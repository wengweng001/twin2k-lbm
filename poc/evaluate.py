"""Score the POC model against the comparator ladder from docs/03_evaluation.md.

Reports, on the same test participants and the same valid pairs:
  B1 population mode   (computed on TRAIN pids only)
  B4 copy-forward      (= the human test-retest reference line)
  base model, untuned  (does fine-tuning add anything?)
  fine-tuned model
plus the variance ratio, because accuracy alone selects for flattening.

Usage:
  python poc/evaluate.py --arm held_out
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def device(choice: str = "auto") -> torch.device:
    """See poc/train.py: MPS is not auto-selected because it benchmarked slower and
    less stable than CPU at this model size."""
    if choice != "auto":
        return torch.device(choice)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def digit_ids(tok) -> dict[int, int]:
    return {d: tok.encode(str(d), add_special_tokens=False)[0] for d in range(1, 10)}


@torch.no_grad()
def predict(model, tok, rows, max_len: int, batch_size: int, dev) -> np.ndarray:
    """Distribution over each item's valid options, from the logits at the answer slot."""
    dig = digit_ids(tok)
    model.eval()
    probs: list[np.ndarray] = []
    for s in range(0, len(rows), batch_size):
        chunk = rows[s: s + batch_size]
        enc = [tok.encode(r["prompt"] + " ", add_special_tokens=False)[-max_len:]
               for r in chunk]
        n = max(len(e) for e in enc)
        ids = torch.full((len(chunk), n), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(chunk), n), dtype=torch.long)
        for i, e in enumerate(enc):  # left-pad: answer slot is the final position
            ids[i, n - len(e):] = torch.tensor(e)
            att[i, n - len(e):] = 1
        logits = model(input_ids=ids.to(dev), attention_mask=att.to(dev),
                       logits_to_keep=1).logits[:, -1, :]
        for i, r in enumerate(chunk):
            cand = [dig[v] for v in r["valid"]]
            p = torch.softmax(logits[i, cand].float(), dim=-1).cpu().numpy()
            probs.append(p)
        if s % (batch_size * 40) == 0:
            print(f"  scored {s + len(chunk):,}/{len(rows):,}", flush=True)
    return probs


def per_column(rows, pred: list[int]) -> tuple[float, dict]:
    """Per-column accuracy, then equal-weight mean over columns."""
    by: dict[str, list] = {}
    for r, p in zip(rows, pred):
        by.setdefault(r["col"], []).append(p == r["answer"])
    accs = {c: float(np.mean(v)) for c, v in by.items()}
    return float(np.mean(list(accs.values()))), accs


def variance_ratio(rows, pred: list[int]) -> tuple[float, float]:
    """Median var(predicted)/var(human) over columns, and the collapse fraction.

    Collapse fraction is the share of columns on which the system emits a single
    constant answer for every participant. A median variance ratio of 0.00 is not a
    rounding artefact -- it means more than half the columns are collapsed.
    """
    by: dict[str, list] = {}
    for r, p in zip(rows, pred):
        by.setdefault(r["col"], [[], []])
        by[r["col"]][0].append(p)
        by[r["col"]][1].append(r["answer"])
    out, collapsed, n = [], 0, 0
    for c, (p, t) in by.items():
        vt = np.var(t)
        if vt > 1e-9:
            out.append(np.var(p) / vt)
            n += 1
            collapsed += len(set(p)) == 1
    return float(np.median(out)), (collapsed / n if n else float("nan"))


def boot_ci(rows, a: list[int], b: list[int], n_boot=400, seed=0):
    """CI on (acc_a - acc_b), resampling participants."""
    rng = np.random.default_rng(seed)
    pids = sorted({r["pid"] for r in rows})
    idx: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        idx.setdefault(r["pid"], []).append(i)
    ha = np.array([x == r["answer"] for x, r in zip(a, rows)], dtype=float)
    hb = np.array([x == r["answer"] for x, r in zip(b, rows)], dtype=float)
    cols = np.array([r["col"] for r in rows])
    uniq = {c: i for i, c in enumerate(sorted(set(cols)))}
    cidx = np.array([uniq[c] for c in cols])

    def mean_by_col(h, sel):
        s = np.bincount(cidx[sel], weights=h[sel], minlength=len(uniq))
        n = np.bincount(cidx[sel], minlength=len(uniq))
        m = n > 0
        return float(np.mean(s[m] / n[m]))

    vals = []
    for _ in range(n_boot):
        draw = rng.choice(pids, size=len(pids), replace=True)
        sel = np.concatenate([idx[p] for p in draw])
        vals.append(mean_by_col(ha, sel) - mean_by_col(hb, sel))
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["held_out", "same_item"], default="held_out")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--base", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--n-test", type=int, default=1800)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-len", type=int, default=192)
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    a = p.parse_args()

    ckpt = Path(a.ckpt) if a.ckpt else HERE / f"ckpt_{a.arm}"
    meta = json.loads((DATA / "meta.json").read_text())
    rows = [json.loads(l) for l in (DATA / f"{a.arm}_test.jsonl").read_text().splitlines() if l]
    rng = np.random.default_rng(0)
    if a.n_test and len(rows) > a.n_test:
        rows = [rows[i] for i in rng.choice(len(rows), a.n_test, replace=False)]
    dev = device(a.device)
    print(f"arm={a.arm}  {len(rows):,} test examples  "
          f"{len({r['pid'] for r in rows})} participants  device {dev}")

    # ---- non-model baselines ----
    pop = meta["pop_mode"]
    pred_pop = [int(pop.get(r["col"], r["valid"][0])) for r in rows]
    pred_cf = [r["copyfwd"] for r in rows]

    results = {}
    acc_pop, _ = per_column(rows, pred_pop)
    acc_cf, _ = per_column(rows, pred_cf)
    results["B1 population mode"] = (acc_pop, *variance_ratio(rows, pred_pop))
    results["B4 copy-forward (= retest line)"] = (acc_cf, *variance_ratio(rows, pred_cf))

    # ---- models ----
    preds = {}
    for name, path in [("base model (untuned)", a.base), ("fine-tuned", str(ckpt))]:
        if name == "fine-tuned" and not ckpt.exists():
            print(f"!! {ckpt} not found; run poc/train.py --arm {a.arm} first")
            continue
        tok = AutoTokenizer.from_pretrained(path)
        tok.pad_token = tok.pad_token or tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(path).to(dev)
        print(f"\nscoring {name} ...")
        probs = predict(model, tok, rows, a.max_len, a.batch_size, dev)
        pr = [r["valid"][int(np.argmax(q))] for r, q in zip(rows, probs)]
        preds[name] = pr
        acc, _ = per_column(rows, pr)
        results[name] = (acc, *variance_ratio(rows, pr))
        del model

    # ---- CIs for the fine-tuned model against each baseline ----
    ci = {}
    if "fine-tuned" in preds:
        ft = preds["fine-tuned"]
        ci["vs B1"] = boot_ci(rows, ft, pred_pop)
        ci["vs B4"] = boot_ci(rows, ft, pred_cf)
        if "base model (untuned)" in preds:
            ci["vs base"] = boot_ci(rows, ft, preds["base model (untuned)"])

    band = acc_cf - acc_pop
    lines = ["", "=" * 74,
             f"POC results  |  arm = {a.arm}  |  {len(rows):,} test pairs, "
             f"{len({r['pid'] for r in rows})} held-out participants", "=" * 74,
             f"{'system':38s} {'accuracy':>10s} {'var ratio':>11s} {'collapsed':>11s} {'NS':>8s}"]
    for k, (acc, vr, cf_frac) in results.items():
        ns = (acc - acc_pop) / band if band > 1e-9 else float("nan")
        lines.append(f"{k:38s} {acc:10.4f} {vr:11.2f} {cf_frac:10.0%} {ns:8.2f}")
    lines += ["", f"personalisation band (B4 - B1) = {band:.4f}",
              "NS: 0 = no personalisation, 1 = copy-forward-equivalent, >1 = denoising",
              "collapsed = share of columns given one constant answer for everyone"]
    for k, (lo, hi) in ci.items():
        lines.append(f"fine-tuned {k}: 95% CI [{lo:+.4f}, {hi:+.4f}]"
                     f"{'  (significant)' if lo > 0 or hi < 0 else '  (not significant)'}")
    lines.append("=" * 74)
    print("\n".join(lines))

    out = HERE / f"results_{a.arm}.json"
    out.write_text(json.dumps({
        "arm": a.arm, "n_test": len(rows),
        "n_test_pids": len({r["pid"] for r in rows}),
        "band": band,
        "results": {k: {"accuracy": v[0], "var_ratio": v[1], "collapsed_frac": v[2]}
                    for k, v in results.items()},
        "ci": ci,
    }, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
