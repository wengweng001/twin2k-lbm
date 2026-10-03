"""Build the POC training/eval slice.

Task slice: wave-4 columns with small integer support (<=9 options), which covers the
binary and short-ordinal families -- the bulk of the evaluation set and the only families
where a single-token answer is a faithful representation.

Two POC arms:
  arm A "same_item"  -- the person's own earlier answer to the target item is in the prompt.
                        Copying is available. Tests denoising (capability C1).
  arm B "held_out"   -- that answer is removed; only other items are shown.
                        Copying is impossible for that person. This is a no-copy
                        repeated-item POC, not the full S2 unseen-block split from
                        docs/03_evaluation.md; target items still appear in training
                        examples for other participants.

Splits are on pid, never on (person, question).

Writes poc/data/{arm}_{split}.jsonl and poc/data/meta.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import ROOT, annotate_columns, column_meta, load_catalog, load_responses  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)

MAX_OPTIONS = 9      # answer must be a single digit 1-9
N_PRIOR = 8          # prior answers shown per prompt
TEST_FRAC = 0.2
SEED = 0
DEMO_COLS = ["QID13", "QID12", "QID14", "QID11", "QID15"]


def short(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> None:
    rng = random.Random(SEED)
    w13, w4 = load_responses()
    lab13, _ = load_responses(labels=True)
    meta = column_meta()
    ann = annotate_columns(w13, w4)
    catalog = {q["QuestionID"]: q for q in load_catalog()}

    # ---- eligible target columns ----
    targets = []
    for col in w4.columns:
        if ann.loc[col, "scale"] not in ("binary", "ordinal_short"):
            continue
        vals = pd.concat([w13[col], w4[col]]).dropna()
        if vals.empty or not np.allclose(vals, vals.round()):
            continue
        lo, hi = int(vals.min()), int(vals.max())
        if lo < 1 or hi > MAX_OPTIONS or hi <= lo:
            continue
        targets.append(col)

    # ---- option labels per column, read off the observed numeric->label pairing ----
    options: dict[str, dict[int, str]] = {}
    for col in targets:
        qid = meta.loc[col, "qid"] if col in meta.index else None
        opts = (catalog.get(qid) or {}).get("Options") or []
        pairs = {}
        if col in lab13.columns:
            j = pd.concat([w13[col], lab13[col]], axis=1, keys=["n", "l"]).dropna()
            for n, l in j.groupby("n")["l"].first().items():
                pairs[int(n)] = short(l, 42)
        for i, o in enumerate(opts, start=1):
            pairs.setdefault(i, short(o, 42))
        vals = pd.concat([w13[col], w4[col]]).dropna()
        options[col] = {v: pairs.get(v, str(v))
                        for v in range(int(vals.min()), int(vals.max()) + 1)}

    # ---- demographics line per person ----
    demo = {}
    for pid in w13.index:
        bits = [str(lab13.loc[pid, c]) for c in DEMO_COLS
                if c in lab13.columns and pd.notna(lab13.loc[pid, c])]
        demo[pid] = short(", ".join(bits), 150)

    # ---- pid split ----
    pids = sorted(w13.index)
    rng.shuffle(pids)
    n_test = int(len(pids) * TEST_FRAC)
    test_pids, train_pids = set(pids[:n_test]), set(pids[n_test:])

    # Matrix/Slider sub-items share a parent QuestionID and therefore share question
    # text. Without the sub-item suffix two different questions render identically, and
    # the held-out arm cannot be distinguished from a leak.
    col_qid = {c: (meta.loc[c, "qid"] if c in meta.index else c) for c in targets}
    q_text = {}
    for c in targets:
        base = short(meta.loc[c, "text"] if c in meta.index else c, 150)
        sub = c[len(col_qid[c]) + 1:] if c != col_qid[c] else ""
        q_text[c] = f"{base} (item {sub})" if sub else base

    # Near-duplicate key: parent QuestionID plus the question-text prefix. The catalog
    # gives each pricing column its own QuestionID even when two of them ask about the
    # same product, so QuestionID alone does not identify duplicates.
    dup_key = {c: (col_qid[c], q_text[c][:80]) for c in targets}
    blocked_by = {t: {c for c in targets
                      if dup_key[c][0] == dup_key[t][0] or dup_key[c][1] == dup_key[t][1]}
                  for t in targets}

    def prior_block(pid, target, arm) -> str:
        # held_out removes every near-duplicate of the target, not just the target
        # column: a sibling Matrix row or a repeat of the same product is a restatement
        # of the question, and leaving it in makes the arm indistinguishable from a leak.
        blocked = blocked_by[target] if arm == "held_out" else {target}
        pool = [c for c in targets if c not in blocked and pd.notna(w13.loc[pid, c])]
        rng.shuffle(pool)
        lines = []
        if arm == "same_item" and pd.notna(w13.loc[pid, target]):
            v = int(w13.loc[pid, target])
            lines.append(f"- {q_text[target]} -> {options[target].get(v, v)}")
        for c in pool[: N_PRIOR - len(lines)]:
            v = int(w13.loc[pid, c])
            lines.append(f"- {q_text[c]} -> {options[c].get(v, v)}")
        return "\n".join(lines)

    def build(arm: str) -> dict[str, list[dict]]:
        out = {"train": [], "test": []}
        for col in targets:
            opt = options[col]
            opt_str = "  ".join(f"{k}) {v}" for k, v in opt.items())
            valid = sorted(opt)
            for pid in w13.index:
                y, prior = w4.loc[pid, col], w13.loc[pid, col]
                if pd.isna(y) or pd.isna(prior):
                    continue
                y = int(y)
                if y not in opt:
                    continue
                prompt = (
                    f"Person: {demo.get(pid,'')}\n"
                    f"Their earlier survey answers:\n{prior_block(pid, col, arm)}\n"
                    f"Question: {q_text[col]}\n"
                    f"Options: {opt_str}\n"
                    f"Answer:"
                )
                out["train" if pid in train_pids else "test"].append({
                    "pid": int(pid), "col": col, "prompt": prompt,
                    "answer": y, "valid": valid, "copyfwd": int(prior),
                })
        return out

    meta_out = {
        "n_target_columns": len(targets), "columns": targets,
        "n_train_pids": len(train_pids), "n_test_pids": len(test_pids),
        "seed": SEED, "n_prior": N_PRIOR, "max_options": MAX_OPTIONS,
    }
    for arm in ("same_item", "held_out"):
        d = build(arm)
        for split, rows in d.items():
            rng.shuffle(rows)
            p = OUT / f"{arm}_{split}.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in rows))
            meta_out[f"{arm}_{split}"] = len(rows)
            print(f"{p.name:26s} {len(rows):>7,} examples")

    # population mode per column, from TRAIN pids only -- computing it on all pids
    # would leak the test set's marginal into the baseline.
    tr = sorted(train_pids)
    meta_out["pop_mode"] = {c: int(w13.loc[tr, c].mode().iloc[0])
                            for c in targets if w13.loc[tr, c].notna().any()}
    (OUT / "meta.json").write_text(json.dumps(meta_out, indent=1))
    print(f"\n{len(targets)} target columns, "
          f"{len(train_pids)} train / {len(test_pids)} test participants")


if __name__ == "__main__":
    main()
