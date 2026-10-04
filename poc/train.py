"""Fine-tune a tiny causal LM to answer wave-4 items as a specific person.

This is the lightweight prototype, not the proposed Large Behavior Model. The default
SmolLM2-135M model is chosen because it runs end-to-end on a free Colab T4 and is large
enough to validate the data, leakage guards, training loop, and evaluation ladder. The
main modelling plan in reports/02_modeling_plan.md uses a 7B/8B-class model or a hybrid
psychometric + LLM system.

The answer is always a single digit, so the whole task collapses to predicting one token.
Loss is applied to that token only; the prompt is masked. At eval time the logits at that
position are restricted to the item's valid options, which gives a proper distribution
over answers for free -- that is what the variance-ratio metric needs.

Usage:
  python poc/train.py --arm held_out
  python poc/train.py --arm held_out --n-train 10000 --batch-size 16 --device cuda
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def device(choice: str = "auto") -> torch.device:
    """CUDA if present, else CPU.

    MPS is deliberately not auto-selected. Benchmarked on an M-series laptop at this
    model size it was both slower than CPU and unstable -- step time degraded from 1.6s
    to 4.1s over five steps and the allocator hit its 9 GB watermark -- because the
    backward pass thrashes the cache. Pass --device mps to override.
    """
    if choice != "auto":
        return torch.device(choice)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def digit_ids(tok) -> dict[int, int]:
    """Token id for '1'..'9'. The prompt ends with a trailing space so the answer is a
    bare digit; most BPE vocabularies keep ' 1' as two tokens but '1' as one."""
    out = {}
    for d in range(1, 10):
        ids = tok.encode(str(d), add_special_tokens=False)
        assert len(ids) == 1, f"'{d}' is not a single token for this tokenizer: {ids}"
        out[d] = ids[0]
    return out


class Rows(Dataset):
    def __init__(self, path: Path, tok, max_len: int, limit: int | None):
        # Stream and stop at `limit`. The train files are ~240 MB; reading them whole to
        # keep the first few thousand rows is a large and pointless allocation.
        self.rows = []
        with path.open() as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))
                if limit and len(self.rows) >= limit:
                    break
        self.tok, self.max_len, self.dig = tok, max_len, digit_ids(tok)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        # Truncate from the LEFT: the question and options sit at the end of the prompt
        # and must never be cut.
        ids = self.tok.encode(r["prompt"] + " ", add_special_tokens=False)[-self.max_len:]
        return {"input_ids": ids, "target": self.dig[r["answer"]]}


def collate(batch, pad_id: int):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):  # left-pad so the answer slot is the final position
        k = len(b["input_ids"])
        ids[i, n - k:] = torch.tensor(b["input_ids"])
        att[i, n - k:] = 1
    return {"input_ids": ids, "attention_mask": att,
            "target": torch.tensor([b["target"] for b in batch])}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["held_out", "same_item"], default="held_out")
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--n-train", type=int, default=2400)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-len", type=int, default=192)
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    dev = device(a.device)
    out_dir = Path(a.out) if a.out else HERE / f"ckpt_{a.arm}"
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model).to(dev)
    n_par = sum(x.numel() for x in model.parameters())
    assert n_par < 5e8, f"{n_par/1e6:.0f}M params exceeds the 0.5B cap"
    print(f"model {a.model}  {n_par/1e6:.0f}M params  device {dev}")

    ds = Rows(DATA / f"{a.arm}_train.jsonl", tok, a.max_len, a.n_train)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = len(dl) * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total,
                                                pct_start=0.05)
    print(f"arm={a.arm}  {len(ds):,} examples  {total} steps")

    model.train()
    t0, run = time.time(), []
    for ep in range(a.epochs):
        for i, b in enumerate(dl):
            tgt = b.pop("target").to(dev)
            b = {k: v.to(dev) for k, v in b.items()}
            # Only the final position matters, so keep one position of logits. Computing
            # logits for all ~320 positions over a 49k vocab is what exhausts MPS memory.
            logits = model(**b, logits_to_keep=1).logits[:, -1, :]
            loss = torch.nn.functional.cross_entropy(logits.float(), tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run.append(loss.item())
            # MPS caches aggressively and will hit its watermark within a few hundred
            # steps without this, even though the live set is under 3 GB.
            if dev.type == "mps" and i % 20 == 0:
                torch.mps.empty_cache()
            if i % 50 == 0:
                el = time.time() - t0
                done = ep * len(dl) + i + 1
                print(f"  step {done}/{total}  loss {sum(run[-50:])/len(run[-50:]):.4f}"
                      f"  {el:.0f}s  eta {el/done*(total-done):.0f}s", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "train_meta.json").write_text(json.dumps({
        "arm": a.arm, "base_model": a.model, "n_params": n_par, "n_train": len(ds),
        "epochs": a.epochs, "batch_size": a.batch_size, "lr": a.lr,
        "max_len": a.max_len, "final_loss": sum(run[-50:]) / len(run[-50:]),
        "seconds": round(time.time() - t0, 1),
    }, indent=1))
    print(f"saved -> {out_dir}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
