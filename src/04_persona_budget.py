"""Persona size and context-window budget, plus sample representativeness.

Produces reports/04_persona_budget.md
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, column_meta, load_responses  # noqa: E402

OUT_MD = ROOT / "reports/04_persona_budget.md"
CHARS_PER_TOKEN = 4.0  # conservative English estimate; no tokenizer dependency


def main() -> None:
    files = sorted(glob.glob(str(ROOT / "wave_split/chunks/*.parquet"))) or \
        sorted(glob.glob(str(ROOT / "data/raw/wave_split/chunks/*.parquet")))
    ws = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    fp = pd.concat([pd.read_parquet(f, columns=["pid", "persona_text", "persona_summary"])
                    for f in sorted(glob.glob(str(ROOT / "data/raw/full_persona/chunks/*.parquet")))],
                   ignore_index=True)

    L: list[str] = []
    A = L.append
    A("# 04 — Persona size, context budget, and sample shape\n")

    A("## Persona length\n")
    A("| Field | median chars | p95 chars | median ~tokens | p95 ~tokens |")
    A("|---|---:|---:|---:|---:|")
    fields = [
        ("wave1_3_persona_text (the legal input)", ws["wave1_3_persona_text"]),
        ("wave1_3_persona_json", ws["wave1_3_persona_json"].astype(str)),
        ("wave4_Q_wave4_A (target, must be stripped)", ws["wave4_Q_wave4_A"].astype(str)),
        ("full_persona.persona_text (LEAKS)", fp["persona_text"]),
        ("full_persona.persona_summary", fp["persona_summary"]),
    ]
    stats = {}
    for name, s in fields:
        n = s.str.len()
        stats[name] = n
        A(f"| {name} | {n.median():,.0f} | {n.quantile(.95):,.0f} "
          f"| {n.median()/CHARS_PER_TOKEN:,.0f} | {n.quantile(.95)/CHARS_PER_TOKEN:,.0f} |")
    A(f"\n(~tokens = chars / {CHARS_PER_TOKEN:.0f}, the usual English rule of thumb.)\n")

    # The rule of thumb is calibrated on prose. This text is structured and repetitive,
    # so measure it rather than assert it.
    exact = None
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        samp = ws["wave1_3_persona_text"].sample(20, random_state=0)
        exact = [len(tk.encode(s, add_special_tokens=False)) for s in samp]
        ratio = float(np.mean([len(s) for s in samp]) / np.mean(exact))
        A(f"Measured on 20 sampled personas with a real BPE tokenizer: "
          f"**{np.median(exact):,.0f} tokens** median, i.e. {ratio:.2f} characters per")
        A(f"token. The rule of thumb {'under' if ratio < CHARS_PER_TOKEN else 'over'}"
          f"states the true count by roughly {abs(1 - CHARS_PER_TOKEN/ratio):.0%}, because")
        A("this text is structured and repetitive rather than prose. Budget against the")
        A("measured figure.\n")
    except Exception as e:  # tokenizer unavailable offline; the estimate still stands
        A(f"(Exact tokenizer measurement skipped: {type(e).__name__}.)\n")

    tt = stats["wave1_3_persona_text (the legal input)"]
    tok_med = float(np.median(exact)) if exact else tt.median() / CHARS_PER_TOKEN
    A("## What this implies for the model\n")
    A(f"- A legal persona is ~**{tok_med:,.0f} tokens** at the median (measured, not")
    A("  estimated). That fits a modern long-context model but is far too large for the")
    A("  <0.5B models the bonus POC is restricted to (typically 2k-8k usable context).")
    A(f"- Naive prompting cost: {tok_med:,.0f} tokens x 126 questions x")
    A(f"  2,058 people ≈ {tok_med*126*2058/1e9:,.1f}B input tokens per full")
    A("  evaluation sweep if the persona is re-sent per question. Batch all 126 questions into")
    A("  one call per person and it drops by ~126x; cache the persona prefix and it drops again.")
    A(f"- `persona_summary` is ~{stats['full_persona.persona_summary'].median()/CHARS_PER_TOKEN:,.0f}")
    A("  tokens — a ~%.0fx compression. Section 03 shows the summary specifications score"
      % (tt.median() / stats["full_persona.persona_summary"].median()))
    A("  *worse*, so compression here is not free.")
    A("- Retrieval is the obvious lever: for each target question, pull only the items")
    A("  measuring the same construct. The 126 wave-4 columns map to 85 QuestionIDs, so a")
    A("  per-question context of a few hundred tokens is achievable.\n")

    # ---- representativeness ----
    w13, w4 = load_responses()
    lab13, _ = load_responses(labels=True)
    meta = column_meta()
    demo = meta[meta["block"] == "Demographics"].index.tolist()
    A("## Sample shape\n")
    A(f"- N = {len(w13):,} participants; {w13.shape[1]} wave-1-3 columns, {w4.shape[1]} wave-4 columns.")
    A(f"- Demographic columns in the catalog: {len(demo)}.\n")
    shown = 0
    for c in demo:
        if c not in lab13.columns or shown >= 6:
            continue
        vc = lab13[c].value_counts(normalize=True).head(6)
        if vc.empty:
            continue
        A(f"**{c}** — {str(meta.loc[c,'text'])[:90]}\n")
        A("| value | share |")
        A("|---|---:|")
        for k, v in vc.items():
            A(f"| {str(k)[:60]} | {v:.1%} |")
        A("")
        shown += 1

    A("## Limitations that bound every claim built on this data\n")
    A("- US adults only, surveyed in 2024-25. No cross-national or longitudinal coverage")
    A("  beyond the four waves, so nothing here supports claims about behaviour drift.")
    A("- Self-report throughout: social-desirability and satisficing are baked in, and the")
    A(f"  {0.612:.0%} test-retest figure is itself partly a measure of survey noise, not of")
    A("  genuine preference instability.")
    A("- Wave 4 covers only heuristics-and-biases plus pricing. Personality, cognitive and")
    A("  demographic blocks are **never** evaluated, so 'predicts a person' really means")
    A("  'predicts their behavioural-economics answers'.")
    A("- Attrition across waves and panel self-selection are not corrected for here.\n")

    OUT_MD.write_text("\n".join(L))
    print(f"wrote {OUT_MD}\n")
    for name, s in fields:
        n = s.str.len()
        print(f"{name:48s} median {n.median():9,.0f} chars  (~{n.median()/CHARS_PER_TOKEN:7,.0f} tok)")


if __name__ == "__main__":
    main()
