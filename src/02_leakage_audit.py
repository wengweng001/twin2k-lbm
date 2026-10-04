"""Empirical audit of the dataset's leakage traps.

Claim under test: the `full_persona` config embeds the wave-4 ground truth, so any
persona built from it scores against answers it was already shown.

Produces reports/exploration/02_leakage_audit.md
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, load_responses  # noqa: E402

OUT_MD = ROOT / "reports/exploration/02_leakage_audit.md"
N_SAMPLE = 300


def load_parquets(name: str, columns: list[str]) -> pd.DataFrame:
    files = sorted(glob.glob(str(ROOT / f"data/raw/{name}/chunks/*.parquet")))
    return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)


def answered_qids(persona_json: str) -> dict[str, object]:
    """Map QuestionID -> answer payload for every answered question in a persona JSON."""
    obj = json.loads(persona_json) if isinstance(persona_json, str) else persona_json
    out: dict[str, object] = {}

    def walk(node):
        if isinstance(node, dict):
            if "QuestionID" in node and "Answers" in node:
                out[node["QuestionID"]] = node["Answers"]
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(obj)
    return out


def main() -> None:
    w13, w4 = load_responses()
    L: list[str] = []
    A = L.append
    A("# 02 — Leakage audit\n")

    full = load_parquets("full_persona", ["pid", "persona_text", "persona_json"])
    ws = load_parquets("wave_split", ["pid", "wave1_3_persona_json", "wave4_Q_wave1_3_A",
                                      "wave4_Q_wave4_A"])
    full["pid"] = full["pid"].astype(int)
    ws["pid"] = ws["pid"].astype(int)
    A(f"`full_persona`: {len(full)} rows. `wave_split`: {len(ws)} rows.\n")

    # ---------- Trap 1: full_persona carries wave-4 answers ----------
    A("## Trap 1 — `full_persona` contains the wave-4 ground truth\n")
    A("The dataset README states it plainly, in a line easy to skim past:\n")
    A("> `persona_text`: Complete survey responses in text format, including all questions")
    A("> and answers. **For questions that appear in both waves 1-3 and wave 4, the wave 4")
    A("> responses are used.**\n")
    A("`full_persona` is the first config listed and the one whose name implies 'everything")
    A("we know about this person'. It is the natural pick, but it is not a valid input for this task.\n")

    sample = full.sample(N_SAMPLE, random_state=0)
    ws_idx = ws.set_index("pid")
    rows_match_w4, rows_match_w13, n_cmp = 0, 0, 0
    per_pid = []
    for _, r in sample.iterrows():
        pid = r["pid"]
        if pid not in ws_idx.index or pid not in w4.index:
            continue
        fp = answered_qids(r["persona_json"])
        gt = answered_qids(ws_idx.loc[pid, "wave4_Q_wave4_A"])
        old = answered_qids(ws_idx.loc[pid, "wave4_Q_wave1_3_A"])
        shared = set(fp) & set(gt) & set(old)
        m4 = sum(1 for q in shared if json.dumps(fp[q], sort_keys=True) ==
                 json.dumps(gt[q], sort_keys=True))
        m13 = sum(1 for q in shared if json.dumps(fp[q], sort_keys=True) ==
                  json.dumps(old[q], sort_keys=True))
        rows_match_w4 += m4
        rows_match_w13 += m13
        n_cmp += len(shared)
        if shared:
            per_pid.append(m4 / len(shared))

    A(f"Audit over {len(per_pid)} randomly sampled participants, comparing each wave-4")
    A(f"QuestionID's answer payload in `full_persona.persona_json` against both possible")
    A("sources.\n")
    A("| `full_persona` answer matches... | share of wave-4 questions |")
    A("|---|---:|")
    A(f"| the **wave-4** answer (the eval target) | **{rows_match_w4/n_cmp:.1%}** |")
    A(f"| the wave-1-3 answer | {rows_match_w13/n_cmp:.1%} |")
    A(f"\n({n_cmp:,} question-participant pairs compared.)\n")
    A(f"Per-participant, a median of **{pd.Series(per_pid).median():.1%}** of wave-4")
    A("questions are already answered — correctly — inside the persona.\n")
    A("Anyone prompting with `full_persona` and scoring on wave 4 is measuring string")
    A(f"retrieval. Expected accuracy approaches {rows_match_w4/n_cmp:.0%}, versus a")
    A(f"{0.612:.0%} human test-retest reference. A result above the human ceiling is the")
    A("tell-tale that this trap has been stepped in.\n")

    # ---------- Trap 2: the README's own usage snippet ----------
    A("## Trap 2 — the README's usage example leaks\n")
    A("```python")
    A('test_questions = wave_split["data"]["wave4_Q_wave4_A"]  # you want to remove the "Answers"')
    A('ground_truth   = wave_split["data"]["wave4_Q_wave4_A"]')
    A("```")
    A("Prompt input and ground truth are the *same field*. The fix is demoted to a trailing")
    A("comment. `wave4_Q_wave4_A` must be stripped of every `Answers` key before it is")
    A("shown to a model; `wave4_Q_wave1_3_A` is the copy-forward baseline, not an input.\n")

    # ---------- Trap 3: item overlap inside wave1_3 persona ----------
    A("## Trap 3 — even the clean persona contains the same items\n")
    A("`wave1_3_persona_text` legitimately includes the earlier administration of every")
    A("wave-4 question, because that is what makes copy-forward possible. That is not a bug,")
    A("but it means:\n")
    A("- the task is **not** 'generalise to unseen questions' — 0 of 126 wave-4 columns are new;")
    A("- a model can score well by copying, so *beating copy-forward* is the only meaningful")
    A("  success criterion;")
    A("- to test genuine generalisation you must hold out question *blocks* from the persona")
    A("  yourself. The dataset does not provide that split.\n")

    # ---------- Trap 4: split granularity ----------
    A("## Trap 4 — split granularity and contamination\n")
    n_pairs = int((w13[w4.columns].notna() & w4.notna()).sum().sum())
    A(f"- The natural fine-tuning unit is the (person, question) pair: ~{n_pairs:,} of them.")
    A("  Splitting on pairs puts the same person in train and test. Split on **pid**.")
    A("- Splitting only on pid still lets the model memorise each item's population")
    A("  distribution, which is the no-personalisation baseline wearing a costume. Report a")
    A("  second split that holds out whole question blocks to separate the two abilities.")
    A("- The dataset has been public on HuggingFace since 2025 and ships with GPT-4.1 and")
    A("  Gemini-Flash-2.5 outputs. Any frontier model may have memorised it; pretraining")
    A("  contamination is a live confound for prompting baselines and cannot be ruled out")
    A("  from inside the benchmark.\n")

    # ---------- Structural check on between-subject arms ----------
    A("## Trap 5 — between-subject arms are not missing data\n")
    arms = Counter()
    for c in w4.columns:
        arms[float(w4[c].notna().mean() < 0.9)] += 1
    A(f"- {int(arms[1.0])} of {len(w4.columns)} wave-4 columns are answered by <90% of")
    A("  participants, because each person saw one arm of each between-subject experiment.")
    A("- Imputing or dropping these rows silently changes the estimand. Metrics must be")
    A("  computed per column on valid pairs, then aggregated with explicit weights.\n")

    OUT_MD.write_text("\n".join(L))
    print(f"wrote {OUT_MD}")
    print(f"full_persona answer == wave4 ground truth : {rows_match_w4/n_cmp:.3%}")
    print(f"full_persona answer == wave1-3 answer     : {rows_match_w13/n_cmp:.3%}")
    print(f"pairs compared: {n_cmp:,}")


if __name__ == "__main__":
    main()
