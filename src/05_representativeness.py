"""Sample representativeness against US adult population marginals.

Answers deliverable 1's 'distributions and representativeness' requirement: who is this
panel, and who does a model trained on it therefore speak for?

Produces reports/exploration/05_representativeness.md and data/derived/representativeness.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, US_BENCHMARKS, load_responses  # noqa: E402

OUT_MD = ROOT / "reports/exploration/05_representativeness.md"
OUT_CSV = ROOT / "data/derived/representativeness.csv"

# Demographic QIDs -> benchmark key. Region/race labels are long; matched on prefix.
DEMO_COLS = {
    "QID11": "region",
    "QID12": "sex",
    "QID13": "age",
    "QID14": "education",
    "QID15": "race",
}


def bucket(value: str, keys: list[str]) -> str | None:
    """Map a survey label onto a benchmark category by prefix match."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    for k in keys:
        if v == k or v.startswith(k):
            return k
    return None


def main() -> None:
    lab13, _ = load_responses(labels=True)

    rows = []
    for col, key in DEMO_COLS.items():
        if col not in lab13.columns:
            continue
        bench = US_BENCHMARKS[key]
        mapped = lab13[col].map(lambda v: bucket(v, list(bench)))
        n_unmapped = int(lab13[col].notna().sum() - mapped.notna().sum())
        share = mapped.value_counts(normalize=True)
        for cat, p_us in bench.items():
            p_s = float(share.get(cat, 0.0))
            rows.append({
                "variable": key, "column": col, "category": cat,
                "sample": p_s, "us_adult": p_us,
                "diff_pp": (p_s - p_us) * 100,
                "ratio": p_s / p_us if p_us else np.nan,
                "unmapped_n": n_unmapped,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    L: list[str] = []
    A = L.append
    A("# 05 — Representativeness\n")
    A(f"Panel: N = {len(lab13):,} US adults. Compared against US adult population")
    A("marginals (ACS 2023 / CPS 2023; educational attainment is for age 25+). The")
    A("benchmarks are approximate and are used to show the direction and rough size of")
    A("skew, not to reweight anything.\n")
    A("Total-variation distance (half the sum of absolute differences) summarises each")
    A("variable: 0 means the marginal matches, 0.10 means a tenth of the sample would")
    A("have to be reassigned to match.\n")

    A("| Variable | TV distance | largest gap |")
    A("|---|---:|---|")
    for v, g in df.groupby("variable"):
        tv = g["diff_pp"].abs().sum() / 200
        worst = g.loc[g["diff_pp"].abs().idxmax()]
        A(f"| {v} | {tv:.3f} | {worst['category']}: {worst['sample']:.1%} vs "
          f"{worst['us_adult']:.1%} ({worst['diff_pp']:+.1f}pp) |")
    A("")

    for v, g in df.groupby("variable"):
        A(f"### {v}\n")
        A("| category | sample | US adults | diff (pp) | ratio |")
        A("|---|---:|---:|---:|---:|")
        for _, r in g.iterrows():
            A(f"| {r['category']} | {r['sample']:.1%} | {r['us_adult']:.1%} "
              f"| {r['diff_pp']:+.1f} | {r['ratio']:.2f}x |")
        un = int(g["unmapped_n"].iloc[0])
        if un:
            A(f"\n{un} responses did not map to a benchmark category and are excluded "
              f"from this variable's shares.")
        A("")

    edu = df[df["variable"] == "education"]
    low = edu[edu["category"] == "Less than high school"]
    age = df[df["variable"] == "age"]
    old = age[age["category"] == "65+"]

    A("## What this means for the model\n")
    A("Two skews are large enough to change how results should be read:\n")
    if len(low):
        r = low.iloc[0]
        A(f"- **Education.** 'Less than high school' is {r['sample']:.1%} of the panel "
          f"against {r['us_adult']:.1%} of US adults — a {r['ratio']:.2f}x")
        A("  under-representation. The least-educated decile of the country is effectively")
        A("  absent. Cognitive-task and heuristics-and-biases results are exactly the")
        A("  measures most sensitive to this, and they are the entire wave-4 evaluation set.")
    if len(old):
        r = old.iloc[0]
        A(f"- **Age.** 65+ is {r['sample']:.1%} against {r['us_adult']:.1%} "
          f"({r['diff_pp']:+.1f}pp). Older adults are under-sampled.")
    A("")
    A("Consequences, stated plainly:\n")
    A("1. A behaviour model fit here is a model of an online-panel population that is")
    A("   more educated and younger than the US, not of the US.")
    A("2. Post-stratification weights can repair *aggregate* estimates for these marginals.")
    A("   They cannot repair individual-level prediction for a demographic cell that")
    A("   contains almost no training examples.")
    A("3. Combined with the flattening documented in report 03 — simulated populations at")
    A("   roughly half the real variance — minority positions are compressed twice: once by")
    A("   who was sampled, once by the model's pull toward the modal answer. Any use of")
    A("   this system to speak for an under-represented group is unsupported.\n")
    A("These marginals are also the only handle available for the fairness slice of the")
    A("evaluation: report accuracy *per demographic cell*, not just pooled, and treat a")
    A("cell with a materially worse score as a blocking result rather than a footnote.\n")

    OUT_MD.write_text("\n".join(L))
    print(f"wrote {OUT_MD}\nwrote {OUT_CSV}\n")
    for v, g in df.groupby("variable"):
        print(f"{v:12s} TV={g['diff_pp'].abs().sum()/200:.3f}  "
              f"worst={g.loc[g['diff_pp'].abs().idxmax(),'category'][:28]:28s} "
              f"{g['diff_pp'].abs().max():+.1f}pp")


if __name__ == "__main__":
    main()
