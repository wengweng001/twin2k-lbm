"""Figures for the exploration report.

Produces figures/*.png. These figures are deliberately simple: each one is meant to
support one claim in the README, not to show every available number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT  # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

INK = "#1F2933"
MUTED = "#7B8794"
GRID = "#E4E7EB"
BLUE = "#2F6FED"
GREEN = "#2F855A"
RED = "#C2410C"
GOLD = "#B7791F"
GREY = "#CBD2D9"

plt.rcParams.update({
    "figure.dpi": 160,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def fig_retest_by_scale(df: pd.DataFrame) -> None:
    order = [s for s in ["binary", "ordinal_short", "ordinal_long", "continuous", "text"]
             if s in set(df["scale"])]
    summary = (df.groupby("scale")["retest_exact"]
               .agg(["mean", "median", "count"])
               .reindex(order))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    y = np.arange(len(summary))
    bars = ax.barh(y, summary["mean"], color=BLUE, alpha=.88, height=.55)
    overall = df["retest_exact"].mean()
    ax.axvline(overall, color=INK, lw=1.2, ls="--")
    ax.text(overall + .01, len(summary) - .35, f"overall {overall:.3f}", fontsize=8,
            va="center", color=INK)

    labels = [f"{idx.replace('_', ' ')}  (n={int(row['count'])})"
              for idx, row in summary.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("exact agreement between earlier answer and wave 4")
    ax.set_title("Human test-retest varies a lot by answer type", loc="left", pad=10)
    ax.grid(axis="x", color=GRID, lw=.8)
    ax.set_axisbelow(True)

    for b, v in zip(bars, summary["mean"]):
        ax.text(v + .015, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center",
                fontsize=8, color=INK)
    save(fig, "retest_by_scale.png")


def fig_baseline_ladder(spec: pd.DataFrame) -> None:
    df = spec.copy()
    df["label"] = df.index.str.replace(" – ", " - ", regex=False).str[:42]
    df = df.sort_values("llm_acc", ascending=True)
    pop = float(df["popmode_acc"].mean())
    copy = float(df["copyfwd_acc"].mean())

    fig, ax = plt.subplots(figsize=(8.3, 5.7))
    y = np.arange(len(df))
    colors = np.where(df["llm_acc"] >= pop, BLUE, RED)
    ax.barh(y, df["llm_acc"], color=colors, height=.62, alpha=.9)

    ax.axvspan(pop, copy, color=BLUE, alpha=.08)
    ax.axvline(pop, color=INK, lw=1.2, ls=":")
    ax.axvline(copy, color=INK, lw=1.2, ls="--")
    ax.text(pop - .002, len(df) + .05, f"population mode {pop:.3f}",
            ha="right", va="bottom", fontsize=8)
    ax.text(copy, len(df) + .05, f"copy-forward {copy:.3f}",
            ha="center", va="bottom", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=7.7)
    ax.set_xlim(0.39, 0.64)
    ax.set_ylim(-.8, len(df) + .8)
    ax.set_xlabel("exact-match accuracy")
    ax.set_title("Published LLM runs do not beat copy-forward", loc="left", pad=10)
    ax.grid(axis="x", color=GRID, lw=.8)
    ax.set_axisbelow(True)

    ax.text((pop + copy) / 2, -.55, "personalisation band", ha="center", va="center",
            fontsize=8, color=BLUE)
    for yi, v in zip(y, df["llm_acc"]):
        ax.text(v + .004, yi, f"{v:.3f}", va="center", fontsize=7.4)
    save(fig, "baseline_ladder.png")


def fig_accuracy_vs_variance(spec: pd.DataFrame) -> None:
    df = spec.copy()
    r = float(np.corrcoef(df["var_ratio"], df["llm_acc"])[0, 1])
    pop = float(df["popmode_acc"].mean())
    copy = float(df["copyfwd_acc"].mean())

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.axvspan(.85, 1.15, color=GREEN, alpha=.10)
    ax.axhline(pop, color=INK, lw=1.1, ls=":")
    ax.axhline(copy, color=INK, lw=1.1, ls="--")
    ax.text(.03, pop + .002, "population mode", va="bottom", ha="left", fontsize=8)
    ax.text(.03, copy + .002, "copy-forward", va="bottom", ha="left", fontsize=8)

    above = df["llm_acc"] >= pop
    ax.scatter(df.loc[~above, "var_ratio"], df.loc[~above, "llm_acc"], s=52,
               color=RED, alpha=.88, label="below population mode")
    ax.scatter(df.loc[above, "var_ratio"], df.loc[above, "llm_acc"], s=52,
               color=BLUE, alpha=.88, label="above population mode")

    # Label only the points a reviewer is likely to ask about.
    label_rows = {
        df["llm_acc"].idxmax(): "best acc",
        df["var_ratio"].idxmax(): "best variance",
        df["var_ratio"].idxmin(): "most collapsed",
    }
    for name, tag in label_rows.items():
        row = df.loc[name]
        offset = (7, 8)
        if tag == "best variance":
            offset = (-74, 8)
        ax.annotate(tag, (row["var_ratio"], row["llm_acc"]), xytext=offset,
                    textcoords="offset points", fontsize=8)

    ax.set_xlim(0, 1.25)
    ax.set_ylim(0.39, 0.62)
    ax.set_xlabel("variance ratio  (prediction variance / human variance)")
    ax.set_ylabel("exact-match accuracy")
    ax.set_title(f"Accuracy and variance fidelity are almost unrelated (r={r:+.2f})",
                 loc="left", pad=10)
    ax.grid(color=GRID, lw=.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    save(fig, "accuracy_vs_variance.png")


def fig_representativeness(rep: pd.DataFrame) -> None:
    show = ["education", "age", "race"]
    d = rep[rep["variable"].isin(show)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))

    for ax, var in zip(axes, show):
        g = d[d["variable"] == var].copy()
        g = g.sort_values("diff_pp")
        y = np.arange(len(g))
        ax.barh(y, g["diff_pp"], color=np.where(g["diff_pp"] >= 0, BLUE, GOLD), height=.55)
        ax.axvline(0, color=INK, lw=.9)
        ax.set_yticks(y)
        ax.set_yticklabels(g["category"].str[:24], fontsize=7.5)
        ax.set_xlabel("sample minus US adults (pp)")
        tv = g["diff_pp"].abs().sum() / 200
        ax.set_title(f"{var}  TV={tv:.3f}", loc="left", fontsize=9.5)
        ax.grid(axis="x", color=GRID, lw=.8)
        ax.set_axisbelow(True)

    fig.suptitle("The panel is close on quotas, but education and age still matter",
                 x=.01, ha="left", fontsize=11, color=INK)
    save(fig, "representativeness.png")


def main() -> None:
    df = pd.read_csv(ROOT / "data/derived/per_column_retest.csv")
    spec = pd.read_csv(ROOT / "data/derived/spec_scores.csv").set_index("spec")
    spec = spec[~spec.duplicated("dup_group")] if "dup_group" in spec else spec
    rep = pd.read_csv(ROOT / "data/derived/representativeness.csv")

    fig_retest_by_scale(df)
    fig_baseline_ladder(spec)
    fig_accuracy_vs_variance(spec)
    fig_representativeness(rep)
    for p in sorted(FIG.glob("*.png")):
        print(f"wrote {p.relative_to(ROOT)}  ({p.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
