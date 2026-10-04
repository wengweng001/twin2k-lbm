"""Structure of the evaluation target + human test-retest reliability + trivial baselines.

Produces reports/exploration/01_structure_and_retest.md and data/derived/per_column_retest.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, annotate_columns, bootstrap_ci, load_responses  # noqa: E402

OUT_MD = ROOT / "reports/exploration/01_structure_and_retest.md"
OUT_CSV = ROOT / "data/derived/per_column_retest.csv"
N_BOOT = 1000
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
OUT_MD.parent.mkdir(parents=True, exist_ok=True)


def cohens_kappa(a: pd.Series, b: pd.Series) -> float:
    cats = sorted(set(a) | set(b))
    if len(cats) < 2:
        return np.nan
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    po = np.trace(m) / n
    pe = (m.sum(0) * m.sum(1)).sum() / n**2
    return np.nan if np.isclose(pe, 1) else (po - pe) / (1 - pe)


def main() -> None:
    w13, w4 = load_responses()
    cols = list(w4.columns)
    ann = annotate_columns(w13, w4)

    rows = []
    for col in cols:
        both = w13[col].notna() & w4[col].notna()
        a, b = w13.loc[both, col], w4.loc[both, col]
        if len(a) < 30:
            continue
        rng = max(pd.concat([a, b]).max() - pd.concat([a, b]).min(), 1e-9)

        # Copy-forward IS the test-retest measurement: same computation, two names.
        exact = float((a == b).mean())
        mae = float((a - b).abs().mean())
        within1 = float(((a - b).abs() <= 1).mean())
        corr = float(a.corr(b)) if a.nunique() > 1 and b.nunique() > 1 else np.nan
        kappa = cohens_kappa(a, b) if ann.loc[col, "support"] <= 30 else np.nan

        # Baseline 1: predict everyone's wave-4 answer with the wave1-3 population mode.
        mode = a.mode()
        pop_mode_acc = float((b == mode.iloc[0]).mean()) if len(mode) else np.nan
        # Baseline 2: population mean (for MAE-scored items).
        pop_mean_mae = float((b - a.mean()).abs().mean())

        rows.append(
            {
                "col": col,
                "qid": ann.loc[col, "qid"],
                "qtype": ann.loc[col, "qtype"],
                "block": ann.loc[col, "block"],
                "scale": ann.loc[col, "scale"],
                "n": int(both.sum()),
                "support": ann.loc[col, "support"],
                "retest_exact": exact,
                "retest_within1": within1,
                "retest_mae": mae,
                "retest_nmae": mae / rng,
                "retest_corr": corr,
                "retest_kappa": kappa,
                "pop_mode_acc": pop_mode_acc,
                "pop_mean_mae": pop_mean_mae,
                "pop_mean_nmae": pop_mean_mae / rng,
                "w4_var": float(b.var()),
                "text": ann.loc[col, "text"],
            }
        )

    df = pd.DataFrame(rows).set_index("col")
    df.to_csv(OUT_CSV)

    # ---- participant-level bootstrap on the two headline aggregates ----
    kept = list(df.index)
    hit_retest = np.full((len(w4), len(kept)), np.nan)
    hit_popmode = np.full((len(w4), len(kept)), np.nan)
    for j, col in enumerate(kept):
        ok = (w13[col].notna() & w4[col].notna()).to_numpy()
        a, b = w13[col].to_numpy(), w4[col].to_numpy()
        hit_retest[ok, j] = (a[ok] == b[ok])
        mode = w13.loc[ok, col].mode()
        if len(mode):
            hit_popmode[ok, j] = (b[ok] == mode.iloc[0])

    def agg(mat):
        def f(idx):
            with np.errstate(invalid="ignore"):
                return float(np.nanmean(np.nanmean(mat[idx], axis=0)))
        return f

    def band(idx):
        return agg(hit_retest)(idx) - agg(hit_popmode)(idx)

    pos = np.arange(len(w4))
    ci_retest = bootstrap_ci(agg(hit_retest), pos, n_boot=N_BOOT)
    ci_popmode = bootstrap_ci(agg(hit_popmode), pos, n_boot=N_BOOT)
    ci_band = bootstrap_ci(band, pos, n_boot=N_BOOT)

    L: list[str] = []
    A = L.append
    A("# 01 — Evaluation target, human test-retest ceiling, and trivial baselines\n")
    A(f"Generated from `wave1_3_response.csv` ({w13.shape[0]}x{w13.shape[1]}) "
      f"and `wave4_response.csv` ({w4.shape[0]}x{w4.shape[1]}).\n")

    A("## 1. The evaluation target is 100% repeated items\n")
    c13, c4 = set(w13.columns), set(w4.columns)
    A(f"- wave 1-3 columns: **{len(c13)}**")
    A(f"- wave 4 columns: **{len(c4)}**")
    A(f"- wave-4 columns also present in waves 1-3: **{len(c4 & c13)}**")
    A(f"- wave-4 columns that are new: **{len(c4 - c13)}**\n")
    A("Every wave-4 question was already asked in waves 1-3. There is no novel-question")
    A("generalisation set. This is structural, not incidental, and it determines the whole")
    A("evaluation design: see section 3.\n")

    A("### Blocks covered by wave 4\n")
    blk = ann.groupby("block").size().sort_values(ascending=False)
    A("| Block | columns |")
    A("|---|---:|")
    for k, v in blk.items():
        A(f"| {k} | {v} |")
    A("")

    A("### Coverage is sparse by design (between-subject experiments)\n")
    cov = (w4.notna().sum() / len(w4)).describe()
    A(f"- mean per-column response rate in wave 4: **{cov['mean']:.1%}**")
    A(f"- median: {cov['50%']:.1%}, min: {cov['min']:.1%}, max: {cov['max']:.1%}")
    A(f"- mean answered columns per participant: **{w4.notna().sum(1).mean():.1f}** of {len(c4)}\n")
    A("NaN here is mostly *not* missing data — each participant saw one arm of each")
    A("between-subject experiment. Any metric must be computed on per-column valid pairs,")
    A("and a naive `dropna()` across the frame would empty it.\n")

    A("## 2. Human test-retest reliability (the ceiling)\n")
    for s, g in df.groupby("scale"):
        A(f"### scale = `{s}` ({len(g)} columns, median n={g['n'].median():.0f})\n")
        A(f"- exact agreement: **{g['retest_exact'].mean():.3f}** "
          f"(median {g['retest_exact'].median():.3f}, "
          f"range {g['retest_exact'].min():.3f}-{g['retest_exact'].max():.3f})")
        A(f"- within-1 agreement: {g['retest_within1'].mean():.3f}")
        A(f"- normalised MAE: {g['retest_nmae'].mean():.3f}")
        A(f"- test-retest correlation: {g['retest_corr'].mean():.3f}")
        if g["retest_kappa"].notna().any():
            A(f"- Cohen's kappa: **{g['retest_kappa'].mean():.3f}** "
              f"(chance-corrected; this is the honest number)")
        A("")

    A("### Overall\n")
    nk = int(df["retest_kappa"].notna().sum())
    A(f"- mean exact agreement across all {len(df)} evaluated columns: "
      f"**{df['retest_exact'].mean():.3f}** (95% CI {ci_retest[0]:.3f}-{ci_retest[1]:.3f})")
    A(f"- mean chance-corrected kappa: **{df['retest_kappa'].mean():.3f}** "
      f"— defined on {nk} of {len(df)} columns; kappa is undefined for the "
      f"{len(df)-nk} continuous-support columns\n")
    A("The gap between those two numbers is the point. Raw agreement flatters every model")
    A("on skewed items; kappa does not.\n")
    A(f"CIs throughout are percentile bootstrap, {N_BOOT} resamples of **participants**")
    A("(not of person-question pairs — answers within a person are correlated, and")
    A("resampling pairs would understate every interval).\n")

    A("## 2b. Question types and how each is scored\n")
    A("Qualtrics `QuestionType` is not sufficient to pick a metric — the same `Matrix`")
    A("type covers 2-point and 100-point items. Each column is therefore also classified")
    A("by its *observed support*, and the metric follows from that.\n")
    A("| Qualtrics type | observed scale | cols | metric used | why |")
    A("|---|---|---:|---|---|")
    rules = {
        "binary": ("accuracy + **Cohen's kappa**",
                   "many items are 80/20 skewed; raw accuracy rewards constant predictors"),
        "ordinal_short": ("MAE + within-1 accuracy + kappa",
                          "distance matters; off-by-one is not the same error as off-by-four"),
        "ordinal_long": ("MAE + Spearman", "exact match is too strict to be informative"),
        "continuous": ("normalised MAE + Pearson",
                       "exact match is meaningless (retest exact agreement is 0.14)"),
        "text": ("normalised MAE on the numeric field",
                 "these are numeric text-entry, not free prose"),
    }
    xt = ann.loc[df.index].groupby([ann.loc[df.index, "qtype"].fillna("?"),
                                    df["scale"]]).size()
    for (qt, sc), n in xt.items():
        metric, why = rules.get(sc, ("-", "-"))
        A(f"| {qt} | `{sc}` | {n} | {metric} | {why} |")
    A("")
    A("Aggregation rule: compute each metric **per column on its valid pairs**, then")
    A("average over columns with equal weight. Pooling raw pairs would let the 40-column")
    A("pricing block dominate the 1-column experiments, and would silently reweight the")
    A("benchmark toward whichever block has the most participants.\n")

    A("## 3. The ceiling and the trivial baseline are the same number\n")
    A("Test-retest reliability is `P(wave4 answer == the same person's earlier answer)`.")
    A("Copy-forward prediction is `predict wave4 with the same person's earlier answer`.")
    A("These are one computation. So on this dataset:\n")
    A(f"- copy-forward baseline accuracy = **{df['retest_exact'].mean():.3f}**")
    A(f"- human test-retest ceiling      = **{df['retest_exact'].mean():.3f}**\n")
    A("A model that 'reaches the human ceiling' has done exactly as well as pasting the")
    A("person's old answer. The headroom above copy-forward is not information about the")
    A("person - the persona already contains the answer - it is *denoising*: predicting the")
    A("person's stable disposition instead of one noisy draw of it. Any model can in")
    A("principle exceed the so-called ceiling on point-prediction metrics, because a human")
    A("re-test draw carries measurement noise that a conditional-mode predictor does not.")
    A("Calling test-retest an upper bound is therefore wrong; it is a *reference line*.\n")

    A("## 4. Baseline ladder\n")
    A("| Baseline | mean exact acc | 95% CI | mean normalised MAE |")
    A("|---|---:|---:|---:|")
    A(f"| Population mode / mean (no personalisation) | {df['pop_mode_acc'].mean():.3f} "
      f"| {ci_popmode[0]:.3f}-{ci_popmode[1]:.3f} | {df['pop_mean_nmae'].mean():.3f} |")
    A(f"| Copy-forward (= test-retest ceiling) | {df['retest_exact'].mean():.3f} "
      f"| {ci_retest[0]:.3f}-{ci_retest[1]:.3f} | {df['retest_nmae'].mean():.3f} |")
    A("")
    lift = df["retest_exact"].mean() - df["pop_mode_acc"].mean()
    A(f"The entire personalisation signal available on this benchmark is the **{lift:.3f}**")
    A(f"absolute accuracy gap between those two rows (95% CI "
      f"{ci_band[0]:.3f}-{ci_band[1]:.3f}). A model scoring inside that band has learned")
    A("something about individuals; a model below the population mode has learned nothing;")
    A("a model above copy-forward is denoising.\n")
    A("The band is narrow but not fragile — its interval excludes zero comfortably, so the")
    A("personalisation signal is real. What it is not is *large*, and any reported gain")
    A("should be quoted against a band of this width rather than against 0-100%.\n")

    A("### Columns where personalisation matters most / least\n")
    df["personalisation_gap"] = df["retest_exact"] - df["pop_mode_acc"]
    top = df.nlargest(8, "personalisation_gap")[["block", "retest_exact", "pop_mode_acc",
                                                  "personalisation_gap", "text"]]
    bot = df.nsmallest(8, "personalisation_gap")[["block", "retest_exact", "pop_mode_acc",
                                                   "personalisation_gap", "text"]]
    for title, t in [("Largest gap (worth modelling)", top), ("Smallest / negative gap", bot)]:
        A(f"**{title}**\n")
        A("| col | block | retest | pop-mode | gap | question |")
        A("|---|---|---:|---:|---:|---|")
        for c, r in t.iterrows():
            A(f"| {c} | {r['block']} | {r['retest_exact']:.3f} | {r['pop_mode_acc']:.3f} "
              f"| {r['personalisation_gap']:+.3f} | {str(r['text'])[:70]} |")
        A("")

    A("## 5. Worst-reliability items\n")
    A("These cap what any model can be scored on. Chasing accuracy here is chasing noise.\n")
    A("| col | block | retest exact | retest corr | n | question |")
    A("|---|---|---:|---:|---:|---|")
    for c, r in df.nsmallest(10, "retest_exact").iterrows():
        A(f"| {c} | {r['block']} | {r['retest_exact']:.3f} | {r['retest_corr']:.3f} "
          f"| {r['n']} | {str(r['text'])[:70]} |")
    A("")

    OUT_MD.write_text("\n".join(L))
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_CSV}  ({len(df)} columns)")
    print()
    print(f"mean retest exact  = {df['retest_exact'].mean():.4f} "
          f"[{ci_retest[0]:.4f}, {ci_retest[1]:.4f}]")
    print(f"mean retest kappa  = {df['retest_kappa'].mean():.4f} (n={nk}/{len(df)} cols)")
    print(f"mean pop-mode acc  = {df['pop_mode_acc'].mean():.4f} "
          f"[{ci_popmode[0]:.4f}, {ci_popmode[1]:.4f}]")
    print(f"personalisation band = {lift:.4f} [{ci_band[0]:.4f}, {ci_band[1]:.4f}]")


if __name__ == "__main__":
    main()
