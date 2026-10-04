"""Additional comparator baselines for the natural wave1-3 -> wave4 task.

Produces:
  reports/exploration/06_comparator_baselines.md
  data/derived/comparator_baselines.csv

These are deliberately cheap non-LLM baselines. They answer the rubric question
"what should the model be compared against?" more directly than another LLM run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, annotate_columns, column_meta, load_responses  # noqa: E402

OUT_MD = ROOT / "reports/exploration/06_comparator_baselines.md"
OUT_CSV = ROOT / "data/derived/comparator_baselines.csv"
K = 50
MIN_CELL_N = 20
DEMO_COLS = ["QID13", "QID12", "QID14"]


def mode_or_nan(s: pd.Series) -> float:
    m = s.dropna().mode()
    return float(m.iloc[0]) if len(m) else np.nan


def exact_by_col(w4: pd.DataFrame, pred: pd.DataFrame) -> pd.Series:
    vals = {}
    for col in w4.columns:
        ok = w4[col].notna() & pred[col].notna()
        vals[col] = float((w4.loc[ok, col] == pred.loc[ok, col]).mean()) if ok.any() else np.nan
    return pd.Series(vals)


def nmae_by_col(w4: pd.DataFrame, pred: pd.DataFrame) -> pd.Series:
    vals = {}
    for col in w4.columns:
        ok = w4[col].notna() & pred[col].notna()
        if not ok.any():
            vals[col] = np.nan
            continue
        rng = max(pd.concat([w4[col], pred[col]]).max() - pd.concat([w4[col], pred[col]]).min(), 1e-9)
        vals[col] = float((w4.loc[ok, col] - pred.loc[ok, col]).abs().mean() / rng)
    return pd.Series(vals)


def demographic_cell_mode(w13: pd.DataFrame, w4: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Predict each item from same age x sex x education cell, falling back to global mode.

    The participant's own wave1-3 answer to the target item is excluded from both the cell
    and fallback calculation. Otherwise a singleton cell would silently become copy-forward.
    """
    cell = labels[DEMO_COLS].astype(str).agg(" | ".join, axis=1)
    pred = pd.DataFrame(index=w4.index, columns=w4.columns, dtype=float)

    for col in w4.columns:
        values = w13[col]
        global_counts = values.dropna().value_counts()
        if global_counts.empty:
            continue
        global_mode = float(global_counts.idxmax())
        cell_counts = values.groupby(cell).value_counts(dropna=True)

        for pid in w4.index:
            if pd.isna(w4.loc[pid, col]):
                continue
            own_cell = cell.loc[pid]
            try:
                counts = cell_counts.loc[own_cell].copy()
            except KeyError:
                counts = pd.Series(dtype=float)
            own_value = values.loc[pid]
            if pd.notna(own_value) and own_value in counts.index:
                counts.loc[own_value] -= 1
                counts = counts[counts > 0]
            if counts.sum() >= MIN_CELL_N:
                pred.loc[pid, col] = float(counts.idxmax())
                continue

            # Fallback: leave-one-out global mode.
            g = global_counts.copy()
            if pd.notna(own_value) and own_value in g.index:
                g.loc[own_value] -= 1
                g = g[g > 0]
            pred.loc[pid, col] = float(g.idxmax()) if len(g) else global_mode
        if (list(w4.columns).index(col) + 1) % 20 == 0:
            print(f"demographic baseline scored {list(w4.columns).index(col) + 1}/{len(w4.columns)} columns",
                  flush=True)
    return pred


def knn_response_mode(w13: pd.DataFrame, w4: pd.DataFrame, k: int = K) -> pd.DataFrame:
    """kNN over the wave1-3 response vector, excluding the target column.

    Numeric columns are z-scored and missing values are mean-imputed after z-scoring
    (therefore zero). For each target item and person, pick the k most similar *other*
    participants who answered that target in wave1-3, then predict their modal earlier
    answer. This is a cheap "people similar to you answered..." baseline.
    """
    x = w13.apply(pd.to_numeric, errors="coerce")
    mu = x.mean(axis=0)
    sd = x.std(axis=0).replace(0, np.nan)
    z = ((x - mu) / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    z = z.to_numpy(dtype=np.float32)
    full_sim = z @ z.T
    pids = list(w13.index)
    pos = {pid: i for i, pid in enumerate(pids)}
    col_pos = {c: i for i, c in enumerate(w13.columns)}

    pred = pd.DataFrame(index=w4.index, columns=w4.columns, dtype=float)
    for j, col in enumerate(w4.columns):
        xcol = z[:, col_pos[col]] if col in col_pos else np.zeros(len(pids))
        sim = full_sim - np.outer(xcol, xcol)
        np.fill_diagonal(sim, -np.inf)
        has_target = w13[col].notna().to_numpy()

        for pid in w4.index:
            if pd.isna(w4.loc[pid, col]):
                continue
            i = pos[pid]
            eligible = has_target.copy()
            eligible[i] = False
            idx = np.flatnonzero(eligible)
            if len(idx) == 0:
                pred.loc[pid, col] = np.nan
                continue
            kk = min(k, len(idx))
            near = idx[np.argpartition(sim[i, idx], -kk)[-kk:]]
            pred.loc[pid, col] = mode_or_nan(w13.iloc[near][col])
        if (j + 1) % 20 == 0:
            print(f"kNN scored {j + 1}/{len(w4.columns)} columns", flush=True)
    return pred


def main() -> None:
    w13, w4 = load_responses()
    labels, _ = load_responses(labels=True)
    ann = annotate_columns(w13, w4)
    meta = column_meta()

    print("scoring demographic-cell baseline ...")
    pred_demo = demographic_cell_mode(w13, w4, labels)
    print("scoring kNN response-vector baseline ...")
    pred_knn = knn_response_mode(w13, w4)

    base = pd.read_csv(ROOT / "data/derived/per_column_retest.csv").set_index("col")
    out = pd.DataFrame(index=w4.columns)
    out["block"] = ann["block"]
    out["scale"] = ann["scale"]
    out["pop_mode_acc"] = base["pop_mode_acc"]
    out["copy_forward_acc"] = base["retest_exact"]
    out["demo_cell_acc"] = exact_by_col(w4, pred_demo)
    out["knn50_acc"] = exact_by_col(w4, pred_knn)
    out["pop_mean_nmae"] = base["pop_mean_nmae"]
    out["copy_forward_nmae"] = base["retest_nmae"]
    out["demo_cell_nmae"] = nmae_by_col(w4, pred_demo)
    out["knn50_nmae"] = nmae_by_col(w4, pred_knn)
    out["text"] = [meta.loc[c, "text"] if c in meta.index else c for c in out.index]
    out.to_csv(OUT_CSV)

    summary = out[["pop_mode_acc", "demo_cell_acc", "knn50_acc", "copy_forward_acc"]].mean()
    nmae = out[["pop_mean_nmae", "demo_cell_nmae", "knn50_nmae", "copy_forward_nmae"]].mean()
    by_scale = out.groupby("scale")[["pop_mode_acc", "demo_cell_acc", "knn50_acc", "copy_forward_acc"]].mean()

    L: list[str] = []
    A = L.append
    A("# 06 — Additional comparator baselines\n")
    A("These baselines are not model proposals; they are comparators. The point is to make")
    A("the evaluation ladder concrete before spending more compute on LLMs.\n")
    A("## Headline exact-match accuracy\n")
    A("| Comparator | Mean exact accuracy | Mean normalised MAE |")
    A("|---|---:|---:|")
    rows = [
        ("B1 population mode / mean", "pop_mode_acc", "pop_mean_nmae"),
        ("B2 demographic-cell mode", "demo_cell_acc", "demo_cell_nmae"),
        (f"B3 kNN response vector (k={K})", "knn50_acc", "knn50_nmae"),
        ("B4 copy-forward / test-retest", "copy_forward_acc", "copy_forward_nmae"),
    ]
    for name, acc_col, nmae_col in rows:
        A(f"| {name} | {summary[acc_col]:.3f} | {nmae[nmae_col]:.3f} |")
    A("")
    A("B2 uses age x sex x education cells and excludes the participant's own answer.")
    A("B3 uses the participant's other wave1-3 answers to find similar respondents, then")
    A("predicts the modal earlier answer among the 50 nearest neighbours. The target column")
    A("is removed from the similarity calculation, so this is not copy-forward.\n")
    A("## By answer scale\n")
    A("| Scale | B1 pop | B2 demo | B3 kNN | B4 copy-forward |")
    A("|---|---:|---:|---:|---:|")
    for scale, row in by_scale.iterrows():
        A(f"| {scale} | {row['pop_mode_acc']:.3f} | {row['demo_cell_acc']:.3f} | "
          f"{row['knn50_acc']:.3f} | {row['copy_forward_acc']:.3f} |")
    A("")
    best = out.assign(knn_lift=out["knn50_acc"] - out["pop_mode_acc"]).nlargest(8, "knn_lift")
    A("## Where kNN adds the most over population mode\n")
    A("| col | block | pop | kNN | copy-forward | question |")
    A("|---|---|---:|---:|---:|---|")
    for col, row in best.iterrows():
        A(f"| {col} | {row['block']} | {row['pop_mode_acc']:.3f} | {row['knn50_acc']:.3f} | "
          f"{row['copy_forward_acc']:.3f} | {str(row['text'])[:80]} |")
    A("")
    A("## Interpretation\n")
    if summary["knn50_acc"] > summary["demo_cell_acc"]:
        A("The response-vector baseline beats the demographic-cell baseline, which means")
        A("there is useful behavioural signal in the persona beyond demographics. That is")
        A("the signal an LBM should exploit.")
    else:
        A("The response-vector baseline does not beat the demographic-cell baseline, which")
        A("would imply that this cheap similarity heuristic is not extracting much extra")
        A("personalisation signal.")
    A("Neither cheap baseline approaches copy-forward, so the repeated-item benchmark still")
    A("leaves a large denoising gap. These rows should be reported next to any LLM result.")

    OUT_MD.write_text("\n".join(L))
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
