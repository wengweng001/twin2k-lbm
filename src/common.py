"""Shared loaders and column typing for the Twin-2K-500 analysis."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / "data/raw/question_catalog_and_human_response_csv"
SIM_DIR = ROOT / "data/raw/LLM_simulation_results"
WAVE_SPLIT_DIR = ROOT / "data/raw/wave_split/chunks"
FULL_PERSONA_DIR = ROOT / "data/raw/full_persona/chunks"


@lru_cache(maxsize=None)
def load_responses(labels: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    suffix = "_label" if labels else ""
    w13 = pd.read_csv(CSV_DIR / f"wave1_3_response{suffix}.csv", low_memory=False).set_index("pid")
    w4 = pd.read_csv(CSV_DIR / f"wave4_response{suffix}.csv", low_memory=False).set_index("pid")
    return w13, w4


@lru_cache(maxsize=None)
def load_catalog() -> list[dict]:
    return json.loads((CSV_DIR / "question_catalog.json").read_text())


@lru_cache(maxsize=None)
def column_meta() -> pd.DataFrame:
    """One row per CSV column, carrying the question metadata it came from."""
    rows = []
    for q in load_catalog():
        n_opt = len(q.get("Options") or [])
        for col in q.get("csv_columns") or []:
            rows.append(
                {
                    "col": col,
                    "qid": q["QuestionID"],
                    "qtype": q.get("QuestionType"),
                    "block": q.get("BlockName"),
                    "source": q.get("source"),
                    "n_options": n_opt,
                    "text": (q.get("QuestionText") or "")[:160],
                }
            )
    return pd.DataFrame(rows).drop_duplicates("col").set_index("col")


def scale_type(col: str, values: pd.Series) -> str:
    """Classify a column into a measurement family that dictates its metric.

    Driven by observed support rather than declared Qualtrics type: the same
    QuestionType is used for 2-point and 100-point items.
    """
    v = values.dropna()
    if v.empty:
        return "empty"
    n_unique = v.nunique()
    is_int = np.allclose(v, v.round())
    if col.endswith("_TEXT") and not is_int:
        return "text"
    if n_unique <= 2:
        return "binary"
    if is_int and n_unique <= 11 and v.max() <= 11:
        return "ordinal_short"
    if is_int and n_unique <= 30:
        return "ordinal_long"
    return "continuous"


def bootstrap_ci(stat_fn, pids: np.ndarray, n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI, resampling *participants* — the independent unit.

    Resampling (person, question) pairs would ignore within-person correlation and
    understate the interval.
    """
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        draw = rng.choice(pids, size=len(pids), replace=True)
        v = stat_fn(draw)
        if v == v:  # skip NaN
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


# US adult population marginals, for the representativeness check.
# Sources: US Census Bureau ACS 2023 1-year estimates and Current Population Survey
# 2023 (educational attainment, age 25+). Figures are approximate and used only to
# show direction and rough size of sample skew, not for reweighting.
US_BENCHMARKS: dict[str, dict[str, float]] = {
    "age": {"18-29": 0.206, "30-49": 0.331, "50-64": 0.243, "65+": 0.220},
    "sex": {"Male": 0.494, "Female": 0.506},
    "region": {"South": 0.384, "West": 0.238, "Midwest": 0.207, "Northeast": 0.171},
    "education": {
        "Less than high school": 0.090,
        "High school graduate": 0.275,
        "Some college, no degree": 0.148,
        "Associate's degree": 0.087,
        "College graduate/some postgrad": 0.234,
        "Postgraduate": 0.146,
    },
    "race": {"White": 0.587, "Hispanic": 0.194, "Black": 0.123, "Asian": 0.062},
}


def annotate_columns(w13: pd.DataFrame, w4: pd.DataFrame) -> pd.DataFrame:
    """Per-column table for the 126 evaluated (wave-4) columns."""
    meta = column_meta()
    out = []
    for col in w4.columns:
        a, b = w13[col], w4[col]
        both = a.notna() & b.notna()
        m = meta.loc[col] if col in meta.index else None
        out.append(
            {
                "col": col,
                "qid": None if m is None else m["qid"],
                "qtype": None if m is None else m["qtype"],
                "block": None if m is None else m["block"],
                "text": None if m is None else m["text"],
                "n_both": int(both.sum()),
                "n_w13": int(a.notna().sum()),
                "n_w4": int(b.notna().sum()),
                "scale": scale_type(col, pd.concat([a, b])),
                "support": float(pd.concat([a, b]).dropna().nunique()),
                "vmin": float(pd.concat([a, b]).min()),
                "vmax": float(pd.concat([a, b]).max()),
            }
        )
    return pd.DataFrame(out).set_index("col")
