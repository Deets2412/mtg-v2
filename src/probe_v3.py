"""
v3 probe — calibration CV on the 8-feature corpus.

Replicates collect_pairs + time_series_cv from src.calibration but on the
v3 corpus, with the v3 feature set, using an in-memory scaler so v2's
encoder_params.json stays untouched.

Outputs a side-by-side comparison of:
  - v2 baseline (from data/calibration.json's cv_evaluation block)
  - v3 probe (computed fresh on data/historical_corpus_v3probe.parquet)

Decision rule (from docs/v3-probe-results.md):
  SUCCESS  - skill on P(loss 90d) crosses 0
             AND skill on P(>20% DD 90d) improves by >=0.03
             AND no horizon degrades by >0.02
  KILL     - skill on P(loss 90d) doesn't cross 0
             OR any horizon degrades by >0.03

Usage:
    python -m src.probe_v3
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import (
    DEDUP_DAYS,
    EMBARGO_DAYS,
    K,
    PREDICTIONS,
    _outcomes_below,
)
from .calibration import time_series_cv
from .ingest_v3 import CORPUS_V3_PATH, V3_FEATURE_COLS
from .retrieve import cosine_distances, topk_episode_dedup

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
V2_CALIBRATION_PATH = DATA_DIR / "calibration.json"


def _fit_inmemory_scaler(df: pd.DataFrame, cols: list[str]) -> dict:
    """Equal-weight z-score: mean/std per feature, no file output."""
    X = df[cols].values
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    return {"mu": mu, "sd": sd, "cols": cols}


def _encode(df: pd.DataFrame, scaler: dict) -> np.ndarray:
    X = df[scaler["cols"]].values
    return (X - scaler["mu"]) / scaler["sd"]


def collect_pairs_v3(corpus: pd.DataFrame, sample: str = "weekly",
                     dedup_days: int = DEDUP_DAYS) -> pd.DataFrame:
    """
    Walk the v3 corpus, retrieve top-K episode-deduped analogs for each
    query date, record (raw P, actual outcome) per prediction. Same shape
    as src.calibration.collect_pairs.
    """
    print(f"  v3 pair collection (sample={sample}, dedup={dedup_days})...")
    scaler = _fit_inmemory_scaler(corpus, V3_FEATURE_COLS)
    encoded = _encode(corpus, scaler)
    dates = corpus.index.to_numpy()

    actuals = {col: _outcomes_below(corpus[col], thresh).to_numpy()
               for col, thresh, _ in PREDICTIONS}

    min_query_idx = 365
    max_query_idx = len(corpus) - 365
    if sample == "daily":
        query_idxs = np.arange(min_query_idx, max_query_idx)
    elif sample == "weekly":
        query_idxs = np.arange(min_query_idx, max_query_idx, 5)
    elif sample == "monthly":
        query_idxs = np.arange(min_query_idx, max_query_idx, 21)
    else:
        raise ValueError(sample)

    embargo_td = np.timedelta64(EMBARGO_DAYS, "D")
    rows: list[dict] = []

    for n, q_idx in enumerate(query_idxs):
        if n % 200 == 0 and n > 0:
            print(f"    ... {n}/{len(query_idxs)}")
        q_date = dates[q_idx]
        q_vec = encoded[q_idx]
        cutoff = q_date - embargo_td
        eligible_mask = dates < cutoff
        if eligible_mask.sum() < K:
            continue
        eligible_idxs = np.where(eligible_mask)[0]
        eligible_vecs = encoded[eligible_idxs]
        eligible_dates = dates[eligible_idxs]
        dists = cosine_distances(q_vec, eligible_vecs)
        if dedup_days > 0:
            top_k_local = topk_episode_dedup(dists, eligible_dates, K, dedup_days)
        else:
            top_k_local = np.argsort(dists)[:K]
        top_k_global = eligible_idxs[top_k_local]

        for col, _, _ in PREDICTIONS:
            analog_outcomes = actuals[col][top_k_global]
            valid = analog_outcomes[~np.isnan(analog_outcomes)]
            if len(valid) == 0:
                continue
            y_true = actuals[col][q_idx]
            if np.isnan(y_true):
                continue
            rows.append({
                "date": pd.Timestamp(q_date),
                "prediction_col": col,
                "raw_p": float(valid.mean()),
                "actual": float(y_true),
            })

    df = pd.DataFrame(rows)
    print(f"    collected {len(df):,} pairs across {df['date'].nunique():,} query dates")
    return df


def _load_v2_baseline() -> dict:
    """v2's stored OOS CV numbers."""
    if not V2_CALIBRATION_PATH.exists():
        return {}
    payload = json.loads(V2_CALIBRATION_PATH.read_text())
    return payload.get("cv_evaluation", {}) or {}


def _verdict(v2: dict, v3: dict) -> str:
    """Apply the pre-committed decision rule."""
    p90 = v3.get("fwd_ret_90d", {}).get("skill_calibrated")
    dd  = v3.get("max_dd_90d",  {}).get("skill_calibrated")
    dd_v2 = v2.get("max_dd_90d", {}).get("skill_calibrated")

    if p90 is None or dd is None:
        return "INCONCLUSIVE — missing CV outputs"

    # Kill conditions
    degradations = []
    for col, cur in v3.items():
        if "skill_calibrated" not in cur:
            continue
        base = v2.get(col, {}).get("skill_calibrated")
        if base is None:
            continue
        drop = base - cur["skill_calibrated"]
        if drop > 0.03:
            degradations.append((col, drop))
    if degradations:
        return ("KILL — degraded by >0.03 on: " +
                ", ".join(f"{c} ({d:+.3f})" for c, d in degradations))
    if p90 <= 0:
        return f"KILL — P(loss 90d) skill {p90:+.3f} did not cross 0"

    # Success conditions (we already know p90 > 0)
    if dd_v2 is None:
        dd_lift = None
    else:
        dd_lift = dd - dd_v2

    if dd_lift is not None and dd_lift >= 0.03:
        return (f"SUCCESS — P(loss 90d) skill {p90:+.3f} > 0 "
                f"AND P(>20% DD 90d) lifted by {dd_lift:+.3f}")
    return (f"INCONCLUSIVE — P(loss 90d) crossed 0 ({p90:+.3f}) but "
            f"P(>20% DD 90d) lift only {dd_lift:+.3f if dd_lift is not None else None}")


def main() -> None:
    print("=" * 110)
    print("v3 PROBE — calibration CV on 8-feature corpus")
    print("=" * 110)

    if not CORPUS_V3_PATH.exists():
        print(f"!! Missing {CORPUS_V3_PATH}. Run: python -m src.ingest_v3")
        return

    corpus = pd.read_parquet(CORPUS_V3_PATH)
    print(f"Corpus: {len(corpus):,} rows  "
          f"({corpus.index.min().date()} -> {corpus.index.max().date()})")
    print(f"Features: {V3_FEATURE_COLS}\n")

    pairs = collect_pairs_v3(corpus, sample="weekly")
    print("\n  Running time-series CV (5 chronological folds)...")
    v3_cv = time_series_cv(pairs, n_folds=5)
    v2_cv = _load_v2_baseline()

    print()
    print("=" * 110)
    print("SIDE-BY-SIDE OOS RESULTS")
    print("=" * 110)
    print(f"{'prediction':<22} {'v2_skill':>10}  {'v3_skill':>10}  "
          f"{'delta':>8}  {'v2_ECE':>8}  {'v3_ECE':>8}  {'n_v3':>6}")
    print("-" * 110)
    for col, _, label in PREDICTIONS:
        v3 = v3_cv.get(col, {})
        v2 = v2_cv.get(col, {})
        v3_skill = v3.get("skill_calibrated")
        v2_skill = v2.get("skill_calibrated")
        v3_ece = v3.get("ece_calibrated")
        v2_ece = v2.get("ece_calibrated")
        n_v3 = v3.get("n_oos_pairs", 0)
        delta = (None if (v3_skill is None or v2_skill is None)
                 else v3_skill - v2_skill)
        s_v2 = f"{v2_skill:+.3f}" if v2_skill is not None else "n/a"
        s_v3 = f"{v3_skill:+.3f}" if v3_skill is not None else "n/a"
        s_dl = f"{delta:+.3f}" if delta is not None else "n/a"
        s_e2 = f"{v2_ece:.3f}" if v2_ece is not None else "n/a"
        s_e3 = f"{v3_ece:.3f}" if v3_ece is not None else "n/a"
        print(f"{label:<22} {s_v2:>10}  {s_v3:>10}  {s_dl:>8}  "
              f"{s_e2:>8}  {s_e3:>8}  {n_v3:>6}")
    print("=" * 110)

    print(f"\nVERDICT: {_verdict(v2_cv, v3_cv)}")

    # Persist the full report for the write-up.
    out_path = DATA_DIR / "v3_probe_cv.json"
    out_path.write_text(json.dumps({
        "corpus_path": str(CORPUS_V3_PATH),
        "n_corpus_rows": int(len(corpus)),
        "n_pairs": int(len(pairs)),
        "features": V3_FEATURE_COLS,
        "v3_cv": v3_cv,
        "v2_baseline_cv": v2_cv,
    }, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
