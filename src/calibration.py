"""
Sprint 1 follow-on (originally tagged v3): isotonic recalibration of v2's
raw probabilities.

The backtest showed v2's raw probabilities are over-confident at both
tails: predicts 95% probability of loss -> actual rate ~50%; predicts 2%
-> actual ~20%. Episode dedup closes about half the gap. This module
closes the rest by fitting a per-prediction monotone mapping from raw P
to calibrated P.

Defensibility: the mapping is fit and evaluated with TIME-SERIES CROSS
VALIDATION. For each fold, isotonic is fit on data strictly earlier than
the fold and evaluated on the fold. The reported Brier skill score is
OUT-OF-SAMPLE -- no look-ahead. The final production model is fit on all
data; its expected behaviour is approximated by the CV result.

Output (data/calibration.json):
  {
    "models": {
      "fwd_ret_30d": {"x": [...], "y": [...]},     // isotonic breakpoints
      "fwd_ret_90d": {"x": [...], "y": [...]},
      "fwd_ret_12m": {"x": [...], "y": [...]},
      "max_dd_90d":  {"x": [...], "y": [...]},
    },
    "cv_evaluation": {<per-prediction out-of-sample metrics>},
    "fit_at": ISO-8601,
    "n_pairs": int,
    "fit_params": {"k": 20, "embargo_days": 90, "dedup_days": 30},
  }

publish.py loads this file and applies the mapping at display time via
numpy.interp so no sklearn dependency leaks into the publish path.

Usage:
    python -m src.calibration                  # fit + evaluate + persist
    python -m src.calibration --evaluate-only  # CV report only
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .backtest import (
    DEDUP_DAYS,
    EMBARGO_DAYS,
    K,
    PREDICTIONS,
    _outcomes_below,
)
from .encoder import FEATURE_COLS, encode, fit_scaler
from .retrieve import CORPUS_PATH, cosine_distances, topk_episode_dedup

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CALIBRATION_PATH = DATA_DIR / "calibration.json"


def collect_pairs(sample: str = "weekly", dedup_days: int = DEDUP_DAYS) -> pd.DataFrame:
    """
    Run the backtest retrieval and return a tidy DataFrame:
        columns: date, prediction_col, raw_p, actual
    One row per (query date, prediction) combination.
    """
    print(f"Collecting (raw_p, actual) pairs (sample={sample}, dedup={dedup_days})...")
    corpus = pd.read_parquet(CORPUS_PATH)
    scaler = fit_scaler(corpus)
    encoded = encode(corpus[FEATURE_COLS], scaler)
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
            print(f"  ... {n}/{len(query_idxs)}")
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
    print(f"  collected {len(df):,} pairs across {df['date'].nunique():,} query dates\n")
    return df


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(((p - y) ** 2).mean())


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(p)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        n = mask.sum()
        if n == 0:
            continue
        gap = abs(y[mask].mean() - p[mask].mean())
        ece += n * gap
    return float(ece / n_total) if n_total > 0 else 0.0


def time_series_cv(pairs: pd.DataFrame, n_folds: int = 5) -> dict:
    """
    Chronological cross-validation. Split dates into n_folds chronological
    blocks. For each fold (except the first, which has no training data),
    fit isotonic on prior folds, apply to this fold, collect calibrated
    predictions and compare to raw + climatology.
    """
    pairs = pairs.sort_values("date").reset_index(drop=True)
    unique_dates = pairs["date"].drop_duplicates().sort_values().to_numpy()
    fold_edges = np.array_split(unique_dates, n_folds)
    fold_id_per_date = {}
    for fi, dates_in_fold in enumerate(fold_edges):
        for d in dates_in_fold:
            fold_id_per_date[d] = fi
    pairs["fold"] = pairs["date"].map(fold_id_per_date)

    out: dict = {}
    for col, _, label in PREDICTIONS:
        sub = pairs[pairs["prediction_col"] == col].copy()
        climatology_p = sub["actual"].mean()

        # Collect out-of-sample calibrated predictions for folds 1..n-1
        oos_records: list[dict] = []
        for fi in range(1, n_folds):
            train = sub[sub["fold"] < fi]
            test = sub[sub["fold"] == fi]
            if len(train) < 30 or len(test) == 0:
                continue
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(train["raw_p"].values, train["actual"].values)
            calibrated = iso.transform(test["raw_p"].values)
            for raw, cal, y, d in zip(
                test["raw_p"].values, calibrated, test["actual"].values, test["date"].values
            ):
                oos_records.append({
                    "date": d, "fold": fi, "raw_p": raw,
                    "calibrated_p": float(cal), "actual": float(y),
                })

        oos = pd.DataFrame(oos_records)
        if len(oos) == 0:
            out[col] = {"n": 0, "label": label}
            continue
        y = oos["actual"].to_numpy()
        raw = oos["raw_p"].to_numpy()
        cal = oos["calibrated_p"].to_numpy()
        clim = np.full_like(y, climatology_p)

        out[col] = {
            "label": label,
            "n_oos_pairs": int(len(oos)),
            "climatology": float(climatology_p),
            "brier_raw": brier(raw, y),
            "brier_calibrated": brier(cal, y),
            "brier_climatology": brier(clim, y),
            "skill_raw": 1 - brier(raw, y) / brier(clim, y) if brier(clim, y) > 0 else 0,
            "skill_calibrated": 1 - brier(cal, y) / brier(clim, y) if brier(clim, y) > 0 else 0,
            "ece_raw": expected_calibration_error(raw, y),
            "ece_calibrated": expected_calibration_error(cal, y),
            "sharpness_raw": float(raw.std()),
            "sharpness_calibrated": float(cal.std()),
        }
    return out


def fit_final_models(pairs: pd.DataFrame) -> dict:
    """
    Fit isotonic on ALL data per prediction. Return serializable
    {col: {"x": [...], "y": [...]}} suitable for numpy.interp at runtime.
    """
    models = {}
    for col, _, _ in PREDICTIONS:
        sub = pairs[pairs["prediction_col"] == col]
        if len(sub) < 30:
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(sub["raw_p"].values, sub["actual"].values)
        # Persist as breakpoints: thresholds + values.
        # sklearn exposes X_thresholds_ and y_thresholds_ post-fit.
        x = iso.X_thresholds_.astype(float).tolist()
        y = iso.y_thresholds_.astype(float).tolist()
        models[col] = {"x": x, "y": y}
    return models


def apply_calibration(raw_p: float, model: dict | None) -> float:
    """
    Map a raw probability to a calibrated one via piecewise-linear interp
    over the persisted isotonic breakpoints. Identity if no model.
    """
    if model is None or not model.get("x"):
        return raw_p
    return float(np.clip(np.interp(raw_p, model["x"], model["y"]), 0.0, 1.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", choices=["daily", "weekly", "monthly"], default="weekly")
    ap.add_argument("--evaluate-only", action="store_true",
                    help="Run CV but don't persist final models.")
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    pairs = collect_pairs(sample=args.sample)
    cv_report = time_series_cv(pairs, n_folds=args.n_folds)

    print("=" * 110)
    print("Out-of-sample CV evaluation (chronological folds, no look-ahead)")
    print("=" * 110)
    print(f"{'prediction':<22} {'n':>6}  {'Brier_raw':>10}  {'Brier_cal':>10}  "
          f"{'Brier_clim':>11}  {'skill_raw':>10}  {'skill_cal':>10}  "
          f"{'ECE_raw':>8}  {'ECE_cal':>8}")
    print("=" * 110)
    for col, _, _ in PREDICTIONS:
        r = cv_report.get(col)
        if not r or r.get("n_oos_pairs", 0) == 0:
            continue
        print(f"{r['label']:<22} {r['n_oos_pairs']:>6}  "
              f"{r['brier_raw']:>10.4f}  {r['brier_calibrated']:>10.4f}  "
              f"{r['brier_climatology']:>11.4f}  "
              f"{r['skill_raw']:>+10.3f}  {r['skill_calibrated']:>+10.3f}  "
              f"{r['ece_raw']:>8.4f}  {r['ece_calibrated']:>8.4f}")

    if args.evaluate_only:
        print("\n(--evaluate-only set; not persisting models)")
        return

    print("\nFitting final isotonic models on full dataset...")
    models = fit_final_models(pairs)
    payload = {
        "models": models,
        "cv_evaluation": cv_report,
        "fit_at": datetime.now(timezone.utc).isoformat(),
        "n_pairs": int(len(pairs)),
        "fit_params": {"k": K, "embargo_days": EMBARGO_DAYS, "dedup_days": DEDUP_DAYS},
    }
    CALIBRATION_PATH.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {CALIBRATION_PATH}")


if __name__ == "__main__":
    main()
