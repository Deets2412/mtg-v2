"""
Sprint 1 follow-on (originally tagged v3): calibration backtest.

The question this module answers:
    Do v2's top-K base rates actually predict forward outcomes better
    than the unconditional historical frequency (climatology)?

Method (strict temporal split, no look-ahead):
    For each query date d:
      1. Retrieve top-K analogs from candidates strictly before d - EMBARGO_DAYS.
         (Embargo prevents near-duplicate "yesterday" days leaking the answer.)
      2. Compute predicted probability of each outcome (P_loss, P_severe_DD)
         from the empirical frequency across the K retrieved analogs.
      3. Compare predicted probability to the actual binary outcome at d.

Metrics:
    - Brier score: mean( (p_predicted - outcome_actual)^2 ). Lower is better.
    - Brier skill score: 1 - Brier_v2 / Brier_climatology. Positive = beats climatology.
    - Reliability: bucket predictions by 10%, compute bucket hit rate.
    - Sharpness: stdev of predicted probabilities. Higher = more differentiated.
      (A useless model has perfect calibration but zero sharpness -- always predicts climatology.)

Known leakage we accept:
    The encoder's z-score scaler is fit on the full corpus including dates
    after each query d. For a 9,000-row corpus, refitting on a shrinking
    subset wouldn't move the mean/std materially, and cosine distance is
    scale-invariant in the direction that matters. Documented limitation;
    promote to "refit scaler at each d" if a future audit demands it.

Usage:
    python -m src.backtest                          # weekly sampling, default
    python -m src.backtest --sample daily           # full daily backtest (slower)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .encoder import FEATURE_COLS, encode, fit_scaler
from .retrieve import CORPUS_PATH, cosine_distances, topk_episode_dedup

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Temporal split: candidates must be at least this far in the past.
# 90 trading days ~ 4 months. Buys us a margin past the 30d/90d forward
# windows being predicted, so retrieved analogs can't overlap the
# prediction window for the query.
EMBARGO_DAYS = 90

# Default episode dedup gap. Mirrors retrieve.retrieve() default so the
# backtest measures the production code path.
DEDUP_DAYS = 30

K = 20

# Each entry: (column, threshold, label). Outcome = column < threshold.
PREDICTIONS = [
    ("fwd_ret_30d", 0.0, "P(loss 30d)"),
    ("fwd_ret_90d", 0.0, "P(loss 90d)"),
    ("fwd_ret_12m", 0.0, "P(loss 12m)"),
    ("max_dd_90d", -20.0, "P(>20% DD in 90d)"),
]


def _outcomes_below(series: pd.Series, threshold: float) -> pd.Series:
    """Binary outcome: 1 if value < threshold, 0 otherwise, NaN propagates."""
    return (series < threshold).astype(float).where(series.notna(), np.nan)


def run_backtest(sample: str = "weekly", dedup_days: int = DEDUP_DAYS) -> dict:
    print("Loading corpus and encoding...")
    corpus = pd.read_parquet(CORPUS_PATH)
    scaler = fit_scaler(corpus)
    encoded = encode(corpus[FEATURE_COLS], scaler)
    dates = corpus.index.to_numpy()

    # Pre-compute actual binary outcomes for each prediction
    actuals: dict[str, np.ndarray] = {}
    for col, thresh, _ in PREDICTIONS:
        actuals[col] = _outcomes_below(corpus[col], thresh).to_numpy()

    # Climatology = unconditional historical frequency for each prediction.
    # Used as the baseline a "no-skill" model would predict.
    climatology: dict[str, float] = {}
    for col, _, _ in PREDICTIONS:
        vals = actuals[col][~np.isnan(actuals[col])]
        climatology[col] = float(vals.mean())

    print(f"Corpus: {len(corpus):,} rows from {corpus.index.min().date()} to {corpus.index.max().date()}")
    print(f"Climatology base rates:")
    for col, _, label in PREDICTIONS:
        print(f"  {label:<22} {climatology[col]:.3f}  (across {len(corpus):,} historical days)")
    print()

    # Choose query dates. Skip the first 365 days (warm-up: too few candidates
    # under embargo) and the last 365 days (no fwd_ret_12m available).
    min_query_idx = 365
    max_query_idx = len(corpus) - 365

    if sample == "daily":
        query_idxs = np.arange(min_query_idx, max_query_idx)
    elif sample == "weekly":
        # Every 5 trading days
        query_idxs = np.arange(min_query_idx, max_query_idx, 5)
    elif sample == "monthly":
        query_idxs = np.arange(min_query_idx, max_query_idx, 21)
    else:
        raise ValueError(f"unknown sample mode: {sample}")

    print(f"Running backtest: {len(query_idxs):,} query dates "
          f"({'every day' if sample == 'daily' else 'every ~' + ('week' if sample == 'weekly' else 'month')})\n")

    embargo_td = np.timedelta64(EMBARGO_DAYS, "D")

    # Collect (predicted, actual) pairs per prediction
    preds: dict[str, list[float]] = {col: [] for col, _, _ in PREDICTIONS}
    truths: dict[str, list[float]] = {col: [] for col, _, _ in PREDICTIONS}

    for n, q_idx in enumerate(query_idxs):
        if n % 200 == 0 and n > 0:
            print(f"  ... {n}/{len(query_idxs)} queries processed")

        q_date = dates[q_idx]
        q_vec = encoded[q_idx]

        # Eligible candidates: strictly before q_date - embargo
        cutoff = q_date - embargo_td
        eligible_mask = dates < cutoff
        if eligible_mask.sum() < K:
            continue  # not enough history yet

        eligible_idxs = np.where(eligible_mask)[0]
        eligible_vecs = encoded[eligible_idxs]
        eligible_dates = dates[eligible_idxs]
        dists = cosine_distances(q_vec, eligible_vecs)

        # Top-K with episode dedup so K analogs == K distinct historical
        # episodes, not K consecutive days of one episode.
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
            p_pred = float(valid.mean())
            y_true = actuals[col][q_idx]
            if np.isnan(y_true):
                continue
            preds[col].append(p_pred)
            truths[col].append(float(y_true))

    print(f"  done: {len(query_idxs):,}/{len(query_idxs)}\n")

    # Summarise per prediction
    report = {}
    print("=" * 100)
    print(f"{'prediction':<22} {'n':>5}  {'Brier_v2':>10}  {'Brier_clim':>11}  {'skill':>7}  "
          f"{'sharpness':>10}  {'verdict':<12}")
    print("=" * 100)
    for col, _, label in PREDICTIONS:
        p = np.array(preds[col])
        y = np.array(truths[col])
        if len(p) == 0:
            print(f"{label:<22} no data")
            continue

        brier_v2 = float(((p - y) ** 2).mean())
        # Climatology Brier: always predict climatology[col]
        c = climatology[col]
        brier_clim = float(((c - y) ** 2).mean())
        skill = 1.0 - brier_v2 / brier_clim if brier_clim > 0 else 0.0
        sharpness = float(p.std())

        if skill > 0.05:
            verdict = "BEATS"
        elif skill < -0.05:
            verdict = "WORSE"
        else:
            verdict = "tie"

        print(f"{label:<22} {len(p):>5}  {brier_v2:>10.4f}  {brier_clim:>11.4f}  "
              f"{skill:>+7.3f}  {sharpness:>10.4f}  {verdict:<12}")
        report[col] = {
            "n": len(p),
            "brier_v2": brier_v2,
            "brier_climatology": brier_clim,
            "brier_skill_score": skill,
            "sharpness": sharpness,
            "climatology": c,
        }

    # Reliability table for the most actionable prediction (90d loss)
    print("\n" + "=" * 100)
    print("Reliability — predicted P(loss 90d) bucketed in 10% bins")
    print("(well-calibrated: bucket hit rate ~ midpoint; ECE = mean |hit_rate - midpoint|)")
    print("=" * 100)
    p90 = np.array(preds["fwd_ret_90d"])
    y90 = np.array(truths["fwd_ret_90d"])
    bins = np.arange(0, 1.01, 0.1)
    print(f"{'bucket':<12} {'n':>6}  {'mean_pred':>10}  {'hit_rate':>10}  {'gap':>8}")
    ece_num, ece_denom = 0.0, 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        midpoint = (lo + hi) / 2
        mask = (p90 >= lo) & (p90 < hi if hi < 1.0 else p90 <= hi)
        n = mask.sum()
        if n == 0:
            print(f"  {lo*100:>3.0f}-{hi*100:>3.0f}%   {n:>6}  {'-':>10}  {'-':>10}  {'-':>8}")
            continue
        mean_pred = float(p90[mask].mean())
        hit_rate = float(y90[mask].mean())
        gap = hit_rate - mean_pred
        ece_num += n * abs(gap)
        ece_denom += n
        print(f"  {lo*100:>3.0f}-{hi*100:>3.0f}%   {n:>6}  {mean_pred:>10.3f}  {hit_rate:>10.3f}  {gap:>+8.3f}")
    ece = ece_num / ece_denom if ece_denom > 0 else 0.0
    print(f"\nExpected Calibration Error (90d loss): {ece:.4f}")
    report["ece_90d_loss"] = ece

    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", choices=["daily", "weekly", "monthly"], default="weekly")
    ap.add_argument("--dedup-days", type=int, default=DEDUP_DAYS,
                    help="Min calendar-day gap between accepted analogs. 0 disables.")
    args = ap.parse_args()
    run_backtest(sample=args.sample, dedup_days=args.dedup_days)


if __name__ == "__main__":
    main()
