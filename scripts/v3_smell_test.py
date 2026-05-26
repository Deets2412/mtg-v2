"""
Ad-hoc smell test for the v3 probe corpus.

Z-scores each feature (equal weight), computes cosine distance from each
target date to all prior dates, prints top-K nearest analogs.

This deliberately does NOT use the existing src.encoder / src.retrieve
modules to avoid polluting v2's encoder_params.json. The probe stays
isolated until it proves itself.

Per docs/v3-probe-results.md — the smell tests are a HARD GATE. If
March 2020 stops pulling October 2008 as a near analog, the v3 features
are not adding signal, they're adding noise that drowns out the
equity-state signal.

Usage:
    python -m scripts.v3_smell_test
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "historical_corpus_v3probe.parquet"

FEATURE_COLS = [
    "vix",
    "spx_ytd_pct",
    "spx_dd_from_52w",
    "macro_shock",
    "slope_2y10y",
    "credit_spread",
    "credit_spread_chg_90d",
    "dxy",
]

# Smell-test target dates and their expected analogs (from CLAUDE.md).
SMELL_TESTS = [
    ("2020-03-16", "COVID crash", ["2008-10", "2011-09", "2018-12", "2018-02"]),
    ("2008-10-13", "Lehman week", ["2020-03", "1998-08", "1998-09"]),
    ("2018-01-26", "January 2018 melt-up", ["1999", "2007"]),
    ("2017-07-03", "Mid-2017 calm", ["2005", "1995"]),
]


def main() -> None:
    df = pd.read_parquet(CORPUS)
    print(f"Loaded {len(df):,} rows from {CORPUS.name}")
    print(f"Range: {df.index.min().date()} -> {df.index.max().date()}\n")

    # Equal-weight z-score.
    X = df[FEATURE_COLS].values
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0  # macro_shock is mostly zero
    Z = (X - mu) / sd

    # Normalize rows for cosine.
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Zn = Z / norms

    for target_str, label, expected_substrings in SMELL_TESTS:
        try:
            target_idx = df.index.get_indexer([pd.Timestamp(target_str)],
                                              method="nearest")[0]
        except Exception as e:
            print(f"!! {target_str} ({label}): could not find date in corpus ({e})")
            continue

        target_date = df.index[target_idx]
        q = Zn[target_idx]

        # Cosine distance to all dates strictly earlier (no look-ahead).
        prior_mask = np.arange(len(df)) < target_idx
        # Also exclude near-window (30 days either side of target) to avoid
        # retrieving "yesterday" as the nearest analog.
        cutoff = target_date - pd.Timedelta(days=30)
        prior_mask &= np.asarray(df.index < cutoff)

        sims = Zn[prior_mask] @ q
        dists = 1 - sims
        prior_dates = df.index[prior_mask]

        # Episode dedup: keep top K but require >= 30 calendar days between
        # selected analogs. Mirrors src.retrieve.topk_episode_dedup.
        sorted_idx = np.argsort(dists)
        picked: list[int] = []
        picked_dates: list[pd.Timestamp] = []
        for idx in sorted_idx:
            d = prior_dates[idx]
            if any(abs((d - pd_).days) < 30 for pd_ in picked_dates):
                continue
            picked.append(idx)
            picked_dates.append(d)
            if len(picked) >= 8:
                break
        order = np.array(picked)

        print(f"=== {target_str} ({label}) ===")
        # Print query state
        print(f"    query state:  VIX={df.iloc[target_idx]['vix']:.1f}  "
              f"YTD={df.iloc[target_idx]['spx_ytd_pct']:+.1f}%  "
              f"DD52w={df.iloc[target_idx]['spx_dd_from_52w']:+.1f}%  "
              f"slope={df.iloc[target_idx]['slope_2y10y']:+.2f}  "
              f"credit={df.iloc[target_idx]['credit_spread']:.2f}  "
              f"DXY={df.iloc[target_idx]['dxy']:.1f}")
        print(f"    top 8 analogs:")
        top_dates = [prior_dates[o] for o in order]
        for rank, (o, d) in enumerate(zip(order, top_dates), 1):
            row = df.loc[d]
            print(f"      {rank}. {d.date()}  dist={dists[o]:.4f}  "
                  f"VIX={row['vix']:.1f} YTD={row['spx_ytd_pct']:+.1f}% "
                  f"DD={row['spx_dd_from_52w']:+.1f}% slope={row['slope_2y10y']:+.2f} "
                  f"credit={row['credit_spread']:.2f}")

        # Check expected hit — match if any top analog falls within +/- 45
        # calendar days of an expected period (handled as the 1st of that
        # month). E.g. "2008-10" matches anything 2008-09-15 to 2008-11-14.
        hit_targets: list[str] = []
        for exp in expected_substrings:
            if len(exp) == 4:                       # year only, e.g. "1999"
                exp_dates = [pd.Timestamp(f"{exp}-06-15")]
                window_days = 200                   # wide — whole year
            else:                                   # "YYYY-MM"
                exp_dates = [pd.Timestamp(f"{exp}-15")]
                window_days = 45
            for d in top_dates:
                if any(abs((d - ed).days) < window_days for ed in exp_dates):
                    hit_targets.append(exp)
                    break
        verdict = "PASS" if hit_targets else "FAIL"
        print(f"    {verdict}  (expected: {expected_substrings}; hit: {hit_targets})\n")


if __name__ == "__main__":
    main()
