"""
v3 probe — ingest with cross-asset features.

Adds four cross-asset features to the v2 corpus:
  - slope_2y10y         : DGS10 - DGS2 (Treasury curve slope, percentage points)
  - credit_spread       : BAA - AAA   (Moody's corporate spread, percentage points)
  - credit_spread_chg_90d : 90 trading-day change in credit_spread
  - dxy                 : ICE Dollar Index spot close (level)

Why these four (decided in the probe scope, see docs/v3-probe-results.md):
  - 2y10y slope: well-documented recession leading indicator. Slope (not levels)
    avoids the regime-shift confound between ZIRP and 5%+ rates.
  - BAA-AAA spread: pre-1996 proxy for HY OAS. HY OAS only goes back to 1996,
    which would chop 6 years off the corpus and break the smell tests
    (Oct 2008 ↔ March 2020). BAA-AAA goes back to 1953 and is correlated.
  - 90d change in credit spread: a spread of 100bps means something different
    when it just widened from 50bps vs. when it just tightened from 200bps.
  - DXY: dollar strength as a global liquidity/risk proxy.

Sources:
  - DGS2, DGS10, BAA, AAA: FRED (no auth, public CSV endpoint)
  - DXY: Yahoo Finance ticker "DX-Y.NYB" (ICE Dollar Index)

Output: data/historical_corpus_v3probe.parquet
        (separate file — v2's data/historical_corpus.parquet is untouched)

Usage:
    python -m src.ingest_v3
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from .ingest import (
    DATA_DIR,
    compute_dd_from_52w_high,
    compute_forward_outcomes,
    compute_spx_ytd,
    fetch_market_data,
    flag_macro_shock,
)

warnings.filterwarnings("ignore", category=FutureWarning)

CORPUS_V3_PATH = DATA_DIR / "historical_corpus_v3probe.parquet"

V3_FEATURE_COLS = [
    "vix",
    "spx_ytd_pct",
    "spx_dd_from_52w",
    "macro_shock",
    "slope_2y10y",
    "credit_spread",
    "credit_spread_chg_90d",
    "dxy",
]


# ---------------------------------------------------------------------------
# FRED fetcher
# ---------------------------------------------------------------------------

def fetch_fred_series(series_id: str, start: str = "1989-01-01") -> pd.Series:
    """
    Pull a daily series from FRED's public CSV endpoint.

    Returns a date-indexed Series with NaN for missing/holiday days.
    Indexed by naive timestamps (no timezone).
    """
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={start}"
    )
    print(f"  Fetching FRED {series_id} since {start}...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(io.BytesIO(resp.content))
    # FRED CSV columns: DATE,<series_id>  (older endpoint) OR
    #                   observation_date,<series_id> (newer fredgraph).
    date_col = "DATE" if "DATE" in df.columns else "observation_date"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    # FRED uses "." for missing observations.
    series = pd.to_numeric(df[series_id], errors="coerce")
    series.name = series_id
    print(f"    Got {series.notna().sum():,} non-null observations "
          f"({series.index.min().date()} -> {series.index.max().date()})")
    return series


# ---------------------------------------------------------------------------
# DXY fetcher
# ---------------------------------------------------------------------------

def fetch_dxy(start: str = "1989-01-01") -> pd.Series:
    """ICE Dollar Index spot close. yfinance ticker DX-Y.NYB."""
    print(f"  Fetching DXY (DX-Y.NYB) since {start}...")
    s = yf.Ticker("DX-Y.NYB").history(start=start, auto_adjust=False)["Close"]
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s.name = "dxy"
    s = s.dropna()
    print(f"    Got {len(s):,} rows of DXY data "
          f"({s.index.min().date()} -> {s.index.max().date()})")
    return s


# ---------------------------------------------------------------------------
# Cross-asset feature assembly
# ---------------------------------------------------------------------------

def fetch_cross_asset(spx_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Pull rates + credit + FX series and align them onto the SPX trading-day
    index.

    Daily series (DGS2, DGS10, DXY): forward-fill up to 5 trading days to
    bridge FRED-vs-NYSE holiday differences. Anything bigger is a real gap
    and should stay NaN.

    Monthly series (Moody's BAA, AAA): forward-fill UNLIMITED. These are
    published once per month and hold their value until next observation —
    that's how every consumer of these series treats them. The probe uses
    monthly-resolution credit on purpose because daily HY OAS (BAMLH0A0HYM2)
    only goes back to 1996 and would chop 6 years off the corpus.
    """
    daily_series = {
        "dgs2":  fetch_fred_series("DGS2"),
        "dgs10": fetch_fred_series("DGS10"),
        "dxy":   fetch_dxy(),
    }
    monthly_series = {
        "baa":   fetch_fred_series("BAA"),
        "aaa":   fetch_fred_series("AAA"),
    }

    df = pd.DataFrame(index=spx_index)
    for name, s in daily_series.items():
        df[name] = s.reindex(spx_index).ffill(limit=5)
    for name, s in monthly_series.items():
        df[name] = s.reindex(spx_index).ffill()  # unlimited — monthly values hold
    return df


# ---------------------------------------------------------------------------
# Derived v3 features
# ---------------------------------------------------------------------------

def compute_v3_features(cross_asset: pd.DataFrame) -> pd.DataFrame:
    """Build the v3 feature columns from raw rates/credit/FX series."""
    out = pd.DataFrame(index=cross_asset.index)
    out["slope_2y10y"] = cross_asset["dgs10"] - cross_asset["dgs2"]
    out["credit_spread"] = cross_asset["baa"] - cross_asset["aaa"]
    out["credit_spread_chg_90d"] = out["credit_spread"].diff(63)  # ~90 trading days
    out["dxy"] = cross_asset["dxy"]
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_v3_corpus() -> pd.DataFrame:
    """Assemble historical_corpus_v3probe.parquet (v2 base + cross-asset features)."""
    print("Building v3 probe corpus...\n")

    # v2 base: VIX, SPX, derived equity features
    df = fetch_market_data()
    print("  Computing v2 derived features...")
    df["spx_ytd_pct"] = compute_spx_ytd(df["spx_close"])
    df["spx_dd_from_52w"] = compute_dd_from_52w_high(df["spx_close"])
    df["macro_shock"] = flag_macro_shock(df.index)

    # v3 additions: rates, credit, FX, derived
    print("\n  Fetching cross-asset series...")
    cross = fetch_cross_asset(df.index)
    v3 = compute_v3_features(cross)
    df = df.join(v3)

    # Forward outcomes (unchanged from v2)
    print("\n  Computing forward outcomes...")
    outcomes = compute_forward_outcomes(df["spx_close"])
    df = df.join(outcomes)

    # Drop rows missing any v3 encoder feature. NB: credit_spread_chg_90d needs
    # 63 prior days of credit_spread, so first ~3 months of v3 corpus will go.
    before = len(df)
    df = df.dropna(subset=V3_FEATURE_COLS)
    print(f"\n  Dropped {before - len(df)} rows missing v3 encoder features "
          f"({before:,} -> {len(df):,})")

    # Drop rows missing near-term forward outcomes
    before = len(df)
    df = df.dropna(subset=["fwd_ret_30d", "fwd_ret_90d"])
    print(f"  Dropped {before - len(df)} rows missing near-term forward outcomes "
          f"({before:,} -> {len(df):,})")

    df.to_parquet(CORPUS_V3_PATH)
    print(f"\nWrote {len(df):,} rows -> {CORPUS_V3_PATH}")
    print(f"Date range: {df.index.min().date()} -> {df.index.max().date()}")
    print(f"\nv3 feature summary:")
    print(df[V3_FEATURE_COLS].describe().round(3).to_string())
    return df


if __name__ == "__main__":
    build_v3_corpus()
