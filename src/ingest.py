"""
Sprint 1: Ingest historical market data and build the corpus parquet.

Output:
    data/historical_corpus.parquet

Columns:
    date (index), vix, spx_close,
    spx_ytd_pct, spx_dd_from_52w, macro_shock,
    fwd_ret_30d, fwd_ret_90d, fwd_ret_12m,
    max_dd_90d, max_dd_180d, vol_realized_30d

Note: Fear & Greed was originally a 5th input. CNN's public endpoint only
returns ~1y of history, not since 2011. F&G is deferred to v3
(reconstruct from sub-ingredients). v2 runs on 4 inputs.

Corpus starts 1990-01-01 (VIX inception). Pre-2011 history matters for
the smell tests (March 2020's nearest analog should be Oct 2008).

Usage:
    python -m src.ingest
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
CORPUS_PATH = DATA_DIR / "historical_corpus.parquet"

# Known macro shock dates. Each date triggers a ~5-trading-day shock window.
# Deliberately conservative — extend by hand as needed.
# v3 task: rule-based classification from headlines, not manual.
MACRO_SHOCK_DATES = {
    # Pre-2011 (added with the 1990 corpus extension)
    "1990-08-02",  # Iraq invades Kuwait
    "1997-10-27",  # Asian crisis mini-crash (Dow -7%)
    "1998-08-31",  # LTCM / Russia default fallout
    "2001-09-11",  # 9/11
    "2008-09-15",  # Lehman collapse
    "2008-09-29",  # House rejects TARP (Dow -7%)
    "2008-10-06",  # Worst week of GFC
    "2010-05-06",  # Flash crash
    # 2011 onwards
    "2011-08-05",  # US debt downgrade
    "2015-08-24",  # China devaluation flash crash
    "2016-06-24",  # Brexit
    "2016-11-09",  # Trump election
    "2018-02-05",  # Volmageddon
    "2020-02-24",  # COVID acceleration begins
    "2020-03-09",  # COVID circuit breaker 1
    "2020-03-12",  # COVID circuit breaker 2
    "2020-03-16",  # COVID circuit breaker 3
    "2022-02-24",  # Russia invades Ukraine
    "2023-03-10",  # SVB collapse
}

# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def fetch_market_data(start: str = "1990-01-01") -> pd.DataFrame:
    """Pull VIX and SPX daily closes from yfinance."""
    print(f"  Fetching VIX + SPX since {start}...")

    vix = yf.Ticker("^VIX").history(start=start, auto_adjust=False)["Close"]
    spx = yf.Ticker("^GSPC").history(start=start, auto_adjust=False)["Close"]

    # yfinance returns tz-aware indices; normalise to naive for clean joins.
    for s in (vix, spx):
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)

    df = pd.concat([vix, spx], axis=1)
    df.columns = ["vix", "spx_close"]
    df.index.name = "date"
    df = df.dropna()
    print(f"  Got {len(df):,} rows of market data")
    return df


# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------

def compute_spx_ytd(spx: pd.Series) -> pd.Series:
    """For each date, SPX % return since first trading day of that calendar year."""
    df = spx.to_frame("spx_close").copy()
    df["year"] = df.index.year
    df["start_of_year"] = df.groupby("year")["spx_close"].transform("first")
    return (df["spx_close"] / df["start_of_year"] - 1) * 100


def compute_dd_from_52w_high(spx: pd.Series) -> pd.Series:
    """Drawdown % from 52-week (252 trading day) rolling high."""
    rolling_high = spx.rolling(252, min_periods=20).max()
    return (spx / rolling_high - 1) * 100


def flag_macro_shock(dates: pd.DatetimeIndex) -> pd.Series:
    """Binary flag — 1 on a known shock date or within 7 calendar days after."""
    flag = pd.Series(0, index=dates, dtype=int)
    for d in MACRO_SHOCK_DATES:
        d = pd.Timestamp(d)
        # 7 calendar days ≈ 5 trading days; covers the immediate aftermath.
        mask = (flag.index >= d) & (flag.index <= d + pd.Timedelta(days=7))
        flag.loc[mask] = 1
    return flag


# ---------------------------------------------------------------------------
# Forward outcomes
# ---------------------------------------------------------------------------

def _fwd_max_dd(spx: pd.Series, window: int) -> pd.Series:
    """Max drawdown from today's price over the next `window` trading days."""
    out = pd.Series(index=spx.index, dtype=float)
    prices = spx.values
    n = len(prices)
    for i in range(n - window):
        fwd = prices[i : i + window + 1]
        peak = fwd[0]
        trough = fwd.min()
        out.iloc[i] = (trough / peak - 1) * 100
    return out


def compute_forward_outcomes(spx: pd.Series) -> pd.DataFrame:
    """For each date, realised forward returns + drawdowns + realised vol."""
    df = pd.DataFrame(index=spx.index)

    df["fwd_ret_30d"] = (spx.shift(-21) / spx - 1) * 100
    df["fwd_ret_90d"] = (spx.shift(-63) / spx - 1) * 100
    df["fwd_ret_12m"] = (spx.shift(-252) / spx - 1) * 100

    df["max_dd_90d"] = _fwd_max_dd(spx, 63)
    df["max_dd_180d"] = _fwd_max_dd(spx, 126)

    daily_returns = spx.pct_change()
    df["vol_realized_30d"] = (
        daily_returns.shift(-21).rolling(21).std() * np.sqrt(252) * 100
    )

    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_corpus() -> pd.DataFrame:
    """Assemble historical_corpus.parquet."""
    print("Building historical corpus...\n")

    df = fetch_market_data()

    print("  Computing derived features...")
    df["spx_ytd_pct"] = compute_spx_ytd(df["spx_close"])
    df["spx_dd_from_52w"] = compute_dd_from_52w_high(df["spx_close"])
    df["macro_shock"] = flag_macro_shock(df.index)

    print("  Computing forward outcomes...")
    outcomes = compute_forward_outcomes(df["spx_close"])
    df = df.join(outcomes)

    # Drop rows missing the encoder features (first ~19 days lack a 52w high).
    feature_cols = ["vix", "spx_ytd_pct", "spx_dd_from_52w", "macro_shock"]
    before = len(df)
    df = df.dropna(subset=feature_cols)
    if before - len(df):
        print(f"  Dropped {before - len(df)} rows missing encoder features")

    # KEEP rows missing forward outcomes (last ~3 months). They can't serve
    # as historical analogs — we don't know their future yet — but they're
    # valid QUERY rows: today's market state is here even though today's
    # 90-day forward return obviously hasn't happened. retrieve.py masks
    # these out of the candidate pool; publish.py picks one as the query.

    df.to_parquet(CORPUS_PATH)
    print(f"\nWrote {len(df):,} rows -> {CORPUS_PATH}")
    print(f"Date range: {df.index.min().date()} -> {df.index.max().date()}")
    print(f"Columns: {list(df.columns)}")
    return df


if __name__ == "__main__":
    build_corpus()
