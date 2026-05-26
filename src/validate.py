"""
Sprint 1 deliverable: show top-20 analogs for "today" + run smell tests.

"Today" = the latest date in the corpus. Smell tests come from CLAUDE.md
and are the bar for accepting the encoder weights as not-obviously-wrong.

Usage:
    python -m src.validate
"""

from __future__ import annotations

import pandas as pd

from .retrieve import OUTCOME_COLS, retrieve, query_vector
from .encoder import FEATURE_COLS, DISTANCE_WEIGHTS, fit_scaler
from .retrieve import CORPUS_PATH


SMELL_TESTS = [
    ("2020-03-16", "COVID circuit breaker -- expect Oct 2008, late-2011, Dec 2018, Feb 2018"),
    ("2018-01-26", "January 2018 melt-up -- expect late-1999, mid-2007, other low-vol highs"),
    ("2017-07-03", "Mid-2017 calm -- expect mid-1995, mid-2005, other low-vol grinds"),
    ("2008-10-10", "Lehman week -- expect March 2020, late-1998 LTCM"),
]


def _fmt(x: float, width: int = 7, prec: int = 2, sign: bool = True) -> str:
    if pd.isna(x):
        return f"{'nan':>{width}}"
    return f"{x:{'+' if sign else ''}{width}.{prec}f}"


def print_query_context(q: dict) -> None:
    print(f"Query date: {q['date']}")
    print(
        f"  VIX={q['vix']:.2f}  "
        f"YTD={q['spx_ytd_pct']:+.2f}%  "
        f"DD52w={q['spx_dd_from_52w']:+.2f}%  "
        f"shock={int(q['macro_shock'])}"
    )


def print_analogs(analogs: pd.DataFrame, k_print: int | None = None) -> None:
    """Pretty-print the analog table. k_print=None prints all."""
    rows = analogs if k_print is None else analogs.head(k_print)
    print(
        f"  {'date':<11} {'dist':>5}  {'VIX':>5}  {'YTD':>7}  {'DD52w':>7}  {'shk':>3}  "
        f"{'fwd30d':>7}  {'fwd90d':>7}  {'fwd12m':>7}  {'maxDD90d':>8}  {'maxDD180d':>9}"
    )
    print("  " + "-" * 96)
    for d, r in rows.iterrows():
        print(
            f"  {str(d.date()):<11} "
            f"{_fmt(r.distance, 5, 3, sign=False)}  "
            f"{_fmt(r.vix, 5, 2, sign=False)}  "
            f"{_fmt(r.spx_ytd_pct)}  "
            f"{_fmt(r.spx_dd_from_52w)}  "
            f"{int(r.macro_shock):>3}  "
            f"{_fmt(r.fwd_ret_30d)}  "
            f"{_fmt(r.fwd_ret_90d)}  "
            f"{_fmt(r.fwd_ret_12m)}  "
            f"{_fmt(r.max_dd_90d, 8)}  "
            f"{_fmt(r.max_dd_180d, 9)}"
        )


def print_outcome_summary(analogs: pd.DataFrame) -> None:
    """Empirical base rates across the top-K — Sprint 2 will format these for UI."""
    print("\n  Top-K outcome distribution (the compliance-fence numbers):")
    for col in OUTCOME_COLS:
        vals = analogs[col].dropna()
        if vals.empty:
            print(f"    {col:<20} n=0")
            continue
        # P(loss) for return cols, P(>20% dd) for dd cols, just stats for vol
        extra = ""
        if col.startswith("fwd_ret_"):
            p_loss = (vals < 0).mean() * 100
            extra = f"  P(loss)={p_loss:5.1f}%"
        elif col.startswith("max_dd_"):
            p_severe = (vals < -20).mean() * 100
            extra = f"  P(<-20%)={p_severe:5.1f}%"
        print(
            f"    {col:<20} n={len(vals):2d}  "
            f"median={vals.median():+7.2f}  "
            f"p25={vals.quantile(0.25):+7.2f}  "
            f"p75={vals.quantile(0.75):+7.2f}  "
            f"min={vals.min():+7.2f}  "
            f"max={vals.max():+7.2f}{extra}"
        )


def run_one(date: str, label: str, k: int = 20, k_print: int = 10) -> None:
    print("\n" + "=" * 100)
    print(label)
    print("=" * 100)
    q = query_vector(date)
    print_query_context(q)
    analogs, _ = retrieve(date, k=k)
    print(f"\nTop {k_print} of {k} analogs:")
    print_analogs(analogs, k_print=k_print)
    print_outcome_summary(analogs)


def main() -> None:
    # Make sure scaler is fit before we start (so the very first retrieve doesn't
    # silently fit a new one).
    corpus = pd.read_parquet(CORPUS_PATH)
    fit_scaler(corpus)

    print(f"Corpus: {len(corpus):,} rows, "
          f"{corpus.index.min().date()} -> {corpus.index.max().date()}")
    print(f"Features: {FEATURE_COLS}")
    print(f"Weights:  {DISTANCE_WEIGHTS}")

    # Today = latest date with the features we need (forward outcomes optional
    # for the query itself).
    today = corpus.index.max()
    run_one(str(today.date()), f'"Today" -- latest corpus date ({today.date()})',
            k=20, k_print=20)

    # Smell tests from CLAUDE.md
    print("\n\n" + "#" * 100)
    print("# SMELL TESTS")
    print("#" * 100)
    for date, label in SMELL_TESTS:
        run_one(date, label, k=20, k_print=10)


if __name__ == "__main__":
    main()
