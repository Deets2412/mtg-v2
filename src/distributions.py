"""
Sprint 2: forward outcome distributions from top-K analogs.

Every number this module produces is a literal empirical frequency across
the top-K retrieved analogs. Nothing here is a forecast.

Per CLAUDE.md compliance fence: every visible number must answer
"from what historical sample did this come?" — the answer is always
"the top-K nearest analogs."
"""

from __future__ import annotations

import pandas as pd

# Thresholds at which we report a probability. Chosen to be the kind of
# round numbers a human would actually quote.
LOSS_THRESHOLDS_RET = [0.0, -5.0, -10.0]      # P(return < threshold)
DD_THRESHOLDS = [-10.0, -20.0, -30.0]          # P(max drawdown < threshold)

OUTCOME_COLS_RETURNS = ["fwd_ret_30d", "fwd_ret_90d", "fwd_ret_12m"]
OUTCOME_COLS_DRAWDOWNS = ["max_dd_90d", "max_dd_180d"]
OUTCOME_COL_VOL = "vol_realized_30d"

DISCLAIMER = "Historical base rates from {n} nearest analogs. Not forecasts."


def _stats(series: pd.Series) -> dict:
    vals = series.dropna()
    if vals.empty:
        return {"n": 0}
    return {
        "n": int(len(vals)),
        "median": float(vals.median()),
        "p10": float(vals.quantile(0.10)),
        "p25": float(vals.quantile(0.25)),
        "p75": float(vals.quantile(0.75)),
        "p90": float(vals.quantile(0.90)),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
    }


def _prob_below(series: pd.Series, threshold: float) -> dict:
    vals = series.dropna()
    if vals.empty:
        return {"n": 0, "p": None, "k": 0}
    k = int((vals < threshold).sum())
    return {"n": int(len(vals)), "k": k, "p": float(k / len(vals))}


def base_rates_from_analogs(analogs: pd.DataFrame) -> dict:
    """
    Compute the empirical base-rate summary for a top-K analog set.

    Returned dict is the compliance-safe payload — every number is a literal
    historical frequency across the analog set with no smoothing or modelling.
    """
    out = {
        "n_analogs": int(len(analogs)),
        "disclaimer": DISCLAIMER.format(n=len(analogs)),
        "returns": {},
        "drawdowns": {},
        "vol": {},
    }

    for col in OUTCOME_COLS_RETURNS:
        s = analogs[col]
        out["returns"][col] = {
            "stats": _stats(s),
            "p_below": {
                f"{th:+.0f}%": _prob_below(s, th) for th in LOSS_THRESHOLDS_RET
            },
        }

    for col in OUTCOME_COLS_DRAWDOWNS:
        s = analogs[col]
        out["drawdowns"][col] = {
            "stats": _stats(s),
            "p_below": {
                f"{th:+.0f}%": _prob_below(s, th) for th in DD_THRESHOLDS
            },
        }

    if OUTCOME_COL_VOL in analogs.columns:
        out["vol"][OUTCOME_COL_VOL] = {"stats": _stats(analogs[OUTCOME_COL_VOL])}

    return out
