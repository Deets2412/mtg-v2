"""
Sprint 1: Hand-set encoder.

Z-score each feature against the corpus distribution, then scale by hand-set
distance weights. The result is a 4-vector ready for cosine retrieval.

Scaler params (per-feature mean and std) are persisted to JSON so the same
parameters get applied to query and corpus alike, and so a re-run is
reproducible without re-fitting against a moving target.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SCALER_PATH = DATA_DIR / "encoder_params.json"

# Order matters — encode() returns columns in this order.
FEATURE_COLS = ["vix", "spx_ytd_pct", "spx_dd_from_52w", "macro_shock"]

# Hand-set distance weights. Starting values — tune by inspection.
# Reasoning:
#   - vix: primary regime signal, weight up
#   - spx_dd_from_52w: tracks regime closely, complementary to VIX
#   - macro_shock: binary but informative when set
#   - spx_ytd_pct: calendar-contaminated (-5% YTD in March != -5% YTD in Dec),
#                  weight down
DISTANCE_WEIGHTS = {
    "vix": 2.0,
    "spx_dd_from_52w": 1.5,
    "macro_shock": 1.5,
    "spx_ytd_pct": 0.5,
}


def fit_scaler(corpus: pd.DataFrame) -> dict:
    """Fit z-score params per feature on the full corpus. Persist to JSON."""
    params = {
        col: {
            "mean": float(corpus[col].mean()),
            "std": float(corpus[col].std()),
        }
        for col in FEATURE_COLS
    }
    SCALER_PATH.parent.mkdir(exist_ok=True)
    SCALER_PATH.write_text(json.dumps(params, indent=2))
    return params


def load_scaler() -> dict:
    """Load persisted z-score params."""
    return json.loads(SCALER_PATH.read_text())


def encode(
    rows: pd.DataFrame | pd.Series,
    scaler: dict,
    weights: dict | None = None,
) -> np.ndarray:
    """
    Apply z-score then weight each feature.

    rows:    DataFrame with FEATURE_COLS as columns, or a single-row Series.
    scaler:  dict from fit_scaler / load_scaler.
    weights: override DISTANCE_WEIGHTS (e.g. for sensitivity probes).
    returns: ndarray of shape (n_rows, len(FEATURE_COLS)).
    """
    if isinstance(rows, pd.Series):
        rows = rows.to_frame().T

    w_map = weights if weights is not None else DISTANCE_WEIGHTS

    out = np.empty((len(rows), len(FEATURE_COLS)), dtype=float)
    for i, col in enumerate(FEATURE_COLS):
        m = scaler[col]["mean"]
        s = scaler[col]["std"]
        w = w_map[col]
        out[:, i] = ((rows[col].to_numpy() - m) / s) * w
    return out
