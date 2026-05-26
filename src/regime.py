"""
Sprint 2: k-means regime labelling, k=6.

Fit once on the full corpus in encoded (weighted z-scored) space.
Sort clusters along a stress axis and assign the human-readable labels
from CLAUDE.md: Panic / Fear / Anxiety / Neutral / Greed / Euphoria.

Centroids + label mapping are persisted as JSON. Loading == labelling.

Usage:
    fit_regime(corpus)              # one-off, writes data/regime_model.json
    label_one(features_dict)         # returns ("Neutral", cluster_id, dist)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .encoder import (
    DISTANCE_WEIGHTS,
    FEATURE_COLS,
    encode,
    fit_scaler,
    load_scaler,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REGIME_PATH = DATA_DIR / "regime_model.json"

REGIME_LABELS = ["Panic", "Fear", "Anxiety", "Neutral", "Greed", "Euphoria"]
K = len(REGIME_LABELS)


def _stress_score(centroid: np.ndarray) -> float:
    """
    Per-centroid stress score. Higher = more stressed.

    In encoded space, each component is already weighted z-score, so:
      + high VIX -> stress
      + low (negative) DD-from-52w -> stress  (subtract because DD is negative under stress)
      + high macro_shock -> stress
      + low SPX YTD -> weak stress signal (subtract)
    """
    i_vix = FEATURE_COLS.index("vix")
    i_dd = FEATURE_COLS.index("spx_dd_from_52w")
    i_shock = FEATURE_COLS.index("macro_shock")
    i_ytd = FEATURE_COLS.index("spx_ytd_pct")
    return centroid[i_vix] - centroid[i_dd] + centroid[i_shock] - 0.25 * centroid[i_ytd]


def fit_regime(corpus: pd.DataFrame, random_state: int = 42) -> dict:
    """
    Fit k-means on encoded corpus, label clusters by stress rank, persist.
    Returns the model dict that was written.
    """
    # Make sure scaler exists and load it (encoder.encode needs it).
    scaler = fit_scaler(corpus)
    X = encode(corpus[FEATURE_COLS], scaler)

    km = KMeans(n_clusters=K, n_init=20, random_state=random_state)
    cluster_ids = km.fit_predict(X)
    centroids = km.cluster_centers_  # shape (K, n_features)

    # Rank centroids by stress score (descending: most stressed first)
    scores = np.array([_stress_score(c) for c in centroids])
    rank = np.argsort(-scores)  # indices sorted by descending score

    # Build mapping: raw cluster id -> human label
    id_to_label = {int(raw_id): REGIME_LABELS[i] for i, raw_id in enumerate(rank)}

    # Per-cluster summary stats (useful for sanity checking the labels)
    summaries = []
    for raw_id in range(K):
        mask = cluster_ids == raw_id
        members = corpus[mask]
        summaries.append({
            "cluster_id": int(raw_id),
            "label": id_to_label[int(raw_id)],
            "stress_score": float(scores[raw_id]),
            "n_days": int(mask.sum()),
            "pct_of_corpus": float(mask.mean() * 100),
            "vix_median": float(members["vix"].median()),
            "spx_ytd_pct_median": float(members["spx_ytd_pct"].median()),
            "spx_dd_from_52w_median": float(members["spx_dd_from_52w"].median()),
            "macro_shock_rate": float(members["macro_shock"].mean()),
        })

    model = {
        "k": K,
        "labels_in_stress_order": REGIME_LABELS,
        "feature_cols": FEATURE_COLS,
        "distance_weights": DISTANCE_WEIGHTS,
        "centroids": centroids.tolist(),
        "id_to_label": id_to_label,
        "cluster_summary": sorted(summaries, key=lambda r: -r["stress_score"]),
        "random_state": random_state,
        "fitted_on_n_days": int(len(corpus)),
        "fitted_on_date_range": [
            str(corpus.index.min().date()),
            str(corpus.index.max().date()),
        ],
    }
    REGIME_PATH.write_text(json.dumps(model, indent=2))
    return model


def load_regime() -> dict:
    return json.loads(REGIME_PATH.read_text())


def label_one(features: dict | pd.Series) -> dict:
    """
    Label a single row (today's features). Returns:
        {"label": str, "cluster_id": int, "distance_to_centroid": float}
    """
    model = load_regime()
    scaler = load_scaler()

    if isinstance(features, dict):
        features = pd.Series(features)
    enc = encode(features[FEATURE_COLS], scaler)[0]  # (n_features,)

    centroids = np.array(model["centroids"])
    dists = np.linalg.norm(centroids - enc, axis=1)
    raw_id = int(np.argmin(dists))
    return {
        "label": model["id_to_label"][str(raw_id)],
        "cluster_id": raw_id,
        "distance_to_centroid": float(dists[raw_id]),
    }


def label_corpus(corpus: pd.DataFrame) -> pd.Series:
    """Label every row in the corpus. Useful for backtesting / Sprint 3."""
    model = load_regime()
    scaler = load_scaler()
    X = encode(corpus[FEATURE_COLS], scaler)
    centroids = np.array(model["centroids"])
    # (n_days, K) distance matrix
    dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    raw_ids = dists.argmin(axis=1)
    labels = [model["id_to_label"][str(int(i))] for i in raw_ids]
    return pd.Series(labels, index=corpus.index, name="regime")
