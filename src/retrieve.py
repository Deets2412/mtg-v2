"""
Sprint 1: cosine retrieval over the encoded corpus.

Load corpus parquet, encode every row once, then provide top-K nearest
analogs for any query date.

Usage:
    from src.retrieve import retrieve
    analogs, query_date = retrieve("2020-03-16", k=20)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .encoder import (
    DISTANCE_WEIGHTS,
    FEATURE_COLS,
    SCALER_PATH,
    encode,
    fit_scaler,
    load_scaler,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPUS_PATH = DATA_DIR / "historical_corpus.parquet"

OUTCOME_COLS = [
    "fwd_ret_30d",
    "fwd_ret_90d",
    "fwd_ret_12m",
    "max_dd_90d",
    "max_dd_180d",
    "vol_realized_30d",
]


def cosine_distances(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    Cosine distance = 1 - cosine similarity.

    query:      (d,)
    candidates: (n, d)
    returns:    (n,) distances in [0, 2]
    """
    q_norm = query / (np.linalg.norm(query) + 1e-12)
    c_norms = np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-12
    c_unit = candidates / c_norms
    sim = c_unit @ q_norm
    return 1.0 - sim


def topk_episode_dedup(
    distances: np.ndarray,
    dates: np.ndarray,
    k: int,
    dedup_days: int = 30,
) -> np.ndarray:
    """
    Top-K nearest by distance, but skip any candidate within `dedup_days`
    of an already-accepted analog. This stops 8 consecutive days from the
    same crisis from counting as 8 independent observations.

    distances: (n,) distance for each candidate.
    dates:     (n,) datetime64 array aligned to distances.
    k:         number of distinct-episode analogs to return.
    dedup_days: minimum calendar-day gap between accepted analogs.

    returns: (k,) integer indices into the input arrays, ordered by distance.
             May return fewer than k if not enough distinct episodes exist.
    """
    sorted_idx = np.argsort(distances)
    accepted: list[int] = []
    accepted_dates: list[np.datetime64] = []
    gap = np.timedelta64(dedup_days, "D")

    for idx in sorted_idx:
        if not np.isfinite(distances[idx]):
            break
        cand_date = dates[idx]
        too_close = any(
            abs(cand_date - d) < gap for d in accepted_dates
        )
        if too_close:
            continue
        accepted.append(int(idx))
        accepted_dates.append(cand_date)
        if len(accepted) == k:
            break

    return np.array(accepted, dtype=int)


def _load_corpus_and_scaler() -> tuple[pd.DataFrame, dict]:
    corpus = pd.read_parquet(CORPUS_PATH)
    if not SCALER_PATH.exists():
        fit_scaler(corpus)
    return corpus, load_scaler()


def retrieve(
    query_date: str | pd.Timestamp,
    k: int = 20,
    exclude_window_days: int = 30,
    dedup_episode_days: int = 30,
    weights: dict | None = None,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Top-K analogs to query_date by cosine distance over weighted z-scored features,
    with episode deduplication so the K analogs represent K distinct historical
    episodes rather than K consecutive days of the same episode.

    query_date:          ISO date or Timestamp. If not in corpus, uses nearest.
    k:                   number of analogs to return.
    exclude_window_days: drop candidates within +/- this many calendar days of
                         the query date. Stops "yesterday" from being your
                         nearest analog to "today" -- adjacent days are nearly
                         identical and tell you nothing.
    dedup_episode_days:  minimum calendar-day gap between any two accepted
                         analogs. Without this, the 20 nearest analogs to
                         a March 2020 query are 8 consecutive Oct 2008 days
                         plus 7 consecutive late-Mar 2020 days plus a few
                         stragglers -- 2 episodes counted 15 ways. Set to
                         0 to disable. Default 30 (~6 trading weeks).
    weights:             override DISTANCE_WEIGHTS (sensitivity probes).

    Returns:
        (analogs_df, resolved_query_date)
        analogs_df has the query date's row stripped, indexed by analog date,
        with columns: distance, *FEATURE_COLS, *OUTCOME_COLS.
    """
    corpus, scaler = _load_corpus_and_scaler()
    encoded = encode(corpus[FEATURE_COLS], scaler, weights=weights)

    query_ts = pd.Timestamp(query_date)
    if query_ts not in corpus.index:
        idx = corpus.index.get_indexer([query_ts], method="nearest")[0]
        query_ts = corpus.index[idx]

    q_idx = corpus.index.get_loc(query_ts)
    q_vec = encoded[q_idx]

    dists = cosine_distances(q_vec, encoded)

    # Mask out the query date and its immediate neighbourhood.
    window = pd.Timedelta(days=exclude_window_days)
    mask = (corpus.index >= query_ts - window) & (corpus.index <= query_ts + window)
    dists[mask] = np.inf

    if dedup_episode_days > 0:
        top_k_idx = topk_episode_dedup(
            dists, corpus.index.to_numpy(), k, dedup_episode_days
        )
    else:
        top_k_idx = np.argsort(dists)[:k]

    result = corpus.iloc[top_k_idx][FEATURE_COLS + OUTCOME_COLS].copy()
    result.insert(0, "distance", dists[top_k_idx])
    return result, query_ts


def query_vector(query_date: str | pd.Timestamp) -> dict:
    """Return the resolved query row's raw features, for printing."""
    corpus, _ = _load_corpus_and_scaler()
    query_ts = pd.Timestamp(query_date)
    if query_ts not in corpus.index:
        idx = corpus.index.get_indexer([query_ts], method="nearest")[0]
        query_ts = corpus.index[idx]
    row = corpus.loc[query_ts]
    return {
        "date": query_ts.date(),
        **{c: float(row[c]) for c in FEATURE_COLS},
    }
