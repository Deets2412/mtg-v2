"""
Sprint 2: daily batch job that writes today's snapshot for v1's display layer.

Inputs:  data/historical_corpus.parquet, data/encoder_params.json,
         data/regime_model.json
Outputs: data/today.json              -- summary (regime, base rates, closest analog)
         data/today_analogs.parquet   -- 20 analog rows (date, distance, features, outcomes)

The v1 display layer reads these two files. Schema is intentionally stable
so v2 can swap in silently behind the existing UI.

Usage:
    python -m src.publish              # uses latest corpus date
    python -m src.publish 2020-03-16   # backfill / sanity-check any date
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from .distributions import DISCLAIMER, base_rates_from_analogs
from .email_publish import send_snapshot_email
from .encoder import FEATURE_COLS, fit_scaler
from .regime import fit_regime, label_one, REGIME_PATH
from .retrieve import CORPUS_PATH, OUTCOME_COLS, retrieve

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TODAY_JSON = DATA_DIR / "today.json"
TODAY_ANALOGS = DATA_DIR / "today_analogs.parquet"
CALIBRATION_PATH = DATA_DIR / "calibration.json"

# Horizons we trust enough to publish a calibrated probability for.
# Per src/calibration.py CV: 12m has negative skill even after isotonic
# (relationship between state and 12m outcome is too unstable). For 12m we
# strip the probability fields entirely and rely on climatology + the
# distribution stats instead.
TRUSTED_PROB_COLS = {"fwd_ret_30d", "fwd_ret_90d", "max_dd_90d"}
UNTRUSTED_PROB_COLS = {"fwd_ret_12m"}

SCHEMA_VERSION = "v2.1"  # bumped: calibrated probabilities + cv_evaluation block


def _ensure_models(corpus: pd.DataFrame) -> None:
    """Make sure scaler and regime model exist; fit them if not."""
    fit_scaler(corpus)
    if not REGIME_PATH.exists():
        fit_regime(corpus)


def _load_calibration() -> dict | None:
    """Load isotonic calibration models. Returns None if not yet fit."""
    if not CALIBRATION_PATH.exists():
        return None
    try:
        return json.loads(CALIBRATION_PATH.read_text())
    except Exception:
        return None


def _interp_calibrate(raw_p: float | None, model: dict | None) -> float | None:
    """Apply piecewise-linear isotonic mapping. Identity if no model."""
    if raw_p is None:
        return None
    if model is None or not model.get("x"):
        return float(raw_p)
    import numpy as np  # local import to keep top tidy
    return float(np.clip(np.interp(raw_p, model["x"], model["y"]), 0.0, 1.0))


def _apply_calibration_to_base_rates(base_rates: dict, calibration: dict | None) -> dict:
    """
    Walk the base_rates structure and:
      - For trusted prediction columns: add `p_calibrated` alongside raw `p`
        in every p_below entry.
      - For untrusted columns (12m): strip the p_below block entirely. The
        descriptive stats (median/p25/p75/etc.) remain — those are real
        sample statistics, not probability claims.
    """
    models = (calibration or {}).get("models", {})

    for section_name in ("returns", "drawdowns"):
        for col, payload in list(base_rates[section_name].items()):
            if col in UNTRUSTED_PROB_COLS:
                # Drop the probability block; keep stats.
                payload.pop("p_below", None)
                payload["probability_status"] = (
                    "withheld_no_skill: out-of-sample CV showed v2 cannot "
                    "predict this horizon better than climatology"
                )
                continue
            if col not in TRUSTED_PROB_COLS:
                continue
            model = models.get(col)
            for threshold, prob in payload.get("p_below", {}).items():
                if prob.get("p") is not None:
                    prob["p_raw"] = prob["p"]
                    prob["p"] = _interp_calibrate(prob["p"], model)
                    prob["calibrated"] = model is not None
    return base_rates


def publish(as_of: str | pd.Timestamp | None = None, k: int = 20) -> dict:
    corpus = pd.read_parquet(CORPUS_PATH)
    _ensure_models(corpus)

    if as_of is None:
        as_of_ts = corpus.index.max()
    else:
        as_of_ts = pd.Timestamp(as_of)
        if as_of_ts not in corpus.index:
            idx = corpus.index.get_indexer([as_of_ts], method="nearest")[0]
            as_of_ts = corpus.index[idx]

    query_row = corpus.loc[as_of_ts]
    query_features = {c: float(query_row[c]) for c in FEATURE_COLS}

    # Retrieve + label
    analogs, _ = retrieve(as_of_ts, k=k)
    regime = label_one(query_features)
    base_rates = base_rates_from_analogs(analogs)

    # Apply isotonic recalibration (if calibration.json exists). For trusted
    # horizons this overwrites raw `p` with a calibrated value and preserves
    # the raw under `p_raw`. For 12m (untrusted per CV), strips probability
    # fields entirely — the empirical median/percentile stats still appear.
    calibration = _load_calibration()
    base_rates = _apply_calibration_to_base_rates(base_rates, calibration)

    closest = analogs.iloc[0]
    closest_payload = {
        "date": str(closest.name.date()),
        "distance": float(closest["distance"]),
        "features": {c: float(closest[c]) for c in FEATURE_COLS},
        "outcomes": {
            c: (float(closest[c]) if pd.notna(closest[c]) else None)
            for c in OUTCOME_COLS
        },
    }

    # Self-describing credibility statement — surface the out-of-sample
    # CV evaluation so downstream consumers (v1's LLM, email template, any
    # human reader) can see exactly how well-calibrated each probability is.
    calibration_meta = None
    if calibration:
        cv = calibration.get("cv_evaluation", {})
        calibration_meta = {
            "fit_at": calibration.get("fit_at"),
            "fit_params": calibration.get("fit_params"),
            "n_pairs": calibration.get("n_pairs"),
            "method": "isotonic regression, time-series 5-fold CV (chronological, no look-ahead)",
            "per_prediction_out_of_sample": {
                col: {
                    "label": v.get("label"),
                    "brier_skill_score": v.get("skill_calibrated"),
                    "expected_calibration_error": v.get("ece_calibrated"),
                    "n_oos_pairs": v.get("n_oos_pairs"),
                    "climatology": v.get("climatology"),
                }
                for col, v in cv.items()
            },
            "interpretation": (
                "Probabilities for fwd_ret_30d, fwd_ret_90d, and max_dd_90d "
                "are out-of-sample calibrated. Brier skill score near 0 means "
                "tied with climatology (not better, not worse). Low ECE means "
                "the displayed probability matches the empirical hit rate. "
                "fwd_ret_12m probabilities are withheld — even after calibration "
                "they fail to beat climatology out-of-sample."
            ),
        }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "as_of_date": str(as_of_ts.date()),
        "query_features": query_features,
        "regime": regime,
        "closest_analog": closest_payload,
        "base_rates": base_rates,
        "calibration": calibration_meta,
        "k": k,
        "disclaimer": DISCLAIMER.format(n=k) + (
            " Probabilities are isotonic-recalibrated against historical hit "
            "rates with time-series CV (see `calibration` block for per-"
            "prediction out-of-sample skill)." if calibration else ""
        ),
    }

    # Write both artifacts atomically-ish (rename after write)
    TODAY_JSON.write_text(json.dumps(summary, indent=2, default=str))
    analogs.to_parquet(TODAY_ANALOGS)

    return summary


def _format_summary(s: dict) -> str:
    q = s["query_features"]
    r = s["regime"]
    c = s["closest_analog"]
    lines = [
        f"As of:   {s['as_of_date']}",
        f"State:   VIX={q['vix']:.2f}  YTD={q['spx_ytd_pct']:+.2f}%  "
        f"DD52w={q['spx_dd_from_52w']:+.2f}%  shock={int(q['macro_shock'])}",
        f"Regime:  {r['label']}  (cluster {r['cluster_id']}, "
        f"d-to-centroid={r['distance_to_centroid']:.3f})",
        f"Closest: {c['date']}  (distance {c['distance']:.4f})",
        f"         outcomes -> fwd30d={c['outcomes']['fwd_ret_30d']:+.2f}%  "
        f"fwd90d={c['outcomes']['fwd_ret_90d']:+.2f}%  "
        + (f"fwd12m={c['outcomes']['fwd_ret_12m']:+.2f}%"
           if c['outcomes']['fwd_ret_12m'] is not None else "fwd12m=nan"),
        "",
        f"Base rates across {s['k']} analogs (the compliance-safe numbers):",
    ]
    br = s["base_rates"]
    for col, payload in br["returns"].items():
        stats = payload["stats"]
        if "p_below" in payload:
            p_loss = payload["p_below"]["+0%"]
            tail = (
                f"P(loss)={p_loss['p']:.0%}  (k={p_loss['k']}/{p_loss['n']})"
                + ("  [calibrated]" if p_loss.get("calibrated") else "")
            )
        else:
            tail = f"P(loss)=withheld ({payload.get('probability_status', 'n/a')[:30]}...)"
        lines.append(
            f"  {col:<14} median={stats['median']:+6.2f}  "
            f"p25={stats['p25']:+6.2f}  p75={stats['p75']:+6.2f}  {tail}"
        )
    for col, payload in br["drawdowns"].items():
        stats = payload["stats"]
        if "p_below" in payload:
            p_20 = payload["p_below"]["-20%"]
            tail = (
                f"P(<-20%)={p_20['p']:.0%}  (k={p_20['k']}/{p_20['n']})"
                + ("  [calibrated]" if p_20.get("calibrated") else "")
            )
        else:
            tail = "P(<-20%)=withheld"
        lines.append(
            f"  {col:<14} median={stats['median']:+6.2f}  "
            f"p25={stats['p25']:+6.2f}  {tail}"
        )
    lines.append("")
    lines.append(f"  {s['disclaimer']}")
    return "\n".join(lines)


def main() -> None:
    # Positional arg = as_of date (override "today"). --no-email skips the
    # Resend send (useful for backfills / ad-hoc inspection without spamming
    # the inbox). Default behaviour: send.
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    as_of = args[0] if args else None
    skip_email = "--no-email" in flags

    s = publish(as_of)
    print(_format_summary(s))
    print(f"\nWrote {TODAY_JSON}")
    print(f"Wrote {TODAY_ANALOGS}")

    if skip_email:
        print("\n[email] skipped (--no-email)")
        return

    print("\nSending snapshot email via Resend...")
    result = send_snapshot_email(s)
    if result.get("sent"):
        print(f"[email] sent id={result.get('id')} to={result.get('to')}")
    else:
        print(f"[email] not sent: {result.get('reason')}")


if __name__ == "__main__":
    main()
