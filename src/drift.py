"""
Drift detector.

Re-runs the same time-series CV that produced data/calibration.json, on
the *current* corpus. Compares the fresh CV report to the stored one
and flags drift on two axes:

  1. Relative drift — has skill_calibrated degraded materially vs. what
     calibration.json promised? (E.g. when calibration was fit we
     reported a Brier skill of -0.02 for fwd_ret_30d; today's CV on the
     extended corpus shows -0.10. The displayed number on the snapshot
     is now lying about its own performance.)

  2. Absolute floor — is skill_calibrated below a hard cliff or
     ece_calibrated above one, regardless of where it started? A
     prediction that's currently uncalibrated is bad even if it
     started that way.

Exits 0 with status="ok" if no drift detected.
Exits 1 with status="drift" and a structured report otherwise.

Output is JSON on stdout in both cases — the GHA workflow parses it
and opens/updates a GitHub Issue when drift is found.

Usage:
    python -m src.drift                # default thresholds, prints report
    python -m src.drift --quiet        # just exit code, no report
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

from .calibration import CALIBRATION_PATH, collect_pairs, time_series_cv


@contextlib.contextmanager
def _stdout_to_stderr():
    """
    Temporarily redirect stdout -> stderr. The collect_pairs and
    time_series_cv functions in src.calibration print progress to
    stdout; we want them on stderr so stdout stays clean JSON for the
    workflow to parse.
    """
    saved = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = saved

# Thresholds. Both must be exceeded for relative-drift to fire — a
# prediction can wobble within a small band without it being meaningful.
SKILL_DEGRADATION_THRESHOLD = 0.05  # skill_calibrated dropped by >= this much
ECE_DEGRADATION_THRESHOLD = 0.05    # ece_calibrated rose by >= this much

# Absolute-floor thresholds. If breached, fire regardless of baseline.
ABSOLUTE_SKILL_FLOOR = -0.25  # skill_calibrated below this = clearly worse than climatology
ABSOLUTE_ECE_CEILING = 0.20   # ece_calibrated above this = poorly calibrated


def _load_baseline() -> dict | None:
    """Stored CV report from data/calibration.json."""
    if not CALIBRATION_PATH.exists():
        return None
    try:
        return json.loads(CALIBRATION_PATH.read_text())
    except Exception:
        return None


def _assess(current: dict, baseline: dict) -> list[dict]:
    """
    For each prediction, compare current CV result to baseline CV result.
    Return a list of issue records (empty list = no drift).
    """
    issues: list[dict] = []
    baseline_cv = (baseline or {}).get("cv_evaluation", {}) or {}

    for col, cur in current.items():
        if cur.get("n_oos_pairs", 0) == 0:
            continue

        base = baseline_cv.get(col, {})
        cur_skill = cur.get("skill_calibrated")
        cur_ece = cur.get("ece_calibrated")
        base_skill = base.get("skill_calibrated")
        base_ece = base.get("ece_calibrated")

        flags: list[str] = []

        # Relative drift — only if we have a baseline to compare to.
        if base_skill is not None and cur_skill is not None:
            skill_drop = base_skill - cur_skill
            if skill_drop >= SKILL_DEGRADATION_THRESHOLD:
                flags.append(
                    f"skill dropped by {skill_drop:+.3f} "
                    f"(baseline {base_skill:+.3f} → current {cur_skill:+.3f})"
                )
        if base_ece is not None and cur_ece is not None:
            ece_rise = cur_ece - base_ece
            if ece_rise >= ECE_DEGRADATION_THRESHOLD:
                flags.append(
                    f"ECE rose by {ece_rise:+.3f} "
                    f"(baseline {base_ece:.3f} → current {cur_ece:.3f})"
                )

        # Absolute floor — independent of baseline.
        if cur_skill is not None and cur_skill < ABSOLUTE_SKILL_FLOOR:
            flags.append(
                f"current skill {cur_skill:+.3f} below floor "
                f"{ABSOLUTE_SKILL_FLOOR:+.3f}"
            )
        if cur_ece is not None and cur_ece > ABSOLUTE_ECE_CEILING:
            flags.append(
                f"current ECE {cur_ece:.3f} above ceiling "
                f"{ABSOLUTE_ECE_CEILING:.3f}"
            )

        if flags:
            issues.append({
                "prediction": col,
                "label": cur.get("label", col),
                "current": {
                    "skill_calibrated": cur_skill,
                    "ece_calibrated": cur_ece,
                    "n_oos_pairs": cur.get("n_oos_pairs"),
                },
                "baseline": {
                    "skill_calibrated": base_skill,
                    "ece_calibrated": base_ece,
                    "n_oos_pairs": base.get("n_oos_pairs"),
                    "fit_at": (baseline or {}).get("fit_at"),
                },
                "flags": flags,
            })

    return issues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress JSON output, just exit code.")
    args = ap.parse_args()

    baseline = _load_baseline()
    if baseline is None:
        report = {
            "status": "no_baseline",
            "message": "data/calibration.json missing or unreadable — "
                       "run python -m src.calibration first.",
        }
        if not args.quiet:
            print(json.dumps(report, indent=2))
        return 2

    # Fresh CV on current corpus, same params as the stored calibration was fit with.
    # Redirect progress prints to stderr so stdout stays clean JSON.
    with _stdout_to_stderr():
        pairs = collect_pairs(sample="weekly")
        current = time_series_cv(pairs, n_folds=5)

    issues = _assess(current, baseline)

    report = {
        "status": "drift" if issues else "ok",
        "baseline_fit_at": baseline.get("fit_at"),
        "current_n_pairs": int(len(pairs)),
        "thresholds": {
            "skill_degradation": SKILL_DEGRADATION_THRESHOLD,
            "ece_degradation": ECE_DEGRADATION_THRESHOLD,
            "skill_floor": ABSOLUTE_SKILL_FLOOR,
            "ece_ceiling": ABSOLUTE_ECE_CEILING,
        },
        "per_prediction_current": {
            col: {
                "skill_calibrated": v.get("skill_calibrated"),
                "ece_calibrated": v.get("ece_calibrated"),
                "n_oos_pairs": v.get("n_oos_pairs"),
            }
            for col, v in current.items()
        },
        "issues": issues,
    }

    if not args.quiet:
        print(json.dumps(report, indent=2, default=str))

    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
