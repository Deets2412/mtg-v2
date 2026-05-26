"""
Sprint 1: weight sensitivity probe.

For each feature, scale its weight up and down vs the baseline. For each
test query, measure how much the top-K set churns. If the rankings are
stable across plausible perturbations, the weights are in a stable basin
and not worth retuning before Sprint 2.

Metrics per query x perturbation:
  - jaccard_20: overlap of top-20 vs baseline (1.0 = identical set)
  - top1_same:  did the #1 analog change
  - top3_overlap: how many of baseline top-3 are still in perturbed top-3

Aggregated over all test queries.

Usage:
    python -m src.sensitivity
"""

from __future__ import annotations

import numpy as np

from .encoder import DISTANCE_WEIGHTS, FEATURE_COLS
from .retrieve import retrieve

TEST_QUERIES = [
    # Extremes
    ("2020-03-16", "March 2020 (full crisis)"),
    ("2008-10-10", "Lehman week (full crisis)"),
    ("2018-01-26", "Jan 2018 melt-up"),
    ("2017-07-03", "Mid-2017 calm"),
    # Intermediate / ambiguous regimes -- the harder test
    ("2011-08-22", "Aug 2011 debt-downgrade aftermath"),
    ("2018-12-24", "Dec 2018 Fed-pivot bottom"),
    ("2022-10-13", "Oct 2022 slow-bleed bear low"),
    ("2026-02-13", "Today (latest corpus date)"),
]

PERTURBATIONS = [
    ("baseline", 1.00),
    ("-50%", 0.50),
    ("-25%", 0.75),
    ("+25%", 1.25),
    ("+50%", 1.50),
]


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def make_weights(feature: str, scale: float) -> dict:
    """Scale one feature's weight, keep the others at baseline."""
    w = dict(DISTANCE_WEIGHTS)
    w[feature] = DISTANCE_WEIGHTS[feature] * scale
    return w


def run_query(query_date: str, weights: dict) -> tuple[set, list]:
    analogs, _ = retrieve(query_date, k=20, weights=weights)
    dates = [d for d in analogs.index]
    return set(dates), dates


def probe() -> None:
    print(f"Baseline weights: {DISTANCE_WEIGHTS}\n")

    # Pre-compute baseline top-20 sets per query
    baselines = {}
    for date, _label in TEST_QUERIES:
        s, lst = run_query(date, DISTANCE_WEIGHTS)
        baselines[date] = (s, lst)

    # Per-feature perturbation table
    print("=" * 110)
    print("Per-feature perturbation -- average Jaccard(top-20) across all queries")
    print("=" * 110)
    print(f"{'feature':<22}", end="")
    for label, _ in PERTURBATIONS:
        print(f"{label:>10}", end="")
    print(f"   {'min':>6}  {'top1_flip_rate':>15}")

    for feature in FEATURE_COLS:
        print(f"{feature:<22}", end="")
        jaccards = []
        top1_flips_per_pert = []
        for label, scale in PERTURBATIONS:
            jacs = []
            top1_flips = 0
            w = make_weights(feature, scale)
            for date, _ in TEST_QUERIES:
                s_base, lst_base = baselines[date]
                s_pert, lst_pert = run_query(date, w)
                jacs.append(jaccard(s_base, s_pert))
                if lst_base[0] != lst_pert[0]:
                    top1_flips += 1
            mean_jac = float(np.mean(jacs))
            print(f"{mean_jac:>10.3f}", end="")
            jaccards.append(mean_jac)
            top1_flips_per_pert.append(top1_flips)
        # min Jaccard across perturbations, total top-1 flips across non-baseline perts
        worst = min(jaccards[1:])  # skip baseline
        flip_rate = sum(top1_flips_per_pert[1:]) / (4 * len(TEST_QUERIES))
        print(f"   {worst:>6.3f}  {flip_rate:>14.0%}")
    print()

    # Per-query detail under WORST-CASE perturbation (all weights jittered ±50%)
    print("=" * 110)
    print("Per-query stability under WORST-CASE adversarial perturbation")
    print("(each weight independently halved or doubled -- try 16 corners, report worst)")
    print("=" * 110)
    print(f"{'query':<42} {'baseline #1':<13} {'worst-corner #1':<18} {'worst jac':>10}  {'#1 stable':>10}")

    # 2^4 = 16 corner perturbations: each weight x 0.5 or x 2.0
    corners = []
    for mask in range(16):
        scales = {f: (2.0 if (mask >> i) & 1 else 0.5)
                  for i, f in enumerate(FEATURE_COLS)}
        w = {f: DISTANCE_WEIGHTS[f] * scales[f] for f in FEATURE_COLS}
        corners.append((scales, w))

    for date, label in TEST_QUERIES:
        s_base, lst_base = baselines[date]
        worst_jac = 1.0
        worst_top1 = lst_base[0]
        top1_stable = True
        for scales, w in corners:
            s_pert, lst_pert = run_query(date, w)
            j = jaccard(s_base, s_pert)
            if j < worst_jac:
                worst_jac = j
                worst_top1 = lst_pert[0]
            if lst_pert[0] != lst_base[0]:
                top1_stable = False
        short = label[:40]
        print(f"{short:<42} {str(lst_base[0].date()):<13} "
              f"{str(worst_top1.date()):<18} {worst_jac:>10.3f}  "
              f"{'yes' if top1_stable else 'NO':>10}")


if __name__ == "__main__":
    probe()
