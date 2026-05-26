# v3 probe — results

**Date:** 2026-05-26
**Branch:** `v3-probe` (never merged to main — see verdict)
**Time spent:** ~3 hours (compressed from the weekend-shaped scope)

## Hypothesis being tested

> Adding rates (2y10y slope), credit (Moody's BAA-AAA spread + 90d change),
> and FX (DXY) features to the encoder produces a meaningful improvement in
> out-of-sample Brier skill on the horizons v1 actually displays.

## Pre-committed decision rule

- **Success:** P(loss 90d) skill crosses 0 AND P(>20% DD 90d) lifts ≥0.03 AND
  no horizon degrades by >0.02.
- **Kill:** P(loss 90d) skill doesn't cross 0 OR any horizon degrades by >0.03.

## Result

**KILL.** Every horizon got worse. Three of four breached the >0.03 degradation threshold.

| Prediction | v2 baseline (skill) | v3 probe (skill) | Delta | v2 ECE | v3 ECE |
|---|---|---|---|---|---|
| P(loss 30d) | -0.020 | -0.068 | **-0.048** | 0.025 | 0.075 |
| P(loss 90d) | -0.059 | -0.078 | -0.019 | 0.110 | 0.117 |
| P(loss 12m) | -0.180 | -0.236 | **-0.056** | 0.170 | 0.222 |
| P(>20% DD 90d) | -0.014 | -0.041 | **-0.027** | 0.020 | 0.024 |

Skill values are out-of-sample Brier skill scores (5-fold chronological CV).
Negative = worse than climatology. ECE is calibrated, not raw.

## What we tested

- **Corpus:** `data/historical_corpus_v3probe.parquet`, 9,017 rows,
  1990-05-03 → 2026-02-23.
- **Features added (4):** `slope_2y10y` (DGS10-DGS2), `credit_spread`
  (BAA-AAA, monthly fwd-filled), `credit_spread_chg_90d` (63-day diff
  of credit_spread), `dxy` (ICE Dollar Index, daily).
- **Total encoder dimension:** 4 → 8.
- **Weights:** equal z-score (no learned weights — that was a separate probe).
- **Retrieval:** same as v2 (cosine, top-20, episode-deduped by ≥30 days).
- **CV:** same as `src.calibration` — 5 chronological folds, 90-day embargo.

## Smell tests (pre-calibration sanity check)

| Target | Top-3 analogs (deduped, 30-day gap) | Expected (per CLAUDE.md) | Verdict |
|---|---|---|---|
| 2020-03-16 (COVID) | 1998-08-31, 2001-09-17, 2008-09-29 | 2008-10, 2011-09, 2018-12, 2018-02 | PASS |
| 2008-10-13 (Lehman) | 2001-09-17, 1990-08-06, 1998-09-04 | 2020-03, 1998-08, 1998-09 | PASS |
| 2018-01-26 (melt-up) | 1996-06-28, 1996-08-22, 1996-04-25 | 1999, 2007 | FAIL |
| 2017-07-03 (calm) | 2017-06-01, 2017-05-02, 1993-11-01 | 2005, 1995 | FAIL |

The crisis smell tests pass strongly — the new features pull crisis
periods together regardless of era. The calm smell tests fail because
the richer encoder pulls feature-state matches (1996 low-vol grind ↔
2018 low-vol grind) rather than the narrative-era matches CLAUDE.md
expected (1999 late dot-com, 2007 pre-GFC). This is arguably correct
behaviour — Jan 2018 wasn't yet a euphoria peak, it was a calm bull-
market grind that would become one by Feb 2018 — but it does show the
encoder is shifting how analogs are chosen, which is a separate concern
from the calibration result.

## Honest interpretation

Three things this result rules out:

1. **You cannot get v3's predictive edge for free.** Adding well-chosen
   features with equal weights is the simplest possible test of "do
   these features help?" — and the answer is no, they actively dilute
   the signal that the 4-feature encoder had.

2. **The richer feature space increases ECE on every horizon.** This is
   the more telling number than skill. The displayed probabilities
   become less reliable, not more. That breaks the compliance fence
   that v2 was built around.

3. **The v2 architecture is at a tighter local optimum than a casual
   inspection suggests.** That's actually meaningful — it means the
   hand-set weights and feature choices in v2 are doing real work, not
   just being defaulted.

What this result doesn't rule out:

- **Learned weights might still help.** Equal-weight z-score puts the
  same emphasis on credit_spread that it puts on VIX, even though VIX
  is doing 10x more work in the v2 baseline. A weight-learning probe
  could re-discover something close to v2 weights for the original four
  features and small-but-nonzero weights for the new four. That's a
  separate experiment (~1 weekend).
- **Daily credit might help.** Monthly BAA-AAA can't react to fast
  events like Lehman week. The post-1996 corpus with daily HY OAS
  would test this — at the cost of cutting the corpus in half.
- **Different features entirely.** Term premium, real yields, breadth
  indicators (% above 200dma), put/call ratio — none of these were in
  the probe.

## Decision

**Do not merge `v3-probe` to main.** v2 is the answer for now.

**Next-step options, ordered by my best read of payoff-per-weekend:**

1. **Stop. Accept v2.** The probe answered the question. v2 is calibrated,
   compliant, and stable. Continued v3 work risks turning a working
   maintenance-only tool into a moving target. Spend weekend cycles on
   other projects.

2. **Probe #2 — learned weights on the 8-feature set.** ~1 weekend. Black-box
   optimisation (random search or Bayesian opt) over feature weights
   minimising OOS Brier from the same CV. If learned weights find non-
   trivial weights for the new features, v3 is real. If learned weights
   collapse back toward "all weight on the original 4," v2 is provably
   optimal in this feature space.

3. **Probe #3 — different features.** Term premium, % above 200dma, put/call.
   ~1 weekend per feature group. Higher exploration cost, less clear
   prior.

4. **Probe #4 — Fear & Greed reconstruction.** Restores the 5th input v2
   deliberately deferred. ~1-2 weekends. Independent of #2 and #3, can
   be combined.

My recommendation: **#1 (stop) or #2 (learned weights).** Both respect
the finding. #2 has the cleanest information value — if it doesn't work
either, v3 is genuinely dead and you've spent two weekends to be sure
rather than two months.

## Artifacts

- `src/ingest_v3.py` — v3 corpus builder
- `src/probe_v3.py` — calibration CV runner
- `scripts/v3_smell_test.py` — encoder sanity check
- `data/historical_corpus_v3probe.parquet` — the 8-feature corpus (gitignored)
- `data/v3_probe_cv.json` — full CV report from this run
