# MTG v2 — Retrieval Engine

## What this is

A retrieval-based rewrite of the Market Temp Gauge backend.

- **v1** is a thresholded composite (`if VIX >= 30: score += 35` ...). Hand-engineered weights, manually-chosen analogs, fixed base-rate citations.
- **v2** replaces the composite with a two-tower retrieval engine over the historical corpus. Display layer (regime gauge, closest analog, base rates) stays exactly as designed for v1 — only the engine swaps.

Architecture borrowed from xAI's open-sourced X-algorithm patterns (Phoenix two-tower retrieval, multi-action prediction with negative weights, candidate isolation). The architecture is the gift — we are not porting Rust.

## Why this is better

The composite collapses VIX-30-in-March-2020, Feb 2018, and October 2008 into the same number. They have completely different forward path distributions. Retrieval over a historical corpus distinguishes them because the embedding sees the full conditioning context, not just one feature crossing a threshold.

The compliance fence also holds better: every number on screen becomes a literal empirical base rate from retrieved analogs, not a composite score that reads as judgment.

## Locked decisions (do not re-litigate)

| Decision | Choice | Why |
|---|---|---|
| Inputs | 4: VIX, SPX YTD %, SPX drawdown from 52w high, macro shock binary | Originally 5 (incl. Fear & Greed); F&G dropped — CNN endpoint only serves ~1y of history, not since 2011. Reconstruction → v3. Anything new must displace one. |
| Encoder | Z-scored 4-vector with hand-set distance weights | No learned encoder. v3 problem. |
| Distance | Cosine | Standard for embedding retrieval. |
| K | 20 | Tune by inspection in Sprint 2. |
| Storage | Parquet + numpy | No FAISS, no vector DB. Corpus is ~9,000 rows (1990 onwards). |
| Regime labels | k-means, k=6 (Panic / Fear / Anxiety / Neutral / Greed / Euphoria) | Fit once on full history. |
| Corpus start | 1990-01-01 | VIX inception. Picks up dot-com, 9/11, GFC, flash crash — Oct 2008 is now reachable as a March 2020 analog. Originally 2011 (F&G limit); revised when F&G was dropped. |

## Compliance fence (non-negotiable)

- Every output column must be an empirical observation across retrieved analogs.
- No probability that isn't a literal historical frequency.
- **Internal review test:** every visible number must answer "from what historical sample did this come?" If it can't, it doesn't ship.
- Copy template: *"Across [N] similar historical episodes, [X]% experienced..."*
- Standing masthead disclaimer: *"Historical base rates from N nearest analogs. Not forecasts."*

## Sprint structure

### Sprint 1 (Weekend 1): Data + retrieval — DONE

- Ingest VIX and SPX daily since 1990 (Fear & Greed deferred — see locked decisions)
- Compute derived features: SPX YTD %, drawdown from 52w high, macro shock flag
- Pre-compute forward outcomes for each date (30d / 90d / 12m returns, max drawdown over 90d/180d, realised vol)
- Persist as `data/historical_corpus.parquet`
- Build `encode()` and `retrieve(k=20)` in `src/`
- Validation script (`src/validate.py`) and sensitivity probe (`src/sensitivity.py`) — smell tests pass, weights in a stable basin

### Sprint 2 (Weekend 2): Distributions + regime + wire-in — DONE

- ✅ From top-K, compute forward return distribution + downside probabilities (`src/distributions.py`)
- ✅ K-means regime labelling, k=6, fit once on full history (`src/regime.py`)
- ✅ Daily batch publisher (`src/publish.py`) writes `data/today.json` + `data/today_analogs.parquet`
- ✅ Wire-in on v1 side complete — see `Market Temp Gauge/src/sources/mtg-v2-analog.ts` and `CLAUDE.md` over there. Calibration A/B is tagged via `scans.evidence_version`.

### Sprint 2.5 (added late): defensibility — DONE

The calibration backtest (originally v3) ran and falsified the assumption
that raw retrieval probabilities were predictive. Headline result:

- Raw v2 P(loss) was WORSE than climatology on 3 of 4 predictions (Brier
  skill scores -0.17 to -0.20). Heavily over-confident at both tails:
  predicted 95% → actual 49%; predicted 2% → actual 22%.

Two-part fix, both shipped:

1. **Episode dedup in retrieval (`src/retrieve.py:topk_episode_dedup`)** —
   K analogs now represent K distinct historical episodes (≥30 calendar
   days apart), not K consecutive days of one episode. Halved the
   calibration gap on its own.
2. **Isotonic recalibration with time-series CV (`src/calibration.py`)** —
   per-prediction monotone mapping from raw P to calibrated P. Fit and
   evaluated with chronological 5-fold CV — every reported skill score
   is OUT-OF-SAMPLE. Final production models persisted to
   `data/calibration.json`; loaded by `publish.py` and applied at display
   time via `numpy.interp` so no sklearn dependency leaks into publish.

Final out-of-sample numbers (defensible):

| Prediction | Calibrated Brier skill | Calibrated ECE | Status |
|---|---|---|---|
| P(loss 30d) | -0.020 | 0.025 | Well-calibrated, ties climatology |
| P(loss 90d) | -0.059 | 0.111 | Marginal — kept |
| P(loss 12m) | -0.181 | 0.171 | **Withheld** — no skill at this horizon |
| P(>20% DD 90d) | -0.014 | 0.020 | Well-calibrated, ties climatology |

What this means for the compliance fence: every displayed probability is
either calibrated against historical hit rates (with the OOS skill + ECE
visible in the artifact's `calibration` block) or explicitly withheld with
reason. The displayed numbers are now defensible as descriptive empirical
frequencies, not implicit forecasts.

Recalibration cadence: re-run `python -m src.calibration` quarterly or after
any structural change to encoder/retrieval. Static fit is fine in between.

### Out of scope for v2 (deliberate)

- Learned encoder weights → v3 (now with a concrete loss function: minimise OOS Brier from `src/calibration.py` CV)
- Sector / region breakdowns → v3
- ~~Calibration backtest~~ → **shipped in Sprint 2.5** (`src/backtest.py` + `src/calibration.py`)
- Daily delta reporting → v3
- Regime-conditional or recency-weighted retrieval → v3
- Cross-asset features (rates, FX, credit) → v3
- **Fear & Greed reconstruction** from sub-ingredients (SPX vs 125dMA, VIX vs 50dMA, 52w highs/lows, put/call ratio, HY-OAS, etc.) → v3. CNN's published series only goes back ~1y, so the 5th input has to be rebuilt before it can rejoin the encoder. If 4-input smell tests fail, this gets promoted to non-negotiable.

## Smell tests (must pass before swap-in)

| Period | Expected nearest analogs |
|---|---|
| March 2020 (COVID crash) | October 2008, September 2011, December 2018, February 2018 |
| October 2008 (Lehman week) | March 2020, August/September 1998 (LTCM) |
| January 2018 melt-up | Other late-cycle euphoria periods (late 1999, mid-2007) |
| Mid-2017 calm | Other low-vol grinds (mid-2005, mid-1995) |

If any of these fail, encoder weights are wrong. Fix before shipping.

## Tech stack

- Python 3.11+
- `pandas`, `numpy`, `pyarrow`
- `yfinance` (VIX, SPX)
- `requests` (CNN Fear & Greed endpoint)
- `scikit-learn` (k-means, Sprint 2)

No FastAPI, no vector DB, no serving infrastructure. v2 is a daily batch job that writes a parquet of "today's analogs + their outcome distribution." v1's display layer reads from that file.

## Working style

- Brutal honesty over hedging. No "I think maybe perhaps."
- Direct, no fluff. No marketing language in code comments, commits, or docs.
- Push back if I'm wrong about something.
- If a decision is ambiguous, ask one sharp question — don't generate three options for me to pick from.
- If something in this file is contradicted by what I say in chat, flag the contradiction before silently overriding.
