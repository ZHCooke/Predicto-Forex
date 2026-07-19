# Forex Modeling Project — CLAUDE.md

## Purpose

Build a home research pipeline for systematic FX trading: acquire clean historical
data, engineer features, build/backtest statistical or ML models, validate
out-of-sample with proper walk-forward methodology, and (only after that)
paper-trade before any live capital is considered.

This is a research-first project. No live trading, no broker API integration,
until the backtest/validation stage is solid and reviewed.

---

## 1. Data Sources

### Primary: Dukascopy tick data
Dukascopy is the source of truth — free, ECN-sourced tick data, widely regarded
as one of the better retail-accessible feeds.

- Web tool (manual/small pulls only, not for bulk): 
  https://www.dukascopy.com/swiss/english/marketwatch/historical/
- Programmatic access (preferred — use one of these, don't scrape the web UI):
  - `dukascopy-python` (PyPI) — https://pypi.org/project/dukascopy-python/
    - `fetch()` for static historical pulls, `live_fetch()` for streaming.
    - Supports tick-level and aggregated (M1/M5/M15/M30/H1/H4/D/W/M) intervals.
  - `dukascopy-node` (Node/CLI, if we end up wanting a non-Python fetch step) —
    https://www.dukascopy-node.app/
    - Historical data back to 1990s-2000s depending on instrument, 1000+
      instruments (FX, commodities, indices, crypto).

Do NOT use forexsb.com as the primary source — it's a secondary compilation of
Dukascopy ticks (capped at 200,000 bars/series) built for a specific EA-builder
ecosystem. Fine as a sanity-check cross-reference, not as the pipeline's source
of truth.

### Secondary / cross-checks
- MetaTrader History Center (MT4/MT5) — useful for cross-validating bar
  construction against a widely-used retail platform, not primary.
- OANDA v20 REST API — free for account holders, useful if we want a second
  independent data vendor to check for Dukascopy-specific artifacts.

---

## 2. Project Structure

```
fx-research/
├── CLAUDE.md                  # this file — keep updated each session
├── README.md                  # short human-facing summary
├── pyproject.toml / requirements.txt
├── config/
│   └── instruments.yaml       # symbol list, timeframes, date ranges
├── data/
│   ├── raw/                   # untouched pulls from dukascopy-python, gitignored
│   ├── interim/                # cleaned/resampled bars, gitignored
│   └── processed/              # feature matrices ready for modeling, gitignored
├── src/
│   ├── ingest/
│   │   ├── fetch_dukascopy.py  # wraps dukascopy-python, handles retries/rate limits
│   │   └── validate_raw.py     # gap detection, duplicate ticks, timezone checks
│   ├── features/
│   │   └── build_features.py   # returns, realized vol, technical/microstructure features
│   ├── models/
│   │   ├── baseline.py         # simple benchmarks (buy-hold, random, naive momentum)
│   │   └── ...                 # actual model(s) — TBD by strategy
│   ├── backtest/
│   │   ├── walk_forward.py     # rolling/expanding window train-test splits
│   │   ├── costs.py            # spread, slippage, swap/rollover modeling
│   │   └── metrics.py          # Sharpe, max drawdown, Calmar, hit rate, etc.
│   └── sizing/
│       └── kelly.py            # position sizing — reuse logic from sports betting pipeline
├── notebooks/                  # exploratory only, nothing here feeds production
├── tests/
└── logs/
```

---

## 3. Environment Setup

- Python 3.11+, virtualenv or conda.
- Core deps: `dukascopy-python`, `pandas`, `numpy`, `pyarrow` (for parquet storage
  of tick/bar data — CSV will not scale), `scikit-learn`, `matplotlib`.
- Optional depending on model choice: `xgboost`/`lightgbm` for tree models,
  `torch` if we go neural, `statsmodels` for classical time series (ARIMA/GARCH
  baselines).
- Store all raw pulls as parquet, partitioned by instrument/year, not CSV —
  tick data at scale will blow up CSV read times.

---

## 4. Methodology — Non-Negotiables

These come directly from lessons learned on the DESI/Fisher pipeline work —
same discipline applies here:

1. **No lookahead bias.** Any feature must only use information available
   strictly before the timestamp it's predicting. Build a unit test that
   asserts this for every feature function.
2. **Walk-forward validation, not a single train/test split.** Use rolling or
   expanding windows; report performance variance across windows, not just a
   mean.
3. **Transaction costs from day one.** Every backtest includes spread,
   estimated slippage, and swap/rollover — never report a "gross" Sharpe
   without a "net" Sharpe next to it.
4. **Kill hypotheses cheaply before scaling up.** Validate on a short
   date range / single pair before running the full multi-instrument,
   multi-year backtest. Mirrors the "kill hypotheses cheaply before
   committing HPC resources" approach used on Sciama.
5. **Position sizing via fractional Kelly**, reusing/adapting the Kelly
   criterion module from the sports betting pipelines. Cap max position size
   and max drawdown explicitly — don't let the model size itself unbounded.
6. **Explicit uncertainty flagging.** Every backtest result should report a
   confidence interval or resampled distribution (e.g. block bootstrap on
   returns), not a single point estimate.
7. **Paper trade before live.** No real capital until a strategy has passed
   walk-forward validation AND a live paper-trading period matching backtest
   expectations within a pre-registered tolerance.

---

## 4a. Suggested Skeleton — `src/ingest/fetch_dukascopy.py`

This is a starting skeleton, not a finished spec. If you (Claude Code / whatever
model is running this) have a better idea for structuring the ingestion layer —
different retry strategy, async fetching, a different on-disk layout, better
error handling, whatever — improve it. This is a sketch to save you a blank
page, not a constraint to follow literally.

```python
"""
fetch_dukascopy.py

Wraps dukascopy-python to pull historical bar/tick data and write it to
partitioned parquet under data/raw/. Should be safe to re-run (idempotent —
skip or overwrite partitions that already exist, don't duplicate).
"""

from datetime import datetime
from pathlib import Path
import time

import pandas as pd
import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_FX_MAJORS_EUR_USD  # etc.

RAW_DATA_DIR = Path("data/raw")

# Map friendly symbol names -> dukascopy_python instrument constants.
# Extend as needed; keep in sync with config/instruments.yaml.
INSTRUMENT_MAP = {
    "EURUSD": INSTRUMENT_FX_MAJORS_EUR_USD,
    # "GBPUSD": INSTRUMENT_FX_MAJORS_GBP_USD,
    # "USDJPY": INSTRUMENT_FX_MAJORS_USD_JPY,
}


def fetch_range(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str,
    max_retries: int = 3,
    retry_backoff_s: float = 5.0,
) -> pd.DataFrame:
    """
    Fetch one symbol/timeframe/date range from Dukascopy with retries.
    Raises after max_retries exhausted — caller decides whether to skip
    or halt the whole pull.
    """
    instrument = INSTRUMENT_MAP[symbol]

    for attempt in range(1, max_retries + 1):
        try:
            df = dukascopy_python.fetch(
                instrument=instrument,
                interval=timeframe,   # e.g. dukascopy_python.INTERVAL_MIN_15
                offer_side=dukascopy_python.OFFER_SIDE_BID,
                start=start,
                end=end,
            )
            return df
        except Exception as exc:
            if attempt == max_retries:
                raise
            wait = retry_backoff_s * attempt
            print(f"[{symbol}] attempt {attempt} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)


def write_partition(df: pd.DataFrame, symbol: str, timeframe: str, year: int) -> Path:
    """Write one symbol/timeframe/year partition to parquet. Idempotent."""
    out_dir = RAW_DATA_DIR / symbol / timeframe
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{year}.parquet"
    df.to_parquet(out_path, index=True)
    return out_path


def pull_symbol(
    symbol: str,
    timeframe: str,
    start_year: int,
    end_year: int,
    overwrite: bool = False,
) -> None:
    """
    Loop year-by-year (keeps memory bounded and gives natural checkpoints
    for a multi-year pull) and write each year's partition.
    """
    for year in range(start_year, end_year + 1):
        out_path = RAW_DATA_DIR / symbol / timeframe / f"{year}.parquet"
        if out_path.exists() and not overwrite:
            print(f"[{symbol}][{timeframe}][{year}] already exists, skipping")
            continue

        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31, 23, 59, 59)
        df = fetch_range(symbol, start, end, timeframe)
        path = write_partition(df, symbol, timeframe, year)
        print(f"[{symbol}][{timeframe}][{year}] wrote {len(df)} rows -> {path}")


if __name__ == "__main__":
    # Smoke test per the session plan: small range, one pair, before
    # committing to a full historical pull.
    pull_symbol("EURUSD", timeframe="15min", start_year=2026, end_year=2026)
```

Known gaps in this skeleton, deliberately left for whoever builds it out:
- No handling yet for tick-level pulls specifically (higher volume, probably
  needs day-level partitions instead of year-level).
- No timezone normalization step — needs to happen either here or in
  `validate_raw.py`, decide which and document it.
- No config-file wiring (`config/instruments.yaml` isn't read yet — this
  hardcodes one symbol as a starting point).
- No logging framework — just prints. Fine for a first pass, swap in proper
  logging once the pipeline stabilizes.

---

## 5. Open Questions / To Decide Early

- Which instrument(s) to start with — majors only (EURUSD, GBPUSD, USDJPY) or
  broader universe? Recommend starting narrow (1-2 pairs) to validate pipeline
  before scaling.
- Target holding period / frequency: intraday (M1-M15), swing (H1-H4), or
  daily+? This drives both feature engineering and how seriously transaction
  costs bite.
- Model class: classical statistical (mean-reversion/momentum signals,
  GARCH-based vol models) vs. ML (gradient boosting on engineered features) vs.
  something closer to the SBI-style approach from the DESI work. Worth
  starting with strong simple baselines before anything complex.
- Broker for eventual paper/live testing (affects realistic cost assumptions
  in the backtest) — not needed yet, but backtest cost assumptions should
  target a specific broker's actual spread/leverage terms once chosen.

---

## 6. Session Log

_(Claude Code: append a dated entry here each session — what was done, what
was found, what's next. Same pattern as the Sciama Fisher pipeline handoff
file.)_

### 2026-07-19 — Project initialized
- Decided on Dukascopy (via `dukascopy-python`) as primary data source.
- Decided against forexsb.com as primary source (secondary/compiled data,
  200k-bar cap) — useful only as a cross-check.
- Repo structure and methodology non-negotiables drafted (see above).
- Next: scaffold repo, write `fetch_dukascopy.py`, pull a small test range
  (e.g. 1 month EURUSD M15) and run `validate_raw.py` gap/duplicate checks
  before committing to a full historical pull.

### 2026-07-19 — Skeleton built, end-to-end smoke run green
Scaffolded the full repo per s2 and wired it end to end. 52 tests passing.

**Built:** ingest (`fetch_dukascopy`, `validate_raw`), features, baselines,
walk-forward splitter, costs, metrics, Kelly sizing, backtest engine, and
`src/run_pipeline.py` as the end-to-end entrypoint. Config is wired through
`config/instruments.yaml` via `src/config.py`; logging via `src/logging_setup.py`.

**Decisions made (previously open):**
- **Timezone normalization lives in ingest, not validation.** `fetch_dukascopy`
  emits a tz-aware UTC index named `timestamp`; `validate_raw` asserts that
  rather than fixing it, so a silent repair can't mask a bad pull.
- **Tick partitions are day-level** (`data/raw/SYM/tick/YYYY/YYYY-MM-DD.parquet`),
  bars stay year-level. An instrument-year of ticks won't fit in memory.
- **`validate_raw` reports, never repairs**, and exits non-zero on ERROR so it
  can gate CI. Data gaps are warnings; structural defects are errors.
- **Execution convention: `strategy_return[t] = position[t-1] * market_return[t]`.**
  Documented at the top of `engine.py` — this is the project's easiest way to
  fake a good backtest, so it's guarded by a test.
- Note `dukascopy_python.fetch()` already retries internally and paginates via
  `limit`; our wrapper's retry loop is only for whole-call failures.

**Both critical invariants are mutation-tested**, not just asserted:
- Making a feature window centered → 5 lookahead tests fail. ✓
- Dropping the position shift in `engine.py` → a stale (already-realized)
  signal reports 91% hit rate and Sharpe ~180, and the guard test fails. ✓

**Smoke run** (1 month EURUSD M15, 2028 bars, validation clean, 5 folds):
all baselines ran; `random` went from −6.0 gross Sharpe to −20.8 net, which is
the check that costs are actually biting. Every CI straddles zero except the
ones that are clearly negative — as expected on one month of data. **No signal
is claimed here; this run only proves the plumbing is correct.**

**Next:**
- Pull real history (2015→now, EURUSD M15) and re-run — current numbers are
  from one month and are statistically meaningless.
- Decide holding period (s5, still open) — it drives feature windows and how
  hard costs bite. `breakeven_edge()` in `costs.py` is the cheap first filter.
- `BARS_PER_YEAR` and `TIMEFRAME_MINUTES` are hardcoded estimates; derive them
  from the actual FX calendar once there's real multi-year data to measure.
- Swap/rollover is a flat annual rate in `CostModel` — replace with real
  per-pair carry once a broker is chosen (s5).
- No model beyond ridge yet; baselines must be beaten before adding complexity.

### 2026-07-19 (later) — Multi-year data pulled; all baselines fail net of costs

**Data:** EURUSD M15, 2015-01-01 → 2026-07-01. 286,701 bars, 12 year-partitions,
validation clean (0 errors). Per-year counts are consistent (~24,900-25,000);
price range 0.954-1.255 spans the 2022 parity break, so the data is genuine.
The 18 flagged gaps are all New Year / Christmas closures — expected, not defects.

**Bug found and fixed — partition idempotency was existence-only.** The 1-month
smoke-test `2025.parquet` (0.09 MB vs ~0.8 MB for full years) was silently
skipped by the full pull, leaving 11 months missing from the middle of the
dataset. `covers_range()` now checks a partition's actual span against the
requested range and re-fetches if it under-covers, with a 4-day tolerance for
holiday edges. Verified live: re-running the pull re-fetched only 2025 and left
the other 11 partitions untouched. Regression test in
`tests/test_partition_coverage.py`. 60 tests passing.

**Result: no baseline shows tradeable edge. Costs dominate completely.**

Initial run (train 20k / test 5k, ~53 folds), net Sharpe: buy_and_hold -0.11,
mean_reversion -1.51, momentum -3.66, ridge -9.05, random -15.93.

Ridge is the instructive case: **+3.17 gross Sharpe → -9.05 net**. Diagnosis
confirmed this is real, not a cost-model bug:
- predicted edge 0.17 pips vs **1.0 pip breakeven** — signal is 6x too small
  to pay for its own trading
- mean holding period **1.36 bars** (~20 min); it flips almost every bar
- turnover 17,947 units/yr x 0.5bp = 89.7% drag, matching the measured 90.7%
  (i.e. the arithmetic is internally consistent)
- gross +24.4%/yr is real but buried under 90.7% of costs

**Horizon sweep** (15m/1h/4h/1d/1w x ridge/momentum, with `min_edge` set to
breakeven). Setting `min_edge=breakeven_edge()` is a large lever — it cut 15m
ridge turnover 17,947 → 66 and net Sharpe -9.05 → -0.04. But **no combination
produced a net Sharpe whose CI excludes zero.** Longer horizons raise the
edge/breakeven ratio (5.4x at 1w) without improving gross Sharpe, i.e. ridge
isn't finding anything real — it's just predicting less often.

**Do not be fooled by 15m momentum** (net Sharpe 0.303, CI [0.303, 0.607]).
`fold_std` is NaN and turnover is 0.187 units/YEAR — it trades about once every
five years, so that Sharpe is computed on a near-empty return series from a
single fold. Degenerate artifact, not a signal. Any future leaderboard should
disqualify runs with near-zero turnover or NaN fold variance rather than
ranking them.

This is the pipeline working correctly: it is refusing to manufacture signal
from naive features on a liquid major. That is the expected outcome and the
reason the cost model and CIs were built in from day one.

**Next:**
- The bottleneck is signal quality, not model complexity. An xgboost on these
  same features will very likely also fail — do not reach for it first.
- Most promising directions, in order: (a) cross-pair / carry / rates-differential
  features rather than pure price-derived ones; (b) a regime filter that trades
  only in specific sessions or vol states, cutting turnover hard; (c) genuinely
  longer horizons (multi-day swing) where the 1-pip cost floor stops dominating.
- Add a turnover/degeneracy guard to `run_pipeline` so artifacts like the
  momentum case above can't reach a leaderboard.
- `min_edge` should probably default to `breakeven_edge(cost_model)` rather than
  0.0 — sizing on sub-breakeven predictions is never rational.
- Still open from s5: holding period is now partly answered (1-bar M15 is
  definitively dead), but no horizon tested is alive either.

### 2026-07-19 (later still) — Cost surface measured; horizon is the binding constraint

**Required information coefficient (IC) by horizon**, EURUSD, using measured
1-pip breakeven against realized vol at each horizon:

| horizon | vol (pips) | required IC |
|---|---|---|
| 15m | 5.2 | **0.192** |
| 1h | 10.2 | 0.098 |
| 4h | 20.3 | 0.049 |
| 1d | 48.6 | 0.021 |
| 1w | 107.9 | 0.009 |

A realistic IC for a good FX signal is 0.02-0.05. **At M15 we needed 0.19 —
roughly 5-10x beyond what is achievable.** We were not failing to find signal;
we were fishing where no achievable signal could have paid.

Notably our M15 ridge achieved IC ~0.033 (0.17 pips edge / 5.2 pips vol), which
is a *respectable* number — it lost only because the hurdle was 0.19. But at 1d
ridge's gross Sharpe went negative, so these price-derived features have no
skill at longer horizons. **Both changes are needed: longer horizon AND better
features. Neither alone is sufficient.**

**Measured spreads (2024, bid vs ask, 1h bars) — our 0.6-pip assumption was wrong:**

| pair | median spread | 1d vol (pips) | breakeven | required IC @1d |
|---|---|---|---|---|
| USDJPY | 0.60 | 100.9 | 1.00 | **0.010** |
| EURUSD | 0.20 | 39.0 | 0.60 | 0.015 |
| GBPUSD | 0.80 | 48.3 | 1.20 | 0.025 |

- EURUSD median spread is **0.20 pips, not the 0.6 we assumed** (0.6 is ~p90).
  The cost model is ~3x too pessimistic at median, ~right at the tail.
- **Spread varies ~7x by hour**: EURUSD 0.20 @14h UTC (London-NY overlap) vs
  1.45 @21h (rollover). GBPUSD 0.70→2.90, USDJPY 0.50→4.20. A flat spread
  constant is simply wrong; `CostModel` should take an hour-of-day profile,
  and 21h UTC should probably be excluded from trading entirely.
- Caveat on USDJPY: 2024 was an exceptional vol regime (BoJ intervention, carry
  unwind). Re-measure across 2015-2026 before treating 0.010 as durable.

**Pair choice matters far less than horizon.** Best-to-worst pair spans
0.010-0.025 required IC (2.5x). Horizon spans 0.009-0.192 (21x). Chasing
"less efficient" exotic pairs is a trap — spread grows faster than opportunity.

**Feature direction — the real bottleneck.** Every current feature is a
transform of a single price path (momentum, z-score, RSI, ATR, range position),
so they carry no information the price series doesn't already contain. Ranked
by expected value:
1. **Rates/carry differential** (2y yield spread, policy rates; free from FRED).
   The most robustly documented FX return driver. Slow-moving, suits daily+.
2. **Cross-pair USD factor** — decompose EURUSD into EUR-strength and
   USD-strength via a basket (GBPUSD, AUDUSD, USDJPY...). Genuinely new
   information relative to EURUSD alone; FX is a relative-value market.
3. **COT positioning** (CFTC, weekly, free) — speculative positioning extremes
   are a documented reversal signal.
4. **Measured spread / microstructure** — now that we can pull ask side and
   ticks: spread as a regime feature, tick-volume imbalance, trade intensity.
5. **Macro event calendar** (NFP, FOMC, ECB) — returns around scheduled events
   are structurally different; worth a flag or an exclusion.

### 2026-07-19 (session 4) — FRED rates wired in; carry financing fixed

**Built:** `src/ingest/fetch_fred.py` (no API key needed — public fredgraph CSV
endpoint), `resample_bars()` in build_features, `with_macro=True` on
`assemble_dataset`, and real signed carry in `apply_costs`. 91 tests passing.

**Usable FRED series (daily, non-revised):** `DGS2`, `DGS10`, `DFF`, `ECBDFR`.
Euro-area and JP yield series on FRED are monthly AND lag ~6 months in
publication (`IRLTLT01EZM156N` last point 2026-01-01), so they are unusable
for a daily strategy. `INTDSRJPM193N` died in 2017. EURUSD differential is
therefore `ECBDFR - DFF`, both daily policy rates, symmetric.
Sanity-checked against history: ECB deposit rate spans -0.50% (NIRP era) to
4.00% (2023); Fed funds 0.04% to 5.33%. Correct.

**Alignment causality** is mutation-tested (`tests/test_fred_alignment.py`):
switching ffill→bfill, or dropping the 1-day lag, both fail the guards. FRED
serves latest-revised data, so only market/policy rates are used —
do NOT add GDP/CPI/payrolls without switching to ALFRED vintages.

**Finding 1: rate features are NOT a directional predictor of spot.**
`corr(f_rate_diff, fwd_ret) = -0.005` against a required IC of 0.021. Adding
the 6 macro features made ridge strictly worse (4h 0.083 → -0.189 net Sharpe;
1D -0.002 → -0.468) — extra noise, no signal. Momentum/buy-hold were
byte-identical with and without, the correct consistency check.

**Finding 2 (my error, now fixed): the first carry test was mis-specified.**
Carry earns the differential as INCOME; our target was spot return only and
`CostModel` charged a flat symmetric 1%/yr swap. That makes carry untestable
by construction — the thing being harvested never entered the P&L. `apply_costs`
now takes `carry_annual` (signed, real differential) and returns a `carry`
column. Sign convention: EUR rates average 1.45% BELOW USD, so long EURUSD
bleeds and short earns.

**Finding 3: a candidate, NOT a validated edge.** Daily EURUSD, momentum,
real carry, measured 0.2-pip spread: net Sharpe **0.584, CI [0.102, 1.121]** —
the first CI in this project that excludes zero. Turnover 68/yr, 10 folds, so
not degenerate like the earlier artifact.

**Do not act on this yet. Two serious caveats:**
1. **Multiple comparisons.** We have now tested ~33 configurations (5 baselines,
   10 horizon-sweep, 12 macro/timeframe, 6 carry). At 95% confidence, ~1.7 are
   expected to clear zero BY CHANCE. One CI barely above zero (lower bound
   0.102) is exactly what noise predicts. This needs a Bonferroni-style
   correction or, better, a held-out period never touched during search.
2. **Most of the improvement is a cost-model change, not signal.** Momentum's
   carry P&L is -0.005 (≈zero). It went 0.465 → 0.584 mainly because real
   signed financing replaced a flat 1%/yr charge that had been applied
   regardless of direction. More accurate, but it flatters results by removing
   a systematic drag rather than by finding anything.

**Next:**
- **Carve out a true holdout** (e.g. 2023-2026) and do not run a single
  backtest on it until a hypothesis is pre-registered. Everything to date has
  seen all the data; the ~33 configurations have contaminated the full sample.
- Apply a multiple-comparisons correction to any leaderboard.
- Add the degeneracy guard (near-zero turnover / NaN fold variance) that the
  earlier momentum artifact motivated — still not done.
- `min_edge` should default to `breakeven_edge(cost_model)` — still not done.
- Cross-pair USD factor is the untested feature with the best prior; rates are
  now known not to work directionally on their own.

### 2026-07-19 (session 5) — Holdout sealed; h001 came back INCONCLUSIVE

**Built:** `src/backtest/holdout.py` (seal date, pre-registration, append-only
access log, Bonferroni helper), `BacktestResult.degeneracy`, and `min_edge`
now defaults to `breakeven_edge(cost_model)`. 90 tests passing.

**Holdout:** sealed at **2023-01-01**. Research 2015-01-02..2022-12-31 (2501
daily bars), holdout 2023-01-02..2026-07-02 (1095 bars, 30% of sample).
Access is logged to `logs/holdout_access.log`; every read is counted and
reported alongside any result.

**Research-only leaderboard (pre-2023)** picked `naive_momentum`: net Sharpe
0.794, CI [0.107, 1.626], 6 folds, turnover 67/yr, non-degenerate. The new
degeneracy guard correctly flagged `buy_and_hold` (turnover 0.173/yr).

**h001 pre-registered** at predicted net Sharpe 0.40 ± 0.50, then evaluated
once. Result: **INCONCLUSIVE, not confirmed.**

Observed Sharpe 0.552 — nominally inside tolerance — but the degeneracy guard
fired on all of: in-market only **0.89% of bars**, 1 of 3 folds producing a
finite Sharpe, NaN across-fold variance, and a total net return of **0.7% over
3.5 years**. The strategy essentially stopped trading in the holdout: momentum
predictions fell below breakeven almost everywhere once `min_edge` defaulted
to the real threshold. A Sharpe computed on ~1% exposure is not evidence.

**This exposed a flaw in the framework itself.** `evaluate_against_prereg`
originally returned `confirmed: true` for this — a pre-registration system that
rubber-stamps a degenerate result is worse than none, because it launders noise
as validation. It now requires a clean degeneracy report and returns an explicit
verdict of CONFIRMED / REFUTED / INCONCLUSIVE (degenerate). Regression test in
`tests/test_holdout.py::test_degenerate_result_cannot_be_confirmed`.

**Important caveat on this holdout:** ~33 configurations were searched over the
FULL 2015-2026 sample in sessions 3-4, before the seal existed. The formal
selection for h001 used research data only, but the analyst's priors had
already seen the holdout. This holdout is therefore *partially burnt* and is
best-effort, not pristine. Genuinely clean validation now requires forward
paper trading (s4.7) on data that does not yet exist.

**Status: no validated edge. h001 neither confirmed nor refuted — it was not
really tested.** The honest read of five sessions is that naive price-derived
features on EURUSD do not clear costs at any horizon tried, and the one
candidate that survived research selection evaporated when the breakeven filter
was applied properly.

**Next:**
- h001 is spent as an id. A retest needs a NEW hypothesis id and should size
  the position differently (e.g. lower `min_edge`, or a signal with enough
  amplitude to clear breakeven) so the holdout actually gets exercised.
- Cross-pair / currency-factor features (see s7 below) are the best remaining
  untested idea and should be developed on research data only.
- Consider a second seal (e.g. 2025+) reserved for whatever comes after the
  cross-pair work, since 2023-2026 is now partially spent.

---

## 7. On Multi-Currency (Triangular) Relationships

Asked in session 5: "is it worth trying to work the relationship between three
currencies?" Short answer: **yes, but not as arbitrage.** Three distinct ideas
get conflated here and they have very different prospects.

**1. Triangular arbitrage — dead. Do not pursue.**
EURUSD x USDJPY = EURJPY must hold by no-arbitrage. Any deviation is closed by
co-located HFT in microseconds, and the deviation is smaller than our 0.2-0.6
pip spread before we could act. We measured retail costs at ~0.6-1.2 pips
round trip; the mispricing is a fraction of that. This is structurally
unavailable to a retail research pipeline and should not be attempted.

**2. Currency factor decomposition — the best untested idea we have.**
EURUSD is a *ratio*: a move can come from EUR strength or USD weakness, and the
single price series cannot distinguish them. With a basket (EURUSD, GBPUSD,
AUDUSD, USDJPY, USDCHF...) you can decompose returns into per-currency strength
factors — if every USD pair moves together it is a dollar move, not a euro move.

This is genuinely *exogenous information relative to the EURUSD price path*,
which is exactly what every feature we have built so far lacks (see session 4:
all price-derived features are transforms of one series and carry no new
information). It is the same reason rates were worth trying, but with a much
stronger prior because it is mechanically related to the thing we are
predicting rather than macro-linked to it.

**3. Cross-sectional ranking — the well-documented academic version.**
The FX factor literature (carry, momentum, value) finds these work markedly
better *cross-sectionally* — rank currencies, go long the top and short the
bottom — than as time-series signals on a single pair. Our entire project so
far has been time-series on one pair, i.e. the weaker formulation.

**Cost warning, which is the catch.** Trading N pairs multiplies spread cost by
roughly N, and triangular positions are *redundant*: an EURJPY position is
already implied by EURUSD + USDJPY exposure. Expressing a view through three
legs pays three spreads for two legs' worth of exposure. Any implementation
must net exposures down to the minimum spanning set before trading. Given that
costs have killed every strategy tested so far, this is the binding constraint,
not the signal.

**Recommended approach if pursued:** pull 5-6 USD majors, build per-currency
strength factors, test cross-sectional ranking at daily horizon, net positions
to minimum legs, and develop entirely on pre-2023 research data.

---

### 2026-07-19 (session 6) — Two cost-side bugs fixed; earlier results were too harsh

**Built:** `src/ingest/measure_spread.py`, hour-aware `CostModel`
(`spread_profile`, `cost_per_unit_series`, `from_measured_profile`),
`mean_holding_bars`, `amortized_breakeven_edge`, and an `allowed_hours`
session filter in the engine. 109 tests passing.

**Bug 1 — `min_edge` was ~7x too strict.** It compared each bar's predicted
edge against the FULL round-trip breakeven, but positions persist ~7.1 bars, so
the spread is paid once per 7 bars. The correct per-bar hurdle is cost/H.
Against the amortized figure 95.7% of bars clear versus 72.4% before. This is
what silently zeroed the h001 holdout test (in-market 0.89% of bars).

**Consequence: several session-4 conclusions were biased pessimistic.** Re-run
on research data with the fix plus measured spreads, 4h ridge goes **-3.075 ->
-0.301** and 1d momentum **-0.044 -> +0.278**. The horizon sweep was penalising
strategies for costs they would not have paid.

**Measured spread profile (EURUSD, 2019-2022, ~25k hourly obs):** flat at
**0.30 pips from 06:00-19:00 UTC**, 0.40 overnight, then **1.20 / 1.50 / 0.60
at 20:00 / 21:00 / 22:00** (rollover). Cheapest-to-dearest ratio 5.0x, but only
THREE hours are expensive. Cached at `data/raw/spread/EURUSD_hourly.json`.

This corrected an earlier recommendation: session 4 suggested restricting to
12:00-16:00 UTC based on a 2024-only sample. That would have been **wrong** —
it would cut ~80% of trading opportunities for no cost benefit, because the
day is essentially flat outside rollover. The right rule is to exclude 20-22
UTC only. Measure before restricting.

**Bug 2 — the session filter's first implementation made things worse.**
Zeroing positions during blocked hours forces a flatten-and-re-enter every day.
Spread is paid on TRADES, not on exposure, so holding through the rollover is
free while closing out costs a full round trip. Measured: 4h mean-reversion
turnover 162 -> 552/yr and net Sharpe +0.380 -> +0.086. Fixed by FREEZING the
position (NaN + ffill) instead. After the fix the filter improves net Sharpe in
5 of 6 configurations and reduces turnover, as intended.

Both bugs are the same species: **a filter that costs more than it saves.**
Worth watching for in anything added next.

**Research-only leaderboard after both fixes (2015-2022):**
best is 4h mean-reversion at net Sharpe **0.451**, then 1d momentum 0.278.
Everything else is negative. **Nothing has a CI excluding zero.**

**Robustness red flag:** 1d momentum scored 0.794 with a 750/250 splitter in
session 5 but 0.278 with 600/200 here. A result that swings that much on an
arbitrary split parameter is behaving like noise, not signal. Any future
candidate should be checked across several splitter settings BEFORE it is
pre-registered.

**Next:**
- Config count is now ~50. Bonferroni alpha is 0.001; a candidate needs a
  ~99.9% interval excluding zero. Nothing is close.
- Cross-pair currency factors (s7) remain the best untested idea, and now
  inherit a correctly-specified cost model.
- Still worth agreeing an explicit stopping rule (proposed: net Sharpe > 0.5 on
  a clean holdout with turnover > 20/yr and stability across splitters, or stop
  and treat the pipeline as the deliverable).

---

## 8. Success Criteria (AGREED 2026-07-19)

A strategy is worth paper trading if and ONLY if it clears all four:

1. **Net Sharpe > 0.5** on a clean holdout, after all costs.
2. **Turnover > 20/yr** — no degenerate near-zero-trading artifacts.
3. **Stable across splitters** — must survive several train/test window sizes.
   Added because 1d momentum scored 0.794 at 750/250 but 0.278 at 600/200,
   which is noise behaviour.
4. **No degeneracy flags** from `BacktestResult.degeneracy`.

Owner has stated the goal is profitability even if it takes a long time, so
the search continues rather than stopping at the first null result. The bar
above is what converts "looks good" into "worth risking money on". It may be
revised, but ONLY before a result is seen, never after.

---

## 9. Transferable Techniques From Sibling Modules

Reviewed `PL-AFL-Module` and `PL-Tennis-Module` (2026-07-19) for approaches
worth importing. Four are genuinely relevant; one is a trap.

**9.1 Elo-style currency strength ratings (from PL-Tennis-Module).**
`ATPBetting/Python/elo_features.py` maintains a per-player rating updated after
each head-to-head. This maps almost exactly onto the cross-pair problem: every
FX pair's move is a "match" between two currencies, and a recursive rating gives
per-currency strength. This is the natural implementation of the s7 currency
factor idea. Needs adapting from win/loss to a continuous margin (the size of
the move matters, not just its sign) — the AFL margin models are the reference
for that.

**9.2 Isotonic calibration (from PL-AFL-Module `models/calibration.py`).**
Probably the highest-value single import. Our Kelly sizer consumes RAW predicted
magnitudes (`mu/sigma^2`), and we have direct evidence they are miscalibrated:
at 1d horizon ridge predicted edges 2.09x breakeven while delivering NEGATIVE
gross Sharpe — confidently wrong. Isotonic regression maps raw model output onto
empirically-observed outcomes without assuming a functional form. Calibrating
before sizing should fix systematically oversized positions.

**9.3 Ensemble over feature subsets with disagreement (AFL `models/ensemble.py`).**
Trains N models on different feature subsets and reports mean plus spread
(`pred_margin_std`, p10/p90, `ensemble_agreement_pct`). Two wins for us:
it satisfies the s4.6 uncertainty requirement per-prediction rather than only
in aggregate, AND model disagreement is a natural "do not trade" filter. Since
costs are our binding constraint, trading only when models agree attacks the
problem from the right side.

**9.4 Paper-trading ledger (Tennis `model/paper_trading.py`, `paper_ledger.py`).**
Existing structure for the s4.7 paper-trading requirement. Reuse rather than
rebuild when we get there.

**9.5 THE TRAP — feature sweeps (Tennis `model/feature_sweep.py`).**
Randomised feature-group search that reweights toward top performers. Powerful,
and exactly the wrong tool for us right now. We have already searched ~50
configurations; Bonferroni alpha is 0.001. A sweep evaluates hundreds more and
would almost certainly surface something that looks significant and is not.
If used at all it must run ONLY on research data, with the configuration count
tracked and the correction applied — and the winner still has to clear s8 on
the sealed holdout.

**On model complexity:** we are currently using only ridge regression plus
trivial baselines. That was deliberate (baselines first), and the evidence still
says the bottleneck is signal, not model class — ridge already achieves a
respectable IC of ~0.034; it fails on costs and stability, not on capacity. A
GBM on the same features would very likely fail the same way. The imports above
are therefore about calibration, uncertainty and NEW information (currency
strength), not about a bigger model.

**Relevant literature (recalled, NOT verified — worth a search before relying
on any specific figure):** Lustig, Roussanov & Verdelhan on common risk factors
in currency markets (the "dollar factor" and carry factor); Menkhoff, Sarno,
Schmeling & Schrimpf on carry trades, FX volatility, and currency momentum;
Asness, Moskowitz & Pedersen "Value and Momentum Everywhere" which includes
currencies. **Important caveat:** these document returns on broad portfolios
often including emerging/high-yield currencies, frequently gross of realistic
retail costs, and the carry factor notably crashed in 2008. "Documented in the
literature" does not imply "survives retail costs on G10 majors" — which is
precisely the constraint that has killed everything we have tested.

---

### 2026-07-19 (session 7) — Cross-pair build started

Enabled G10 majors in config and `INSTRUMENT_MAP`: GBPUSD, USDJPY, AUDUSD,
USDCHF, USDCAD, NZDUSD (plus existing EURUSD). Added `PAIR_LEGS` mapping each
pair to its (base, quote) currency so a pair return can be attributed to two
currencies rather than one instrument.

Pulling **1h** bars 2015-2026 for the six new pairs — 1h is ample for daily/4h
factor work and ~12x faster than M15. EURUSD retains full M15 history.

**Data validated:** all six pairs 2015-2026, ~71,700 1h bars each, zero errors.
Price ranges are historically genuine — USDCHF's 0.761 low is the Jan-2015 SNB
de-peg, GBPUSD's 1.04 the 2022 mini-budget crash, USDJPY's 162.8 the 2024 peak.

**Built `src/features/currency_strength.py`.** Solves `r_pair = s_base - s_quote`
across the basket by least squares with a sum(s)=0 identification constraint,
rather than a sequential Elo update — we observe the exact margin, so the
algebra is available where tennis only has win/loss. The Elo idea survives as
the rating layer (EWMA decay == K-factor). 121 tests passing.

**Decomposition verified exactly:** `|s_EUR - s_USD - r_EURUSD| = 2.08e-17`,
machine precision. Cumulative 2015-2022 strengths are economically coherent:
CHF +19.0% (safe haven), USD +11.5% (dollar era), GBP **-13.7%** (Brexit),
commodity currencies AUD/NZD/CAD all negative. The factors are measuring
something real.

**A headline number that did not survive scrutiny.** `f_ccy_dispersion_*`
showed IC ~0.05, the strongest in the project. Year-by-year it is
-0.079, -0.041, -0.015, **+0.029, +0.028**, -0.122, +0.015, -0.015 — only 62%
sign agreement, with the pooled figure driven by 2015 (SNB) and 2020 (COVID).
Cause: dispersion is UNSIGNED, so it cannot distinguish "USD dominating" from
"EUR dominating"; it merely coincided with dollar strength in crisis years.
Rejected.

**A rolling z-score variant did not help.** Motivated by a real diagnostic (raw
`f_ccy_diff_5` averaged IC -0.055 within years but far less pooled, the
signature of level drift), but `f_ccy_diff_z_5` fell to 50% sign agreement.
Kept in the module, not used. Caveat noted: adding the z-score columns extended
the warm-up and dropped ~240 early rows, so before/after IC figures are not
quite the same sample.

**The actual result — first stable signal in the project, but below the bar.**
`f_ccy_diff_5` (EUR-minus-USD strength vs the basket, 5-bar halflife) has 88%
sign agreement and a per-year mean IC of **-0.055**, roughly 3x the ~0.02
required at daily horizon. Negative sign = short-term reversal: when EUR has
strengthened against USD relative to the basket, EURUSD tends to fall back.

Backtested across four splitter settings (criterion s8.3), research data only:

| split | ccy_reversion | ridge on all 20 strength features |
|---|---|---|
| 500/150 | **0.210** | 0.364 |
| 600/200 | **0.180** | 0.191 |
| 750/250 | **0.273** | -0.198 |
| 900/300 | **0.288** | -0.526 |

The single-feature reversion model is **stable** — 0.18 to 0.29 across every
split, turnover ~85/yr, no degeneracy flags. That is the first time anything in
this project has held its shape when the splitter moved. Ridge on all 20
features swings from +0.36 to -0.53: classic overfitting, and further evidence
that capacity is not the constraint.

**Verdict against s8: 3 of 4 criteria met.**
- (1) Net Sharpe > 0.5 — **FAILED**, ~0.24. All CIs straddle zero.
- (2) Turnover > 20/yr — passed (~85).
- (3) Stable across splitters — passed, for the first time.
- (4) No degeneracy flags — passed.

Not a pass. Do NOT pre-register this against the holdout yet — spending a
holdout look on a 0.24 candidate wastes the one clean test we have.

**Next (principled improvements, NOT more search — config count is now ~60):**
- Apply isotonic calibration (s9.2) before Kelly sizing. Still unapplied, and
  it is the highest-prior fix: sizing consumes raw magnitudes we know are
  unreliable.
- Ensemble with a disagreement filter (s9.3) — trade only on agreement, which
  attacks the cost side rather than hunting more signal.
- Combine `f_ccy_diff_5` with the price/carry features in ONE model rather than
  testing families separately.
- Resist the feature sweep (s9.5). At ~60 configurations the correction is
  already punishing.

### 2026-07-19 (session 8) — Both AFL imports built; both made things WORSE

Built `src/models/calibration.py` (isotonic return calibration, chronological
held-out calibration slice) and `src/models/ensemble.py` (feature-family
ensemble, direction-agreement filter), wired `min_agreement` into the engine.
139 tests passing, including that a skill-less model collapses toward a
constant under calibration and therefore stops trading.

**Result on the daily EURUSD cross-pair strategy, research data, 3 splitters:**

| variant | mean net Sharpe | std across splits |
|---|---|---|
| **A. base reversion (`f_ccy_diff_5`)** | **+0.257** | **0.041** |
| B. + isotonic calibration | -0.593 | 0.194 |
| C. family ensemble (ridge members) | -0.391 | 0.205 |
| D. ensemble + agreement filter | +0.022 | 0.421 |
| E. ensemble + calibration + agreement | -0.562 | 0.255 |

**The simplest model is still the best, and by far the most stable** (std 0.041
against 0.19-0.42 for everything else). Variant D produced a 0.506 at one
splitter and -0.26/-0.18 at the others — the signature of noise, not signal.

**I predicted calibration would be "probably the highest-value single import".
That was wrong, and the reason is instructive.** The AFL setting is thousands of
matches, a GBM with genuinely non-linear miscalibration, and a large calibration
sample. Ours is ~2,400 daily bars, ONE weak feature, and a near-linear
relationship. Isotonic regression is non-parametric — it spends roughly as many
degrees of freedom as it has calibration rows — so on a 150-270 row slice it
overfits, and holding that slice back also starves the base fit. The technique
was sound; the regime was wrong.

Same story for the ensemble: averaging three ridge members does not fix
overfitting when the members themselves overfit (already known — ridge on the
20 strength features swung +0.36 to -0.53 across splitters).

**Transferable lesson: our binding constraints are sample size and signal
strength, not model calibration or averaging. Techniques that ADD flexibility
should be expected to hurt here.** Import complexity only alongside more data
or a stronger signal.

Config count ~72. Bonferroni alpha 0.0007.

**Status vs s8 unchanged: 3 of 4.** Base reversion still ~0.26 against a 0.5
bar, CIs still straddle zero. Nothing here justifies a holdout look.

**Next — the honest options, in order of expected value:**
1. **More data, not more technique.** ~2,400 daily bars is the real limit. The
   same strategy on 6 more pairs (each with its own `f_ccy_diff`) would give
   ~7x the observations from ONE hypothesis, not seven — a cross-sectional
   portfolio rather than seven separate tests. This is both the standard form
   in the FX factor literature and the only move that attacks sample size.
2. Intraday (4h) cross-pair strength: ~6x more bars, though costs bite harder.
3. Accept ~0.26 is what this signal is worth and revisit the s8 bar.
