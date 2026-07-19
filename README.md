# Predicto-Forex

Research pipeline for systematic FX trading: pull clean Dukascopy history,
engineer leak-free features, and test hypotheses under statistical scrutiny
strong enough to kill our own results.

**Research only.** No broker integration, no live trading. `CLAUDE.md` §0 is
the entry point — it carries current state, unfinished work, and the traps
we've actually hit.

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
pytest                                            # 200 tests
```

## Usage

```bash
# 1. Pull data (idempotent — coverage-aware, safe to re-run)
python -m src.ingest.fetch_dukascopy --symbol EURUSD --timeframe 1h \
    --start 2015-01-01 --end 2026-07-01

# 2. Validate (non-zero exit if anything is structurally wrong)
python -m src.ingest.validate_raw --symbol EURUSD --timeframe 1h

# 3. Supporting data
python -m src.ingest.fetch_fred            # interest rates
python -m src.ingest.fetch_cot             # CFTC positioning
python -m src.ingest.measure_spread        # hourly spread profile

# 4. Walk-forward backtest, net of costs
python -m src.run_pipeline --symbol EURUSD --timeframe 15min \
    --train-size 20000 --test-size 5000
```

## Where the numbers come from

Nothing here is assumed if it can be measured.

- **Spread is measured, not guessed.** EURUSD is 0.30 pips through most of the
  day and 1.50 at the 21:00 UTC rollover — a 5× swing a flat constant cannot
  represent.
- **Prices are mid, not bid.** Bid-only quotes manufacture fake signal at
  illiquid hours (t = 7.96 on bid vs 1.57 on mid for the same "effect").
- **Costs amortize over the holding period.** A position held 7 bars pays one
  round trip, not seven.

## The five invariants

Everything else is replaceable.

1. **No lookahead.** Features at bar `t` use only data through `t`. Tests
   corrupt the future and assert nothing before the cut moves.
2. **Positions are shifted one bar.** A signal from `t`'s close earns `t`→`t+1`.
3. **Net is always reported beside gross.**
4. **Per-prediction significance before Sharpe.** Sharpe needs ~17 years to
   resolve a 0.5 — it cannot adjudicate anything at this sample size.
5. **The holdout (2023-01-01 onward) is sealed.** One look per pre-registered
   hypothesis, and every access is logged.

Invariants 1 and 2 are **mutation-tested** — deliberately breaking each makes
the corresponding tests fail, so they are not passing vacuously.

## Status

**No validated edge.** A screen of 82 price/carry/strength features found one
nominal hit where chance gives 4.1 — fewer than randomness produces. Momentum,
mean-reversion, RSI, ATR, carry, currency strength and positioning are all
rigorously null.

The one survivor is not a price pattern: **intraday seasonality at the London
open**, a structural flow effect. It is unspent and awaiting the holdout test.
See `CLAUDE.md` §0.3.
