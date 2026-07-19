# Predicto-Forex

Research pipeline for systematic FX trading: pull clean Dukascopy history,
engineer leak-free features, and backtest models under walk-forward validation
with transaction costs and explicit uncertainty.

**Research only.** No broker integration, no live trading. See `CLAUDE.md` for
the methodology rules this codebase is built to enforce.

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
pytest
```

## Usage

```bash
# 1. Pull data (idempotent — existing partitions are skipped)
python -m src.ingest.fetch_dukascopy --symbol EURUSD --timeframe 15min \
    --start 2025-06-01 --end 2025-06-30

# 2. Validate it (non-zero exit if anything is structurally wrong)
python -m src.ingest.validate_raw --symbol EURUSD --timeframe 15min

# 3. Walk-forward backtest every baseline, net of costs
python -m src.run_pipeline --symbol EURUSD --timeframe 15min \
    --train-size 800 --test-size 200
```

Symbols, timeframes, date ranges, pip sizes and spread assumptions live in
`config/instruments.yaml`.

## Layout

| Path | Purpose |
|---|---|
| [src/ingest/fetch_dukascopy.py](src/ingest/fetch_dukascopy.py) | Dukascopy → partitioned parquet, retries, UTC normalization |
| [src/ingest/validate_raw.py](src/ingest/validate_raw.py) | Gap / duplicate / OHLC / timezone checks. Reports, never repairs |
| [src/features/build_features.py](src/features/build_features.py) | Momentum, vol, RSI, ATR, session encodings. Strictly backward-looking |
| [src/models/baseline.py](src/models/baseline.py) | Buy-hold, random, momentum, mean-reversion, ridge |
| [src/backtest/walk_forward.py](src/backtest/walk_forward.py) | Rolling / expanding splits with embargo |
| [src/backtest/costs.py](src/backtest/costs.py) | Spread, slippage, swap |
| [src/backtest/metrics.py](src/backtest/metrics.py) | Sharpe, Sortino, drawdown, Calmar + block-bootstrap CIs |
| [src/backtest/engine.py](src/backtest/engine.py) | Fold → fit → predict → size → cost → metrics |
| [src/sizing/kelly.py](src/sizing/kelly.py) | Fractional Kelly, hard-capped, with drawdown throttle |

## The three invariants

Everything else is replaceable; these are not.

1. **No lookahead.** Features at bar `t` use only data through `t`.
   `tests/test_no_lookahead.py` corrupts the future and asserts no feature
   value at or before the cut changes.
2. **Positions are shifted one bar.** A signal from bar `t`'s close earns
   `t`→`t+1`. `tests/test_engine.py::test_stale_signal_is_not_profitable`
   fails loudly if that shift is ever dropped.
3. **Net is always reported beside gross.** `BacktestResult.report()` prints
   both plus a bootstrap CI; there is no code path that reports gross alone.

Invariants 1 and 2 have been mutation-tested — deliberately breaking each one
makes the corresponding tests fail, so they are not passing vacuously.
