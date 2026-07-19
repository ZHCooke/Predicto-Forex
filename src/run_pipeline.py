"""
run_pipeline.py

End-to-end smoke run: load raw bars -> features -> walk-forward backtest of
every baseline -> gross/net report with confidence intervals.

    python -m src.run_pipeline --symbol EURUSD --timeframe 15min

Per CLAUDE.md s4.4 ("kill hypotheses cheaply"), this is deliberately meant to
be run on a short range and a single pair first. If a model can't clear the
baselines here, it doesn't earn a full multi-year multi-instrument run.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.backtest.costs import CostModel
from src.backtest.engine import run_backtest
from src.backtest.walk_forward import WalkForwardSplitter
from src.config import load_instruments
from src.features.build_features import BARS_PER_YEAR, assemble_dataset, log_returns
from src.ingest.validate_raw import load_raw, validate
from src.logging_setup import setup_logging
from src.models.baseline import BASELINES
from src.sizing.kelly import KellyConfig

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FX research pipeline")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", default="15min")
    parser.add_argument("--horizon", type=int, default=1, help="target horizon in bars")
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--mode", choices=["expanding", "rolling"], default="expanding")
    parser.add_argument("--kelly-fraction", type=float, default=0.25)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    setup_logging(logfile="pipeline.log")

    bars = load_raw(args.symbol, args.timeframe)

    if not args.skip_validation:
        report = validate(bars, args.symbol, args.timeframe)
        report.log()
        if not report.ok:
            raise SystemExit(
                "raw data failed validation; fix it or re-run with --skip-validation"
            )

    X, y = assemble_dataset(bars, timeframe=args.timeframe, horizon=args.horizon)
    market_returns = log_returns(bars["close"]).reindex(X.index)
    log.info("dataset: %d rows x %d features", len(X), X.shape[1])

    cfg = load_instruments().get(args.symbol)
    bars_per_year = BARS_PER_YEAR.get(args.timeframe, 252)
    cost_model = CostModel(
        pip=cfg.pip if cfg else 0.0001,
        spread_pips=cfg.typical_spread_pips if cfg else 0.6,
        bars_per_year=bars_per_year,
    )
    splitter = WalkForwardSplitter(
        train_size=args.train_size,
        test_size=args.test_size,
        # Embargo must cover the target horizon or the last training labels
        # leak information from the test window.
        embargo=args.horizon,
        mode=args.mode,
    )
    kelly = KellyConfig(fraction=args.kelly_fraction)

    rows = []
    for name, model_cls in BASELINES.items():
        try:
            result = run_backtest(
                X, y, market_returns,
                model_factory=model_cls,
                splitter=splitter,
                cost_model=cost_model,
                kelly=kelly,
                bars_per_year=bars_per_year,
            )
        except Exception as exc:  # noqa: BLE001 - one bad baseline shouldn't kill the run
            log.exception("model %s failed: %s", name, exc)
            continue

        log.info("\n=== %s ===\n%s", name, result.report())
        rows.append(
            {
                "model": name,
                "gross_sharpe": result.gross.sharpe,
                "net_sharpe": result.net.sharpe,
                "net_sharpe_lo": result.net_sharpe_ci[0],
                "net_sharpe_hi": result.net_sharpe_ci[1],
                "fold_sharpe_std": result.fold_metrics["sharpe"].std(),
                "net_return": result.net.total_return,
                "max_dd": result.net.max_drawdown,
            }
        )

    if rows:
        table = pd.DataFrame(rows).set_index("model").sort_values("net_sharpe", ascending=False)
        log.info("\n=== leaderboard (net, after costs) ===\n%s",
                 table.to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()
