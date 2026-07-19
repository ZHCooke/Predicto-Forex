"""
engine.py

Ties everything together: walk-forward folds -> fit model on train -> predict
on test -> Kelly-size -> apply costs -> metrics.

THE EXECUTION TIMING CONVENTION (the single most important thing in here):

    Features at bar t are computed from data up to and including the close
    of bar t. A position can therefore only be OPENED at the close of t, and
    it earns the market move from close(t) to close(t+1).

    strategy_return[t] = position[t-1] * market_return[t]

    i.e. positions are shifted forward one bar before being multiplied by
    returns. Getting this wrong is the classic way to produce a backtest with
    a Sharpe of 8 that loses money live.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.backtest.costs import (
    CostModel,
    amortized_breakeven_edge,
    apply_costs,
    breakeven_edge,
    mean_holding_bars,
)
from src.backtest.metrics import PerformanceSummary, bootstrap_ci, summarize
from src.backtest.walk_forward import Split, WalkForwardSplitter, summarize_folds
from src.models.baseline import BaseModel
from src.models.ensemble import apply_agreement_filter
from src.sizing.kelly import KellyConfig, size_positions

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Everything a backtest produces. Never report `gross` without `net`."""

    returns: pd.DataFrame          # gross / spread_cost / swap_cost / net per bar
    positions: pd.Series
    predictions: pd.Series
    fold_metrics: pd.DataFrame     # one row per walk-forward fold
    gross: PerformanceSummary
    net: PerformanceSummary
    net_sharpe_ci: tuple[float, float]
    agreement: pd.Series | None = None

    @property
    def fold_summary(self) -> pd.DataFrame:
        return summarize_folds(self.fold_metrics)

    @property
    def degeneracy(self) -> list[str]:
        """
        Reasons this result is a statistical artifact rather than a strategy.

        Motivated by a real near-miss: a momentum config reported net Sharpe
        0.303 with a positive CI while trading 0.187 units per YEAR — roughly
        once every five years. Its Sharpe came from a near-empty return series
        in a single fold. Anything flagged here must be disqualified from a
        leaderboard, not ranked on it.
        """
        reasons: list[str] = []

        if self.net.turnover < 1.0:
            reasons.append(f"turnover {self.net.turnover:.3f}/yr — barely trades")

        n_active = int((self.positions != 0).sum())
        frac_active = n_active / max(1, len(self.positions))
        if frac_active < 0.01:
            reasons.append(f"in-market only {100 * frac_active:.2f}% of bars")

        valid_folds = self.fold_metrics["sharpe"].notna().sum()
        if valid_folds < 3:
            reasons.append(f"only {valid_folds} folds produced a finite Sharpe")

        if len(self.fold_metrics) > 1 and pd.isna(self.fold_metrics["sharpe"].std()):
            reasons.append("across-fold variance is NaN — cannot assess stability")

        if self.net.n_bars < 100:
            reasons.append(f"only {self.net.n_bars} return observations")

        return reasons

    @property
    def is_degenerate(self) -> bool:
        return bool(self.degeneracy)

    def report(self) -> str:
        from src.backtest.metrics import format_report

        head = format_report(self.gross, self.net, self.net_sharpe_ci)
        folds = self.fold_summary.loc[["sharpe", "net_return"]].to_string(float_format="%.4f")
        return f"{head}\n\nAcross {len(self.fold_metrics)} folds:\n{folds}"


def run_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    market_returns: pd.Series,
    model_factory,
    splitter: WalkForwardSplitter,
    cost_model: CostModel | None = None,
    kelly: KellyConfig | None = None,
    bars_per_year: int = 24_800,
    vol_window: int = 96,
    carry_annual: pd.Series | None = None,
    allowed_hours: list[int] | None = None,
    min_agreement: float | None = None,
) -> BacktestResult:
    """
    Walk-forward backtest of `model_factory()` over aligned X / y / market_returns.

    `model_factory` is a zero-arg callable returning a fresh, unfitted model —
    a new instance is built per fold so nothing carries across folds.
    """
    cost_model = cost_model or CostModel(bars_per_year=bars_per_year)
    kelly = kelly or KellyConfig()

    X, y = X.align(y, join="inner", axis=0)
    market_returns = market_returns.reindex(X.index)

    # Trailing realized vol drives Kelly's denominator. Computed on the full
    # series but strictly backward-looking, and shifted so bar t uses vol
    # estimated from bars up to t-1 only.
    vol = market_returns.rolling(vol_window).std().shift(1)

    n_folds = splitter.n_splits(len(X))
    if n_folds == 0:
        raise ValueError(f"dataset of {len(X)} rows yields no folds for {splitter!r}")
    log.info("running %d walk-forward folds over %d rows", n_folds, len(X))

    all_predictions: list[pd.Series] = []
    all_agreement: list[pd.Series] = []
    fold_rows: list[dict] = []

    for split in splitter.split(X):
        X_train, y_train = X.iloc[split.train_idx], y.iloc[split.train_idx]
        X_test = X.iloc[split.test_idx]

        model: BaseModel = model_factory()
        model.fit(X_train, y_train)

        # If the model reports per-prediction uncertainty (EnsembleModel does),
        # suppress bars its members disagree on BEFORE sizing — a disputed bar
        # then produces no position, no trade and no cost.
        if min_agreement is not None and hasattr(model, "predict_with_uncertainty"):
            unc = model.predict_with_uncertainty(X_test)
            preds = apply_agreement_filter(
                unc["mean"].rename(getattr(model, "name", "ensemble")),
                unc["agreement"],
                min_agreement,
            )
            all_agreement.append(unc["agreement"])
        else:
            preds = model.predict(X_test)

        all_predictions.append(preds)

        fold_rows.append(_fold_metrics(split, preds, market_returns, vol, cost_model, kelly, bars_per_year))

    predictions = pd.concat(all_predictions).sort_index()
    # Test windows are non-overlapping by default, but guard anyway.
    predictions = predictions[~predictions.index.duplicated(keep="first")]

    # Resolve an unset min_edge to the AMORTIZED breakeven: a position held H
    # bars pays one round trip, so the per-bar hurdle is cost/H. Estimated from
    # the raw predictions so the threshold does not depend on itself.
    if kelly.min_edge is None:
        holding = mean_holding_bars(predictions)
        kelly = replace(kelly, min_edge=amortized_breakeven_edge(cost_model, holding))
        log.info(
            "min_edge unset; holding %.1f bars -> amortized hurdle %.6f "
            "(round-trip breakeven %.6f)",
            holding, kelly.min_edge, breakeven_edge(cost_model),
        )

    positions = size_positions(predictions, vol.reindex(predictions.index), kelly)

    # Session filter: do not INITIATE or ADJUST during structurally expensive
    # hours — freeze the existing position instead.
    #
    # Closing out at a blocked hour was the obvious first implementation and it
    # is wrong: spread is paid on TRADES, not on exposure, so holding through
    # the rollover is free while flattening and re-entering costs a full round
    # trip every day. Measured, that mistake tripled turnover (4h mean-reversion
    # 162 -> 552/yr) and turned a +0.380 net Sharpe into +0.086. Freezing gets
    # the intended benefit (never trading at a 5x spread) at no cost.
    if allowed_hours is not None:
        blocked = ~pd.Series(positions.index.hour, index=positions.index).isin(allowed_hours)
        n_blocked = int(blocked.sum())
        if n_blocked:
            log.info("session filter froze %d/%d bars (%.1f%%)",
                     n_blocked, len(positions), 100 * n_blocked / len(positions))
        # NaN out blocked bars then forward-fill: the position simply persists.
        positions = positions.where(~blocked).ffill().fillna(0.0)

    # The shift that encodes the execution convention documented above.
    gross = (positions.shift(1) * market_returns.reindex(positions.index)).dropna()
    returns = apply_costs(gross, positions, cost_model, carry_annual=carry_annual)

    net = returns["net"].dropna()
    lo, hi, _ = bootstrap_ci(net, bars_per_year, block_size=min(96, max(2, len(net) // 10)))

    return BacktestResult(
        returns=returns,
        positions=positions,
        predictions=predictions,
        fold_metrics=pd.DataFrame(fold_rows).set_index("fold"),
        gross=summarize(returns["gross"], bars_per_year, positions),
        net=summarize(net, bars_per_year, positions),
        net_sharpe_ci=(lo, hi),
        agreement=(pd.concat(all_agreement).sort_index() if all_agreement else None),
    )


def _fold_metrics(
    split: Split,
    preds: pd.Series,
    market_returns: pd.Series,
    vol: pd.Series,
    cost_model: CostModel,
    kelly: KellyConfig,
    bars_per_year: int,
) -> dict:
    """Per-fold net performance, so we can report across-fold variance."""
    pos = size_positions(preds, vol.reindex(preds.index), kelly)
    gross = (pos.shift(1) * market_returns.reindex(pos.index)).dropna()

    if gross.empty or gross.std() == 0:
        return {"fold": split.fold, "sharpe": np.nan, "net_return": np.nan, "hit_rate": np.nan}

    net = apply_costs(gross, pos, cost_model)["net"].dropna()
    stats = summarize(net, bars_per_year, pos)
    return {
        "fold": split.fold,
        "sharpe": stats.sharpe,
        "net_return": stats.total_return,
        "hit_rate": stats.hit_rate,
        "n_test": len(split.test_idx),
    }
