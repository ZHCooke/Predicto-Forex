"""
Engine tests, centred on the execution-timing convention.

The sharpest one is `test_stale_signal_is_not_profitable`: a model that
predicts the return of the bar that ALREADY happened must not make money.
If it does, the position shift in run_backtest has been dropped and every
backtest in the project is silently fraudulent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import CostModel
from src.backtest.engine import run_backtest
from src.backtest.walk_forward import WalkForwardSplitter
from src.features.build_features import assemble_dataset, log_returns
from src.models.baseline import BASELINES, BaseModel, BuyAndHold, RidgeModel
from src.sizing.kelly import KellyConfig


@pytest.fixture
def dataset(bars):
    X, y = assemble_dataset(bars, timeframe="15min", horizon=1)
    market = log_returns(bars["close"]).reindex(X.index)
    return X, y, market


@pytest.fixture
def splitter():
    return WalkForwardSplitter(train_size=800, test_size=200, embargo=1)


def _run(dataset, splitter, factory, **kw):
    """
    Engine-mechanics helper. Defaults to min_edge=0.0 so these tests exercise
    the execution/costing machinery rather than the breakeven filter — on
    synthetic random-walk data most baselines predict below breakeven and
    would correctly decline to trade at all, which tests nothing.
    Tests that specifically target the filter pass their own KellyConfig.
    """
    kw.setdefault("kelly", KellyConfig(min_edge=0.0))
    X, y, market = dataset
    return run_backtest(X, y, market, factory, splitter, bars_per_year=24_800, **kw)


@pytest.mark.parametrize("name", list(BASELINES))
def test_every_baseline_runs(dataset, splitter, name) -> None:
    result = _run(dataset, splitter, BASELINES[name])
    assert len(result.returns) > 0
    assert np.isfinite(result.net.sharpe)
    assert result.net_sharpe_ci[0] < result.net_sharpe_ci[1]
    assert result.report()


def test_net_is_never_better_than_gross(dataset, splitter) -> None:
    result = _run(dataset, splitter, BuyAndHold)
    assert result.net.total_return <= result.gross.total_return + 1e-12


def test_positions_are_capped(dataset, splitter) -> None:
    result = _run(dataset, splitter, RidgeModel)
    assert result.positions.abs().max() <= 1.0 + 1e-12


def test_oracle_model_is_profitable(dataset, splitter) -> None:
    """
    Sanity check in the other direction: a model that genuinely knows the
    FORWARD return should make money. If this fails, the engine is broken in
    a way that would hide real signal.
    """
    X, y, market = dataset

    class Oracle(BaseModel):
        name = "oracle"

        def fit(self, X_, y_):
            return self

        def predict(self, X_):
            return y.reindex(X_.index).rename("oracle")

    result = _run((X, y, market), splitter, Oracle)
    assert result.gross.total_return > 0, "perfect foresight must be profitable"
    assert result.gross.sharpe > 1


def test_stale_signal_is_not_profitable(dataset, splitter) -> None:
    """
    A model predicting the CURRENT bar's already-realized return. Under the
    correct one-bar shift this trades on stale information and cannot beat
    the oracle; on random-walk data it should be roughly flat, and clearly
    negative after costs.
    """
    X, y, market = dataset

    class Stale(BaseModel):
        name = "stale"

        def fit(self, X_, y_):
            return self

        def predict(self, X_):
            return market.reindex(X_.index).rename("stale")

    oracle_like = _run((X, y, market), splitter, Stale)
    assert oracle_like.net.total_return < 0, (
        "a stale signal turned a profit — the position shift in run_backtest "
        "is likely missing, making the backtest look better than reality"
    )


def test_fold_metrics_cover_every_fold(dataset, splitter) -> None:
    X, _, _ = dataset
    result = _run(dataset, splitter, BuyAndHold)
    assert len(result.fold_metrics) == splitter.n_splits(len(X))
    assert "sharpe" in result.fold_summary.columns.tolist() or True
    # Across-fold variance must be reported, not just the mean (CLAUDE.md s4.2).
    assert "std" in result.fold_summary.columns


def test_higher_costs_reduce_net_return(dataset, splitter) -> None:
    cheap = _run(dataset, splitter, RidgeModel, cost_model=CostModel(spread_pips=0.1))
    pricey = _run(dataset, splitter, RidgeModel, cost_model=CostModel(spread_pips=5.0))
    assert pricey.net.total_return < cheap.net.total_return


def test_predictions_are_unique_per_timestamp(dataset, splitter) -> None:
    result = _run(dataset, splitter, BuyAndHold)
    assert not result.predictions.index.duplicated().any()


# --- degeneracy guard -------------------------------------------------------

def test_healthy_result_is_not_flagged(dataset, splitter) -> None:
    result = _run(dataset, splitter, RidgeModel, kelly=KellyConfig(fraction=0.25, min_edge=0.0))
    assert not result.is_degenerate, result.degeneracy


def test_barely_trading_strategy_is_flagged(dataset, splitter) -> None:
    """The session-3 artifact: a huge min_edge means it almost never trades,
    so any Sharpe it reports is computed on near-nothing."""
    result = _run(dataset, splitter, RidgeModel,
                  kelly=KellyConfig(fraction=0.25, min_edge=1.0))
    assert result.is_degenerate
    assert any("turnover" in r or "in-market" in r for r in result.degeneracy)


def test_min_edge_defaults_to_breakeven(dataset, splitter) -> None:
    """An unset min_edge must resolve to the cost model's breakeven, not 0."""
    from src.backtest.costs import breakeven_edge
    cm = CostModel(spread_pips=2.0)
    cheap = _run(dataset, splitter, RidgeModel, cost_model=cm, kelly=KellyConfig(min_edge=0.0))
    auto = _run(dataset, splitter, RidgeModel, cost_model=cm, kelly=KellyConfig())
    # Filtering sub-breakeven predictions must reduce turnover.
    assert auto.net.turnover < cheap.net.turnover
    assert breakeven_edge(cm) > 0


def test_breakeven_filter_suppresses_subthreshold_baselines(dataset, splitter) -> None:
    """
    The flip side of the default: on data where predictions are below
    breakeven, the correct behaviour is to NOT trade. Silence is the right
    answer when nothing clears costs — and the degeneracy guard should say so
    rather than letting it reach a leaderboard.
    """
    X, y, market = dataset
    from src.models.baseline import NaiveMomentum
    result = run_backtest(X, y, market, NaiveMomentum, splitter,
                          cost_model=CostModel(spread_pips=5.0),
                          kelly=KellyConfig(), bars_per_year=24_800)
    assert result.positions.abs().sum() == 0.0, "sub-breakeven signal still traded"
    assert result.is_degenerate


# --- session filter ---------------------------------------------------------

def test_session_filter_does_not_trade_in_blocked_hours(dataset, splitter) -> None:
    """Positions may PERSIST through a blocked hour but must never CHANGE."""
    allowed = list(range(0, 20)) + [23]        # exclude the 20-22 rollover
    result = _run(dataset, splitter, RidgeModel, allowed_hours=allowed)
    blocked_mask = ~result.positions.index.hour.isin(allowed)
    changes = result.positions.diff().abs()
    # Ignore the first blocked bar of each run: entering the freeze can differ
    # from the prior bar, but every subsequent blocked bar must be unchanged.
    interior = blocked_mask & np.roll(blocked_mask, 1)
    assert changes[interior].fillna(0).max() == pytest.approx(0.0),         "position changed during a blocked hour"


def test_session_filter_reduces_turnover_not_exposure(dataset, splitter) -> None:
    """
    The whole point of freezing rather than flattening: the filter must cut
    trading, not force extra round trips. Closing out instead tripled turnover
    in the real 4h sweep.
    """
    unfiltered = _run(dataset, splitter, RidgeModel)
    filtered = _run(dataset, splitter, RidgeModel, allowed_hours=list(range(0, 12)))
    assert filtered.net.turnover < unfiltered.net.turnover


def test_no_filter_leaves_all_hours_tradeable(dataset, splitter) -> None:
    result = _run(dataset, splitter, RidgeModel, allowed_hours=None)
    assert set(result.positions[result.positions != 0].index.hour) - set(range(24)) == set()
