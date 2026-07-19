"""Costs must always reduce returns; Kelly sizing must always stay capped."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import CostModel, apply_costs, breakeven_edge, turnover
from src.backtest.metrics import bootstrap_ci, max_drawdown, summarize
from src.sizing.kelly import KellyConfig, drawdown_throttle, size_positions


@pytest.fixture
def idx():
    return pd.date_range("2024-01-01", periods=500, freq="15min", tz="UTC")


def test_costs_never_increase_returns(idx) -> None:
    rng = np.random.default_rng(0)
    gross = pd.Series(rng.normal(0, 0.001, len(idx)), index=idx)
    positions = pd.Series(rng.choice([-1.0, 0.0, 1.0], len(idx)), index=idx)

    out = apply_costs(gross, positions, CostModel())
    assert (out["net"] <= out["gross"] + 1e-12).all()
    assert (out["spread_cost"] >= 0).all()
    assert (out["swap_cost"] >= 0).all()


def test_holding_a_constant_position_incurs_no_spread_after_entry(idx) -> None:
    gross = pd.Series(0.0, index=idx)
    positions = pd.Series(1.0, index=idx)

    out = apply_costs(gross, positions, CostModel())
    # Entry from flat is charged once; no further turnover afterwards.
    assert out["spread_cost"].iloc[0] > 0
    assert out["spread_cost"].iloc[1:].sum() == pytest.approx(0.0)


def test_initial_entry_is_charged(idx) -> None:
    positions = pd.Series([1.0] * len(idx), index=idx)
    assert turnover(positions).iloc[0] == pytest.approx(1.0), "entry from flat must cost"


def test_flipping_position_costs_double(idx) -> None:
    positions = pd.Series([1.0, -1.0] * (len(idx) // 2), index=idx)
    t = turnover(positions)
    assert t.iloc[1] == pytest.approx(2.0)


def test_breakeven_edge_is_positive() -> None:
    assert breakeven_edge(CostModel()) > 0


def test_kelly_respects_max_position(idx) -> None:
    # Absurd edge with tiny vol — must still clamp.
    preds = pd.Series(1.0, index=idx)
    vol = pd.Series(1e-9, index=idx)

    pos = size_positions(preds, vol, KellyConfig(fraction=0.5, max_position=0.75))
    assert pos.abs().max() <= 0.75 + 1e-12


def test_kelly_sign_follows_prediction(idx) -> None:
    preds = pd.Series(np.linspace(-0.01, 0.01, len(idx)), index=idx)
    vol = pd.Series(0.001, index=idx)
    pos = size_positions(preds, vol, KellyConfig())

    nonzero = preds != 0
    assert (np.sign(pos[nonzero]) == np.sign(preds[nonzero])).all()


def test_min_edge_zeroes_noise(idx) -> None:
    preds = pd.Series(1e-9, index=idx)
    vol = pd.Series(0.001, index=idx)
    pos = size_positions(preds, vol, KellyConfig(min_edge=1e-4))
    assert (pos == 0).all()


def test_kelly_never_returns_nan(idx) -> None:
    preds = pd.Series([np.nan] * len(idx), index=idx)
    vol = pd.Series(0.0, index=idx)
    assert not size_positions(preds, vol, KellyConfig()).isna().any()


def test_fraction_must_be_valid() -> None:
    with pytest.raises(ValueError):
        KellyConfig(fraction=0.0)
    with pytest.raises(ValueError):
        KellyConfig(fraction=1.5)


def test_drawdown_throttle_is_causal(idx) -> None:
    # A catastrophic loss at the very last bar cannot change earlier sizing.
    returns = pd.Series(0.0, index=idx)
    positions = pd.Series(1.0, index=idx)
    base = drawdown_throttle(positions, returns)

    returns_shocked = returns.copy()
    returns_shocked.iloc[-1] = -0.9
    shocked = drawdown_throttle(positions, returns_shocked)

    pd.testing.assert_series_equal(base.iloc[:-1], shocked.iloc[:-1])


def test_drawdown_throttle_engages(idx) -> None:
    returns = pd.Series(-0.01, index=idx)  # steady bleed, breaches 20% quickly
    positions = pd.Series(1.0, index=idx)
    throttled = drawdown_throttle(positions, returns, max_drawdown=0.20, throttle=0.5)
    assert throttled.iloc[-1] == pytest.approx(0.5)


def test_max_drawdown_is_non_positive(idx) -> None:
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    assert max_drawdown(returns) <= 0


def test_summary_and_ci_are_finite(idx) -> None:
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0.0001, 0.001, len(idx)), index=idx)

    stats = summarize(returns, bars_per_year=24_800)
    assert np.isfinite(stats.sharpe)
    assert 0 <= stats.hit_rate <= 1

    lo, hi, dist = bootstrap_ci(returns, 24_800, n_resamples=200, block_size=48)
    assert lo < hi
    assert len(dist) == 200


# --- real carry financing (replaces the flat swap when supplied) -------------

def test_carry_is_income_not_cost_when_differential_is_positive(idx) -> None:
    """Long a pair whose base rate exceeds its quote rate must EARN carry."""
    gross = pd.Series(0.0, index=idx)
    positions = pd.Series(1.0, index=idx)
    carry_annual = pd.Series(3.0, index=idx)  # base yields 3% more

    out = apply_costs(gross, positions, CostModel(), carry_annual=carry_annual)
    assert out["carry"].iloc[1:].gt(0).all(), "positive differential must pay the long"


def test_carry_sign_flips_with_position(idx) -> None:
    gross = pd.Series(0.0, index=idx)
    carry_annual = pd.Series(-1.45, index=idx)  # EUR below USD, the real case

    long = apply_costs(gross, pd.Series(1.0, index=idx), CostModel(),
                       carry_annual=carry_annual)["carry"]
    short = apply_costs(gross, pd.Series(-1.0, index=idx), CostModel(),
                        carry_annual=carry_annual)["carry"]

    # Negative differential: long bleeds, short earns.
    assert long.iloc[1:].lt(0).all()
    assert short.iloc[1:].gt(0).all()
    assert long.iloc[1:].add(short.iloc[1:]).abs().max() < 1e-15


def test_carry_magnitude_annualizes_correctly(idx) -> None:
    bpy = 260
    gross = pd.Series(0.0, index=idx)
    out = apply_costs(gross, pd.Series(1.0, index=idx),
                      CostModel(bars_per_year=bpy),
                      carry_annual=pd.Series(2.6, index=idx))
    assert out["carry"].iloc[1] == pytest.approx(0.026 / bpy)


def test_carry_uses_the_position_actually_held(idx) -> None:
    """Carry accrues on the position held OVER the bar, i.e. position[t-1]."""
    gross = pd.Series(0.0, index=idx)
    pos = pd.Series(0.0, index=idx)
    pos.iloc[5:] = 1.0
    out = apply_costs(gross, pos, CostModel(), carry_annual=pd.Series(3.0, index=idx))

    assert out["carry"].iloc[5] == 0.0, "carry charged before the position existed"
    assert out["carry"].iloc[6] > 0.0


def test_flat_swap_still_applies_when_no_carry_given(idx) -> None:
    gross = pd.Series(0.0, index=idx)
    out = apply_costs(gross, pd.Series(1.0, index=idx), CostModel())
    assert (out["swap_cost"] > 0).all()
    assert (out["carry"] == 0).all()
