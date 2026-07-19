"""
Two cost-side corrections from session 6:

1. The per-bar sizing hurdle must be the round-trip cost AMORTIZED over the
   holding period. Using the full round-trip cost per bar is too strict by
   exactly the holding period, and it is what silently zeroed the h001 holdout
   test (in-market 0.89% of bars).
2. Spread varies by UTC hour (EURUSD: 0.30 most of the day, 1.50 at the 21:00
   rollover). A flat constant misprices both the liquid window and the
   rollover.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import (
    CostModel,
    amortized_breakeven_edge,
    apply_costs,
    breakeven_edge,
    mean_holding_bars,
)

# Roughly the measured EURUSD profile: flat by day, spiking at rollover.
PROFILE = {h: 0.30 for h in range(24)}
PROFILE.update({0: 0.40, 1: 0.40, 2: 0.40, 3: 0.40, 4: 0.40, 5: 0.40,
                20: 1.20, 21: 1.50, 22: 0.60, 23: 0.40})


@pytest.fixture
def hourly_index():
    return pd.date_range("2024-01-01", periods=24 * 30, freq="1h", tz="UTC")


# --- holding-period estimation ---------------------------------------------

def test_alternating_signal_holds_one_bar() -> None:
    s = pd.Series([1.0, -1.0] * 50)
    assert mean_holding_bars(s) == pytest.approx(1.0, abs=0.1)


def test_persistent_signal_holds_longer() -> None:
    s = pd.Series([1.0] * 10 + [-1.0] * 10 + [1.0] * 10)
    assert mean_holding_bars(s) == pytest.approx(10.0, abs=1.0)


def test_constant_signal_never_flips() -> None:
    s = pd.Series([1.0] * 100)
    assert mean_holding_bars(s) >= 50


def test_holding_is_at_least_one() -> None:
    assert mean_holding_bars(pd.Series([1.0])) >= 1.0
    assert mean_holding_bars(pd.Series([], dtype=float)) >= 1.0


def test_holding_estimate_ignores_magnitude() -> None:
    """Only the sign pattern matters, not how large the predictions are."""
    small = pd.Series([1e-9, 1e-9, -1e-9, -1e-9] * 20)
    large = pd.Series([5.0, 5.0, -5.0, -5.0] * 20)
    assert mean_holding_bars(small) == pytest.approx(mean_holding_bars(large))


# --- amortization -----------------------------------------------------------

def test_amortized_hurdle_is_lower_than_round_trip() -> None:
    cm = CostModel()
    assert amortized_breakeven_edge(cm, holding_bars=7.0) < breakeven_edge(cm)


def test_amortized_hurdle_scales_inversely_with_holding() -> None:
    cm = CostModel()
    assert amortized_breakeven_edge(cm, 10.0) == pytest.approx(
        amortized_breakeven_edge(cm, 5.0) / 2
    )


def test_holding_one_bar_recovers_round_trip() -> None:
    """With a one-bar holding period the amortized hurdle IS the round trip."""
    cm = CostModel()
    assert amortized_breakeven_edge(cm, 1.0) == pytest.approx(breakeven_edge(cm))


def test_safety_factor_tightens() -> None:
    cm = CostModel()
    base = amortized_breakeven_edge(cm, 7.0)
    assert amortized_breakeven_edge(cm, 7.0, safety_factor=2.0) == pytest.approx(2 * base)


def test_nonpositive_holding_rejected() -> None:
    with pytest.raises(ValueError):
        amortized_breakeven_edge(CostModel(), 0.0)


# --- hour-aware spread ------------------------------------------------------

def test_rollover_hour_costs_more_than_midday(hourly_index) -> None:
    cm = CostModel(spread_profile=PROFILE)
    costs = cm.cost_per_unit_series(hourly_index)
    at_21 = costs[hourly_index.hour == 21].iloc[0]
    at_14 = costs[hourly_index.hour == 14].iloc[0]
    assert at_21 > at_14


def test_flat_model_is_hour_invariant(hourly_index) -> None:
    costs = CostModel(spread_pips=0.6).cost_per_unit_series(hourly_index)
    assert costs.nunique() == 1


def test_expensive_hours_identifies_rollover() -> None:
    cm = CostModel(spread_profile=PROFILE)
    assert set(cm.expensive_hours(max_multiple=1.5)) == {20, 21, 22}


def test_trading_at_rollover_costs_more_in_the_backtest(hourly_index) -> None:
    """The profile must actually reach the P&L, not just the helper."""
    gross = pd.Series(0.0, index=hourly_index)
    flip = pd.Series(np.where(np.arange(len(hourly_index)) % 2 == 0, 1.0, -1.0),
                     index=hourly_index)

    flat = apply_costs(gross, flip, CostModel(spread_pips=0.30))["spread_cost"].sum()
    profiled = apply_costs(gross, flip, CostModel(spread_profile=PROFILE))["spread_cost"].sum()
    assert profiled > flat, "hourly profile did not reach spread_cost"


def test_from_measured_profile_roundtrips(tmp_path) -> None:
    path = tmp_path / "prof.json"
    path.write_text(json.dumps({str(k): v for k, v in PROFILE.items()}))
    cm = CostModel.from_measured_profile(path, pip=1e-4)
    assert cm.spread_profile[21] == 1.50
    assert set(cm.expensive_hours()) == {20, 21, 22}


def test_unknown_hour_falls_back_to_flat_spread(hourly_index) -> None:
    partial = {12: 0.30}
    cm = CostModel(spread_pips=0.9, spread_profile=partial)
    costs = cm.cost_per_unit_series(hourly_index)
    assert costs[hourly_index.hour == 12].iloc[0] < costs[hourly_index.hour == 3].iloc[0]
    assert costs.notna().all()
