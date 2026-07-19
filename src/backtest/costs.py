"""
costs.py

Transaction cost model. CLAUDE.md s4.3: no backtest reports a gross figure
without a net one beside it, so every equity curve in this project routes
through `apply_costs`.

Three components:
  spread    — half-spread paid on entry and again on exit, i.e. charged in
              proportion to how much the position CHANGES, not to its size.
  slippage  — extra adverse fill beyond the quoted spread, also per unit traded.
  swap      — overnight financing, charged per bar held, on absolute exposure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    """
    All costs in price-fraction terms (i.e. 0.0001 == 1bp of notional).

    spread_pips / slippage_pips are given in pips and converted using `pip`,
    which keeps config human-readable for JPY pairs (pip = 0.01) too.
    """

    pip: float = 0.0001
    spread_pips: float = 0.6
    slippage_pips: float = 0.2
    # Annualized financing cost on held exposure; converted per-bar below.
    swap_annual_rate: float = 0.01
    bars_per_year: int = 24_800  # 15min default
    # Optional measured spread by UTC hour. When present it overrides the flat
    # `spread_pips` for per-bar costing. EURUSD is ~0.30 pips for most of the
    # day but 1.50 at the 21:00 rollover — a 5x swing that a flat constant
    # cannot represent. See src/ingest/measure_spread.py.
    spread_profile: Mapping[int, float] | None = field(default=None, compare=False)

    @classmethod
    def from_measured_profile(
        cls, path: str | Path, pip: float = 1e-4, **kwargs
    ) -> "CostModel":
        """Build a cost model from a profile written by measure_spread.py."""
        profile = {int(k): float(v) for k, v in json.loads(Path(path).read_text()).items()}
        median = float(np.median(list(profile.values())))
        return cls(pip=pip, spread_pips=median, spread_profile=profile, **kwargs)

    @property
    def cost_per_unit_traded(self) -> float:
        """
        Cost of changing position by 1.0 units of notional, using the flat
        spread. This is the scalar used for breakeven arithmetic; per-bar
        costing goes through `cost_per_unit_series` so it can vary by hour.
        """
        # Crossing the spread once costs half-spread relative to mid, plus slippage.
        return (self.spread_pips / 2 + self.slippage_pips) * self.pip / self._ref_price()

    def cost_per_unit_series(self, index: pd.DatetimeIndex) -> pd.Series:
        """Per-bar trading cost, hour-aware when a measured profile is set."""
        if self.spread_profile is None:
            return pd.Series(self.cost_per_unit_traded, index=index)

        spreads = pd.Series(index.hour, index=index).map(self.spread_profile)
        spreads = spreads.fillna(self.spread_pips)
        return (spreads / 2 + self.slippage_pips) * self.pip / self._ref_price()

    def expensive_hours(self, max_multiple: float = 1.5) -> list[int]:
        """UTC hours whose spread exceeds `max_multiple` x the cheapest hour."""
        if self.spread_profile is None:
            return []
        floor = min(self.spread_profile.values())
        return sorted(h for h, v in self.spread_profile.items() if v > floor * max_multiple)

    def _ref_price(self) -> float:
        # Costs are expressed as a fraction of price. For a 1.0-ish major this
        # is ~1; overridden via `for_instrument` when a real price is known.
        return 1.0

    @property
    def swap_per_bar(self) -> float:
        return self.swap_annual_rate / self.bars_per_year


def turnover(positions: pd.Series) -> pd.Series:
    """
    Absolute change in position per bar. The first bar's turnover is the full
    initial position — entering from flat is a real trade and must be charged.
    """
    return positions.diff().abs().fillna(positions.abs())


def apply_costs(
    gross_returns: pd.Series,
    positions: pd.Series,
    model: CostModel,
    carry_annual: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Convert a gross strategy return series into gross/net alongside the cost
    breakdown.

    `gross_returns` is expected to already be position-weighted (i.e.
    position[t-1] * market_return[t]) — see backtest.engine.run_backtest.

    `carry_annual` is the SIGNED annualized rate differential (base rate minus
    quote rate, in percent) aligned to the bars. When supplied it replaces the
    flat `swap_annual_rate` with real financing:

        holding EURUSD long  = long EUR, short USD -> earn EUR, pay USD
        so carry P&L per bar = position * (r_EUR - r_USD) / bars_per_year

    This matters: a carry strategy's entire thesis is that the differential is
    INCOME, not a cost. Charging a flat symmetric swap makes carry untestable
    by construction, because the thing you are trying to harvest never enters
    the P&L. Note the sign — with EUR rates below USD rates, being long EURUSD
    bleeds and being short earns.

    Returns columns: gross, spread_cost, swap_cost, carry, net.
    """
    positions = positions.reindex(gross_returns.index).fillna(0.0)
    prev = positions.shift(1).fillna(0.0)  # the position actually held over the bar

    spread_cost = turnover(positions) * model.cost_per_unit_series(gross_returns.index)

    if carry_annual is None:
        swap_cost = positions.abs() * model.swap_per_bar
        carry = pd.Series(0.0, index=gross_returns.index)
    else:
        # Real financing: signed, and earned on the position held over the bar.
        swap_cost = pd.Series(0.0, index=gross_returns.index)
        rate = carry_annual.reindex(gross_returns.index).ffill().fillna(0.0) / 100.0
        carry = prev * rate / model.bars_per_year

    net = gross_returns - spread_cost - swap_cost + carry
    return pd.DataFrame(
        {
            "gross": gross_returns,
            "spread_cost": spread_cost,
            "swap_cost": swap_cost,
            "carry": carry,
            "net": net,
        }
    )


def breakeven_edge(model: CostModel) -> float:
    """
    Per-TRADE return a strategy must earn just to cover costs on a full round
    trip. Useful as a fast sanity check: if the predicted edge is below this,
    the hypothesis is dead before backtesting (CLAUDE.md s4.4).

    NOTE: this is per trade, not per bar. Do not use it directly as a per-bar
    sizing filter — see `amortized_breakeven_edge`.
    """
    return 2 * model.cost_per_unit_traded


def mean_holding_bars(predictions: pd.Series) -> float:
    """
    Average number of consecutive bars the signal keeps the same sign.

    Estimated from the RAW predictions, before any sizing filter is applied,
    which keeps it independent of the threshold it is about to be used to
    compute. Estimating it from realized positions instead would be circular:
    the filter changes the holding period, which changes the filter.
    """
    sign = np.sign(predictions.dropna())
    sign = sign[sign != 0]
    if len(sign) < 2:
        return 1.0
    # Number of sign runs = number of flips + 1.
    n_runs = int((sign.diff() != 0).sum())
    return max(1.0, len(sign) / max(1, n_runs))


def amortized_breakeven_edge(
    model: CostModel, holding_bars: float, safety_factor: float = 1.0
) -> float:
    """
    The correct PER-BAR sizing hurdle.

    A position held for H bars pays one round trip, not H of them, so a per-bar
    prediction only has to clear cost/H — not the full round-trip cost. Using
    the unamortized figure as a per-bar filter is too strict by exactly the
    holding period.

    This was a real bug, not a theoretical nicety: with H = 7.1 on daily
    EURUSD momentum, the unamortized filter admitted 72% of bars while the
    correct one admits 96%. In the 2023-2026 holdout the over-strict version
    zeroed the strategy almost entirely (in-market 0.89% of bars), turning a
    pre-registered test into a non-event.

    `safety_factor` > 1 tightens the hurdle if you want margin for the holding
    period being shorter out-of-sample than it was in-sample.
    """
    if holding_bars <= 0:
        raise ValueError("holding_bars must be positive")
    return safety_factor * breakeven_edge(model) / holding_bars
