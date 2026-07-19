"""
kelly.py

Fractional Kelly position sizing (CLAUDE.md s4.5), adapted from the sports
betting pipelines. Two differences from the discrete-odds betting case:

  1. FX returns are continuous, so the Kelly fraction is the Gaussian form
     f* = mu / sigma^2 rather than the (bp - q)/b binary-outcome form.
  2. Everything is hard-capped. Full Kelly is optimal only if mu and sigma are
     known exactly; they are estimated, and estimation error in mu is
     quadratic in its effect on growth. We size at a fraction of Kelly and
     clamp — never let the model size itself unbounded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class KellyConfig:
    """
    fraction:      multiple of full Kelly to actually bet. 0.25-0.5 is the
                   usual defensible range; 1.0 is not a serious setting.
    max_position:  hard cap on absolute position, in units of notional.
    min_edge:      predicted returns below this magnitude are treated as noise
                   and sized to zero — keeps costs from eating a null signal.
    vol_floor:     lower bound on sigma, so a quiet window can't produce a
                   near-infinite position.
    """

    fraction: float = 0.25
    max_position: float = 1.0
    # None means "resolve to breakeven_edge(cost_model)" in run_backtest.
    # Sizing on a sub-breakeven prediction is never rational — the trade cannot
    # pay for itself even if the prediction is exactly right — so 0.0 is a bad
    # default and is only kept reachable for tests that want it explicitly.
    min_edge: float | None = None
    vol_floor: float = 1e-6

    def __post_init__(self) -> None:
        if not 0 < self.fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        if self.max_position <= 0:
            raise ValueError("max_position must be positive")


def kelly_fraction(mu: float | np.ndarray, sigma: float | np.ndarray) -> float | np.ndarray:
    """
    Full Kelly for a continuous return with mean mu and stdev sigma: mu/sigma^2.
    This is the UNCAPPED value — callers should not use it directly for sizing.
    """
    sigma = np.asarray(sigma, dtype=float)
    var = np.where(sigma > 0, sigma**2, np.nan)
    return np.asarray(mu, dtype=float) / var


def size_positions(
    predicted_return: pd.Series,
    predicted_vol: pd.Series,
    config: KellyConfig | None = None,
) -> pd.Series:
    """
    Turn per-bar predicted return and volatility into capped position sizes.

    Both inputs must be known at the bar they are indexed on — this function
    does no shifting, so the caller is responsible for making sure the
    prediction for bar t was generated from data available at t.
    """
    cfg = config or KellyConfig()
    vol = predicted_vol.reindex(predicted_return.index).clip(lower=cfg.vol_floor)

    raw = pd.Series(
        kelly_fraction(predicted_return.to_numpy(), vol.to_numpy()),
        index=predicted_return.index,
    )
    sized = (raw * cfg.fraction).clip(-cfg.max_position, cfg.max_position)

    # Zero out anything inside the noise band, and any bar we couldn't size.
    # An unresolved min_edge (None) means no filter — run_backtest normally
    # resolves it to breakeven before we get here.
    min_edge = 0.0 if cfg.min_edge is None else cfg.min_edge
    sized = sized.where(predicted_return.abs() >= min_edge, 0.0)
    return sized.fillna(0.0)


def drawdown_throttle(
    positions: pd.Series,
    realized_returns: pd.Series,
    max_drawdown: float = 0.20,
    throttle: float = 0.5,
) -> pd.Series:
    """
    Scale positions down after the equity curve breaches `max_drawdown`.

    The drawdown at bar t is computed from returns strictly BEFORE t (shifted
    by one), so this is causal and safe to use inside a backtest.
    """
    eq = (1 + realized_returns.fillna(0)).cumprod()
    dd = (eq / eq.cummax() - 1).shift(1).fillna(0.0)
    scale = pd.Series(np.where(dd <= -abs(max_drawdown), throttle, 1.0), index=dd.index)
    return positions * scale.reindex(positions.index).fillna(1.0)
