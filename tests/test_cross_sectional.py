"""
The cross-sectional book has algebraic properties that must hold exactly:
dollar-neutrality, correct replication of the intended currency exposure, and
netting of redundant legs. Those are testable independently of whether the
strategy makes money, which is what makes them worth pinning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.currency_strength import PAIR_LEGS, currency_returns, strength_ratings
from src.models.cross_sectional import (
    build_cross_sectional_positions,
    cross_sectional_scores,
    currency_positions_from_scores,
    currency_to_pair_positions,
    portfolio_diagnostics,
    portfolio_returns,
    portfolio_turnover,
)

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD"]


@pytest.fixture
def pair_returns():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=400, freq="D", tz="UTC")
    return pd.DataFrame(rng.normal(0, 0.005, (len(idx), len(PAIRS))), index=idx, columns=PAIRS)


def test_scores_are_dollar_neutral(pair_returns) -> None:
    """Demeaned scores must sum to zero — for every long there is a short."""
    ccy = currency_returns(pair_returns)
    scores = cross_sectional_scores(strength_ratings(ccy, 5.0))
    assert scores.dropna().sum(axis=1).abs().max() < 1e-12


def test_gross_exposure_is_constant(pair_returns) -> None:
    ccy = currency_returns(pair_returns)
    scores = cross_sectional_scores(strength_ratings(ccy, 5.0))
    w = currency_positions_from_scores(scores, gross_exposure=1.0).dropna()
    gross = w.abs().sum(axis=1)
    assert gross[gross > 0].std() < 1e-9


def test_reversion_fades_strength(pair_returns) -> None:
    """The strongest currency must be SHORTED under reversion."""
    ccy = currency_returns(pair_returns)
    scores = cross_sectional_scores(strength_ratings(ccy, 5.0)).dropna()
    w = currency_positions_from_scores(scores, reversion=True)

    row = scores.iloc[-1]
    strongest, weakest = row.idxmax(), row.idxmin()
    assert w.iloc[-1][strongest] < 0
    assert w.iloc[-1][weakest] > 0


def test_momentum_mode_does_the_opposite(pair_returns) -> None:
    ccy = currency_returns(pair_returns)
    scores = cross_sectional_scores(strength_ratings(ccy, 5.0)).dropna()
    rev = currency_positions_from_scores(scores, reversion=True)
    mom = currency_positions_from_scores(scores, reversion=False)
    pd.testing.assert_frame_equal(rev, -mom)


def test_pair_positions_replicate_currency_exposure(pair_returns) -> None:
    """
    The defining property: converting currency weights to pair trades and back
    must reproduce the intended exposure. If this fails the book is not holding
    what the model asked for.
    """
    pos, ccy_w = build_cross_sectional_positions(pair_returns)
    diag = portfolio_diagnostics(pos.dropna(), ccy_w.dropna())
    assert diag["max_replication_residual"] < 1e-9


def test_book_is_net_neutral(pair_returns) -> None:
    """
    Every long is offset by a short, so total exposure nets to zero. This is
    "dollar-neutral" in the EQUITY sense (net notional cancels).
    """
    pos, ccy_w = build_cross_sectional_positions(pair_returns)
    diag = portfolio_diagnostics(pos.dropna(), ccy_w.dropna())
    assert diag["max_abs_net_exposure"] < 1e-9


def test_usd_exposure_is_a_real_view_not_zero(pair_returns) -> None:
    """
    The counterpart, and the trap this originally fell into: net neutrality
    does NOT imply zero USD exposure. USD is scored like any other currency, so
    if the dollar has run up, reversion shorts it — deliberately. A test
    asserting zero USD exposure would be asserting the strategy has no view.
    """
    pos, ccy_w = build_cross_sectional_positions(pair_returns)
    diag = portfolio_diagnostics(pos.dropna(), ccy_w.dropna())
    assert diag["mean_abs_usd_exposure"] > 0, "USD must be tradeable like any other leg"
    # But it must not dominate the book either.
    assert diag["mean_abs_usd_exposure"] < diag["mean_gross_ccy_exposure"]


def test_netting_reduces_traded_exposure(pair_returns) -> None:
    """
    The cost argument: expressing the book through minimum-norm pair trades
    must require LESS gross trading than the currency exposure implies, because
    redundant dollar legs cancel. Without netting, N pairs cost ~N spreads.
    """
    pos, ccy_w = build_cross_sectional_positions(pair_returns)
    diag = portfolio_diagnostics(pos.dropna(), ccy_w.dropna())
    assert diag["netting_ratio"] < 1.0, "netting did not reduce traded exposure"


def test_a_pure_dollar_view_needs_no_cross_trades() -> None:
    """
    Sanity case with an obvious answer: wanting only USD exposure should be
    expressible through the USD pairs, not through unrelated crosses.
    """
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CHF", "CAD"]
    w = pd.DataFrame(0.0, index=idx, columns=currencies)
    w["USD"] = 0.7
    for c in ["EUR", "GBP", "JPY", "AUD", "NZD", "CHF", "CAD"]:
        w[c] = -0.1

    pos = currency_to_pair_positions(w, PAIRS)
    assert np.isfinite(pos.to_numpy()).all()
    diag = portfolio_diagnostics(pos, w)
    assert diag["max_replication_residual"] < 1e-9


def test_portfolio_returns_apply_the_execution_shift(pair_returns) -> None:
    """Positions from bar t must earn bar t+1's move, never bar t's."""
    pos = pd.DataFrame(0.0, index=pair_returns.index, columns=PAIRS)
    pos.iloc[5, 0] = 1.0  # long EURUSD on bar 5 only

    r = pair_returns.copy()
    r.iloc[:] = 0.0
    r.iloc[6, 0] = 0.01  # market moves on bar 6

    out = portfolio_returns(pos, r)
    assert out.loc[r.index[6]] == pytest.approx(0.01)
    assert out.loc[r.index[5]] == pytest.approx(0.0)


def test_stale_signal_earns_nothing(pair_returns) -> None:
    """A position taken AFTER the move must not capture it."""
    pos = pd.DataFrame(0.0, index=pair_returns.index, columns=PAIRS)
    r = pair_returns.copy()
    r.iloc[:] = 0.0
    r.iloc[6, 0] = 0.01
    pos.iloc[6, 0] = 1.0  # entered on the bar that already moved

    out = portfolio_returns(pos, r)
    assert out.abs().max() == pytest.approx(0.0)


def test_turnover_charges_the_initial_entry(pair_returns) -> None:
    pos = pd.DataFrame(0.5, index=pair_returns.index, columns=PAIRS)
    t = portfolio_turnover(pos)
    assert t.iloc[0] == pytest.approx(0.5 * len(PAIRS))
    assert t.iloc[1:].sum() == pytest.approx(0.0)


def test_positions_are_causal(pair_returns) -> None:
    pos, _ = build_cross_sectional_positions(pair_returns)
    corrupted = pair_returns.copy()
    corrupted.iloc[200:] *= 30
    pos2, _ = build_cross_sectional_positions(corrupted)
    pd.testing.assert_frame_equal(pos.iloc[:200], pos2.iloc[:200])
