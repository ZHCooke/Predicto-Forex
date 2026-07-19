"""
The currency decomposition must satisfy an exact algebraic identity, which
makes it unusually testable: if s_EUR - s_USD does not reproduce the EURUSD
return, the decomposition is simply wrong.

Also re-checks causality, since this is a new feature family and lookahead is
the failure mode that would flatter every downstream result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.currency_strength import (
    PAIR_LEGS,
    build_strength_features,
    currency_returns,
    strength_ratings,
)

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD"]


@pytest.fixture
def pair_returns():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=500, freq="D", tz="UTC")
    return pd.DataFrame(rng.normal(0, 0.005, (len(idx), len(PAIRS))), index=idx, columns=PAIRS)


def test_decomposition_reproduces_every_pair(pair_returns) -> None:
    """The defining identity: r_pair == s_base - s_quote, for all pairs."""
    ccy = currency_returns(pair_returns)
    for p in PAIRS:
        base, quote = PAIR_LEGS[p]
        implied = ccy[base] - ccy[quote]
        pd.testing.assert_series_equal(
            implied, pair_returns[p], check_names=False, atol=1e-12
        )


def test_strengths_sum_to_zero(pair_returns) -> None:
    """Identification constraint: strengths are relative to the basket."""
    ccy = currency_returns(pair_returns)
    assert ccy.sum(axis=1).abs().max() < 1e-12


def test_common_dollar_move_is_attributed_to_the_dollar(pair_returns) -> None:
    """
    The whole point of the exercise. If every non-USD currency is flat and only
    the dollar moves, the decomposition must say so — a single pair could not
    distinguish this from its own currency moving.
    """
    idx = pair_returns.index
    r = pd.DataFrame(0.0, index=idx, columns=PAIRS)
    # USD strengthens 1% against everything.
    for p in PAIRS:
        base, quote = PAIR_LEGS[p]
        r[p] = -0.01 if quote == "USD" else 0.01

    ccy = currency_returns(r)
    assert (ccy["USD"] > 0).all()
    # Every other currency should be equally, and negatively, affected.
    others = ccy.drop(columns=["USD"])
    assert (others < 0).all().all()
    assert others.std(axis=1).max() < 1e-12


def test_single_currency_move_is_isolated(pair_returns) -> None:
    """A pure EUR move must not be attributed to the dollar."""
    idx = pair_returns.index
    r = pd.DataFrame(0.0, index=idx, columns=PAIRS)
    r["EURUSD"] = 0.01  # only EUR moves, against USD

    ccy = currency_returns(r)
    # EUR is the biggest mover; USD moves far less than EUR does.
    assert (ccy["EUR"] > 0).all()
    assert (ccy["EUR"].abs() > ccy["USD"].abs()).all()


def test_missing_pair_invalidates_the_bar(pair_returns) -> None:
    """A gap must produce NaN, never a silently mis-attributed decomposition."""
    r = pair_returns.copy()
    r.iloc[10, r.columns.get_loc("USDJPY")] = np.nan
    ccy = currency_returns(r)
    assert ccy.iloc[10].isna().all()
    assert ccy.iloc[11].notna().all()


def test_unknown_currency_rejected(pair_returns) -> None:
    with pytest.raises(ValueError, match="not covered"):
        currency_returns(pair_returns, currencies=["USD", "EUR", "SEK"])


def test_ratings_are_causal(pair_returns) -> None:
    """Corrupt the future; the past must not move."""
    ccy = currency_returns(pair_returns)
    base = strength_ratings(ccy, halflife=20)

    corrupted = ccy.copy()
    corrupted.iloc[250:] += 10.0
    after = strength_ratings(corrupted, halflife=20)

    pd.testing.assert_frame_equal(base.iloc[:250], after.iloc[:250])


def test_features_are_causal(pair_returns) -> None:
    base = build_strength_features(pair_returns, "EURUSD")

    corrupted = pair_returns.copy()
    corrupted.iloc[300:] *= 50
    after = build_strength_features(corrupted, "EURUSD")

    pd.testing.assert_frame_equal(base.iloc[:300], after.iloc[:300])


def test_features_use_the_prefix_and_are_finite(pair_returns) -> None:
    X = build_strength_features(pair_returns, "EURUSD")
    assert all(c.startswith("f_") for c in X.columns)
    assert np.isfinite(X.dropna().to_numpy()).all()
    assert len(X) == len(pair_returns)


def test_shorter_halflife_reacts_faster(pair_returns) -> None:
    ccy = currency_returns(pair_returns)
    fast = strength_ratings(ccy, halflife=2)
    slow = strength_ratings(ccy, halflife=100)
    assert fast.std().mean() > slow.std().mean()


def test_works_with_a_partial_basket(pair_returns) -> None:
    """Fewer pairs must still decompose the currencies they do cover."""
    subset = pair_returns[["EURUSD", "GBPUSD", "USDJPY"]]
    ccy = currency_returns(subset)
    assert set(ccy.columns) == {"USD", "EUR", "GBP", "JPY"}
    implied = ccy["EUR"] - ccy["USD"]
    pd.testing.assert_series_equal(implied, subset["EURUSD"], check_names=False, atol=1e-12)


def test_dispersion_measures_trend_concentration(pair_returns) -> None:
    """
    Dispersion of the RATINGS is high when one currency persistently dominates
    (a sustained dollar trend drives USD far from the basket) and low when
    moves are noise that washes out of the EWMA.

    Note this is the opposite of the naive reading — dispersion is a
    trend-strength indicator, not an idiosyncrasy indicator. Documented here
    because the first version of this test asserted it backwards.
    """
    idx = pair_returns.index
    common = pd.DataFrame(0.0, index=idx, columns=PAIRS)
    for p in PAIRS:
        _, quote = PAIR_LEGS[p]
        common[p] = -0.01 if quote == "USD" else 0.01

    rng = np.random.default_rng(1)
    idio = pd.DataFrame(rng.normal(0, 0.01, (len(idx), len(PAIRS))), index=idx, columns=PAIRS)

    d_common = build_strength_features(common)["f_ccy_dispersion_20"].iloc[-1]
    d_idio = build_strength_features(idio)["f_ccy_dispersion_20"].iloc[-1]
    assert d_common > d_idio, "a sustained one-currency trend must raise dispersion"
    # And the dominant currency must be the one that actually moved.
    ccy = currency_returns(common)
    assert ccy.iloc[-1].idxmax() == "USD"
