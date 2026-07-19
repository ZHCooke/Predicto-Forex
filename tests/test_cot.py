"""
COT positioning: the publication lag is the whole risk.

The report snapshots Tuesday but is not published until Friday evening. Using
it from Tuesday hands the model three days of hindsight and would produce a
spectacular, entirely fake backtest. These tests pin the lag and the
normalisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ingest.fetch_cot import CONTRACT_CURRENCY, align_to_bars, build_cot_features


@pytest.fixture
def cot():
    """Weekly Tuesday snapshots for two currencies."""
    dates = pd.date_range("2020-01-07", periods=100, freq="7D", tz="UTC")
    rows = []
    for ccy, base in (("EUR", 0.1), ("JPY", -0.2)):
        rows.append(
            pd.DataFrame(
                {
                    "report_date": dates,
                    "currency": ccy,
                    "open_interest": 500_000.0,
                    "net_lev": base + np.linspace(0, 0.3, len(dates)),
                    "net_asset_mgr": base,
                    "net_dealer": -base,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def bars():
    return pd.date_range("2020-01-01", "2021-06-01", freq="1D", tz="UTC")


def test_contract_names_are_exact_not_prefixes() -> None:
    """
    Regression test for a real bug: prefix-matching "EURO FX" also caught
    "EURO FX/BRITISH POUND XRATE" and "EURO FX/JAPANESE YEN XRATE", doubling
    the EUR row count by blending three unrelated instruments.
    """
    for name in CONTRACT_CURRENCY:
        assert "XRATE" not in name
        assert name.endswith("EXCHANGE"), f"{name} is not a full contract name"


def test_nzd_has_both_historical_names() -> None:
    """CFTC renamed NEW ZEALAND DOLLAR -> NZ DOLLAR; matching only one made the
    series appear to stop in 2022."""
    nzd = [k for k, v in CONTRACT_CURRENCY.items() if v == "NZD"]
    assert len(nzd) == 2


def test_bar_never_sees_an_unpublished_report(cot, bars) -> None:
    """A bar may only use a report published strictly before it."""
    aligned = align_to_bars(cot, bars, "EUR", release_lag_days=6)
    eur = cot[cot.currency == "EUR"].set_index("report_date")["net_lev"]

    for ts in [bars[100], bars[300], bars[-1]]:
        val = aligned.loc[ts, "net_lev"]
        source = eur.index[np.isclose(eur.to_numpy(), val)][0]
        # Tuesday snapshot published Friday: at minimum 3 days must have passed.
        assert (ts - source).days >= 3, f"bar {ts} used a report from {source}"


def test_tuesday_report_is_not_visible_on_wednesday(cot, bars) -> None:
    aligned = align_to_bars(cot, bars, "EUR", release_lag_days=6)
    eur = cot[cot.currency == "EUR"].set_index("report_date")["net_lev"]

    tuesday = pd.Timestamp("2020-06-02", tz="UTC")
    wednesday = tuesday + pd.Timedelta(days=1)
    if tuesday in eur.index:
        assert aligned.loc[wednesday, "net_lev"] != eur.loc[tuesday]


def test_short_lag_is_rejected(cot, bars) -> None:
    with pytest.raises(ValueError, match="leaks"):
        align_to_bars(cot, bars, "EUR", release_lag_days=1)


def test_future_positioning_cannot_change_the_past(cot, bars) -> None:
    base = align_to_bars(cot, bars, "EUR", release_lag_days=6)
    cut = pd.Timestamp("2020-09-01", tz="UTC")

    corrupted = cot.copy()
    mask = (corrupted.currency == "EUR") & (corrupted.report_date >= cut)
    corrupted.loc[mask, "net_lev"] += 100.0
    after = align_to_bars(corrupted, bars, "EUR", release_lag_days=6)

    m = bars < cut
    pd.testing.assert_frame_equal(base[m], after[m])


def test_forward_fill_only_never_backfill(cot, bars) -> None:
    """Between weekly reports the last KNOWN value must persist."""
    aligned = align_to_bars(cot, bars, "EUR", release_lag_days=6)
    week = aligned.loc["2020-06-15":"2020-06-19", "net_lev"]
    assert week.nunique() == 1


def test_positions_are_normalised_by_open_interest(cot, bars) -> None:
    """Raw contract counts trend with market growth; normalised ones do not."""
    aligned = align_to_bars(cot, bars, "EUR", release_lag_days=6)
    assert aligned["net_lev"].abs().max() <= 1.0


def test_missing_currency_raises(cot, bars) -> None:
    with pytest.raises(KeyError):
        align_to_bars(cot, bars, "SEK", release_lag_days=6)


def test_features_are_prefixed_and_usd_leg_is_implicit(cot, bars) -> None:
    """
    There is no USD contract — USD is the other side of every one of these
    futures — so a USD leg contributes no column of its own.
    """
    X = build_cot_features(bars, "EUR", "USD", cot=cot)
    assert all(c.startswith("f_") for c in X.columns)
    assert any("base" in c for c in X.columns)
    assert not any("quote" in c for c in X.columns)
    assert "f_cot_diff_lev" in X.columns


def test_cross_pair_uses_both_legs(cot, bars) -> None:
    X = build_cot_features(bars, "EUR", "JPY", cot=cot)
    assert any("base" in c for c in X.columns)
    assert any("quote" in c for c in X.columns)
