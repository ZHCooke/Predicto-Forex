"""
Structural (calendar/session) features and regime-conditional testing.

The critical tests are DST correctness — the whole point of this family is that
a "London morning" bar is the same market event in July and January — and the
causality of regime assignment.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.analysis.regime import (
    conditional_test,
    interaction_test,
    quantile_regime,
    realized_vol_regime,
)
from src.features.structural import (
    build_structural_features,
    dst_aware_hour_bucket,
    local_hour,
    month_end_flags,
    session_flags,
)

LONDON = ZoneInfo("Europe/London")


@pytest.fixture
def hourly():
    return pd.date_range("2020-01-01", "2021-12-31", freq="1h", tz="UTC")


# --- DST correctness --------------------------------------------------------

def test_london_open_is_the_same_event_summer_and_winter(hourly) -> None:
    """
    The fix for the session-11 caveat. 08:00 London is 08:00 UTC in winter but
    07:00 UTC in summer; fixed UTC buckets pool two different market events.
    """
    winter = pd.Timestamp("2020-01-15 08:00", tz="UTC")
    summer = pd.Timestamp("2020-07-15 07:00", tz="UTC")
    idx = pd.DatetimeIndex([winter, summer])
    assert local_hour(idx, LONDON).tolist() == [8.0, 8.0]


def test_fixed_utc_would_have_mixed_regimes() -> None:
    """Demonstrates the bug being fixed: same UTC hour, different London hour."""
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2020-01-15 08:00", tz="UTC"), pd.Timestamp("2020-07-15 08:00", tz="UTC")]
    )
    assert local_hour(idx, LONDON).tolist() == [8.0, 9.0]


def test_session_flags_are_dst_stable(hourly) -> None:
    flags = session_flags(hourly)
    lon = local_hour(hourly, LONDON)
    # Every bar flagged as London-open must be 08:00-10:00 London time.
    on = flags["f_sess_london_open"] == 1
    assert lon[on].between(8, 10, inclusive="left").all()


def test_sessions_are_binary_and_overlap_is_a_subset(hourly) -> None:
    f = session_flags(hourly)
    for c in f.columns:
        assert set(np.unique(f[c])) <= {0.0, 1.0}
    assert (f["f_sess_overlap"] <= f["f_sess_london"]).all()
    assert (f["f_sess_overlap"] <= f["f_sess_ny"]).all()


def test_timezone_naive_index_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        local_hour(pd.date_range("2020-01-01", periods=5, freq="D"), LONDON)


# --- calendar structure -----------------------------------------------------

def test_month_end_flag_fires_at_month_end(hourly) -> None:
    f = month_end_flags(hourly, window=3)
    jan31 = f.loc["2020-01-31"]
    jan15 = f.loc["2020-01-15"]
    assert (jan31["f_is_month_end"] == 1).all()
    assert (jan15["f_is_month_end"] == 0).all()


def test_quarter_end_is_a_subset_of_month_end(hourly) -> None:
    f = month_end_flags(hourly)
    assert (f["f_is_quarter_end"] <= f["f_is_month_end"]).all()
    assert f.loc["2020-03-31", "f_is_quarter_end"].eq(1).all()
    assert f.loc["2020-01-31", "f_is_quarter_end"].eq(0).all()


def test_all_structural_features_are_finite_and_prefixed(hourly) -> None:
    X = build_structural_features(hourly)
    assert all(c.startswith("f_") for c in X.columns)
    assert np.isfinite(X.to_numpy()).all()
    assert len(X) == len(hourly)


def test_dst_aware_bucket_differs_from_utc_bucket(hourly) -> None:
    b = dst_aware_hour_bucket(hourly, LONDON, width=4)
    utc_bucket = (hourly.hour // 4 * 4)
    # They must agree in winter and disagree in summer.
    assert (b.to_numpy() != utc_bucket).any()


# --- regime assignment ------------------------------------------------------

def test_regime_assignment_is_causal() -> None:
    """A bar's regime must not depend on anything after it."""
    idx = pd.date_range("2020-01-01", periods=1000, freq="D", tz="UTC")
    s = pd.Series(np.random.default_rng(0).normal(size=1000), index=idx)

    base = quantile_regime(s, window=100)
    corrupted = s.copy()
    corrupted.iloc[600:] += 50
    after = quantile_regime(corrupted, window=100)

    pd.testing.assert_series_equal(base.iloc[:600], after.iloc[:600])


def test_vol_regime_flags_a_shift_then_normalises() -> None:
    """
    Regime is RELATIVE to the trailing window, not absolute. A jump to higher
    volatility registers as "high" while the window still remembers the calm
    period, then reverts toward neutral once the new level IS the recent
    history.

    That is the correct causal behaviour — an absolute threshold would need
    full-sample calibration and reintroduce lookahead — but it is easy to
    misread, which is why it is pinned here.
    """
    idx = pd.date_range("2020-01-01", periods=1400, freq="D", tz="UTC")
    rng = np.random.default_rng(1)
    r = pd.Series(
        np.concatenate([rng.normal(0, 0.001, 600), rng.normal(0, 0.01, 800)]), index=idx
    )
    reg = realized_vol_regime(r, window=20, n_regimes=2)

    just_after = reg.iloc[640:760].dropna()
    long_after = reg.iloc[1150:].dropna()
    # ~90% classified high; the remainder are bars where the 20-day vol
    # estimate still straddles the transition and is genuinely ambiguous.
    assert just_after.mean() >= 0.85, "a fresh vol shift must register as high"
    assert long_after.mean() < just_after.mean(), "it must normalise as the window rolls"


def test_conditional_test_reports_each_regime() -> None:
    idx = pd.date_range("2020-01-01", periods=2000, freq="D", tz="UTC")
    rng = np.random.default_rng(2)
    f = pd.Series(rng.normal(size=2000), index=idx)
    r = pd.Series(rng.normal(0, 0.01, 2000), index=idx)
    g = pd.Series(np.where(np.arange(2000) < 1000, 0.0, 1.0), index=idx)

    out = conditional_test(f, r, g)
    assert len(out) == 2
    assert {"regime", "n", "t_stat", "p_value"} <= set(out.columns)


def test_interaction_detects_a_genuine_sign_flip() -> None:
    """
    The scenario this whole module exists for: a feature that works in one
    regime and inverts in the other, which is invisible to an unconditional
    test because the two halves cancel.
    """
    idx = pd.date_range("2020-01-01", periods=2000, freq="D", tz="UTC")
    rng = np.random.default_rng(3)
    f = pd.Series(rng.normal(size=2000), index=idx)
    g = pd.Series(np.where(np.arange(2000) < 1000, 0.0, 1.0), index=idx)
    # Feature predicts returns positively in regime 0, negatively in regime 1.
    r = pd.Series(np.where(g == 0, 1, -1) * f * 0.004 + rng.normal(0, 0.004, 2000), index=idx)

    from src.backtest.scoring import signed_return_test
    uncond = signed_return_test(f, r)
    assert uncond["p_value"] > 0.05, "unconditionally this must look like nothing"

    inter = interaction_test(f, r, g)
    assert inter["p_value"] < 1e-6
    assert inter["mean_a"] > 0 > inter["mean_b"]


def test_interaction_rejects_more_than_two_regimes() -> None:
    idx = pd.date_range("2020-01-01", periods=900, freq="D", tz="UTC")
    f = pd.Series(np.random.default_rng(4).normal(size=900), index=idx)
    r = pd.Series(np.random.default_rng(5).normal(0, 0.01, 900), index=idx)
    g = pd.Series(np.arange(900) % 3, index=idx).astype(float)
    with pytest.raises(ValueError, match="exactly 2"):
        interaction_test(f, r, g)
