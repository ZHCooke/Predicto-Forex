"""
Macro alignment is a prime lookahead vector: a daily series joined onto
intraday bars will silently leak the future if it is backfilled, or aligned
with zero lag, or reindexed with `method="nearest"`.

These tests pin the causal contract without touching the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ingest.fetch_fred import align_to_bars


@pytest.fixture
def bar_index():
    return pd.date_range("2024-01-01", "2024-02-01", freq="15min", tz="UTC")


@pytest.fixture
def daily():
    """A rate series that steps up by 1.0 each day — easy to trace."""
    idx = pd.date_range("2023-12-01", "2024-03-01", freq="D", tz="UTC")
    return pd.Series(np.arange(len(idx), dtype=float), index=idx, name="RATE")


def test_bar_never_sees_a_future_rate(bar_index, daily) -> None:
    aligned = align_to_bars(daily, bar_index, lag_days=1)

    for ts in [bar_index[0], bar_index[500], bar_index[-1]]:
        value = aligned.loc[ts]
        # Recover the stamp date of the value used, and assert it is strictly
        # in the past relative to the bar.
        source_date = daily.index[daily.to_numpy() == value][0]
        assert source_date < ts, f"bar {ts} used a rate stamped {source_date}"


def test_lag_is_at_least_one_full_day(bar_index, daily) -> None:
    aligned = align_to_bars(daily, bar_index, lag_days=1)
    ts = pd.Timestamp("2024-01-15 00:00", tz="UTC")
    # Value at midnight on the 15th must be the 14th's reading, not the 15th's.
    assert aligned.loc[ts] == daily.loc[pd.Timestamp("2024-01-14", tz="UTC")]


def test_larger_lag_shifts_further_back(bar_index, daily) -> None:
    ts = pd.Timestamp("2024-01-15 12:00", tz="UTC")
    a1 = align_to_bars(daily, bar_index, lag_days=1).loc[ts]
    a5 = align_to_bars(daily, bar_index, lag_days=5).loc[ts]
    assert a5 == a1 - 4  # series steps by 1.0/day


def test_zero_lag_is_rejected(bar_index, daily) -> None:
    with pytest.raises(ValueError, match="lag_days"):
        align_to_bars(daily, bar_index, lag_days=0)


def test_future_values_cannot_change_the_past(bar_index, daily) -> None:
    """The decisive test: corrupt the macro series' future, and nothing at or
    before the cut may move."""
    cut = pd.Timestamp("2024-01-15", tz="UTC")
    corrupted = daily.copy()
    corrupted.loc[corrupted.index >= cut] += 1000.0

    base = align_to_bars(daily, bar_index, lag_days=1)
    after = align_to_bars(corrupted, bar_index, lag_days=1)

    mask = bar_index < cut
    pd.testing.assert_series_equal(base[mask], after[mask])


def test_weekend_holds_fridays_value_not_mondays(bar_index, daily) -> None:
    """Forward-fill only. A weekend bar must carry the last KNOWN rate, never
    reach forward to Monday's."""
    sparse = daily[daily.index.dayofweek < 5]  # business days only
    aligned = align_to_bars(sparse, bar_index, lag_days=1)

    sat = pd.Timestamp("2024-01-13 12:00", tz="UTC")  # Saturday
    fri = pd.Timestamp("2024-01-12", tz="UTC")        # the preceding business day
    mon = pd.Timestamp("2024-01-15", tz="UTC")
    assert aligned.loc[sat] == sparse.loc[fri]
    assert aligned.loc[sat] != sparse.loc[mon], "weekend reached forward to Monday"


def test_step_changes_are_not_smeared(bar_index) -> None:
    """A policy rate is a step function; alignment must preserve the step, not
    interpolate it into a ramp."""
    idx = pd.date_range("2023-12-01", "2024-03-01", freq="D", tz="UTC")
    step = pd.Series(
        np.where(idx < pd.Timestamp("2024-01-20", tz="UTC"), 4.0, 4.5),
        index=idx, name="POLICY",
    )
    aligned = align_to_bars(step, bar_index, lag_days=1)
    assert set(aligned.dropna().unique()) <= {4.0, 4.5}, "alignment interpolated a step"


def test_output_is_indexed_on_the_bars(bar_index, daily) -> None:
    aligned = align_to_bars(daily, bar_index, lag_days=1)
    assert aligned.index.equals(bar_index)
    assert aligned.name == "RATE"
