"""
A partition that exists but under-covers its year must be re-fetched, not
skipped. Regression test for a real bug: a one-month smoke-test pull left
2025.parquet in place, and an existence-only idempotency check would have
skipped it during the full 2015-2026 pull, silently leaving eleven months
missing from the middle of the dataset.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.ingest.fetch_dukascopy import covers_range, write_partition


def _write(tmp_path, start: str, end: str, freq: str = "15min"):
    idx = pd.date_range(start, end, freq=freq, tz="UTC", name="timestamp")
    df = pd.DataFrame({"close": 1.1}, index=idx)
    return write_partition(df, tmp_path / "part.parquet")


def test_full_year_partition_covers(tmp_path) -> None:
    path = _write(tmp_path, "2025-01-01", "2025-12-31")
    assert covers_range(path, date(2025, 1, 1), date(2025, 12, 31))


def test_one_month_partition_does_not_cover_the_year(tmp_path) -> None:
    path = _write(tmp_path, "2025-06-01", "2025-06-30")
    assert not covers_range(path, date(2025, 1, 1), date(2025, 12, 31))


def test_partition_missing_the_tail_does_not_cover(tmp_path) -> None:
    path = _write(tmp_path, "2025-01-01", "2025-08-31")
    assert not covers_range(path, date(2025, 1, 1), date(2025, 12, 31))


def test_partition_missing_the_head_does_not_cover(tmp_path) -> None:
    path = _write(tmp_path, "2025-04-01", "2025-12-31")
    assert not covers_range(path, date(2025, 1, 1), date(2025, 12, 31))


def test_holiday_edges_are_tolerated(tmp_path) -> None:
    """FX doesn't trade Jan 1 or the last day or two of the year — a partition
    starting Jan 2 and ending Dec 30 is complete, not short."""
    path = _write(tmp_path, "2025-01-02", "2025-12-30")
    assert covers_range(path, date(2025, 1, 1), date(2025, 12, 31))


def test_partial_year_request_is_satisfied_by_matching_data(tmp_path) -> None:
    """A partial final year: asking only through June is met by June data."""
    path = _write(tmp_path, "2026-01-01", "2026-06-30")
    assert covers_range(path, date(2026, 1, 1), date(2026, 6, 30))


def test_empty_partition_does_not_cover(tmp_path) -> None:
    df = pd.DataFrame(
        {"close": []}, index=pd.DatetimeIndex([], tz="UTC", name="timestamp")
    )
    path = write_partition(df, tmp_path / "empty.parquet")
    assert not covers_range(path, date(2025, 1, 1), date(2025, 12, 31))


def test_unreadable_partition_does_not_cover(tmp_path) -> None:
    bad = tmp_path / "corrupt.parquet"
    bad.write_bytes(b"not a parquet file")
    assert not covers_range(bad, date(2025, 1, 1), date(2025, 12, 31))
