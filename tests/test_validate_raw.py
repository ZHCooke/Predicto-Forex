"""validate_raw must actually catch the defects it claims to catch."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingest.fetch_dukascopy import normalize_index
from src.ingest.validate_raw import validate


def test_clean_bars_pass(bars) -> None:
    rep = validate(bars, "TEST", "15min")
    assert rep.ok, rep.errors


def test_naive_index_is_an_error(bars) -> None:
    naive = bars.tz_localize(None)
    rep = validate(naive, "TEST", "15min")
    assert any("timezone-naive" in e for e in rep.errors)


def test_non_utc_index_is_an_error(bars) -> None:
    rep = validate(bars.tz_convert("Europe/London"), "TEST", "15min")
    assert any("expected UTC" in e for e in rep.errors)


def test_duplicate_timestamps_are_an_error(bars) -> None:
    dupes = pd.concat([bars, bars.iloc[:5]]).sort_index()
    rep = validate(dupes, "TEST", "15min")
    assert any("duplicate" in e for e in rep.errors)


def test_unsorted_index_is_an_error(bars) -> None:
    rep = validate(bars.iloc[::-1], "TEST", "15min")
    assert any("sorted" in e for e in rep.errors)


def test_broken_ohlc_is_an_error(bars) -> None:
    broken = bars.copy()
    broken.iloc[10, broken.columns.get_loc("high")] = broken["low"].iloc[10] - 1
    rep = validate(broken, "TEST", "15min")
    assert any("high <" in e for e in rep.errors)


def test_missing_columns_are_an_error(bars) -> None:
    rep = validate(bars.drop(columns=["close"]), "TEST", "15min")
    assert any("missing OHLC" in e for e in rep.errors)


def test_nan_prices_are_an_error(bars) -> None:
    holed = bars.copy()
    holed.iloc[3, holed.columns.get_loc("close")] = float("nan")
    rep = validate(holed, "TEST", "15min")
    assert any("NaN" in e for e in rep.errors)


def test_long_gap_is_a_warning_not_an_error(bars) -> None:
    gapped = pd.concat([bars.iloc[:100], bars.iloc[500:]])
    rep = validate(gapped, "TEST", "15min")
    assert rep.ok, "a data gap is a warning, not a hard failure"
    assert any("gaps longer than a weekend" in w for w in rep.warnings)


def test_empty_frame_is_an_error() -> None:
    rep = validate(pd.DataFrame(), "TEST", "15min")
    assert not rep.ok


def test_normalize_index_makes_frames_valid(bars) -> None:
    """The ingest-side normalizer should produce something validate() accepts."""
    messy = pd.concat([bars.tz_localize(None), bars.tz_localize(None).iloc[:10]])
    messy = messy.sample(frac=1, random_state=0)  # shuffle order too

    rep = validate(normalize_index(messy), "TEST", "15min")
    assert rep.ok, rep.errors
