"""
structural.py

Calendar and session features — the "when", not the "what".

WHY THIS FAMILY. Session 11 screened 82 price-derived features and found
essentially nothing (1 nominal hit where chance gives 4.1). The single survivor
was intraday seasonality at the London open, which is not a price pattern at
all: it is a STRUCTURAL regularity, existing because of how the market is
organised rather than because of what prices did yesterday.

That is a signpost. This module builds features from market structure —
sessions, month-end, expiries — none of which require any data beyond a
calendar, and none of which have been mined into oblivion the way moving
averages have.

DAYLIGHT SAVING IS HANDLED HERE, AND IT MATTERS. London opens at 08:00 UTC in
winter and 07:00 UTC in summer; New York's afternoon shifts likewise. The
session-11 result used fixed UTC buckets and therefore mixed two regimes
together, which either dilutes the effect or flatters it. Everything here is
computed in the relevant LOCAL market time via zoneinfo, so a "London morning"
bar is the same market event all year round.
"""

from __future__ import annotations

import logging
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

LONDON = ZoneInfo("Europe/London")
NEW_YORK = ZoneInfo("America/New_York")
TOKYO = ZoneInfo("Asia/Tokyo")


def local_hour(index: pd.DatetimeIndex, tz: ZoneInfo) -> pd.Series:
    """Hour of day in a market's own timezone, DST included."""
    if index.tz is None:
        raise ValueError("index must be timezone-aware (UTC)")
    local = index.tz_convert(tz)
    return pd.Series(local.hour + local.minute / 60.0, index=index)


def session_flags(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Which trading session each bar falls in, using local market hours so the
    boundaries survive daylight saving.

    Sessions overlap deliberately — the London/NY overlap is the most liquid
    window of the day and is worth isolating.
    """
    lon = local_hour(index, LONDON)
    ny = local_hour(index, NEW_YORK)
    tok = local_hour(index, TOKYO)

    out = pd.DataFrame(index=index)
    out["f_sess_tokyo"] = ((tok >= 9) & (tok < 15)).astype(float)
    out["f_sess_london"] = ((lon >= 8) & (lon < 16)).astype(float)
    out["f_sess_ny"] = ((ny >= 8) & (ny < 17)).astype(float)
    out["f_sess_overlap"] = (out["f_sess_london"] * out["f_sess_ny"])
    # The first two hours after the London open — where the session-11 effect
    # lives, now defined in London time rather than fixed UTC.
    out["f_sess_london_open"] = ((lon >= 8) & (lon < 10)).astype(float)
    out["f_sess_ny_close"] = ((ny >= 15) & (ny < 17)).astype(float)
    return out


def month_end_flags(index: pd.DatetimeIndex, window: int = 3) -> pd.DataFrame:
    """
    Proximity to month and quarter end.

    Month-end drives large, price-insensitive rebalancing flows: funds hedging
    international holdings adjust currency exposure to benchmark on the last
    business days. Those flows are mechanical rather than informed, which is
    exactly the kind of counterparty a systematic strategy wants.
    """
    dates = pd.Series(index.date, index=index)
    bdays = pd.Series(index.normalize(), index=index)

    # Business days remaining in the month, counted on the actual calendar.
    month_ends = bdays.groupby([index.year, index.month]).transform("max")
    days_to_end = (month_ends - bdays).dt.days

    out = pd.DataFrame(index=index)
    out["f_days_to_month_end"] = days_to_end.clip(upper=15) / 15.0
    out["f_is_month_end"] = (days_to_end <= window).astype(float)
    out["f_is_month_start"] = (pd.Series(index.day, index=index) <= window).astype(float)
    out["f_is_quarter_end"] = (
        out["f_is_month_end"] * pd.Series(index.month.isin([3, 6, 9, 12]).astype(float), index=index)
    )
    out["f_is_year_end"] = (
        out["f_is_month_end"] * pd.Series((index.month == 12).astype(float), index=index)
    )
    return out


def expiry_flags(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    FX option expiry effects.

    Vanilla FX options cut at 10:00 New York time. Around that cut, dealers
    hedging large strikes can pin or push spot, and the flow is again
    mechanical. The monthly expiry (third Wednesday) concentrates the largest
    open interest.
    """
    ny = local_hour(index, NEW_YORK)
    dow = pd.Series(index.dayofweek, index=index)
    dom = pd.Series(index.day, index=index)

    out = pd.DataFrame(index=index)
    # The 10am NY cut, and the two hours leading into it.
    out["f_ny_cut"] = ((ny >= 9) & (ny < 11)).astype(float)
    # Third Wednesday: the standard monthly expiry.
    out["f_monthly_expiry"] = ((dow == 2) & (dom >= 15) & (dom <= 21)).astype(float)
    out["f_expiry_window"] = out["f_monthly_expiry"] * out["f_ny_cut"]
    return out


def week_flags(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Day-of-week structure, encoded as flags rather than a linear ramp.

    A linear `dayofweek` feature (which is what we used until now) forces a
    monotone relationship across the week, which has no economic justification.
    Monday and Friday behave differently from midweek in FX — gap risk into the
    weekend, and position squaring on Friday afternoon.
    """
    dow = pd.Series(index.dayofweek, index=index)
    out = pd.DataFrame(index=index)
    for i, name in enumerate(["mon", "tue", "wed", "thu", "fri"]):
        out[f"f_dow_{name}"] = (dow == i).astype(float)
    ny = local_hour(index, NEW_YORK)
    out["f_friday_pm"] = ((dow == 4) & (ny >= 12)).astype(float)
    out["f_monday_am"] = ((dow == 0) & (ny < 12)).astype(float)
    return out


def build_structural_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """All structural features for a bar index. Requires only the timestamps."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")

    out = pd.concat(
        [
            session_flags(index),
            month_end_flags(index),
            expiry_flags(index),
            week_flags(index),
        ],
        axis=1,
    )
    assert all(c.startswith("f_") for c in out.columns)
    return out


def dst_aware_hour_bucket(index: pd.DatetimeIndex, tz: ZoneInfo = LONDON, width: int = 4) -> pd.Series:
    """
    Bucket bars by LOCAL market hour rather than fixed UTC.

    This is the direct fix for the session-11 caveat: the London-open effect was
    measured in fixed UTC buckets, so summer and winter bars were pooled
    despite being different market events. Bucketing on local time aligns them.
    """
    lh = local_hour(index, tz)
    return (lh // width * width).astype(int).rename("local_bucket")
