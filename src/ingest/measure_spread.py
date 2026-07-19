"""
measure_spread.py

Measures the real bid-ask spread by hour of day and caches it.

Why this exists: `CostModel` originally assumed a flat 0.6-pip spread. Direct
measurement showed EURUSD's median is 0.20 pips, and that it varies ~7x across
the day — 0.20 at the London-NY overlap versus 1.45 at the 21:00 UTC rollover.
A flat constant is therefore wrong in both directions: too pessimistic in the
liquid window, far too optimistic at rollover.

Since costs have been the binding constraint on every strategy tested, getting
this profile right matters more than any model change.

MEASURED ON RESEARCH DATA ONLY (pre-2023). The holdout is sealed, and although
a spread profile is a microstructure fact rather than a signal, keeping the
discipline uniform avoids having to argue about it later.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.backtest.holdout import HOLDOUT_START
from src.config import RAW_DATA_DIR
from src.ingest.fetch_dukascopy import fetch_range
from src.logging_setup import setup_logging

log = logging.getLogger(__name__)

SPREAD_DIR = RAW_DATA_DIR / "spread"


def measure_hourly_spread(
    symbol: str,
    pip: float,
    start_year: int = 2019,
    end_year: int = 2022,
    use_cache: bool = True,
) -> pd.Series:
    """
    Median spread in pips for each hour of the UTC day.

    Uses 1h bars rather than ticks: we need a stable central estimate per hour
    bucket, not the full intraday distribution, and 1h keeps the pull cheap.
    """
    if end_year >= HOLDOUT_START.year:
        raise ValueError(
            f"end_year {end_year} reaches into the sealed holdout "
            f"({HOLDOUT_START}); measure on research data only"
        )

    path = SPREAD_DIR / f"{symbol}_hourly.json"
    if use_cache and path.exists():
        log.info("[%s] loading cached spread profile %s", symbol, path)
        return pd.Series(
            {int(k): v for k, v in json.loads(path.read_text()).items()}
        ).sort_index()

    frames = []
    for year in range(start_year, end_year + 1):
        s = datetime(year, 1, 1, tzinfo=timezone.utc)
        e = datetime(year, 12, 31, 23, 59, tzinfo=timezone.utc)
        bid = fetch_range(symbol, s, e, "1h", "bid")
        ask = fetch_range(symbol, s, e, "1h", "ask")
        j = bid[["close"]].join(ask[["close"]], lsuffix="_b", rsuffix="_a").dropna()
        frames.append(j)
        log.info("[%s][%d] %d hourly observations", symbol, year, len(j))

    joined = pd.concat(frames)
    spread_pips = (joined["close_a"] - joined["close_b"]) / pip
    # Median, not mean: spread has a fat right tail (news, rollover, gaps) and
    # the mean would be dragged by episodes we would not be trading through.
    profile = spread_pips.groupby(joined.index.hour).median()
    profile.index.name = "hour_utc"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(k): float(v) for k, v in profile.items()}, indent=2))
    log.info("[%s] wrote profile -> %s", symbol, path)
    return profile


def best_trading_hours(profile: pd.Series, max_multiple: float = 1.5) -> list[int]:
    """
    Hours whose spread is within `max_multiple` of the day's cheapest hour.

    This is the trading window: outside it we are paying a materially worse
    price for the same signal, which on a strategy with a thin edge is the
    difference between viable and not.
    """
    floor = profile.min()
    return sorted(int(h) for h, v in profile.items() if v <= floor * max_multiple)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure hourly bid-ask spread")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--pip", type=float, default=1e-4)
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    setup_logging(logfile="ingest.log")
    profile = measure_hourly_spread(
        args.symbol, args.pip, args.start_year, args.end_year, use_cache=not args.refresh
    )
    hours = best_trading_hours(profile)
    log.info("\n%s", profile.to_string(float_format="%.2f"))
    log.info("cheapest hour %dh (%.2f pips), dearest %dh (%.2f pips), ratio %.1fx",
             profile.idxmin(), profile.min(), profile.idxmax(), profile.max(),
             profile.max() / profile.min())
    log.info("recommended trading hours (within 1.5x of cheapest): %s", hours)


if __name__ == "__main__":
    main()
