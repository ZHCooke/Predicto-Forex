"""
fetch_dukascopy.py

Wraps dukascopy-python to pull historical bar/tick data and write it to
partitioned parquet under data/raw/. Safe to re-run: a partition that already
exists is skipped unless `overwrite=True`.

Timezone policy (decided here, not in validate_raw.py):
    Dukascopy serves UTC. Everything written under data/raw/ carries a
    tz-aware UTC DatetimeIndex named "timestamp". Downstream code may assume
    UTC and never has to guess. `validate_raw.py` asserts this rather than
    fixing it.

On-disk layout:
    bars:  data/raw/<SYMBOL>/<timeframe>/<year>.parquet
    ticks: data/raw/<SYMBOL>/tick/<year>/<YYYY-MM-DD>.parquet

Tick partitions are day-level because a single instrument-year of ticks is
tens of millions of rows and will not fit comfortably in memory.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import dukascopy_python
import pandas as pd
from dukascopy_python import instruments as duka_instruments

from src.config import RAW_DATA_DIR, load_instruments
from src.logging_setup import setup_logging

log = logging.getLogger(__name__)

# Friendly symbol -> dukascopy_python instrument constant.
# Keep in sync with config/instruments.yaml.
INSTRUMENT_MAP: dict[str, str] = {
    "EURUSD": duka_instruments.INSTRUMENT_FX_MAJORS_EUR_USD,
    "GBPUSD": duka_instruments.INSTRUMENT_FX_MAJORS_GBP_USD,
    "USDJPY": duka_instruments.INSTRUMENT_FX_MAJORS_USD_JPY,
    "AUDUSD": duka_instruments.INSTRUMENT_FX_MAJORS_AUD_USD,
    "USDCHF": duka_instruments.INSTRUMENT_FX_MAJORS_USD_CHF,
    "USDCAD": duka_instruments.INSTRUMENT_FX_MAJORS_USD_CAD,
    "NZDUSD": duka_instruments.INSTRUMENT_FX_MAJORS_NZD_USD,
}

# Which currency is base vs quote in each pair's quoting convention. Needed to
# turn a pair return into a statement about two CURRENCIES rather than one
# instrument — the basis of the cross-pair strength work (CLAUDE.md s7).
PAIR_LEGS: dict[str, tuple[str, str]] = {
    "EURUSD": ("EUR", "USD"),
    "GBPUSD": ("GBP", "USD"),
    "AUDUSD": ("AUD", "USD"),
    "NZDUSD": ("NZD", "USD"),
    "USDJPY": ("USD", "JPY"),
    "USDCHF": ("USD", "CHF"),
    "USDCAD": ("USD", "CAD"),
}

# Friendly timeframe -> dukascopy_python interval constant.
INTERVAL_MAP: dict[str, str] = {
    "tick": dukascopy_python.INTERVAL_TICK,
    "1min": dukascopy_python.INTERVAL_MIN_1,
    "5min": dukascopy_python.INTERVAL_MIN_5,
    "15min": dukascopy_python.INTERVAL_MIN_15,
    "30min": dukascopy_python.INTERVAL_MIN_30,
    "1h": dukascopy_python.INTERVAL_HOUR_1,
    "4h": dukascopy_python.INTERVAL_HOUR_4,
    "1d": dukascopy_python.INTERVAL_DAY_1,
}

OFFER_SIDE_MAP: dict[str, str] = {
    "bid": dukascopy_python.OFFER_SIDE_BID,
    "ask": dukascopy_python.OFFER_SIDE_ASK,
}


def normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Force a tz-aware UTC DatetimeIndex named 'timestamp', sorted, deduped."""
    df = df.copy()
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df.index = idx.rename("timestamp")
    df = df[~df.index.duplicated(keep="first")]
    return df.sort_index()


def fetch_range(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str,
    offer_side: str = "bid",
    max_attempts: int = 3,
    retry_backoff_s: float = 5.0,
) -> pd.DataFrame:
    """
    Fetch one symbol/timeframe/date range from Dukascopy.

    dukascopy_python.fetch() already retries individual chunk downloads; this
    outer loop exists for failures that take down the whole call (DNS, proxy,
    connection reset mid-pagination). Raises once attempts are exhausted —
    the caller decides whether to skip the partition or halt the pull.
    """
    if symbol not in INSTRUMENT_MAP:
        raise KeyError(f"unknown symbol {symbol!r}; add it to INSTRUMENT_MAP")
    if timeframe not in INTERVAL_MAP:
        raise KeyError(f"unknown timeframe {timeframe!r}; add it to INTERVAL_MAP")

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            df = dukascopy_python.fetch(
                instrument=INSTRUMENT_MAP[symbol],
                interval=INTERVAL_MAP[timeframe],
                offer_side=OFFER_SIDE_MAP[offer_side],
                start=start,
                end=end,
            )
            return normalize_index(df)
        except Exception as exc:  # noqa: BLE001 - want to retry on anything transient
            last_exc = exc
            if attempt == max_attempts:
                break
            wait = retry_backoff_s * attempt
            log.warning(
                "[%s][%s] attempt %d/%d failed (%s); retrying in %.0fs",
                symbol, timeframe, attempt, max_attempts, exc, wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"fetch failed for {symbol} {timeframe} {start:%Y-%m-%d}..{end:%Y-%m-%d}"
    ) from last_exc


def partition_path(symbol: str, timeframe: str, period: int | date) -> Path:
    """Path for one partition. `period` is a year for bars, a date for ticks."""
    if timeframe == "tick":
        assert isinstance(period, date), "tick partitions are day-level"
        return RAW_DATA_DIR / symbol / "tick" / str(period.year) / f"{period}.parquet"
    return RAW_DATA_DIR / symbol / timeframe / f"{period}.parquet"


def write_partition(df: pd.DataFrame, path: Path) -> Path:
    """Write one partition to parquet, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)
    return path


# A partition is only "already done" if it actually spans the range we want.
# Tolerance covers the normal case where a year's data starts a day or two in
# (New Year holiday) or ends early on the final Friday.
COVERAGE_TOLERANCE = timedelta(days=4)


def covers_range(path: Path, want_start: date, want_end: date) -> bool:
    """
    True if an existing partition already spans [want_start, want_end].

    Existence alone is NOT sufficient: a partition written by an earlier,
    narrower pull (e.g. a one-month smoke test) would otherwise be skipped
    forever, leaving a silent hole in the middle of the dataset.
    """
    try:
        idx = pd.read_parquet(path, columns=[]).index
    except Exception as exc:  # noqa: BLE001 - unreadable partition = re-fetch it
        log.warning("could not read %s (%s); will re-fetch", path, exc)
        return False

    if len(idx) == 0:
        return False

    have_start, have_end = idx.min().date(), idx.max().date()
    return (
        have_start <= want_start + COVERAGE_TOLERANCE
        and have_end >= want_end - COVERAGE_TOLERANCE
    )


def _pull_bars(
    symbol: str, timeframe: str, start: date, end: date, offer_side: str, overwrite: bool
) -> None:
    """Year-by-year: bounded memory and natural checkpoints for long pulls."""
    for year in range(start.year, end.year + 1):
        out_path = partition_path(symbol, timeframe, year)

        # Clamp the year window to the configured range so partial first/last
        # years don't silently pull data outside what was asked for.
        y_start = max(start, date(year, 1, 1))
        y_end = min(end, date(year, 12, 31))

        if out_path.exists() and not overwrite:
            if covers_range(out_path, y_start, y_end):
                log.info("[%s][%s][%d] exists and covers range, skipping",
                         symbol, timeframe, year)
                continue
            log.warning(
                "[%s][%s][%d] exists but does NOT cover %s..%s; re-fetching",
                symbol, timeframe, year, y_start, y_end,
            )

        df = fetch_range(
            symbol,
            datetime.combine(y_start, datetime.min.time(), tzinfo=timezone.utc),
            datetime.combine(y_end, datetime.max.time(), tzinfo=timezone.utc),
            timeframe,
            offer_side=offer_side,
        )
        if df.empty:
            log.warning("[%s][%s][%d] returned no rows, not writing", symbol, timeframe, year)
            continue
        write_partition(df, out_path)
        log.info("[%s][%s][%d] wrote %d rows -> %s", symbol, timeframe, year, len(df), out_path)


def _pull_ticks(
    symbol: str, start: date, end: date, offer_side: str, overwrite: bool
) -> None:
    """Day-by-day. Weekends return empty and are skipped without writing."""
    day = start
    while day <= end:
        out_path = partition_path(symbol, "tick", day)
        if out_path.exists() and not overwrite:
            log.info("[%s][tick][%s] exists, skipping", symbol, day)
            day += timedelta(days=1)
            continue

        df = fetch_range(
            symbol,
            datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc),
            datetime.combine(day, datetime.max.time(), tzinfo=timezone.utc),
            "tick",
            offer_side=offer_side,
        )
        if df.empty:
            log.debug("[%s][tick][%s] no rows (weekend/holiday?)", symbol, day)
        else:
            write_partition(df, out_path)
            log.info("[%s][tick][%s] wrote %d rows", symbol, day, len(df))
        day += timedelta(days=1)


def pull_symbol(
    symbol: str,
    timeframe: str,
    start: date,
    end: date,
    offer_side: str = "bid",
    overwrite: bool = False,
) -> None:
    """Pull one symbol/timeframe over [start, end] into partitioned parquet."""
    if timeframe == "tick":
        _pull_ticks(symbol, start, end, offer_side, overwrite)
    else:
        _pull_bars(symbol, timeframe, start, end, offer_side, overwrite)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull Dukascopy data to data/raw/")
    parser.add_argument("--symbol", help="Override config; e.g. EURUSD")
    parser.add_argument("--timeframe", help="Override config; e.g. 15min or tick")
    parser.add_argument("--start", type=date.fromisoformat, help="YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, help="YYYY-MM-DD")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    setup_logging(logfile="ingest.log")
    configs = load_instruments()

    if args.symbol:
        cfg = configs.get(args.symbol)
        if cfg is None:
            raise SystemExit(
                f"{args.symbol} not enabled in config/instruments.yaml"
            )
        targets = {args.symbol: cfg}
    else:
        targets = configs

    if not targets:
        raise SystemExit("no enabled instruments in config/instruments.yaml")

    for symbol, cfg in targets.items():
        pull_symbol(
            symbol,
            timeframe=args.timeframe or cfg.timeframe,
            start=args.start or cfg.start,
            end=args.end or cfg.end,
            offer_side=cfg.offer_side,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
