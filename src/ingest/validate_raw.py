"""
validate_raw.py

Quality gate on anything under data/raw/. Checks structure, timezone, ordering,
duplicates, OHLC sanity and calendar gaps. Reports rather than repairs — a
silent auto-fix here is how bad data reaches a backtest.

Run:  python -m src.ingest.validate_raw --symbol EURUSD --timeframe 15min
Exit code is non-zero if any ERROR-severity issue is found, so this can gate CI.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR
from src.logging_setup import setup_logging

log = logging.getLogger(__name__)

OHLC_COLUMNS = ["open", "high", "low", "close"]

# Bars expected per timeframe over a full FX week (Sun 22:00 - Fri 22:00 UTC),
# used only to size the "suspiciously large gap" threshold.
TIMEFRAME_MINUTES = {
    "1min": 1, "5min": 5, "15min": 15, "30min": 30,
    "1h": 60, "4h": 240, "1d": 1440,
}


@dataclass
class ValidationReport:
    symbol: str
    timeframe: str
    n_rows: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def log(self) -> None:
        tag = f"[{self.symbol}][{self.timeframe}]"
        log.info("%s %d rows, %d errors, %d warnings",
                 tag, self.n_rows, len(self.errors), len(self.warnings))
        for msg in self.errors:
            log.error("%s %s", tag, msg)
        for msg in self.warnings:
            log.warning("%s %s", tag, msg)


def load_raw(symbol: str, timeframe: str) -> pd.DataFrame:
    """Concatenate every partition for a symbol/timeframe into one frame."""
    root = RAW_DATA_DIR / symbol / timeframe
    if not root.exists():
        raise FileNotFoundError(f"no raw data at {root}")
    parts = sorted(root.rglob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no parquet partitions under {root}")
    df = pd.concat([pd.read_parquet(p) for p in parts]).sort_index()
    log.info("[%s][%s] loaded %d rows from %d partitions", symbol, timeframe, len(df), len(parts))
    return df


def validate(df: pd.DataFrame, symbol: str, timeframe: str) -> ValidationReport:
    """Run all checks against an in-memory raw frame."""
    rep = ValidationReport(symbol=symbol, timeframe=timeframe, n_rows=len(df))

    if df.empty:
        rep.error("frame is empty")
        return rep

    _check_index(df, rep)
    if timeframe != "tick":
        _check_ohlc(df, rep)
        _check_gaps(df, timeframe, rep)
    return rep


def _check_index(df: pd.DataFrame, rep: ValidationReport) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        rep.error(f"index is {type(df.index).__name__}, expected DatetimeIndex")
        return
    if df.index.tz is None:
        rep.error("index is timezone-naive; fetch_dukascopy should emit tz-aware UTC")
    elif str(df.index.tz) != "UTC":
        rep.error(f"index timezone is {df.index.tz}, expected UTC")

    if not df.index.is_monotonic_increasing:
        rep.error("index is not sorted ascending")

    n_dupes = int(df.index.duplicated().sum())
    if n_dupes:
        rep.error(f"{n_dupes} duplicate timestamps")


def _check_ohlc(df: pd.DataFrame, rep: ValidationReport) -> None:
    missing = [c for c in OHLC_COLUMNS if c not in df.columns]
    if missing:
        rep.error(f"missing OHLC columns: {missing}")
        return

    n_nan = int(df[OHLC_COLUMNS].isna().any(axis=1).sum())
    if n_nan:
        rep.error(f"{n_nan} rows with NaN in OHLC")

    n_nonpos = int((df[OHLC_COLUMNS] <= 0).any(axis=1).sum())
    if n_nonpos:
        rep.error(f"{n_nonpos} rows with non-positive prices")

    # high must bound open/close/low, low must floor them.
    bad_high = (df["high"] < df[["open", "close", "low"]].max(axis=1)).sum()
    bad_low = (df["low"] > df[["open", "close", "high"]].min(axis=1)).sum()
    if bad_high:
        rep.error(f"{int(bad_high)} rows where high < max(open, close, low)")
    if bad_low:
        rep.error(f"{int(bad_low)} rows where low > min(open, close, high)")

    if "volume" in df.columns:
        n_zero_vol = int((df["volume"] == 0).sum())
        if n_zero_vol:
            rep.warn(f"{n_zero_vol} bars with zero volume")


def _check_gaps(df: pd.DataFrame, timeframe: str, rep: ValidationReport) -> None:
    """
    Flag gaps beyond a normal weekend. FX closes ~Fri 22:00 UTC and reopens
    ~Sun 22:00 UTC, so a ~48h gap every week is expected, not a defect.
    """
    step_min = TIMEFRAME_MINUTES.get(timeframe)
    if step_min is None:
        rep.warn(f"no gap threshold defined for timeframe {timeframe!r}, skipping gap check")
        return

    deltas = df.index.to_series().diff().dropna()
    weekend = pd.Timedelta(hours=49)
    suspicious = deltas[(deltas > weekend)]

    if len(suspicious):
        worst = suspicious.sort_values(ascending=False).head(5)
        rep.warn(
            f"{len(suspicious)} gaps longer than a weekend; largest: "
            + ", ".join(f"{ts:%Y-%m-%d %H:%M}(+{d})" for ts, d in worst.items())
        )

    # Bars arriving faster than the timeframe means overlapping/misaligned data.
    too_fast = deltas[deltas < pd.Timedelta(minutes=step_min)]
    if len(too_fast):
        rep.error(f"{len(too_fast)} intervals shorter than the {timeframe} bar width")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate raw Dukascopy partitions")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)
    args = parser.parse_args()

    setup_logging(logfile="validate.log")
    df = load_raw(args.symbol, args.timeframe)
    rep = validate(df, args.symbol, args.timeframe)
    rep.log()
    raise SystemExit(0 if rep.ok else 1)


if __name__ == "__main__":
    main()
