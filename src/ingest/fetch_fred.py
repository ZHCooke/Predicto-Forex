"""
fetch_fred.py

Pulls interest-rate series from FRED (St. Louis Fed) and aligns them onto FX
bar timestamps WITHOUT lookahead.

Why rates: every feature in build_features.py is a transform of a single price
path, so none of them carry information the price doesn't already contain.
Rate differentials are exogenous to the price series and are the most robustly
documented driver of FX returns (the carry mechanism: capital flows toward the
higher-yielding currency).

No API key needed — the fredgraph CSV endpoint is public.

TWO LOOKAHEAD TRAPS, both handled in `align_to_bars`:

1. Publication lag. FRED daily series are stamped with a US business date and
   no time of day. A value stamped 2024-01-04 was not necessarily knowable at
   00:00 UTC on 2024-01-04. We shift the macro index forward by `lag_days`
   (default 1) before aligning, so a bar can only ever see a rate stamped at
   least a full day earlier.

2. Revision/vintage bias. FRED serves the LATEST REVISED value for a series.
   Using today's revised number at a 2018 timestamp is lookahead. This is
   severe for GDP/payrolls (revised for years) and largely a non-issue for
   MARKET rates like DGS2, which are not revised. We therefore restrict
   ourselves to market and policy rates. Do NOT add GDP/CPI/payrolls here
   without switching to ALFRED vintage data first.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR
from src.logging_setup import setup_logging

log = logging.getLogger(__name__)

MACRO_DIR = RAW_DATA_DIR / "macro"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

# Only daily, non-revised market/policy rates. Euro-area and JP yield series on
# FRED are monthly AND lag ~6 months in publication, so they are unusable for a
# daily strategy — deliberately excluded.
SERIES = {
    "DGS2": "US 2-year Treasury constant maturity yield",
    "DGS10": "US 10-year Treasury constant maturity yield",
    "DFF": "US effective federal funds rate",
    "ECBDFR": "ECB deposit facility rate",
}

# Rate differentials that define a pair's carry. Positive = base currency of
# the quote convention yields more than the counter currency.
PAIR_RATE_LEGS: dict[str, tuple[str, str]] = {
    # EURUSD: EUR leg is the ECB policy rate, USD leg the Fed funds rate.
    # Both are daily policy rates, so this differential is symmetric.
    "EURUSD": ("ECBDFR", "DFF"),
}


def fetch_series(series_id: str, use_cache: bool = True) -> pd.Series:
    """Fetch one FRED series, caching to data/raw/macro/<id>.parquet."""
    path = MACRO_DIR / f"{series_id}.parquet"
    if use_cache and path.exists():
        log.info("[%s] loading cached %s", series_id, path)
        return pd.read_parquet(path).iloc[:, 0]

    url = FRED_CSV.format(series_id=series_id)
    log.info("[%s] fetching %s", series_id, url)
    raw = pd.read_csv(url, parse_dates=[0], index_col=0)

    # FRED encodes missing observations as "." — coerce and drop them.
    s = pd.to_numeric(raw.iloc[:, 0], errors="coerce").dropna()
    s.index = pd.DatetimeIndex(s.index).tz_localize("UTC")
    s.index.name = "date"
    s.name = series_id

    path.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_parquet(path)
    log.info("[%s] %d obs, %s..%s -> %s",
             series_id, len(s), s.index.min().date(), s.index.max().date(), path)
    return s


def align_to_bars(
    series: pd.Series, bar_index: pd.DatetimeIndex, lag_days: int = 1
) -> pd.Series:
    """
    Put a daily macro series onto FX bar timestamps, causally.

    The macro index is shifted forward by `lag_days` and then forward-filled
    onto the bars, so bar t sees only values stamped at least `lag_days`
    before t. Forward-fill only — a backfill would let a weekend inherit
    Monday's rate, which is lookahead.
    """
    if lag_days < 1:
        raise ValueError("lag_days must be >= 1; a same-day rate may not be knowable")

    lagged = series.copy()
    lagged.index = lagged.index + pd.Timedelta(days=lag_days)
    # reindex(method="ffill") takes the last value at or BEFORE each bar.
    return lagged.reindex(bar_index, method="ffill").rename(series.name)


def build_rate_features(
    bar_index: pd.DatetimeIndex,
    symbol: str = "EURUSD",
    lag_days: int = 1,
    change_windows: tuple[int, ...] = (5, 20, 60),
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Carry / rates features aligned to `bar_index`, all prefixed `f_` so they
    drop straight into the existing feature matrix.

    `change_windows` are in DAYS (not bars) — rate moves are daily phenomena.
    """
    if symbol not in PAIR_RATE_LEGS:
        raise KeyError(f"no rate legs defined for {symbol!r}; add to PAIR_RATE_LEGS")

    base_id, quote_id = PAIR_RATE_LEGS[symbol]
    base = fetch_series(base_id, use_cache)
    quote = fetch_series(quote_id, use_cache)

    # Compute the differential on the DAILY grid first, then align once.
    # Doing it after alignment would smear step changes across intraday bars.
    daily = pd.concat([base, quote], axis=1).sort_index().ffill().dropna()
    diff = (daily[base_id] - daily[quote_id]).rename("rate_diff")

    out = pd.DataFrame(index=bar_index)
    out["f_rate_diff"] = align_to_bars(diff, bar_index, lag_days)

    for w in change_windows:
        # Change in the differential over w calendar days. Momentum in the
        # differential is generally more informative than its level.
        out[f"f_rate_diff_chg_{w}d"] = align_to_bars(
            diff.diff(w).rename(f"chg{w}"), bar_index, lag_days
        )

    # Level of each leg: the differential alone can't distinguish "both high"
    # from "both low", and the absolute rate level matters for carry appetite.
    out["f_rate_base"] = align_to_bars(daily[base_id], bar_index, lag_days)
    out["f_rate_quote"] = align_to_bars(daily[quote_id], bar_index, lag_days)

    assert all(c.startswith("f_") for c in out.columns)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull FRED rate series to data/raw/macro/")
    parser.add_argument("--refresh", action="store_true", help="ignore cache")
    args = parser.parse_args()

    setup_logging(logfile="ingest.log")
    for sid, desc in SERIES.items():
        s = fetch_series(sid, use_cache=not args.refresh)
        log.info("[%s] %s | latest %.3f on %s", sid, desc, s.iloc[-1], s.index[-1].date())


if __name__ == "__main__":
    main()
