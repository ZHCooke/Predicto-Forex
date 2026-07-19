"""
fetch_cot.py

CFTC Commitments of Traders — weekly speculative positioning in FX futures.

WHY THIS IS DIFFERENT FROM EVERYTHING ELSE WE HAVE. Session 11 established that
our entire price-derived corpus carries no directional information (1 nominal
hit in 82 tests where chance gives 4.1). The diagnosis was that we only had
public price history, which is the most-mined dataset in finance. COT is the
one genuinely different thing available free: it reports WHO IS POSITIONED
HOW, which is closer to order flow than anything derivable from a price series.

The documented hypothesis is contrarian: when leveraged speculators reach a
positioning extreme, they are collectively vulnerable to a squeeze, and
subsequent returns tend to reverse. That is a claim about market structure, not
about price patterns — the same category as the London-open effect, which is
the only thing that has survived scrutiny here.

TWO PUBLICATION TRAPS, BOTH HANDLED.

1. The report is a snapshot of TUESDAY, published FRIDAY at 15:30 Eastern.
   Using it from Tuesday is a three-day lookahead and would produce a
   spectacular, entirely fake backtest. `align_to_bars` therefore releases each
   observation only from the following Monday, which is conservative even
   against the Friday-evening release.

2. Positions are reported in CONTRACTS, and open interest grows over the
   decades. A raw net-position series therefore trends, and a model would learn
   the trend rather than the positioning. Everything here is normalised — as a
   share of open interest, or as a percentile of its own trailing history.
"""

from __future__ import annotations

import argparse
import io
import logging
import urllib.request
import zipfile
from datetime import date

import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR
from src.logging_setup import setup_logging

log = logging.getLogger(__name__)

COT_DIR = RAW_DATA_DIR / "cot"
HISTORY_URL = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"
# The plain URL 403s without this; CFTC blocks unidentified clients.
HEADERS = {"User-Agent": "Mozilla/5.0 (FX research pipeline)"}

# EXACT CFTC contract names -> currency.
#
# Exact, not prefix, matching. `startswith("EURO FX")` also catches
# "EURO FX/BRITISH POUND XRATE" and "EURO FX/JAPANESE YEN XRATE" — cross-rate
# contracts against a different counter currency entirely. That bug silently
# doubled the EUR row count (1407 vs 654) and would have blended three
# unrelated instruments into one positioning series.
#
# NZD appears under two names across the history: CFTC renamed
# "NEW ZEALAND DOLLAR" to "NZ DOLLAR", which is why matching only the old name
# made the series appear to end in 2022. Both are mapped.
CONTRACT_CURRENCY = {
    "EURO FX - CHICAGO MERCANTILE EXCHANGE": "EUR",
    "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE": "GBP",
    "BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE": "GBP",
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE": "JPY",
    "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE": "CHF",
    "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "CAD",
    "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "AUD",
    "NZ DOLLAR - CHICAGO MERCANTILE EXCHANGE": "NZD",
    "NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE": "NZD",
}

# Trader categories in the financial-futures report. Leveraged money is the
# speculative cohort the contrarian hypothesis is about; asset managers are
# slower real-money flow and are kept for contrast.
CATEGORIES = {
    "lev": ("Lev_Money_Positions_Long_All", "Lev_Money_Positions_Short_All"),
    "asset_mgr": ("Asset_Mgr_Positions_Long_All", "Asset_Mgr_Positions_Short_All"),
    "dealer": ("Dealer_Positions_Long_All", "Dealer_Positions_Short_All"),
}

OPEN_INTEREST = "Open_Interest_All"
DATE_COL = "Report_Date_as_YYYY-MM-DD"
NAME_COL = "Market_and_Exchange_Names"


def _fetch_year(year: int) -> pd.DataFrame:
    url = HISTORY_URL.format(year=year)
    log.info("[COT] fetching %s", url)
    req = urllib.request.Request(url, headers=HEADERS)
    blob = urllib.request.urlopen(req, timeout=120).read()

    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = z.namelist()[0]
        return pd.read_csv(io.BytesIO(z.read(name)), low_memory=False)


def fetch_cot(
    start_year: int = 2014, end_year: int = 2026, use_cache: bool = True
) -> pd.DataFrame:
    """
    Download and tidy COT positioning for the FX majors.

    Returns one row per (report_date, currency) with net positioning by
    category, normalised by open interest.
    """
    path = COT_DIR / f"cot_{start_year}_{end_year}.parquet"
    if use_cache and path.exists():
        log.info("[COT] loading cached %s", path)
        return pd.read_parquet(path)

    frames = []
    for year in range(start_year, end_year + 1):
        try:
            frames.append(_fetch_year(year))
        except Exception as exc:  # noqa: BLE001 - a missing future year is normal
            log.warning("[COT] %d unavailable (%s)", year, exc)

    if not frames:
        raise RuntimeError("no COT data could be downloaded")

    raw = pd.concat(frames, ignore_index=True)
    raw.columns = [c.strip().strip('"') for c in raw.columns]
    raw[NAME_COL] = raw[NAME_COL].astype(str).str.upper().str.strip()

    rows = []
    for contract, ccy in CONTRACT_CURRENCY.items():
        # Exact equality — see the note on CONTRACT_CURRENCY.
        sub = raw[raw[NAME_COL] == contract].copy()
        if sub.empty:
            log.debug("[COT] no rows for %s (may be a renamed variant)", contract)
            continue

        out = pd.DataFrame(
            {"report_date": pd.to_datetime(sub[DATE_COL], errors="coerce")}
        )
        oi = pd.to_numeric(sub[OPEN_INTEREST], errors="coerce")
        out["open_interest"] = oi.to_numpy()

        for cat, (lcol, scol) in CATEGORIES.items():
            long_ = pd.to_numeric(sub[lcol], errors="coerce").to_numpy()
            short = pd.to_numeric(sub[scol], errors="coerce").to_numpy()
            # Normalised by open interest so the series is comparable across
            # decades. Raw contract counts trend with market growth.
            out[f"net_{cat}"] = (long_ - short) / oi.to_numpy()

        out["currency"] = ccy
        rows.append(out.dropna(subset=["report_date"]))

    df = pd.concat(rows, ignore_index=True).sort_values(["currency", "report_date"])
    df["report_date"] = df["report_date"].dt.tz_localize("UTC")

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("[COT] %d rows, %s..%s -> %s", len(df),
             df.report_date.min().date(), df.report_date.max().date(), path)
    return df


def align_to_bars(
    cot: pd.DataFrame, bar_index: pd.DatetimeIndex, currency: str, release_lag_days: int = 6
) -> pd.DataFrame:
    """
    Put a currency's COT series onto bar timestamps, causally.

    THE LAG IS THE WHOLE POINT. The report snapshots Tuesday's positions but is
    not published until Friday 15:30 Eastern. Aligning on the report date would
    hand the model three days of hindsight and produce a fake backtest.
    `release_lag_days=6` releases Tuesday's snapshot from the following Monday
    — conservative even against the Friday release, so a timezone or holiday
    edge case cannot leak.
    """
    if release_lag_days < 4:
        raise ValueError(
            "release_lag_days < 4 leaks: the Tuesday snapshot is not public "
            "until Friday evening"
        )

    sub = cot[cot["currency"] == currency].copy()
    if sub.empty:
        raise KeyError(f"no COT data for {currency!r}")

    sub = sub.set_index("report_date").sort_index()
    sub.index = sub.index + pd.Timedelta(days=release_lag_days)

    cols = [c for c in sub.columns if c.startswith("net_") or c == "open_interest"]
    return sub[cols].reindex(bar_index, method="ffill")


def build_cot_features(
    bar_index: pd.DatetimeIndex,
    currency_base: str,
    currency_quote: str,
    cot: pd.DataFrame | None = None,
    percentile_window: int = 156,
    release_lag_days: int = 6,
) -> pd.DataFrame:
    """
    Positioning features for a pair, prefixed `f_`.

    The headline is the DIFFERENTIAL between the two legs' speculative
    positioning, which is what a pair actually expresses, plus each leg's
    percentile within its own trailing history — the percentile is what makes
    "an extreme" meaningful, since the absolute level drifts with market
    structure.

    `percentile_window` is in WEEKS (156 = three years).
    """
    cot = fetch_cot() if cot is None else cot
    out = pd.DataFrame(index=bar_index)

    legs = {}
    for role, ccy in (("base", currency_base), ("quote", currency_quote)):
        if ccy == "USD":
            # There is no USD contract: USD is the other side of every one of
            # these futures, so its positioning is implicit rather than reported.
            continue
        try:
            legs[role] = align_to_bars(cot, bar_index, ccy, release_lag_days)
        except KeyError:
            log.warning("no COT series for %s", ccy)

    for role, df in legs.items():
        for cat in CATEGORIES:
            col = f"net_{cat}"
            if col not in df:
                continue
            out[f"f_cot_{role}_{cat}"] = df[col]
            # Percentile within trailing history: an "extreme" only means
            # something relative to what is normal for this contract.
            roll = df[col].rolling(percentile_window * 5, min_periods=100)
            out[f"f_cot_{role}_{cat}_pct"] = (
                (df[col] - roll.min()) / (roll.max() - roll.min()).replace(0, np.nan)
            )

    # The differential: what the PAIR expresses. Only defined when both legs
    # have a reported contract.
    if "base" in legs and "quote" in legs:
        out["f_cot_diff_lev"] = legs["base"]["net_lev"] - legs["quote"]["net_lev"]
    elif "base" in legs:
        # USD is the quote leg, so long-base positioning IS the pair view.
        out["f_cot_diff_lev"] = legs["base"]["net_lev"]
    elif "quote" in legs:
        out["f_cot_diff_lev"] = -legs["quote"]["net_lev"]

    assert all(c.startswith("f_") for c in out.columns)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch CFTC COT positioning")
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    setup_logging(logfile="ingest.log")
    df = fetch_cot(args.start_year, args.end_year, use_cache=not args.refresh)
    log.info("\n%s", df.groupby("currency")["report_date"].agg(["min", "max", "size"]).to_string())


if __name__ == "__main__":
    main()
