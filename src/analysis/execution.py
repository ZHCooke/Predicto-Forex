"""
execution.py

Tick-level execution analysis: what fill would we ACTUALLY have got?

WHY THIS EXISTS. After 21 years of data the London-open effect is settled —
1.314 pips selection-free, p = 1e-9, no decay. The entire remaining question is
execution cost, and specifically SLIPPAGE, which is the only input separating
Sharpe 0.337 from Sharpe 0.534 with a confidence interval excluding zero.

Slippage cannot be measured from bar data. But it CAN be decomposed, and one
half of it is measurable right now from tick data:

  MARKET slippage  - the spread actually available at the instant of the trade,
                     and how far price moves while an order is worked.
                     MEASURABLE HERE.
  BROKER slippage  - latency, requotes, rejections, markup on the raw feed.
                     Requires live execution; no amount of historical data
                     substitutes for it.

A broker DEMO account does not answer this either: demo servers typically fill
at the quoted price instantly, which flatters exactly the number we care about.
That is why this module exists instead of a demo harness.

WHAT WE MEASURE, PER TRADING DAY
  - the true bid/ask at the decision instant (08:00:00 London)
  - the spread distribution across the execution window, not just at one tick
  - how far price drifts in the seconds after the decision (adverse selection:
    if price runs away from us the moment we decide, a passive order misses and
    an aggressive one pays for it)
  - fills under three honest execution styles, from pessimistic to optimistic

EXECUTION STYLES, and the direction convention. The strategy SHORTS EURUSD at
the open and buys it back four hours later.
  aggressive : sell at the bid, buy back at the ask -> pays the full spread twice
  mid        : both legs at mid -> pays nothing, unachievable, an upper bound
  passive    : sell at the ask, buy back at the bid -> EARNS the spread, but only
               if the order is filled, which is not guaranteed

The truth is between aggressive and passive. Reporting all three brackets the
answer rather than pretending to a precision we do not have.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.ingest.fetch_dukascopy import fetch_range

log = logging.getLogger(__name__)

LONDON = ZoneInfo("Europe/London")


def london_instant(day: date, hour: int) -> datetime:
    """The UTC instant corresponding to a given London wall-clock hour."""
    return datetime.combine(day, time(hour, 0), tzinfo=LONDON).astimezone(timezone.utc)


def fetch_day_ticks(symbol: str, day: date) -> pd.DataFrame:
    """One day of ticks with bid and ask."""
    start = datetime.combine(day, time(0, 0), tzinfo=timezone.utc)
    end = datetime.combine(day, time(23, 59, 59), tzinfo=timezone.utc)
    df = fetch_range(symbol, start, end, "tick", "bid")
    if df.empty:
        return df
    return df.rename(columns={"bidPrice": "bid", "askPrice": "ask"})


def window_stats(ticks: pd.DataFrame, at: datetime, window_s: int = 60, pip: float = 1e-4) -> dict:
    """
    Execution conditions around an intended trade time.

    `window_s` is how long we allow ourselves to work the order. Longer means
    more chance of a passive fill but more exposure to price drifting away.
    """
    if ticks.empty:
        return {}

    lo = at - timedelta(seconds=window_s)
    hi = at + timedelta(seconds=window_s)
    w = ticks.loc[(ticks.index >= lo) & (ticks.index <= hi)]
    if w.empty:
        return {}

    # The tick in force at the decision instant: the last one at or before it.
    prior = ticks.loc[ticks.index <= at]
    if prior.empty:
        return {}
    at_tick = prior.iloc[-1]

    spread = (w["ask"] - w["bid"]) / pip
    mid = (w["ask"] + w["bid"]) / 2
    mid_at = (at_tick["ask"] + at_tick["bid"]) / 2

    after = w.loc[w.index > at]
    drift = ((after["ask"] + after["bid"]) / 2 - mid_at) / pip if len(after) else pd.Series(dtype=float)

    return {
        "n_ticks": len(w),
        "spread_at": (at_tick["ask"] - at_tick["bid"]) / pip,
        "spread_median": float(spread.median()),
        "spread_p90": float(spread.quantile(0.9)),
        "bid_at": float(at_tick["bid"]),
        "ask_at": float(at_tick["ask"]),
        "mid_at": float(mid_at),
        # Adverse drift: how far mid moves in our favour (negative for a short
        # means price fell, which is good) during the working window.
        "drift_mean": float(drift.mean()) if len(drift) else np.nan,
        "drift_p90_adverse": float(drift.quantile(0.9)) if len(drift) else np.nan,
        "mid_range": float((mid.max() - mid.min()) / pip),
    }


def simulate_day(
    symbol: str,
    day: date,
    entry_hour: int = 8,
    hold_hours: int = 4,
    window_s: int = 60,
    pip: float = 1e-4,
) -> dict | None:
    """
    Reconstruct one day's trade from ticks under three execution styles.

    Returns per-style P&L in pips for a SHORT, so positive = profit.
    """
    ticks = fetch_day_ticks(symbol, day)
    if ticks.empty:
        return None

    t_in = london_instant(day, entry_hour)
    t_out = t_in + timedelta(hours=hold_hours)

    a = window_stats(ticks, t_in, window_s, pip)
    b = window_stats(ticks, t_out, window_s, pip)
    if not a or not b:
        return None

    # Short: open by SELLING, close by BUYING. P&L = sell price - buy price.
    styles = {
        # Cross the spread both ways — the pessimistic, always-achievable case.
        "aggressive": (a["bid_at"], b["ask_at"]),
        # Both legs at mid — unachievable, an upper bound on any real fill.
        "mid": (a["mid_at"], b["mid_at"]),
        # Rest passively on both legs — earns the spread IF filled.
        "passive": (a["ask_at"], b["bid_at"]),
    }

    out = {
        "day": day,
        "spread_entry": a["spread_at"],
        "spread_exit": b["spread_at"],
        "entry_drift_mean": a["drift_mean"],
        "entry_range": a["mid_range"],
        "n_ticks_entry": a["n_ticks"],
    }
    for name, (sell, buy) in styles.items():
        out[f"pnl_{name}"] = (sell - buy) / pip
    return out


def run_execution_study(
    symbol: str = "EURUSD",
    days: list[date] | None = None,
    entry_hour: int = 8,
    hold_hours: int = 4,
    window_s: int = 60,
    pip: float = 1e-4,
) -> pd.DataFrame:
    """Run `simulate_day` over a sample of days and return the raw records."""
    rows = []
    for i, d in enumerate(days or [], 1):
        try:
            r = simulate_day(symbol, d, entry_hour, hold_hours, window_s, pip)
        except Exception as exc:  # noqa: BLE001 - one bad day must not stop the study
            log.warning("%s failed: %s", d, exc)
            continue
        if r:
            rows.append(r)
        if i % 10 == 0:
            log.info("processed %d/%d days", i, len(days))
    return pd.DataFrame(rows)


def sample_trading_days(start: date, end: date, n: int, seed: int = 0) -> list[date]:
    """
    A random sample of weekdays.

    Random rather than consecutive so the sample is not dominated by one
    market regime, and seeded so the study is reproducible.
    """
    all_days = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(all_days), size=min(n, len(all_days)), replace=False)
    return sorted(all_days[i].date() for i in picked)
