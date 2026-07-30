"""
cross_asset.py

Session-window returns for the cross-asset SESSION-TRANSMISSION lead study
(CLAUDE.md s16).

THE IDEA. In liquid G10 FX the currency almost always leads other asset classes,
not the other way round — FX is faster and deeper. The one exception is
information that arrives while a currency's home market is ASLEEP: a risk or
commodity move made during the US afternoon, when Europe and Asia are closed,
cannot be reacted to until those markets next open. That is the only structurally
defensible LEAD, and it is a sibling of the project's one real finding (the
London-open effect, s11) — signal in retail FX lives in how the market is
ORGANISED, not in transforms of past prices.

So we do NOT test "does gold predict EURUSD daily" (that is contemporaneous
co-movement, untradeable). We test "does the US-session move in asset X predict
the NEXT open of currency Y, the one that was closed while X was moving".

HOW LOOKAHEAD IS MADE IMPOSSIBLE. Every session window carries its real
start/end UTC timestamps. A predictor session is paired only with a target
session whose window STARTS at or after the predictor's window ENDS
(`pair_lead`, merge_asof forward). The predictor value is therefore fully known
before the target window begins, by construction — there is no clock arithmetic
to get wrong, and DST/timezone/weekend edges are handled automatically (a Friday
US afternoon pairs with the following Monday open, since that is genuinely the
next session).

WHY THE OVERLAP TRAP (s0.5 #1) DOES NOT APPLY HERE. That trap fires when an
H-bar forward return is sampled every bar, so neighbours share (H-1)/H of their
content. Here there is exactly ONE observation per session per day and the
windows are disjoint, so consecutive observations share nothing. n_effective is
the honest number of days.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.ingest.validate_raw import load_raw

log = logging.getLogger(__name__)

LONDON = ZoneInfo("Europe/London")
NEW_YORK = ZoneInfo("America/New_York")
TOKYO = ZoneInfo("Asia/Tokyo")
SYDNEY = ZoneInfo("Australia/Sydney")

# Cross-asset predictors and FX targets alike are stored as 1h MID bars. Mid is
# mandatory: bid-only prices fabricate signal at illiquid hours (s0.5 #2), and
# these windows include some thin ones.
MID_TIMEFRAME = "1h_mid"


@dataclass(frozen=True)
class Window:
    """A daily session window, defined in a market's OWN local time (DST-aware).

    `tz` is the market clock; [hour_start, hour_end) are local hours. Defining
    the window in local time means a 'London morning' is the same market event
    in summer and winter, rather than drifting an hour with the clocks — the
    same correction that mattered for the London-open effect (s12).
    """

    tz: ZoneInfo
    hour_start: float
    hour_end: float
    label: str

    def __post_init__(self) -> None:
        if not (0 <= self.hour_start < self.hour_end <= 24):
            raise ValueError(f"bad window hours: {self.hour_start}..{self.hour_end}")


def load_mid_close(symbol: str) -> pd.Series:
    """The 1h mid close for one symbol (FX pair or cross-asset CFD), UTC index."""
    df = load_raw(symbol, MID_TIMEFRAME)
    return df["close"].rename(symbol)


def session_frame(symbol: str, window: Window, weekday: int | None = None) -> pd.DataFrame:
    """
    Log return realised during `window` on each local trading day.

    Returns one row per local session-day, indexed by that local date, with:
        start_utc : timestamp of the first bar in the window (UTC)
        end_utc   : timestamp of the last  bar in the window (UTC)
        ret       : log(last_close / first_open) over the window

    `ret` is fully determined at `end_utc`. First-bar OPEN to last-bar CLOSE
    captures the whole window move; open and close are point-in-time and exact
    even on mid bars (s13 note). `weekday` optionally restricts to a single day
    of the local week (0 = Monday) — used by the weekend-gap tests.
    """
    df = load_raw(symbol, MID_TIMEFRAME)
    if df.index.tz is None:
        raise ValueError("index must be tz-aware UTC")

    local = df.index.tz_convert(window.tz)
    lh = local.hour + local.minute / 60.0
    mask = (lh >= window.hour_start) & (lh < window.hour_end)
    if weekday is not None:
        mask &= local.dayofweek == weekday

    sub = df[mask]
    if sub.empty:
        return pd.DataFrame(columns=["start_utc", "end_utc", "ret"])

    day_key = pd.Index(sub.index.tz_convert(window.tz).date, name="session_date")
    grp_first = sub.groupby(day_key).agg(
        first_ts=("open", lambda s: s.index.min()),
        last_ts=("open", lambda s: s.index.max()),
        first_open=("open", "first"),
        last_close=("close", "last"),
    )
    out = pd.DataFrame(
        {
            "start_utc": grp_first["first_ts"],
            "end_utc": grp_first["last_ts"],
            "ret": np.log(grp_first["last_close"] / grp_first["first_open"]),
        }
    )
    return out.dropna()


def pair_lead(pred: pd.DataFrame, tgt: pd.DataFrame, max_gap_hours: float = 84.0) -> pd.DataFrame:
    """
    Pair each predictor session with the NEXT target session that starts at or
    after the predictor ends. Lookahead-proof by construction.

    `max_gap_hours` = 84 bridges a normal weekend (Fri afternoon -> Mon open is
    ~58h) and the odd holiday, without letting a match reach absurdly far. Rows
    with no target inside the window are dropped.
    """
    p = pred.sort_values("end_utc").reset_index()
    t = tgt.sort_values("start_utc").reset_index(drop=True)
    p["end_utc"] = pd.to_datetime(p["end_utc"], utc=True)
    t["start_utc"] = pd.to_datetime(t["start_utc"], utc=True)

    merged = pd.merge_asof(
        p,
        t.rename(columns={"ret": "tgt_ret", "start_utc": "tgt_start", "end_utc": "tgt_end"}),
        left_on="end_utc",
        right_on="tgt_start",
        direction="forward",
        tolerance=pd.Timedelta(hours=max_gap_hours),
    ).dropna(subset=["tgt_ret"])

    merged = merged.rename(columns={"ret": "pred_ret"})
    merged["gap_hours"] = (merged["tgt_start"] - merged["end_utc"]).dt.total_seconds() / 3600.0
    return merged[["end_utc", "tgt_start", "gap_hours", "pred_ret", "tgt_ret"]]


def pair_contemporaneous(pred: pd.DataFrame, tgt: pd.DataFrame) -> pd.DataFrame:
    """
    Pair predictor and target on the SAME session (identical window/day).

    For the deliberately-untradeable contemporaneous CONTROLS: if the same-window
    correlation is strong while the lead versions are ~0, that proves the
    cross-asset link is co-movement, not predictability.
    """
    p = pred.rename(columns={"ret": "pred_ret"})
    t = tgt.rename(columns={"ret": "tgt_ret"})
    j = p.join(t["tgt_ret"], how="inner")
    j["gap_hours"] = 0.0
    return j[["start_utc", "end_utc", "gap_hours", "pred_ret", "tgt_ret"]].rename(
        columns={"start_utc": "tgt_start"}
    )
