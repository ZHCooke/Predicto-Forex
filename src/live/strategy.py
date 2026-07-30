"""
strategy.py

The London-open rule as an executable spec, and the DST-aware timing that turns
it into concrete UTC instants.

THE TIMING IS THE DANGEROUS PART. Session 15 (s0.0d) found that Dukascopy bars
are START-labelled: the bar stamped 08:00 London OPENS at 08:00 and CLOSES at
09:00. The research uses the CLOSE of the 08:00 bar, so the actual trade is:

    ENTER at 09:00 London, EXIT at 13:00 London.

NOT 08:00-12:00, as earlier prose loosely said. The first tick study ran an hour
early on both legs and produced -5.6 pips, which is what caught the bug. Any live
implementation MUST use 09:00 London for entry. That correction is encoded here
as the default, and locked by a test.

All times are computed in Europe/London local time and converted to UTC, so the
trade fires at the same MARKET moment in summer and winter (08:00 UTC in summer,
09:00 UTC in winter) rather than drifting with the clocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

LONDON = ZoneInfo("Europe/London")


@dataclass(frozen=True)
class LondonOpenRule:
    """
    Short EURUSD at the London open, hold four hours. A pure calendar rule with
    no fitted parameters.

    entry_hour_london / exit_hour_london are LOCAL London wall-clock hours. The
    defaults (9 -> 13) are the s0.0d-corrected trade, not the loose 8 -> 12.
    """

    symbol: str = "EURUSD"
    direction: int = -1                 # -1 = short
    entry_hour_london: int = 9
    exit_hour_london: int = 13
    pip: float = 1e-4

    def __post_init__(self) -> None:
        if self.direction not in (-1, 1):
            raise ValueError("direction must be +1 or -1")
        if not 0 <= self.entry_hour_london < self.exit_hour_london <= 24:
            raise ValueError("need 0 <= entry < exit <= 24 (London hours)")

    @property
    def hold_hours(self) -> int:
        return self.exit_hour_london - self.entry_hour_london

    def times_for_day(self, day: date) -> tuple[datetime, datetime]:
        """(entry_utc, exit_utc) for a given London calendar day, DST-aware."""
        entry = datetime.combine(day, time(self.entry_hour_london), tzinfo=LONDON)
        exit_ = datetime.combine(day, time(self.exit_hour_london), tzinfo=LONDON)
        return entry.astimezone(timezone.utc), exit_.astimezone(timezone.utc)

    def next_trade(self, now_utc: datetime) -> tuple[datetime, datetime]:
        """
        The next (entry_utc, exit_utc) at or after `now_utc`, skipping weekends.

        If today's entry is still in the future, use today; otherwise roll
        forward to the next weekday. Holidays are NOT handled here — the broker
        will reject or the day will simply have no fill, and the ledger records
        that. Encoding a holiday calendar is a separate, broker-specific concern.
        """
        if now_utc.tzinfo is None:
            raise ValueError("now_utc must be tz-aware")
        now_utc = now_utc.astimezone(timezone.utc)

        day = now_utc.astimezone(LONDON).date()
        for _ in range(8):  # at most a week of weekend/rollover hops
            if _is_weekday(day):
                entry, exit_ = self.times_for_day(day)
                if entry >= now_utc:
                    return entry, exit_
            day += timedelta(days=1)
        raise RuntimeError("no trading day found within a week — check the clock")


def _is_weekday(day: date) -> bool:
    """FX trades Monday-Friday. Sunday-evening opens are ignored: the London
    open is always a weekday event."""
    return day.weekday() < 5
