"""
commission.py

Broker commission, in PIPS, kept deliberately separate from spread and slippage.

WHY THIS MATTERS — A CORRECTION TO THE REPO'S COST STORY (session 17).
s0.0d/e reported a tick-MEASURED round-trip cost of 0.345 pips and treated that
as "the cost". It is not. That figure is the raw bid-ask SPREAD on Dukascopy's
ECN feed and EXCLUDES commission, which at a real broker is the dominant, fixed
cost:

    IBKR:            0.20 bps of notional per side, $2.00 minimum
                     = 0.20 pips/side = 0.40 pips round trip at >= 1 standard lot
    IC / Pepperstone: $3.50 per 100k per side (no minimum)
                     = 0.35 pips/side = 0.70 pips round trip, any size

So the honest retail all-in cost is ~0.6-0.8 pips round trip, right back at the
project's original 0.80 assumption — and against a 1.314-pip selection-free edge
that leaves the strategy borderline. Viability now hinges on PASSIVE fills
(earning the spread to offset commission), which is exactly what the live
experiment must test.

THE MINIMUM IS A TRAP AT MICRO SIZE. IBKR's $2 minimum is 0.20 pips/side at one
standard lot but 20 pips/side at 0.01 lots. A "$20-30 micro-lot experiment"
(s0.0e) at IBKR therefore pays ruinous PROPORTIONAL commission — which does NOT
mean the strategy is dead, because commission is known analytically from the fee
schedule and is size-dependent. The experiment measures SLIPPAGE (a price effect,
size-independent); commission is then projected to production size. This model
keeps the two separate so that projection is honest.
"""

from __future__ import annotations

from dataclasses import dataclass

STANDARD_LOT_UNITS = 100_000


@dataclass(frozen=True)
class CommissionModel:
    """
    Per-side commission for one broker, convertible to pips at any trade size.

    `rate_bps` is basis points of notional per side; `min_usd` is the per-side
    minimum. `pip_value_per_std_lot_usd` is the value of one pip on one standard
    lot (≈ $10 for USD-quoted majors).
    """

    name: str
    rate_bps: float
    min_usd: float
    pip_value_per_std_lot_usd: float = 10.0

    def per_side_usd(self, lots: float, price: float = 1.0) -> float:
        notional = lots * STANDARD_LOT_UNITS * price
        return max(self.rate_bps * 1e-4 * notional, self.min_usd)

    def per_side_pips(self, lots: float, price: float = 1.0) -> float:
        """Commission for one leg expressed in pips at this trade size."""
        pip_value = self.pip_value_per_std_lot_usd * lots
        if pip_value <= 0:
            raise ValueError("lots must be positive")
        return self.per_side_usd(lots, price) / pip_value

    def round_trip_pips(self, lots: float, price: float = 1.0) -> float:
        return 2 * self.per_side_pips(lots, price)


# Reference brokers (verified July 2026 — re-check before relying on them, fee
# schedules move). US-accessible set is limited by CFTC/NFA to IBKR, OANDA,
# Forex.com, IG US, Schwab; IC Markets / Pepperstone do NOT accept US residents.
IBKR = CommissionModel("IBKR", rate_bps=0.20, min_usd=2.00)
IC_MARKETS = CommissionModel("IC_Markets", rate_bps=0.35, min_usd=0.0)   # $3.50/100k, no min
PEPPERSTONE = CommissionModel("Pepperstone", rate_bps=0.35, min_usd=0.0)
OANDA_SPREAD_ONLY = CommissionModel("OANDA_spread_only", rate_bps=0.0, min_usd=0.0)  # cost is in the spread

BROKERS = {b.name: b for b in [IBKR, IC_MARKETS, PEPPERSTONE, OANDA_SPREAD_ONLY]}


def lots_for_representative_commission(model: CommissionModel, price: float = 1.0) -> float:
    """
    Trade size at/above which the per-side commission stops being inflated by
    `min_usd` and settles at its rate-based (production-representative) value.

    The minimum dominates while rate*notional < min_usd; the crossover is
        lots = min_usd / (rate_bps * 1e-4 * STANDARD_LOT_UNITS * price)
    (≈ 1.0 standard lot for IBKR). Zero-minimum brokers are cost-invariant with
    size, so any positive size is representative — return the smallest.
    """
    if model.min_usd <= 0:
        return 0.01
    lots = model.min_usd / (model.rate_bps * 1e-4 * STANDARD_LOT_UNITS * price)
    return round(lots, 4)
