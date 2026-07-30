"""
Live / paper execution layer for the London-open EURUSD rule (CLAUDE.md s17).

The research is settled (s0.0b-e): the effect is real over 21 years and clears
the s8 bar at the tick-MEASURED market cost. What is NOT settled is what a real
broker charges once commission, fill quality and slippage are included — and that
cannot be measured from historical bars. This package is the instrument for that:
a broker-agnostic execution loop, a paper ledger, and a cost model that keeps
MEASURED slippage separate from MODELLED commission so a micro-lot experiment can
be projected to production size.

Nothing here places a real order until a concrete Broker connector is written and
credentials are supplied. `PaperBroker` is the default and is safe.
"""
