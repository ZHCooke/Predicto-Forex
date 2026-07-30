"""
ledger.py

The trade ledger — the instrument the whole live experiment is for.

It decomposes every trade into three parts that must not be conflated:

  MID EDGE        direction * (mid_exit - mid_entry). The strategy's raw edge at
                  the mid, unpolluted by execution. Should track the ~1.3-pip
                  selection-free backtest; if it does not, the effect has decayed
                  or the timing is wrong.
  MARKET COST     how far the actual fills sat from the mid (spread crossed, or
                  spread EARNED on a passive fill). MEASURED, and size-independent
                  because it is a price effect. This is what the experiment exists
                  to pin down.
  COMMISSION      broker fee, MODELLED from the fee schedule (commission.py). Size
                  DEPENDENT — ruinous at micro size under a per-trade minimum,
                  small at production size — so it is projected separately rather
                  than measured, letting a cheap micro-lot run speak to production.

net = mid_edge - market_cost - commission. Keeping the three apart is what lets a
$20-30 micro experiment answer the production-size question honestly (s0.0e).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import LOG_DIR
from src.live.commission import CommissionModel

log = logging.getLogger(__name__)

LEDGER_PATH = LOG_DIR / "live_ledger.jsonl"


def _leg_slippage_pips(side: int, mid: float, fill: float, pip: float) -> float:
    """Signed execution cost of one leg vs the mid, in pips (positive = worse).

    A sell that fills below mid, or a buy that fills above mid, cost us; a passive
    fill on the far side of the mid is NEGATIVE (we earned the spread)."""
    raw = (mid - fill) if side == -1 else (fill - mid)
    return raw / pip


@dataclass
class TradeRecord:
    day: str
    entry_ts: str | None
    exit_ts: str | None
    direction: int                 # -1 short
    lots: float
    order_kind: str                # "market" | "limit"
    mid_entry: float
    mid_exit: float
    fill_entry: float | None
    fill_exit: float | None
    filled: bool
    pip: float = 1e-4
    note: str = ""

    # --- derived, all in pips, all execution-cost-free where noted -----------
    def mid_edge_pips(self) -> float:
        return self.direction * (self.mid_exit - self.mid_entry) / self.pip

    def market_cost_pips(self) -> float | None:
        """Round-trip slippage vs mid (entry leg + exit leg). None if unfilled."""
        if not self.filled or self.fill_entry is None or self.fill_exit is None:
            return None
        entry_leg = _leg_slippage_pips(self.direction, self.mid_entry, self.fill_entry, self.pip)
        exit_leg = _leg_slippage_pips(-self.direction, self.mid_exit, self.fill_exit, self.pip)
        return entry_leg + exit_leg

    def realized_gross_pips(self) -> float | None:
        """Fill-to-fill P&L, before commission. None if unfilled."""
        if not self.filled or self.fill_entry is None or self.fill_exit is None:
            return None
        return self.direction * (self.fill_exit - self.fill_entry) / self.pip

    def net_pips(self, commission: CommissionModel, lots: float | None = None) -> float | None:
        """realized gross minus commission at `lots` (defaults to the traded size)."""
        gross = self.realized_gross_pips()
        if gross is None:
            return None
        price = (self.mid_entry + self.mid_exit) / 2
        return gross - commission.round_trip_pips(lots or self.lots, price)


class Ledger:
    """Append-only JSONL trade log plus experiment-grade summaries."""

    def __init__(self, path: Path = LEDGER_PATH):
        self.path = path

    def append(self, rec: TradeRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(rec)) + "\n")
        log.info("ledger: recorded %s filled=%s", rec.day, rec.filled)

    def load(self) -> list[TradeRecord]:
        if not self.path.exists():
            return []
        recs = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                recs.append(TradeRecord(**json.loads(line)))
        return recs

    def summary(
        self,
        commission: CommissionModel,
        production_lots: float = 1.0,
        market_cost_budget_pips: float = 0.36,
    ) -> dict:
        """
        Experiment read-out: fill rate, measured market cost, mid-edge sanity,
        and net at BOTH the traded size and a projected production size.

        `market_cost_budget_pips` is the round-trip slippage budget the strategy
        can absorb and still clear the s8 bar; 0.36 corresponds to the ~0.18
        pip/leg figure from s0.0e. A market cost at or below it (ideally NEGATIVE,
        from passive fills) is the pass condition for execution quality.
        """
        recs = self.load()
        if not recs:
            return {"n": 0}

        filled = [r for r in recs if r.filled]
        mkt = np.array([r.market_cost_pips() for r in filled], dtype=float)
        mid_edge = np.array([r.mid_edge_pips() for r in filled], dtype=float)
        net_traded = np.array([r.net_pips(commission) for r in filled], dtype=float)
        net_prod = np.array([r.net_pips(commission, production_lots) for r in filled], dtype=float)

        n_f = len(filled)
        return {
            "n": len(recs),
            "n_filled": n_f,
            "fill_rate": n_f / len(recs),
            "market_cost_pips_median": float(np.median(mkt)) if n_f else np.nan,
            "market_cost_pips_mean": float(np.mean(mkt)) if n_f else np.nan,
            "market_cost_budget": market_cost_budget_pips,
            "market_cost_within_budget": bool(n_f and np.median(mkt) <= market_cost_budget_pips),
            "mid_edge_pips_mean": float(np.mean(mid_edge)) if n_f else np.nan,
            "commission_rt_traded_pips": commission.round_trip_pips(filled[0].lots) if n_f else np.nan,
            "commission_rt_production_pips": commission.round_trip_pips(production_lots),
            "net_pips_mean_traded": float(np.mean(net_traded)) if n_f else np.nan,
            "net_pips_mean_production": float(np.mean(net_prod)) if n_f else np.nan,
            "broker": commission.name,
            "production_lots": production_lots,
        }
