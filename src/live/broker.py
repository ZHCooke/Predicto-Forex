"""
broker.py

A minimal broker interface plus a PaperBroker that fills against quotes, so the
whole execution loop can run and be measured without risking money — and a
LiveBroker stub marking exactly where a real OANDA/IBKR connector plugs in.

FILL CONVENTIONS (the honest ones from src/analysis/execution.py):
    market SELL -> fills at the BID  (crosses the spread, pays it)
    market BUY  -> fills at the ASK
    limit  SELL at L -> fills only if the market BID reaches L within the working
                        window (you are resting an offer; a buyer lifts it).
                        EARNS spread relative to mid, but may NOT fill.
    limit  BUY  at L -> fills only if the ASK falls to L.

The passive (limit) path is the one that matters: it is the only way to offset
commission (see commission.py), and whether it actually fills in the London-open
hour is precisely the unknown the live experiment exists to measure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable


@dataclass(frozen=True)
class Quote:
    ts: datetime
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True)
class Order:
    side: int              # +1 buy, -1 sell
    lots: float
    kind: str = "market"   # "market" | "limit"
    limit_price: float | None = None

    def __post_init__(self) -> None:
        if self.side not in (-1, 1):
            raise ValueError("side must be +1 (buy) or -1 (sell)")
        if self.kind not in ("market", "limit"):
            raise ValueError("kind must be 'market' or 'limit'")
        if self.kind == "limit" and self.limit_price is None:
            raise ValueError("limit order needs a limit_price")


@dataclass(frozen=True)
class Fill:
    filled: bool
    price: float | None
    ts: datetime | None
    decision_mid: float
    reason: str = ""


class Broker(ABC):
    """Interface a live connector must implement."""

    @abstractmethod
    def market_order(self, order: Order, quote: Quote) -> Fill: ...

    @abstractmethod
    def limit_order(self, order: Order, working_quotes: Iterable[Quote]) -> Fill: ...


class PaperBroker(Broker):
    """Simulated fills. No network, no money."""

    def market_order(self, order: Order, quote: Quote) -> Fill:
        price = quote.bid if order.side == -1 else quote.ask
        return Fill(True, price, quote.ts, quote.mid, reason="market")

    def limit_order(self, order: Order, working_quotes: Iterable[Quote]) -> Fill:
        quotes = list(working_quotes)
        if not quotes:
            raise ValueError("limit_order needs at least one working quote")
        decision_mid = quotes[0].mid
        limit = order.limit_price
        for q in quotes:
            # Sell limit fills when a buyer reaches up to our price (bid >= L);
            # buy limit fills when the offer drops to our price (ask <= L).
            if order.side == -1 and q.bid >= limit:
                return Fill(True, limit, q.ts, decision_mid, reason="limit_hit")
            if order.side == 1 and q.ask <= limit:
                return Fill(True, limit, q.ts, decision_mid, reason="limit_hit")
        return Fill(False, None, None, decision_mid, reason="limit_unfilled")


class LiveBroker(Broker):
    """
    STUB. A real connector (OANDA v20 REST, IBKR TWS/Gateway) implements these.

    Deliberately raises: this project's rule is no broker integration until the
    validation stage is solid and a broker is chosen (CLAUDE.md purpose statement).
    When implemented it must (a) time-stamp the decision mid BEFORE sending the
    order, so slippage is measured against the price we actually saw, and (b) log
    every requote/rejection — those are broker slippage that no backtest can show.
    """

    def market_order(self, order: Order, quote: Quote) -> Fill:
        raise NotImplementedError("wire a real broker connector here before live trading")

    def limit_order(self, order: Order, working_quotes: Iterable[Quote]) -> Fill:
        raise NotImplementedError("wire a real broker connector here before live trading")
