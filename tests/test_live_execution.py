"""
Tests for the live/paper execution layer (CLAUDE.md s17).

The timing and cost-decomposition logic is where silent, expensive bugs live
(s0.0d ran the trade an hour early and lost 5.6 pips). These lock it down.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.live.broker import Fill, Order, PaperBroker, Quote
from src.live.commission import IBKR, IC_MARKETS, lots_for_representative_commission
from src.live.ledger import TradeRecord
from src.live.strategy import LondonOpenRule


# --- timing: the s0.0d correction and DST -----------------------------------

def test_entry_is_0900_london_not_0800():
    rule = LondonOpenRule()
    assert rule.entry_hour_london == 9 and rule.exit_hour_london == 13
    assert rule.hold_hours == 4


def test_summer_entry_is_0800_utc():
    # London is BST (UTC+1) in July -> 09:00 London = 08:00 UTC.
    entry, exit_ = LondonOpenRule().times_for_day(datetime(2024, 7, 10).date())
    assert entry.astimezone(timezone.utc).hour == 8
    assert exit_.astimezone(timezone.utc).hour == 12


def test_winter_entry_is_0900_utc():
    # London is GMT (UTC+0) in January -> 09:00 London = 09:00 UTC.
    entry, exit_ = LondonOpenRule().times_for_day(datetime(2024, 1, 10).date())
    assert entry.astimezone(timezone.utc).hour == 9
    assert exit_.astimezone(timezone.utc).hour == 13


def test_next_trade_skips_weekend():
    rule = LondonOpenRule()
    # Saturday 2024-07-13 -> next trade is Monday the 15th.
    nxt, _ = rule.next_trade(datetime(2024, 7, 13, 12, tzinfo=timezone.utc))
    assert nxt.astimezone(timezone.utc).date() == datetime(2024, 7, 15).date()


def test_next_trade_rolls_past_today_if_entry_gone():
    rule = LondonOpenRule()
    # Weekday, but after 08:00 UTC (summer) -> today's entry has passed.
    nxt, _ = rule.next_trade(datetime(2024, 7, 10, 10, tzinfo=timezone.utc))
    assert nxt.astimezone(timezone.utc).date() == datetime(2024, 7, 11).date()


# --- commission: the micro-lot minimum trap ---------------------------------

def test_ibkr_commission_is_small_at_standard_size():
    # $2 min on 1 lot = 0.20 pips/side = 0.40 round trip.
    assert IBKR.per_side_pips(1.0) == pytest.approx(0.20, abs=1e-6)
    assert IBKR.round_trip_pips(1.0) == pytest.approx(0.40, abs=1e-6)


def test_ibkr_minimum_is_ruinous_at_micro_size():
    # $2 min on 0.01 lots: pip value is $0.10, so 20 pips PER SIDE.
    assert IBKR.per_side_pips(0.01) == pytest.approx(20.0, abs=1e-6)


def test_zero_minimum_broker_is_size_invariant():
    # IC Markets has no minimum: same pips/side at micro and standard size.
    assert IC_MARKETS.per_side_pips(0.01) == pytest.approx(IC_MARKETS.per_side_pips(1.0), abs=1e-9)
    assert IC_MARKETS.round_trip_pips(1.0) == pytest.approx(0.70, abs=1e-6)


def test_representative_size_kills_minimum_distortion():
    lots = lots_for_representative_commission(IBKR)
    assert IBKR.per_side_pips(lots) == pytest.approx(0.20, abs=0.02)
    assert lots_for_representative_commission(IC_MARKETS) == 0.01  # no minimum


# --- paper broker fills ------------------------------------------------------

def _q(bid, ask, h=8):
    return Quote(datetime(2024, 7, 10, h, tzinfo=timezone.utc), bid, ask)


def test_market_sell_fills_at_bid():
    fill = PaperBroker().market_order(Order(-1, 1.0), _q(1.0800, 1.0801))
    assert fill.filled and fill.price == 1.0800


def test_limit_sell_fills_only_if_bid_reaches_it():
    quotes = [_q(1.0800, 1.0801), _q(1.08005, 1.08015), _q(1.08025, 1.08035)]
    # Resting sell at mid+ (1.08025): last quote's bid reaches it -> fills.
    hit = PaperBroker().limit_order(Order(-1, 1.0, "limit", 1.08025), quotes)
    assert hit.filled and hit.price == 1.08025
    # A sell limit above every bid never fills.
    miss = PaperBroker().limit_order(Order(-1, 1.0, "limit", 1.0810), quotes)
    assert not miss.filled


# --- ledger cost decomposition ----------------------------------------------

def _trade(fill_entry, fill_exit, kind="market"):
    return TradeRecord(
        day="2024-07-10", entry_ts="", exit_ts="", direction=-1, lots=1.0,
        order_kind=kind, mid_entry=1.08005, mid_exit=1.07905,
        fill_entry=fill_entry, fill_exit=fill_exit, filled=True,
    )


def test_mid_edge_is_execution_free():
    # Short, mid fell 10 pips -> +10 pip mid edge regardless of fills.
    assert _trade(1.0800, 1.0791).mid_edge_pips() == pytest.approx(10.0, abs=1e-6)


def test_aggressive_fills_cost_the_spread():
    # Sell at bid 1.0800 (0.05 below mid), buy at ask 1.0791 (0.05 above mid).
    t = _trade(1.0800, 1.0791)
    assert t.market_cost_pips() == pytest.approx(1.0, abs=1e-6)  # 0.5 + 0.5 pip


def test_passive_fills_earn_the_spread_negative_cost():
    # Sell at ask 1.0801 (above mid), buy at bid 1.0790 (below mid) -> earned.
    t = _trade(1.0801, 1.0790, kind="limit")
    assert t.market_cost_pips() < 0


def test_net_pips_subtracts_commission():
    t = _trade(1.0800, 1.0791)
    gross = t.realized_gross_pips()
    net = t.net_pips(IBKR, lots=1.0)
    # Commission is charged on USD notional, so it is priced at the trade's mid,
    # not assumed at parity — ~0.43 pips rt at EURUSD ~1.08, not exactly 0.40.
    price = (t.mid_entry + t.mid_exit) / 2
    assert net == pytest.approx(gross - IBKR.round_trip_pips(1.0, price), abs=1e-9)


def test_unfilled_trade_has_no_cost():
    t = TradeRecord(
        day="2024-07-10", entry_ts="", exit_ts="", direction=-1, lots=1.0,
        order_kind="limit", mid_entry=1.08, mid_exit=1.079,
        fill_entry=None, fill_exit=None, filled=False,
    )
    assert t.market_cost_pips() is None and t.net_pips(IBKR) is None
