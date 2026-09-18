"""The simulated broker's fill model and account arithmetic."""

from __future__ import annotations

import contextlib
from datetime import timedelta

import pytest

from competition.broker import SimConfig, SimulatedBroker
from competition.broker.alpaca import (
    _f,
    _ts,
    parse_bar,
    parse_order,
    parse_position,
    parse_quote,
)
from competition.broker.base import InsufficientFunds, OrderRejected
from competition.types import (
    OrderIntent,
    OrderStatus,
    OrderType,
    Quote,
    Side,
    TimeInForce,
    utcnow,
)


@pytest.fixture
def prices():
    return {"AAPL": 100.0, "MSFT": 400.0}


@pytest.fixture
def broker(prices):
    now = {"t": utcnow()}

    def quotes(sym):
        p = prices.get(sym)
        if p is None:
            return None
        return Quote(sym, now["t"], p * 0.999, p * 1.001, 500, 500)

    b = SimulatedBroker(5000.0, quotes, config=SimConfig(slippage_bps=2.0),
                        clock=lambda: now["t"])
    b._test_now = now      # type: ignore[attr-defined]
    return b


# --------------------------------------------------------------------------- #
# market orders
# --------------------------------------------------------------------------- #


def test_notional_market_buy_fills_at_the_ask_plus_slippage(broker):
    o = broker.submit(OrderIntent("AAPL", "buy", notional=2000))
    assert o.status is OrderStatus.FILLED
    assert o.filled_avg_price == pytest.approx(100.0 * 1.001 * 1.0002, rel=1e-6)
    # A $2,000 notional order must spend at most $2,000, not $2,000 plus
    # slippage -- it is sized against the expected fill price.
    spent = o.filled_qty * o.filled_avg_price
    assert spent == pytest.approx(2000.0, rel=1e-6)
    assert spent <= 2000.0 + 1e-6


def test_equity_tracks_the_mark(broker, prices):
    broker.submit(OrderIntent("AAPL", "buy", notional=2000))
    before = broker.equity
    prices["AAPL"] = 110.0
    assert broker.equity > before
    prices["AAPL"] = 90.0
    assert broker.equity < before


def test_cash_and_equity_conserve_on_a_round_trip(broker):
    broker.submit(OrderIntent("AAPL", "buy", notional=2000))
    broker.submit(OrderIntent("AAPL", "sell", qty=broker.qty_of("AAPL")))
    acct = broker.account()
    assert acct.positions == ()
    # Two crossings of a 0.2% spread plus 2bps slippage each way.
    assert acct.cash == pytest.approx(5000.0, rel=3e-3)
    assert acct.cash < 5000.0            # you always pay the spread


def test_average_entry_price_is_weighted(broker, prices):
    broker.submit(OrderIntent("AAPL", "buy", qty=10))
    prices["AAPL"] = 200.0
    broker.submit(OrderIntent("AAPL", "buy", qty=10))
    pos = broker.account().position("AAPL")
    assert pos.qty == pytest.approx(20)
    assert pos.avg_entry_price == pytest.approx((100.12 + 200.24) / 2, rel=1e-3)


def test_realized_pl_is_booked_on_the_sell(broker, prices):
    broker.submit(OrderIntent("AAPL", "buy", qty=10))
    prices["AAPL"] = 120.0
    broker.submit(OrderIntent("AAPL", "sell", qty=10))
    assert broker.realized_pl == pytest.approx(10 * (120 * 0.999 * 0.9998 - 100.12),
                                               rel=1e-2)


# --------------------------------------------------------------------------- #
# long-only discipline
# --------------------------------------------------------------------------- #


def test_selling_nothing_is_rejected(broker):
    with pytest.raises(OrderRejected, match="nothing held"):
        broker.submit(OrderIntent("AAPL", "sell", qty=5))


def test_oversell_is_trimmed_to_flat_never_short(broker):
    broker.submit(OrderIntent("AAPL", "buy", qty=10))
    o = broker.submit(OrderIntent("AAPL", "sell", qty=1_000_000))
    assert o.filled_qty == pytest.approx(10)
    assert broker.qty_of("AAPL") == 0.0


def test_buying_beyond_cash_is_clipped_and_never_overdraws(broker):
    o = broker.submit(OrderIntent("AAPL", "buy", notional=100_000))
    assert o.filled_qty * o.filled_avg_price == pytest.approx(5000, rel=1e-2)
    assert broker.account().cash >= 0.0
    # Whatever is left may still be spent, but never more than is there.
    left = broker.account().cash
    with contextlib.suppress(InsufficientFunds):
        broker.submit(OrderIntent("MSFT", "buy", notional=1000))
    assert broker.account().cash >= -1e-9
    assert broker.account().cash <= left


def test_a_buy_that_can_only_afford_dust_is_refused(prices):
    def quotes(sym):
        p = prices.get(sym)
        return Quote(sym, utcnow(), p * 0.999, p * 1.001, 500, 500) if p else None

    # 50c of cash cannot buy a dollar of stock, so refuse rather than create
    # an unsellable fractional crumb.
    b = SimulatedBroker(0.50, quotes)
    with pytest.raises(InsufficientFunds):
        b.submit(OrderIntent("AAPL", "buy", notional=1000))
    assert b.positions() == []


def test_no_sequence_of_orders_can_overdraw(broker, prices):
    for _ in range(40):
        for sym in ("AAPL", "MSFT"):
            with contextlib.suppress(InsufficientFunds):
                broker.submit(OrderIntent(sym, "buy", notional=900))
            assert broker.account().cash >= -1e-9, "account went negative"
    assert broker.account().equity > 0


def test_no_quote_is_rejected(broker):
    with pytest.raises(OrderRejected, match="no tradable quote"):
        broker.submit(OrderIntent("ZZZZ", "buy", notional=100))


# --------------------------------------------------------------------------- #
# limit orders
# --------------------------------------------------------------------------- #


def test_limit_buy_rests_until_the_ask_comes_to_it(broker, prices):
    o = broker.submit(OrderIntent("AAPL", "buy", qty=5, order_type="limit",
                                  limit_price=95.0))
    assert o.status is OrderStatus.NEW and o.filled_qty == 0
    assert broker.account().buying_power < broker.account().cash   # cash committed
    prices["AAPL"] = 94.0
    broker.sync()
    assert o.status is OrderStatus.FILLED
    assert o.filled_avg_price <= 95.0        # price improvement passed through


def test_limit_sell_rests_until_the_bid_reaches_it(broker, prices):
    broker.submit(OrderIntent("AAPL", "buy", qty=10))
    o = broker.submit(OrderIntent("AAPL", "sell", qty=5, order_type="limit",
                                  limit_price=120.0))
    assert o.status is OrderStatus.NEW
    prices["AAPL"] = 130.0
    broker.sync()
    assert o.status is OrderStatus.FILLED
    assert o.filled_avg_price >= 120.0


def test_an_immediately_crossing_limit_fills_at_once(broker):
    o = broker.submit(OrderIntent("AAPL", "buy", qty=5, order_type="limit",
                                  limit_price=200.0))
    assert o.status is OrderStatus.FILLED
    assert o.filled_avg_price == pytest.approx(100.1, rel=1e-3)   # pays the ask


def test_cancel_and_cancel_all(broker):
    a = broker.submit(OrderIntent("AAPL", "buy", qty=1, order_type="limit",
                                  limit_price=50.0))
    b = broker.submit(OrderIntent("MSFT", "buy", qty=1, order_type="limit",
                                  limit_price=200.0))
    assert len(broker.open_orders()) == 2
    broker.cancel(a.id)
    assert [o.id for o in broker.open_orders()] == [b.id]
    assert broker.cancel_all() == 1
    assert broker.open_orders() == []


def test_day_orders_expire_at_the_close(broker):
    broker.submit(OrderIntent("AAPL", "buy", qty=1, order_type="limit",
                              limit_price=50.0, tif="day"))
    broker.submit(OrderIntent("MSFT", "buy", qty=1, order_type="limit",
                              limit_price=100.0, tif="gtc"))
    assert broker.end_of_day() == 1
    assert len(broker.open_orders()) == 1
    assert broker.open_orders()[0].tif is TimeInForce.GTC


def test_stale_orders_can_be_expired_by_ttl(broker):
    broker.submit(OrderIntent("AAPL", "buy", qty=1, order_type="limit",
                              limit_price=50.0))
    assert broker.expire_stale_orders(9999) == 0
    broker._test_now["t"] = broker._test_now["t"] + timedelta(seconds=200)
    assert broker.expire_stale_orders(120) == 1


# --------------------------------------------------------------------------- #
# bulk actions and lifecycle
# --------------------------------------------------------------------------- #


def test_close_position_and_close_all(broker):
    broker.submit(OrderIntent("AAPL", "buy", notional=1000))
    broker.submit(OrderIntent("MSFT", "buy", notional=1000))
    assert len(broker.positions()) == 2
    broker.close_position("AAPL")
    assert [p.symbol for p in broker.positions()] == ["MSFT"]
    broker.close_all_positions()
    assert broker.positions() == []


def test_close_all_cancels_working_orders(broker):
    broker.submit(OrderIntent("AAPL", "buy", qty=1, order_type="limit",
                              limit_price=50.0))
    broker.submit(OrderIntent("MSFT", "buy", notional=500))
    broker.close_all_positions(cancel_orders=True)
    assert broker.open_orders() == []
    assert broker.positions() == []


def test_reset_for_round_restores_the_bankroll(broker):
    broker.submit(OrderIntent("AAPL", "buy", notional=3000))
    broker.reset_for_round(5000.0)
    acct = broker.account()
    assert acct.cash == 5000.0 and acct.equity == 5000.0
    assert acct.positions == () and broker.fills == []
    assert broker.realized_pl == 0.0


def test_whole_share_mode_floors_quantities(prices):
    def quotes(sym):
        p = prices.get(sym)
        return Quote(sym, utcnow(), p * 0.999, p * 1.001, 500, 500) if p else None

    b = SimulatedBroker(5000.0, quotes, config=SimConfig(fractional=False))
    o = b.submit(OrderIntent("MSFT", "buy", notional=1000))     # $400 each
    assert o.filled_qty == 2.0


def test_participation_cap_produces_partial_fills(prices):
    def quotes(sym):
        return Quote(sym, utcnow(), 99.9, 100.1, 50, 50)

    b = SimulatedBroker(50_000.0, quotes,
                        config=SimConfig(max_participation=0.5))
    o = b.submit(OrderIntent("AAPL", "buy", qty=100))
    assert o.status is OrderStatus.PARTIALLY_FILLED
    assert o.filled_qty == pytest.approx(25)
    b.sync()
    assert o.filled_qty == pytest.approx(50)


def test_summary_reports_the_round(broker):
    broker.submit(OrderIntent("AAPL", "buy", notional=1000))
    s = broker.summary()
    assert s["fills"] == 1 and "AAPL" in s["positions"]
    assert s["equity"] == pytest.approx(broker.equity, abs=0.01)   # summary rounds


# --------------------------------------------------------------------------- #
# Alpaca wire-format parsing
# --------------------------------------------------------------------------- #


def test_timestamp_parsing_handles_nanoseconds_and_offsets():
    a = _ts("2026-09-17T13:45:00.123456789Z")
    assert (a.year, a.hour, a.minute, a.microsecond) == (2026, 13, 45, 123456)
    b = _ts("2026-09-17T09:45:00.123456789-04:00")
    assert b.utcoffset().total_seconds() == -4 * 3600
    assert _ts("2026-09-17T13:45:00Z").tzinfo is not None
    assert _ts(None) is not None and _ts("") is not None
    assert _ts("garbage") is not None       # never raises


def test_float_coercion_is_total():
    assert _f(None) == 0.0 and _f("") == 0.0 and _f("abc") == 0.0
    assert _f("3.5") == 3.5 and _f(7) == 7.0
    assert _f(None, 1.5) == 1.5


def test_order_parsing():
    o = parse_order({
        "id": "x", "client_order_id": "c", "symbol": "aapl", "side": "buy",
        "qty": "3", "filled_qty": "1.5", "filled_avg_price": "200.5",
        "type": "limit", "limit_price": "199.9", "time_in_force": "day",
        "status": "partially_filled", "submitted_at": "2026-09-17T13:00:00Z",
    })
    assert o.symbol == "AAPL" and o.side is Side.BUY
    assert o.status is OrderStatus.PARTIALLY_FILLED
    assert o.order_type is OrderType.LIMIT and o.limit_price == 199.9
    assert o.leaves_qty == pytest.approx(1.5)
    assert o.notional_filled == pytest.approx(1.5 * 200.5)


@pytest.mark.parametrize("raw,expected", [
    ("new", OrderStatus.NEW), ("accepted", OrderStatus.NEW),
    ("filled", OrderStatus.FILLED), ("canceled", OrderStatus.CANCELED),
    ("rejected", OrderStatus.REJECTED), ("expired", OrderStatus.EXPIRED),
    ("pending_new", OrderStatus.PENDING), ("something_new", OrderStatus.PENDING),
])
def test_order_status_mapping(raw, expected):
    assert parse_order({"status": raw, "side": "buy"}).status is expected


def test_bar_quote_and_position_parsing():
    bar = parse_bar("AAPL", {"t": "2026-09-17T13:00:00Z", "o": 1, "h": 2, "l": 0.5,
                             "c": 1.5, "v": 100, "n": 5, "vw": 1.4})
    assert (bar.open, bar.high, bar.low, bar.close) == (1.0, 2.0, 0.5, 1.5)
    assert bar.dollar_volume == pytest.approx(140.0)

    q = parse_quote("AAPL", {"t": "2026-09-17T13:00:00Z", "bp": 10, "ap": 10.1,
                             "bs": 3, "as": 7})
    assert q.mid == pytest.approx(10.05)
    assert q.microprice == pytest.approx((10 * 7 + 10.1 * 3) / 10)
    assert q.imbalance == pytest.approx((3 - 7) / 10)
    assert q.is_sane

    p = parse_position({"symbol": "aapl", "qty": "10", "avg_entry_price": "100",
                        "current_price": "110"})
    assert p.symbol == "AAPL" and p.unrealized_pl == pytest.approx(100.0)
    assert p.unrealized_plpc == pytest.approx(0.1)


def test_crossed_and_empty_quotes_are_not_sane():
    assert not Quote("A", utcnow(), 10.0, 9.0).is_sane
    assert not Quote("A", utcnow(), 0.0, 0.0).is_sane
    assert Quote("A", utcnow(), 10.0, 10.0).is_sane
