"""Pre-trade risk checks. Every rejection path, plus the invariants that matter.

The two properties worth proving:
  * a team can never trade outside its assigned universe (Round 3 fairness);
  * a long-only account can never be pushed short or overdrawn, no matter
    what a strategy asks for.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from competition.data.calendar import MarketCalendar
from competition.data.snapshot import MarketSnapshot
from competition.engine.guardrails import Guardrails, RiskState
from competition.types import (
    UTC,
    Account,
    Bar,
    Order,
    OrderIntent,
    OrderStatus,
    OrderType,
    Position,
    Quote,
    RejectReason,
    Side,
)

SYMS = ("AAPL", "MSFT", "GOOGL")
OPEN_TS = datetime(2026, 9, 10, 17, 0, tzinfo=UTC)      # 13:00 ET, mid-session


@pytest.fixture
def cal():
    return MarketCalendar()


@pytest.fixture
def snap(cal):
    bars = {s: (Bar(s, OPEN_TS, 100, 101, 99, 100.0, 1e6, 500, 100.0),) for s in SYMS}
    quotes = {s: Quote(s, OPEN_TS, 99.95, 100.05, 500, 500) for s in SYMS}
    return MarketSnapshot(ts=OPEN_TS, session=cal.session(OPEN_TS), universe=SYMS,
                          bars=bars, quotes=quotes)


@pytest.fixture
def account():
    return Account(cash=2000.0, equity=5000.0, buying_power=2000.0,
                   positions=(Position("AAPL", 30, 100.0, 100.0),))


@pytest.fixture
def rails(cfg):
    return Guardrails(cfg.risk, bankroll=cfg.starting_cash)


@pytest.fixture
def state():
    s = RiskState("t")
    s.roll_session(date(2026, 9, 10), 5000.0)
    return s


def check(rails, state, account, snap, intents, universe=SYMS, orders=()):
    return rails.validate(intents, state=state, account=account, snapshot=snap,
                          universe=universe, open_orders=orders, now=OPEN_TS)


def only_reason(result) -> RejectReason:
    assert len(result.rejected) == 1, result.rejected
    return result.rejected[0].reason


# --------------------------------------------------------------------------- #
# the happy path
# --------------------------------------------------------------------------- #


def test_a_legal_order_passes(rails, state, account, snap):
    r = check(rails, state, account, snap, [OrderIntent("MSFT", "buy", notional=500)])
    assert r.counts == (1, 0)
    assert state.orders_today == 1


def test_a_legal_limit_order_passes(rails, state, account, snap):
    r = check(rails, state, account, snap,
              [OrderIntent("MSFT", "buy", qty=1, order_type="limit", limit_price=99.5)])
    assert r.counts == (1, 0)


# --------------------------------------------------------------------------- #
# universe enforcement -- the Round 3 fairness rail
# --------------------------------------------------------------------------- #


def test_trading_outside_the_universe_is_refused(rails, state, account, snap):
    r = check(rails, state, account, snap, [OrderIntent("TSLA", "buy", notional=500)])
    assert only_reason(r) is RejectReason.OUTSIDE_UNIVERSE


def test_a_narrower_universe_is_enforced(rails, state, account, snap):
    r = check(rails, state, account, snap,
              [OrderIntent("MSFT", "buy", notional=500)], universe=("AAPL",))
    assert only_reason(r) is RejectReason.OUTSIDE_UNIVERSE


def test_snapshot_restriction_hides_other_symbols(snap):
    view = snap.restricted_to(["AAPL"])
    assert view.universe == ("AAPL",)
    assert set(view.bars) == {"AAPL"}
    assert view.quote("MSFT") is None
    assert view.price("MSFT") == 0.0


# --------------------------------------------------------------------------- #
# long-only, never overdrawn
# --------------------------------------------------------------------------- #


def test_selling_what_is_not_held_is_refused(rails, state, account, snap):
    r = check(rails, state, account, snap, [OrderIntent("MSFT", "sell", qty=10)])
    assert only_reason(r) is RejectReason.OVERSELL


def test_overselling_is_trimmed_to_flat(rails, state, account, snap):
    r = check(rails, state, account, snap, [OrderIntent("AAPL", "sell", qty=1000)])
    assert r.counts == (1, 0)
    assert r.accepted[0].qty == pytest.approx(30)
    assert "trimmed" in r.accepted[0].reason


def test_cumulative_sells_cannot_exceed_the_position(rails, state, account, snap):
    intents = [
        OrderIntent("AAPL", "sell", qty=20, order_type="limit", limit_price=100.1,
                    reduce_only=True),
        OrderIntent("AAPL", "sell", qty=20, order_type="limit", limit_price=100.2,
                    reduce_only=True),
        OrderIntent("AAPL", "sell", qty=20, order_type="limit", limit_price=100.3,
                    reduce_only=True),
    ]
    r = check(rails, state, account, snap, intents)
    assert sum(o.qty for o in r.accepted) == pytest.approx(30)
    assert any(x.reason is RejectReason.OVERSELL for x in r.rejected)


def test_resting_sells_count_against_the_position(rails, state, account, snap):
    resting = Order(id="r", client_order_id="c", symbol="AAPL", side=Side.SELL,
                    qty=25, order_type=OrderType.LIMIT, limit_price=100.5,
                    status=OrderStatus.NEW, submitted_at=OPEN_TS)
    r = check(rails, state, account, snap,
              [OrderIntent("AAPL", "sell", qty=10, order_type="limit",
                           limit_price=100.4, reduce_only=True)], orders=[resting])
    assert r.accepted[0].qty == pytest.approx(5)


def test_buying_past_cash_is_refused(rails, state, account, snap):
    r = check(rails, state, account, snap, [OrderIntent("MSFT", "buy", notional=2500)])
    assert only_reason(r) is RejectReason.INSUFFICIENT_CASH


def test_a_batch_cannot_collectively_overspend(rails, state, account, snap):
    intents = [OrderIntent(s, "buy", notional=900) for s in ("MSFT", "GOOGL")] + \
              [OrderIntent("MSFT", "buy", qty=9, order_type="limit", limit_price=100.0)]
    r = check(rails, state, account, snap, intents)
    spent = sum(o.notional or o.qty * 100.0 for o in r.accepted)
    assert spent <= 2000.0 + 1e-6


def test_working_buy_orders_reserve_cash(rails, state, account, snap):
    resting = Order(id="r", client_order_id="c", symbol="MSFT", side=Side.BUY,
                    qty=15, order_type=OrderType.LIMIT, limit_price=100.0,
                    status=OrderStatus.NEW, submitted_at=OPEN_TS)
    r = check(rails, state, account, snap,
              [OrderIntent("GOOGL", "buy", notional=900)], orders=[resting])
    assert only_reason(r) is RejectReason.INSUFFICIENT_CASH


def test_market_sells_fund_later_buys_in_the_same_tick(rails, state, snap):
    account = Account(cash=100.0, equity=5000.0, buying_power=100.0,
                      positions=(Position("AAPL", 30, 100.0, 100.0),))
    r = check(rails, state, account, snap, [
        OrderIntent("AAPL", "sell", qty=30, reduce_only=True),
        OrderIntent("MSFT", "buy", notional=1500),
    ])
    assert r.counts == (2, 0)


def test_resting_limit_sells_do_not_fund_buys(rails, state, snap):
    account = Account(cash=100.0, equity=5000.0, buying_power=100.0,
                      positions=(Position("AAPL", 30, 100.0, 100.0),))
    r = check(rails, state, account, snap, [
        OrderIntent("AAPL", "sell", qty=30, order_type="limit", limit_price=100.5,
                    reduce_only=True),
        OrderIntent("MSFT", "buy", notional=1500),
    ])
    assert len(r.accepted) == 1 and r.accepted[0].side is Side.SELL
    assert only_reason(r) is RejectReason.INSUFFICIENT_CASH


# --------------------------------------------------------------------------- #
# caps
# --------------------------------------------------------------------------- #


def test_position_cap(rails, state, snap, cfg):
    tight = Guardrails(type(cfg.risk)(**{**cfg.risk.__dict__, "max_position_pct": 0.50}),
                       bankroll=cfg.starting_cash)
    account = Account(cash=3000.0, equity=5000.0, buying_power=3000.0,
                      positions=(Position("AAPL", 24, 100.0, 100.0),))
    r = tight.validate([OrderIntent("AAPL", "buy", notional=1000)], state=state,
                       account=account, snapshot=snap, universe=SYMS, now=OPEN_TS)
    assert only_reason(r) is RejectReason.POSITION_CAP


def test_leverage_cap(rails, state, snap, cfg):
    account = Account(cash=4000.0, equity=5000.0, buying_power=4000.0,
                      positions=(Position("AAPL", 45, 100.0, 100.0),))
    r = check(rails, state, account, snap, [OrderIntent("MSFT", "buy", notional=1000)])
    assert r.rejected and r.rejected[0].reason in (
        RejectReason.LEVERAGE_CAP, RejectReason.POSITION_CAP)


def test_notional_bounds(rails, state, account, snap):
    assert only_reason(check(rails, state, account, snap,
                             [OrderIntent("MSFT", "buy", notional=0.5)])
                       ) is RejectReason.MIN_NOTIONAL
    assert only_reason(check(rails, state, account, snap,
                             [OrderIntent("MSFT", "buy", notional=9000)])
                       ) is RejectReason.MAX_NOTIONAL


def test_a_full_position_close_is_exempt_from_the_minimum(rails, state, snap):
    """Dust must always be closable or it can never be cleared."""
    account = Account(cash=100.0, equity=100.0, buying_power=100.0,
                      positions=(Position("AAPL", 0.002, 100.0, 100.0),))
    r = check(rails, state, account, snap,
              [OrderIntent("AAPL", "sell", qty=0.002, reduce_only=True)])
    assert r.counts == (1, 0)


def test_a_partial_sell_below_the_minimum_is_still_refused(rails, state, snap):
    account = Account(cash=0.0, equity=3000.0, buying_power=0.0,
                      positions=(Position("AAPL", 30, 100.0, 100.0),))
    r = check(rails, state, account, snap,
              [OrderIntent("AAPL", "sell", qty=0.005, reduce_only=True)])
    assert only_reason(r) is RejectReason.MIN_NOTIONAL


def test_order_rate_limits(rails, state, account, snap, cfg):
    intents = [OrderIntent("MSFT", "buy", qty=1, order_type="limit",
                           limit_price=99.0 + i * 0.01)
               for i in range(cfg.risk.max_orders_per_tick + 5)]
    r = check(rails, state, account, snap, intents)
    assert len(r.accepted) <= cfg.risk.max_orders_per_tick
    assert any(x.reason is RejectReason.ORDER_RATE_LIMIT for x in r.rejected)


# --------------------------------------------------------------------------- #
# data quality
# --------------------------------------------------------------------------- #


def test_stale_quotes_block_entries_but_not_exits(rails, state, cal):
    stale = Quote("AAPL", OPEN_TS - timedelta(seconds=600), 99.95, 100.05, 500, 500)
    snap = MarketSnapshot(
        ts=OPEN_TS, session=cal.session(OPEN_TS), universe=("AAPL",),
        bars={"AAPL": (Bar("AAPL", OPEN_TS, 100, 101, 99, 100.0, 1e6, 500, 100.0),)},
        quotes={"AAPL": stale},
    )
    account = Account(cash=2000.0, equity=5000.0, buying_power=2000.0,
                      positions=(Position("AAPL", 30, 100.0, 100.0),))
    entry = check(rails, state, account, snap, [OrderIntent("AAPL", "buy", notional=100)],
                  universe=("AAPL",))
    assert only_reason(entry) is RejectReason.STALE_QUOTE
    exit_ = check(rails, state, account, snap,
                  [OrderIntent("AAPL", "sell", qty=5, reduce_only=True)],
                  universe=("AAPL",))
    assert exit_.counts == (1, 0)


def test_crossed_quotes_are_refused(rails, state, account, cal):
    snap = MarketSnapshot(
        ts=OPEN_TS, session=cal.session(OPEN_TS), universe=("AAPL",),
        bars={"AAPL": (Bar("AAPL", OPEN_TS, 100, 101, 99, 100.0, 1e6, 500, 100.0),)},
        quotes={"AAPL": Quote("AAPL", OPEN_TS, 101.0, 99.0, 500, 500)},
    )
    r = check(rails, state, account, snap, [OrderIntent("AAPL", "buy", notional=100)],
              universe=("AAPL",))
    assert only_reason(r) is RejectReason.CROSSED_QUOTE


def test_missing_price_is_refused(rails, state, account, cal):
    snap = MarketSnapshot(ts=OPEN_TS, session=cal.session(OPEN_TS), universe=("AAPL",))
    r = check(rails, state, account, snap, [OrderIntent("AAPL", "buy", notional=100)],
              universe=("AAPL",))
    assert only_reason(r) is RejectReason.NO_QUOTE


def test_nan_sizes_are_refused(rails, state, account, snap):
    assert only_reason(check(rails, state, account, snap,
                             [OrderIntent("MSFT", "buy", notional=float("nan"))])
                       ) is RejectReason.NAN_SIZE
    assert only_reason(check(rails, state, account, snap,
                             [OrderIntent("MSFT", "buy", qty=-5)])
                       ) is RejectReason.NAN_SIZE


def test_fat_finger_limits_are_refused(rails, state, account, snap):
    assert only_reason(check(rails, state, account, snap,
                             [OrderIntent("MSFT", "buy", qty=1, order_type="limit",
                                          limit_price=50.0)])
                       ) is RejectReason.LIMIT_TOO_FAR


def test_duplicate_intents_are_collapsed(rails, state, account, snap):
    r = check(rails, state, account, snap,
              [OrderIntent("MSFT", "buy", notional=100)] * 3)
    assert len(r.accepted) == 1
    assert sum(1 for x in r.rejected if x.reason is RejectReason.DUPLICATE_INTENT) == 2


def test_closed_market_blocks_everything(rails, state, account, cal):
    closed = datetime(2026, 9, 12, 17, 0, tzinfo=UTC)          # Saturday
    snap = MarketSnapshot(
        ts=closed, session=cal.session(closed), universe=("AAPL",),
        bars={"AAPL": (Bar("AAPL", closed, 100, 101, 99, 100.0, 1e6, 500, 100.0),)},
        quotes={"AAPL": Quote("AAPL", closed, 99.95, 100.05, 500, 500)},
    )
    r = rails.validate([OrderIntent("AAPL", "buy", notional=100)], state=state,
                       account=account, snapshot=snap, universe=("AAPL",), now=closed)
    assert only_reason(r) is RejectReason.MARKET_CLOSED


def test_untradable_assets_are_refused(state, account, snap, cfg):
    rails = Guardrails(cfg.risk, bankroll=cfg.starting_cash, is_tradable=lambda s: s != "MSFT")
    r = check(rails, state, account, snap, [OrderIntent("MSFT", "buy", notional=100)])
    assert only_reason(r) is RejectReason.ASSET_NOT_TRADABLE


# --------------------------------------------------------------------------- #
# kill switch
# --------------------------------------------------------------------------- #


def test_kill_switch_trips_and_blocks(rails, state, snap, cfg):
    crashed = Account(cash=100.0, equity=5000.0 * (1 - cfg.risk.daily_loss_kill_switch_pct)
                      - 1.0, buying_power=100.0, positions=())
    msg = rails.check_kill_switch(state, crashed, today=date(2026, 9, 10))
    assert msg and "kill switch" in msg
    assert state.is_halted
    r = check(rails, state, crashed, snap, [OrderIntent("AAPL", "buy", notional=50)])
    assert only_reason(r) is RejectReason.KILL_SWITCH


def test_kill_switch_does_not_trip_above_the_threshold(rails, state, cfg):
    ok = Account(cash=100.0, equity=5000.0 * 0.9, buying_power=100.0, positions=())
    assert rails.check_kill_switch(state, ok, today=date(2026, 9, 10)) is None
    assert not state.is_halted


def test_kill_switch_clears_on_the_next_session(rails, state, cfg):
    crashed = Account(cash=0.0, equity=1000.0, buying_power=0.0, positions=())
    rails.check_kill_switch(state, crashed, today=date(2026, 9, 10))
    assert state.is_halted
    state.roll_session(date(2026, 9, 11), 1000.0)
    assert not state.is_halted


def test_session_roll_resets_the_daily_order_count(state):
    state.orders_today = 500
    assert state.roll_session(date(2026, 9, 11), 5000.0)
    assert state.orders_today == 0
    assert not state.roll_session(date(2026, 9, 11), 5000.0)


def test_rejections_are_tallied(rails, state, account, snap):
    check(rails, state, account, snap, [OrderIntent("TSLA", "buy", notional=100)])
    check(rails, state, account, snap, [OrderIntent("MSFT", "sell", qty=1)])
    tally = state.summary()["rejections"]
    assert tally["outside_universe"] == 1 and tally["oversell"] == 1
