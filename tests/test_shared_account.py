"""Teams sharing one real broker account.

With nine real accounts, Alpaca guarantees one team cannot spend another's
money. With a shared account that guarantee is *this code*, so these tests
are the guarantee. The properties that must hold:

  * a team can only ever spend its own cash and sell its own shares;
  * no per-team operation can liquidate or cancel another team's book;
  * the sum of the virtual books reconciles against the real account;
  * opposing orders are crossed internally with cash conserved exactly,
    because Alpaca rejects them as wash trades inside one account;
  * the split survives a crash -- it exists nowhere but in this process.
"""

from __future__ import annotations

import contextlib

import pytest

from competition.broker.base import BrokerError, InsufficientFunds, OrderRejected
from competition.broker.shared import (
    SharedAccount,
    VirtualBook,
    tag_for,
    team_from_tag,
)
from competition.broker.simulated import SimConfig, SimulatedBroker
from competition.types import OrderIntent, Quote, Side, utcnow

TEAMS = ["alpha", "beta", "gamma"]
BANKROLL = 5000.0


@pytest.fixture
def prices():
    return {"AAPL": 100.0, "MSFT": 400.0, "GOOGL": 200.0}


@pytest.fixture
def shared(prices):
    """Three teams sharing one simulated account of 3 x $5,000."""
    def quote_of(symbol):
        p = prices.get(symbol)
        return Quote(symbol, utcnow(), p * 0.999, p * 1.001, 500, 500) if p else None

    real = SimulatedBroker(BANKROLL * len(TEAMS), quote_of, name="group-a",
                           config=SimConfig(slippage_bps=0.0))
    return SharedAccount(real, TEAMS, bankroll=BANKROLL,
                         price_of=lambda s: prices.get(s, 0.0),
                         quote_of=quote_of, name="group-a")


def books(shared):
    return {k: shared.virtual_broker(k) for k in TEAMS}


def buy(shared, team, symbol, notional):
    order = shared.stage(team, OrderIntent(symbol, "buy", notional=notional))
    shared.flush()
    shared.poll_fills()
    return order


# --------------------------------------------------------------------------- #
# tagging
# --------------------------------------------------------------------------- #


def test_client_order_id_round_trips_the_team():
    for key in ("alpha", "mean_reverter", "q_learner", "vol_breakout"):
        assert team_from_tag(tag_for(key)) == key


def test_foreign_client_order_ids_are_not_claimed():
    assert team_from_tag("some-other-system-id") is None
    assert team_from_tag("") is None
    assert team_from_tag("cmp-nohex") is None


# --------------------------------------------------------------------------- #
# each team starts with its own bankroll
# --------------------------------------------------------------------------- #


def test_every_book_starts_at_the_bankroll(shared):
    for broker in books(shared).values():
        account = broker.account()
        assert account.cash == pytest.approx(BANKROLL)
        assert account.equity == pytest.approx(BANKROLL)
        assert account.positions == ()


def test_required_capital_is_the_sum(shared):
    assert shared.required_capital() == pytest.approx(BANKROLL * len(TEAMS))


def test_the_real_account_holds_the_pool(shared):
    assert shared.broker.account().cash == pytest.approx(BANKROLL * len(TEAMS))


# --------------------------------------------------------------------------- #
# isolation: the whole point
# --------------------------------------------------------------------------- #


def test_a_team_cannot_spend_more_than_its_own_cash(shared, prices):
    """The real account holds 3x, but each team may only reach its own share.

    An oversized order is clipped to what the team can afford rather than
    rejected -- the same behaviour as a real broker -- so the guarantee is
    tested on the resulting size, not on an exception.
    """
    order = shared.stage("alpha", OrderIntent("AAPL", "buy",
                                              notional=BANKROLL * 2))
    spend = order.qty * prices["AAPL"]
    assert spend <= BANKROLL + 1e-6, f"alpha reached ${spend:,.2f} of a $5,000 book"
    shared.flush()
    shared.poll_fills()
    assert books(shared)["alpha"].account().cash >= -1e-6
    # And the other teams are untouched.
    for other in ("beta", "gamma"):
        assert books(shared)[other].account().cash == pytest.approx(BANKROLL)


def test_a_team_with_no_cash_is_refused(shared):
    buy(shared, "alpha", "AAPL", BANKROLL)
    assert shared.available_cash("alpha") < 1.0
    with pytest.raises(InsufficientFunds):
        shared.stage("alpha", OrderIntent("MSFT", "buy", notional=1000))


def test_no_sequence_of_orders_can_overdraw_a_book(shared):
    for _ in range(30):
        for team in TEAMS:
            for symbol in ("AAPL", "MSFT", "GOOGL"):
                with contextlib.suppress(InsufficientFunds, OrderRejected):
                    shared.stage(team, OrderIntent(symbol, "buy", notional=900))
        shared.flush()
        shared.poll_fills()
        for team in TEAMS:
            assert shared.books[team].cash >= -1e-6, f"{team} went negative"
    assert shared.reconcile().ok, shared.reconcile().describe()


def test_spending_by_one_team_does_not_reduce_another(shared):
    buy(shared, "alpha", "AAPL", 4000)
    assert books(shared)["alpha"].account().cash == pytest.approx(1000, abs=10)
    for other in ("beta", "gamma"):
        assert books(shared)[other].account().cash == pytest.approx(BANKROLL)


def test_a_team_cannot_sell_shares_it_does_not_hold(shared):
    buy(shared, "alpha", "AAPL", 2000)
    assert books(shared)["alpha"].account().qty("AAPL") > 0
    with pytest.raises(OrderRejected, match="holds"):
        shared.stage("beta", OrderIntent("AAPL", "sell", qty=5))


def test_a_team_cannot_oversell_its_own_position(shared):
    buy(shared, "alpha", "AAPL", 1000)
    held = books(shared)["alpha"].account().qty("AAPL")
    order = shared.stage("alpha", OrderIntent("AAPL", "sell", qty=held * 10))
    assert order.qty == pytest.approx(held, rel=1e-6)


def test_positions_are_per_team(shared):
    buy(shared, "alpha", "AAPL", 1000)
    buy(shared, "beta", "MSFT", 1000)
    assert [p.symbol for p in books(shared)["alpha"].positions()] == ["AAPL"]
    assert [p.symbol for p in books(shared)["beta"].positions()] == ["MSFT"]
    assert books(shared)["gamma"].positions() == []
    # The real account holds both.
    assert {p.symbol for p in shared.broker.positions()} == {"AAPL", "MSFT"}


def test_two_teams_can_hold_the_same_symbol_independently(shared):
    buy(shared, "alpha", "AAPL", 1000)
    buy(shared, "beta", "AAPL", 2000)
    a = books(shared)["alpha"].account().qty("AAPL")
    b = books(shared)["beta"].account().qty("AAPL")
    assert b == pytest.approx(2 * a, rel=1e-3)
    real = shared.broker.account().qty("AAPL")
    assert real == pytest.approx(a + b, rel=1e-6)


def test_closing_all_positions_touches_only_that_team(shared):
    """The dangerous one: the broker's own close_all would wipe everyone."""
    buy(shared, "alpha", "AAPL", 1000)
    buy(shared, "beta", "MSFT", 1000)
    buy(shared, "gamma", "GOOGL", 1000)
    books(shared)["beta"].close_all_positions()
    shared.flush()
    shared.poll_fills()
    assert books(shared)["beta"].positions() == []
    assert books(shared)["alpha"].positions(), "alpha was liquidated too"
    assert books(shared)["gamma"].positions(), "gamma was liquidated too"


def test_cancel_all_touches_only_that_team(shared, prices):
    shared.stage("alpha", OrderIntent("AAPL", "buy", qty=1,
                                      order_type="limit", limit_price=50.0))
    shared.stage("beta", OrderIntent("AAPL", "buy", qty=1,
                                     order_type="limit", limit_price=51.0))
    shared.flush()
    assert len(books(shared)["alpha"].open_orders()) == 1
    assert len(books(shared)["beta"].open_orders()) == 1
    books(shared)["alpha"].cancel_all()
    assert books(shared)["alpha"].open_orders() == []
    assert len(books(shared)["beta"].open_orders()) == 1, "beta's order was cancelled"


def test_a_team_cannot_cancel_another_teams_order(shared):
    shared.stage("beta", OrderIntent("AAPL", "buy", qty=1,
                                     order_type="limit", limit_price=50.0))
    shared.flush()
    victim = books(shared)["beta"].open_orders()[0]
    books(shared)["alpha"].cancel(victim.id)
    assert len(books(shared)["beta"].open_orders()) == 1


def test_open_orders_are_scoped_to_the_team(shared):
    shared.stage("alpha", OrderIntent("AAPL", "buy", qty=1,
                                      order_type="limit", limit_price=50.0))
    shared.stage("beta", OrderIntent("MSFT", "buy", qty=1,
                                     order_type="limit", limit_price=200.0))
    shared.flush()
    assert {o.symbol for o in books(shared)["alpha"].open_orders()} == {"AAPL"}
    assert {o.symbol for o in books(shared)["beta"].open_orders()} == {"MSFT"}
    assert books(shared)["gamma"].open_orders() == []


def test_staged_and_resting_orders_reserve_only_that_teams_cash(shared):
    shared.stage("alpha", OrderIntent("AAPL", "buy", notional=4000))
    assert shared.available_cash("alpha") == pytest.approx(1000, abs=20)
    assert shared.available_cash("beta") == pytest.approx(BANKROLL)


# --------------------------------------------------------------------------- #
# internal crossing
# --------------------------------------------------------------------------- #


def test_opposing_orders_are_crossed_internally(shared, prices):
    """Alpaca would reject these as a wash trade, so they never reach it."""
    buy(shared, "alpha", "AAPL", 2000)
    held = books(shared)["alpha"].account().qty("AAPL")
    before_orders = len(shared._owner)

    shared.stage("alpha", OrderIntent("AAPL", "sell", qty=held))
    shared.stage("beta", OrderIntent("AAPL", "buy", qty=held))
    shared.flush()
    shared.poll_fills()

    # Nothing new went to the broker: the two teams traded with each other.
    assert len(shared._owner) == before_orders
    assert books(shared)["alpha"].account().qty("AAPL") == pytest.approx(0, abs=1e-6)
    assert books(shared)["beta"].account().qty("AAPL") == pytest.approx(held, rel=1e-6)
    assert shared.crossed_total > 0


def test_crossing_conserves_cash_exactly(shared, prices):
    """The mid is the only price at which the books still reconcile."""
    buy(shared, "alpha", "AAPL", 2000)
    total_before = sum(b.cash for b in shared.books.values())
    held = books(shared)["alpha"].account().qty("AAPL")

    shared.stage("alpha", OrderIntent("AAPL", "sell", qty=held))
    shared.stage("beta", OrderIntent("AAPL", "buy", qty=held))
    shared.flush()
    shared.poll_fills()

    total_after = sum(b.cash for b in shared.books.values())
    assert total_after == pytest.approx(total_before, abs=1e-6)
    assert shared.reconcile().ok, shared.reconcile().describe()


def test_crossing_is_pro_rata_not_first_come(shared, prices):
    """Order-independence matters: the engine deliberately rotates team order."""
    buy(shared, "alpha", "AAPL", 3000)
    held = books(shared)["alpha"].account().qty("AAPL")
    # alpha sells everything; beta and gamma each want half of it.
    shared.stage("alpha", OrderIntent("AAPL", "sell", qty=held))
    shared.stage("beta", OrderIntent("AAPL", "buy", qty=held / 2))
    shared.stage("gamma", OrderIntent("AAPL", "buy", qty=held / 2))
    shared.flush()
    shared.poll_fills()
    b = books(shared)["beta"].account().qty("AAPL")
    g = books(shared)["gamma"].account().qty("AAPL")
    assert b == pytest.approx(g, rel=1e-6), "crossing favoured whoever acted first"


def test_only_the_residual_reaches_the_broker(shared, prices):
    buy(shared, "alpha", "AAPL", 2000)
    held = books(shared)["alpha"].account().qty("AAPL")
    shared.stage("alpha", OrderIntent("AAPL", "sell", qty=held))
    shared.stage("beta", OrderIntent("AAPL", "buy", qty=held * 2))
    shared.flush()
    shared.poll_fills()
    # beta wanted 2x, crossed 1x internally, so 1x was bought from the market.
    assert books(shared)["beta"].account().qty("AAPL") == pytest.approx(
        held * 2, rel=1e-2)
    assert books(shared)["alpha"].account().qty("AAPL") == pytest.approx(0, abs=1e-6)


def test_crossed_volume_is_recorded_per_team(shared):
    buy(shared, "alpha", "AAPL", 2000)
    held = books(shared)["alpha"].account().qty("AAPL")
    shared.stage("alpha", OrderIntent("AAPL", "sell", qty=held))
    shared.stage("beta", OrderIntent("AAPL", "buy", qty=held))
    shared.flush()
    assert shared.books["alpha"].crossed_notional > 0
    assert shared.books["beta"].crossed_notional > 0
    assert shared.summary()["crossed_notional"] > 0


def test_a_limit_away_from_the_mid_is_not_crossed(shared, prices):
    """A bid well below the mid is not willing to pay the mid."""
    buy(shared, "alpha", "AAPL", 2000)
    held = books(shared)["alpha"].account().qty("AAPL")
    shared.stage("alpha", OrderIntent("AAPL", "sell", qty=held))
    shared.stage("beta", OrderIntent("AAPL", "buy", qty=held,
                                     order_type="limit", limit_price=50.0))
    shared.flush()
    shared.poll_fills()
    # beta's lowball bid did not get filled by alpha at the mid.
    assert books(shared)["beta"].account().qty("AAPL") == pytest.approx(0, abs=1e-6)


def test_crossing_can_be_disabled(shared, prices):
    shared.cross_internally = False
    buy(shared, "alpha", "AAPL", 2000)
    held = books(shared)["alpha"].account().qty("AAPL")
    shared.stage("alpha", OrderIntent("AAPL", "sell", qty=held))
    shared.stage("beta", OrderIntent("AAPL", "buy", qty=held))
    orders = shared.flush()
    # Both went to the broker -- which on real Alpaca is the wash-trade path.
    assert len(orders) == 2


# --------------------------------------------------------------------------- #
# reconciliation
# --------------------------------------------------------------------------- #


def test_reconciles_after_ordinary_trading(shared, prices):
    buy(shared, "alpha", "AAPL", 1500)
    buy(shared, "beta", "MSFT", 2500)
    buy(shared, "gamma", "AAPL", 800)
    report = shared.reconcile()
    assert report.ok, report.describe()
    assert "reconciled" in report.describe()


def test_reconciliation_detects_drift(shared):
    buy(shared, "alpha", "AAPL", 1000)
    shared.books["alpha"].cash += 999.0          # inject a discrepancy
    report = shared.reconcile()
    assert not report.ok
    assert report.cash_drift == pytest.approx(999.0, abs=1e-6)
    assert "cash drift" in report.describe()


def test_reconciliation_detects_position_drift(shared):
    buy(shared, "alpha", "AAPL", 1000)
    shared.books["alpha"].lot("AAPL").qty += 5
    report = shared.reconcile()
    assert not report.ok
    assert "AAPL" in report.position_drift


def test_a_sell_never_mints_cash(shared, prices):
    """If attribution ever drifted, a sell must not credit phantom shares."""
    book = shared.books["alpha"]
    before = book.cash
    from competition.types import Order, OrderStatus
    phantom = Order(id="x", client_order_id=tag_for("alpha"), symbol="AAPL",
                    side=Side.SELL, qty=10, filled_qty=10,
                    filled_avg_price=100.0, status=OrderStatus.FILLED)
    shared._apply_fill("alpha", phantom, 10, 100.0)
    assert book.cash == pytest.approx(before), "cash was minted from nothing"


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #


def test_reset_flattens_the_real_account_once_per_wave(shared):
    buy(shared, "alpha", "AAPL", 1000)
    buy(shared, "beta", "MSFT", 1000)
    brokers = books(shared)
    # The engine resets each team in turn.
    for team in TEAMS:
        brokers[team].reset_for_round(BANKROLL)
    assert shared.broker.positions() == [], "the real account was not flattened"
    for team in TEAMS:
        account = brokers[team].account()
        assert account.cash == pytest.approx(BANKROLL)
        assert account.positions == ()


def test_reset_wave_does_not_wipe_a_team_reset_earlier(shared):
    """The bug this guards: flattening on every call kills the first team."""
    brokers = books(shared)
    brokers["alpha"].reset_for_round(BANKROLL)
    buy(shared, "alpha", "AAPL", 1000)
    held = books(shared)["alpha"].account().qty("AAPL")
    assert held > 0
    brokers["beta"].reset_for_round(BANKROLL)
    brokers["gamma"].reset_for_round(BANKROLL)
    # alpha's position is still hers; the wave completed without re-flattening.
    assert books(shared)["alpha"].account().qty("AAPL") == pytest.approx(held)


def test_state_survives_a_round_trip(shared):
    buy(shared, "alpha", "AAPL", 1500)
    buy(shared, "beta", "MSFT", 2000)
    blob = shared.to_dict()

    def quote_of(symbol):
        return Quote(symbol, utcnow(), 99.9, 100.1, 500, 500)

    fresh = SharedAccount(
        SimulatedBroker(BANKROLL * 3, quote_of, config=SimConfig()),
        TEAMS, bankroll=BANKROLL, price_of=lambda s: 100.0, name="group-a",
    )
    fresh.load(blob)
    for team in TEAMS:
        assert fresh.books[team].cash == pytest.approx(shared.books[team].cash)
        assert fresh.books[team].lots.keys() == shared.books[team].lots.keys()
        for symbol, lot in shared.books[team].lots.items():
            assert fresh.books[team].lots[symbol].qty == pytest.approx(lot.qty)


def test_the_split_is_lost_without_persistence(shared):
    """Why the books must be checkpointed: the broker does not know the split."""
    buy(shared, "alpha", "AAPL", 2000)
    real_qty = shared.broker.account().qty("AAPL")
    assert real_qty > 0
    # A fresh SharedAccount over the same broker has no idea whose it is.
    naive = SharedAccount(shared.broker, TEAMS, bankroll=BANKROLL,
                          price_of=lambda s: 100.0)
    assert all(b.qty("AAPL") == 0 for b in naive.books.values())


def test_summary_reports_the_shared_costs(shared):
    buy(shared, "alpha", "AAPL", 1000)
    summary = shared.summary()
    assert summary["teams"] == TEAMS
    assert summary["required_capital"] == pytest.approx(BANKROLL * 3)
    assert "crossed_notional" in summary
    assert "wash_rejections" in summary
    assert "forced_cancels" in summary


def test_unknown_team_is_refused(shared):
    with pytest.raises(BrokerError, match="no virtual book"):
        shared.book("not_a_team")
    with pytest.raises(BrokerError, match="no virtual book"):
        shared.virtual_broker("not_a_team")


def test_a_shared_account_needs_at_least_one_team(prices):
    with pytest.raises(ValueError, match="at least one team"):
        SharedAccount(SimulatedBroker(1000.0, lambda s: None), [],
                      bankroll=BANKROLL, price_of=lambda s: 100.0)


def test_book_serialisation_drops_empty_lots(shared):
    buy(shared, "alpha", "AAPL", 1000)
    held = books(shared)["alpha"].account().qty("AAPL")
    shared.stage("alpha", OrderIntent("AAPL", "sell", qty=held))
    shared.flush()
    shared.poll_fills()
    assert shared.books["alpha"].to_dict()["lots"] == {}


def test_virtual_book_equity_marks_at_the_current_price(prices):
    book = VirtualBook("alpha", cash=1000.0)
    book.lot("AAPL").buy(10, 100.0)
    assert book.equity(lambda s: 100.0) == pytest.approx(2000.0)
    assert book.equity(lambda s: 120.0) == pytest.approx(2200.0)
