"""One Alpaca account, partitioned into several virtual books.

Alpaca caps paper accounts at three per login, so nine teams cannot always get
nine accounts. This module lets a group of teams share one real account while
each keeps its own cash, positions and equity -- one custody account, many
internal books, which is how a multi-strategy desk actually operates.

The honest costs, stated up front
---------------------------------
**Isolation becomes software-enforced.** With nine real accounts, Alpaca
guarantees that one team cannot spend another's money. Here, that guarantee is
this file. It is heavily tested, but it is code rather than a broker.

**Opposing orders must be crossed internally.** Alpaca rejects a buy and a
sell on the same symbol inside one account as a potential wash trade (HTTP
403, "opposite side market/stop order exists") -- and that applies to paper
accounts. With nine teams on Round 1's three symbols, conflicts are the normal
case, not an edge case. So staged intents are netted per symbol each tick and
the overlap is crossed between teams at the **mid**, with only the residual
sent to the market.

Crossing at the mid is the only price that conserves cash exactly: the buyer's
debit equals the seller's credit, so the sum of the virtual books still
reconciles against a real account that saw no trade. The side effect is that
both crossed teams avoid the half-spread they would have paid externally. That
is symmetric and rule-based rather than discretionary, and crossed volume is
recorded per team so it shows up in the reports instead of hiding.

**The market maker is structurally disadvantaged.** When a team's market order
would collide with another team's resting limit, the resting quote is
cancelled so the market order can proceed -- exits must never be blocked. The
scalper is the only team that rests quotes, and it requotes every 20 seconds,
so the cost is small but it is real and it falls on one competitor.

Fewer teams per account means less of all of the above. Three accounts of
three beats one account of nine.
"""

from __future__ import annotations

import contextlib
import logging
import math
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from ..types import (
    Account,
    Fill,
    Order,
    OrderIntent,
    OrderStatus,
    OrderType,
    Position,
    Side,
    utcnow,
)
from .base import Broker, BrokerError, InsufficientFunds, OrderRejected

log = logging.getLogger("competition.shared")

#: `client_order_id` prefix, so a fill can always be traced to a team.
TAG = "cmp"


def tag_for(team_key: str) -> str:
    return f"{TAG}-{team_key}-{uuid.uuid4().hex[:16]}"


def team_from_tag(client_order_id: str) -> str | None:
    """Recover the owning team from a client order id."""
    if not client_order_id or not client_order_id.startswith(f"{TAG}-"):
        return None
    rest = client_order_id[len(TAG) + 1:]
    # Team keys are lower_snake_case and the suffix is hex, so split on the
    # final hyphen.
    if "-" not in rest:
        return None
    return rest.rsplit("-", 1)[0] or None


# --------------------------------------------------------------------------- #


@dataclass
class Lot:
    """A virtual position with a weighted-average cost basis."""

    qty: float = 0.0
    cost: float = 0.0

    @property
    def avg_price(self) -> float:
        return (self.cost / self.qty) if self.qty > 1e-12 else 0.0

    def buy(self, qty: float, price: float) -> None:
        self.qty += qty
        self.cost += qty * price

    def sell(self, qty: float, price: float) -> float:
        """Reduce the lot. Returns realised P&L."""
        qty = min(qty, self.qty)
        if qty <= 0:
            return 0.0
        avg = self.avg_price
        self.cost = max(self.cost - qty * avg, 0.0)
        self.qty = max(self.qty - qty, 0.0)
        if self.qty <= 1e-12:
            self.qty, self.cost = 0.0, 0.0
        return qty * (price - avg)


@dataclass
class VirtualBook:
    """One team's share of a shared account."""

    team_key: str
    cash: float
    lots: dict[str, Lot] = field(default_factory=dict)
    realized_pl: float = 0.0
    #: Notional crossed against another team rather than the market. Recorded
    #: so the effect of internal crossing is visible, not hidden.
    crossed_notional: float = 0.0
    fees: float = 0.0
    #: Cumulative fills, matching `SimulatedBroker.fills`. The engine dedupes
    #: against this by fill identity, so it must accumulate rather than drain.
    fills: list[Fill] = field(default_factory=list)

    def qty(self, symbol: str) -> float:
        lot = self.lots.get(symbol)
        return lot.qty if lot else 0.0

    def lot(self, symbol: str) -> Lot:
        return self.lots.setdefault(symbol, Lot())

    def positions(self, price_of: Callable[[str], float]) -> tuple[Position, ...]:
        out = []
        for symbol, lot in sorted(self.lots.items()):
            if lot.qty <= 1e-12:
                continue
            mark = price_of(symbol) or lot.avg_price
            out.append(Position(symbol=symbol, qty=round(lot.qty, 9),
                                avg_entry_price=lot.avg_price, current_price=mark))
        return tuple(out)

    def equity(self, price_of: Callable[[str], float]) -> float:
        return self.cash + sum(p.market_value for p in self.positions(price_of))

    def to_dict(self) -> dict:
        return {
            "team_key": self.team_key,
            "cash": round(self.cash, 6),
            "realized_pl": round(self.realized_pl, 6),
            "crossed_notional": round(self.crossed_notional, 4),
            "fees": round(self.fees, 6),
            "lots": {s: {"qty": round(l.qty, 9), "cost": round(l.cost, 6)}
                     for s, l in self.lots.items() if l.qty > 1e-12},
        }

    @classmethod
    def from_dict(cls, blob: Mapping) -> VirtualBook:
        book = cls(team_key=str(blob.get("team_key", "")),
                   cash=float(blob.get("cash", 0.0)))
        book.realized_pl = float(blob.get("realized_pl", 0.0))
        book.crossed_notional = float(blob.get("crossed_notional", 0.0))
        book.fees = float(blob.get("fees", 0.0))
        for symbol, lot in (blob.get("lots") or {}).items():
            book.lots[str(symbol)] = Lot(qty=float(lot.get("qty", 0.0)),
                                         cost=float(lot.get("cost", 0.0)))
        return book


@dataclass
class Staged:
    """An intent waiting for the tick's flush."""

    team_key: str
    intent: OrderIntent
    qty: float
    provisional_id: str
    created: datetime = field(default_factory=utcnow)


@dataclass
class Reconciliation:
    """Do the virtual books add up to what the broker actually holds?"""

    cash_real: float
    cash_virtual: float
    positions_real: dict[str, float]
    positions_virtual: dict[str, float]
    tolerance: float = 1.0

    @property
    def cash_drift(self) -> float:
        return self.cash_virtual - self.cash_real

    @property
    def position_drift(self) -> dict[str, float]:
        keys = set(self.positions_real) | set(self.positions_virtual)
        return {
            k: self.positions_virtual.get(k, 0.0) - self.positions_real.get(k, 0.0)
            for k in sorted(keys)
            if abs(self.positions_virtual.get(k, 0.0)
                   - self.positions_real.get(k, 0.0)) > 1e-6
        }

    @property
    def ok(self) -> bool:
        return abs(self.cash_drift) <= self.tolerance and not self.position_drift

    def describe(self) -> str:
        if self.ok:
            return (f"reconciled: cash {self.cash_virtual:,.2f} vs "
                    f"{self.cash_real:,.2f}, positions match")
        parts = [f"cash drift {self.cash_drift:+,.2f}"]
        if self.position_drift:
            parts.append("position drift " + ", ".join(
                f"{k} {v:+.4f}" for k, v in self.position_drift.items()))
        return "; ".join(parts)


# --------------------------------------------------------------------------- #


class SharedAccount:
    """Owns one real broker and partitions it into per-team virtual books.

    This is the only object that talks to the underlying broker. In
    particular it never calls the account-wide `close_all_positions()` or an
    unscoped `cancel_all()` -- either would wipe out every team sharing the
    account.
    """

    def __init__(
        self,
        broker: Broker,
        team_keys: Sequence[str],
        *,
        bankroll: float,
        price_of: Callable[[str], float],
        cross_internally: bool = True,
        name: str = "shared",
        quote_of: Callable[[str], object] | None = None,
    ):
        if not team_keys:
            raise ValueError("a shared account needs at least one team")
        self.broker = broker
        self.name = name
        self.bankroll = float(bankroll)
        self.price_of = price_of
        self.quote_of = quote_of
        self.cross_internally = cross_internally
        self.books: dict[str, VirtualBook] = {
            key: VirtualBook(team_key=key, cash=float(bankroll)) for key in team_keys
        }
        self._staged: list[Staged] = []
        #: broker order id -> owning team
        self._owner: dict[str, str] = {}
        #: broker order id -> filled qty already applied to a book
        self._applied: dict[str, float] = {}
        #: broker order id -> the intent that produced it
        self._intent: dict[str, OrderIntent] = {}
        self.crossed_total = 0.0
        self.wash_rejections = 0
        self.forced_cancels = 0
        #: Teams already reset in the current wave. The engine calls
        #: `reset_for_round` once per team, but the real account must only be
        #: flattened once.
        self._reset_wave: set[str] = set()

    # ------------------------------------------------------------------ #

    @property
    def team_keys(self) -> tuple[str, ...]:
        return tuple(self.books)

    #: Fallback cushion when no quote is available, in basis points. A buy
    #: sized against the mid but filled at the ask overspends by the half
    #: spread, which is how a $5,000 book ends up a few dollars overdrawn.
    NO_QUOTE_CUSHION_BPS = 30.0

    def ref_price(self, symbol: str, side: Side) -> float:
        """The price an order of this side should be *sized* against.

        Sizing against the mid is wrong: a market buy pays the ask. Over a
        whole round that half-spread error accumulates until a book goes
        slightly negative, which then fails reconciliation against the real
        account. So use the actual touch when a quote is available, and a
        cushioned mid when it is not.
        """
        mid = self.price_of(symbol)
        if self.quote_of is not None:
            quote = self.quote_of(symbol)
            bid = getattr(quote, "bid", 0.0) or 0.0
            ask = getattr(quote, "ask", 0.0) or 0.0
            if bid > 0 and ask >= bid:
                return ask if side is Side.BUY else bid
        if mid <= 0:
            return 0.0
        cushion = self.NO_QUOTE_CUSHION_BPS / 10_000.0
        return mid * (1 + cushion) if side is Side.BUY else mid * (1 - cushion)

    def book(self, team_key: str) -> VirtualBook:
        try:
            return self.books[team_key]
        except KeyError as e:
            raise BrokerError(f"{self.name}: no virtual book for {team_key}") from e

    def virtual_broker(self, team_key: str) -> VirtualBroker:
        return VirtualBroker(self, team_key)

    def required_capital(self) -> float:
        return self.bankroll * len(self.books)

    # ------------------------------------------------------------------ #
    # staging
    # ------------------------------------------------------------------ #

    def stage(self, team_key: str, intent: OrderIntent) -> Order:
        """Validate against the team's own book and queue for the flush."""
        book = self.book(team_key)
        if self.price_of(intent.symbol) <= 0:
            raise OrderRejected(f"{self.name}: no price for {intent.symbol}")
        # Size against what the order will actually pay or receive, never the
        # mid -- see `ref_price`.
        ref = intent.limit_price or self.ref_price(intent.symbol, intent.side)
        if ref <= 0:
            raise OrderRejected(f"{self.name}: no usable price for {intent.symbol}")

        qty = intent.qty if intent.qty is not None else intent.notional / ref
        qty = math.floor(qty * 1e6) / 1e6
        if qty <= 0:
            raise OrderRejected(f"{self.name}: {intent.symbol} sized to zero")

        if intent.side is Side.SELL:
            # A team may only ever sell what its own book holds, minus what it
            # has already staged or resting. This is the isolation guarantee.
            available = (book.qty(intent.symbol)
                         - self._staged_qty(team_key, intent.symbol, Side.SELL)
                         - self._resting_qty(team_key, intent.symbol, Side.SELL))
            if qty > available + 1e-6:
                if available <= 1e-6:
                    raise OrderRejected(
                        f"{self.name}: {team_key} holds "
                        f"{book.qty(intent.symbol):g} {intent.symbol} and has "
                        f"already committed all of it"
                    )
                qty = math.floor(available * 1e6) / 1e6
        else:
            need = qty * ref
            free = self.available_cash(team_key)
            if need > free + 1e-6:
                if free <= 1.0:
                    raise InsufficientFunds(
                        f"{self.name}: {team_key} needs ${need:,.2f}, has "
                        f"${free:,.2f} of its own cash"
                    )
                qty = math.floor(free / ref * 0.9995 * 1e6) / 1e6
                if qty * ref < 1.0:
                    raise InsufficientFunds(
                        f"{self.name}: {team_key} needs ${need:,.2f}, has "
                        f"${free:,.2f} of its own cash"
                    )

        provisional = f"stage-{uuid.uuid4().hex[:12]}"
        self._staged.append(Staged(team_key, intent, qty, provisional))
        return Order(
            id=provisional,
            client_order_id=provisional,
            symbol=intent.symbol,
            side=intent.side,
            qty=qty,
            order_type=intent.order_type,
            limit_price=intent.limit_price,
            tif=intent.tif,
            status=OrderStatus.PENDING,
            reason=intent.reason,
            tag=intent.tag,
        )

    def _staged_qty(self, team_key: str, symbol: str, side: Side) -> float:
        return sum(s.qty for s in self._staged
                   if s.team_key == team_key and s.intent.symbol == symbol
                   and s.intent.side is side)

    def _resting_qty(self, team_key: str, symbol: str, side: Side) -> float:
        total = 0.0
        for order in self._live_orders():
            if (self._owner.get(order.id) == team_key and order.symbol == symbol
                    and order.side is side):
                total += order.leaves_qty
        return total

    def available_cash(self, team_key: str) -> float:
        """A team's cash, less what it has staged or resting on the buy side."""
        book = self.book(team_key)
        committed = sum(
            s.qty * (s.intent.limit_price
                     or self.ref_price(s.intent.symbol, Side.BUY))
            for s in self._staged
            if s.team_key == team_key and s.intent.side is Side.BUY
        )
        for order in self._live_orders():
            if self._owner.get(order.id) == team_key and order.side is Side.BUY:
                committed += order.leaves_qty * (
                    order.limit_price or self.ref_price(order.symbol, Side.BUY))
        return max(book.cash - committed, 0.0)

    # ------------------------------------------------------------------ #
    # the flush: net, cross, submit
    # ------------------------------------------------------------------ #

    def flush(self) -> list[Order]:
        """Net the tick's staged intents, cross internally, submit the rest."""
        if not self._staged:
            return []
        staged, self._staged = self._staged, []
        submitted: list[Order] = []

        by_symbol: dict[str, list[Staged]] = {}
        for item in staged:
            by_symbol.setdefault(item.intent.symbol, []).append(item)

        for symbol, items in by_symbol.items():
            remaining = self._cross(symbol, items) if self.cross_internally else items
            for item in remaining:
                order = self._submit_one(item)
                if order is not None:
                    submitted.append(order)
        return submitted

    def _cross(self, symbol: str, items: Sequence[Staged]) -> list[Staged]:
        """Match opposing intents against each other at the mid, pro rata.

        Pro rata rather than first-come: it makes the outcome independent of
        the order teams happened to act in, which the engine deliberately
        rotates.
        """
        mid = self.price_of(symbol)
        if mid <= 0:
            return list(items)

        buys, sells, others = [], [], []
        for item in items:
            # Partition by identity: `Staged` is a dataclass, so a value-based
            # `in` test would conflate two teams staging identical intents.
            if self._crossable(item, mid):
                (buys if item.intent.side is Side.BUY else sells).append(item)
            else:
                others.append(item)
        if not buys or not sells:
            return list(items)

        buy_total = sum(i.qty for i in buys)
        sell_total = sum(i.qty for i in sells)
        crossed = min(buy_total, sell_total)
        if crossed <= 1e-9:
            return list(items)

        leftovers: list[Staged] = list(others)
        for group, total in ((buys, buy_total), (sells, sell_total)):
            for item in group:
                share = crossed * (item.qty / total) if total > 0 else 0.0
                share = min(share, item.qty)
                if share > 1e-9:
                    self._apply_cross(item, share, mid)
                rest = item.qty - share
                if rest > 1e-9:
                    leftovers.append(
                        Staged(item.team_key, item.intent, rest, item.provisional_id)
                    )
        log.info(
            "%s: crossed %.6f %s internally at %.4f between %d team(s)",
            self.name, crossed, symbol, mid, len({i.team_key for i in items}),
        )
        return leftovers

    def _crossable(self, item: Staged, mid: float) -> bool:
        """Would this intent accept the mid as its price?"""
        if item.intent.order_type is OrderType.MARKET:
            return True
        limit = item.intent.limit_price or 0.0
        if item.intent.side is Side.BUY:
            return limit >= mid          # willing to pay at least the mid
        return limit <= mid              # willing to receive at most the mid

    def _apply_cross(self, item: Staged, qty: float, price: float) -> None:
        """Book an internal cross. Cash is conserved exactly."""
        book = self.book(item.team_key)
        symbol = item.intent.symbol
        if item.intent.side is Side.BUY:
            cost = qty * price
            book.cash -= cost
            book.lot(symbol).buy(qty, price)
        else:
            book.cash += qty * price
            book.realized_pl += book.lot(symbol).sell(qty, price)
        book.crossed_notional += qty * price
        self.crossed_total += qty * price
        book.fills.append(
            Fill(order_id=f"cross-{item.provisional_id}", symbol=symbol,
                 side=item.intent.side, qty=qty, price=price, ts=utcnow())
        )

    def _submit_one(self, item: Staged) -> Order | None:
        """Send one residual intent to the real broker, avoiding a wash reject."""
        symbol = item.intent.symbol
        self._clear_conflicts(item)

        intent = OrderIntent(
            symbol=symbol,
            side=item.intent.side,
            qty=round(item.qty, 6),
            order_type=item.intent.order_type,
            limit_price=item.intent.limit_price,
            tif=item.intent.tif,
            reason=item.intent.reason,
            tag=item.intent.tag,
            reduce_only=item.intent.reduce_only,
        )
        try:
            order = self.broker.submit(intent, client_order_id=tag_for(item.team_key))
        except OrderRejected as e:
            blob = str(e).lower()
            if "wash" in blob or "opposite side" in blob:
                self.wash_rejections += 1
                log.warning(
                    "%s: %s %s rejected as a potential wash trade; it will be "
                    "retried next tick", self.name, item.team_key, symbol,
                )
            else:
                log.info("%s: %s %s rejected: %s", self.name, item.team_key,
                         symbol, e)
            return None
        except BrokerError as e:
            log.error("%s: submit failed for %s %s: %s", self.name,
                      item.team_key, symbol, e)
            return None

        self._owner[order.id] = item.team_key
        self._applied[order.id] = 0.0
        self._intent[order.id] = item.intent
        order.reason = item.intent.reason
        order.tag = item.intent.tag
        return order

    def _clear_conflicts(self, item: Staged) -> None:
        """Cancel another team's resting order that would trigger a wash reject.

        Exits must never be blocked by a different team's quote, so the resting
        order gives way. Only the market maker rests quotes, and it requotes
        every tick -- but this does fall on one competitor, and it is counted.
        """
        opposite = item.intent.side.flip()
        for order in self._live_orders():
            if order.symbol != item.intent.symbol or order.side is not opposite:
                continue
            owner = self._owner.get(order.id)
            if owner == item.team_key:
                continue          # a team's own book is its own problem
            try:
                self.broker.cancel(order.id)
                self.forced_cancels += 1
                log.info(
                    "%s: cancelled %s's resting %s %s to let %s's %s through",
                    self.name, owner or "?", order.side.value, order.symbol,
                    item.team_key, item.intent.side.value,
                )
            except BrokerError as e:
                log.debug("%s: could not cancel %s: %s", self.name, order.id, e)

    # ------------------------------------------------------------------ #
    # fills
    # ------------------------------------------------------------------ #

    def _live_orders(self) -> list[Order]:
        try:
            return list(self.broker.open_orders())
        except BrokerError:
            return []

    def poll_fills(self) -> dict[str, list[Fill]]:
        """Attribute new broker fills to the right virtual book.

        Returns only the fills newly discovered on this call; each book also
        keeps the cumulative list the engine dedupes against.
        """
        fresh: dict[str, list[Fill]] = {k: [] for k in self.books}
        orders: list[Order] = []
        with contextlib.suppress(BrokerError):
            orders.extend(self.broker.open_orders())
        closed = getattr(self.broker, "closed_orders", None)
        if callable(closed):
            with contextlib.suppress(BrokerError):
                orders.extend(closed(limit=200))
        else:
            inner = getattr(self.broker, "_orders", None)
            if isinstance(inner, dict):
                orders.extend(inner.values())

        for order in orders:
            team = self._owner.get(order.id) or team_from_tag(order.client_order_id)
            if team is None or team not in self.books:
                continue
            already = self._applied.get(order.id, 0.0)
            delta = order.filled_qty - already
            if delta <= 1e-9:
                continue
            price = order.filled_avg_price or self.price_of(order.symbol)
            if price <= 0:
                continue
            self._applied[order.id] = order.filled_qty
            self._owner.setdefault(order.id, team)
            self._apply_fill(team, order, delta, price)
            fill = Fill(order_id=order.id, symbol=order.symbol, side=order.side,
                        qty=delta, price=price, ts=order.updated_at)
            self.book(team).fills.append(fill)
            fresh[team].append(fill)
        return fresh

    def _apply_fill(self, team: str, order: Order, qty: float, price: float) -> None:
        book = self.book(team)
        if order.side is Side.BUY:
            book.cash -= qty * price
            book.lot(order.symbol).buy(qty, price)
            return
        # Never credit cash for stock the book does not hold: that would mint
        # money and silently break reconciliation against the real account.
        held = book.qty(order.symbol)
        if qty > held + 1e-6:
            log.error(
                "%s: %s filled a sell of %.6f %s but its book holds %.6f -- "
                "crediting only what it held. This means attribution drifted; "
                "check the reconciliation report.",
                self.name, team, qty, order.symbol, held,
            )
            qty = held
        if qty <= 1e-9:
            return
        book.cash += qty * price
        book.realized_pl += book.lot(order.symbol).sell(qty, price)

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def sync(self, now: datetime | None = None) -> None:
        self.broker.sync(now)
        self.poll_fills()

    def reset(self, bankroll: float | None = None) -> None:
        """Flatten the real account and reset every virtual book."""
        if bankroll is not None:
            self.bankroll = float(bankroll)
        self._staged.clear()
        self._owner.clear()
        self._applied.clear()
        self._intent.clear()
        for key in self.books:
            self.books[key] = VirtualBook(team_key=key, cash=self.bankroll)
        self.crossed_total = 0.0
        self.wash_rejections = 0
        self.forced_cancels = 0
        self._reset_wave.clear()
        self._flatten_real()

    def _flatten_real(self) -> None:
        try:
            self.broker.cancel_all()
            self.broker.close_all_positions(cancel_orders=True)
        except BrokerError as e:
            log.error("%s: could not flatten the shared account: %s", self.name, e)

    def reset_team(self, team_key: str, cash: float) -> None:
        """Reset one team's book, flattening the real account once per wave.

        The engine loops over teams calling `reset_for_round`. Flattening the
        shared account on every call would be N round trips and N-1 no-ops --
        and worse, the second call would liquidate positions the first team had
        just been given. So the first team of a wave does the real work.
        """
        if not self._reset_wave:
            self._staged.clear()
            self._owner.clear()
            self._applied.clear()
            self._intent.clear()
            self._flatten_real()
        self._reset_wave.add(team_key)
        self.books[team_key] = VirtualBook(team_key=team_key, cash=float(cash))
        if self._reset_wave >= set(self.books):
            self._reset_wave.clear()

    def reconcile(self) -> Reconciliation:
        """Check the books add up to what the broker holds."""
        try:
            real = self.broker.account()
        except BrokerError as e:
            raise BrokerError(f"{self.name}: cannot read the shared account: {e}") from e
        virtual_cash = sum(b.cash for b in self.books.values())
        virtual_pos: dict[str, float] = {}
        for book in self.books.values():
            for symbol, lot in book.lots.items():
                if lot.qty > 1e-12:
                    virtual_pos[symbol] = virtual_pos.get(symbol, 0.0) + lot.qty
        return Reconciliation(
            cash_real=real.cash,
            cash_virtual=virtual_cash,
            positions_real={p.symbol: p.qty for p in real.positions},
            positions_virtual=virtual_pos,
            tolerance=max(0.01 * self.bankroll, 1.0),
        )

    def summary(self) -> dict:
        return {
            "name": self.name,
            "teams": list(self.books),
            "bankroll_each": self.bankroll,
            "required_capital": self.required_capital(),
            "crossed_notional": round(self.crossed_total, 2),
            "wash_rejections": self.wash_rejections,
            "forced_cancels": self.forced_cancels,
            "books": {k: b.to_dict() for k, b in self.books.items()},
        }

    def to_dict(self) -> dict:
        """Persisted state. Without this a crash loses the per-team split."""
        return {
            "version": 1,
            "bankroll": self.bankroll,
            "crossed_total": self.crossed_total,
            "wash_rejections": self.wash_rejections,
            "forced_cancels": self.forced_cancels,
            "books": {k: b.to_dict() for k, b in self.books.items()},
            "owner": dict(self._owner),
            "applied": dict(self._applied),
        }

    def load(self, blob: Mapping) -> None:
        if not blob:
            return
        self.bankroll = float(blob.get("bankroll", self.bankroll))
        self.crossed_total = float(blob.get("crossed_total", 0.0))
        self.wash_rejections = int(blob.get("wash_rejections", 0))
        self.forced_cancels = int(blob.get("forced_cancels", 0))
        for key, payload in (blob.get("books") or {}).items():
            if key in self.books:
                self.books[key] = VirtualBook.from_dict(payload)
        self._owner = {str(k): str(v) for k, v in (blob.get("owner") or {}).items()}
        self._applied = {str(k): float(v)
                         for k, v in (blob.get("applied") or {}).items()}
        log.info("%s: restored %d virtual book(s)", self.name, len(self.books))


# --------------------------------------------------------------------------- #


class VirtualBroker(Broker):
    """One team's `Broker` view of a shared account.

    Every account-wide operation is narrowed to this team's own holdings. In
    particular `close_all_positions` sells this team's quantities one symbol at
    a time rather than calling the broker's account-wide close, which would
    liquidate every team sharing the account.
    """

    def __init__(self, shared: SharedAccount, team_key: str):
        self.shared = shared
        self.team_key = team_key
        self.name = f"{shared.name}:{team_key}"
        shared.book(team_key)          # fail fast if the team has no book

    # -- state ------------------------------------------------------------ #

    @property
    def book(self) -> VirtualBook:
        return self.shared.book(self.team_key)

    def account(self) -> Account:
        book = self.book
        positions = book.positions(self.shared.price_of)
        equity = book.cash + sum(p.market_value for p in positions)
        return Account(
            cash=round(book.cash, 6),
            equity=round(equity, 6),
            buying_power=round(self.shared.available_cash(self.team_key), 6),
            positions=positions,
        )

    def positions(self) -> list[Position]:
        return list(self.book.positions(self.shared.price_of))

    @property
    def baseline_equity(self) -> float:
        return self.shared.bankroll

    # -- orders ----------------------------------------------------------- #

    def submit(self, intent: OrderIntent, *, client_order_id: str | None = None) -> Order:
        return self.shared.stage(self.team_key, intent)

    def open_orders(self, symbol: str | None = None) -> list[Order]:
        sym = symbol.upper() if symbol else None
        out = []
        for order in self.shared._live_orders():
            if self.shared._owner.get(order.id) != self.team_key:
                continue
            if sym and order.symbol != sym:
                continue
            out.append(order)
        return out

    def cancel(self, order_id: str) -> None:
        if self.shared._owner.get(order_id) != self.team_key:
            log.debug("%s: refusing to cancel %s, not this team's order",
                      self.name, order_id)
            return
        self.shared.broker.cancel(order_id)

    def cancel_all(self, symbol: str | None = None) -> int:
        n = 0
        for order in self.open_orders(symbol):
            self.cancel(order.id)
            n += 1
        return n

    # -- bulk ------------------------------------------------------------- #

    def close_position(self, symbol: str) -> Order | None:
        held = self.book.qty(symbol)
        if held <= 1e-9:
            return None
        self.cancel_all(symbol)
        return self.submit(OrderIntent(symbol=symbol, side=Side.SELL,
                                       qty=round(held, 6),
                                       reason="close_position", reduce_only=True))

    def close_all_positions(self, *, cancel_orders: bool = True) -> list[Order]:
        # Never `self.shared.broker.close_all_positions()` -- that would
        # liquidate every team sharing this account.
        if cancel_orders:
            self.cancel_all()
        out = []
        for position in self.positions():
            try:
                order = self.close_position(position.symbol)
            except OrderRejected:
                continue
            if order:
                out.append(order)
        return out

    # -- lifecycle -------------------------------------------------------- #

    def sync(self, now: datetime | None = None) -> None:
        self.shared.sync(now)

    def flush(self) -> list[Order]:
        return self.shared.flush()

    def reset_for_round(self, starting_cash: float) -> None:
        """Reset this team's book; the real account is flattened once per wave."""
        self.shared.reset_team(self.team_key, starting_cash)

    @property
    def fills(self) -> list[Fill]:
        """This team's cumulative fills, as `SimulatedBroker.fills` provides.

        Polling here (rather than draining a queue) keeps the engine's
        dedupe-by-identity logic working unchanged, and means one team reading
        its fills cannot consume another's.
        """
        self.shared.poll_fills()
        return self.shared.book(self.team_key).fills

    @property
    def supports_fractional(self) -> bool:
        return self.shared.broker.supports_fractional

    @property
    def supports_notional_orders(self) -> bool:
        return False        # sized to shares before submission
