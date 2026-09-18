"""Deterministic offline broker.

Purpose: run the entire three-round competition -- all eight teams, every
tick -- with no network and no keys, so the field can be validated, backtested
and regression-tested before a dollar of paper money moves. Every team is
filled by the *same* model with the *same* slippage, so sim results are
comparable even though they are not real.

Fill model
----------
* market buy  -> ask * (1 + slip),  market sell -> bid * (1 - slip)
* limit buy   -> fills when ask <= limit, at min(limit, ask); price improvement
                 is passed through, which is what actually happens on a resting
                 bid that gets hit.
* limit sell  -> fills when bid >= limit, at max(limit, bid)
* `day` orders are cancelled by `end_of_day()`; `gtc` survives.
* optional per-bar participation cap models the fact that you cannot buy
  the whole tape with one clip.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from ..types import (
    Account,
    Fill,
    Order,
    OrderIntent,
    OrderStatus,
    OrderType,
    Position,
    Quote,
    Side,
    TimeInForce,
    utcnow,
)
from .base import Broker, InsufficientFunds, OrderRejected

#: Returns the current quote for a symbol, or None if it is not trading.
QuoteSource = Callable[[str], "Quote | None"]


@dataclass
class _Lot:
    """Running position with a weighted-average cost basis."""

    qty: float = 0.0
    cost: float = 0.0  # total dollars paid for the open qty

    @property
    def avg_price(self) -> float:
        return (self.cost / self.qty) if self.qty > 0 else 0.0


@dataclass
class SimConfig:
    slippage_bps: float = 2.0
    commission_per_share: float = 0.0
    #: Fraction of a bar's volume one order may consume (None = unlimited).
    max_participation: float | None = None
    #: Allow fractional shares, as Alpaca does for most large caps.
    fractional: bool = True
    #: Reject sells beyond the held quantity (long-only accounts).
    allow_short: bool = False
    #: Minimum tradable increment when fractional is off.
    lot_size: float = 1.0


class SimulatedBroker(Broker):
    """In-process portfolio + matching engine for one team."""

    _ids = itertools.count(1)

    def __init__(
        self,
        starting_cash: float,
        quote_source: QuoteSource,
        *,
        name: str = "sim",
        config: SimConfig | None = None,
        clock: Callable[[], datetime] = utcnow,
    ):
        self.name = name
        self.cfg = config or SimConfig()
        self._quote_of = quote_source
        self._clock = clock
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)
        self._lots: dict[str, _Lot] = {}
        self._orders: dict[str, Order] = {}
        self.fills: list[Fill] = []
        self.realized_pl = 0.0
        self.fees_paid = 0.0
        self.rejections: list[tuple[datetime, str, str]] = []
        self._volume_used: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # pricing
    # ------------------------------------------------------------------ #

    def _quote(self, symbol: str) -> Quote:
        q = self._quote_of(symbol)
        if q is None or not q.is_sane:
            raise OrderRejected(f"{self.name}: no tradable quote for {symbol}")
        return q

    def mark(self, symbol: str) -> float:
        q = self._quote_of(symbol)
        if q is not None and q.mid > 0:
            return q.mid
        lot = self._lots.get(symbol)
        return lot.avg_price if lot else 0.0

    # ------------------------------------------------------------------ #
    # account
    # ------------------------------------------------------------------ #

    def positions(self) -> list[Position]:
        out = []
        for sym, lot in self._lots.items():
            if abs(lot.qty) < 1e-9:
                continue
            out.append(
                Position(
                    symbol=sym,
                    qty=round(lot.qty, 9),
                    avg_entry_price=lot.avg_price,
                    current_price=self.mark(sym) or lot.avg_price,
                )
            )
        return sorted(out, key=lambda p: p.symbol)

    def account(self) -> Account:
        pos = tuple(self.positions())
        equity = self.cash + sum(p.market_value for p in pos)
        # Long-only, no margin: buying power is settled cash minus the cash
        # already committed to working buy limits.
        committed = sum(
            (o.limit_price or self.mark(o.symbol)) * o.leaves_qty
            for o in self._orders.values()
            if o.status.is_open and o.side is Side.BUY
        )
        return Account(
            cash=round(self.cash, 6),
            equity=round(equity, 6),
            buying_power=round(max(self.cash - committed, 0.0), 6),
            positions=pos,
        )

    @property
    def equity(self) -> float:
        return self.account().equity

    def qty_of(self, symbol: str) -> float:
        lot = self._lots.get(symbol)
        return lot.qty if lot else 0.0

    # ------------------------------------------------------------------ #
    # orders
    # ------------------------------------------------------------------ #

    def _round_qty(self, qty: float, symbol: str) -> float:
        # Always round DOWN. Rounding a share count up, even at the sixth
        # decimal, means spending marginally more than the notional asked
        # for -- and an account deploying its last dollar would overdraw.
        if self.cfg.fractional:
            return math.floor(qty * 1e6) / 1e6
        step = max(self.cfg.lot_size, 1e-9)
        return float(math.floor(qty / step)) * step

    def submit(self, intent: OrderIntent, *, client_order_id: str | None = None) -> Order:
        q = self._quote(intent.symbol)
        ref = q.ask if intent.side is Side.BUY else q.bid
        if intent.order_type is OrderType.LIMIT and intent.limit_price:
            ref = intent.limit_price

        qty = intent.qty
        if qty is None:
            if ref <= 0:
                raise OrderRejected(f"{self.name}: cannot size {intent.symbol}, no price")
            # Size against the price we expect to *pay*, not the quote: a
            # market order sized at the raw ask and then filled at ask +
            # slippage spends more than the notional asked for, which can
            # overdraw an account deploying its last dollar.
            expected = (
                self._slip(ref, intent.side)
                if intent.order_type is OrderType.MARKET else ref
            )
            qty = intent.notional / max(expected, 0.01)
        qty = self._round_qty(qty, intent.symbol)
        if qty <= 0:
            raise OrderRejected(f"{self.name}: {intent.symbol} sized to zero shares")

        if intent.side is Side.SELL and not self.cfg.allow_short:
            held = self.qty_of(intent.symbol)
            if qty > held + 1e-6:
                if held <= 1e-9:
                    raise OrderRejected(f"{self.name}: cannot sell {intent.symbol}, nothing held")
                qty = self._round_qty(held, intent.symbol)  # trim to flat, never short

        if intent.side is Side.BUY:
            fill_ref = (
                self._slip(ref, intent.side)
                if intent.order_type is OrderType.MARKET else ref
            )
            need = qty * fill_ref + qty * self.cfg.commission_per_share
            avail = self.account().buying_power
            if need > avail + 1e-6:
                # Clip to what is affordable -- but refuse outright if that
                # leaves less than a dollar of notional. Silently filling one
                # cent of stock is worse than a clean rejection: it creates
                # unsellable dust and hides the funding problem.
                affordable = max(
                    avail / (fill_ref + self.cfg.commission_per_share) * 0.9995, 0.0
                )
                qty = self._round_qty(affordable, intent.symbol)
                if qty <= 0 or qty * fill_ref < 1.0:
                    raise InsufficientFunds(
                        f"{self.name}: {intent.describe()} needs ${need:,.2f}, "
                        f"have ${avail:,.2f}"
                    )

        oid = f"sim-{next(self._ids):07d}"
        order = Order(
            id=oid,
            client_order_id=client_order_id or oid,
            symbol=intent.symbol,
            side=intent.side,
            qty=qty,
            order_type=intent.order_type,
            limit_price=intent.limit_price,
            tif=intent.tif,
            status=OrderStatus.NEW,
            submitted_at=self._clock(),
            updated_at=self._clock(),
            reason=intent.reason,
        )
        self._orders[oid] = order
        if intent.order_type is OrderType.MARKET:
            self._try_fill(order)
        else:
            # A resting limit can still be immediately executable.
            self._try_fill(order)
        return order

    def open_orders(self, symbol: str | None = None) -> list[Order]:
        sym = symbol.upper() if symbol else None
        return [
            o for o in self._orders.values()
            if o.status.is_open and (sym is None or o.symbol == sym)
        ]

    def cancel(self, order_id: str) -> None:
        o = self._orders.get(order_id)
        if o and o.status.is_open:
            o.status = OrderStatus.CANCELED if o.filled_qty == 0 else OrderStatus.FILLED
            o.updated_at = self._clock()

    def cancel_all(self, symbol: str | None = None) -> int:
        n = 0
        for o in self.open_orders(symbol):
            self.cancel(o.id)
            n += 1
        return n

    # ------------------------------------------------------------------ #
    # matching
    # ------------------------------------------------------------------ #

    def _slip(self, price: float, side: Side) -> float:
        adj = 1.0 + side.sign * self.cfg.slippage_bps / 10_000.0
        return max(price * adj, 0.01)

    def _try_fill(self, order: Order) -> None:
        if not order.status.is_open:
            return
        q = self._quote_of(order.symbol)
        if q is None or not q.is_sane:
            return

        if order.order_type is OrderType.MARKET:
            price = self._slip(q.ask if order.side is Side.BUY else q.bid, order.side)
        else:
            lim = order.limit_price or 0.0
            if order.side is Side.BUY:
                if q.ask > lim:
                    return                      # not yet executable
                price = min(lim, q.ask)         # price improvement
            else:
                if q.bid < lim:
                    return
                price = max(lim, q.bid)
        if price <= 0:
            return

        qty = order.leaves_qty
        cap = self.cfg.max_participation
        if cap:
            book_size = (q.ask_size if order.side is Side.BUY else q.bid_size) or 0.0
            if book_size > 0:
                qty = min(qty, max(book_size * cap, 0.0))
        if qty <= 1e-9:
            return

        self._apply_fill(order, qty, price)

    def _apply_fill(self, order: Order, qty: float, price: float) -> None:
        fee = qty * self.cfg.commission_per_share
        lot = self._lots.setdefault(order.symbol, _Lot())

        if order.side is Side.BUY:
            cost = qty * price + fee
            if cost > self.cash + 1e-6:
                qty = max((self.cash - fee) / price, 0.0)
                if qty <= 1e-9:
                    return
                cost = qty * price + fee
            self.cash -= cost
            lot.qty += qty
            lot.cost += qty * price
        else:
            sell_qty = min(qty, lot.qty) if not self.cfg.allow_short else qty
            if sell_qty <= 1e-9:
                return
            qty = sell_qty
            avg = lot.avg_price
            proceeds = qty * price - fee
            self.cash += proceeds
            self.realized_pl += qty * (price - avg) - fee
            lot.cost = max(lot.cost - qty * avg, 0.0)
            lot.qty = max(lot.qty - qty, 0.0)
            if lot.qty <= 1e-9:
                lot.qty, lot.cost = 0.0, 0.0

        self.fees_paid += fee
        prev_notional = order.filled_avg_price * order.filled_qty
        order.filled_qty += qty
        order.filled_avg_price = (prev_notional + qty * price) / order.filled_qty
        order.status = OrderStatus.FILLED if order.leaves_qty <= 1e-9 else OrderStatus.PARTIALLY_FILLED
        order.updated_at = self._clock()
        self.fills.append(
            Fill(
                order_id=order.id,
                symbol=order.symbol,
                side=order.side,
                qty=qty,
                price=price,
                ts=self._clock(),
                fee=fee,
            )
        )

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def sync(self, now: datetime | None = None) -> None:
        """Re-check every working order against the current quotes."""
        for order in list(self._orders.values()):
            if order.status.is_open:
                self._try_fill(order)

    def expire_stale_orders(self, ttl_seconds: float) -> int:
        """Cancel working orders older than `ttl_seconds` (used by the scalper)."""
        now, n = self._clock(), 0
        for o in self.open_orders():
            if (now - o.submitted_at).total_seconds() > ttl_seconds:
                self.cancel(o.id)
                n += 1
        return n

    def end_of_day(self) -> int:
        """Cancel `day` orders, as the venue does at the close."""
        n = 0
        for o in self.open_orders():
            if o.tif is TimeInForce.DAY:
                self.cancel(o.id)
                n += 1
        self._volume_used.clear()
        return n

    def close_position(self, symbol: str) -> Order | None:
        held = self.qty_of(symbol)
        if held <= 1e-9:
            return None
        self.cancel_all(symbol)
        return self.submit(
            OrderIntent(symbol=symbol, side=Side.SELL, qty=held, reason="close_position",
                        reduce_only=True)
        )

    def close_all_positions(self, *, cancel_orders: bool = True) -> list[Order]:
        if cancel_orders:
            self.cancel_all()
        out = []
        for p in self.positions():
            try:
                o = self.close_position(p.symbol)
            except OrderRejected:
                continue
            if o:
                out.append(o)
        return out

    def reset_for_round(self, starting_cash: float) -> None:
        self.cancel_all()
        self._lots.clear()
        self._orders.clear()
        self.fills.clear()
        self.cash = float(starting_cash)
        self.starting_cash = float(starting_cash)
        self.realized_pl = 0.0
        self.fees_paid = 0.0
        self.rejections.clear()
        self._volume_used.clear()

    @property
    def baseline_equity(self) -> float:
        return self.starting_cash

    @property
    def supports_fractional(self) -> bool:
        return self.cfg.fractional

    @property
    def supports_notional_orders(self) -> bool:
        return True

    # ------------------------------------------------------------------ #

    def summary(self) -> dict:
        acct = self.account()
        return {
            "name": self.name,
            "cash": round(acct.cash, 2),
            "equity": round(acct.equity, 2),
            "return_pct": round((acct.equity / self.starting_cash - 1.0) * 100, 4)
            if self.starting_cash else 0.0,
            "realized_pl": round(self.realized_pl, 2),
            "fees": round(self.fees_paid, 4),
            "fills": len(self.fills),
            "positions": {p.symbol: round(p.qty, 4) for p in acct.positions},
        }
