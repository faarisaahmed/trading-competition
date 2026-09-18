"""The strategy contract.

Design rule for this competition: **every team gets the same plumbing.** The
things that are easy to get subtly wrong -- converting a target weight into a
share count, respecting the minimum notional, not double-sending an order
that is already working, rounding fractional shares -- live here, once, and
are shared. A team wins or loses on its *ideas*, not on whether its author
remembered to subtract the existing position before sizing.

What a strategy actually implements is small:

    tick_seconds     how often it wants to be called
    warmup_bars      how much history it needs before it will trade
    PARAMS           its typed parameter schema (documents itself)
    on_tick(ctx)     -> list[OrderIntent]

Everything else -- risk rails, universe enforcement, order submission,
logging, the ledger -- is the engine's job.
"""

from __future__ import annotations

import abc
import logging
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import RiskConfig, TeamConfig
from ..data.snapshot import MarketSnapshot
from ..types import (
    Account,
    Bar,
    Fill,
    Order,
    OrderIntent,
    OrderType,
    Position,
    Quote,
    Side,
    TimeInForce,
)
from ..util import indicators as ind

# --------------------------------------------------------------------------- #
# parameters
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Param:
    """One declared parameter: name, default, and what it means."""

    name: str
    default: Any
    doc: str = ""
    minimum: float | None = None
    maximum: float | None = None

    def coerce(self, value: Any) -> Any:
        if isinstance(self.default, bool):
            return bool(value)
        if isinstance(self.default, int) and not isinstance(value, bool):
            v = int(value)
        elif isinstance(self.default, float):
            v = float(value)
        elif isinstance(self.default, (list, tuple)):
            return type(self.default)(value)
        else:
            return value
        if self.minimum is not None and v < self.minimum:
            raise ValueError(f"{self.name}={v} below minimum {self.minimum}")
        if self.maximum is not None and v > self.maximum:
            raise ValueError(f"{self.name}={v} above maximum {self.maximum}")
        return v


class Params:
    """Validated parameter bag built from a strategy's declared schema.

    Unknown keys are a hard error: a typo in `config/teams.yaml` that silently
    left a parameter at its default would be an unfair, invisible handicap.
    """

    def __init__(self, schema: Sequence[Param], values: Mapping[str, Any] | None = None):
        self._schema = {p.name: p for p in schema}
        raw = dict(values or {})
        unknown = set(raw) - set(self._schema)
        if unknown:
            raise ValueError(
                f"unknown parameter(s) {sorted(unknown)}; declared: {sorted(self._schema)}"
            )
        self._values: dict[str, Any] = {}
        for name, p in self._schema.items():
            self._values[name] = p.coerce(raw[name]) if name in raw else p.default

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError as e:
            raise AttributeError(
                f"parameter {name!r} is not declared (have: {sorted(self._values)})"
            ) from e

    def __getitem__(self, name: str) -> Any:
        return self._values[name]

    def get(self, name: str, default: Any = None) -> Any:
        return self._values.get(name, default)

    def as_dict(self) -> dict[str, Any]:
        return dict(self._values)

    def document(self) -> list[tuple[str, Any, str]]:
        return [(p.name, self._values[p.name], p.doc) for p in self._schema.values()]


# --------------------------------------------------------------------------- #
# context
# --------------------------------------------------------------------------- #


@dataclass
class StrategyContext:
    """Everything a strategy may read on a tick, plus order-building helpers."""

    team_key: str
    round_id: int
    snapshot: MarketSnapshot
    account: Account
    universe: tuple[str, ...]
    starting_cash: float
    baseline_equity: float
    risk: RiskConfig
    rng: random.Random
    state: dict[str, Any]
    log: logging.Logger
    open_orders: tuple[Order, ...] = ()
    tick_index: int = 0
    round_progress: float = 0.0     # 0.0 at the open of day 1, 1.0 at the final close
    sessions_remaining: int = 0
    is_final_session: bool = False
    fractionable: frozenset[str] = frozenset()
    #: Order ids the strategy wants cancelled. The engine drains this after
    #: `on_tick` returns, before it submits any new intents -- so a strategy
    #: that works resting quotes (the scalper) can pull a stale bid and post
    #: a fresh one in the same tick without ever being double-filled.
    cancel_requests: list[str] = field(default_factory=list)
    #: Cash already promised to intents built earlier in THIS tick, and cash
    #: that same-tick market sells will release. A fresh context is built per
    #: tick, so these always start at zero.
    _reserved: float = 0.0
    _credited: float = 0.0

    # -- portfolio reads --------------------------------------------------- #

    @property
    def ts(self):
        return self.snapshot.ts

    @property
    def session(self):
        return self.snapshot.session

    @property
    def equity(self) -> float:
        return self.account.equity

    @property
    def cash(self) -> float:
        return self.account.cash

    @property
    def buying_power(self) -> float:
        """Spendable cash, net of everything already promised this tick.

        Subtracts (a) cash tied up in working buy orders and (b) cash already
        allocated to intents this strategy built earlier in this same tick;
        adds back (c) proceeds from same-tick *market* sells, which the engine
        submits before any buys.

        Without (b), a strategy filling three position slots in one loop sizes
        each entry against the full balance, and the guardrails reject two of
        them. That was showing up as a wall of `insufficient_cash` rejections
        for every team that opens more than one position at a time.
        """
        committed = sum(
            (o.limit_price or self.price(o.symbol)) * o.leaves_qty
            for o in self.open_orders
            if o.side is Side.BUY
        )
        available = min(self.account.buying_power, self.account.cash)
        return max(available - committed - self._reserved + self._credited, 0.0)

    def reserve(self, amount: float) -> None:
        """Mark cash as promised to an intent built this tick."""
        if amount > 0:
            self._reserved += amount

    def credit(self, amount: float) -> None:
        """Mark cash that a same-tick market sell will release."""
        if amount > 0:
            self._credited += amount

    @property
    def round_return(self) -> float:
        base = self.baseline_equity or self.starting_cash
        return (self.equity / base - 1.0) if base > 0 else 0.0

    def position(self, symbol: str) -> Position | None:
        return self.account.position(symbol.upper())

    def qty(self, symbol: str) -> float:
        return self.account.qty(symbol.upper())

    def holds(self, symbol: str) -> bool:
        return abs(self.qty(symbol)) > 1e-9

    def weight(self, symbol: str) -> float:
        """Position market value as a fraction of equity."""
        e = self.equity
        return (self.account.market_value(symbol.upper()) / e) if e > 0 else 0.0

    @property
    def gross_weight(self) -> float:
        return self.account.leverage

    @property
    def held(self) -> tuple[str, ...]:
        return self.account.held_symbols

    @property
    def n_positions(self) -> int:
        return len(self.held)

    def unrealized_pct(self, symbol: str) -> float:
        p = self.position(symbol)
        return p.unrealized_plpc if p else 0.0

    # -- market reads ------------------------------------------------------ #

    def price(self, symbol: str) -> float:
        return self.snapshot.price(symbol)

    def quote(self, symbol: str) -> Quote | None:
        return self.snapshot.quote(symbol)

    def bars(self, symbol: str, kind: str = "primary") -> tuple[Bar, ...]:
        return self.snapshot.series(symbol, kind)

    def closes(self, symbol: str, kind: str = "primary", n: int | None = None) -> list[float]:
        return self.snapshot.closes(symbol, kind, n)

    def tradable(self, symbol: str) -> bool:
        return symbol.upper() in set(self.universe) and self.snapshot.tradable(symbol)

    def candidates(self, min_bars: int, kind: str = "primary") -> list[str]:
        """Universe members with enough history to act on."""
        return [s for s in self.snapshot.ready(min_bars, kind) if self.tradable(s)]

    # -- cancellation ------------------------------------------------------ #

    def request_cancel(self, order_id: str) -> None:
        if order_id and order_id not in self.cancel_requests:
            self.cancel_requests.append(order_id)

    def request_cancel_symbol(self, symbol: str, side: Side | None = None) -> int:
        """Cancel this team's working orders on one symbol. Returns the count."""
        sym = symbol.upper()
        n = 0
        for o in self.open_orders:
            if o.symbol == sym and (side is None or o.side is side):
                self.request_cancel(o.id)
                n += 1
        return n

    def request_cancel_all(self) -> int:
        for o in self.open_orders:
            self.request_cancel(o.id)
        return len(self.open_orders)

    def expire_orders_older_than(self, seconds: float) -> int:
        """Cancel working orders that have rested longer than `seconds`."""
        n = 0
        for o in self.open_orders:
            if (self.ts - o.submitted_at).total_seconds() > seconds:
                self.request_cancel(o.id)
                n += 1
        return n

    def orders_for(self, symbol: str, side: Side | None = None) -> tuple[Order, ...]:
        sym = symbol.upper()
        return tuple(
            o for o in self.open_orders
            if o.symbol == sym and (side is None or o.side is side)
        )

    def has_open_order(self, symbol: str, side: Side | None = None) -> bool:
        sym = symbol.upper()
        return any(
            o.symbol == sym and (side is None or o.side is side) for o in self.open_orders
        )

    def open_order_qty(self, symbol: str, side: Side) -> float:
        sym = symbol.upper()
        return sum(o.leaves_qty for o in self.open_orders if o.symbol == sym and o.side is side)

    # -- per-symbol scratch state ----------------------------------------- #

    def sym_state(self, symbol: str) -> dict[str, Any]:
        """Persistent per-symbol dict (stops, entry bars, cooldowns...)."""
        book = self.state.setdefault("_symbols", {})
        return book.setdefault(symbol.upper(), {})

    def remember(self, key: str, value: Any) -> None:
        self.state[key] = value

    def recall(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def bar_clock(self) -> int:
        """Monotone bar counter, used for bar-based cooldowns and time stops."""
        return int(self.state.get("_bar_clock", 0))

    def advance_bar_clock(self) -> int:
        n = self.bar_clock() + 1
        self.state["_bar_clock"] = n
        return n

    def on_cooldown(self, symbol: str, key: str = "cooldown_until") -> bool:
        return self.bar_clock() < int(self.sym_state(symbol).get(key, 0))

    def set_cooldown(self, symbol: str, bars: int, key: str = "cooldown_until") -> None:
        self.sym_state(symbol)[key] = self.bar_clock() + max(int(bars), 0)

    # -- order construction ------------------------------------------------ #

    def min_notional(self) -> float:
        return max(self.risk.min_order_notional, 1.0)

    def max_notional(self) -> float:
        return min(self.risk.max_order_notional, self.risk.max_position_pct * self.equity)

    def _shares_for(self, symbol: str, notional: float, price: float) -> float:
        """Notional -> share count, honouring whole-share-only symbols."""
        if price <= 0:
            return 0.0
        qty = notional / price
        if symbol.upper() in self.fractionable:
            return round(qty, 6)
        return float(math.floor(qty))

    def buy_notional(self, symbol: str, notional: float, *, reason: str = "",
                     tag: str = "") -> OrderIntent | None:
        """Market buy for a dollar amount, clipped to spendable cash.

        Whatever it commits is reserved, so a later call in the same tick
        cannot spend the same dollar twice.
        """
        sym = symbol.upper()
        px = self.price(sym)
        if px <= 0:
            return None
        cap = min(notional, self.buying_power, self.max_notional())
        if cap < self.min_notional():
            return None
        if sym in self.fractionable:
            # Floor to the cent, never round. `round(993.415, 2)` gives
            # 993.42, which is one cent MORE than the cash on hand, and the
            # guardrails then reject the order for insufficient funds. That
            # single rounding direction was producing a rejection on every
            # tick a team tried to deploy its last dollar.
            spend = math.floor(cap * 100.0) / 100.0
            if spend < self.min_notional():
                return None
            self.reserve(spend)
            return OrderIntent(symbol=sym, side=Side.BUY, notional=spend,
                               reason=reason, tag=tag)
        qty = self._shares_for(sym, cap, px)
        if qty < 1 or qty * px < self.min_notional():
            return None
        self.reserve(qty * px)
        return OrderIntent(symbol=sym, side=Side.BUY, qty=qty, reason=reason, tag=tag)

    def sell_qty(self, symbol: str, qty: float, *, reason: str = "",
                 tag: str = "") -> OrderIntent | None:
        """Market sell of `qty` shares, trimmed to what is actually held."""
        sym = symbol.upper()
        held = self.qty(sym)
        if held <= 1e-9:
            return None
        amount = min(qty, held)
        if sym not in self.fractionable:
            amount = float(math.floor(amount))
            # Whole-share accounts cannot leave a fractional tail behind, so
            # if rounding down would orphan the rest, sell the lot.
            if held - amount < 1.0 and held - amount > 1e-9:
                amount = held
        if amount <= 1e-9:
            return None
        px = self.price(sym)
        if px > 0 and amount * px < self.min_notional() and amount < held:
            return None            # dust; not worth an order unless closing out
        return OrderIntent(symbol=sym, side=Side.SELL, qty=round(amount, 6),
                           reason=reason, tag=tag, reduce_only=True)

    def close(self, symbol: str, *, reason: str = "exit", tag: str = "") -> OrderIntent | None:
        """Flatten a symbol completely."""
        held = self.qty(symbol)
        if held <= 1e-9:
            return None
        return OrderIntent(symbol=symbol.upper(), side=Side.SELL, qty=round(held, 6),
                           reason=reason, tag=tag, reduce_only=True)

    def limit(
        self,
        symbol: str,
        side: Side | str,
        qty: float,
        price: float,
        *,
        reason: str = "",
        tag: str = "",
        tif: TimeInForce = TimeInForce.DAY,
        replace_open: bool = False,
    ) -> OrderIntent | None:
        """A passive limit order, size-checked and tick-rounded."""
        sym = symbol.upper()
        s = Side(side) if isinstance(side, str) else side
        px = round(max(price, 0.01), 2)
        qty = round(qty, 6) if sym in self.fractionable else float(math.floor(qty))
        if qty <= 0:
            return None
        if s is Side.SELL:
            held = self.qty(sym)
            qty = min(qty, held)
            if qty <= 1e-9:
                return None
            # Never leave a remainder too small to sell later: offer the lot
            # instead. Otherwise repeated partial exits grind a position down
            # to fractional dust that no order can ever clear.
            if (held - qty) * px < self.min_notional():
                qty = held
        if qty * px < self.min_notional():
            return None
        if s is Side.BUY:
            if qty * px > self.buying_power + 1e-6:
                return None
            self.reserve(qty * px)
        return OrderIntent(
            symbol=sym, side=s, qty=qty, order_type=OrderType.LIMIT, limit_price=px,
            tif=tif, reason=reason, tag=tag, replace_open=replace_open,
            reduce_only=(s is Side.SELL),
        )

    def rebalance_to_weight(
        self,
        symbol: str,
        target_weight: float,
        *,
        tolerance: float = 0.02,
        reason: str = "",
        tag: str = "",
    ) -> OrderIntent | None:
        """One order that moves a position toward `target_weight` of equity.

        The shared sizing path. Returns None when the gap is inside
        `tolerance` (so strategies do not churn on noise), when the trade
        would be below the venue's minimum notional, or when there is no cash.
        """
        sym = symbol.upper()
        px = self.price(sym)
        if px <= 0 or self.equity <= 0:
            return None
        target_weight = max(target_weight, 0.0)          # long-only competition
        target_weight = min(target_weight, self.risk.max_position_pct)
        current = self.weight(sym)
        gap = target_weight - current
        if abs(gap) < tolerance:
            return None
        delta_notional = gap * self.equity
        if delta_notional > 0:
            return self.buy_notional(sym, delta_notional, reason=reason, tag=tag)
        want = abs(delta_notional)
        held_value = self.account.market_value(sym)
        # If we are nearly flattening anyway, flatten -- avoids dust positions
        # that cannot be sold later because they are below the minimum.
        if held_value - want < self.min_notional() * 2:
            return self.close(sym, reason=reason or "flatten", tag=tag)
        return self.sell_qty(sym, want / px, reason=reason, tag=tag)

    def allocate(
        self,
        weights: Mapping[str, float],
        *,
        tolerance: float = 0.02,
        reason: str = "",
        exit_others: bool = True,
        cash_buffer: float = 0.01,
    ) -> list[OrderIntent]:
        """Move the whole book toward a target weight map.

        Sells are emitted before buys so the cash they release is available
        within the same tick, and the requested weights are scaled down if
        they would breach the gross-leverage rail.
        """
        wanted = {s.upper(): max(w, 0.0) for s, w in weights.items() if w > 0}
        cap = max(self.risk.max_gross_leverage - cash_buffer, 0.0)
        total = sum(wanted.values())
        if total > cap > 0:
            wanted = {s: w * cap / total for s, w in wanted.items()}

        sells: list[OrderIntent] = []
        buys: list[OrderIntent] = []
        if exit_others:
            for sym in self.held:
                if sym not in wanted:
                    o = self.close(sym, reason=reason or "not in target set")
                    if o:
                        sells.append(o)

        # Credit the cash the exits release, haircut for slippage. The engine
        # submits sells before buys within a tick, so that cash really is
        # available; resting limit sells are not credited because they may
        # never fill.
        for o in sells:
            if o.order_type is OrderType.MARKET:
                self.credit((o.qty or 0.0) * self.price(o.symbol) * 0.995)

        for sym, w in sorted(wanted.items(), key=lambda kv: -kv[1]):
            o = self.rebalance_to_weight(sym, w, tolerance=tolerance, reason=reason)
            if o is None:
                continue
            if o.side is Side.SELL:
                sells.append(o)
                if o.order_type is OrderType.MARKET:
                    self.credit((o.qty or 0.0) * self.price(sym) * 0.995)
                continue
            buys.append(o)
        return sells + buys

    def flatten_all(self, *, reason: str = "flatten") -> list[OrderIntent]:
        out = []
        for sym in self.held:
            o = self.close(sym, reason=reason)
            if o:
                out.append(o)
        return out


# --------------------------------------------------------------------------- #
# strategy
# --------------------------------------------------------------------------- #


class Strategy(abc.ABC):
    """Base class for every competitor."""

    #: How often the engine should call `on_tick`, in seconds.
    DEFAULT_TICK_SECONDS: int = 300
    #: Declared parameter schema. Subclasses override.
    PARAMS: tuple[Param, ...] = ()
    #: Short description surfaced in reports.
    DESCRIPTION: str = ""

    def __init__(self, team: TeamConfig, params: Mapping[str, Any] | None = None):
        self.team = team
        self.key = team.key
        self.name = team.name
        try:
            self.p = Params(self.PARAMS, params if params is not None else team.params)
        except ValueError as e:
            raise ValueError(f"team {team.key}: {e}") from e
        self.log = logging.getLogger(f"competition.strategy.{team.key}")
        self._fills: list[Fill] = []

    # -- declared behaviour ------------------------------------------------ #

    @property
    def tick_seconds(self) -> int:
        return int(self.p.get("tick_seconds", self.DEFAULT_TICK_SECONDS))

    @property
    def warmup_bars(self) -> int:
        return int(self.p.get("min_bars_required", 50))

    @abc.abstractmethod
    def on_tick(self, ctx: StrategyContext) -> list[OrderIntent]:
        """Return the orders this strategy wants placed right now.

        Contract: pure with respect to the outside world. Read `ctx`, mutate
        `ctx.state`, return intents. Do not submit orders, sleep, or touch the
        network -- the engine owns all of that, and a strategy that blocks
        gets timed out and skipped for the tick.
        """

    # -- optional hooks ---------------------------------------------------- #

    def on_round_start(self, ctx: StrategyContext) -> None:
        """Called once, before the first tick of a round."""

    def on_round_end(self, ctx: StrategyContext) -> None:
        """Called once, after the final tick (post-liquidation)."""

    def on_fill(self, fill: Fill, ctx: StrategyContext) -> None:
        """Called for each fill the engine observes for this team."""
        self._fills.append(fill)

    def on_session_start(self, ctx: StrategyContext) -> None:
        """Called on the first tick of each trading session."""

    def on_session_end(self, ctx: StrategyContext) -> None:
        """Called on the last tick of each trading session."""

    # -- learning persistence --------------------------------------------- #

    def state_dict(self) -> dict[str, Any]:
        """Serialisable learned state, persisted between runs (see QLearner)."""
        return {}

    def load_state(self, blob: Mapping[str, Any]) -> None:
        """Restore what `state_dict` produced."""

    # -- introspection ----------------------------------------------------- #

    def describe(self) -> str:
        lines = [f"{self.name} [{self.key}] tick={self.tick_seconds}s warmup={self.warmup_bars}"]
        if self.DESCRIPTION:
            lines.append("  " + " ".join(self.DESCRIPTION.split()))
        for name, value, doc in self.p.document():
            lines.append(f"    {name:<28} {value!r:<24} {doc}")
        return "\n".join(lines)

    # -- small shared helpers --------------------------------------------- #

    def sync_bar_clock(self, ctx: StrategyContext) -> bool:
        """Advance the shared bar counter when a new primary bar has printed.

        Ticks can fire several times inside one bar (the scalper runs every
        20s on 5-minute bars). Bar-based logic -- cooldowns, time stops, "held
        for N bars" -- must count bars, not ticks, or the same parameter would
        mean different things to different teams. Returns True on the tick
        that opens a new bar.
        """
        latest = None
        for sym in ctx.universe:
            series = ctx.bars(sym)
            if series and (latest is None or series[-1].ts > latest):
                latest = series[-1].ts
        if latest is None:
            return False
        stamp = latest.isoformat()
        if ctx.state.get("_last_bar_ts") == stamp:
            return False
        ctx.state["_last_bar_ts"] = stamp
        ctx.advance_bar_clock()
        return True

    @staticmethod
    def atr_stop(bars: Sequence[Bar], mult: float, period: int = 14) -> float:
        """Long stop price `mult` ATRs below the last close."""
        if not bars:
            return 0.0
        return max(bars[-1].close - mult * ind.atr(bars, period), 0.01)

    @staticmethod
    def inverse_vol_weight(
        bars: Sequence[Bar], target_vol: float, *, base: float, cap: float, period: int = 20
    ) -> float:
        """Scale `base` so the position's per-bar vol is near `target_vol`."""
        if not bars:
            return 0.0
        vol = ind.realized_vol([b.close for b in bars], period)
        if vol <= 1e-6:
            return min(base, cap)
        return max(min(base * (target_vol / vol), cap), 0.0)

    def trail_stop(
        self, ctx: StrategyContext, symbol: str, mult: float, period: int = 14,
        key: str = "stop",
    ) -> float:
        """Maintain a ratcheting ATR trailing stop in per-symbol state."""
        bars = ctx.bars(symbol)
        if not bars:
            return float(ctx.sym_state(symbol).get(key, 0.0))
        candidate = self.atr_stop(bars, mult, period)
        st = ctx.sym_state(symbol)
        current = float(st.get(key, 0.0))
        st[key] = max(current, candidate)      # ratchet: never loosens
        return st[key]
