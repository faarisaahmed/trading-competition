"""Core value types shared by the data feed, strategies, brokers and engine.

Everything here is a frozen (or near-frozen) dataclass with no behaviour beyond
cheap derived properties. Strategies receive these and nothing else -- they
never see a live network handle, which is what makes the whole competition
replayable from the ledger.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

UTC = timezone.utc


def utcnow() -> datetime:
    return datetime.now(tz=UTC)


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

    def flip(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


class OrderStatus(str, Enum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"
    PENDING = "pending"

    @property
    def is_open(self) -> bool:
        return self in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING)

    @property
    def is_terminal(self) -> bool:
        return not self.is_open


@dataclass(frozen=True, slots=True)
class Bar:
    """One OHLCV bar. `ts` is the bar's *open* timestamp, always UTC."""

    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int = 0
    vwap: float = 0.0

    @property
    def typical(self) -> float:
        return (self.high + self.low + self.close) / 3.0

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def ret(self) -> float:
        """Open-to-close return of this bar."""
        return (self.close / self.open - 1.0) if self.open > 0 else 0.0

    @property
    def dollar_volume(self) -> float:
        return self.volume * (self.vwap or self.close)

    def as_row(self) -> tuple:
        return (self.symbol, self.ts.isoformat(), self.open, self.high, self.low,
                self.close, self.volume, self.trade_count, self.vwap)


@dataclass(frozen=True, slots=True)
class Quote:
    """Top-of-book snapshot."""

    symbol: str
    ts: datetime
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.ask or self.bid

    @property
    def spread(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return max(self.ask - self.bid, 0.0)
        return 0.0

    @property
    def relative_spread(self) -> float:
        m = self.mid
        return (self.spread / m) if m > 0 else math.inf

    @property
    def microprice(self) -> float:
        """Size-weighted mid: leans toward the side with less resting size."""
        bs, asz = self.bid_size, self.ask_size
        if self.bid > 0 and self.ask > 0 and (bs + asz) > 0:
            return (self.bid * asz + self.ask * bs) / (bs + asz)
        return self.mid

    @property
    def imbalance(self) -> float:
        """(bid_size - ask_size) / total, in [-1, 1]. Positive = buy pressure."""
        total = self.bid_size + self.ask_size
        return ((self.bid_size - self.ask_size) / total) if total > 0 else 0.0

    @property
    def is_sane(self) -> bool:
        return self.bid > 0 and self.ask > 0 and self.ask >= self.bid


@dataclass(frozen=True, slots=True)
class Trade:
    symbol: str
    ts: datetime
    price: float
    size: float


@dataclass(frozen=True, slots=True)
class NewsItem:
    id: str
    ts: datetime
    headline: str
    summary: str
    source: str
    symbols: tuple[str, ...]
    url: str = ""

    @property
    def text(self) -> str:
        return f"{self.headline}. {self.summary}".strip()


@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float

    @property
    def is_long(self) -> bool:
        return self.qty > 0

    @property
    def is_short(self) -> bool:
        return self.qty < 0

    @property
    def market_value(self) -> float:
        return self.qty * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.qty * self.avg_entry_price

    @property
    def unrealized_pl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def unrealized_plpc(self) -> float:
        cb = abs(self.cost_basis)
        return (self.unrealized_pl / cb) if cb > 0 else 0.0


@dataclass(frozen=True, slots=True)
class Account:
    cash: float
    equity: float
    buying_power: float
    positions: tuple[Position, ...] = ()

    def position(self, symbol: str) -> Position | None:
        for p in self.positions:
            if p.symbol == symbol:
                return p
        return None

    def qty(self, symbol: str) -> float:
        p = self.position(symbol)
        return p.qty if p else 0.0

    def market_value(self, symbol: str) -> float:
        p = self.position(symbol)
        return p.market_value if p else 0.0

    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.market_value) for p in self.positions)

    @property
    def net_exposure(self) -> float:
        return sum(p.market_value for p in self.positions)

    @property
    def leverage(self) -> float:
        return (self.gross_exposure / self.equity) if self.equity > 0 else 0.0

    @property
    def held_symbols(self) -> tuple[str, ...]:
        return tuple(p.symbol for p in self.positions if p.qty != 0)


@dataclass(slots=True)
class OrderIntent:
    """What a strategy *wants* to do. The engine validates before submitting.

    Exactly one of `qty` or `notional` must be set. `notional` is convenient
    for fractional sizing ("put $412 into AAPL"); the engine converts it to a
    share count using the current quote when the broker or order type cannot
    take a notional directly.
    """

    symbol: str
    side: Side
    qty: float | None = None
    notional: float | None = None
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    tif: TimeInForce = TimeInForce.DAY
    reason: str = ""
    tag: str = ""
    # Strategies that manage their own resting quotes set this to have the
    # engine cancel any of the team's live orders on this symbol first.
    replace_open: bool = False
    # Set by a strategy when the intent is an exit; guardrails allow exits
    # through some risk blocks that would stop new risk being added.
    reduce_only: bool = False

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper().strip()
        if isinstance(self.side, str):
            self.side = Side(self.side)
        if isinstance(self.order_type, str):
            self.order_type = OrderType(self.order_type)
        if isinstance(self.tif, str):
            self.tif = TimeInForce(self.tif)
        if (self.qty is None) == (self.notional is None):
            raise ValueError(
                f"OrderIntent for {self.symbol} needs exactly one of qty/notional "
                f"(got qty={self.qty}, notional={self.notional})"
            )
        if self.order_type is OrderType.LIMIT and not self.limit_price:
            raise ValueError(f"limit order for {self.symbol} requires limit_price")

    def describe(self) -> str:
        size = f"{self.qty:g}sh" if self.qty is not None else f"${self.notional:,.2f}"
        px = f" @ {self.limit_price:.4f}" if self.limit_price else " @ mkt"
        return f"{self.side.value.upper()} {size} {self.symbol}{px}"


@dataclass(slots=True)
class Order:
    """A submitted order as the broker sees it."""

    id: str
    client_order_id: str
    symbol: str
    side: Side
    qty: float
    filled_qty: float = 0.0
    filled_avg_price: float = 0.0
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    tif: TimeInForce = TimeInForce.DAY
    status: OrderStatus = OrderStatus.NEW
    submitted_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    #: Carried over from the originating intent, so the ledger records *why*
    #: a strategy placed each order, not just what it placed.
    reason: str = ""
    tag: str = ""

    @property
    def leaves_qty(self) -> float:
        return max(self.qty - self.filled_qty, 0.0)

    @property
    def notional_filled(self) -> float:
        return self.filled_qty * self.filled_avg_price


@dataclass(frozen=True, slots=True)
class Fill:
    order_id: str
    symbol: str
    side: Side
    qty: float
    price: float
    ts: datetime
    fee: float = 0.0

    @property
    def notional(self) -> float:
        return self.qty * self.price

    @property
    def signed_qty(self) -> float:
        return self.qty * self.side.sign


@dataclass(frozen=True, slots=True)
class EquityPoint:
    ts: datetime
    equity: float
    cash: float
    gross_exposure: float


class RejectReason(str, Enum):
    """Why the engine refused to forward an intent to the broker."""

    OUTSIDE_UNIVERSE = "outside_universe"
    SHORTING_DISABLED = "shorting_disabled"
    INSUFFICIENT_CASH = "insufficient_cash"
    LEVERAGE_CAP = "leverage_cap"
    POSITION_CAP = "position_cap"
    ORDER_RATE_LIMIT = "order_rate_limit"
    MIN_NOTIONAL = "min_notional"
    MAX_NOTIONAL = "max_notional"
    NO_QUOTE = "no_quote"
    STALE_QUOTE = "stale_quote"
    CROSSED_QUOTE = "crossed_quote"
    LIMIT_TOO_FAR = "limit_too_far"
    MARKET_CLOSED = "market_closed"
    ASSET_NOT_TRADABLE = "asset_not_tradable"
    ASSET_HALTED = "asset_halted"
    DUPLICATE_INTENT = "duplicate_intent"
    OVERSELL = "oversell"
    KILL_SWITCH = "kill_switch"
    STRATEGY_ERROR = "strategy_error"
    STRATEGY_TIMEOUT = "strategy_timeout"
    BROKER_ERROR = "broker_error"
    NAN_SIZE = "nan_size"


@dataclass(frozen=True, slots=True)
class Rejection:
    intent: OrderIntent
    reason: RejectReason
    detail: str = ""


def mid_or_close(quote: Quote | None, bars: Sequence[Bar] | None) -> float:
    """Best available reference price: live mid, else last bar close, else 0."""
    if quote is not None and quote.mid > 0:
        return quote.mid
    if bars:
        return bars[-1].close
    return 0.0


def closes(bars: Iterable[Bar]) -> list[float]:
    return [b.close for b in bars]
