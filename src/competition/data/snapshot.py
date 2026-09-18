"""The immutable market view handed to every strategy on a tick.

One `MarketSnapshot` is built per tick by the data hub and passed -- the same
object -- to all eight teams. That is the mechanical guarantee behind the
"identical tape" fairness rule: there is exactly one copy of the data, no team
can refresh it, and nobody can see a later print than anyone else.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..types import Bar, NewsItem, Quote
from .calendar import SessionInfo


@dataclass(frozen=True)
class MarketSnapshot:
    """Everything a strategy is allowed to know at time `ts`."""

    ts: datetime
    session: SessionInfo
    universe: tuple[str, ...]
    bars: Mapping[str, tuple[Bar, ...]] = field(default_factory=dict)
    fast_bars: Mapping[str, tuple[Bar, ...]] = field(default_factory=dict)
    daily_bars: Mapping[str, tuple[Bar, ...]] = field(default_factory=dict)
    quotes: Mapping[str, Quote] = field(default_factory=dict)
    news: Mapping[str, tuple[NewsItem, ...]] = field(default_factory=dict)
    #: Symbols the feed could not price this tick -- strategies should skip them.
    missing: frozenset[str] = frozenset()

    # -- prices ------------------------------------------------------------ #

    def quote(self, symbol: str) -> Quote | None:
        return self.quotes.get(symbol.upper())

    def price(self, symbol: str) -> float:
        """Best reference price: live mid, else last primary close, else 0."""
        sym = symbol.upper()
        q = self.quotes.get(sym)
        if q is not None and q.mid > 0:
            return q.mid
        series = self.bars.get(sym) or self.fast_bars.get(sym) or self.daily_bars.get(sym)
        return series[-1].close if series else 0.0

    def tradable(self, symbol: str) -> bool:
        sym = symbol.upper()
        if sym in self.missing:
            return False
        q = self.quotes.get(sym)
        return bool(q and q.is_sane) or self.price(sym) > 0

    # -- series ------------------------------------------------------------ #

    def series(self, symbol: str, kind: str = "primary") -> tuple[Bar, ...]:
        table = {"primary": self.bars, "fast": self.fast_bars, "daily": self.daily_bars}[kind]
        return table.get(symbol.upper(), ())

    def closes(self, symbol: str, kind: str = "primary", n: int | None = None) -> list[float]:
        s = self.series(symbol, kind)
        vals = [b.close for b in s]
        return vals[-n:] if n else vals

    def highs(self, symbol: str, kind: str = "primary", n: int | None = None) -> list[float]:
        s = self.series(symbol, kind)
        vals = [b.high for b in s]
        return vals[-n:] if n else vals

    def lows(self, symbol: str, kind: str = "primary", n: int | None = None) -> list[float]:
        s = self.series(symbol, kind)
        vals = [b.low for b in s]
        return vals[-n:] if n else vals

    def volumes(self, symbol: str, kind: str = "primary", n: int | None = None) -> list[float]:
        s = self.series(symbol, kind)
        vals = [b.volume for b in s]
        return vals[-n:] if n else vals

    def last_bar(self, symbol: str, kind: str = "primary") -> Bar | None:
        s = self.series(symbol, kind)
        return s[-1] if s else None

    def has_history(self, symbol: str, n: int, kind: str = "primary") -> bool:
        return len(self.series(symbol, kind)) >= n

    def ready(self, n: int, kind: str = "primary") -> list[str]:
        """Universe members with at least `n` bars and a usable price."""
        return [s for s in self.universe if self.has_history(s, n, kind) and self.tradable(s)]

    # -- news -------------------------------------------------------------- #

    def news_for(self, symbol: str, *, within_hours: float | None = None) -> tuple[NewsItem, ...]:
        items = self.news.get(symbol.upper(), ())
        if within_hours is None:
            return items
        cutoff = self.ts - timedelta(hours=within_hours)
        return tuple(n for n in items if n.ts >= cutoff)

    # -- construction ------------------------------------------------------ #

    def restricted_to(self, symbols: Sequence[str]) -> MarketSnapshot:
        """A view limited to `symbols`.

        The engine calls this per team so a strategy physically cannot read
        data for a ticker outside the universe it was assigned -- Round 3's
        dealt hands are enforced at the data layer, not just at order entry.
        """
        allowed = tuple(dict.fromkeys(s.upper() for s in symbols))
        keep = set(allowed)
        return MarketSnapshot(
            ts=self.ts,
            session=self.session,
            universe=allowed,
            bars={k: v for k, v in self.bars.items() if k in keep},
            fast_bars={k: v for k, v in self.fast_bars.items() if k in keep},
            daily_bars={k: v for k, v in self.daily_bars.items() if k in keep},
            quotes={k: v for k, v in self.quotes.items() if k in keep},
            news={k: v for k, v in self.news.items() if k in keep},
            missing=frozenset(s for s in self.missing if s in keep),
        )

    def describe(self) -> str:
        return (
            f"<snapshot {self.ts:%Y-%m-%d %H:%M:%S%z} universe={len(self.universe)} "
            f"quoted={len(self.quotes)} bars={sum(len(v) for v in self.bars.values())} "
            f"news={sum(len(v) for v in self.news.values())} open={self.session.is_open}>"
        )
