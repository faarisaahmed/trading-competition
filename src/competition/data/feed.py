"""Market-data feeds: one live (Alpaca), one replay (offline / backtest).

Both produce a `MarketSnapshot`. The engine builds ONE snapshot per tick from
whichever feed is active and hands the same object to every team, so the
fairness guarantee is identical in live and replay mode.

The live feed is deliberately stingy with requests: eight teams sharing a
200 req/min budget means bars are refetched only when a bar boundary has
actually passed, quotes are batched 200-at-a-time, and news is polled on its
own slower cadence.
"""

from __future__ import annotations

import abc
import bisect
import logging
import re
import threading
import time as _time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta

from ..types import UTC, Bar, NewsItem, Quote, utcnow
from .calendar import MarketCalendar
from .snapshot import MarketSnapshot

log = logging.getLogger("competition.feed")

_TF = re.compile(r"^(\d+)(Min|Hour|Day|Week|Month)$")


def timeframe_seconds(tf: str) -> int:
    """Alpaca timeframe string -> seconds. '5Min' -> 300."""
    m = _TF.match(tf)
    if not m:
        raise ValueError(f"bad timeframe {tf!r}")
    n, unit = int(m.group(1)), m.group(2)
    mult = {"Min": 60, "Hour": 3600, "Day": 86400, "Week": 604800, "Month": 2592000}[unit]
    return n * mult


class Feed(abc.ABC):
    """Source of a `MarketSnapshot`."""

    @abc.abstractmethod
    def snapshot(self, universe: Sequence[str], now: datetime | None = None) -> MarketSnapshot:
        ...

    @property
    @abc.abstractmethod
    def calendar(self) -> MarketCalendar:
        ...

    def prime(self, universe: Sequence[str]) -> None:
        """Warm caches before the first tick so tick 1 isn't data-starved."""

    def close(self) -> None:
        ...


# --------------------------------------------------------------------------- #
# live
# --------------------------------------------------------------------------- #


class AlpacaFeed(Feed):
    """Live Alpaca data with incremental bar caching.

    Not a per-team object: the engine constructs exactly one and shares it.
    """

    def __init__(
        self,
        reader,
        *,
        calendar: MarketCalendar | None = None,
        primary_timeframe: str = "5Min",
        fast_timeframe: str = "1Min",
        slow_timeframe: str = "1Day",
        history_bars_primary: int = 500,
        history_bars_fast: int = 240,
        history_days_slow: int = 400,
        news_lookback_hours: int = 72,
        news_limit_per_symbol: int = 50,
        news_poll_seconds: float = 120.0,
        quote_max_age_seconds: float = 120.0,
    ):
        self.reader = reader
        self._calendar = calendar or MarketCalendar.from_alpaca(reader.client)
        self.primary_tf = primary_timeframe
        self.fast_tf = fast_timeframe
        self.slow_tf = slow_timeframe
        self.history_bars_primary = history_bars_primary
        self.history_bars_fast = history_bars_fast
        self.history_days_slow = history_days_slow
        self.news_lookback_hours = news_lookback_hours
        self.news_limit_per_symbol = news_limit_per_symbol
        self.news_poll_seconds = news_poll_seconds
        self.quote_max_age_seconds = quote_max_age_seconds

        self._bars: dict[str, dict[str, list[Bar]]] = {"primary": {}, "fast": {}, "daily": {}}
        self._bar_fetched_at: dict[str, float] = {}
        self._news: dict[str, list[NewsItem]] = {}
        self._news_fetched_at = 0.0
        self._lock = threading.Lock()

    @property
    def calendar(self) -> MarketCalendar:
        return self._calendar

    # -- bar caching ------------------------------------------------------- #

    def _cache_key(self, kind: str) -> str:
        return kind

    def _needs_bar_refresh(self, kind: str, now: datetime) -> bool:
        tf = {"primary": self.primary_tf, "fast": self.fast_tf, "daily": self.slow_tf}[kind]
        period = timeframe_seconds(tf)
        last = self._bar_fetched_at.get(self._cache_key(kind), 0.0)
        if last == 0.0:
            return True
        # Refetch once per bar period (daily bars: at most every 15 minutes).
        interval = min(period, 900) if kind == "daily" else max(period * 0.5, 30)
        return (_time.monotonic() - last) >= interval

    def _fetch_bars(self, kind: str, symbols: Sequence[str], now: datetime) -> None:
        tf = {"primary": self.primary_tf, "fast": self.fast_tf, "daily": self.slow_tf}[kind]
        if kind == "daily":
            start = now - timedelta(days=int(self.history_days_slow * 1.5) + 10)
            limit = self.history_days_slow
        else:
            bars = self.history_bars_primary if kind == "primary" else self.history_bars_fast
            # Reach back over enough calendar days to cover `bars` session bars.
            per_session = max(int(6.5 * 3600 / timeframe_seconds(tf)), 1)
            sessions_needed = bars / per_session + 2
            start = now - timedelta(days=int(sessions_needed * 1.6) + 4)
            limit = bars
        try:
            fetched = self.reader.bars(symbols, tf, limit=limit, start=start, end=now)
        except Exception as e:  # noqa: BLE001 -- a stale cache beats a dead engine
            log.warning("bar fetch failed for %s (%s): %s", kind, tf, e)
            return
        table = self._bars[kind]
        for sym, rows in fetched.items():
            if rows:
                table[sym] = rows[-limit:]
        self._bar_fetched_at[self._cache_key(kind)] = _time.monotonic()

    # -- news -------------------------------------------------------------- #

    def _fetch_news(self, symbols: Sequence[str]) -> None:
        try:
            items = self.reader.news(
                symbols, hours=self.news_lookback_hours, limit=self.news_limit_per_symbol
            )
        except Exception as e:  # noqa: BLE001
            log.warning("news fetch failed: %s", e)
            return
        table: dict[str, list[NewsItem]] = defaultdict(list)
        wanted = {s.upper() for s in symbols}
        for item in items:
            for sym in item.symbols:
                if sym in wanted:
                    table[sym].append(item)
        for rows in table.values():
            rows.sort(key=lambda n: n.ts, reverse=True)
        self._news = dict(table)
        self._news_fetched_at = _time.monotonic()

    # -- snapshot ---------------------------------------------------------- #

    def prime(self, universe: Sequence[str]) -> None:
        now = utcnow()
        syms = [s.upper() for s in universe]
        with self._lock:
            for kind in ("daily", "primary", "fast"):
                self._fetch_bars(kind, syms, now)
            self._fetch_news(syms)

    def snapshot(self, universe: Sequence[str], now: datetime | None = None) -> MarketSnapshot:
        now = (now or utcnow()).astimezone(UTC)
        syms = [s.upper() for s in dict.fromkeys(universe)]
        with self._lock:
            for kind in ("primary", "fast", "daily"):
                missing_cached = [s for s in syms if s not in self._bars[kind]]
                if self._needs_bar_refresh(kind, now):
                    self._fetch_bars(kind, syms, now)
                elif missing_cached:
                    self._fetch_bars(kind, missing_cached, now)
            if (_time.monotonic() - self._news_fetched_at) >= self.news_poll_seconds:
                self._fetch_news(syms)

            try:
                quotes = self.reader.latest_quotes(syms)
            except Exception as e:  # noqa: BLE001
                log.warning("quote fetch failed: %s", e)
                quotes = {}

            missing: set[str] = set()
            clean: dict[str, Quote] = {}
            for sym in syms:
                q = quotes.get(sym)
                if q is None or not q.is_sane:
                    missing.add(sym)
                    continue
                age = (now - q.ts).total_seconds()
                if age > self.quote_max_age_seconds:
                    # Keep it (strategies can see the age) but flag as missing
                    # so guardrails will not trade on it.
                    missing.add(sym)
                clean[sym] = q

            return MarketSnapshot(
                ts=now,
                session=self._calendar.session(now),
                universe=tuple(syms),
                bars={s: tuple(self._bars["primary"].get(s, ())) for s in syms},
                fast_bars={s: tuple(self._bars["fast"].get(s, ())) for s in syms},
                daily_bars={s: tuple(self._bars["daily"].get(s, ())) for s in syms},
                quotes=clean,
                news={s: tuple(self._news.get(s, ())) for s in syms},
                missing=frozenset(missing),
            )


# --------------------------------------------------------------------------- #
# replay
# --------------------------------------------------------------------------- #


class ReplayFeed(Feed):
    """Serves history as of `now` with no lookahead.

    Bars are pre-sorted once; each tick uses `bisect` to find the last bar
    whose *close* time is <= now, which is the only bar a live strategy could
    have seen. Quotes are synthesised from that bar's close with a
    deterministic spread, so the simulated broker has something to fill on.
    """

    def __init__(
        self,
        primary: Mapping[str, Sequence[Bar]],
        *,
        fast: Mapping[str, Sequence[Bar]] | None = None,
        daily: Mapping[str, Sequence[Bar]] | None = None,
        news: Mapping[str, Sequence[NewsItem]] | None = None,
        calendar: MarketCalendar | None = None,
        primary_timeframe: str = "5Min",
        fast_timeframe: str = "1Min",
        spread_bps: float = 4.0,
        quote_size: float = 500.0,
        history_bars: int = 500,
    ):
        self._calendar = calendar or MarketCalendar()
        self.primary_tf_seconds = timeframe_seconds(primary_timeframe)
        self.fast_tf_seconds = timeframe_seconds(fast_timeframe)
        self.spread_bps = spread_bps
        self.quote_size = quote_size
        self.history_bars = history_bars
        self._store: dict[str, dict[str, list[Bar]]] = {
            "primary": {k.upper(): sorted(v, key=lambda b: b.ts) for k, v in primary.items()},
            "fast": {k.upper(): sorted(v, key=lambda b: b.ts) for k, v in (fast or {}).items()},
            "daily": {k.upper(): sorted(v, key=lambda b: b.ts) for k, v in (daily or {}).items()},
        }
        self._index: dict[str, dict[str, list[datetime]]] = {
            kind: {sym: [b.ts for b in rows] for sym, rows in table.items()}
            for kind, table in self._store.items()
        }
        self._news = {k.upper(): sorted(v, key=lambda n: n.ts) for k, v in (news or {}).items()}
        self._news_index = {k: [n.ts for n in v] for k, v in self._news.items()}
        # (kind, symbol) -> (end_index, sliced tuple). The engine ticks as fast
        # as its quickest team (20s for the market maker) but bars only become
        # visible every 5 minutes, so the same window gets rebuilt ~15 times
        # per bar. On an 80-symbol Round 3 that was ~190M list-element copies
        # per round; memoising on the bisect index removes essentially all of
        # it and changes nothing about what a strategy sees.
        self._window_cache: dict[tuple[str, str], tuple[int, tuple[Bar, ...]]] = {}

    @property
    def calendar(self) -> MarketCalendar:
        return self._calendar

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self._store["primary"]))

    def span(self) -> tuple[datetime, datetime] | None:
        stamps = [ts for idx in self._index["primary"].values() for ts in (idx[:1] + idx[-1:])]
        return (min(stamps), max(stamps)) if stamps else None

    def _bar_period(self, kind: str) -> int:
        return {"primary": self.primary_tf_seconds,
                "fast": self.fast_tf_seconds,
                "daily": 86400}[kind]

    def _visible(self, kind: str, sym: str, now: datetime, limit: int) -> tuple[Bar, ...]:
        rows = self._store[kind].get(sym)
        if not rows:
            return ()
        stamps = self._index[kind][sym]
        # A bar stamped at t is only complete (and therefore visible) at
        # t + period. This is the single most important line for backtest
        # honesty -- without it every strategy sees the future.
        cutoff = now - timedelta(seconds=self._bar_period(kind))
        i = bisect.bisect_right(stamps, cutoff)
        if i <= 0:
            return ()
        key = (kind, sym)
        cached = self._window_cache.get(key)
        if cached is not None and cached[0] == i and len(cached[1]) >= min(i, limit):
            return cached[1] if len(cached[1]) <= limit else cached[1][-limit:]
        window = tuple(rows[max(i - limit, 0):i])
        self._window_cache[key] = (i, window)
        return window

    def _synth_quote(self, sym: str, bar: Bar, now: datetime) -> Quote:
        half = bar.close * (self.spread_bps / 2.0) / 10_000.0
        return Quote(
            symbol=sym,
            ts=now,
            bid=round(max(bar.close - half, 0.01), 4),
            ask=round(bar.close + half, 4),
            bid_size=self.quote_size,
            ask_size=self.quote_size,
        )

    def snapshot(self, universe: Sequence[str], now: datetime | None = None) -> MarketSnapshot:
        if now is None:
            raise ValueError("ReplayFeed requires an explicit `now`")
        now = now.astimezone(UTC)
        syms = [s.upper() for s in dict.fromkeys(universe)]
        bars, fast, daily, quotes, news = {}, {}, {}, {}, {}
        missing: set[str] = set()
        for sym in syms:
            p = self._visible("primary", sym, now, self.history_bars)
            bars[sym] = p
            fast[sym] = self._visible("fast", sym, now, self.history_bars)
            daily[sym] = self._visible("daily", sym, now, 400)
            ref = p[-1] if p else (fast[sym][-1] if fast[sym] else (daily[sym][-1] if daily[sym] else None))
            if ref is None or ref.close <= 0:
                missing.add(sym)
            else:
                quotes[sym] = self._synth_quote(sym, ref, now)
            items = self._news.get(sym)
            if items:
                j = bisect.bisect_right(self._news_index[sym], now)
                news[sym] = tuple(items[max(j - 100, 0):j])
            else:
                news[sym] = ()
        return MarketSnapshot(
            ts=now,
            session=self._calendar.session(now),
            universe=tuple(syms),
            bars=bars, fast_bars=fast, daily_bars=daily,
            quotes=quotes, news=news,
            missing=frozenset(missing),
        )

    def quote_source(self, clock):
        """A `QuoteSource` for `SimulatedBroker`, driven by a shared clock.

        `clock` is a zero-arg callable returning the engine's current time, so
        broker fills and strategy views stay on the same instant.
        """

        def source(symbol: str) -> Quote | None:
            sym = symbol.upper()
            now = clock()
            for kind in ("fast", "primary", "daily"):
                vis = self._visible(kind, sym, now, 1)
                if vis:
                    return self._synth_quote(sym, vis[-1], now)
            return None

        return source

    # -- construction helpers --------------------------------------------- #

    @classmethod
    def from_alpaca(
        cls,
        reader,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        *,
        primary_timeframe: str = "5Min",
        fast_timeframe: str = "1Min",
        with_news: bool = True,
        calendar: MarketCalendar | None = None,
        **kw,
    ) -> ReplayFeed:
        """Download a window once and replay it offline -- the backtest path."""
        primary = reader.bars(symbols, primary_timeframe, limit=10000, start=start, end=end)
        fast = reader.bars(symbols, fast_timeframe, limit=10000, start=start, end=end)
        daily = reader.bars(
            symbols, "1Day", limit=500, start=start - timedelta(days=600), end=end
        )
        news_table: dict[str, list[NewsItem]] = defaultdict(list)
        if with_news:
            hours = max(int((end - start).total_seconds() / 3600) + 72, 72)
            try:
                for item in reader.news(symbols, hours=hours, limit=50):
                    for sym in item.symbols:
                        news_table[sym].append(item)
            except Exception as e:  # noqa: BLE001
                log.warning("historical news unavailable: %s", e)
        return cls(
            primary, fast=fast, daily=daily, news=news_table,
            primary_timeframe=primary_timeframe, fast_timeframe=fast_timeframe,
            calendar=calendar, **kw,
        )
