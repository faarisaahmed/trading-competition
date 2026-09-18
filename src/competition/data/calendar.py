"""US equity market calendar and session state.

Two sources, same interface:
  * `MarketCalendar.from_alpaca(client)` -- authoritative, uses /v2/calendar
    and /v2/clock, so half-days and unscheduled closures are handled.
  * `MarketCalendar()` -- offline NYSE rules, computed (not hard-coded) for
    any year, so backtests and CI need no network.

The engine needs this for three things: knowing when to tick, knowing how
long until the close (the scalper and the round-end liquidator both care),
and counting the trading sessions inside a seven-calendar-day round.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from ..types import UTC

NY = ZoneInfo("America/New_York")

REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)


# --------------------------------------------------------------------------- #
# holiday computation
# --------------------------------------------------------------------------- #


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th `weekday` (Mon=0) of a month; n<0 counts back from the end."""
    if n > 0:
        d = date(year, month, 1)
        offset = (weekday - d.weekday()) % 7
        return d + timedelta(days=offset + 7 * (n - 1))
    nxt = date(year + (month == 12), (month % 12) + 1, 1)
    d = nxt - timedelta(days=1)
    offset = (d.weekday() - weekday) % 7
    return d - timedelta(days=offset + 7 * (-n - 1))


def _easter(year: int) -> date:
    """Anonymous Gregorian algorithm."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _observed(d: date) -> date:
    """NYSE shifts Saturday holidays to Friday and Sunday holidays to Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@functools.lru_cache(maxsize=64)
def nyse_holidays(year: int) -> frozenset[date]:
    """Full-day NYSE closures for `year`, computed from the published rules."""
    out = {
        _observed(date(year, 1, 1)),                       # New Year's Day
        _nth_weekday(year, 1, 0, 3),                       # MLK Jr. Day
        _nth_weekday(year, 2, 0, 3),                       # Washington's Birthday
        _easter(year) - timedelta(days=2),                 # Good Friday
        _nth_weekday(year, 5, 0, -1),                      # Memorial Day
        _observed(date(year, 7, 4)),                       # Independence Day
        _nth_weekday(year, 9, 0, 1),                       # Labor Day
        _nth_weekday(year, 11, 3, 4),                      # Thanksgiving
        _observed(date(year, 12, 25)),                     # Christmas
    }
    if year >= 2022:
        out.add(_observed(date(year, 6, 19)))              # Juneteenth
    return frozenset(out)


@functools.lru_cache(maxsize=64)
def nyse_early_closes(year: int) -> frozenset[date]:
    """1:00pm ET closes: day after Thanksgiving, Christmas Eve, July 3."""
    out: set[date] = set()
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    out.add(thanksgiving + timedelta(days=1))
    for d in (date(year, 12, 24), date(year, 7, 3)):
        if d.weekday() < 5 and d not in nyse_holidays(year):
            out.add(d)
    return frozenset(d for d in out if d.weekday() < 5)


# --------------------------------------------------------------------------- #
# session state
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SessionInfo:
    """Where `now` sits relative to the trading session."""

    now: datetime
    session_date: date
    is_trading_day: bool
    is_open: bool
    open_at: datetime | None
    close_at: datetime | None
    is_early_close: bool = False
    next_open: datetime | None = None

    @property
    def minutes_to_close(self) -> float:
        if not self.close_at:
            return float("inf")
        return (self.close_at - self.now).total_seconds() / 60.0

    @property
    def minutes_since_open(self) -> float:
        if not self.open_at:
            return 0.0
        return max((self.now - self.open_at).total_seconds() / 60.0, 0.0)

    @property
    def seconds_to_open(self) -> float:
        target = self.open_at if (self.open_at and self.now < self.open_at) else self.next_open
        if not target:
            return float("inf")
        return max((target - self.now).total_seconds(), 0.0)

    @property
    def in_opening_range(self) -> bool:
        return self.is_open and self.minutes_since_open <= 30.0

    @property
    def in_closing_range(self) -> bool:
        return self.is_open and self.minutes_to_close <= 30.0

    def within_minutes_of_close(self, minutes: float) -> bool:
        return self.is_open and self.minutes_to_close <= minutes

    @property
    def ny_time(self) -> time:
        return self.now.astimezone(NY).time()


# --------------------------------------------------------------------------- #
# calendar
# --------------------------------------------------------------------------- #


class MarketCalendar:
    """Session lookups, offline by default and Alpaca-backed when available."""

    def __init__(self, *, sessions: dict[date, tuple[time, time]] | None = None):
        #: Explicit overrides (populated from /v2/calendar when online).
        self._sessions: dict[date, tuple[time, time]] = dict(sessions or {})

    # -- construction ------------------------------------------------------ #

    @classmethod
    def from_alpaca(cls, client, *, start: date | None = None, days: int = 400) -> MarketCalendar:
        """Pull the real calendar. Falls back to offline rules on any failure."""
        cal = cls()
        s = start or (date.today() - timedelta(days=30))
        e = s + timedelta(days=days)
        try:
            rows = client.calendar(s.isoformat(), e.isoformat())
        except Exception:  # noqa: BLE001 -- offline rules are a fine fallback
            return cal
        for row in rows or []:
            try:
                d = date.fromisoformat(str(row["date"]))
                o = time.fromisoformat(str(row.get("open", "09:30")))
                c = time.fromisoformat(str(row.get("close", "16:00")))
            except (KeyError, ValueError):
                continue
            cal._sessions[d] = (o, c)
        return cal

    # -- day queries ------------------------------------------------------- #

    def is_trading_day(self, d: date) -> bool:
        if d in self._sessions:
            return True
        if self._sessions and d.year in {k.year for k in self._sessions}:
            # We have authoritative data for this year and this date is absent.
            return False
        return d.weekday() < 5 and d not in nyse_holidays(d.year)

    def session_times(self, d: date) -> tuple[datetime, datetime] | None:
        """(open, close) as timezone-aware datetimes, or None if closed."""
        if not self.is_trading_day(d):
            return None
        if d in self._sessions:
            o, c = self._sessions[d]
        else:
            o = REGULAR_OPEN
            c = EARLY_CLOSE if d in nyse_early_closes(d.year) else REGULAR_CLOSE
        return (
            datetime.combine(d, o, tzinfo=NY).astimezone(UTC),
            datetime.combine(d, c, tzinfo=NY).astimezone(UTC),
        )

    def is_early_close(self, d: date) -> bool:
        times = self.session_times(d)
        if not times:
            return False
        return times[1].astimezone(NY).time() < REGULAR_CLOSE

    def next_trading_day(self, d: date, *, inclusive: bool = False) -> date:
        cur = d if inclusive else d + timedelta(days=1)
        for _ in range(400):
            if self.is_trading_day(cur):
                return cur
            cur += timedelta(days=1)
        raise RuntimeError(f"no trading day found after {d}")

    def previous_trading_day(self, d: date, *, inclusive: bool = False) -> date:
        cur = d if inclusive else d - timedelta(days=1)
        for _ in range(400):
            if self.is_trading_day(cur):
                return cur
            cur -= timedelta(days=1)
        raise RuntimeError(f"no trading day found before {d}")

    def trading_days(self, start: date, end: date) -> list[date]:
        out, cur = [], start
        while cur <= end:
            if self.is_trading_day(cur):
                out.append(cur)
            cur += timedelta(days=1)
        return out

    def count_sessions(self, start: date, end: date) -> int:
        return len(self.trading_days(start, end))

    # -- session state ----------------------------------------------------- #

    def session(self, now: datetime | None = None) -> SessionInfo:
        now = (now or datetime.now(tz=UTC)).astimezone(UTC)
        d = now.astimezone(NY).date()
        times = self.session_times(d)
        if times is None:
            nxt = self.next_trading_day(d)
            nxt_times = self.session_times(nxt)
            return SessionInfo(
                now=now, session_date=d, is_trading_day=False, is_open=False,
                open_at=None, close_at=None,
                next_open=nxt_times[0] if nxt_times else None,
            )
        open_at, close_at = times
        is_open = open_at <= now < close_at
        next_open = open_at
        if now >= close_at:
            nxt = self.next_trading_day(d)
            nt = self.session_times(nxt)
            next_open = nt[0] if nt else None
        return SessionInfo(
            now=now, session_date=d, is_trading_day=True, is_open=is_open,
            open_at=open_at, close_at=close_at,
            is_early_close=self.is_early_close(d), next_open=next_open,
        )

    def tick_times(
        self, start: date, end: date, *, seconds: int, include_close: bool = True
    ) -> list[datetime]:
        """Every tick timestamp in [start, end] at `seconds` cadence.

        Used by the replay engine so a backtest visits exactly the instants a
        live run would have.
        """
        out: list[datetime] = []
        for d in self.trading_days(start, end):
            times = self.session_times(d)
            if not times:
                continue
            o, c = times
            t = o
            while t < c:
                out.append(t)
                t += timedelta(seconds=seconds)
            if include_close and (not out or out[-1] != c):
                out.append(c - timedelta(seconds=1))
        return out

    def describe_window(self, start: date, end: date) -> str:
        days = self.trading_days(start, end)
        halves = [d.isoformat() for d in days if self.is_early_close(d)]
        note = f", early closes: {', '.join(halves)}" if halves else ""
        return f"{len(days)} sessions {start}..{end}{note}"
