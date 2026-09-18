"""The season calendar: when each round opens, closes, and what happens between.

The competition is meant to run unattended for three weeks. That needs one
authoritative answer to "what should be happening right now?" -- computed from
the wall clock and the market calendar, never from a human deciding. This
module is that answer.

A round is a *window of trading sessions*, not a span of wall-clock hours: a
round that began Monday 09:30 ends at Friday's close, and the four overnights
in between are not part of anyone's week. Weekends and holidays simply are not
trading time, so they neither advance nor penalise a round.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from .data.calendar import MarketCalendar

Phase = Literal["before", "pre_open", "open", "after_close", "between", "done"]


@dataclass(frozen=True)
class RoundWindow:
    """One round's place on the calendar."""

    round_id: int
    name: str
    start_date: date
    end_date: date
    opens_at: datetime      # UTC: the bell on start_date
    closes_at: datetime     # UTC: the bell on end_date
    sessions: int

    def contains(self, when: datetime) -> bool:
        return self.opens_at <= when < self.closes_at

    def is_past(self, when: datetime) -> bool:
        return when >= self.closes_at

    def is_future(self, when: datetime) -> bool:
        return when < self.opens_at


@dataclass(frozen=True)
class SeasonState:
    """What the competition should be doing at one instant."""

    now: datetime
    phase: Phase
    current: RoundWindow | None
    next: RoundWindow | None

    @property
    def round_id(self) -> int | None:
        return self.current.round_id if self.current else None

    @property
    def seconds_to_start(self) -> float | None:
        """Until the next round's opening bell."""
        if self.next is None:
            return None
        return max(0.0, (self.next.opens_at - self.now).total_seconds())

    @property
    def seconds_to_end(self) -> float | None:
        """Until the running round's final bell."""
        if self.current is None:
            return None
        return max(0.0, (self.current.closes_at - self.now).total_seconds())

    @property
    def is_trading(self) -> bool:
        return self.phase == "open"

    @property
    def is_finished(self) -> bool:
        return self.phase == "done"


class SeasonSchedule:
    """The three round windows, and where the clock sits among them."""

    def __init__(self, windows: list[RoundWindow], calendar: MarketCalendar):
        if not windows:
            raise ValueError("a season needs at least one round")
        self.windows = sorted(windows, key=lambda w: w.opens_at)
        self.calendar = calendar
        for a, b in zip(self.windows, self.windows[1:], strict=False):
            if b.opens_at < a.closes_at:
                raise ValueError(
                    f"round {b.round_id} opens before round {a.round_id} closes; "
                    f"rounds may not overlap"
                )

    # -- construction ------------------------------------------------------ #

    @classmethod
    def build(
        cls,
        *,
        rounds: list[tuple[int, str]],
        first_start: date,
        sessions_per_round: int,
        calendar: MarketCalendar,
        gap_sessions: int = 0,
    ) -> SeasonSchedule:
        """Lay `rounds` end to end, each `sessions_per_round` sessions long.

        `first_start` is nudged forward to the next trading day if it lands on
        a weekend or holiday, so "start Monday" survives a Monday holiday
        without silently losing a session.
        """
        if sessions_per_round < 1:
            raise ValueError("a round needs at least one session")
        windows: list[RoundWindow] = []
        cursor = calendar.next_trading_day(first_start, inclusive=True)
        for rid, name in rounds:
            days = _take_sessions(calendar, cursor, sessions_per_round)
            start, end = days[0], days[-1]
            open_times = calendar.session_times(start)
            close_times = calendar.session_times(end)
            if not open_times or not close_times:
                raise ValueError(f"round {rid}: no market session on {start}/{end}")
            windows.append(RoundWindow(
                round_id=rid,
                name=name,
                start_date=start,
                end_date=end,
                opens_at=open_times[0],
                closes_at=close_times[1],
                sessions=len(days),
            ))
            nxt = calendar.next_trading_day(end)
            for _ in range(gap_sessions):
                nxt = calendar.next_trading_day(nxt)
            cursor = nxt
        return cls(windows, calendar)

    # -- lookup ------------------------------------------------------------ #

    def window(self, round_id: int) -> RoundWindow:
        for w in self.windows:
            if w.round_id == round_id:
                return w
        raise KeyError(f"round {round_id} is not on the schedule")

    def state(self, now: datetime) -> SeasonState:
        """Resolve the clock to a phase. This is the scheduler's whole brain."""
        running = next((w for w in self.windows if w.contains(now)), None)
        upcoming = next((w for w in self.windows if w.is_future(now)), None)

        if running is None:
            if upcoming is None:
                return SeasonState(now, "done", None, None)
            phase: Phase = "before" if not any(
                w.is_past(now) for w in self.windows) else "between"
            return SeasonState(now, phase, None, upcoming)

        session = self.calendar.session(now)
        if session.is_open:
            inner: Phase = "open"
        elif session.is_trading_day and session.open_at and now < session.open_at:
            inner = "pre_open"
        else:
            inner = "after_close"
        return SeasonState(now, inner, running, upcoming)

    # -- presentation ------------------------------------------------------ #

    def progress(self, round_id: int, now: datetime) -> tuple[int, int]:
        """(sessions elapsed, total) for a round -- for "day 3 of 5"."""
        w = self.window(round_id)
        if now < w.opens_at:
            return 0, w.sessions
        done = self.calendar.count_sessions(w.start_date, min(now.date(), w.end_date))
        return min(done, w.sessions), w.sessions

    def describe(self) -> list[str]:
        out = []
        for w in self.windows:
            out.append(
                f"round {w.round_id}  {w.name:<14} "
                f"{w.start_date:%a %d %b} -> {w.end_date:%a %d %b}  "
                f"{w.sessions} sessions"
            )
        return out


def _take_sessions(cal: MarketCalendar, start: date, n: int) -> list[date]:
    """The next `n` trading days from `start` inclusive."""
    days: list[date] = []
    d = cal.next_trading_day(start, inclusive=True)
    guard = 0
    while len(days) < n:
        days.append(d)
        d = cal.next_trading_day(d)
        guard += 1
        if guard > n * 10 + 60:      # a year of holidays would not do this
            raise ValueError(f"cannot find {n} sessions from {start}")
    return days


def humanise(seconds: float | None) -> str:
    """A countdown a person can read at a glance."""
    if seconds is None:
        return "--"
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}h {mins:02d}m"
    if hours:
        return f"{hours}h {mins:02d}m {secs:02d}s"
    return f"{mins}m {secs:02d}s"


def next_deadline(state: SeasonState) -> tuple[str, datetime | None]:
    """(label, instant) of the thing the clock is counting down to."""
    if state.phase in ("before", "between"):
        return ("starts in", state.next.opens_at if state.next else None)
    if state.current is not None:
        return ("ends in", state.current.closes_at)
    return ("season complete", None)
