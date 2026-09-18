"""NYSE calendar. Computed from published rules, so it must match reality."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from competition.data.calendar import (
    NY,
    MarketCalendar,
    _easter,
    nyse_early_closes,
    nyse_holidays,
)
from competition.types import UTC

# Published Easter Sundays -- Good Friday is two days earlier.
KNOWN_EASTER = {
    2020: (4, 12), 2021: (4, 4), 2022: (4, 17), 2023: (4, 9), 2024: (3, 31),
    2025: (4, 20), 2026: (4, 5), 2027: (3, 28), 2028: (4, 16), 2029: (4, 1),
    2030: (4, 21),
}


@pytest.mark.parametrize("year,expected", sorted(KNOWN_EASTER.items()))
def test_easter_matches_published_dates(year, expected):
    e = _easter(year)
    assert (e.month, e.day) == expected


def test_2026_holidays_are_the_published_nine_plus_juneteenth():
    got = sorted(nyse_holidays(2026))
    assert got == [
        date(2026, 1, 1),    # New Year's Day
        date(2026, 1, 19),   # MLK Jr Day
        date(2026, 2, 16),   # Washington's Birthday
        date(2026, 4, 3),    # Good Friday
        date(2026, 5, 25),   # Memorial Day
        date(2026, 6, 19),   # Juneteenth
        date(2026, 7, 3),    # Independence Day observed (Jul 4 is a Saturday)
        date(2026, 9, 7),    # Labor Day
        date(2026, 11, 26),  # Thanksgiving
        date(2026, 12, 25),  # Christmas
    ]


def test_weekend_holidays_shift_to_a_weekday():
    # 2027-12-25 is a Saturday -> observed Friday the 24th.
    assert date(2027, 12, 24) in nyse_holidays(2027)
    # 2022-01-01 was a Saturday -> observed Friday 2021-12-31, so Jan 1 2022
    # itself is not in the 2022 set as a weekday closure.
    assert date(2022, 1, 1) not in nyse_holidays(2022)


def test_juneteenth_only_from_2022():
    assert date(2021, 6, 18) not in nyse_holidays(2021)
    assert any(d.month == 6 and d.day in (18, 19, 20) for d in nyse_holidays(2023))


def test_early_closes():
    early = nyse_early_closes(2026)
    assert date(2026, 11, 27) in early          # day after Thanksgiving
    assert date(2026, 12, 24) in early          # Christmas Eve
    assert all(d.weekday() < 5 for d in early)


def test_no_trading_on_holidays_or_weekends(calendar):
    assert not calendar.is_trading_day(date(2026, 9, 7))    # Labor Day
    assert not calendar.is_trading_day(date(2026, 9, 12))   # Saturday
    assert not calendar.is_trading_day(date(2026, 9, 13))   # Sunday
    assert calendar.is_trading_day(date(2026, 9, 8))        # Tuesday


def test_seven_calendar_days_gives_four_or_five_sessions(calendar):
    # The competition round is 7 calendar days. Over a normal week that is 5
    # sessions; a week containing a holiday gives 4.
    assert calendar.count_sessions(date(2026, 9, 14), date(2026, 9, 20)) == 5
    assert calendar.count_sessions(date(2026, 9, 7), date(2026, 9, 13)) == 4


def test_session_times_are_930_to_1600_eastern(calendar):
    o, c = calendar.session_times(date(2026, 9, 8))
    assert o.astimezone(NY).strftime("%H:%M") == "09:30"
    assert c.astimezone(NY).strftime("%H:%M") == "16:00"


def test_early_close_session_ends_at_1300(calendar):
    o, c = calendar.session_times(date(2026, 11, 27))
    assert c.astimezone(NY).strftime("%H:%M") == "13:00"
    assert calendar.is_early_close(date(2026, 11, 27))


def test_session_state_mid_day(calendar):
    ts = datetime(2026, 9, 10, 17, 0, tzinfo=UTC)      # 13:00 ET
    s = calendar.session(ts)
    assert s.is_open and s.is_trading_day
    assert s.minutes_since_open == pytest.approx(210, abs=1)
    assert s.minutes_to_close == pytest.approx(180, abs=1)
    assert not s.in_opening_range and not s.in_closing_range


def test_session_state_in_opening_and_closing_ranges(calendar):
    o, c = calendar.session_times(date(2026, 9, 10))
    assert calendar.session(o + timedelta(minutes=5)).in_opening_range
    assert calendar.session(c - timedelta(minutes=5)).in_closing_range
    assert calendar.session(c - timedelta(minutes=5)).within_minutes_of_close(10)


def test_session_state_when_closed(calendar):
    s = calendar.session(datetime(2026, 9, 12, 17, 0, tzinfo=UTC))   # Saturday
    assert not s.is_trading_day and not s.is_open
    assert s.next_open is not None
    assert s.next_open.astimezone(NY).date() == date(2026, 9, 14)
    assert s.minutes_to_close == float("inf")


def test_next_and_previous_trading_day(calendar):
    assert calendar.next_trading_day(date(2026, 9, 4)) == date(2026, 9, 8)   # skips Labor Day
    assert calendar.previous_trading_day(date(2026, 9, 8)) == date(2026, 9, 4)
    assert calendar.next_trading_day(date(2026, 9, 8), inclusive=True) == date(2026, 9, 8)


def test_tick_times_stay_inside_sessions(calendar):
    ticks = calendar.tick_times(date(2026, 9, 8), date(2026, 9, 9), seconds=300)
    assert len(ticks) == 2 * 78 + 2        # 78 five-minute bars + a close tick each
    for t in ticks:
        s = calendar.session(t)
        assert s.is_trading_day
        assert s.open_at <= t <= s.close_at


def test_alpaca_calendar_falls_back_when_offline():
    class Broken:
        def calendar(self, *_a, **_k):
            raise RuntimeError("network down")

    cal = MarketCalendar.from_alpaca(Broken())
    # Falls back to computed rules rather than exploding.
    assert cal.is_trading_day(date(2026, 9, 8))
    assert not cal.is_trading_day(date(2026, 9, 7))


def test_explicit_sessions_override_computed_rules():
    cal = MarketCalendar(sessions={date(2026, 9, 7): (
        __import__("datetime").time(9, 30), __import__("datetime").time(16, 0))})
    # An override makes Labor Day a session, which is what /v2/calendar would do
    # if the exchange ever published one.
    assert cal.is_trading_day(date(2026, 9, 7))
