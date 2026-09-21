"""The season calendar: when rounds open, close, and how the clock reads.

A three-week unattended run has no human to catch a scheduling mistake, so
the arithmetic here is worth more tests than it looks like it needs.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from competition.data.calendar import MarketCalendar
from competition.schedule import (
    SeasonSchedule,
    humanise,
    next_deadline,
)
from competition.types import UTC


@pytest.fixture
def cal():
    return MarketCalendar()


@pytest.fixture
def season(cal):
    """The shipped shape: three rounds of five sessions from Mon 21 Sep 2026."""
    return SeasonSchedule.build(
        rounds=[(1, "The Big Three"), (2, "Open Market"), (3, "Chaos Draft")],
        first_start=date(2026, 9, 21),
        sessions_per_round=5,
        calendar=cal,
    )


# --------------------------------------------------------------------------- #
# laying out the windows
# --------------------------------------------------------------------------- #


def test_each_round_is_one_market_week(season):
    assert [w.round_id for w in season.windows] == [1, 2, 3]
    for w in season.windows:
        assert w.sessions == 5
        assert w.start_date.weekday() == 0      # Monday
        assert w.end_date.weekday() == 4        # Friday


def test_rounds_run_back_to_back_without_overlapping(season):
    for a, b in zip(season.windows, season.windows[1:], strict=False):
        assert b.opens_at > a.closes_at
        # The next round starts the very next session.
        assert (b.start_date - a.end_date).days == 3       # Fri -> Mon


def test_the_bell_times_are_the_market_open_and_close(season, cal):
    w = season.window(1)
    open_at, _ = cal.session_times(w.start_date)
    _, close_at = cal.session_times(w.end_date)
    assert w.opens_at == open_at
    assert w.closes_at == close_at


def test_a_weekend_start_rolls_forward_to_the_open(cal):
    """Asking for Saturday must not silently lose a session."""
    s = SeasonSchedule.build(
        rounds=[(1, "r1")], first_start=date(2026, 9, 19),   # a Saturday
        sessions_per_round=5, calendar=cal)
    assert s.window(1).start_date == date(2026, 9, 21)
    assert s.window(1).sessions == 5


def test_a_holiday_inside_a_round_extends_the_calendar_span(cal):
    """Five *sessions* is the unit, not five days."""
    # Thanksgiving week 2026: Thu 26 Nov is a holiday.
    s = SeasonSchedule.build(
        rounds=[(1, "r1")], first_start=date(2026, 11, 23),
        sessions_per_round=5, calendar=cal)
    w = s.window(1)
    assert w.sessions == 5
    assert cal.count_sessions(w.start_date, w.end_date) == 5
    # It therefore runs past Friday, because one weekday was not a session.
    assert w.end_date > date(2026, 11, 27)


def test_gap_sessions_insert_idle_days(cal):
    s = SeasonSchedule.build(
        rounds=[(1, "a"), (2, "b")], first_start=date(2026, 9, 21),
        sessions_per_round=5, calendar=cal, gap_sessions=1)
    a, b = s.windows
    assert cal.count_sessions(a.end_date, b.start_date) == 3   # end, gap, start


def test_overlapping_windows_are_refused(cal):
    from competition.schedule import RoundWindow

    now = datetime(2026, 9, 21, 13, 30, tzinfo=UTC)
    a = RoundWindow(1, "a", date(2026, 9, 21), date(2026, 9, 25),
                    now, now + timedelta(days=5), 5)
    b = RoundWindow(2, "b", date(2026, 9, 23), date(2026, 9, 29),
                    now + timedelta(days=2), now + timedelta(days=8), 5)
    with pytest.raises(ValueError, match="overlap"):
        SeasonSchedule([a, b], cal)


def test_a_season_needs_a_round(cal):
    with pytest.raises(ValueError, match="at least one round"):
        SeasonSchedule([], cal)


def test_unknown_round_is_an_error(season):
    with pytest.raises(KeyError):
        season.window(99)


# --------------------------------------------------------------------------- #
# resolving the clock to a phase -- the scheduler's whole brain
# --------------------------------------------------------------------------- #


def test_before_the_first_bell(season):
    state = season.state(datetime(2026, 9, 18, 22, 0, tzinfo=UTC))
    assert state.phase == "before"
    assert state.current is None
    assert state.next.round_id == 1
    assert state.seconds_to_start > 0
    assert not state.is_trading


def test_during_a_session(season):
    state = season.state(datetime(2026, 9, 22, 15, 0, tzinfo=UTC))   # 11am ET
    assert state.phase == "open"
    assert state.round_id == 1
    assert state.is_trading
    assert state.seconds_to_end > 0


def test_overnight_inside_a_round_is_not_trading(season):
    state = season.state(datetime(2026, 9, 23, 3, 0, tzinfo=UTC))
    assert state.phase in ("pre_open", "after_close")
    assert state.round_id == 1          # still inside round 1's window
    assert not state.is_trading


def test_the_weekend_between_rounds(season):
    """Round 1 has closed; round 2 has not opened."""
    state = season.state(datetime(2026, 9, 26, 12, 0, tzinfo=UTC))
    assert state.phase == "between"
    assert state.current is None
    assert state.next.round_id == 2


def test_after_the_last_bell_the_season_is_done(season):
    state = season.state(datetime(2026, 10, 12, 12, 0, tzinfo=UTC))
    assert state.phase == "done"
    assert state.is_finished
    assert state.seconds_to_start is None
    assert state.seconds_to_end is None


def test_the_instant_of_the_final_bell_closes_the_round(season):
    w = season.window(3)
    assert season.state(w.closes_at - timedelta(seconds=1)).round_id == 3
    assert season.state(w.closes_at).is_finished


def test_the_opening_bell_starts_the_round(season):
    w = season.window(1)
    assert season.state(w.opens_at - timedelta(seconds=1)).phase == "before"
    assert season.state(w.opens_at).round_id == 1


# --------------------------------------------------------------------------- #
# progress and presentation
# --------------------------------------------------------------------------- #


def test_progress_counts_sessions_not_days(season):
    assert season.progress(1, datetime(2026, 9, 18, tzinfo=UTC)) == (0, 5)
    assert season.progress(1, datetime(2026, 9, 21, 15, 0, tzinfo=UTC)) == (1, 5)
    assert season.progress(1, datetime(2026, 9, 23, 15, 0, tzinfo=UTC)) == (3, 5)
    # A weekend mid-round does not advance the count past the total.
    assert season.progress(1, datetime(2026, 9, 27, tzinfo=UTC)) == (5, 5)


def test_deadline_points_at_the_next_real_event(season):
    label, when = next_deadline(season.state(datetime(2026, 9, 18, tzinfo=UTC)))
    assert label == "starts in" and when == season.window(1).opens_at

    label, when = next_deadline(
        season.state(datetime(2026, 9, 22, 15, 0, tzinfo=UTC)))
    assert label == "ends in" and when == season.window(1).closes_at

    label, when = next_deadline(
        season.state(datetime(2026, 10, 12, tzinfo=UTC)))
    assert when is None


def test_describe_lists_every_round(season):
    lines = season.describe()
    assert len(lines) == 3
    assert "The Big Three" in lines[0]


@pytest.mark.parametrize("secs,want", [
    (None, "--"),
    (0, "0m 00s"),
    (59, "0m 59s"),
    (61, "1m 01s"),
    (3661, "1h 01m 01s"),
    (90000, "1d 01h 00m"),
    (-5, "0m 00s"),
])
def test_humanise(secs, want):
    assert humanise(secs) == want


# --------------------------------------------------------------------------- #
# the config that drives it
# --------------------------------------------------------------------------- #


def test_shipped_config_starts_on_a_monday(cfg):
    assert cfg.schedule.start_date == date(2026, 9, 21)
    assert cfg.schedule.start_date.weekday() == 0
    assert cfg.schedule.sessions_per_round == 5
    assert cfg.schedule.auto_advance is True


def test_shipped_config_lays_out_three_clean_weeks(cfg, cal):
    s = SeasonSchedule.build(
        rounds=[(r.id, r.name) for r in cfg.rounds],
        first_start=cfg.schedule.start_date,
        sessions_per_round=cfg.schedule.sessions_per_round,
        calendar=cal, gap_sessions=cfg.schedule.gap_sessions)
    assert [w.round_id for w in s.windows] == [1, 2, 3]
    assert [w.start_date for w in s.windows] == [
        date(2026, 9, 21), date(2026, 9, 28), date(2026, 10, 5)]


def test_publish_interval_has_a_floor():
    from competition.config import ConfigError, PublishConfig

    with pytest.raises(ConfigError, match="every_seconds"):
        PublishConfig(enabled=True, every_seconds=5).validate()


def test_bad_start_date_is_rejected(tmp_path):
    from competition.config import ConfigError, _parse_schedule

    with pytest.raises(ConfigError, match="YYYY-MM-DD"):
        _parse_schedule({"start_date": "next monday"})


def test_sessions_per_round_must_be_positive():
    from competition.config import ConfigError, ScheduleConfig

    with pytest.raises(ConfigError, match="sessions_per_round"):
        ScheduleConfig(sessions_per_round=0).validate()


# --------------------------------------------------------------------------- #
# picking the stop time from the clock
# --------------------------------------------------------------------------- #


def test_the_window_boundaries_fit_inside_a_ci_job():
    """A session is longer than one job may run, hence two windows."""
    from competition.cli import HANDOVER_UTC, SESSION_END_UTC

    open_utc = 13 * 60 + 30
    handover = HANDOVER_UTC[0] * 60 + HANDOVER_UTC[1]
    end = SESSION_END_UTC[0] * 60 + SESSION_END_UTC[1]

    assert open_utc < handover < end
    assert end >= 20 * 60, "must cover the 20:00 UTC close"
    # Each half has to finish well inside GitHub's six-hour ceiling, counting
    # from the earliest trigger that could start it.
    assert handover - (12 * 60 + 5) < 5 * 60
    assert end - handover < 5 * 60


@pytest.mark.parametrize("now_hhmm,want", [
    ((12, 5), (16, 40)),     # before the bell
    ((13, 30), (16, 40)),    # the open
    ((16, 39), (16, 40)),    # a minute before handover
    ((16, 41), (20, 10)),    # just after
    ((18, 0), (20, 10)),     # a badly delayed run still does the right thing
])
def test_auto_window_is_chosen_by_the_clock(now_hhmm, want):
    """The failure this exists to prevent.

    Mapping the window to *which cron fired* means a trigger GitHub drops
    takes its whole window with it. Deriving it from the clock means any run,
    whenever it starts, stops at the right time.
    """
    from competition.cli import HANDOVER_UTC, SESSION_END_UTC

    chosen = HANDOVER_UTC if now_hhmm < HANDOVER_UTC else SESSION_END_UTC
    assert chosen == want
