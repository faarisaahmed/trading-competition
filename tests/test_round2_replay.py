"""The first Round 2 traded nothing; these pin down every part of the fix.

* Daily bars were requested without a start, which Alpaca reads as "today
  only": one bar per name, so the five-day liquidity check emptied the pool.
* The round is replayed a week later via `schedule.round_starts`, with the
  dead run struck by `comp void-round`.
* The benchmark holds SPY -- the S&P 500 -- in Rounds 2 and 3.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from competition.broker.alpaca import AlpacaDataReader, _lookback_start
from competition.config import ConfigError, TeamConfig, load_config
from competition.data.calendar import MarketCalendar
from competition.engine.ledger import Ledger
from competition.engine.universe import UniverseResolver
from competition.schedule import SeasonSchedule
from competition.types import UTC

# --------------------------------------------------------------------------- #
# bars: never let Alpaca default the start to "today"
# --------------------------------------------------------------------------- #


class _RecordingClient:
    class creds:
        feed = "iex"

    def __init__(self):
        self.params = []

    def data_paginate(self, path, params, key):
        self.params.append(dict(params))
        return iter(())


def test_bars_without_a_start_reach_back_far_enough():
    client = _RecordingClient()
    end = datetime(2026, 10, 2, 20, 0, tzinfo=UTC)
    AlpacaDataReader(client).bars(["AAPL"], "1Day", limit=25, end=end)
    start = datetime.fromisoformat(client.params[0]["start"])
    # 25 sessions is five weeks of calendar; anything under that is the bug.
    assert end - start >= timedelta(days=35)


def test_an_explicit_start_is_left_alone():
    client = _RecordingClient()
    start = datetime(2026, 9, 1, tzinfo=UTC)
    AlpacaDataReader(client).bars(["AAPL"], "1Day", limit=25, start=start)
    assert client.params[0]["start"] == start.isoformat()


@pytest.mark.parametrize("tf,limit,min_days", [
    ("1Day", 400, 560),     # the pickers' history: 400 sessions ~ 80 weeks
    ("5Min", 500, 6),       # 500 bars ~ 6.4 sessions
    ("1Min", 240, 1),
])
def test_lookback_covers_the_requested_bars(tf, limit, min_days):
    end = datetime(2026, 10, 5, 14, 0, tzinfo=UTC)
    assert (end - _lookback_start(tf, limit, end)).days >= min_days


# --------------------------------------------------------------------------- #
# the benchmark holds the S&P 500
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def shipped():
    return load_config()


def _team(cfg, key):
    return next(t for t in cfg.teams if t.key == key)


@pytest.mark.parametrize("round_id", [2, 3])
def test_benchmark_holds_spy_in_rounds_2_and_3(shipped, round_id):
    rnd = next(r for r in shipped.rounds if r.id == round_id)
    resolver = UniverseResolver(shipped)       # no provider: it must not need one
    u = resolver.resolve(rnd, _team(shipped, "benchmark"), session=date(2026, 10, 5))
    assert u.symbols == ("SPY",)
    assert u.source == "fixed(team)"


def test_benchmark_still_trades_the_big_three_in_round_1(shipped):
    rnd = next(r for r in shipped.rounds if r.id == 1)
    u = UniverseResolver(shipped).resolve(
        rnd, _team(shipped, "benchmark"), session=date(2026, 9, 21))
    assert u.symbols == tuple(rnd.symbols)


def test_only_an_unscored_team_may_pin_its_universe():
    team = TeamConfig(key="cheater", name="x", strategy="a:B", picker="a:B",
                      env_prefix="X", scored=True, fixed_universe={2: ("SPY",)})
    with pytest.raises(ConfigError, match="unscored"):
        team.validate()


# --------------------------------------------------------------------------- #
# postponing a round
# --------------------------------------------------------------------------- #

ROUNDS = [(1, "The Big Three"), (2, "Open Market"), (3, "Chaos Draft")]


def test_a_postponed_round_pushes_the_rest_back():
    s = SeasonSchedule.build(
        rounds=ROUNDS, first_start=date(2026, 9, 21), sessions_per_round=5,
        calendar=MarketCalendar(), round_starts={2: date(2026, 10, 5)},
    )
    assert [(w.start_date, w.end_date) for w in s.windows] == [
        (date(2026, 9, 21), date(2026, 9, 25)),
        (date(2026, 10, 5), date(2026, 10, 9)),
        (date(2026, 10, 12), date(2026, 10, 16)),
    ]
    # The week in between is a gap, not a round.
    assert s.state(datetime(2026, 9, 30, 15, 0, tzinfo=UTC)).phase == "between"


def test_a_round_cannot_be_pulled_into_the_one_before():
    with pytest.raises(ValueError, match="cannot start"):
        SeasonSchedule.build(
            rounds=ROUNDS, first_start=date(2026, 9, 21), sessions_per_round=5,
            calendar=MarketCalendar(), round_starts={2: date(2026, 9, 23)},
        )


def test_the_shipped_schedule_replays_round_2_from_5_october(shipped):
    from competition import cli

    s = cli._build_schedule(shipped, MarketCalendar())
    assert s.window(2).start_date == date(2026, 10, 5)
    assert s.window(3).start_date == date(2026, 10, 12)


# --------------------------------------------------------------------------- #
# voiding the dead run
# --------------------------------------------------------------------------- #


def _played(ledger: Ledger, round_id: int, start: date, *, scored: bool) -> None:
    ledger.open_round(round_id=round_id, start=start, end=start + timedelta(days=4),
                      mode="live", broker="alpaca", rules_hash="h")
    if scored:
        for i, team in enumerate(("a", "b")):
            ledger.record_result(round_id, team, start_equity=5000, end_equity=5000,
                                 return_pct=0.0, place=i + 1, points=13.0)
        ledger.close_round(round_id)


@pytest.mark.parametrize("scored", [True, False])
def test_void_strikes_the_round_and_keeps_the_audit_trail(tmp_path, scored):
    ledger = Ledger(tmp_path / "l.sqlite", run_id="r1")
    _played(ledger, 1, date(2026, 9, 21), scored=True)
    ledger.run_id = "r2"
    _played(ledger, 2, date(2026, 9, 28), scored=scored)

    runs, moved = ledger.void_round(2, reason="empty pool")

    assert runs == 1
    assert moved == (2 if scored else 0)
    assert ledger.results_for_round(2) == []
    assert len(ledger.results_for_round(1)) == 2           # untouched
    assert ledger.resumable_round(2) is None               # nothing to rejoin
    assert ledger.round_progress(2)["status"] == "void"
    kept = ledger.query("SELECT reason FROM voided_results WHERE round_id = 2")
    assert [r["reason"] for r in kept] == ["empty pool"] * moved


def test_the_replay_scores_cleanly_after_a_void(tmp_path):
    ledger = Ledger(tmp_path / "l.sqlite", run_id="old")
    _played(ledger, 2, date(2026, 9, 28), scored=True)
    ledger.void_round(2, reason="empty pool")
    ledger.run_id = "new"
    _played(ledger, 2, date(2026, 10, 5), scored=True)
    rows = ledger.results_for_round(2)
    assert {r["run_id"] for r in rows} == {"new"}
    assert len(rows) == 2


def test_void_round_is_a_dry_run_without_yes(tmp_path):
    from competition.cli import main

    path = tmp_path / "l.sqlite"
    ledger = Ledger(path, run_id="r2")
    _played(ledger, 2, date(2026, 9, 28), scored=True)
    ledger.close()

    assert main(["--quiet", "--ledger", str(path), "void-round", "2",
                 "--reason", "x"]) == 0
    assert len(Ledger(path).results_for_round(2)) == 2
    assert main(["--quiet", "--ledger", str(path), "void-round", "2",
                 "--reason", "x", "--yes"]) == 0
    assert Ledger(path).results_for_round(2) == []
