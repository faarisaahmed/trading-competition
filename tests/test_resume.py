"""Crash resume.

The scenario: a round is four days in, the process dies, you restart it. The
requirement is that the round *continues* -- same baseline, same positions,
same strategy state -- rather than being silently restarted from zero, which
is what the un-checkpointed engine did (it flattens every account on
`prepare_round`).

These tests model the real failure by keeping the *broker* objects alive (a
live Alpaca account still holds its positions after your laptop sleeps) while
building an entirely new engine over the same ledger.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from competition.broker.simulated import SimConfig, SimulatedBroker
from competition.data.universe import UniverseProvider
from competition.engine import CompetitionEngine, Ledger, UniverseResolver
from competition.engine.checkpoint import (
    TeamCheckpoint,
    load_round_checkpoint,
    safe_state,
)
from competition.types import UTC

ROUND_START = date(2026, 9, 8)
ROUND_END = date(2026, 9, 11)


def make_engine(cfg, feed, ledger_path, brokers, clock, *, teams=None,
                daily=None, news=None, run_id=None):
    ledger = Ledger(ledger_path, run_id=run_id)
    ledger.start_run(round_id=1, mode="test", broker="sim",
                     rules_hash=cfg.rules_hash, seed=1, config={})
    resolver = UniverseResolver(
        cfg, provider=UniverseProvider(),
        daily_bars=lambda syms: {s: tuple((daily or {}).get(s, ())) for s in syms},
        news=lambda syms: {s: tuple((news or {}).get(s, ())) for s in syms},
    )
    engine = CompetitionEngine(
        cfg, feed=feed, brokers=brokers, ledger=ledger, resolver=resolver,
        mode="test", teams=teams or list(cfg.teams), clock=clock,
        equity_snapshot_seconds=1,
    )
    inner = engine.tick

    def tick(rnd, *, now=None, **kw):
        if now is not None:
            clock.now = now
        return inner(rnd, now=now, **kw)

    engine.tick = tick                                  # type: ignore[method-assign]
    return engine, ledger


class Clock:
    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def scenario(cfg, feed, tmp_path, daily, news, calendar):
    """A round ticked partway through, with a checkpoint on disk."""
    path = tmp_path / "resume.sqlite"
    clock = Clock(datetime.combine(ROUND_START, datetime.min.time(), tzinfo=UTC))
    sim = SimConfig(slippage_bps=2.0, fractional=True, allow_short=False)
    brokers = {
        t.key: SimulatedBroker(cfg.starting_cash, feed.quote_source(clock),
                               name=t.key, config=sim, clock=clock)
        for t in cfg.teams
    }
    engine, ledger = make_engine(cfg, feed, path, brokers, clock,
                                 daily=daily, news=news, run_id="run-one")
    rnd = cfg.round(1)
    engine.round_window = (ROUND_START, ROUND_END)
    ledger.open_round(round_id=1, start=ROUND_START, end=ROUND_END, mode="test",
                      broker="sim", rules_hash=cfg.rules_hash)
    engine.prepare_round(rnd, session=ROUND_START)
    engine.start_round(rnd)

    # Tick through most of a session so positions and state exist.
    stamps = calendar.tick_times(ROUND_START, ROUND_START, seconds=300)
    for ts in stamps[: int(len(stamps) * 0.8)]:
        engine.tick(rnd, now=ts, progress=0.2)
    engine.checkpoint(rnd, ticks=len(stamps))

    return {
        "path": path, "brokers": brokers, "clock": clock, "engine": engine,
        "ledger": ledger, "rnd": rnd, "stamps": stamps, "cfg": cfg, "feed": feed,
        "daily": daily, "news": news,
    }


# --------------------------------------------------------------------------- #
# the checkpoint captured something worth restoring
# --------------------------------------------------------------------------- #


def test_the_interrupted_round_actually_did_something(scenario):
    engine = scenario["engine"]
    holders = [k for k, b in scenario["brokers"].items() if b.positions()]
    assert holders, "no team held a position, so the test proves nothing"
    assert any(engine.teams[k].state for k in engine.teams), "no strategy state"


def test_checkpoint_is_written_for_every_team(scenario):
    payload = scenario["ledger"].resumable_round(1)
    assert payload is not None
    assert set(payload["teams"]) == set(scenario["cfg"].team_keys)
    assert payload["status"] == "in_progress"
    assert payload["ticks"] > 0


def test_checkpoint_round_trips_through_json(scenario):
    engine = scenario["engine"]
    for rt in engine.teams.values():
        blob = TeamCheckpoint.from_runtime(rt).to_dict()
        back = TeamCheckpoint.from_dict(blob)
        assert back.team_key == rt.key
        assert back.baseline_equity == pytest.approx(rt.baseline_equity)
        assert back.tick_count == rt.tick_count
        assert set(back.seen_fills) == set(rt.seen_fills)


def test_unserialisable_state_is_dropped_not_fatal():
    class Weird:
        pass

    out = safe_state({"good": 1.5, "nested": {"a": [1, 2]}, "bad": Weird()},
                     who="test")
    assert out["good"] == 1.5 and out["nested"] == {"a": [1, 2]}
    assert "bad" not in out


# --------------------------------------------------------------------------- #
# resuming
# --------------------------------------------------------------------------- #


@pytest.fixture
def resumed(scenario):
    """A brand new engine that rejoined the interrupted round."""
    before = {}
    for key, rt in scenario["engine"].teams.items():
        account = scenario["brokers"][key].account()
        before[key] = {
            "baseline": rt.baseline_equity,
            "state": dict(rt.state),
            "tick_count": rt.tick_count,
            "seen_fills": set(rt.seen_fills),
            "positions": {p.symbol: p.qty for p in account.positions},
            "equity": account.equity,
            "universe": rt.universe,
            "session_open_equity": rt.risk.session_open_equity,
            "orders_today": rt.risk.orders_today,
        }
    scenario["ledger"].close()

    # A different process: new engine, new ledger handle, same brokers.
    engine2, ledger2 = make_engine(
        scenario["cfg"], scenario["feed"], scenario["path"], scenario["brokers"],
        scenario["clock"], daily=scenario["daily"], news=scenario["news"],
        run_id="run-two",
    )
    checkpoint = load_round_checkpoint(ledger2.resumable_round(1))
    assert checkpoint is not None and checkpoint.is_resumable
    engine2.prepare_round(scenario["rnd"], session=ROUND_START, resume=checkpoint)
    engine2.start_round(scenario["rnd"])
    # `**scenario` FIRST, then the overrides -- the other way round silently
    # hands the tests the pre-resume engine.
    return {**scenario, "engine": engine2, "ledger": ledger2, "before": before}


def test_resume_does_not_flatten_the_accounts(resumed):
    """The whole point: positions survive."""
    for key, snapshot in resumed["before"].items():
        now = {p.symbol: p.qty
               for p in resumed["brokers"][key].account().positions}
        assert now == pytest.approx(snapshot["positions"]), (
            f"{key} lost its positions on resume")


def test_resume_restores_the_baseline_not_the_current_equity(resumed):
    """If the baseline were re-read live, a team's loss would be forgiven."""
    for key, snapshot in resumed["before"].items():
        rt = resumed["engine"].teams[key]
        assert rt.baseline_equity == pytest.approx(snapshot["baseline"])
    # And at least one team must actually be away from its baseline, or this
    # test could pass trivially.
    moved = [k for k, v in resumed["before"].items()
             if abs(v["equity"] - v["baseline"]) > 0.01]
    assert moved, "no team had moved off its baseline"


def test_resume_restores_strategy_state(resumed):
    for key, snapshot in resumed["before"].items():
        rt = resumed["engine"].teams[key]
        if not snapshot["state"]:
            continue
        # Compared through the serialiser, since that is what was persisted.
        assert rt.state == safe_state(snapshot["state"], who=key)


def test_resume_restores_risk_state(resumed):
    for key, snapshot in resumed["before"].items():
        rt = resumed["engine"].teams[key]
        assert rt.risk.session_open_equity == pytest.approx(
            snapshot["session_open_equity"])
        assert rt.risk.orders_today == snapshot["orders_today"]


def test_resume_restores_tick_count_and_universe(resumed):
    for key, snapshot in resumed["before"].items():
        rt = resumed["engine"].teams[key]
        assert rt.tick_count == snapshot["tick_count"]
        assert rt.universe == snapshot["universe"]


def test_resume_does_not_replay_old_fills(resumed):
    """Restored `seen_fills` means `on_fill` is not called again for history."""
    for key, snapshot in resumed["before"].items():
        rt = resumed["engine"].teams[key]
        assert rt.seen_fills == snapshot["seen_fills"]
    engine = resumed["engine"]
    rnd = resumed["rnd"]
    engine.tick(rnd, now=resumed["stamps"][-1], progress=0.9)
    for key in resumed["before"]:
        replayed = [f for f in engine.teams[key].fills
                    if f"{f.order_id}:{f.ts.isoformat()}:{f.qty:.6f}"
                    in resumed["before"][key]["seen_fills"]]
        assert not replayed, f"{key} replayed an already-seen fill"


def test_resume_skips_on_round_start(resumed):
    assert resumed["engine"].resuming is True


def test_a_resumed_round_can_be_finished(resumed):
    engine, rnd = resumed["engine"], resumed["rnd"]
    for ts in resumed["stamps"][-3:]:
        engine.tick(rnd, now=ts, progress=0.95)
    result = engine.finish_round(rnd, started_at=resumed["stamps"][0],
                                 ticks=engine._ticks)
    assert len(result.teams) == len(resumed["cfg"].teams)
    for team in result.teams:
        # Returns must be measured from the ORIGINAL baseline.
        assert team.start_equity == pytest.approx(
            resumed["before"][team.team_key]["baseline"])
    # And the round is no longer resumable.
    assert resumed["ledger"].resumable_round(1) is None


# --------------------------------------------------------------------------- #
# a fresh start is still a fresh start
# --------------------------------------------------------------------------- #


def test_without_resume_the_round_restarts_from_zero(scenario):
    """The old behaviour must still be reachable, and must be explicit."""
    positions_before = {
        k: {p.symbol: p.qty for p in b.account().positions}
        for k, b in scenario["brokers"].items()
    }
    assert any(positions_before.values())
    scenario["ledger"].close()

    engine2, _ledger2 = make_engine(
        scenario["cfg"], scenario["feed"], scenario["path"], scenario["brokers"],
        scenario["clock"], daily=scenario["daily"], news=scenario["news"],
        run_id="run-three",
    )
    engine2.prepare_round(scenario["rnd"], session=ROUND_START)  # no resume
    assert engine2.resuming is False
    for key, broker in scenario["brokers"].items():
        assert broker.account().positions == (), f"{key} was not flattened"
        assert broker.account().cash == pytest.approx(scenario["cfg"].starting_cash)


# --------------------------------------------------------------------------- #
# refusing an unsafe resume
# --------------------------------------------------------------------------- #


def _checkpoint(**over):
    base = dict(round_id=1, run_id="r", start_date="2026-09-08",
                end_date="2026-09-11", mode="live", broker="alpaca",
                rules_hash="hash", status="in_progress", ticks=10,
                updated_at="2026-09-10T14:00:00+00:00",
                teams={"a": {"team_key": "a", "baseline_equity": 5000.0}})
    base.update(over)
    return load_round_checkpoint(base)


def test_refuses_a_resume_when_the_rulebook_changed():
    cp = _checkpoint()
    ok, why = cp.compatible_with(round_id=1, start=date(2026, 9, 8),
                                 end=date(2026, 9, 11), rules_hash="different",
                                 team_keys={"a"})
    assert not ok and "rulebook changed" in why


def test_refuses_a_resume_into_a_different_window():
    cp = _checkpoint()
    ok, why = cp.compatible_with(round_id=1, start=date(2026, 9, 15),
                                 end=date(2026, 9, 11), rules_hash="hash",
                                 team_keys={"a"})
    assert not ok and "started" in why


def test_refuses_a_resume_into_a_different_round():
    cp = _checkpoint()
    ok, why = cp.compatible_with(round_id=2, start=date(2026, 9, 8),
                                 end=date(2026, 9, 11), rules_hash="hash",
                                 team_keys={"a"})
    assert not ok and "round" in why


def test_refuses_a_resume_with_a_missing_team():
    cp = _checkpoint()
    ok, why = cp.compatible_with(round_id=1, start=date(2026, 9, 8),
                                 end=date(2026, 9, 11), rules_hash="hash",
                                 team_keys={"a", "b"})
    assert not ok and "no checkpoint for team" in why


def test_refuses_a_resume_with_an_unexpected_team():
    cp = _checkpoint(teams={"a": {"team_key": "a", "baseline_equity": 1.0},
                            "z": {"team_key": "z", "baseline_equity": 1.0}})
    ok, why = cp.compatible_with(round_id=1, start=date(2026, 9, 8),
                                 end=date(2026, 9, 11), rules_hash="hash",
                                 team_keys={"a"})
    assert not ok and "not in this run" in why


def test_a_completed_round_is_not_resumable(cfg, tmp_path):
    ledger = Ledger(tmp_path / "done.sqlite")
    ledger.start_run(round_id=1, mode="test", broker="sim",
                     rules_hash=cfg.rules_hash, seed=1, config={})
    ledger.open_round(round_id=1, start=ROUND_START, end=ROUND_END, mode="test",
                      broker="sim", rules_hash=cfg.rules_hash)
    ledger.save_checkpoint(1, "trend_rider", {"baseline_equity": 5000.0})
    assert ledger.resumable_round(1) is not None
    ledger.close_round(1, status="complete")
    assert ledger.resumable_round(1) is None
    ledger.close()


def test_corrupt_checkpoint_is_ignored_not_fatal(cfg, tmp_path):
    ledger = Ledger(tmp_path / "corrupt.sqlite")
    ledger.start_run(round_id=1, mode="test", broker="sim",
                     rules_hash=cfg.rules_hash, seed=1, config={})
    ledger.open_round(round_id=1, start=ROUND_START, end=ROUND_END, mode="test",
                      broker="sim", rules_hash=cfg.rules_hash)
    with ledger._tx() as cur:
        cur.execute(
            "INSERT INTO round_state (round_id, run_id, team_key, updated_at, "
            "blob_json) VALUES (1, ?, 'trend_rider', '2026-09-10', '{not json')",
            (ledger.run_id,),
        )
    payload = ledger.resumable_round(1)
    assert payload is not None and payload["teams"] == {}
    assert load_round_checkpoint(payload).is_resumable is False
    ledger.close()


def test_no_checkpoint_means_no_resume(cfg, tmp_path):
    ledger = Ledger(tmp_path / "empty.sqlite")
    assert ledger.resumable_round(1) is None
    assert load_round_checkpoint(None) is None
    ledger.close()


# --------------------------------------------------------------------------- #
# suspending a round that runs out of wall-clock time
# --------------------------------------------------------------------------- #


def test_suspend_is_not_the_same_as_stop(cfg):
    """The distinction the whole split-job scheme rests on.

    `stop` ends a round: flatten, score, record. `suspend` means only that
    this process is out of time -- the round is unfinished and must be handed
    on intact. Confusing them would liquidate every team at lunchtime and
    score a half-round.
    """
    import inspect

    from competition.engine.runner import CompetitionEngine, RoundSuspended

    sig = inspect.signature(CompetitionEngine.run_live)
    assert "suspend" in sig.parameters and "stop" in sig.parameters

    src = inspect.getsource(CompetitionEngine.run_live)
    stop_at = src.index("stop is not None and stop()")
    suspend_at = src.index("suspend is not None and suspend()")
    # The stop branch breaks out into finish_round; the suspend branch must
    # raise instead, so it can never reach scoring.
    suspend_block = src[suspend_at:suspend_at + 500]
    assert "RoundSuspended" in suspend_block
    assert "checkpoint" in suspend_block
    assert "finish_round" not in suspend_block
    assert stop_at != suspend_at

    e = RoundSuspended(2, 41)
    assert e.round_id == 2 and e.ticks == 41
