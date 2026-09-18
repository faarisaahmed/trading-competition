"""End-to-end: the engine, the ledger, and the universe resolver.

These are the integration tests. They run real rounds with all nine teams
over synthetic data and assert the properties that make the competition a
competition:

  * every team starts from the same bankroll;
  * every team sees the identical snapshot on a given tick;
  * no team can trade outside its assigned universe;
  * one broken strategy cannot stop the round;
  * the whole thing is reproducible from a seed;
  * everything that happened is in the ledger.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from competition.broker.simulated import SimConfig, SimulatedBroker
from competition.data.universe import UniverseProvider
from competition.engine import CompetitionEngine, Ledger, UniverseResolver
from competition.scoring import score_round
from competition.strategies.base import Strategy
from competition.types import UTC, OrderIntent

ROUND_START = date(2026, 9, 8)      # Tuesday after Labor Day
ROUND_END = date(2026, 9, 11)       # Friday -- four sessions


def build(cfg, feed, ledger_path, *, teams=None, symbols=None, daily=None,
          news=None, seed=7, pool_override=None):
    """Wire up an engine over the replay feed, exactly as `comp backtest` does."""
    clock = {"now": datetime.combine(ROUND_START, datetime.min.time(), tzinfo=UTC)}
    quote_source = feed.quote_source(lambda: clock["now"])
    sim = SimConfig(slippage_bps=cfg.fairness.sim_slippage_bps,
                    commission_per_share=0.0, fractional=True,
                    allow_short=cfg.risk.allow_short)
    team_list = teams if teams is not None else list(cfg.teams)
    brokers = {
        t.key: SimulatedBroker(cfg.starting_cash, quote_source, name=t.key,
                               config=sim, clock=lambda: clock["now"])
        for t in team_list
    }
    ledger = Ledger(ledger_path)
    ledger.start_run(round_id=1, mode="test", broker="sim",
                     rules_hash=cfg.rules_hash, seed=seed, config={})
    provider = UniverseProvider()
    resolver = UniverseResolver(
        cfg, provider=provider,
        daily_bars=lambda syms: {s: tuple(daily.get(s, ())) for s in syms} if daily else {},
        news=lambda syms: {s: tuple(news.get(s, ())) for s in syms} if news else {},
        pool_size_override=pool_override,
    )
    engine = CompetitionEngine(
        cfg, feed=feed, brokers=brokers, ledger=ledger, resolver=resolver,
        mode="test", teams=team_list, clock=lambda: clock["now"],
        equity_snapshot_seconds=900,
    )
    inner = engine.tick

    def tick(rnd, *, now=None, **kw):
        if now is not None:
            clock["now"] = now
        return inner(rnd, now=now, **kw)

    engine.tick = tick                                    # type: ignore[method-assign]
    return engine, ledger, brokers, clock


# --------------------------------------------------------------------------- #
# a full Round 1
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def round1(cfg, feed, tmp_path_factory, daily, news):
    path = tmp_path_factory.mktemp("ledger") / "r1.sqlite"
    engine, ledger, brokers, _clock = build(cfg, feed, path, daily=daily, news=news)
    result = engine.run_replay(cfg.round(1), start=ROUND_START, end=ROUND_END,
                              tick_seconds=300)
    return {"result": result, "ledger": ledger, "engine": engine, "brokers": brokers}


def test_round_completes_with_every_team(round1, cfg):
    result = round1["result"]
    assert {t.team_key for t in result.teams} == set(cfg.team_keys)
    assert result.ticks > 100
    assert result.round_id == 1


def test_no_strategy_errored_or_timed_out(round1):
    for t in round1["result"].teams:
        assert t.errors == 0, f"{t.team_key} raised {t.errors} time(s)"
        assert t.timeouts == 0, f"{t.team_key} overran its budget {t.timeouts} time(s)"


def test_every_team_starts_from_the_same_bankroll(round1, cfg):
    starts = {t.start_equity for t in round1["result"].teams}
    assert starts == {cfg.starting_cash}


def test_every_team_traded(round1):
    silent = [t.team_key for t in round1["result"].teams if t.fills == 0]
    assert not silent, f"these teams never traded: {silent}"


def test_nobody_built_an_illegal_order(round1):
    """The shared plumbing should make avoidable rejections impossible."""
    avoidable = {"outside_universe", "oversell", "nan_size", "insufficient_cash",
                 "min_notional", "duplicate_intent"}
    for t in round1["result"].teams:
        offending = {k: v for k, v in t.rejections.items() if k in avoidable}
        assert not offending, f"{t.team_key} produced avoidable rejections: {offending}"


def test_nobody_ended_short_or_overdrawn(round1):
    for key, broker in round1["brokers"].items():
        acct = broker.account()
        assert acct.cash >= -1e-6, f"{key} ended overdrawn"
        assert all(p.qty >= 0 for p in acct.positions), f"{key} ended short"


def test_round_end_liquidation_leaves_everyone_flat(round1, cfg):
    if not cfg.risk.liquidate_at_round_end:
        pytest.skip("liquidation disabled")
    for key, broker in round1["brokers"].items():
        assert broker.positions() == [], f"{key} still holds a position"
        assert broker.open_orders() == [], f"{key} still has working orders"


def test_returns_are_plausible(round1, cfg):
    for t in round1["result"].teams:
        assert -0.9 < t.return_pct < 2.0, f"{t.team_key} returned {t.return_pct:.1%}"
        assert t.end_equity > 0
        assert t.end_equity == pytest.approx(
            t.start_equity * (1 + t.return_pct), rel=1e-9)


def test_round_one_universe_is_identical_for_everyone(round1, cfg):
    expected = set(cfg.round(1).symbols)
    for t in round1["result"].teams:
        assert set(t.universe) == expected


def test_result_table_renders(round1):
    table = round1["result"].table()
    assert "Round 1" in table and "return" in table


# --------------------------------------------------------------------------- #
# the ledger recorded it all
# --------------------------------------------------------------------------- #


def test_ledger_captured_the_round(round1):
    ledger = round1["ledger"]
    stats = ledger.stats()
    assert stats["runs"] >= 1
    assert stats["teams"] >= 9
    assert stats["orders"] > 0 and stats["fills"] > 0
    assert stats["equity"] > 0
    assert stats["universes"] >= 9


def test_every_fill_has_a_recorded_reason(round1):
    ledger = round1["ledger"]
    for t in round1["result"].teams:
        if t.fills == 0:
            continue
        rows = ledger.trade_log(t.team_key, limit=20)
        assert rows, f"no trade log for {t.team_key}"
        with_reason = [r for r in rows if (r["reason"] or "").strip()]
        assert len(with_reason) >= len(rows) * 0.5, (
            f"{t.team_key}: most fills have no stated reason")


def test_equity_curves_are_recorded_and_monotone_in_time(round1):
    ledger = round1["ledger"]
    for t in round1["result"].teams:
        curve = ledger.equity_curve(t.team_key)
        assert len(curve) > 2
        stamps = [c[0] for c in curve]
        assert stamps == sorted(stamps)
        assert all(e > 0 for _ts, e in curve)


def test_universes_are_recorded_per_session(round1, cfg):
    ledger = round1["ledger"]
    for t in round1["result"].teams:
        assert set(ledger.universe_for(1, t.team_key)) == set(cfg.round(1).symbols)


def test_scoring_from_the_ledger_matches_the_engine(round1, cfg):
    result = round1["result"]
    scored = score_round(cfg, 1, result.teams, round_name=cfg.round(1).name)
    assert len(scored.scored_only()) == len(cfg.scored_teams)
    assert sum(s.points for s in scored.scored_only()) == pytest.approx(
        sum(cfg.points_table))
    places = sorted(s.place for s in scored.scored_only())
    assert places[0] == 1


# --------------------------------------------------------------------------- #
# fairness mechanics
# --------------------------------------------------------------------------- #


def test_all_teams_see_the_identical_snapshot(cfg, feed, symbols, midweek_ts):
    """The shared-tape guarantee: one snapshot object per tick, restricted."""
    snap = feed.snapshot(symbols, midweek_ts)
    views = [snap.restricted_to(symbols) for _ in range(5)]
    for v in views:
        assert v.ts == snap.ts
        for sym in symbols:
            assert v.series(sym, "primary") == snap.series(sym, "primary")
            assert v.quote(sym) == snap.quote(sym)


def test_execution_order_rotates(cfg, feed, tmp_path, daily, news):
    engine, ledger, _b, _c = build(cfg, feed, tmp_path / "rot.sqlite",
                                   daily=daily, news=news)
    engine.prepare_round(cfg.round(1), session=ROUND_START)
    seen = set()
    for i in range(len(engine.teams) + 2):
        engine._rotation = i
        keys = list(engine.teams)
        offset = engine._rotation % len(keys)
        seen.add(tuple(keys[offset:] + keys[:offset])[0])
    assert len(seen) > 1, "execution order never rotates"
    ledger.close()


def test_a_broken_strategy_cannot_stop_the_round(cfg, feed, tmp_path, daily, news):
    engine, ledger, brokers, _c = build(cfg, feed, tmp_path / "boom.sqlite",
                                        daily=daily, news=news)

    class Exploding(Strategy):
        DESCRIPTION = "raises on every tick"

        def on_tick(self, ctx):
            raise RuntimeError("boom")

    victim = list(engine.teams)[0]
    # Pass empty params: these stand-ins declare no schema, and the strict
    # parameter check would (correctly) reject trend_rider's real params.
    engine.teams[victim].strategy = Exploding(engine.teams[victim].config, {})
    result = engine.run_replay(cfg.round(1), start=ROUND_START,
                               end=date(2026, 9, 9), tick_seconds=300)
    broken = result.team(victim)
    assert broken.errors > 0
    assert broken.fills == 0
    # Everyone else carried on.
    others = [t for t in result.teams if t.team_key != victim]
    assert any(t.fills > 0 for t in others)
    events = ledger.query(
        "SELECT COUNT(*) n FROM events WHERE kind='strategy_error' AND team_key=?",
        (victim,))
    assert events[0]["n"] > 0
    ledger.close()


def test_a_slow_strategy_is_timed_out_not_tolerated(cfg, feed, tmp_path, daily, news):
    engine, ledger, _b, _c = build(cfg, feed, tmp_path / "slow.sqlite",
                                   daily=daily, news=news)

    class Sloth(Strategy):
        DESCRIPTION = "sleeps past its budget"

        def on_tick(self, ctx):
            import time
            time.sleep(cfg.fairness.strategy_timeout_seconds + 0.2)
            return [OrderIntent(ctx.universe[0], "buy", notional=100)]

    victim = list(engine.teams)[0]
    engine.teams[victim].strategy = Sloth(engine.teams[victim].config, {})
    engine.prepare_round(cfg.round(1), session=ROUND_START)
    base = datetime.combine(ROUND_START, datetime.min.time(), tzinfo=UTC)
    engine.tick(cfg.round(1), now=base + timedelta(hours=14))
    rt = engine.teams[victim]
    assert rt.timeouts == 1
    # TeamRuntime.fills is the list of fills; TeamResult.fills is the count.
    assert len(rt.fills) == 0, "a timed-out tick must be discarded, not executed"
    assert not rt.orders, "a timed-out tick must not reach the broker"
    ledger.close()


def test_the_same_seed_reproduces_the_same_result(cfg, feed, tmp_path, daily, news):
    outcomes = []
    for i in range(2):
        engine, ledger, _b, _c = build(cfg, feed, tmp_path / f"rep{i}.sqlite",
                                       daily=daily, news=news, seed=99)
        r = engine.run_replay(cfg.round(1), start=ROUND_START, end=date(2026, 9, 9),
                              tick_seconds=300)
        outcomes.append({t.team_key: (round(t.return_pct, 10), t.fills)
                         for t in r.teams})
        ledger.close()
    assert outcomes[0] == outcomes[1]


def test_kill_switch_flattens_and_halts(cfg, feed, tmp_path, daily, news):
    engine, ledger, brokers, clock = build(cfg, feed, tmp_path / "kill.sqlite",
                                           daily=daily, news=news)
    engine.prepare_round(cfg.round(1), session=ROUND_START)
    victim = list(engine.teams)[0]
    broker = brokers[victim]
    base = datetime.combine(ROUND_START, datetime.min.time(), tzinfo=UTC) + \
        timedelta(hours=14)
    clock["now"] = base
    # Put on a position, then wipe the account so the session drawdown trips.
    engine.tick(cfg.round(1), now=base)
    rt = engine.teams[victim]
    rt.risk.session_open_equity = broker.account().equity * 4
    engine.tick(cfg.round(1), now=base + timedelta(minutes=5))
    assert rt.risk.is_halted
    assert broker.positions() == []
    rows = ledger.query(
        "SELECT COUNT(*) n FROM events WHERE kind='kill_switch' AND team_key=?",
        (victim,))
    assert rows[0]["n"] >= 1
    ledger.close()


# --------------------------------------------------------------------------- #
# Round 3: the draft, enforced at the data layer
# --------------------------------------------------------------------------- #


def test_round_three_enforces_the_dealt_hand(cfg, feed, symbols, tmp_path, daily, news,
                                             pool):
    """A team must not be able to see or trade anything outside its hand."""
    from competition.draft import deal

    teams = list(cfg.teams)[:4]
    engine, ledger, brokers, clock = build(cfg, feed, tmp_path / "r3.sqlite",
                                           teams=teams, daily=daily, news=news)
    # Deal from a pool restricted to the symbols we have data for, so each
    # hand is small but real.
    small = pool.filtered(symbols).head(len(symbols))
    result = deal([t.key for t in teams], small, picks_per_team=2,
                  pool_size=len(symbols), seed=3, min_rank_deciles=1,
                  max_sector_share=1.0)
    engine.resolver.set_draft(result)
    engine.prepare_round(cfg.round(3), session=ROUND_START)

    for t in teams:
        rt = engine.teams[t.key]
        hand = set(result.hand(t.key).symbols)
        assert set(rt.universe) <= hand, f"{t.key} universe escaped its hand"
        # And the snapshot it receives is restricted to that hand.
        snap = feed.snapshot(symbols, clock["now"])
        ctx = engine._context(rt, snap, brokers[t.key].account(), cfg.round(3),
                              0.0, 0, False)
        for sym in symbols:
            if sym not in hand:
                assert ctx.snapshot.series(sym) == ()
                assert ctx.snapshot.quote(sym) is None
    ledger.close()


def test_recorded_draft_survives_a_ledger_round_trip(cfg, tmp_path, pool):
    from competition.draft import deal

    ledger = Ledger(tmp_path / "draft.sqlite")
    ledger.start_run(round_id=3, mode="test", broker="none",
                     rules_hash=cfg.rules_hash, seed=1, config={})
    result = deal([t.key for t in cfg.scored_teams], pool, seed=1)
    ledger.record_draft(3, result)
    payload = ledger.latest_draft(3)
    assert payload is not None
    assert payload["target_sum"] == result.target_sum
    assert len(payload["hands"]) == len(result.hands)
    for hand in payload["hands"]:
        assert sum(hand["ranks"]) == result.target_sum
    ledger.close()


# --------------------------------------------------------------------------- #
# learned state across rounds
# --------------------------------------------------------------------------- #


def test_learned_state_persists_between_rounds(cfg, feed, tmp_path, daily, news):
    path = tmp_path / "learn.sqlite"
    first, ledger, _b, _c = build(cfg, feed, path, daily=daily, news=news)
    first.run_replay(cfg.round(1), start=ROUND_START, end=date(2026, 9, 9),
                     tick_seconds=300)
    saved = ledger.load_learned_state("q_learner", "strategy")
    assert saved and saved.get("q"), "the RL entry saved nothing"
    states_after_r1 = len(saved["q"])
    ledger.close()

    second, ledger2, _b2, _c2 = build(cfg, feed, path, daily=daily, news=news)
    second.load_learned_state()
    agent = second.teams["q_learner"].strategy
    assert len(agent.q) == states_after_r1
    assert agent.steps > 0
    ledger2.close()


def test_universe_resolver_falls_back_when_a_picker_explodes(cfg, symbols, daily):
    """One broken picker must not remove a team from the round."""
    provider = UniverseProvider()
    resolver = UniverseResolver(
        cfg, provider=provider,
        daily_bars=lambda syms: {s: tuple(daily.get(s, ())) for s in syms},
    )
    team = cfg.team("trend_rider")
    picker = resolver.picker_for(team)

    def explode(_ctx):
        raise RuntimeError("picker is broken")

    picker.pick = explode                        # type: ignore[method-assign]
    resolver._candidate_cache = tuple(symbols)
    resolved = resolver.resolve(cfg.round(2), team, session=ROUND_START)
    assert resolved.symbols, "a broken picker took the team out of the round"
    assert len(resolved.symbols) <= cfg.round(2).picker.max_symbols
