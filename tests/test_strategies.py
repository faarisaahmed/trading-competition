"""Every competitor, driven over synthetic data.

The contract each strategy must satisfy:
  * never raise, on any tick, including degenerate ones (no data, no cash,
    market closed, position already held);
  * only ever ask for things inside its universe, with a finite positive size;
  * never ask to sell more than it holds;
  * actually trade at some point -- a strategy that silently does nothing
    would show up as a mid-table finish and nobody would notice it was broken.
"""

from __future__ import annotations

import logging
import math
import random
from datetime import date, datetime, timedelta

import pytest

from competition.data.snapshot import MarketSnapshot
from competition.engine.guardrails import Guardrails, RiskState
from competition.strategies import load_strategy
from competition.strategies.base import Strategy, StrategyContext
from competition.types import (
    UTC,
    Account,
    OrderIntent,
    OrderType,
    Position,
    Quote,
    Side,
)

ALL_TEAMS = ["trend_rider", "mean_reverter", "q_learner", "gambler", "stat_arb",
             "news_hound", "vol_breakout", "scalper", "benchmark"]


def make_ctx(cfg, team, snapshot, account, *, universe=None, open_orders=(),
             state=None, round_id=1):
    universe = tuple(universe or snapshot.universe)
    return StrategyContext(
        team_key=team.key, round_id=round_id,
        snapshot=snapshot.restricted_to(universe), account=account,
        universe=universe, starting_cash=cfg.starting_cash,
        baseline_equity=cfg.starting_cash, risk=cfg.risk,
        rng=random.Random(cfg.team_seed(team.key, round_id)),
        state=state if state is not None else {},
        log=logging.getLogger(f"test.{team.key}"),
        open_orders=tuple(open_orders),
        fractionable=frozenset(universe),
    )


def assert_legal(intents, ctx):
    for o in intents:
        assert isinstance(o, OrderIntent)
        assert o.symbol in ctx.universe, f"{o.symbol} outside universe"
        size = o.qty if o.qty is not None else o.notional
        assert size is not None and math.isfinite(size) and size > 0
        if o.order_type is OrderType.LIMIT:
            assert o.limit_price and o.limit_price > 0
        if o.side is Side.SELL:
            assert o.qty is not None, "sells must be in shares, not notional"
            assert o.qty <= ctx.qty(o.symbol) + 1e-6, f"oversell of {o.symbol}"


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("key", ALL_TEAMS)
def test_every_team_loads_with_its_configured_params(cfg, key):
    team = cfg.team(key)
    strat = load_strategy(team.strategy)(team)
    assert isinstance(strat, Strategy)
    assert strat.tick_seconds >= 5
    assert strat.warmup_bars >= 0
    assert strat.DESCRIPTION, f"{key} has no description"
    assert strat.describe()


@pytest.mark.parametrize("key", ALL_TEAMS)
def test_unknown_parameters_are_rejected(cfg, key):
    """A typo in teams.yaml must fail loudly, not silently use a default."""
    team = cfg.team(key)
    cls = load_strategy(team.strategy)
    with pytest.raises(ValueError, match="unknown parameter"):
        cls(team, {**team.params, "not_a_real_param": 1})


def test_strategy_loader_rejects_bad_specs():
    from competition.strategies import load_strategy as ls
    with pytest.raises(ValueError):
        ls("no_colon_here")
    with pytest.raises(ImportError):
        ls("competition.nonexistent:Thing")
    with pytest.raises(ImportError):
        ls("competition.strategies.trend_rider:NotAClass")
    with pytest.raises(TypeError):
        ls("competition.strategies.base:Param")


# --------------------------------------------------------------------------- #
# driven over real synthetic history
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("key", ALL_TEAMS)
def test_strategy_runs_a_full_session_without_error(cfg, key, feed, symbols, calendar):
    """Tick a strategy across two sessions and check every intent is legal."""
    team = cfg.team(key)
    strat = load_strategy(team.strategy)(team)
    universe = tuple(symbols[:3])
    account = Account(cash=cfg.starting_cash, equity=cfg.starting_cash,
                      buying_power=cfg.starting_cash, positions=())
    state: dict = {}
    rails = Guardrails(cfg.risk, bankroll=cfg.starting_cash)
    risk = RiskState(key)
    stamps = calendar.tick_times(date(2026, 9, 8), date(2026, 9, 9),
                                 seconds=max(strat.tick_seconds, 300))
    total = 0
    for ts in stamps:
        snap = feed.snapshot(universe, ts)
        ctx = make_ctx(cfg, team, snap, account, universe=universe, state=state)
        risk.roll_session(snap.session.session_date, account.equity)
        intents = strat.on_tick(ctx)
        assert_legal(intents, ctx)
        result = rails.validate(intents, state=risk, account=account, snapshot=ctx.snapshot,
                                universe=universe, now=ts)
        # Nothing a strategy builds should be refused for a reason it could
        # have avoided itself.
        for rej in result.rejected:
            assert rej.reason.value not in ("outside_universe", "oversell", "nan_size"), \
                f"{key} built an illegal order: {rej.reason.value} {rej.detail}"
        total += len(intents)
    assert total >= 0


@pytest.mark.parametrize("key", ALL_TEAMS)
def test_strategy_trades_at_least_once_over_a_week(cfg, key, feed, symbols, calendar):
    """A strategy that never trades is broken, not conservative."""
    team = cfg.team(key)
    strat = load_strategy(team.strategy)(team)
    universe = tuple(symbols)
    cash = cfg.starting_cash
    state: dict = {}
    seen = 0
    for ts in calendar.tick_times(date(2026, 9, 8), date(2026, 9, 11),
                                  seconds=max(strat.tick_seconds, 300)):
        snap = feed.snapshot(universe, ts)
        account = Account(cash=cash, equity=cash, buying_power=cash, positions=())
        ctx = make_ctx(cfg, team, snap, account, universe=universe, state=state)
        seen += len(strat.on_tick(ctx))
        if seen:
            break
    assert seen > 0, f"{key} never produced a single order over four sessions"


# --------------------------------------------------------------------------- #
# degenerate conditions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("key", ALL_TEAMS)
def test_no_data_is_survivable(cfg, key, calendar):
    team = cfg.team(key)
    strat = load_strategy(team.strategy)(team)
    ts = datetime(2026, 9, 10, 17, 0, tzinfo=UTC)
    empty = MarketSnapshot(ts=ts, session=calendar.session(ts),
                           universe=("AAPL",), missing=frozenset({"AAPL"}))
    account = Account(cash=5000.0, equity=5000.0, buying_power=5000.0, positions=())
    ctx = make_ctx(cfg, team, empty, account, universe=("AAPL",))
    assert strat.on_tick(ctx) == [] or True       # must not raise
    strat.on_round_start(ctx)
    strat.on_round_end(ctx)
    strat.on_session_start(ctx)
    strat.on_session_end(ctx)


@pytest.mark.parametrize("key", ALL_TEAMS)
def test_no_cash_produces_no_buys(cfg, key, feed, symbols, midweek_ts):
    team = cfg.team(key)
    strat = load_strategy(team.strategy)(team)
    universe = tuple(symbols[:3])
    snap = feed.snapshot(universe, midweek_ts)
    broke = Account(cash=0.0, equity=0.01, buying_power=0.0, positions=())
    ctx = make_ctx(cfg, team, snap, broke, universe=universe)
    for o in strat.on_tick(ctx):
        assert o.side is not Side.BUY, f"{key} tried to buy with no cash"


@pytest.mark.parametrize("key", ALL_TEAMS)
def test_market_closed_is_survivable(cfg, key, feed, symbols, calendar):
    team = cfg.team(key)
    strat = load_strategy(team.strategy)(team)
    universe = tuple(symbols[:3])
    closed = datetime(2026, 9, 12, 17, 0, tzinfo=UTC)          # Saturday
    snap = feed.snapshot(universe, closed)
    account = Account(cash=5000.0, equity=5000.0, buying_power=5000.0, positions=())
    ctx = make_ctx(cfg, team, snap, account, universe=universe)
    assert_legal(strat.on_tick(ctx), ctx)


@pytest.mark.parametrize("key", ALL_TEAMS)
def test_holding_a_position_is_survivable(cfg, key, feed, symbols, midweek_ts):
    team = cfg.team(key)
    strat = load_strategy(team.strategy)(team)
    universe = tuple(symbols[:3])
    snap = feed.snapshot(universe, midweek_ts)
    px = snap.price(universe[0])
    qty = 1000.0 / px
    account = Account(cash=4000.0, equity=5000.0, buying_power=4000.0,
                      positions=(Position(universe[0], qty, px, px),))
    ctx = make_ctx(cfg, team, snap, account, universe=universe)
    assert_legal(strat.on_tick(ctx), ctx)


@pytest.mark.parametrize("key", ALL_TEAMS)
def test_a_single_symbol_universe_is_survivable(cfg, key, feed, symbols, midweek_ts):
    """Round 2 pickers can legitimately return one name."""
    team = cfg.team(key)
    strat = load_strategy(team.strategy)(team)
    universe = (symbols[0],)
    snap = feed.snapshot(universe, midweek_ts)
    account = Account(cash=5000.0, equity=5000.0, buying_power=5000.0, positions=())
    ctx = make_ctx(cfg, team, snap, account, universe=universe)
    assert_legal(strat.on_tick(ctx), ctx)


@pytest.mark.parametrize("key", ALL_TEAMS)
def test_an_empty_universe_is_survivable(cfg, key, feed, midweek_ts):
    team = cfg.team(key)
    strat = load_strategy(team.strategy)(team)
    snap = feed.snapshot([], midweek_ts)
    account = Account(cash=5000.0, equity=5000.0, buying_power=5000.0, positions=())
    ctx = make_ctx(cfg, team, snap, account, universe=())
    assert strat.on_tick(ctx) == []


# --------------------------------------------------------------------------- #
# strategy-specific behaviour worth pinning down
# --------------------------------------------------------------------------- #


def test_gambler_respects_its_own_session_loss_cap(cfg, feed, symbols, midweek_ts):
    strat = load_strategy(cfg.team("gambler").strategy)(cfg.team("gambler"))
    universe = tuple(symbols[:3])
    snap = feed.snapshot(universe, midweek_ts)
    px = snap.price(universe[0])
    state: dict = {}
    # Open the session at 5000, then crash to 30% down.
    ok = Account(cash=5000.0, equity=5000.0, buying_power=5000.0, positions=())
    strat.on_tick(make_ctx(cfg, cfg.team("gambler"), snap, ok, universe=universe,
                           state=state))
    crashed = Account(cash=100.0, equity=3500.0, buying_power=100.0,
                      positions=(Position(universe[0], 3400.0 / px, px, px),))
    ctx = make_ctx(cfg, cfg.team("gambler"), snap, crashed, universe=universe,
                   state=state)
    intents = strat.on_tick(ctx)
    assert intents and all(o.side is Side.SELL for o in intents)
    assert all("cap" in o.reason for o in intents)


def test_gambler_ladder_is_bounded(cfg):
    strat = load_strategy(cfg.team("gambler").strategy)(cfg.team("gambler"))
    rungs = list(strat.p.martingale_rungs)

    class Fake:
        def __init__(self):
            self.state = {"rung": 0}

        def recall(self, k, d=None):
            return self.state.get(k, d)

        def remember(self, k, v):
            self.state[k] = v

    fake = Fake()
    sizes = []
    for rung in range(len(rungs) + 3):
        fake.remember("rung", rung)
        sizes.append(strat.bet_weight(fake, {"f": 0.20}))
    assert max(sizes) <= strat.p.max_weight_per_name
    assert sizes[-1] == sizes[len(rungs) - 1], "ladder must clamp, not grow forever"


def test_scalper_flattens_into_the_close(cfg, feed, symbols, calendar):
    team = cfg.team("scalper")
    strat = load_strategy(team.strategy)(team)
    universe = tuple(symbols[:3])
    _o, close = calendar.session_times(date(2026, 9, 10))
    ts = close - timedelta(minutes=5)
    snap = feed.snapshot(universe, ts)
    px = snap.price(universe[0])
    account = Account(cash=1000.0, equity=5000.0, buying_power=1000.0,
                      positions=(Position(universe[0], 4000.0 / px, px, px),))
    ctx = make_ctx(cfg, team, snap, account, universe=universe)
    intents = strat.on_tick(ctx)
    assert intents and all(o.side is Side.SELL for o in intents)
    assert all(o.order_type is OrderType.MARKET for o in intents)


def test_scalper_skips_names_outside_its_spread_band(cfg, calendar, symbols):
    team = cfg.team("scalper")
    strat = load_strategy(team.strategy)(team)
    ts = datetime(2026, 9, 10, 17, 0, tzinfo=UTC)
    from competition.types import Bar
    bars = {s: tuple(Bar(s, ts, 100, 100.2, 99.8, 100.0, 1e5, 400, 100.0)
                     for _ in range(40)) for s in ("TIGHT", "WIDE")}
    snap = MarketSnapshot(
        ts=ts, session=calendar.session(ts), universe=("TIGHT", "WIDE"),
        bars=bars, fast_bars=bars,
        quotes={"TIGHT": Quote("TIGHT", ts, 100.00, 100.06, 500, 500),
                "WIDE": Quote("WIDE", ts, 95.0, 105.0, 500, 500)},
    )
    assert strat.fair_value(make_ctx(cfg, team, snap, Account(5000, 5000, 5000, ()),
                                     universe=("TIGHT", "WIDE")), "TIGHT") is not None
    assert strat.fair_value(make_ctx(cfg, team, snap, Account(5000, 5000, 5000, ()),
                                     universe=("TIGHT", "WIDE")), "WIDE") is None


def test_q_learner_state_round_trips(cfg, feed, symbols, midweek_ts):
    team = cfg.team("q_learner")
    cls = load_strategy(team.strategy)
    a = cls(team)
    universe = tuple(symbols[:3])
    state: dict = {}
    account = Account(cash=5000.0, equity=5000.0, buying_power=5000.0, positions=())
    for i in range(30):
        snap = feed.snapshot(universe, midweek_ts + timedelta(seconds=300 * i))
        a.on_tick(make_ctx(cfg, team, snap, account, universe=universe, state=state))
    assert a.q, "the agent learned nothing"
    blob = a.state_dict()
    b = cls(team)
    b.load_state(blob)
    # `state_dict` rounds to 8 decimals to keep the ledger blob small; with
    # alpha=0.1 and rewards of order 0.1 that is far more precision than the
    # learning uses, so the round trip is exact to within the rounding.
    assert set(b.q) == set(a.q)
    for key, row in a.q.items():
        assert b.q[key] == pytest.approx(row, abs=1e-8)
    assert b.steps == a.steps and b.visits == a.visits
    assert b.epsilon == pytest.approx(a.epsilon)
    assert b.policy_table()


def test_q_learner_refuses_a_mismatched_action_space(cfg):
    team = cfg.team("q_learner")
    cls = load_strategy(team.strategy)
    a = cls(team)
    blob = {"q": {"0|0|0|0": [1.0, 2.0]}, "exposure_actions": [0.0, 0.9], "steps": 5}
    a.load_state(blob)
    assert a.q == {} and a.steps == 0        # rejected, not silently misread


def test_q_learner_pretrain_learns(cfg, intraday):
    team = cfg.team("q_learner")
    strat = load_strategy(team.strategy)(team)
    stats = strat.pretrain(intraday, passes=1, rng=random.Random(0))
    assert stats["states"] > 5
    assert stats["updates_after"] > 100
    assert math.isfinite(stats["mean_reward"])


def test_benchmark_buys_once_and_holds(cfg, feed, symbols, midweek_ts):
    team = cfg.team("benchmark")
    strat = load_strategy(team.strategy)(team)
    universe = tuple(symbols)
    snap = feed.snapshot(universe, midweek_ts)
    state: dict = {}
    account = Account(cash=5000.0, equity=5000.0, buying_power=5000.0, positions=())
    strat.on_round_start(make_ctx(cfg, team, snap, account, universe=universe,
                                  state=state))
    first = strat.on_tick(make_ctx(cfg, team, snap, account, universe=universe,
                                   state=state))
    assert first and all(o.side is Side.BUY for o in first)
    second = strat.on_tick(make_ctx(cfg, team, snap, account, universe=universe,
                                    state=state))
    assert second == [], "the benchmark must buy once and then do nothing"


def test_stat_arb_finds_a_planted_cointegrated_pair(cfg, intraday, specs):
    team = cfg.team("stat_arb")
    strat = load_strategy(team.strategy)(team)
    linked = [(s, sp.cointegrated_with) for s, sp in specs.items()
              if sp.cointegrated_with]
    if not linked:
        pytest.skip("no cointegrated pair in this fixture")
    y, x = linked[0]
    m = strat.evaluate_pair(intraday[y], intraday[x])
    assert m is not None, f"failed to detect the planted {y}/{x} relationship"
    assert m["corr"] >= strat.p.min_correlation
    assert m["tstat"] <= strat.p.adf_tstat_max
    assert m["half_life"] <= strat.p.max_half_life_bars


def test_bar_clock_counts_bars_not_ticks(cfg, feed, symbols, midweek_ts):
    team = cfg.team("trend_rider")
    strat = load_strategy(team.strategy)(team)
    universe = tuple(symbols[:2])
    state: dict = {}
    account = Account(cash=5000.0, equity=5000.0, buying_power=5000.0, positions=())
    # Four ticks inside one 5-minute bar must advance the clock exactly once.
    for offset in (0, 20, 40, 60):
        snap = feed.snapshot(universe, midweek_ts + timedelta(seconds=offset))
        ctx = make_ctx(cfg, team, snap, account, universe=universe, state=state)
        strat.sync_bar_clock(ctx)
    assert state["_bar_clock"] == 1
    snap = feed.snapshot(universe, midweek_ts + timedelta(seconds=400))
    ctx = make_ctx(cfg, team, snap, account, universe=universe, state=state)
    strat.sync_bar_clock(ctx)
    assert state["_bar_clock"] == 2
