"""Every stock picker.

Two contracts:
  * mechanical -- return at most `max_symbols`, only from the candidate pool,
    no duplicates, never raise;
  * philosophical -- each picker must select the *kind* of name its strategy
    trades. That is the competition rule ("the picker has to use a similar
    approach to the actual team"), so it is tested, not assumed.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from competition.pickers import load_picker
from competition.pickers.base import Picker, PickerContext, PickResult
from competition.types import Bar, NewsItem, utcnow
from competition.util import indicators as ind

ALL = ["trend_rider", "mean_reverter", "q_learner", "gambler", "stat_arb",
       "news_hound", "vol_breakout", "scalper", "benchmark"]


def ctx_for(daily, symbols, *, news=None, sectors=None, max_symbols=5, round_id=2,
           locked=False, state=None):
    return PickerContext(
        round_id=round_id, candidates=tuple(symbols),
        daily_bars={s: tuple(daily[s]) for s in symbols},
        max_symbols=max_symbols, log=logging.getLogger("test.picker"),
        news={k: tuple(v) for k, v in (news or {}).items()},
        sectors=dict(sectors or {}), state=state if state is not None else {},
        locked_universe=locked,
    )


# --------------------------------------------------------------------------- #
# mechanical contract
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("key", ALL)
def test_picker_loads(cfg, key):
    team = cfg.team(key)
    picker = load_picker(team.picker)(team)
    assert isinstance(picker, Picker)
    assert picker.DESCRIPTION
    assert picker.describe()


@pytest.mark.parametrize("key", ALL)
def test_unknown_picker_params_are_rejected(cfg, key):
    team = cfg.team(key)
    cls = load_picker(team.picker)
    with pytest.raises(ValueError, match="unknown parameter"):
        cls(team, {**team.picker_params, "bogus": 1})


@pytest.mark.parametrize("key", ALL)
@pytest.mark.parametrize("cap", [1, 3, 5, 10])
def test_respects_the_cap_and_the_pool(cfg, key, cap, daily, symbols, news):
    team = cfg.team(key)
    picker = load_picker(team.picker)(team)
    ctx = ctx_for(daily, symbols, news=news, max_symbols=cap)
    result = picker.pick(ctx)
    assert isinstance(result, PickResult)
    assert len(result.symbols) <= cap
    assert len(set(result.symbols)) == len(result.symbols)
    assert set(result.symbols) <= set(symbols)
    assert result.describe()
    assert isinstance(result.to_dict(), dict)


@pytest.mark.parametrize("key", ALL)
def test_empty_pool_is_survivable(cfg, key):
    team = cfg.team(key)
    picker = load_picker(team.picker)(team)
    result = picker.pick(ctx_for({}, [], max_symbols=5))
    assert result.symbols == ()


@pytest.mark.parametrize("key", ALL)
def test_no_history_is_survivable(cfg, key, symbols):
    team = cfg.team(key)
    picker = load_picker(team.picker)(team)
    stub = {s: [Bar(s, utcnow(), 10, 11, 9, 10, 1e6, 100, 10)] for s in symbols}
    result = picker.pick(ctx_for(stub, symbols, max_symbols=5))
    assert len(result.symbols) <= 5      # must not raise


@pytest.mark.parametrize("key", ALL)
def test_round_three_locked_hand_is_respected(cfg, key, daily, symbols, news):
    """In Round 3 the picker may only rank within the ten dealt names."""
    team = cfg.team(key)
    picker = load_picker(team.picker)(team)
    hand = symbols[:4]
    ctx = ctx_for(daily, hand, news=news, max_symbols=4, round_id=3, locked=True)
    result = picker.pick(ctx)
    assert set(result.symbols) <= set(hand)


def test_picker_loader_rejects_bad_specs():
    from competition.pickers import load_picker as lp
    with pytest.raises(ValueError):
        lp("no_colon")
    with pytest.raises(ImportError):
        lp("competition.nope:Thing")
    with pytest.raises(TypeError):
        lp("competition.pickers.base:PickResult")


# --------------------------------------------------------------------------- #
# philosophical contract -- built on a pool with known, planted regimes
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def labelled_pool():
    """A pool where we know exactly which name is which archetype."""
    rng = np.random.default_rng(20260101)
    n = 260

    def bars(sym, closes, trades, range_mult=1.0, gap=0.004, dollar_vol=3e8):
        out = []
        prev = float(closes[0])
        for c in closes:
            c = float(c)
            op = prev * (1 + rng.normal(0, gap))
            span = abs(c - op) + 0.012 * c * range_mult
            hi = max(op, c) + span * 0.5
            lo = max(min(op, c) - span * 0.5, 0.01)
            out.append(Bar(sym, utcnow(), op, hi, lo, c, dollar_vol / max(c, 1),
                           trades, c))
            prev = c
        return out

    def rw(mu, sig, p0=100.0):
        return p0 * np.cumprod(1 + rng.normal(mu, sig, n))

    def ou(level, k, sig):
        s = [level]
        for _ in range(n - 1):
            s.append(s[-1] + -k * (s[-1] - level) + rng.normal(0, sig * level))
        return np.array(s)

    def coil(p0=100.0):
        a = p0 * np.cumprod(1 + rng.normal(0.001, 0.022, 200))
        lvl = a[-1]
        b = [lvl]
        for _ in range(n - 201):
            b.append(b[-1] + -0.6 * (b[-1] - lvl) + rng.normal(0, 0.004 * lvl))
        return np.concatenate([a, np.array(b)])

    pool = {
        "MOMO": bars("MOMO", rw(0.0038, 0.016), 60_000, gap=0.006),
        "REVERT": bars("REVERT", ou(80, 0.30, 0.022), 55_000),
        "WILD": bars("WILD", rw(0.002, 0.070, 40), 70_000, range_mult=3.0, gap=0.035),
        "COILED": bars("COILED", coil(), 45_000, range_mult=0.3, gap=0.002),
        "LIQUID": bars("LIQUID", rw(0.0004, 0.008, 250), 450_000, range_mult=0.5,
                       gap=0.002, dollar_vol=9e8),
        "PAIRA": bars("PAIRA", rw(0.0008, 0.014, 120), 50_000),
        "DOWN": bars("DOWN", rw(-0.003, 0.020), 40_000, gap=0.006),
    }
    base = np.array([b.close for b in pool["PAIRA"]])
    sp = [0.0]
    for _ in range(n - 1):
        sp.append(sp[-1] * 0.82 + rng.normal(0, 0.012))
    pool["PAIRB"] = bars("PAIRB", 60 * np.exp(np.log(base / 120) * 1.1 + np.array(sp)),
                         52_000)
    return pool


@pytest.fixture(scope="module")
def labelled_news():
    now = utcnow()
    from datetime import timedelta
    return {
        "MOMO": [NewsItem(f"m{i}", now - timedelta(hours=2 + i * 3),
                          "MOMO beats estimates and raises guidance", "", "Reuters",
                          ("MOMO",)) for i in range(5)],
        "DOWN": [NewsItem(f"d{i}", now - timedelta(hours=2 + i * 3),
                          "DOWN Corp misses estimates and cuts guidance", "", "Reuters",
                          ("DOWN",)) for i in range(4)],
    }


SECTORS = {"MOMO": "Technology", "REVERT": "Utilities", "WILD": "Energy",
           "COILED": "Industrials", "LIQUID": "Financials", "PAIRA": "Health Care",
           "PAIRB": "Health Care", "DOWN": "Materials"}


def pick(cfg, key, labelled_pool, labelled_news, cap=3):
    team = cfg.team(key)
    picker = load_picker(team.picker)(team)
    ctx = ctx_for(labelled_pool, list(labelled_pool), news=labelled_news,
                  sectors=SECTORS, max_symbols=cap)
    return picker.pick(ctx)


def test_momentum_picker_picks_the_uptrend(cfg, labelled_pool, labelled_news):
    result = pick(cfg, "trend_rider", labelled_pool, labelled_news)
    assert "MOMO" in result.symbols
    assert "DOWN" not in result.symbols, "a momentum screen must not pick a downtrend"


def test_reversion_picker_picks_the_reverting_name(cfg, labelled_pool, labelled_news):
    result = pick(cfg, "mean_reverter", labelled_pool, labelled_news, cap=4)
    assert "REVERT" in result.symbols
    # And it must measure reversion, not just pick anything volatile.
    closes = [b.close for b in labelled_pool["REVERT"]]
    assert ind.half_life(closes) < 12


def test_reversion_picker_rejects_a_trending_name(cfg, labelled_pool, labelled_news):
    team = cfg.team("mean_reverter")
    picker = load_picker(team.picker)(team)
    ctx = ctx_for(labelled_pool, list(labelled_pool), sectors=SECTORS, max_symbols=8)
    result = picker.pick(ctx)
    assert "MOMO" in result.rejected or "MOMO" not in result.symbols


def test_lottery_picker_picks_the_wildest_name(cfg, labelled_pool, labelled_news):
    result = pick(cfg, "gambler", labelled_pool, labelled_news, cap=2)
    assert "WILD" in result.symbols
    assert "LIQUID" not in result.symbols, "a variance seeker must not pick the grinder"


def test_squeeze_picker_prefers_the_coiled_name(cfg, labelled_pool, labelled_news):
    # With one slot it must choose the coil and nothing else.
    top = pick(cfg, "vol_breakout", labelled_pool, labelled_news, cap=1)
    assert top.symbols == ("COILED",)
    # With more slots it may reach further down a small pool, but the coil
    # must still outrank the 7%-a-day lottery ticket.
    result = pick(cfg, "vol_breakout", labelled_pool, labelled_news, cap=3)
    assert result.symbols[0] == "COILED"
    if "WILD" in result.scores:
        assert result.scores["COILED"] > result.scores["WILD"]


def test_liquidity_picker_prefers_the_liquid_grinder(cfg, labelled_pool, labelled_news):
    result = pick(cfg, "scalper", labelled_pool, labelled_news, cap=3)
    assert result.symbols[0] == "LIQUID"


def test_pair_picker_finds_the_planted_pair(cfg, labelled_pool, labelled_news):
    result = pick(cfg, "stat_arb", labelled_pool, labelled_news, cap=8)
    pairs = result.extra.get("pairs", [])
    assert pairs, "no pair found in a pool containing a planted one"
    members = {frozenset((p["y"], p["x"])) for p in pairs}
    assert frozenset({"PAIRA", "PAIRB"}) in members
    assert {"PAIRA", "PAIRB"} <= set(result.symbols)


def test_newsflow_picker_follows_positive_coverage(cfg, labelled_pool, labelled_news):
    result = pick(cfg, "news_hound", labelled_pool, labelled_news, cap=3)
    assert "MOMO" in result.symbols
    assert "DOWN" not in result.symbols, "negative coverage must not be selected"
    assert "COILED" not in result.symbols, "a name with no coverage is unusable"


def test_the_field_does_not_converge_on_one_book(cfg, labelled_pool, labelled_news):
    """Different philosophies must produce different shortlists."""
    picks = {}
    for key in ["trend_rider", "mean_reverter", "gambler", "vol_breakout",
                "scalper", "stat_arb", "news_hound"]:
        picks[key] = frozenset(pick(cfg, key, labelled_pool, labelled_news, cap=3))
    # At least five of the seven shortlists must be distinct.
    assert len(set(picks.values())) >= 5, picks


# --------------------------------------------------------------------------- #
# the learning picker
# --------------------------------------------------------------------------- #


def test_bandit_learns_and_round_trips(cfg, labelled_pool, labelled_news):
    team = cfg.team("q_learner")
    cls = load_picker(team.picker)
    a = cls(team)
    ctx = ctx_for(labelled_pool, list(labelled_pool), sectors=SECTORS, max_symbols=3)
    first = a.pick(ctx)
    assert first.symbols
    assert a.pulls == 0
    a.on_fills({s: 0.05 for s in first.symbols})
    assert a.pulls == len(first.symbols)
    blob = a.state_dict()
    b = cls(team)
    b.load_state(blob)
    assert b.pulls == a.pulls
    np_ok = (b.A == a.A).all() and (b.b == a.b).all()
    assert np_ok


def test_bandit_rejects_a_mismatched_feature_set(cfg):
    team = cfg.team("q_learner")
    a = load_picker(team.picker)(team)
    a.load_state({"feature_set": ["only_one"], "A": [[1.0]], "b": [1.0], "pulls": 9})
    assert a.pulls == 0


def test_bandit_reward_is_clipped(cfg, labelled_pool, labelled_news):
    team = cfg.team("q_learner")
    a = load_picker(team.picker)(team)
    result = a.pick(ctx_for(labelled_pool, list(labelled_pool), max_symbols=2))
    a.on_fills({s: 10_000.0 for s in result.symbols})     # absurd reward
    assert np.isfinite(a.b).all()
    assert abs(a.b).max() < 100
