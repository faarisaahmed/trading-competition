"""Deterministic synthetic market data.

Purpose: run the entire three-round competition -- all eight teams, every
guardrail, the draft, the pickers, the scoring -- with no API key and no
network. That makes the pipeline testable in CI and lets you watch the whole
thing end to end before a single real order is placed.

The generator is not trying to be a realistic market simulator. It is trying
to produce a tape with enough *structure* that each strategy has something to
find: trending names, mean-reverting names, coiled names that break out,
tight liquid names, correlated pairs, and lottery tickets. Anything less and
a dry run would show every team doing nothing, which tests nothing.

Everything is seeded, so the same seed gives byte-identical bars -- which is
what makes a sim leaderboard reproducible.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np

from ..types import UTC, Bar, NewsItem
from .calendar import MarketCalendar

#: The behavioural archetypes the generator can produce.
REGIMES = ("trend_up", "trend_down", "mean_revert", "coil_break", "chop",
           "lottery", "liquid_grind")


def _seed_for(symbol: str, seed: int) -> int:
    h = hashlib.sha256(f"{seed}:{symbol}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**32 - 1)


def assign_regime(symbol: str, seed: int) -> str:
    """Stable regime per (symbol, seed), so a name behaves consistently."""
    return REGIMES[_seed_for(symbol, seed) % len(REGIMES)]


@dataclass
class SyntheticSpec:
    """How to generate one symbol's path."""

    symbol: str
    regime: str
    start_price: float
    daily_vol: float
    daily_drift: float
    trade_count: int
    dollar_volume: float
    #: If set, this symbol tracks `cointegrated_with` plus a stationary spread.
    cointegrated_with: str | None = None
    beta: float = 1.0


def build_specs(
    symbols: Sequence[str],
    *,
    seed: int = 20260101,
    prices: Mapping[str, float] | None = None,
    pair_fraction: float = 0.25,
) -> dict[str, SyntheticSpec]:
    """Assign each symbol a regime, a starting price and (some) pair links."""
    syms = [s.upper() for s in dict.fromkeys(symbols)]
    specs: dict[str, SyntheticSpec] = {}
    for sym in syms:
        rng = np.random.default_rng(_seed_for(sym, seed))
        regime = assign_regime(sym, seed)
        px = float((prices or {}).get(sym) or rng.uniform(18.0, 420.0))
        if regime == "lottery":
            vol, drift = float(rng.uniform(0.045, 0.085)), float(rng.normal(0.0008, 0.004))
        elif regime == "liquid_grind":
            vol, drift = float(rng.uniform(0.008, 0.014)), float(rng.normal(0.0002, 0.0006))
        elif regime == "trend_up":
            vol, drift = float(rng.uniform(0.014, 0.024)), float(rng.uniform(0.0016, 0.0042))
        elif regime == "trend_down":
            vol, drift = float(rng.uniform(0.014, 0.026)), float(-rng.uniform(0.0014, 0.0038))
        elif regime == "mean_revert":
            vol, drift = float(rng.uniform(0.018, 0.030)), 0.0
        elif regime == "coil_break":
            vol, drift = float(rng.uniform(0.012, 0.020)), float(rng.normal(0.0006, 0.001))
        else:                                              # chop
            vol, drift = float(rng.uniform(0.012, 0.022)), float(rng.normal(0.0, 0.0008))
        trades = int(rng.uniform(8_000, 260_000))
        if regime == "liquid_grind":
            trades = int(rng.uniform(180_000, 700_000))
        specs[sym] = SyntheticSpec(
            symbol=sym, regime=regime, start_price=px, daily_vol=vol,
            daily_drift=drift, trade_count=trades,
            dollar_volume=float(rng.uniform(3e7, 9e8)),
        )

    # Wire up some cointegrated pairs so the stat-arb team has something real
    # to find rather than spurious correlations.
    n_pairs = max(int(len(syms) * pair_fraction) // 2, 1 if len(syms) >= 2 else 0)
    rng = np.random.default_rng(seed)
    pool = [s for s in syms if specs[s].regime in ("chop", "liquid_grind", "mean_revert")]
    rng.shuffle(pool)
    for i in range(min(n_pairs, len(pool) // 2)):
        a, b = pool[2 * i], pool[2 * i + 1]
        specs[b].cointegrated_with = a
        specs[b].beta = float(np.random.default_rng(_seed_for(b, seed)).uniform(0.7, 1.4))
    return specs


def _intraday_path(
    rng: np.random.Generator, spec: SyntheticSpec, n_bars: int, bars_per_session: int
) -> np.ndarray:
    """Close-to-close multiplicative path at the bar timeframe."""
    per_bar_vol = spec.daily_vol / math.sqrt(max(bars_per_session, 1))
    per_bar_drift = spec.daily_drift / max(bars_per_session, 1)
    regime = spec.regime

    if regime == "mean_revert":
        # OU around a slowly drifting level.
        kappa = 0.035
        level = np.zeros(n_bars)
        log_level = 0.0
        x = 0.0
        out = np.empty(n_bars)
        for i in range(n_bars):
            log_level += rng.normal(0, per_bar_vol * 0.15)
            x += -kappa * x + rng.normal(0, per_bar_vol * 2.2)
            level[i] = log_level
            out[i] = log_level + x
        return np.exp(out)

    if regime == "coil_break":
        # Alternating quiet consolidations and sharp releases -- the setup the
        # breakout team exists to trade.
        out = np.empty(n_bars)
        log_px = 0.0
        i = 0
        while i < n_bars:
            quiet_len = int(rng.integers(bars_per_session * 2, bars_per_session * 5))
            anchor = log_px
            for _ in range(min(quiet_len, n_bars - i)):
                log_px += -0.25 * (log_px - anchor) + rng.normal(0, per_bar_vol * 0.35)
                out[i] = log_px
                i += 1
                if i >= n_bars:
                    break
            if i >= n_bars:
                break
            # A release, not an explosion: the burst has to be large relative
            # to the coil but still leave the name's overall volatility in the
            # range its spec claims, or the squeeze screens (which look for
            # *low* ATR) would never select it in the first place.
            burst_len = int(rng.integers(max(bars_per_session // 4, 2), bars_per_session))
            direction = 1.0 if rng.random() < 0.58 else -1.0
            for _ in range(min(burst_len, n_bars - i)):
                log_px += direction * abs(rng.normal(per_bar_vol * 0.5, per_bar_vol * 0.35))
                out[i] = log_px
                i += 1
                if i >= n_bars:
                    break
        return np.exp(out)

    if regime == "lottery":
        # Fat tails: a mixture of a calm state and an occasional jump.
        shocks = rng.normal(per_bar_drift, per_bar_vol, n_bars)
        jumps = (rng.random(n_bars) < 0.006) * rng.normal(0, per_bar_vol * 14, n_bars)
        return np.exp(np.cumsum(shocks + jumps))

    # trend_up / trend_down / chop / liquid_grind: drifting random walk with a
    # slowly varying drift so trends persist then fade.
    drift_path = per_bar_drift * (
        1.0 + 0.6 * np.sin(np.linspace(0, rng.uniform(1.0, 3.0) * math.pi, n_bars))
    )
    return np.exp(np.cumsum(rng.normal(0, per_bar_vol, n_bars) + drift_path))


def generate_bars(
    symbols: Sequence[str],
    start: date,
    end: date,
    *,
    timeframe_seconds: int = 300,
    seed: int = 20260101,
    calendar: MarketCalendar | None = None,
    specs: Mapping[str, SyntheticSpec] | None = None,
    prices: Mapping[str, float] | None = None,
    warmup_sessions: int = 40,
) -> dict[str, list[Bar]]:
    """Generate intraday bars over the sessions in [start, end], plus warm-up.

    `warmup_sessions` extra sessions are produced *before* `start` so that on
    the first tick of a round every strategy already has its full history --
    otherwise the first two days would just be warm-up and the round would
    only really be five days long.
    """
    cal = calendar or MarketCalendar()
    specs = dict(specs or build_specs(symbols, seed=seed, prices=prices))

    first = start
    for _ in range(warmup_sessions):
        first = cal.previous_trading_day(first)
    sessions = cal.trading_days(first, end)
    if not sessions:
        return {s.upper(): [] for s in symbols}

    stamps: list[datetime] = []
    for d in sessions:
        times = cal.session_times(d)
        if not times:
            continue
        t, close = times
        while t < close:
            stamps.append(t)
            t += timedelta(seconds=timeframe_seconds)
    n = len(stamps)
    if n == 0:
        return {s.upper(): [] for s in symbols}
    bars_per_session = max(n // len(sessions), 1)

    # Pass 1: the symbols nothing else depends on.
    paths: dict[str, np.ndarray] = {}
    order = sorted(specs, key=lambda s: specs[s].cointegrated_with is not None)
    for sym in order:
        spec = specs[sym]
        rng = np.random.default_rng(_seed_for(sym, seed) + 1)
        anchor = spec.cointegrated_with
        if anchor and anchor in paths:
            # log(y) = beta * log(x) + stationary spread  => genuinely cointegrated
            spread = np.empty(n)
            x = 0.0
            sd = spec.daily_vol / math.sqrt(bars_per_session) * 2.0
            for i in range(n):
                x += -0.05 * x + rng.normal(0, sd)
                spread[i] = x
            paths[sym] = np.exp(spec.beta * np.log(paths[anchor]) + spread)
        else:
            paths[sym] = _intraday_path(rng, spec, n, bars_per_session)

    out: dict[str, list[Bar]] = {}
    for sym, path in paths.items():
        spec = specs[sym]
        rng = np.random.default_rng(_seed_for(sym, seed) + 2)
        closes = spec.start_price * path / path[0]
        rows: list[Bar] = []
        prev_close = float(closes[0])
        per_bar_vol = spec.daily_vol / math.sqrt(bars_per_session)
        avg_bar_volume = spec.dollar_volume / max(bars_per_session, 1)
        for i, ts in enumerate(stamps):
            close = float(closes[i])
            # Overnight gap on the first bar of each session, so the gap-based
            # screens (the lottery picker) have something to measure.
            is_session_open = (
                i == 0 or stamps[i - 1].astimezone(UTC).date() != ts.astimezone(UTC).date()
            )
            if is_session_open and i > 0:
                gap = float(rng.normal(0, spec.daily_vol * 0.45))
                open_px = prev_close * (1.0 + gap)
            else:
                open_px = prev_close
            span = abs(close - open_px) + max(open_px, 1e-6) * abs(
                rng.normal(per_bar_vol * 1.4, per_bar_vol * 0.4)
            )
            high = max(open_px, close) + span * float(rng.uniform(0.2, 0.7))
            low = max(min(open_px, close) - span * float(rng.uniform(0.2, 0.7)), 0.01)
            vol_mult = float(
                np.exp(rng.normal(0, 0.45))
                * (2.4 if is_session_open else 1.0)
                * (1.0 + 2.2 * min(abs(close / max(open_px, 1e-9) - 1.0) / max(per_bar_vol, 1e-9), 3.0) / 3.0)
            )
            shares = max(avg_bar_volume * vol_mult / max(close, 0.01), 1.0)
            rows.append(Bar(
                symbol=sym, ts=ts, open=round(open_px, 4), high=round(high, 4),
                low=round(low, 4), close=round(close, 4), volume=round(shares, 2),
                trade_count=max(int(spec.trade_count / max(bars_per_session, 1) * vol_mult), 1),
                vwap=round((high + low + close) / 3.0, 4),
            ))
            prev_close = close
        out[sym] = rows
    return out


def to_daily(bars: Mapping[str, Sequence[Bar]]) -> dict[str, list[Bar]]:
    """Aggregate intraday bars into daily bars, as the pickers expect."""
    out: dict[str, list[Bar]] = {}
    for sym, rows in bars.items():
        by_day: dict[date, list[Bar]] = {}
        for b in rows:
            by_day.setdefault(b.ts.astimezone(UTC).date(), []).append(b)
        daily: list[Bar] = []
        for d in sorted(by_day):
            group = by_day[d]
            volume = sum(x.volume for x in group)
            notional = sum(x.volume * (x.vwap or x.close) for x in group)
            daily.append(Bar(
                symbol=sym,
                ts=datetime.combine(d, datetime.min.time(), tzinfo=UTC),
                open=group[0].open,
                high=max(x.high for x in group),
                low=min(x.low for x in group),
                close=group[-1].close,
                volume=volume,
                trade_count=sum(x.trade_count for x in group),
                vwap=round(notional / volume, 4) if volume > 0 else group[-1].close,
            ))
        out[sym] = daily
    return out


# --------------------------------------------------------------------------- #
# news
# --------------------------------------------------------------------------- #

_POSITIVE = [
    "{s} beats estimates and raises guidance for the quarter",
    "{s} tops expectations on record revenue",
    "Analysts raise price target on {s} after a strong quarter",
    "{s} announces a share buyback program",
    "{s} wins a major new contract",
    "{s} upgraded to buy on accelerating growth",
]
_NEGATIVE = [
    "{s} misses estimates and cuts guidance",
    "{s} falls short of expectations as demand weakens",
    "Analysts cut price target on {s}",
    "{s} downgraded to sell on margin concerns",
    "SEC probe widens at {s}",
    "{s} warns on the quarter amid supply chain disruption",
]
_NEUTRAL = [
    "{s} reiterates guidance; results in line with expectations",
    "{s} to present at an industry conference",
    "{s} names a new chief operating officer",
    "{s} schedules its quarterly earnings call",
]

_SOURCES = ["Benzinga", "Reuters", "Bloomberg", "CNBC", "Business Wire", "Seeking Alpha"]


def generate_news(
    bars: Mapping[str, Sequence[Bar]],
    *,
    seed: int = 20260101,
    articles_per_session: float = 1.8,
    lead_hours: float = 2.0,
) -> dict[str, list[NewsItem]]:
    """Synthesise a news wire that *leads* price, so the signal is learnable.

    Stories are generated from each session's realised return and stamped
    `lead_hours` before the move they describe. This is deliberately generous:
    a dry run should show the news team trading, so the plumbing can be
    verified. It is not evidence the strategy works on real news.
    """
    out: dict[str, list[NewsItem]] = {}
    counter = 0
    for sym, rows in bars.items():
        rng = np.random.default_rng(_seed_for(sym, seed) + 7)
        by_day: dict[date, list[Bar]] = {}
        for b in rows:
            by_day.setdefault(b.ts.astimezone(UTC).date(), []).append(b)
        items: list[NewsItem] = []
        for d in sorted(by_day):
            group = by_day[d]
            if len(group) < 2 or group[0].open <= 0:
                continue
            session_return = group[-1].close / group[0].open - 1.0
            n_articles = int(rng.poisson(articles_per_session))
            for _ in range(n_articles):
                if session_return > 0.004:
                    template = _POSITIVE[int(rng.integers(len(_POSITIVE)))]
                elif session_return < -0.004:
                    template = _NEGATIVE[int(rng.integers(len(_NEGATIVE)))]
                else:
                    template = _NEUTRAL[int(rng.integers(len(_NEUTRAL)))]
                counter += 1
                ts = group[0].ts - timedelta(hours=float(rng.uniform(0.5, lead_hours)))
                items.append(NewsItem(
                    id=f"syn-{counter:07d}",
                    ts=ts,
                    headline=template.format(s=sym),
                    summary="",
                    source=_SOURCES[int(rng.integers(len(_SOURCES)))],
                    symbols=(sym,),
                    url="",
                ))
        items.sort(key=lambda n: n.ts)
        out[sym] = items
    return out


def generate_daily_history(
    symbols: Sequence[str],
    end: date,
    *,
    sessions: int = 260,
    seed: int = 20260101,
    calendar: MarketCalendar | None = None,
    specs: Mapping[str, SyntheticSpec] | None = None,
    anchor_prices: Mapping[str, float] | None = None,
) -> dict[str, list[Bar]]:
    """Daily bars for the `sessions` trading days ending *before* `end`.

    The pickers screen on daily history -- 63 days for momentum, 120 for
    reversion, 180 for pair formation. Producing that much depth as intraday
    bars would mean ~1.4M bars for a 60-name pool, which is slow and pointless
    because nothing reads the intraday detail that far back. So the dry run
    mirrors what live mode actually does: a long *daily* series for the
    pickers, and a short intraday series for the strategies.

    `anchor_prices` rescales each path so its final close matches the price
    the intraday series starts at, which splices the two series into one
    continuous history rather than two disconnected ones.
    """
    cal = calendar or MarketCalendar()
    specs = dict(specs or build_specs(symbols, seed=seed, prices=anchor_prices))

    days: list[date] = []
    cursor = cal.previous_trading_day(end)
    for _ in range(sessions):
        days.append(cursor)
        cursor = cal.previous_trading_day(cursor)
    days.reverse()
    n = len(days)
    if n == 0:
        return {s.upper(): [] for s in symbols}

    paths: dict[str, np.ndarray] = {}
    order = sorted(specs, key=lambda s: specs[s].cointegrated_with is not None)
    for sym in order:
        spec = specs[sym]
        rng = np.random.default_rng(_seed_for(sym, seed) + 11)
        anchor = spec.cointegrated_with
        if anchor and anchor in paths:
            spread = np.empty(n)
            x = 0.0
            for i in range(n):
                x += -0.12 * x + rng.normal(0, spec.daily_vol * 1.6)
                spread[i] = x
            paths[sym] = np.exp(spec.beta * np.log(paths[anchor]) + spread)
        else:
            # Same regime shapes as the intraday generator, at daily scale.
            if spec.regime == "mean_revert":
                out = np.empty(n)
                lvl = 0.0
                x = 0.0
                for i in range(n):
                    lvl += rng.normal(0, spec.daily_vol * 0.12)
                    x += -0.18 * x + rng.normal(0, spec.daily_vol * 1.8)
                    out[i] = lvl + x
                paths[sym] = np.exp(out)
            elif spec.regime == "coil_break":
                out = np.empty(n)
                log_px = 0.0
                i = 0
                while i < n:
                    quiet = int(rng.integers(8, 22))
                    base = log_px
                    for _ in range(min(quiet, n - i)):
                        log_px += -0.3 * (log_px - base) + rng.normal(0, spec.daily_vol * 0.4)
                        out[i] = log_px
                        i += 1
                        if i >= n:
                            break
                    if i >= n:
                        break
                    burst = int(rng.integers(2, 7))
                    direction = 1.0 if rng.random() < 0.58 else -1.0
                    for _ in range(min(burst, n - i)):
                        log_px += direction * abs(
                            rng.normal(spec.daily_vol * 0.9, spec.daily_vol * 0.5)
                        )
                        out[i] = log_px
                        i += 1
                        if i >= n:
                            break
                paths[sym] = np.exp(out)
            elif spec.regime == "lottery":
                shocks = rng.normal(spec.daily_drift, spec.daily_vol, n)
                jumps = (rng.random(n) < 0.05) * rng.normal(0, spec.daily_vol * 4.0, n)
                paths[sym] = np.exp(np.cumsum(shocks + jumps))
            else:
                drift = spec.daily_drift * (
                    1.0 + 0.7 * np.sin(np.linspace(0, rng.uniform(1.0, 3.0) * math.pi, n))
                )
                paths[sym] = np.exp(np.cumsum(rng.normal(0, spec.daily_vol, n) + drift))

    out: dict[str, list[Bar]] = {}
    for sym, path in paths.items():
        spec = specs[sym]
        rng = np.random.default_rng(_seed_for(sym, seed) + 12)
        target = float((anchor_prices or {}).get(sym) or spec.start_price)
        closes = target * path / path[-1]      # splice: end at the anchor price
        rows: list[Bar] = []
        prev = float(closes[0])
        for i, d in enumerate(days):
            close = float(closes[i])
            open_px = prev * (1.0 + float(rng.normal(0, spec.daily_vol * 0.35)))
            span = abs(close - open_px) + max(open_px, 1e-6) * abs(
                rng.normal(spec.daily_vol * 0.75, spec.daily_vol * 0.25)
            )
            high = max(open_px, close) + span * float(rng.uniform(0.2, 0.6))
            low = max(min(open_px, close) - span * float(rng.uniform(0.2, 0.6)), 0.01)
            vol_mult = float(np.exp(rng.normal(0, 0.35)))
            shares = max(spec.dollar_volume * vol_mult / max(close, 0.01), 1.0)
            rows.append(Bar(
                symbol=sym,
                ts=datetime.combine(d, datetime.min.time(), tzinfo=UTC),
                open=round(open_px, 4), high=round(high, 4), low=round(low, 4),
                close=round(close, 4), volume=round(shares, 2),
                trade_count=max(int(spec.trade_count * vol_mult), 1),
                vwap=round((high + low + close) / 3.0, 4),
            ))
            prev = close
        out[sym] = rows
    return out
