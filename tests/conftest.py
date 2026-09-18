"""Shared fixtures. Everything here is offline and seeded."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from competition.config import load_config  # noqa: E402
from competition.data.calendar import MarketCalendar  # noqa: E402
from competition.data.feed import ReplayFeed  # noqa: E402
from competition.data.synthetic import (  # noqa: E402
    build_specs,
    generate_bars,
    generate_daily_history,
    generate_news,
    to_daily,
)
from competition.data.universe import UniverseSnapshot  # noqa: E402
from competition.types import Account  # noqa: E402

#: A Monday-to-Sunday window with a full five sessions, no holidays.
WINDOW_START = date(2026, 9, 7)
WINDOW_END = date(2026, 9, 13)
SEED = 4242


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "config")


@pytest.fixture(scope="session")
def calendar():
    return MarketCalendar()


@pytest.fixture(scope="session")
def pool():
    return UniverseSnapshot.from_csv(ROOT / "data" / "top500.csv")


@pytest.fixture(scope="session")
def symbols():
    return ["AAPL", "GOOGL", "MSFT", "NVDA", "KO", "JPM", "XOM", "WMT", "PFE", "COST"]


@pytest.fixture(scope="session")
def specs(symbols):
    return build_specs(symbols, seed=SEED)


@pytest.fixture(scope="session")
def intraday(symbols, specs, calendar):
    return generate_bars(
        symbols, WINDOW_START, WINDOW_END, seed=SEED, calendar=calendar,
        specs=specs, warmup_sessions=30,
    )


@pytest.fixture(scope="session")
def daily(symbols, specs, calendar, intraday):
    window = to_daily(intraday)
    first = min(rows[0].ts.date() for rows in intraday.values() if rows)
    anchors = {s: rows[0].open for s, rows in window.items() if rows}
    history = generate_daily_history(
        symbols, first, sessions=220, seed=SEED, calendar=calendar,
        specs=specs, anchor_prices=anchors,
    )
    return {s: list(history.get(s, [])) + list(window.get(s, [])) for s in symbols}


@pytest.fixture(scope="session")
def news(intraday):
    return generate_news(intraday, seed=SEED)


@pytest.fixture(scope="session")
def feed(intraday, daily, news, calendar):
    return ReplayFeed(
        intraday, fast=intraday, daily=daily, news=news, calendar=calendar,
        primary_timeframe="5Min", fast_timeframe="5Min", history_bars=500,
    )


@pytest.fixture(scope="session")
def midweek_ts(calendar):
    """A timestamp mid-session on the Thursday of the window."""
    times = calendar.session_times(date(2026, 9, 10))
    assert times is not None
    return times[0] + timedelta(hours=3)


@pytest.fixture
def snapshot(feed, symbols, midweek_ts):
    return feed.snapshot(symbols, midweek_ts)


@pytest.fixture
def flat_account(cfg):
    return Account(cash=cfg.starting_cash, equity=cfg.starting_cash,
                   buying_power=cfg.starting_cash, positions=())
