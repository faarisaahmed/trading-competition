"""The replay feed. One property matters above all: NO LOOKAHEAD.

A backtest that leaks the future is worse than no backtest, because it is
confidently wrong. These tests check that every series a strategy can reach
is strictly in the past, at every tick, for every symbol and timeframe.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from competition.data.feed import ReplayFeed, timeframe_seconds
from competition.types import UTC


def test_timeframe_parsing():
    assert timeframe_seconds("1Min") == 60
    assert timeframe_seconds("5Min") == 300
    assert timeframe_seconds("1Hour") == 3600
    assert timeframe_seconds("1Day") == 86400
    with pytest.raises(ValueError):
        timeframe_seconds("5minutes")


# --------------------------------------------------------------------------- #
# no lookahead
# --------------------------------------------------------------------------- #


def test_no_primary_bar_is_visible_before_it_completes(feed, symbols, calendar):
    """A bar stamped at t only exists once t + period has passed."""
    period = timedelta(seconds=300)
    stamps = calendar.tick_times(date(2026, 9, 8), date(2026, 9, 11), seconds=300)
    for ts in stamps[::7]:
        snap = feed.snapshot(symbols, ts)
        for sym in symbols:
            for bar in snap.series(sym, "primary"):
                assert bar.ts + period <= ts, (
                    f"{sym} bar at {bar.ts} visible at {ts} -- lookahead!"
                )


def test_no_daily_bar_is_visible_before_its_session_ends(feed, symbols, calendar):
    stamps = calendar.tick_times(date(2026, 9, 8), date(2026, 9, 11), seconds=1800)
    for ts in stamps:
        snap = feed.snapshot(symbols, ts)
        for sym in symbols:
            for bar in snap.series(sym, "daily"):
                assert bar.ts + timedelta(days=1) <= ts


def test_news_is_never_from_the_future(feed, symbols, calendar):
    stamps = calendar.tick_times(date(2026, 9, 8), date(2026, 9, 11), seconds=900)
    for ts in stamps:
        snap = feed.snapshot(symbols, ts)
        for sym in symbols:
            for item in snap.news_for(sym):
                assert item.ts <= ts


def test_quotes_are_stamped_now_and_derived_from_visible_bars(feed, symbols, midweek_ts):
    snap = feed.snapshot(symbols, midweek_ts)
    for sym in symbols:
        q = snap.quote(sym)
        assert q is not None and q.ts == midweek_ts
        assert q.is_sane and q.bid < q.ask
        last = snap.series(sym, "primary")[-1]
        assert q.mid == pytest.approx(last.close, rel=1e-4)


def test_history_grows_monotonically(feed, symbols, calendar):
    stamps = calendar.tick_times(date(2026, 9, 8), date(2026, 9, 9), seconds=300)
    prev = 0
    for ts in stamps:
        n = len(feed.snapshot(symbols[:1], ts).series(symbols[0], "primary"))
        assert n >= min(prev, 500)
        prev = n


def test_memoisation_returns_identical_windows(feed, symbols, midweek_ts):
    """The window cache must not change what a strategy sees."""
    a = feed.snapshot(symbols, midweek_ts)
    b = feed.snapshot(symbols, midweek_ts)
    for sym in symbols:
        assert a.series(sym, "primary") == b.series(sym, "primary")
        assert a.series(sym, "daily") == b.series(sym, "daily")
    # Advance within the same bar: the window must be unchanged.
    c = feed.snapshot(symbols, midweek_ts + timedelta(seconds=20))
    for sym in symbols:
        assert c.series(sym, "primary") == a.series(sym, "primary")
    # Advance past a bar boundary: it must grow.
    d = feed.snapshot(symbols, midweek_ts + timedelta(seconds=310))
    assert len(d.series(symbols[0], "primary")) >= len(a.series(symbols[0], "primary"))


def test_quote_source_tracks_the_shared_clock(feed, symbols, midweek_ts):
    now = {"t": midweek_ts}
    source = feed.quote_source(lambda: now["t"])
    first = source(symbols[0])
    assert first is not None and first.ts == midweek_ts
    now["t"] = midweek_ts + timedelta(hours=1)
    second = source(symbols[0])
    assert second.ts == now["t"]
    assert source("NOT_A_SYMBOL") is None


def test_requires_an_explicit_now(feed, symbols):
    with pytest.raises(ValueError, match="explicit"):
        feed.snapshot(symbols, None)


# --------------------------------------------------------------------------- #
# snapshot helpers
# --------------------------------------------------------------------------- #


def test_snapshot_accessors(snapshot, symbols):
    sym = symbols[0]
    assert snapshot.price(sym) > 0
    assert len(snapshot.closes(sym)) == len(snapshot.series(sym))
    assert len(snapshot.closes(sym, n=10)) == 10
    assert snapshot.last_bar(sym) is snapshot.series(sym)[-1]
    assert snapshot.has_history(sym, 50)
    assert not snapshot.has_history(sym, 10_000)
    assert sym in snapshot.ready(50)
    assert snapshot.tradable(sym)
    assert len(snapshot.highs(sym)) == len(snapshot.lows(sym))
    assert "universe=" in snapshot.describe()


def test_unknown_symbol_is_empty_not_an_error(snapshot):
    assert snapshot.series("NOPE") == ()
    assert snapshot.closes("NOPE") == []
    assert snapshot.price("NOPE") == 0.0
    assert snapshot.quote("NOPE") is None
    assert not snapshot.tradable("NOPE")
    assert snapshot.last_bar("NOPE") is None
    assert snapshot.news_for("NOPE") == ()


def test_restriction_is_a_hard_boundary(snapshot, symbols):
    view = snapshot.restricted_to(symbols[:2])
    assert view.universe == tuple(s.upper() for s in symbols[:2])
    for sym in symbols[2:]:
        assert view.series(sym) == ()
        assert view.quote(sym) is None
        assert not view.tradable(sym)
    assert view.ts == snapshot.ts and view.session is snapshot.session


def test_news_within_hours_filter(feed, symbols, midweek_ts):
    snap = feed.snapshot(symbols, midweek_ts)
    for sym in symbols:
        recent = snap.news_for(sym, within_hours=6)
        allitems = snap.news_for(sym)
        assert len(recent) <= len(allitems)
        for item in recent:
            assert (midweek_ts - item.ts).total_seconds() <= 6 * 3600 + 1


def test_span_and_symbols(feed, symbols):
    assert set(feed.symbols) == {s.upper() for s in symbols}
    span = feed.span()
    assert span is not None and span[0] < span[1]


def test_empty_feed_is_harmless(calendar):
    empty = ReplayFeed({}, calendar=calendar)
    snap = empty.snapshot(["AAPL"], datetime(2026, 9, 10, 17, 0, tzinfo=UTC))
    assert snap.series("AAPL") == ()
    assert "AAPL" in snap.missing
    assert not snap.tradable("AAPL")
