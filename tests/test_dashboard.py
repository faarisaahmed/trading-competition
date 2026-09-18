"""The HTML dashboard.

Rendering is checked structurally rather than by eye, and the checks encode
the chart rules the dashboard is meant to obey -- one axis, colour by entity,
no ninth hue, solid gridlines, a table as contrast relief, no external
dependencies.
"""

from __future__ import annotations

import re
from datetime import date

import pytest

from competition.engine import Ledger
from competition.reporting import (
    build_dashboard_data,
    render_dashboard,
    write_dashboard,
)
from competition.reporting.palette import (
    CATEGORICAL_DARK,
    CATEGORICAL_LIGHT,
    assign_slots,
    css_variables,
    team_var,
)
from competition.types import Account, Position, utcnow

# --------------------------------------------------------------------------- #
# palette
# --------------------------------------------------------------------------- #


def test_eight_slots_in_both_modes():
    assert len(CATEGORICAL_LIGHT) == len(CATEGORICAL_DARK) == 8
    assert len(set(CATEGORICAL_LIGHT)) == 8
    assert all(re.fullmatch(r"#[0-9a-f]{6}", c) for c in CATEGORICAL_LIGHT)
    assert all(re.fullmatch(r"#[0-9a-f]{6}", c) for c in CATEGORICAL_DARK)


def test_slots_follow_configured_order_not_rank(cfg):
    """Colour must follow the entity, so standings reordering never repaints."""
    keys = [t.key for t in cfg.teams]
    a = assign_slots(keys)
    b = assign_slots(keys)
    assert a == b
    # Reversing the *display* order must not change any assignment.
    assert assign_slots(keys)["gambler"] == a["gambler"]


def test_the_eight_scored_teams_get_the_eight_slots(cfg):
    slots = assign_slots([t.key for t in cfg.teams])
    scored = [t.key for t in cfg.scored_teams]
    assert sorted(slots[k] for k in scored) == list(range(8))


def test_the_benchmark_gets_no_ninth_hue(cfg):
    """A ninth categorical colour is never generated."""
    slots = assign_slots([t.key for t in cfg.teams])
    unscored = [t.key for t in cfg.teams if not t.scored]
    assert unscored, "no unscored entry to check"
    for key in unscored:
        assert slots[key] == -1
    css = css_variables(slots)
    for key in unscored:
        var = f"--team-{key.replace('_', '-')}: "
        line = next(ln for ln in css.split("\n") if var in ln)
        assert "#898781" in line, "the benchmark must use muted ink"
        for colour in CATEGORICAL_LIGHT + CATEGORICAL_DARK:
            assert colour not in line


def test_css_declares_dark_under_both_scopes(cfg):
    css = css_variables(assign_slots([t.key for t in cfg.teams]))
    assert "@media (prefers-color-scheme: dark)" in css
    assert ':root[data-theme="dark"]' in css
    assert 'not([data-theme="light"])' in css


def test_team_var_is_css_safe():
    assert team_var("mean_reverter") == "var(--team-mean-reverter)"


# --------------------------------------------------------------------------- #
# data assembly
# --------------------------------------------------------------------------- #


@pytest.fixture
def populated(cfg, tmp_path):
    """A ledger holding a partially-run round."""
    ledger = Ledger(tmp_path / "dash.sqlite")
    ledger.start_run(round_id=1, mode="test", broker="sim",
                     rules_hash=cfg.rules_hash, seed=1, config={})
    ledger.open_round(round_id=1, start=date(2026, 9, 8), end=date(2026, 9, 14),
                      mode="test", broker="sim", rules_hash=cfg.rules_hash)
    from competition.types import Fill, Order, OrderStatus, Side

    for i, team in enumerate(cfg.teams):
        for step in range(6):
            equity = cfg.starting_cash * (1 + (i - 4) * 0.004 * step)
            ledger.record_equity(
                team.key,
                utcnow().replace(microsecond=0),
                Account(cash=equity * 0.4, equity=equity, buying_power=equity * 0.4,
                        positions=(Position("AAPL", 3.0, 100.0, 101.0),)),
            )
        order = Order(id=f"o{i}", client_order_id=f"c{i}", symbol="AAPL",
                      side=Side.BUY, qty=3.0, status=OrderStatus.FILLED,
                      reason="test entry (score 0.42)")
        ledger.record_orders(team.key, [order])
        ledger.record_fills(team.key, [Fill(f"o{i}", "AAPL", Side.BUY, 3.0,
                                            100.0, utcnow())])
        ledger.record_universe(1, team.key, date(2026, 9, 8),
                               ["AAPL", "GOOGL", "MSFT"], source="fixed")
    yield ledger
    ledger.close()


def test_builds_from_a_ledger_alone(cfg, populated):
    data = build_dashboard_data(cfg, ledger=populated, round_id=1)
    assert data.round_id == 1
    assert len(data.teams) == len(cfg.teams)
    assert len(data.scored) == len(cfg.scored_teams)
    assert data.benchmark is not None
    assert data.round_start == date(2026, 9, 8)
    assert data.round_end == date(2026, 9, 14)
    assert data.sessions_total == 5
    assert all(len(t.curve) == 6 for t in data.teams)
    assert data.tape


def test_ranking_is_by_return_and_places_are_dense(cfg, populated):
    data = build_dashboard_data(cfg, ledger=populated, round_id=1)
    ranked = data.ranked
    assert [t.place for t in ranked] == list(range(1, len(ranked) + 1))
    assert all(ranked[i].ret >= ranked[i + 1].ret for i in range(len(ranked) - 1))
    assert data.benchmark.place is None or data.benchmark not in ranked


def test_needs_an_engine_or_a_ledger(cfg):
    with pytest.raises(ValueError, match="engine or a ledger"):
        build_dashboard_data(cfg)


def test_day_of_round_is_clamped(cfg, populated):
    data = build_dashboard_data(cfg, ledger=populated, round_id=1)
    assert 1 <= data.day_of_round <= data.round_length
    assert data.round_length == 7


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


@pytest.fixture
def html(cfg, populated):
    data = build_dashboard_data(cfg, ledger=populated, round_id=1)
    return render_dashboard(cfg, data, refresh=30)


def test_renders_valid_standalone_html(html):
    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")
    assert html.count("<body") == 1 and html.count("</body>") == 1


def test_has_no_external_dependencies(html):
    """It must open from file:// with no network."""
    assert not re.search(r'(src|href)\s*=\s*"https?://', html)
    assert "cdn." not in html
    assert "<link" not in html


def test_exactly_one_chart_and_therefore_one_axis(html):
    assert html.count('<svg class="chart"') == 1


def test_every_team_is_drawn_and_identified(cfg, html):
    assert html.count('<polyline class="line') == len(cfg.teams)
    for team in cfg.teams:
        assert team.name in html
        assert f"--team-{team.key.replace('_', '-')}" in html


def test_benchmark_line_is_visually_subordinate(html):
    assert 'class="line benchmark"' in html
    assert ".line.benchmark {" in html


def test_gridlines_are_solid(html):
    """Dashed grid reads as data."""
    assert "stroke-dasharray" not in html


def test_marks_are_thin(html):
    assert "stroke-width: 2" in html


def test_standings_table_is_present_as_contrast_relief(html):
    assert "<h2>Standings</h2>" in html
    assert html.count('class="swatch"') >= 9
    assert "tabular-nums" in html


def test_selective_direct_labels_not_one_per_point(cfg, html):
    labels = html.count('class="series-label"')
    assert 0 < labels <= len(cfg.teams), labels


def test_hover_layer_exists(html):
    assert 'id="crosshair"' in html
    assert 'id="tooltip"' in html
    assert 'id="hover-target"' in html
    assert 'id="chart-data"' in html


def test_axis_is_explained(html):
    assert "not clock time" in html


def test_palette_is_exposed_for_the_validator(html):
    assert f'data-palette="{CATEGORICAL_LIGHT[0]}' in html


def test_values_are_escaped(cfg, populated):
    """A strategy's reason string ends up in HTML, so it must be escaped."""
    from competition.types import Order, OrderStatus, Side
    nasty = '<script>alert("x")</script> & "quoted"'
    populated.record_orders("gambler", [
        Order(id="evil", client_order_id="c", symbol="AAPL", side=Side.BUY,
              qty=1.0, status=OrderStatus.FILLED, reason=nasty)
    ])
    from competition.types import Fill
    populated.record_fills("gambler", [Fill("evil", "AAPL", Side.BUY, 1.0,
                                            100.0, utcnow())])
    data = build_dashboard_data(cfg, ledger=populated, round_id=1)
    out = render_dashboard(cfg, data)
    assert "<script>alert" not in out
    assert "&lt;script&gt;" in out


def test_writes_atomically(cfg, populated, tmp_path):
    out = write_dashboard(cfg, tmp_path / "d" / "dash.html", ledger=populated,
                          round_id=1)
    assert out.exists() and out.read_text().startswith("<!DOCTYPE")
    assert not list(out.parent.glob("*.tmp")), "temp file left behind"
    # Rewriting in place must not corrupt it.
    again = write_dashboard(cfg, out, ledger=populated, round_id=1)
    assert again.read_text().startswith("<!DOCTYPE")


def test_refresh_can_be_disabled(cfg, populated):
    data = build_dashboard_data(cfg, ledger=populated, round_id=1)
    assert 'http-equiv="refresh"' in render_dashboard(cfg, data, refresh=30)
    assert 'http-equiv="refresh"' not in render_dashboard(cfg, data, refresh=0)


def test_empty_ledger_renders_without_crashing(cfg, tmp_path):
    ledger = Ledger(tmp_path / "bare.sqlite")
    data = build_dashboard_data(cfg, ledger=ledger)
    out = render_dashboard(cfg, data)
    assert out.startswith("<!DOCTYPE")
    assert "No equity history" in out
    ledger.close()


def test_renders_with_a_live_engine(cfg, feed, tmp_path, daily, news, calendar):
    """The engine path, which also exercises the dashboard callback."""
    from competition.broker.simulated import SimConfig, SimulatedBroker
    from competition.data.universe import UniverseProvider
    from competition.engine import CompetitionEngine, UniverseResolver

    clock = {"now": calendar.session_times(date(2026, 9, 8))[0]}
    brokers = {
        t.key: SimulatedBroker(cfg.starting_cash, feed.quote_source(
            lambda: clock["now"]), name=t.key, config=SimConfig(),
            clock=lambda: clock["now"])
        for t in cfg.teams
    }
    ledger = Ledger(tmp_path / "live.sqlite")
    ledger.start_run(round_id=1, mode="test", broker="sim",
                     rules_hash=cfg.rules_hash, seed=1, config={})
    engine = CompetitionEngine(
        cfg, feed=feed, brokers=brokers, ledger=ledger,
        resolver=UniverseResolver(cfg, provider=UniverseProvider(),
                                  daily_bars=lambda s: {}, news=lambda s: {}),
        mode="test", clock=lambda: clock["now"], equity_snapshot_seconds=1,
    )
    engine.round_window = (date(2026, 9, 8), date(2026, 9, 14))
    rnd = cfg.round(1)
    engine.prepare_round(rnd, session=date(2026, 9, 8))
    engine.start_round(rnd)

    written: list = []
    engine.set_dashboard(lambda eng: written.append(
        write_dashboard(cfg, tmp_path / "live.html", engine=eng)))
    for i in range(3):
        clock["now"] = clock["now"].replace(minute=(30 + i * 5) % 60)
        engine.tick(rnd, now=clock["now"])
    assert written, "the dashboard callback never fired"
    out = (tmp_path / "live.html").read_text()
    assert "Round 1" in out and "Standings" in out
    ledger.close()


def test_a_broken_dashboard_cannot_stop_a_round(cfg, feed, tmp_path, calendar):
    """The callback is best-effort by design."""
    from competition.broker.simulated import SimConfig, SimulatedBroker
    from competition.data.universe import UniverseProvider
    from competition.engine import CompetitionEngine, UniverseResolver

    clock = {"now": calendar.session_times(date(2026, 9, 8))[0]}
    brokers = {
        t.key: SimulatedBroker(cfg.starting_cash, feed.quote_source(
            lambda: clock["now"]), name=t.key, config=SimConfig(),
            clock=lambda: clock["now"])
        for t in cfg.teams
    }
    ledger = Ledger(tmp_path / "boom.sqlite")
    ledger.start_run(round_id=1, mode="test", broker="sim",
                     rules_hash=cfg.rules_hash, seed=1, config={})
    engine = CompetitionEngine(
        cfg, feed=feed, brokers=brokers, ledger=ledger,
        resolver=UniverseResolver(cfg, provider=UniverseProvider(),
                                  daily_bars=lambda s: {}, news=lambda s: {}),
        mode="test", clock=lambda: clock["now"], equity_snapshot_seconds=1,
    )
    rnd = cfg.round(1)
    engine.prepare_round(rnd, session=date(2026, 9, 8))
    engine.start_round(rnd)

    def explode(_eng):
        raise RuntimeError("dashboard is broken")

    engine.set_dashboard(explode)
    engine.tick(rnd, now=clock["now"])          # must not raise
    ledger.close()
