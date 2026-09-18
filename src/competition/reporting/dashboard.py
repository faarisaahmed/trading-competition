"""A self-contained HTML dashboard, rewritten on every equity snapshot.

Design decisions worth stating, because each one is a chart rule this file is
deliberately obeying:

* **One y-axis.** Every team starts each round from the same bankroll, so all
  nine equity curves share a dollar scale. There is no second axis anywhere on
  the page -- two scales on one plot invent a correlation that is not in the
  data.
* **Colour follows the entity.** Slots are assigned from the configured team
  order once (see `palette.py`), so the standings table reordering never
  repaints a line. Eight scored teams, eight validated slots, and the
  benchmark in muted ink -- never a generated ninth hue.
* **x is snapshot index, not wall clock.** Equity is sampled during sessions
  only, so a time axis would draw long dead horizontals across every night and
  weekend, making a 4-session round look mostly flat. The axis is ordinal with
  session-boundary ticks, and the label says so.
* **Text wears text tokens.** Values, labels and legends stay in ink; a small
  colour swatch beside them carries identity. Three light-mode slots fall below
  3:1 contrast, so the standings table and the direct labels are the required
  relief.
* **No dependencies.** Inline CSS, hand-built SVG, one small inline script for
  the crosshair. It opens from `file://` with no network, which matters when
  the thing you are monitoring is a week-long unattended run.
"""

from __future__ import annotations

import html
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ..config import CompetitionConfig
from ..types import utcnow
from ..util import indicators as ind
from .palette import (
    CATEGORICAL_LIGHT,
    assign_slots,
    css_variables,
    team_var,
)

log = logging.getLogger("competition.dashboard")

REFRESH_SECONDS = 30


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


@dataclass
class TeamRow:
    key: str
    name: str
    scored: bool
    baseline: float
    equity: float
    curve: list[float] = field(default_factory=list)
    stamps: list[str] = field(default_factory=list)
    positions: dict[str, float] = field(default_factory=dict)
    universe: list[str] = field(default_factory=list)
    fills: int = 0
    traded: float = 0.0
    rejections: dict[str, int] = field(default_factory=dict)
    errors: int = 0
    timeouts: int = 0
    halted: bool = False
    halt_reason: str = ""
    last_trade: str = ""
    points: float = 0.0
    place: int | None = None
    tagline: str = ""
    philosophy: str = ""
    prev_place: int | None = None

    @property
    def ret(self) -> float:
        return (self.equity / self.baseline - 1.0) if self.baseline > 0 else 0.0

    @property
    def pnl(self) -> float:
        return self.equity - self.baseline

    @property
    def max_drawdown(self) -> float:
        return ind.max_drawdown(self.curve) if len(self.curve) > 1 else 0.0

    @property
    def n_positions(self) -> int:
        return sum(1 for q in self.positions.values() if abs(q) > 1e-9)


@dataclass
class DashboardData:
    competition: str
    rules_hash: str
    generated_at: datetime
    round_id: int | None = None
    round_name: str = ""
    round_start: date | None = None
    round_end: date | None = None
    sessions_total: int = 0
    sessions_done: int = 0
    session_open: bool = False
    session_date: date | None = None
    minutes_to_close: float = 0.0
    next_open: datetime | None = None
    ticks: int = 0
    last_tick: datetime | None = None
    mode: str = ""
    broker: str = ""
    resumed: bool = False
    teams: list[TeamRow] = field(default_factory=list)
    tape: list[dict[str, Any]] = field(default_factory=list)
    standings: list[dict[str, Any]] = field(default_factory=list)
    rounds_scored: list[int] = field(default_factory=list)
    #: Season clock, set when a schedule is in play.
    season_phase: str = ""
    deadline_label: str = ""
    deadline_at: datetime | None = None
    season_plan: list[dict[str, Any]] = field(default_factory=list)

    @property
    def scored(self) -> list[TeamRow]:
        return [t for t in self.teams if t.scored]

    @property
    def benchmark(self) -> TeamRow | None:
        for t in self.teams:
            if not t.scored:
                return t
        return None

    @property
    def ranked(self) -> list[TeamRow]:
        rows = sorted(self.scored, key=lambda t: -t.ret)
        for i, row in enumerate(rows, 1):
            row.place = i
        return rows

    @property
    def day_of_round(self) -> int:
        if not self.round_start:
            return 0
        ref = (self.session_date or self.generated_at.date())
        return max(min((ref - self.round_start).days + 1, self.round_length), 1)

    @property
    def round_length(self) -> int:
        if self.round_start and self.round_end:
            return (self.round_end - self.round_start).days + 1
        return 7

    @property
    def round_complete(self) -> bool:
        return bool(self.round_end and self.generated_at.date() > self.round_end)

    @property
    def has_trading(self) -> bool:
        """Has anything actually happened? Distinguishes empty from flat."""
        return bool(self.ticks or self.tape
                    or any(t.fills or t.curve for t in self.teams))

    @property
    def is_live(self) -> bool:
        """True only for a real round against real paper accounts.

        A dashboard you leave open for a week must never be ambiguous about
        whether the money is real. Anything that is not an Alpaca-backed live
        run is simulated, and says so in a banner.
        """
        return self.mode == "live" and self.broker.startswith("alpaca")

    @property
    def data_source(self) -> str:
        if self.is_live:
            return "live Alpaca paper accounts"
        if "synthetic" in self.broker:
            return "simulated fills on synthetic prices"
        if self.broker.startswith("sim"):
            return "simulated fills on historical prices"
        return f"{self.mode or 'unknown'} / {self.broker or 'unknown'}"


def build_dashboard_data(
    cfg: CompetitionConfig,
    *,
    engine=None,
    ledger=None,
    round_id: int | None = None,
    tape_limit: int = 40,
) -> DashboardData:
    """Assemble dashboard state from a live engine, a ledger, or both.

    With an engine it reflects the current tick. With only a ledger it renders
    whatever was last recorded -- so `comp dashboard` works after the fact, and
    after a crash.
    """
    ledger = ledger if ledger is not None else (engine.ledger if engine else None)
    if ledger is None:
        raise ValueError("a dashboard needs an engine or a ledger")

    rnd = engine._round if engine is not None else None
    rid = round_id if round_id is not None else (rnd.id if rnd else None)
    data = DashboardData(
        competition=cfg.name,
        rules_hash=cfg.rules_hash,
        generated_at=utcnow(),
        round_id=rid,
        round_name=(rnd.name if rnd else (cfg.round(rid).name if rid else "")),
        mode=engine.mode if engine is not None else "",
        resumed=bool(getattr(engine, "resuming", False)),
        ticks=int(getattr(engine, "_ticks", 0) or 0),
    )

    # Where did this data actually come from? Taken from the run record and,
    # when an engine is present, corroborated against the broker classes
    # actually in use -- a run cannot label itself live if it is holding
    # simulators.
    runs = ledger.runs()
    if runs:
        data.broker = str(runs[-1]["broker"] or "")
        if not data.mode:
            data.mode = str(runs[-1]["mode"] or "")
    if engine is not None and engine.teams:
        classes = {type(rt.broker).__name__ for rt in engine.teams.values()}
        if classes and not any(c == "AlpacaBroker" for c in classes):
            # Contradict a bogus "alpaca" label, but keep any recorded detail
            # (e.g. "sim/synthetic") -- overwriting it wholesale loses the very
            # information the banner needs to describe the source.
            if data.broker.startswith("alpaca") or not data.broker:
                data.broker = "sim"
            if data.mode == "live":
                data.mode = "simulated"

    # -- round window and the clock ---------------------------------------- #
    window = getattr(engine, "round_window", None) if engine is not None else None
    if window is None and rid is not None:
        row = ledger.round_progress(rid)
        if row is not None:
            try:
                window = (date.fromisoformat(row["start_date"][:10]),
                          date.fromisoformat(row["end_date"][:10]))
                data.ticks = data.ticks or int(row["ticks"] or 0)
            except (ValueError, TypeError, KeyError):
                window = None
    if window:
        data.round_start, data.round_end = window
        # The session clock works with or without an engine: `comp dashboard`
        # run against a ledger still shows where the round stands, it just
        # reads the wall clock instead of the engine's cursor.
        if engine is not None:
            cal = engine.feed.calendar
            now = engine.clock()
        else:
            from ..data.calendar import MarketCalendar

            cal = MarketCalendar()
            now = utcnow()
        sessions = cal.trading_days(*window)
        data.sessions_total = len(sessions)
        session = cal.session(now)
        data.session_open = session.is_open
        data.session_date = session.session_date
        data.minutes_to_close = session.minutes_to_close
        data.next_open = session.next_open
        data.sessions_done = sum(1 for d in sessions if d < session.session_date)

    # -- per-team ----------------------------------------------------------- #
    for team in cfg.teams:
        curve_rows = ledger.equity_curve(team.key, None)
        stamps = [r[0] for r in curve_rows]
        curve = [r[1] for r in curve_rows]
        rt = engine.teams.get(team.key) if engine is not None else None

        baseline = cfg.starting_cash
        equity = curve[-1] if curve else baseline
        positions: dict[str, float] = {}
        universe: list[str] = []
        row = TeamRow(key=team.key, name=team.name, scored=team.scored,
                      tagline=team.tagline, philosophy=team.philosophy,
                      baseline=baseline, equity=equity, curve=curve, stamps=stamps)
        if rt is not None:
            row.baseline = float(rt.baseline_equity or cfg.starting_cash)
            try:
                account = rt.broker.account()
                row.equity = account.equity
                positions = {p.symbol: round(p.qty, 4) for p in account.positions
                             if abs(p.qty) > 1e-9}
            except Exception:  # noqa: BLE001 -- a report must never raise
                pass
            universe = list(rt.universe)
            row.errors = len(rt.errors)
            row.timeouts = rt.timeouts
            row.halted = rt.risk.is_halted
            row.halt_reason = rt.risk.halt_reason
            row.rejections = dict(rt.risk.rejections)
        else:
            universe = ledger.universe_for(rid, team.key) if rid else []
            row.rejections = ledger.rejection_tally(team.key)
        row.positions = positions
        row.universe = universe
        row.fills = ledger.fill_count(team.key)
        row.traded = ledger.traded_notional(team.key)
        recent = ledger.trade_log(team.key, limit=1)
        if recent:
            r = recent[0]
            row.last_trade = (
                f"{r['side'].upper()} {r['qty']:.4g} {r['symbol']} "
                f"@ {r['price']:.2f}"
                + (f" -- {r['reason']}" if r["reason"] else "")
            )
        data.teams.append(row)

    # -- the tape ----------------------------------------------------------- #
    fills: list[dict[str, Any]] = []
    for team in cfg.teams:
        for r in ledger.trade_log(team.key, limit=tape_limit):
            fills.append({
                "ts": r["ts"], "team": team.key, "name": team.name,
                "symbol": r["symbol"], "side": r["side"], "qty": float(r["qty"]),
                "price": float(r["price"]), "reason": r["reason"] or "",
            })
    fills.sort(key=lambda f: f["ts"], reverse=True)
    data.tape = fills[:tape_limit]

    # -- season standings, if any round has been scored --------------------- #
    try:
        from ..cli import _scores_from_ledger
        from ..scoring import build_leaderboard

        scores = _scores_from_ledger(cfg, ledger, [r.id for r in cfg.rounds])
        if scores:
            lb = build_leaderboard(cfg, scores)
            data.rounds_scored = list(lb.rounds)
            data.standings = [s.to_dict() for s in lb.standings]
            points = {s["team"]: s["points"] for s in data.standings}
            for row in data.teams:
                row.points = float(points.get(row.key, 0.0))
    except Exception as e:  # noqa: BLE001
        log.debug("standings unavailable for the dashboard: %s", e)

    return data


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _e(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _money(v: float) -> str:
    return f"${v:,.2f}"


def _pct(v: float) -> str:
    return f"{v:+.2f}%" if abs(v) >= 0.005 else "0.00%"


def _signed(v: float) -> str:
    return f"{v:+,.2f}"


def _delta_class(v: float) -> str:
    return "up" if v > 1e-9 else ("down" if v < -1e-9 else "flat")


def _countdown(data: DashboardData) -> str:
    if data.round_complete:
        return "round complete"
    if data.session_open and data.minutes_to_close < 1:
        return "at the close"
    if data.session_open:
        m = max(data.minutes_to_close, 0)
        return f"{int(m // 60)}h {int(m % 60):02d}m to the close"
    if data.next_open:
        delta = data.next_open - data.generated_at
        secs = max(delta.total_seconds(), 0)
        if secs > 86400:
            return f"opens in {int(secs // 86400)}d {int((secs % 86400) // 3600)}h"
        return f"opens in {int(secs // 3600)}h {int((secs % 3600) // 60):02d}m"
    return "market closed"


def _season_clock(data: DashboardData) -> str:
    """A countdown that ticks in the browser.

    The page only regenerates every ~30s, and a clock that jumps in 30-second
    steps reads as broken. The deadline is emitted as an ISO instant and
    counted down client-side, so it stays smooth between renders and is
    correct in the reader's own timezone.
    """
    if not data.deadline_at:
        if data.season_phase == "done":
            return ('<section class="clock done"><div class="clock-label">'
                    'season complete</div></section>')
        return ""
    iso = data.deadline_at.isoformat()
    phase = {
        "before": "Round 1 starts",
        "between": "next round starts",
        "pre_open": "round ends",
        "open": "round ends",
        "after_close": "round ends",
    }.get(data.season_phase, data.deadline_label or "next")
    live = "live" if data.season_phase == "open" else "idle"
    when = f"{data.deadline_at:%a %d %b %H:%M} UTC"
    return f"""<section class="clock {live}">
  <div class="clock-label">{_e(phase)}</div>
  <div class="clock-value num" data-deadline="{iso}">&mdash;</div>
  <div class="clock-when">{_e(when)}</div>
</section>"""


def _season_plan(data: DashboardData) -> str:
    """The three round windows, so a visitor can see what is coming."""
    if not data.season_plan:
        return ""
    rows = []
    for w in data.season_plan:
        state = str(w.get("state", ""))
        badge = {"done": "done", "live": "live", "upcoming": "upcoming"}.get(state, "")
        rows.append(
            f"<tr class=\"{badge}\">"
            f"<td class=\"num\">{_e(w.get('round_id'))}</td>"
            f"<td>{_e(w.get('name'))}</td>"
            f"<td>{_e(w.get('window'))}</td>"
            f"<td><span class=\"pill {badge}\">{_e(state)}</span></td>"
            f"</tr>"
        )
    body = "\n".join(rows)
    return f"""<section class="panel">
  <h2>Season schedule</h2>
  <table class="plan">
    <thead><tr><th>#</th><th>Round</th><th>Window</th><th>State</th></tr></thead>
    <tbody>
{body}
    </tbody>
  </table>
</section>"""


# --------------------------------------------------------------------------- #
# the equity chart
# --------------------------------------------------------------------------- #

CHART_W, CHART_H = 1080, 380
PAD_L, PAD_R, PAD_T, PAD_B = 68, 150, 18, 38


def _equity_chart(data: DashboardData) -> str:
    """A multi-series line chart. One axis, 2px lines, hairline grid."""
    series = [t for t in data.teams if len(t.curve) >= 2]
    if not series:
        return ('<p class="empty">No equity history recorded yet. '
                'The chart appears after the first snapshot.</p>')

    n = max(len(t.curve) for t in series)
    lo = min(min(t.curve) for t in series)
    hi = max(max(t.curve) for t in series)
    base = data.teams[0].baseline if data.teams else 5000.0
    lo, hi = min(lo, base), max(hi, base)
    span = (hi - lo) or 1.0
    lo -= span * 0.06
    hi += span * 0.06
    span = hi - lo

    def x_of(i: int) -> float:
        return PAD_L + (i / max(n - 1, 1)) * (CHART_W - PAD_L - PAD_R)

    def y_of(v: float) -> float:
        return PAD_T + (1 - (v - lo) / span) * (CHART_H - PAD_T - PAD_B)

    parts: list[str] = []

    # --- gridlines and y ticks (solid hairlines; dashed grid reads as data)
    ticks = 5
    for k in range(ticks + 1):
        v = lo + span * k / ticks
        y = y_of(v)
        parts.append(
            f'<line class="grid" x1="{PAD_L}" y1="{y:.1f}" '
            f'x2="{CHART_W - PAD_R}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="tick y" x="{PAD_L - 10}" y="{y + 4:.1f}">'
            f'${v:,.0f}</text>'
        )

    # --- the bankroll reference line: "zero return" is the thing to beat.
    #     Its label is placed later, in the same de-collision pass as the
    #     series labels, because it shares the right-hand gutter with them.
    yb = y_of(base)
    parts.append(
        f'<line class="baseline-ref" x1="{PAD_L}" y1="{yb:.1f}" '
        f'x2="{CHART_W - PAD_R}" y2="{yb:.1f}"/>'
    )

    # --- x ticks at session boundaries, since x is ordinal not clock time
    boundaries: list[tuple[int, str]] = []
    longest = max(series, key=lambda t: len(t.stamps))
    last_day = ""
    for i, stamp in enumerate(longest.stamps):
        day = stamp[:10]
        if day != last_day:
            boundaries.append((i, day))
            last_day = day
    for i, day in boundaries:
        x = x_of(i)
        if i > 0:
            parts.append(
                f'<line class="session-rule" x1="{x:.1f}" y1="{PAD_T}" '
                f'x2="{x:.1f}" y2="{CHART_H - PAD_B}"/>'
            )
        parts.append(
            f'<text class="tick x" x="{x:.1f}" y="{CHART_H - PAD_B + 20}">'
            f'{_e(day[5:])}</text>'
        )

    # --- axis rule
    parts.append(
        f'<line class="axis" x1="{PAD_L}" y1="{CHART_H - PAD_B}" '
        f'x2="{CHART_W - PAD_R}" y2="{CHART_H - PAD_B}"/>'
    )

    # --- series. Benchmark first so competitors draw over it.
    ordered = sorted(series, key=lambda t: (t.scored, t.ret))
    label_slots: list[tuple[float, str, str, bool]] = []
    end_points: dict[str, tuple[float, float, str]] = {}
    for t in ordered:
        pts = " ".join(f"{x_of(i):.1f},{y_of(v):.1f}" for i, v in enumerate(t.curve))
        cls = "line" + ("" if t.scored else " benchmark")
        stroke = team_var(t.key)
        parts.append(
            f'<polyline class="{cls}" points="{pts}" stroke="{stroke}"/>'
        )
        end_x, end_y = x_of(len(t.curve) - 1), y_of(t.curve[-1])
        parts.append(
            f'<circle class="end-dot" cx="{end_x:.1f}" cy="{end_y:.1f}" r="4.5" '
            f'fill="{stroke}"/>'
        )
        label_slots.append((end_y, t.name, t.key, t.scored))
        end_points[t.key] = (end_x, end_y, stroke)

    # --- direct labels, de-collided vertically. Selective: the podium, the
    #     tail and the benchmark -- never a label on every point. The bankroll
    #     reference label competes for the same gutter, so it is de-collided
    #     alongside them rather than drawn independently.
    ranked_keys = [t.key for t in data.ranked]
    keep = set(ranked_keys[:3]) | set(ranked_keys[-1:])
    bench = data.benchmark
    if bench:
        keep.add(bench.key)
    gutter: list[tuple[float, str, str, str]] = [
        (y, name, "series-label", key) for y, name, key, _scored in label_slots
        if key in keep
    ]
    gutter.append((yb, f"start ${base:,.0f}", "ref-label", ""))
    gutter.sort()

    label_x = CHART_W - PAD_R + 8
    placed: list[float] = []
    for y, text, cls, key in gutter:
        yy = y
        while any(abs(yy - q) < 13 for q in placed):
            yy += 13
        placed.append(yy)
        # Carry identity with a swatch beside the text, not with proximity to
        # an end dot. Nine curves regularly converge -- at this point in the
        # sample round the benchmark and Trend Rider end within a dollar of
        # each other -- so a label placed near "a" dot is routinely read as
        # belonging to the wrong team. The text itself stays in ink.
        anchor = end_points.get(key)
        if anchor is not None:
            parts.append(
                f'<rect class="label-swatch" x="{label_x:.1f}" '
                f'y="{yy - 1.5:.1f}" width="7" height="7" rx="2" '
                f'fill="{anchor[2]}"/>'
            )
            text_x = label_x + 12
        else:
            text_x = label_x
        parts.append(
            f'<text class="{cls}" x="{text_x:.1f}" y="{yy + 4:.1f}">'
            f'{_e(text)}</text>'
        )

    # --- hover layer: crosshair + tooltip, driven by the inline script
    parts.append(
        f'<line id="crosshair" class="crosshair" x1="0" y1="{PAD_T}" x2="0" '
        f'y2="{CHART_H - PAD_B}" style="display:none"/>'
    )
    parts.append(
        f'<rect id="hover-target" x="{PAD_L}" y="{PAD_T}" '
        f'width="{CHART_W - PAD_L - PAD_R}" height="{CHART_H - PAD_T - PAD_B}" '
        f'fill="transparent"/>'
    )

    payload = {
        "n": n,
        "padL": PAD_L, "padR": PAD_R, "w": CHART_W,
        "stamps": longest.stamps,
        "series": [
            {"key": t.key, "name": t.name, "curve": t.curve,
             "baseline": t.baseline, "scored": t.scored}
            for t in sorted(series, key=lambda s: (not s.scored, s.name))
        ],
    }

    return (
        f'<div class="chart-wrap">'
        f'<svg class="chart" viewBox="0 0 {CHART_W} {CHART_H}" '
        f'role="img" aria-label="Equity curve for every team this round" '
        f'preserveAspectRatio="xMidYMid meet">'
        + "".join(parts) +
        '</svg>'
        '<div id="tooltip" class="tooltip" hidden></div>'
        f'<script id="chart-data" type="application/json">'
        f'{json.dumps(payload)}</script>'
        f'</div>'
        f'<p class="axis-note">Horizontal axis is equity-snapshot order, not '
        f'clock time — samples are taken during sessions only, so a time '
        f'axis would draw a flat line across every night and weekend. Vertical '
        f'rules mark session boundaries.</p>'
    )


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #


def _banner(data: DashboardData) -> str:
    """A standing warning whenever the numbers are not real money.

    "Not started yet" and "simulated" are different claims and must not be
    conflated: before the opening bell there is no data at all, and calling
    that simulated would be its own kind of lie.
    """
    if data.is_live:
        return ""
    # Only the schedule can say the season has not begun. Absence of fills is
    # not evidence of that -- a simulated run with no trades is still
    # simulated, and must still carry the warning.
    if data.season_phase == "before" and not data.has_trading:
        return (
            '<section class="banner quiet" role="status">'
            '<strong>The competition has not started yet.</strong> '
            'No round has been run, so every team is shown at its opening '
            'bankroll. Trading begins at the bell shown below.'
            '</section>'
        )
    return (
        '<section class="banner" role="status">'
        '<strong>Simulated data \u2014 this is not a live competition.</strong> '
        f'Source: {_e(data.data_source)}. No orders were sent to a broker and '
        'no account balances changed. Returns here say nothing about how these '
        'strategies would perform on a real tape.'
        '</section>'
    )


def _hero(data: DashboardData) -> str:
    """Title, live badge and the countdown -- the thing you see first."""
    live = data.season_phase == "open"
    badge = ('<span class="badge live"><span class="dot"></span>LIVE</span>'
             if live else
             f'<span class="badge">{_e(_phase_word(data))}</span>')
    round_bit = ""
    if data.round_id:
        done, total = data.sessions_done, data.sessions_total or 5
        pct = min(100.0, 100.0 * done / total) if total else 0.0
        round_bit = (
            f'<div class="hero-round">'
            f'<div class="hero-round-name">Round {data.round_id} &middot; '
            f'{_e(data.round_name)}</div>'
            f'<div class="progress"><span style="width:{pct:.1f}%"></span></div>'
            f'<div class="hero-round-sub">day {max(done, 1)} of {total}</div>'
            f'</div>'
        )
    clock = ""
    if data.deadline_at:
        label = {"before": "Round 1 starts in", "between": "Next round starts in"}.get(
            data.season_phase, "Round ends in")
        clock = (
            f'<div class="hero-clock">'
            f'<div class="hero-clock-label">{_e(label)}</div>'
            f'<div class="hero-clock-value num" '
            f'data-deadline="{data.deadline_at.isoformat()}">&mdash;</div>'
            f'<div class="hero-clock-when">'
            f'{data.deadline_at:%a %d %b %H:%M} UTC</div></div>'
        )
    elif data.season_phase == "done":
        clock = ('<div class="hero-clock"><div class="hero-clock-value">'
                 'Season complete</div></div>')
    return f"""<section class="hero">
  <div class="hero-left">
    <div class="hero-title">{_e(data.competition)} {badge}</div>
    {round_bit}
  </div>
  {clock}
</section>"""


def _phase_word(data: DashboardData) -> str:
    return {
        "before": "not started",
        "between": "between rounds",
        "pre_open": "pre-market",
        "after_close": "closed",
        "done": "finished",
    }.get(data.season_phase, "standby")


_MEDALS = {1: ("gold", "1st"), 2: ("silver", "2nd"), 3: ("bronze", "3rd")}


def _podium(data: DashboardData) -> str:
    """Top three, sized by placing. The centrepiece of the leaderboard."""
    ranked = data.ranked
    if not data.has_trading or len(ranked) < 3:
        return ""
    first, second, third = ranked[0], ranked[1], ranked[2]
    # Visual order is 2nd, 1st, 3rd -- the winner in the middle and tallest.
    steps = []
    for row, height in ((second, "h2"), (first, "h1"), (third, "h3")):
        cls, label = _MEDALS[row.place]
        steps.append(
            f'<div class="step {height}">'
            f'<div class="step-team">'
            f'<span class="swatch" style="background:{team_var(row.key)}"></span>'
            f'<span class="step-name">{_e(row.name)}</span></div>'
            f'<div class="step-ret {_delta_class(row.ret)}">'
            f'{_pct(row.ret * 100)}</div>'
            f'<div class="step-eq">{_money(row.equity)}</div>'
            f'<div class="block {cls}"><span class="place">{label}</span></div>'
            f'</div>'
        )
    return ('<section class="podium">' + "".join(steps) + '</section>')


def _race(data: DashboardData) -> str:
    """Every team as a bar, longest return wins. Reads at a glance."""
    ranked = data.ranked
    bench = data.benchmark
    rows = list(ranked) + ([bench] if bench else [])
    if not rows:
        return ""
    span = max((abs(r.ret) for r in rows), default=0.0) or 0.01
    out = []
    for row in rows:
        frac = min(abs(row.ret) / span, 1.0) * 50.0
        side = "pos" if row.ret >= 0 else "neg"
        medal = ""
        if row.scored and row.place in _MEDALS and data.has_trading:
            medal = f'<span class="medal {_MEDALS[row.place][0]}"></span>'
        place = (str(row.place) if row.scored and data.has_trading else "&mdash;")
        out.append(
            f'<div class="race-row">'
            f'<div class="race-place num">{place}</div>'
            f'<div class="race-name">{medal}'
            f'<span class="swatch" style="background:{team_var(row.key)}"></span>'
            f'{_e(row.name)}'
            f'</div>'
            f'<div class="race-track">'
            f'<div class="race-bar {side}" style="width:{frac:.2f}%"></div>'
            f'</div>'
            f'<div class="race-ret num {_delta_class(row.ret)}">'
            f'{_pct(row.ret * 100)}</div>'
            f'<div class="race-eq num">{_money(row.equity)}</div>'
            f'</div>'
        )
    return ('<div class="race"><div class="race-axis"><span>behind</span>'
            '<span>ahead</span></div>' + "".join(out) + '</div>')


def _tab_bar(tabs: list[tuple[str, str]]) -> str:
    buttons = "".join(
        f'<button class="tab{" on" if i == 0 else ""}" data-tab="{tid}">'
        f'{_e(label)}</button>'
        for i, (tid, label) in enumerate(tabs)
    )
    return f'<nav class="tabs" role="tablist">{buttons}</nav>'


def _team_dossiers(data: DashboardData) -> str:
    """Who each competitor is, in its own words."""
    cards = []
    for row in (data.ranked + ([data.benchmark] if data.benchmark else [])):
        pos = ", ".join(f"{k} {v:g}" for k, v in sorted(row.positions.items())[:6])
        uni = ", ".join(row.universe[:8]) or "&mdash;"
        rank = (f'<span class="dossier-rank">#{row.place}</span>'
                if row.scored and row.place and data.has_trading else "")
        cards.append(f"""<article class="dossier">
  <header>
    <span class="swatch" style="background:{team_var(row.key)}"></span>
    <h3>{_e(row.name)}</h3>{rank}
    <span class="dossier-ret {_delta_class(row.ret)}">{_pct(row.ret * 100)}</span>
  </header>
  <p class="tagline">&ldquo;{_e(row.tagline)}&rdquo;</p>
  <p class="philosophy">{_e(row.philosophy)}</p>
  <dl class="dossier-stats">
    <div><dt>Equity</dt><dd class="num">{_money(row.equity)}</dd></div>
    <div><dt>P&amp;L</dt><dd class="num {_delta_class(row.pnl)}">
      {_signed(row.pnl)}</dd></div>
    <div><dt>Fills</dt><dd class="num">{row.fills}</dd></div>
    <div><dt>Traded</dt><dd class="num">{_money(row.traded)}</dd></div>
    <div><dt>Max DD</dt><dd class="num">{_pct(row.max_drawdown * 100)}</dd></div>
    <div><dt>Season pts</dt><dd class="num">{row.points:g}</dd></div>
  </dl>
  <div class="dossier-holdings">
    <span>Holding</span> {_e(pos) or "flat"}
  </div>
  <div class="dossier-holdings">
    <span>Universe</span> {uni}
  </div>
</article>""")
    return '<div class="dossiers">' + "".join(cards) + "</div>"


def _stat_tiles(data: DashboardData) -> str:
    ranked = data.ranked
    leader = ranked[0] if ranked else None
    bench = data.benchmark
    tiles: list[str] = []

    if data.round_id:
        tiles.append(
            f'<div class="tile"><div class="tile-label">Round {data.round_id}'
            f'</div><div class="tile-value">Day {data.day_of_round}'
            f'<span class="tile-unit">/ {data.round_length}</span></div>'
            f'<div class="tile-sub">{_e(data.round_name)}</div></div>'
        )
    status = "open" if data.session_open else "closed"
    tiles.append(
        f'<div class="tile"><div class="tile-label">Market</div>'
        f'<div class="tile-value {"live" if data.session_open else ""}">'
        f'{status}</div>'
        f'<div class="tile-sub">{_e(_countdown(data))}</div></div>'
    )
    if leader and data.has_trading:
        tiles.append(
            f'<div class="tile"><div class="tile-label">Leading</div>'
            f'<div class="tile-value">{_e(leader.name)}</div>'
            f'<div class="tile-sub {_delta_class(leader.ret)}">'
            f'{_pct(leader.ret * 100)} · {_money(leader.equity)}</div></div>'
        )
    if ranked:
        spread = ranked[0].ret - ranked[-1].ret
        tiles.append(
            f'<div class="tile"><div class="tile-label">Field spread</div>'
            f'<div class="tile-value">{spread * 100:.2f}<span class="tile-unit">'
            f'pp</span></div><div class="tile-sub">first to last</div></div>'
        )
    if bench:
        beat = sum(1 for t in ranked if t.ret > bench.ret)
        tiles.append(
            f'<div class="tile"><div class="tile-label">Beating the benchmark</div>'
            f'<div class="tile-value">{beat}<span class="tile-unit">'
            f'/ {len(ranked)}</span></div>'
            f'<div class="tile-sub {_delta_class(bench.ret)}">benchmark '
            f'{_pct(bench.ret * 100)}</div></div>'
        )

    errors = sum(t.errors for t in data.teams)
    timeouts = sum(t.timeouts for t in data.teams)
    halted = [t.name for t in data.teams if t.halted]
    health = "healthy"
    health_class = "good"
    if halted:
        health, health_class = f"{len(halted)} halted", "critical"
    elif errors or timeouts:
        health, health_class = f"{errors} err / {timeouts} t/o", "warning"
    tiles.append(
        f'<div class="tile"><div class="tile-label">Engine</div>'
        f'<div class="tile-value status-{health_class}">'
        f'<span class="dot"></span>{_e(health)}</div>'
        f'<div class="tile-sub">{data.ticks:,} ticks'
        + (f" · {_e(', '.join(halted))}" if halted else "")
        + '</div></div>'
    )
    return f'<section class="tiles">{"".join(tiles)}</section>'


def _standings_table(data: DashboardData) -> str:
    rows: list[str] = []
    for t in data.ranked + ([data.benchmark] if data.benchmark else []):
        if t is None:
            continue
        place = str(t.place) if t.scored and t.place else "—"
        flag = ""
        if t.halted:
            flag = (f'<span class="badge critical" title="{_e(t.halt_reason)}">'
                    f'halted</span>')
        elif t.timeouts:
            flag = f'<span class="badge warning">{t.timeouts} timeout(s)</span>'
        # Precomputed: a Python 3.10 f-string cannot hold a backslash in its
        # expression part, and these need embedded quotes.
        row_cls = "" if t.scored else ' class="unscored"'
        tag = "" if t.scored else " <em>(unscored)</em>"
        rows.append(
            f'<tr{row_cls}>'
            f'<td class="num">{place}</td>'
            f'<td><span class="swatch" style="background:{team_var(t.key)}"></span>'
            f'{_e(t.name)}{tag}{flag}</td>'
            f'<td class="num">{_money(t.equity)}</td>'
            f'<td class="num {_delta_class(t.ret)}">{_pct(t.ret * 100)}</td>'
            f'<td class="num {_delta_class(t.pnl)}">{_signed(t.pnl)}</td>'
            f'<td class="num">{t.n_positions}</td>'
            f'<td class="num">{t.fills:,}</td>'
            f'<td class="num">{t.max_drawdown * 100:.2f}%</td>'
            f'<td class="num">{t.points:g}</td>'
            f'</tr>'
        )
    return (
        '<section class="panel"><h2>Standings</h2>'
        '<table class="data"><thead><tr>'
        '<th class="num">#</th><th>Team</th><th class="num">Equity</th>'
        '<th class="num">Return</th><th class="num">P&amp;L</th>'
        '<th class="num">Pos</th><th class="num">Fills</th>'
        '<th class="num">Max DD</th><th class="num">Season pts</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table></section>'
    )


def _team_cards(data: DashboardData) -> str:
    cards: list[str] = []
    for t in data.ranked + ([data.benchmark] if data.benchmark else []):
        if t is None:
            continue
        positions = ", ".join(
            f"{sym} {qty:g}" for sym, qty in sorted(t.positions.items())
        ) or "flat"
        universe = ", ".join(t.universe) or "—"
        rejects = ", ".join(f"{k} {v}" for k, v in
                            sorted(t.rejections.items(), key=lambda kv: -kv[1])[:3])
        cards.append(
            f'<article class="card">'
            f'<header><span class="swatch" style="background:{team_var(t.key)}">'
            f'</span><h3>{_e(t.name)}</h3>'
            f'<span class="card-ret {_delta_class(t.ret)}">'
            f'{_pct(t.ret * 100)}</span></header>'
            f'<dl>'
            f'<dt>Equity</dt><dd>{_money(t.equity)}</dd>'
            f'<dt>Positions</dt><dd>{_e(positions)}</dd>'
            f'<dt>Universe</dt><dd class="dim">{_e(universe)}</dd>'
            f'<dt>Traded</dt><dd>{_money(t.traded)} over {t.fills:,} fills</dd>'
            + (f'<dt>Rejections</dt><dd class="dim">{_e(rejects)}</dd>'
               if rejects else '')
            + (f'<dt>Last</dt><dd class="dim">{_e(t.last_trade)}</dd>'
               if t.last_trade else '')
            + '</dl></article>'
        )
    return f'<section class="cards">{"".join(cards)}</section>'


def _tape(data: DashboardData) -> str:
    if not data.tape:
        return ""
    rows = []
    for f in data.tape:
        rows.append(
            f'<tr><td class="dim num">{_e(f["ts"][11:19])}</td>'
            f'<td><span class="swatch sm" style="background:'
            f'{team_var(f["team"])}"></span>{_e(f["name"])}</td>'
            f'<td class="side-{_e(f["side"])}">{_e(f["side"].upper())}</td>'
            f'<td class="num">{f["qty"]:.4g}</td>'
            f'<td>{_e(f["symbol"])}</td>'
            f'<td class="num">{f["price"]:.2f}</td>'
            f'<td class="dim reason">{_e(f["reason"][:110])}</td></tr>'
        )
    return (
        '<section class="panel"><h2>Trade tape</h2>'
        '<table class="data tape"><thead><tr><th>Time</th><th>Team</th>'
        '<th>Side</th><th class="num">Qty</th><th>Symbol</th>'
        '<th class="num">Price</th><th>Why</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></section>'
    )


def _season_table(data: DashboardData) -> str:
    if not data.standings or not data.rounds_scored:
        return ""
    head = "".join(f'<th class="num">R{r}</th>' for r in data.rounds_scored)
    rows = []
    for entry in data.standings:
        returns = entry.get("returns") or {}
        cells = []
        for r in data.rounds_scored:
            # JSON round-trips dict keys to strings, so accept either form.
            value = returns.get(r, returns.get(str(r)))
            if value is None:
                cells.append('<td class="num dim">&mdash;</td>')
            else:
                cells.append(
                    f'<td class="num {_delta_class(value)}">{_pct(value * 100)}</td>'
                )
        cell_html = "".join(cells)
        row_cls = "" if entry.get("scored") else ' class="unscored"'
        rank = entry.get("rank") or "&mdash;"
        swatch = team_var(entry["team"])
        rows.append(
            f'<tr{row_cls}>'
            f'<td class="num">{rank}</td>'
            f'<td><span class="swatch" style="background:{swatch}"></span>'
            f'{_e(entry["name"])}</td>'
            f'<td class="num strong">{entry["points"]:g}</td>{cell_html}'
            f'<td class="num {_delta_class(entry["total_return"])}">'
            f'{_pct(entry["total_return"] * 100)}</td>'
            f'<td class="num">{entry["wins"]}</td></tr>'
        )
    return (
        '<section class="panel"><h2>Season standings</h2>'
        '<table class="data"><thead><tr><th class="num">#</th><th>Team</th>'
        f'<th class="num">Points</th>{head}'
        '<th class="num">Compounded</th><th class="num">Wins</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table></section>'
    )


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #

_SCRIPT = """
(function () {
  // Tabs. Plain buttons toggling panels: no router, no history entries, and
  // the whole page still works if this script never runs (all panels would
  // simply be visible).
  var tabs = document.querySelectorAll('.tab');
  function show(id) {
    document.querySelectorAll('.panel-set').forEach(function (p) {
      p.classList.toggle('hidden', p.id !== 'tab-' + id);
    });
    tabs.forEach(function (t) {
      t.classList.toggle('on', t.getAttribute('data-tab') === id);
    });
    try { localStorage.setItem('comp-tab', id); } catch (e) { /* private mode */ }
  }
  tabs.forEach(function (t) {
    t.addEventListener('click', function () { show(t.getAttribute('data-tab')); });
  });
  // Survive the 30-second auto-refresh on whichever tab the visitor chose.
  try {
    var saved = localStorage.getItem('comp-tab');
    if (saved && document.getElementById('tab-' + saved)) show(saved);
  } catch (e) { /* ignore */ }
})();

(function () {
  // Grow the race bars from zero on load, so a refresh feels like movement
  // rather than a redraw.
  window.requestAnimationFrame(function () {
    document.querySelectorAll('.race-bar').forEach(function (b) {
      var w = b.style.width;
      b.style.width = '0%';
      window.requestAnimationFrame(function () { b.style.width = w; });
    });
  });
})();

(function () {
  // The season countdown. Ticks locally so it stays smooth between the
  // page's periodic regenerations.
  var el = document.querySelector('[data-deadline]');
  if (!el) return;
  var target = new Date(el.getAttribute('data-deadline')).getTime();
  function pad(n) { return (n < 10 ? '0' : '') + n; }
  function tick() {
    var left = Math.max(0, Math.floor((target - Date.now()) / 1000));
    var d = Math.floor(left / 86400);
    var h = Math.floor((left % 86400) / 3600);
    var m = Math.floor((left % 3600) / 60);
    var s = left % 60;
    el.textContent = d > 0
      ? d + 'd ' + pad(h) + ':' + pad(m) + ':' + pad(s)
      : pad(h) + ':' + pad(m) + ':' + pad(s);
    if (left === 0) { el.textContent = '00:00:00'; return; }
    setTimeout(tick, 1000);
  }
  tick();
})();

(function () {
  var node = document.getElementById('chart-data');
  if (!node) return;
  var data = JSON.parse(node.textContent);
  var svg = document.querySelector('.chart');
  var target = document.getElementById('hover-target');
  var cross = document.getElementById('crosshair');
  var tip = document.getElementById('tooltip');
  if (!svg || !target || !tip) return;

  var plotW = data.w - data.padL - data.padR;

  function xOf(i) { return data.padL + (i / Math.max(data.n - 1, 1)) * plotW; }

  function show(evt) {
    var box = svg.getBoundingClientRect();
    var scale = data.w / box.width;
    var svgX = (evt.clientX - box.left) * scale;
    var frac = (svgX - data.padL) / plotW;
    var i = Math.round(frac * Math.max(data.n - 1, 1));
    if (i < 0) i = 0;
    if (i > data.n - 1) i = data.n - 1;

    cross.setAttribute('x1', xOf(i));
    cross.setAttribute('x2', xOf(i));
    cross.style.display = '';

    var rows = data.series.map(function (s) {
      var v = s.curve[Math.min(i, s.curve.length - 1)];
      var ret = s.baseline ? (v / s.baseline - 1) * 100 : 0;
      return { name: s.name, key: s.key, v: v, ret: ret, scored: s.scored };
    }).sort(function (a, b) { return b.ret - a.ret; });

    var stamp = data.stamps[Math.min(i, data.stamps.length - 1)] || '';
    var html = '<div class="tip-head">' + stamp.replace('T', ' ').slice(0, 16) +
               ' UTC</div>';
    rows.forEach(function (r) {
      var cls = r.ret > 0 ? 'up' : (r.ret < 0 ? 'down' : 'flat');
      html += '<div class="tip-row"><span class="swatch sm" style="background:' +
              'var(--team-' + r.key.replace(/_/g, '-') + ')"></span>' +
              '<span class="tip-name">' + r.name + (r.scored ? '' : ' *') +
              '</span><span class="tip-val">$' +
              r.v.toLocaleString(undefined, { minimumFractionDigits: 2,
                                              maximumFractionDigits: 2 }) +
              '</span><span class="tip-ret ' + cls + '">' +
              (r.ret >= 0 ? '+' : '') + r.ret.toFixed(2) + '%</span></div>';
    });
    tip.innerHTML = html;
    tip.hidden = false;

    var wrapBox = svg.parentElement.getBoundingClientRect();
    var px = evt.clientX - wrapBox.left + 16;
    if (px + tip.offsetWidth > wrapBox.width) {
      px = evt.clientX - wrapBox.left - tip.offsetWidth - 16;
    }
    tip.style.left = Math.max(px, 4) + 'px';
    tip.style.top = '12px';
  }

  function hide() { cross.style.display = 'none'; tip.hidden = true; }

  target.addEventListener('mousemove', show);
  target.addEventListener('mouseleave', hide);
  target.addEventListener('touchmove', function (e) {
    if (e.touches.length) { show(e.touches[0]); e.preventDefault(); }
  }, { passive: false });
})();
"""


def _styles(slots: Mapping[str, int]) -> str:
    return f"""
  {css_variables(slots)}

  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 22px 26px 60px;
    background: var(--plane); color: var(--text-primary);
    font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  .num {{ font-variant-numeric: tabular-nums; text-align: right; }}
  h1 {{ font-size: 19px; margin: 0 0 2px; letter-spacing: -0.01em; }}
  h2 {{ font-size: 13px; margin: 0 0 12px; font-weight: 600;
       text-transform: uppercase; letter-spacing: 0.07em;
       color: var(--text-secondary); }}
  h3 {{ font-size: 14px; margin: 0; font-weight: 600; }}
  .sub {{ color: var(--text-secondary); font-size: 12.5px; }}
  .dim {{ color: var(--muted); }}
  .strong {{ font-weight: 650; }}
  header.top {{ display: flex; justify-content: space-between;
                align-items: flex-end; gap: 20px; margin-bottom: 18px;
                flex-wrap: wrap; }}
  .up {{ color: var(--up); }}
  .down {{ color: var(--down); }}
  .flat {{ color: var(--muted); }}

  .panel {{ background: var(--surface); border: 1px solid var(--border);
            border-radius: 10px; padding: 16px 18px; margin-bottom: 16px; }}

  .tiles {{ display: grid; gap: 12px; margin-bottom: 16px;
            grid-template-columns: repeat(auto-fit, minmax(168px, 1fr)); }}
  .tile {{ background: var(--surface); border: 1px solid var(--border);
           border-radius: 10px; padding: 13px 15px; }}
  .tile-label {{ font-size: 11px; text-transform: uppercase;
                 letter-spacing: 0.07em; color: var(--muted); }}
  .tile-value {{ font-size: 23px; font-weight: 600; margin-top: 3px;
                 letter-spacing: -0.02em; }}
  .tile-unit {{ font-size: 13px; font-weight: 400; color: var(--muted);
                margin-left: 4px; }}
  .tile-sub {{ font-size: 12px; color: var(--text-secondary); margin-top: 3px; }}
  .tile-value.live::after {{ content: ""; display: inline-block; width: 8px;
    height: 8px; border-radius: 50%; background: var(--good); margin-left: 8px;
    vertical-align: 3px; }}
  .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%;
          margin-right: 7px; vertical-align: 2px; background: currentColor; }}
  .status-good {{ color: var(--good); }}
  .status-warning {{ color: var(--warning); }}
  .status-critical {{ color: var(--critical); }}

  .chart-wrap {{ position: relative; }}
  .chart {{ width: 100%; height: auto; display: block; overflow: visible; }}
  .grid {{ stroke: var(--grid); stroke-width: 1; }}
  .axis {{ stroke: var(--axis); stroke-width: 1; }}
  .session-rule {{ stroke: var(--grid); stroke-width: 1; }}
  .baseline-ref {{ stroke: var(--axis); stroke-width: 1.5; }}
  .ref-label {{ fill: var(--muted); font-size: 10.5px; }}
  .tick {{ fill: var(--muted); font-size: 10.5px;
           font-variant-numeric: tabular-nums; }}
  .tick.y {{ text-anchor: end; }}
  .tick.x {{ text-anchor: middle; }}
  .line {{ fill: none; stroke-width: 2; stroke-linejoin: round;
           stroke-linecap: round; }}
  .line.benchmark {{ stroke-width: 1.5; opacity: 0.85; }}
  .end-dot {{ stroke: var(--surface); stroke-width: 2; }}
  .series-label {{ font-size: 11px; fill: var(--text-secondary); }}
  .label-swatch {{ stroke: var(--surface); stroke-width: 1; }}
  .crosshair {{ stroke: var(--axis); stroke-width: 1; pointer-events: none; }}
  .axis-note {{ font-size: 11.5px; color: var(--muted); margin: 6px 0 0;
                max-width: 76ch; }}
  .tooltip {{ position: absolute; background: var(--surface);
    border: 1px solid var(--border); border-radius: 8px; padding: 9px 11px;
    font-size: 12px; pointer-events: none; min-width: 264px;
    box-shadow: 0 6px 22px rgba(0,0,0,0.13); z-index: 5; }}
  .tip-head {{ color: var(--muted); font-size: 11px; margin-bottom: 6px;
               font-variant-numeric: tabular-nums; }}
  .tip-row {{ display: flex; align-items: center; gap: 7px;
              line-height: 1.75; }}
  .tip-name {{ flex: 1; color: var(--text-secondary); white-space: nowrap; }}
  .tip-val, .tip-ret {{ font-variant-numeric: tabular-nums; }}
  .tip-ret {{ min-width: 58px; text-align: right; }}

  table.data {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  table.data th {{ text-align: left; font-weight: 600; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted);
    padding: 0 10px 8px; border-bottom: 1px solid var(--border);
    white-space: nowrap; }}
  table.data th.num {{ text-align: right; }}
  table.data td {{ padding: 7px 10px; border-bottom: 1px solid var(--border); }}
  table.data tbody tr:last-child td {{ border-bottom: none; }}
  table.data tr.unscored {{ color: var(--text-secondary); }}
  table.tape td {{ padding: 5px 10px; font-size: 12.5px; }}
  .reason {{ max-width: 44ch; overflow: hidden; text-overflow: ellipsis;
             white-space: nowrap; }}
  .side-buy {{ color: var(--up); font-weight: 600; }}
  .side-sell {{ color: var(--down); font-weight: 600; }}

  .swatch {{ display: inline-block; width: 10px; height: 10px;
             border-radius: 3px; margin-right: 8px; vertical-align: 0; }}
  .swatch.sm {{ width: 8px; height: 8px; margin-right: 6px; }}
  .badge {{ font-size: 10.5px; padding: 1px 6px; border-radius: 20px;
            margin-left: 8px; border: 1px solid currentColor; }}
  .badge.critical {{ color: var(--critical); }}
  .badge.warning {{ color: var(--warning); }}

  .cards {{ display: grid; gap: 12px; margin-bottom: 16px;
            grid-template-columns: repeat(auto-fit, minmax(292px, 1fr)); }}
  .card {{ background: var(--surface); border: 1px solid var(--border);
           border-radius: 10px; padding: 13px 15px; }}
  .card header {{ display: flex; align-items: center; gap: 2px;
                  margin-bottom: 9px; }}
  .card-ret {{ margin-left: auto; font-variant-numeric: tabular-nums;
               font-weight: 600; }}
  .card dl {{ display: grid; grid-template-columns: auto 1fr; gap: 3px 12px;
              margin: 0; font-size: 12.5px; }}
  .card dt {{ color: var(--muted); white-space: nowrap; }}
  .card dd {{ margin: 0; overflow-wrap: anywhere; }}
  .banner {{ background: var(--surface); border: 1px solid var(--warning);
             border-left: 4px solid var(--warning); border-radius: 8px;
             padding: 11px 15px; margin-bottom: 16px; font-size: 13px;
             color: var(--text-secondary); }}
  .banner strong {{ color: var(--text-primary); }}
  .empty {{ color: var(--muted); padding: 34px 0; text-align: center; }}
  footer {{ color: var(--muted); font-size: 11.5px; margin-top: 22px;
            display: flex; gap: 16px; flex-wrap: wrap; }}

  .banner.quiet {{ border-color: var(--border); }}

  /* ---- hero ---------------------------------------------------------- */
  .hero {{
    display: flex; justify-content: space-between; align-items: center;
    gap: 24px; flex-wrap: wrap; padding: 20px 24px; margin-bottom: 16px;
    border-radius: 14px; border: 1px solid var(--border);
    background: linear-gradient(135deg, var(--surface), var(--plane));
  }}
  .hero-title {{
    font-size: 26px; font-weight: 700; letter-spacing: -.01em;
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  }}
  .badge {{
    font-size: 11px; font-weight: 600; letter-spacing: .1em;
    text-transform: uppercase; padding: 4px 10px; border-radius: 999px;
    border: 1px solid var(--border); color: var(--text-secondary);
  }}
  .badge.live {{
    border-color: var(--good); color: var(--good);
    display: inline-flex; align-items: center; gap: 7px;
  }}
  .badge .dot {{
    width: 7px; height: 7px; border-radius: 50%; background: var(--good);
    animation: pulse 1.8s ease-in-out infinite;
  }}
  @keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: .35; transform: scale(.78); }}
  }}
  .hero-round {{ margin-top: 12px; max-width: 420px; }}
  .hero-round-name {{ font-size: 13px; color: var(--text-secondary); }}
  .hero-round-sub {{ font-size: 11px; color: var(--muted); margin-top: 5px; }}
  .progress {{
    height: 6px; border-radius: 99px; background: var(--grid);
    margin-top: 8px; overflow: hidden;
  }}
  .progress span {{
    display: block; height: 100%; border-radius: 99px;
    background: var(--good); transition: width .8s ease;
  }}
  .hero-clock {{ text-align: right; }}
  .hero-clock-label {{
    font-size: 11px; letter-spacing: .1em; text-transform: uppercase;
    color: var(--text-secondary);
  }}
  .hero-clock-value {{
    font-size: 42px; font-weight: 700; line-height: 1.1;
    font-variant-numeric: tabular-nums; color: var(--text-primary);
  }}
  .hero-clock-when {{ font-size: 11px; color: var(--muted); }}

  /* ---- tabs ---------------------------------------------------------- */
  .tabs {{
    display: flex; gap: 4px; margin-bottom: 18px; flex-wrap: wrap;
    border-bottom: 1px solid var(--border);
  }}
  .tab {{
    appearance: none; background: none; border: 0; cursor: pointer;
    font: inherit; font-size: 13px; font-weight: 600; letter-spacing: .01em;
    color: var(--text-secondary); padding: 10px 16px;
    border-bottom: 2px solid transparent; margin-bottom: -1px;
  }}
  .tab:hover {{ color: var(--text-primary); }}
  .tab.on {{ color: var(--text-primary); border-bottom-color: var(--good); }}
  .panel-set.hidden {{ display: none; }}

  /* ---- podium -------------------------------------------------------- */
  .podium {{
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px;
    align-items: end; margin-bottom: 18px;
  }}
  .step {{ text-align: center; }}
  .step-team {{
    display: flex; align-items: center; justify-content: center; gap: 7px;
    font-weight: 600; font-size: 15px;
  }}
  .step-ret {{
    font-size: 26px; font-weight: 700; font-variant-numeric: tabular-nums;
    margin: 2px 0;
  }}
  .step-eq {{
    font-size: 12px; color: var(--muted); margin-bottom: 8px;
    font-variant-numeric: tabular-nums;
  }}
  .block {{
    border-radius: 10px 10px 0 0; display: flex; align-items: flex-start;
    justify-content: center; padding-top: 10px;
    border: 1px solid var(--border); border-bottom: 0;
    width: min(100%, 230px); margin: 0 auto;
  }}
  .h1 .block {{ height: 104px; }}
  .h2 .block {{ height: 74px; }}
  .h3 .block {{ height: 54px; }}
  .block .place {{
    font-size: 12px; font-weight: 700; letter-spacing: .09em;
    text-transform: uppercase; color: #12151a;
  }}
  .block.gold {{ background: #e8b53a; border-color: #e8b53a; }}
  .block.silver {{ background: #b9c0c9; border-color: #b9c0c9; }}
  .block.bronze {{ background: #c98a5e; border-color: #c98a5e; }}

  /* ---- the race ------------------------------------------------------ */
  .race {{ display: flex; flex-direction: column; gap: 3px; }}
  .race-axis {{
    display: flex; justify-content: space-between; font-size: 10px;
    letter-spacing: .09em; text-transform: uppercase; color: var(--muted);
    padding: 0 0 6px;
  }}
  .race-row {{
    display: grid; align-items: center; gap: 12px;
    grid-template-columns: 26px minmax(150px, 210px) 1fr 78px 96px;
    padding: 5px 0; border-bottom: 1px solid var(--grid);
  }}
  .race-place {{ color: var(--muted); font-size: 12px; text-align: right; }}
  .race-name {{
    display: flex; align-items: center; gap: 8px; font-weight: 600;
    font-size: 13px; white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis;
  }}
  .race-name em {{
    color: var(--muted); font-weight: 400; font-size: 10px;
    letter-spacing: .06em; text-transform: uppercase; margin-left: 2px;
  }}
  .race-track {{
    position: relative; height: 16px; background: var(--grid);
    border-radius: 4px;
  }}
  /* Zero sits in the middle: bars grow right for gains, left for losses. */
  .race-track::before {{
    content: ""; position: absolute; left: 50%; top: -2px; bottom: -2px;
    width: 1px; background: var(--axis);
  }}
  .race-bar {{
    position: absolute; top: 0; height: 100%; border-radius: 4px;
    transition: width .9s cubic-bezier(.22, 1, .36, 1);
  }}
  .race-bar.pos {{ left: 50%; background: var(--up); }}
  .race-bar.neg {{ right: 50%; background: var(--down); }}
  .race-ret, .race-eq {{ text-align: right; font-size: 13px; }}
  .race-eq {{ color: var(--text-secondary); }}
  .medal {{
    width: 9px; height: 9px; border-radius: 50%; display: inline-block;
    flex: 0 0 auto;
  }}
  .medal.gold {{ background: #e8b53a; }}
  .medal.silver {{ background: #b9c0c9; }}
  .medal.bronze {{ background: #c98a5e; }}

  /* ---- dossiers ------------------------------------------------------ */
  .dossiers {{
    display: grid; gap: 14px;
    grid-template-columns: repeat(auto-fill, minmax(330px, 1fr));
  }}
  .dossier {{
    border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px;
    background: var(--surface);
  }}
  .dossier header {{
    display: flex; align-items: center; gap: 9px; margin-bottom: 10px;
  }}
  .dossier h3 {{ margin: 0; font-size: 15px; }}
  .dossier-rank {{
    font-size: 11px; color: var(--muted); font-variant-numeric: tabular-nums;
  }}
  .dossier-ret {{
    margin-left: auto; font-size: 17px; font-weight: 700;
    font-variant-numeric: tabular-nums;
  }}
  .tagline {{
    margin: 0 0 8px; font-style: italic; font-size: 13px;
    color: var(--text-primary);
  }}
  .philosophy {{
    margin: 0 0 12px; font-size: 12px; line-height: 1.55;
    color: var(--text-secondary);
  }}
  .dossier-stats {{
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 9px;
    margin: 0 0 10px;
  }}
  .dossier-stats dt {{
    font-size: 10px; letter-spacing: .07em; text-transform: uppercase;
    color: var(--muted);
  }}
  .dossier-stats dd {{ margin: 2px 0 0; font-size: 13px; font-weight: 600; }}
  .dossier-holdings {{
    font-size: 11px; color: var(--text-secondary); padding-top: 7px;
    border-top: 1px solid var(--grid);
  }}
  .dossier-holdings span {{
    color: var(--muted); text-transform: uppercase; letter-spacing: .07em;
    margin-right: 6px; font-size: 10px;
  }}

  @media (prefers-reduced-motion: reduce) {{
    .race-bar, .progress span {{ transition: none; }}
    .badge .dot {{ animation: none; }}
  }}

  @media (max-width: 720px) {{
    .race-row {{ grid-template-columns: 22px 1fr 70px; }}
    .race-track, .race-eq {{ display: none; }}
    .hero-clock {{ text-align: left; }}
    .hero-clock-value {{ font-size: 32px; }}
  }}
  .clock {{
    display: flex; align-items: baseline; gap: 20px; flex-wrap: wrap;
    padding: 14px 18px; margin: 0 0 18px; border-radius: 10px;
    background: var(--surface); border: 1px solid var(--border);
  }}
  .clock.live {{ border-color: var(--good); }}
  .clock-label {{
    font-size: 11px; letter-spacing: .09em; text-transform: uppercase;
    color: var(--text-secondary);
  }}
  .clock-value {{
    font-size: 30px; font-weight: 600; font-variant-numeric: tabular-nums;
    color: var(--text-primary);
  }}
  .clock.live .clock-value {{ color: var(--good); }}
  .clock-when {{ font-size: 12px; color: var(--muted); }}
  .clock.done .clock-label {{ color: var(--text-primary); font-size: 16px; }}
  table.plan {{ width: 100%; border-collapse: collapse; }}
  table.plan th {{
    text-align: left; font-size: 11px; letter-spacing: .08em;
    text-transform: uppercase; color: var(--text-secondary);
    padding: 6px 10px; border-bottom: 1px solid var(--border);
  }}
  table.plan td {{ padding: 8px 10px; border-bottom: 1px solid var(--grid); }}
  table.plan tr.done td {{ color: var(--muted); }}
  .pill {{
    display: inline-block; padding: 2px 9px; border-radius: 999px;
    font-size: 11px; letter-spacing: .04em; border: 1px solid var(--border);
    color: var(--text-secondary);
  }}
  .pill.live {{ border-color: var(--good); color: var(--good); }}
  .pill.done {{ color: var(--muted); }}
"""


def render_dashboard(cfg: CompetitionConfig, data: DashboardData, *,
                     refresh: int = REFRESH_SECONDS) -> str:
    """The whole dashboard as one self-contained HTML string."""
    slots = assign_slots([t.key for t in cfg.teams])
    title = f"{cfg.name}"
    if data.round_id:
        title += f" — Round {data.round_id}"

    round_line = ""
    if data.round_start and data.round_end:
        round_line = (
            f"{data.round_start:%a %d %b} – {data.round_end:%a %d %b} · "
            f"session {data.sessions_done + (1 if data.session_open else 0)}"
            f"/{data.sessions_total or '?'}"
        )

    refresh_tag = (f'<meta http-equiv="refresh" content="{refresh}">'
                   if refresh > 0 else "")
    # Exposed on <body> so the palette validator can be pointed at the page.
    palette_attr = ",".join(CATEGORICAL_LIGHT)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{refresh_tag}
<title>{_e(title)}</title>
<style>{_styles(slots)}</style>
</head>
<body class="viz-root" data-palette="{palette_attr}">

{_hero(data)}
{_banner(data)}

{_tab_bar([
    ("leaderboard", "Leaderboard"),
    ("teams", "The Field"),
    ("season", "Season"),
    ("activity", "Activity"),
])}

<section class="panel-set" id="tab-leaderboard">
  {_podium(data)}
  {_stat_tiles(data)}
  <section class="panel">
    <h2>Standings{_e(round_line and " — " + round_line)}</h2>
    {_race(data)}
  </section>
  <section class="panel">
    <h2>Equity this round</h2>
    {_equity_chart(data)}
  </section>
  {_standings_table(data)}
</section>

<section class="panel-set hidden" id="tab-teams">
  {_team_dossiers(data)}
</section>

<section class="panel-set hidden" id="tab-season">
  {_season_plan(data)}
  {_season_table(data)}
</section>

<section class="panel-set hidden" id="tab-activity">
  {_tape(data)}
</section>

<footer>
  <span>data: {_e(data.data_source)}</span>
  <span>rules: {_e(data.rules_hash)}</span>
  <span>bankroll: {_money(cfg.starting_cash)} per team per round</span>
  <span>updated {data.generated_at:%H:%M:%S} UTC</span>
  <span>auto-refresh: {refresh}s</span>
</footer>
<script>{_SCRIPT}</script>
</body>
</html>
"""


def write_dashboard(
    cfg: CompetitionConfig,
    path: str | Path,
    *,
    engine=None,
    ledger=None,
    round_id: int | None = None,
    refresh: int = REFRESH_SECONDS,
) -> Path:
    """Render and write the dashboard atomically.

    Atomic because a browser refreshing every 30 seconds will otherwise
    occasionally read a half-written file and render a blank page.
    """
    data = build_dashboard_data(cfg, engine=engine, ledger=ledger, round_id=round_id)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(render_dashboard(cfg, data, refresh=refresh))
    tmp.replace(out)
    return out
