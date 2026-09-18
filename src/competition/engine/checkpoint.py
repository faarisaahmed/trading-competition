"""Round checkpoints, so a process death does not void the week.

A round runs for seven calendar days. Over that span a laptop will sleep, an
SSH session will drop, or Python will hit something unexpected. Without
checkpoints the only recovery is `prepare_round(reset_accounts=True)`, which
flattens every account and restarts the round from zero -- a full week of
nine strategies' work, gone.

So the engine writes a checkpoint on every equity snapshot and at every
session boundary, and on startup it looks for an in-progress round to rejoin.

What has to survive, and why each one matters
---------------------------------------------
`baseline_equity`  The round's measurement zero. If this were re-read from
                   the live account on resume, a team down 3% would silently
                   have its loss forgiven and the round's scoring would be
                   wrong for everyone. This is the single most important
                   field in the file.
`strategy state`   Trailing stops, entry prices, cooldowns, the gambler's
                   martingale rung, the pair trader's fitted betas, the
                   Q-learner's per-symbol action state. Losing it means every
                   strategy wakes up thinking it is flat and re-enters
                   positions it already holds.
`risk state`       Session-open equity (so the kill switch measures the right
                   drawdown), the daily order count, and whether the team is
                   currently halted.
`seen_fills`       Fill identities already processed, so a resume does not
                   replay them into `on_fill` and double-count.
`session_seen`     Which sessions have already had their `on_session_start`
                   hook fired.

Positions themselves are NOT checkpointed: they live in the broker account,
which is the authority. The checkpoint only restores the engine's *memory* of
what it was doing.

Everything is JSON, and serialisation is defensive: a strategy that stashes
something unserialisable in `ctx.state` gets that key dropped with a warning
rather than taking down the checkpoint.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from ..types import UTC
from .guardrails import RiskState

log = logging.getLogger("competition.checkpoint")

CHECKPOINT_VERSION = 1


def _jsonable(value: Any, *, path: str = "") -> Any:
    """Coerce a value to something json can hold, or raise TypeError."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v, path=f"{path}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, path=f"{path}[]") for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(v, path=f"{path}{{}}") for v in value)
    raise TypeError(f"{path}: cannot serialise {type(value).__name__}")


def safe_state(state: Mapping[str, Any], *, who: str = "") -> dict[str, Any]:
    """Serialise a strategy's scratch state, dropping anything exotic."""
    out: dict[str, Any] = {}
    for key, value in state.items():
        try:
            out[str(key)] = _jsonable(value, path=str(key))
        except TypeError as e:
            log.warning("%s: dropping unserialisable state key %r (%s)", who, key, e)
    return out


def _as_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _as_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


# --------------------------------------------------------------------------- #


@dataclass
class TeamCheckpoint:
    """One team's resumable state."""

    team_key: str
    baseline_equity: float
    state: dict[str, Any] = field(default_factory=dict)
    picker_state: dict[str, Any] = field(default_factory=dict)
    universe: list[str] = field(default_factory=list)
    universe_source: str = ""
    tick_count: int = 0
    last_tick: str | None = None
    seen_fills: list[str] = field(default_factory=list)
    session_seen: list[str] = field(default_factory=list)
    entry_marks: dict[str, float] = field(default_factory=dict)
    timeouts: int = 0
    error_count: int = 0
    # risk
    session_date: str | None = None
    session_open_equity: float = 0.0
    orders_today: int = 0
    halted_until: str | None = None
    halt_reason: str = ""
    rejections: dict[str, int] = field(default_factory=dict)
    peak_equity: float = 0.0

    @classmethod
    def from_runtime(cls, rt) -> TeamCheckpoint:
        return cls(
            team_key=rt.key,
            baseline_equity=float(rt.baseline_equity),
            state=safe_state(rt.state, who=rt.key),
            picker_state=safe_state(rt.picker_state, who=f"{rt.key}.picker"),
            universe=list(rt.universe),
            universe_source=rt.universe_source,
            tick_count=int(rt.tick_count),
            last_tick=rt.last_tick.isoformat() if rt.last_tick else None,
            seen_fills=sorted(rt.seen_fills),
            session_seen=sorted(d.isoformat() for d in rt.session_seen),
            entry_marks={k: float(v) for k, v in rt.entry_marks.items()},
            timeouts=int(rt.timeouts),
            error_count=len(rt.errors),
            session_date=rt.risk.session_date.isoformat() if rt.risk.session_date else None,
            session_open_equity=float(rt.risk.session_open_equity),
            orders_today=int(rt.risk.orders_today),
            halted_until=(rt.risk.halted_until.isoformat()
                          if rt.risk.halted_until else None),
            halt_reason=rt.risk.halt_reason,
            rejections=dict(rt.risk.rejections),
            peak_equity=float(rt.risk.peak_equity),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": CHECKPOINT_VERSION,
            "team_key": self.team_key,
            "baseline_equity": self.baseline_equity,
            "state": self.state,
            "picker_state": self.picker_state,
            "universe": self.universe,
            "universe_source": self.universe_source,
            "tick_count": self.tick_count,
            "last_tick": self.last_tick,
            "seen_fills": self.seen_fills,
            "session_seen": self.session_seen,
            "entry_marks": self.entry_marks,
            "timeouts": self.timeouts,
            "error_count": self.error_count,
            "risk": {
                "session_date": self.session_date,
                "session_open_equity": self.session_open_equity,
                "orders_today": self.orders_today,
                "halted_until": self.halted_until,
                "halt_reason": self.halt_reason,
                "rejections": self.rejections,
                "peak_equity": self.peak_equity,
            },
        }

    @classmethod
    def from_dict(cls, blob: Mapping[str, Any]) -> TeamCheckpoint:
        risk = blob.get("risk") or {}
        return cls(
            team_key=str(blob.get("team_key", "")),
            baseline_equity=float(blob.get("baseline_equity", 0.0) or 0.0),
            state=dict(blob.get("state") or {}),
            picker_state=dict(blob.get("picker_state") or {}),
            universe=[str(s) for s in (blob.get("universe") or [])],
            universe_source=str(blob.get("universe_source", "")),
            tick_count=int(blob.get("tick_count", 0) or 0),
            last_tick=blob.get("last_tick"),
            seen_fills=[str(x) for x in (blob.get("seen_fills") or [])],
            session_seen=[str(x) for x in (blob.get("session_seen") or [])],
            entry_marks={str(k): float(v)
                         for k, v in (blob.get("entry_marks") or {}).items()},
            timeouts=int(blob.get("timeouts", 0) or 0),
            error_count=int(blob.get("error_count", 0) or 0),
            session_date=risk.get("session_date"),
            session_open_equity=float(risk.get("session_open_equity", 0.0) or 0.0),
            orders_today=int(risk.get("orders_today", 0) or 0),
            halted_until=risk.get("halted_until"),
            halt_reason=str(risk.get("halt_reason", "")),
            rejections={str(k): int(v) for k, v in (risk.get("rejections") or {}).items()},
            peak_equity=float(risk.get("peak_equity", 0.0) or 0.0),
        )

    def apply_to(self, rt, *, equity_curve: list[tuple[datetime, float]] | None = None) -> None:
        """Restore this checkpoint onto a fresh `TeamRuntime`."""
        rt.baseline_equity = self.baseline_equity
        rt.state = dict(self.state)
        rt.picker_state = dict(self.picker_state)
        if self.universe:
            rt.universe = tuple(self.universe)
            rt.universe_source = self.universe_source or rt.universe_source
        rt.tick_count = self.tick_count
        rt.last_tick = _as_dt(self.last_tick)
        rt.seen_fills = set(self.seen_fills)
        rt.session_seen = {d for d in (_as_date(x) for x in self.session_seen) if d}
        rt.entry_marks = dict(self.entry_marks)
        rt.timeouts = self.timeouts
        if equity_curve:
            rt.equity_curve = list(equity_curve)

        risk = RiskState(rt.key)
        risk.session_date = _as_date(self.session_date)
        risk.session_open_equity = self.session_open_equity
        risk.orders_today = self.orders_today
        risk.halted_until = _as_date(self.halted_until)
        risk.halt_reason = self.halt_reason
        risk.rejections = dict(self.rejections)
        risk.peak_equity = self.peak_equity
        rt.risk = risk


@dataclass
class RoundCheckpoint:
    """A round in progress, and every team's place in it."""

    round_id: int
    run_id: str
    start_date: date
    end_date: date
    mode: str
    broker: str
    rules_hash: str
    status: str = "in_progress"
    ticks: int = 0
    updated_at: datetime | None = None
    teams: dict[str, TeamCheckpoint] = field(default_factory=dict)

    @property
    def is_resumable(self) -> bool:
        return self.status == "in_progress" and bool(self.teams)

    def compatible_with(
        self, *, round_id: int, start: date, end: date, rules_hash: str,
        team_keys: set[str],
    ) -> tuple[bool, str]:
        """Is this checkpoint safe to resume into the requested round?

        Every mismatch here is a reason a resume would produce a *different*
        competition than the one that was interrupted, so each is refused
        rather than papered over.
        """
        if self.round_id != round_id:
            return False, f"checkpoint is for round {self.round_id}, not {round_id}"
        if self.start_date != start:
            return False, (f"checkpoint round started {self.start_date}, "
                           f"requested start is {start}")
        if self.end_date != end:
            return False, (f"checkpoint round ends {self.end_date}, "
                           f"requested end is {end}")
        if self.rules_hash != rules_hash:
            return False, (f"the rulebook changed since the checkpoint "
                           f"({self.rules_hash} -> {rules_hash})")
        missing = team_keys - set(self.teams)
        if missing:
            return False, f"no checkpoint for team(s): {', '.join(sorted(missing))}"
        extra = set(self.teams) - team_keys
        if extra:
            return False, (f"checkpoint has team(s) not in this run: "
                           f"{', '.join(sorted(extra))}")
        return True, "compatible"

    def describe(self) -> str:
        return (
            f"round {self.round_id} ({self.status}) {self.start_date}..{self.end_date} "
            f"run={self.run_id} ticks={self.ticks} teams={len(self.teams)} "
            f"updated={self.updated_at:%Y-%m-%d %H:%M}" if self.updated_at else ""
        )


def load_round_checkpoint(payload: Mapping[str, Any] | None) -> RoundCheckpoint | None:
    """Build a `RoundCheckpoint` from `Ledger.resumable_round()`'s payload."""
    if not payload:
        return None
    start = _as_date(payload.get("start_date"))
    end = _as_date(payload.get("end_date"))
    if start is None or end is None:
        log.warning("checkpoint has no usable round window; ignoring")
        return None
    teams: dict[str, TeamCheckpoint] = {}
    for key, blob in (payload.get("teams") or {}).items():
        try:
            teams[str(key)] = TeamCheckpoint.from_dict(blob)
        except (TypeError, ValueError) as e:
            log.warning("checkpoint for %s is unreadable (%s); ignoring", key, e)
    return RoundCheckpoint(
        round_id=int(payload.get("round_id", 0)),
        run_id=str(payload.get("run_id", "")),
        start_date=start,
        end_date=end,
        mode=str(payload.get("mode", "")),
        broker=str(payload.get("broker", "")),
        rules_hash=str(payload.get("rules_hash", "")),
        status=str(payload.get("status", "in_progress")),
        ticks=int(payload.get("ticks", 0) or 0),
        updated_at=_as_dt(payload.get("updated_at")),
        teams=teams,
    )
