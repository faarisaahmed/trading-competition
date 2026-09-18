"""The ledger: an auditable SQLite record of everything that happened.

A competition whose results cannot be reconstructed is a competition whose
results can be argued with. So every decision is written down: the rules hash
in force, the dealt hands, each team's universe per session, every order and
rejection with the strategy's own stated reason, equity snapshots on a fixed
cadence, and the final scoring arithmetic.

Design notes
------------
* One file per competition (`runs/competition.sqlite` by default). SQLite
  because it needs zero setup, is transactional, and the whole thing can be
  handed to someone as a single artefact.
* WAL mode, so a live `comp report` can read while a round is still writing.
* Writes are small and frequent; each is wrapped in its own transaction so a
  crash loses at most the current tick.
* Nothing here is on the hot path of a trading decision -- if the ledger
  fails, the round keeps running and the failure is logged. Recording the
  race must never be the reason a runner trips.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ..types import UTC, Fill, Order, Rejection, utcnow

log = logging.getLogger("competition.ledger")

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    round_id    INTEGER NOT NULL,
    mode        TEXT NOT NULL,
    broker      TEXT NOT NULL,
    rules_hash  TEXT NOT NULL,
    seed        INTEGER NOT NULL,
    config_json TEXT NOT NULL,
    notes       TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS teams (
    run_id      TEXT NOT NULL,
    team_key    TEXT NOT NULL,
    name        TEXT NOT NULL,
    strategy    TEXT NOT NULL,
    picker      TEXT NOT NULL,
    scored      INTEGER NOT NULL,
    params_json TEXT NOT NULL,
    PRIMARY KEY (run_id, team_key)
);

CREATE TABLE IF NOT EXISTS drafts (
    run_id       TEXT NOT NULL,
    round_id     INTEGER NOT NULL,
    dealt_at     TEXT NOT NULL,
    method       TEXT NOT NULL,
    seed         INTEGER NOT NULL,
    target_sum   INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (run_id, round_id)
);

CREATE TABLE IF NOT EXISTS universes (
    run_id       TEXT NOT NULL,
    round_id     INTEGER NOT NULL,
    team_key     TEXT NOT NULL,
    session_date TEXT NOT NULL,
    source       TEXT NOT NULL,
    symbols_json TEXT NOT NULL,
    detail_json  TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (run_id, round_id, team_key, session_date)
);

CREATE TABLE IF NOT EXISTS orders (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    team_key        TEXT NOT NULL,
    ts              TEXT NOT NULL,
    order_id        TEXT NOT NULL,
    client_order_id TEXT NOT NULL DEFAULT '',
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    qty             REAL NOT NULL,
    order_type      TEXT NOT NULL,
    limit_price     REAL,
    status          TEXT NOT NULL,
    reason          TEXT NOT NULL DEFAULT '',
    tag             TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS fills (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id   TEXT NOT NULL,
    team_key TEXT NOT NULL,
    ts       TEXT NOT NULL,
    order_id TEXT NOT NULL,
    symbol   TEXT NOT NULL,
    side     TEXT NOT NULL,
    qty      REAL NOT NULL,
    price    REAL NOT NULL,
    fee      REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS equity (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL,
    team_key       TEXT NOT NULL,
    ts             TEXT NOT NULL,
    equity         REAL NOT NULL,
    cash           REAL NOT NULL,
    gross_exposure REAL NOT NULL,
    positions_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS rejections (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id   TEXT NOT NULL,
    team_key TEXT NOT NULL,
    ts       TEXT NOT NULL,
    symbol   TEXT NOT NULL,
    side     TEXT NOT NULL,
    reason   TEXT NOT NULL,
    detail   TEXT NOT NULL DEFAULT '',
    intent   TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id   TEXT NOT NULL,
    team_key TEXT NOT NULL DEFAULT '',
    ts       TEXT NOT NULL,
    kind     TEXT NOT NULL,
    message  TEXT NOT NULL DEFAULT '',
    data_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS results (
    run_id        TEXT NOT NULL,
    round_id      INTEGER NOT NULL,
    team_key      TEXT NOT NULL,
    start_equity  REAL NOT NULL,
    end_equity    REAL NOT NULL,
    return_pct    REAL NOT NULL,
    place         INTEGER,
    points        REAL,
    scored        INTEGER NOT NULL DEFAULT 1,
    metrics_json  TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (run_id, round_id, team_key)
);

CREATE TABLE IF NOT EXISTS learned_state (
    team_key   TEXT NOT NULL,
    kind       TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    round_id   INTEGER NOT NULL DEFAULT 0,
    blob_json  TEXT NOT NULL,
    PRIMARY KEY (team_key, kind)
);

CREATE INDEX IF NOT EXISTS idx_orders_team   ON orders (run_id, team_key, ts);
CREATE INDEX IF NOT EXISTS idx_fills_team    ON fills (run_id, team_key, ts);
CREATE INDEX IF NOT EXISTS idx_equity_team   ON equity (run_id, team_key, ts);
CREATE INDEX IF NOT EXISTS idx_reject_team   ON rejections (run_id, team_key);
CREATE INDEX IF NOT EXISTS idx_events_run    ON events (run_id, ts);
CREATE INDEX IF NOT EXISTS idx_results_round ON results (round_id, team_key);
"""


def _j(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return json.dumps({"_unserialisable": str(type(obj))})


def _iso(ts: datetime | date | None) -> str:
    if ts is None:
        return utcnow().isoformat()
    if isinstance(ts, datetime):
        return (ts if ts.tzinfo else ts.replace(tzinfo=UTC)).astimezone(UTC).isoformat()
    return ts.isoformat()


class Ledger:
    """Append-only-ish store for a competition's full history."""

    def __init__(self, path: str | Path = "runs/competition.sqlite", *, run_id: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex[:16]
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        with self._tx() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.executescript(SCHEMA)
            c.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    # ------------------------------------------------------------------ #

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    def _safe(self, sql: str, params: Sequence = ()) -> None:
        """Write, swallowing failures. The ledger must never kill a round."""
        try:
            with self._tx() as c:
                c.execute(sql, params)
        except sqlite3.Error as e:
            log.error("ledger write failed (%s): %s", sql.split()[2:4], e)

    def _safe_many(self, sql: str, rows: Sequence[Sequence]) -> None:
        if not rows:
            return
        try:
            with self._tx() as c:
                c.executemany(sql, rows)
        except sqlite3.Error as e:
            log.error("ledger batch write failed: %s", e)

    def query(self, sql: str, params: Sequence = ()) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                return cur.execute(sql, params).fetchall()
            finally:
                cur.close()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # run / team registration
    # ------------------------------------------------------------------ #

    def start_run(
        self, *, round_id: int, mode: str, broker: str, rules_hash: str, seed: int,
        config: Mapping[str, Any], notes: str = "",
    ) -> str:
        self._safe(
            "INSERT OR REPLACE INTO runs "
            "(run_id, started_at, round_id, mode, broker, rules_hash, seed, config_json, notes) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (self.run_id, _iso(utcnow()), round_id, mode, broker, rules_hash, seed,
             _j(config), notes),
        )
        return self.run_id

    def finish_run(self, notes: str = "") -> None:
        self._safe(
            "UPDATE runs SET finished_at = ?, notes = COALESCE(NULLIF(?, ''), notes) "
            "WHERE run_id = ?",
            (_iso(utcnow()), notes, self.run_id),
        )

    def register_team(self, team, *, strategy: str = "", picker: str = "") -> None:
        self._safe(
            "INSERT OR REPLACE INTO teams "
            "(run_id, team_key, name, strategy, picker, scored, params_json) "
            "VALUES (?,?,?,?,?,?,?)",
            (self.run_id, team.key, team.name, strategy or team.strategy,
             picker or team.picker, int(team.scored),
             _j({"params": dict(team.params), "picker_params": dict(team.picker_params),
                 "philosophy": team.philosophy})),
        )

    # ------------------------------------------------------------------ #
    # per-round records
    # ------------------------------------------------------------------ #

    def record_draft(self, round_id: int, draft) -> None:
        self._safe(
            "INSERT OR REPLACE INTO drafts "
            "(run_id, round_id, dealt_at, method, seed, target_sum, payload_json) "
            "VALUES (?,?,?,?,?,?,?)",
            (self.run_id, round_id, _iso(draft.dealt_at), draft.method, draft.seed,
             draft.target_sum, _j(draft.to_dict())),
        )

    def record_universe(
        self, round_id: int, team_key: str, session: date, symbols: Iterable[str],
        *, source: str, detail: Mapping[str, Any] | None = None,
    ) -> None:
        self._safe(
            "INSERT OR REPLACE INTO universes "
            "(run_id, round_id, team_key, session_date, source, symbols_json, detail_json) "
            "VALUES (?,?,?,?,?,?,?)",
            (self.run_id, round_id, team_key, _iso(session), source,
             _j(list(symbols)), _j(dict(detail or {}))),
        )

    def record_orders(self, team_key: str, orders: Iterable[Order]) -> None:
        rows = [
            (self.run_id, team_key, _iso(o.submitted_at), o.id, o.client_order_id,
             o.symbol, o.side.value, o.qty, o.order_type.value, o.limit_price,
             o.status.value, o.reason, o.tag)
            for o in orders
        ]
        self._safe_many(
            "INSERT INTO orders (run_id, team_key, ts, order_id, client_order_id, symbol, "
            "side, qty, order_type, limit_price, status, reason, tag) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )

    def record_fills(self, team_key: str, fills: Iterable[Fill]) -> None:
        rows = [
            (self.run_id, team_key, _iso(f.ts), f.order_id, f.symbol, f.side.value,
             f.qty, f.price, f.fee)
            for f in fills
        ]
        self._safe_many(
            "INSERT INTO fills (run_id, team_key, ts, order_id, symbol, side, qty, price, fee) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )

    def record_equity(self, team_key: str, ts: datetime, account) -> None:
        self._safe(
            "INSERT INTO equity (run_id, team_key, ts, equity, cash, gross_exposure, "
            "positions_json) VALUES (?,?,?,?,?,?,?)",
            (self.run_id, team_key, _iso(ts), account.equity, account.cash,
             account.gross_exposure,
             _j({p.symbol: round(p.qty, 6) for p in account.positions})),
        )

    def record_rejections(self, team_key: str, ts: datetime, rejections: Iterable[Rejection]) -> None:
        rows = [
            (self.run_id, team_key, _iso(ts), r.intent.symbol, r.intent.side.value,
             r.reason.value, r.detail, r.intent.describe())
            for r in rejections
        ]
        self._safe_many(
            "INSERT INTO rejections (run_id, team_key, ts, symbol, side, reason, detail, intent) "
            "VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )

    def record_event(
        self, kind: str, message: str = "", *, team_key: str = "",
        data: Mapping[str, Any] | None = None, ts: datetime | None = None,
    ) -> None:
        self._safe(
            "INSERT INTO events (run_id, team_key, ts, kind, message, data_json) "
            "VALUES (?,?,?,?,?,?)",
            (self.run_id, team_key, _iso(ts), kind, message, _j(dict(data or {}))),
        )

    def record_result(
        self, round_id: int, team_key: str, *, start_equity: float, end_equity: float,
        return_pct: float, place: int | None = None, points: float | None = None,
        scored: bool = True, metrics: Mapping[str, Any] | None = None,
    ) -> None:
        self._safe(
            "INSERT OR REPLACE INTO results (run_id, round_id, team_key, start_equity, "
            "end_equity, return_pct, place, points, scored, metrics_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (self.run_id, round_id, team_key, start_equity, end_equity, return_pct,
             place, points, int(scored), _j(dict(metrics or {}))),
        )

    # ------------------------------------------------------------------ #
    # learned state (survives between rounds -- the RL entries depend on it)
    # ------------------------------------------------------------------ #

    def save_learned_state(
        self, team_key: str, kind: str, blob: Mapping[str, Any], *, round_id: int = 0
    ) -> None:
        self._safe(
            "INSERT OR REPLACE INTO learned_state "
            "(team_key, kind, updated_at, round_id, blob_json) VALUES (?,?,?,?,?)",
            (team_key, kind, _iso(utcnow()), round_id, _j(dict(blob))),
        )

    def load_learned_state(self, team_key: str, kind: str) -> dict[str, Any]:
        rows = self.query(
            "SELECT blob_json FROM learned_state WHERE team_key = ? AND kind = ?",
            (team_key, kind),
        )
        if not rows:
            return {}
        try:
            return json.loads(rows[0]["blob_json"])
        except (json.JSONDecodeError, TypeError):
            log.warning("learned state for %s/%s is corrupt; ignoring", team_key, kind)
            return {}

    # ------------------------------------------------------------------ #
    # reads used by scoring and reporting
    # ------------------------------------------------------------------ #

    def equity_curve(self, team_key: str, round_id: int | None = None) -> list[tuple[str, float]]:
        if round_id is None:
            rows = self.query(
                "SELECT ts, equity FROM equity WHERE run_id = ? AND team_key = ? ORDER BY ts",
                (self.run_id, team_key),
            )
        else:
            rows = self.query(
                "SELECT e.ts, e.equity FROM equity e JOIN runs r ON r.run_id = e.run_id "
                "WHERE e.run_id = ? AND e.team_key = ? AND r.round_id = ? ORDER BY e.ts",
                (self.run_id, team_key, round_id),
            )
        return [(r["ts"], float(r["equity"])) for r in rows]

    def fill_count(self, team_key: str) -> int:
        rows = self.query(
            "SELECT COUNT(*) AS n FROM fills WHERE run_id = ? AND team_key = ?",
            (self.run_id, team_key),
        )
        return int(rows[0]["n"]) if rows else 0

    def traded_notional(self, team_key: str) -> float:
        rows = self.query(
            "SELECT COALESCE(SUM(qty * price), 0) AS v FROM fills "
            "WHERE run_id = ? AND team_key = ?",
            (self.run_id, team_key),
        )
        return float(rows[0]["v"]) if rows else 0.0

    def rejection_tally(self, team_key: str) -> dict[str, int]:
        rows = self.query(
            "SELECT reason, COUNT(*) AS n FROM rejections WHERE run_id = ? AND team_key = ? "
            "GROUP BY reason ORDER BY n DESC",
            (self.run_id, team_key),
        )
        return {r["reason"]: int(r["n"]) for r in rows}

    def results_for_round(self, round_id: int) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM results WHERE round_id = ? ORDER BY place IS NULL, place",
            (round_id,),
        )

    def all_results(self) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM results ORDER BY round_id, place IS NULL, place"
        )

    def runs(self) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM runs ORDER BY started_at")

    def latest_draft(self, round_id: int) -> dict[str, Any] | None:
        rows = self.query(
            "SELECT payload_json FROM drafts WHERE round_id = ? ORDER BY dealt_at DESC LIMIT 1",
            (round_id,),
        )
        if not rows:
            return None
        try:
            return json.loads(rows[0]["payload_json"])
        except json.JSONDecodeError:
            return None

    def universe_for(self, round_id: int, team_key: str) -> list[str]:
        rows = self.query(
            "SELECT symbols_json FROM universes WHERE round_id = ? AND team_key = ? "
            "ORDER BY session_date DESC LIMIT 1",
            (round_id, team_key),
        )
        if not rows:
            return []
        try:
            return list(json.loads(rows[0]["symbols_json"]))
        except json.JSONDecodeError:
            return []

    def trade_log(self, team_key: str, limit: int = 200) -> list[sqlite3.Row]:
        return self.query(
            "SELECT f.ts, f.symbol, f.side, f.qty, f.price, o.reason, o.tag "
            "FROM fills f LEFT JOIN orders o ON o.order_id = f.order_id "
            "AND o.run_id = f.run_id WHERE f.run_id = ? AND f.team_key = ? "
            "ORDER BY f.ts DESC LIMIT ?",
            (self.run_id, team_key, limit),
        )

    def stats(self) -> dict[str, int]:
        out = {}
        for table in ("runs", "teams", "orders", "fills", "equity", "rejections",
                      "events", "results", "universes", "drafts", "learned_state"):
            rows = self.query(f"SELECT COUNT(*) AS n FROM {table}")
            out[table] = int(rows[0]["n"]) if rows else 0
        return out
