"""The competition engine: the loop that actually runs a round.

One tick looks like this, and the order of the steps is the whole fairness
argument:

  1. Build **one** `MarketSnapshot` from the shared feed.
  2. Rotate the team order (so no team is always first to see it).
  3. For each team whose own cadence is due:
       a. restrict the snapshot to that team's universe (Round 3 enforcement
          happens at the data layer, not just at order entry)
       b. read the account, roll the session, check the kill switch
       c. call `strategy.on_tick` under a wall-clock budget
       d. drain the strategy's cancel requests
       e. validate every intent through the shared guardrails
       f. submit what survived; record orders, fills, rejections
  4. Snapshot equity for everyone on the same tick.

Both live and replay modes use this same loop. The only differences are the
feed, the broker, and where `now` comes from -- which is what makes a
backtest a genuine dress rehearsal rather than a different program.
"""

from __future__ import annotations

import contextlib
import logging
import random
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from ..broker.base import Broker, BrokerError, OrderRejected
from ..config import CompetitionConfig, RoundConfig, TeamConfig
from ..data.feed import Feed
from ..data.snapshot import MarketSnapshot
from ..strategies import load_strategy
from ..strategies.base import Strategy, StrategyContext
from ..types import (
    UTC,
    Account,
    Fill,
    Order,
    OrderIntent,
    Rejection,
    RejectReason,
    Side,
    utcnow,
)
from ..util import indicators as ind
from .guardrails import Guardrails, RiskState
from .ledger import Ledger
from .universe import UniverseResolver

log = logging.getLogger("competition.engine")


@dataclass
class TeamRuntime:
    """Everything the engine holds for one team during a round."""

    config: TeamConfig
    strategy: Strategy
    broker: Broker
    risk: RiskState
    universe: tuple[str, ...] = ()
    universe_source: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    picker_state: dict[str, Any] = field(default_factory=dict)
    rng: random.Random = field(default_factory=random.Random)
    baseline_equity: float = 0.0
    last_tick: datetime | None = None
    tick_count: int = 0
    seen_fills: set[str] = field(default_factory=set)
    fills: list[Fill] = field(default_factory=list)
    orders: list[Order] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    timeouts: int = 0
    session_seen: set[date] = field(default_factory=set)
    #: Entry equity per symbol, used to hand picker feedback to the bandit.
    entry_marks: dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.config.key

    def due(self, now: datetime) -> bool:
        if self.last_tick is None:
            return True
        return (now - self.last_tick).total_seconds() >= self.strategy.tick_seconds - 1e-6


@dataclass
class TeamResult:
    """One team's outcome for one round."""

    team_key: str
    name: str
    scored: bool
    start_equity: float
    end_equity: float
    return_pct: float
    fills: int
    traded_notional: float
    max_drawdown: float
    sharpe: float
    rejections: dict[str, int]
    errors: int
    timeouts: int
    universe: tuple[str, ...]
    place: int | None = None
    points: float = 0.0

    def metrics(self) -> dict:
        return {
            "fills": self.fills,
            "traded_notional": round(self.traded_notional, 2),
            "max_drawdown": round(self.max_drawdown, 6),
            "sharpe": round(self.sharpe, 4),
            "rejections": self.rejections,
            "errors": self.errors,
            "timeouts": self.timeouts,
            "universe": list(self.universe),
        }


@dataclass
class RoundResult:
    round_id: int
    round_name: str
    started_at: datetime
    finished_at: datetime
    mode: str
    teams: list[TeamResult] = field(default_factory=list)
    ticks: int = 0
    draft: dict | None = None

    def scored_teams(self) -> list[TeamResult]:
        return [t for t in self.teams if t.scored]

    def team(self, key: str) -> TeamResult:
        for t in self.teams:
            if t.team_key == key:
                return t
        raise KeyError(key)

    def table(self) -> str:
        rows = sorted(self.teams, key=lambda t: (-t.return_pct,))
        width = max((len(t.name) for t in rows), default=10)
        out = [
            f"Round {self.round_id} -- {self.round_name}  ({self.mode}, {self.ticks} ticks)",
            f"{'#':<3} {'team':<{width}} {'return':>9} {'end equity':>12} {'fills':>7} "
            f"{'maxDD':>8} {'pts':>5}",
            "-" * (width + 50),
        ]
        for i, t in enumerate(rows, 1):
            place = "--" if not t.scored else str(t.place or i)
            out.append(
                f"{place:<3} {t.name:<{width}} {t.return_pct:>8.2%} "
                f"{t.end_equity:>12,.2f} {t.fills:>7} {t.max_drawdown:>8.2%} "
                f"{t.points:>5.1f}" + ("" if t.scored else "   (unscored)")
            )
        return "\n".join(out)


class StrategyTimeout(RuntimeError):
    pass


class CompetitionEngine:
    """Runs one round for all teams against a feed and a broker per team."""

    def __init__(
        self,
        cfg: CompetitionConfig,
        *,
        feed: Feed,
        brokers: Mapping[str, Broker],
        ledger: Ledger,
        resolver: UniverseResolver,
        mode: str = "live",
        teams: Sequence[TeamConfig] | None = None,
        clock: Callable[[], datetime] = utcnow,
        is_tradable: Callable[[str], bool] | None = None,
        is_fractionable: Callable[[str], bool] | None = None,
        allow_closed_market: bool = False,
        equity_snapshot_seconds: int = 300,
    ):
        self.cfg = cfg
        self.feed = feed
        self.ledger = ledger
        self.resolver = resolver
        self.mode = mode
        self.clock = clock
        self.equity_snapshot_seconds = equity_snapshot_seconds
        self.guardrails = Guardrails(
            cfg.risk,
            is_tradable=is_tradable,
            is_fractionable=is_fractionable,
            allow_closed_market=allow_closed_market,
        )
        self._is_fractionable = is_fractionable or (lambda _s: True)
        self.teams: dict[str, TeamRuntime] = {}
        for team in (teams if teams is not None else cfg.teams):
            broker = brokers.get(team.key)
            if broker is None:
                log.warning("%s has no broker configured; skipping", team.key)
                continue
            self.teams[team.key] = TeamRuntime(
                config=team,
                strategy=load_strategy(team.strategy)(team),
                broker=broker,
                risk=RiskState(team.key),
            )
        self._last_equity_snapshot: datetime | None = None
        self._rotation = 0

    # ------------------------------------------------------------------ #
    # learned state
    # ------------------------------------------------------------------ #

    def load_learned_state(self) -> None:
        """Restore Q-tables and bandit models so learning spans rounds."""
        for rt in self.teams.values():
            blob = self.ledger.load_learned_state(rt.key, "strategy")
            if blob:
                try:
                    rt.strategy.load_state(blob)
                except Exception as e:  # noqa: BLE001
                    log.warning("%s: could not load strategy state: %s", rt.key, e)
            pblob = self.ledger.load_learned_state(rt.key, "picker")
            if pblob:
                try:
                    self.resolver.picker_for(rt.config).load_state(pblob)
                except Exception as e:  # noqa: BLE001
                    log.warning("%s: could not load picker state: %s", rt.key, e)

    def save_learned_state(self, round_id: int) -> None:
        for rt in self.teams.values():
            try:
                blob = rt.strategy.state_dict()
                if blob:
                    self.ledger.save_learned_state(rt.key, "strategy", blob, round_id=round_id)
            except Exception as e:  # noqa: BLE001
                log.warning("%s: could not save strategy state: %s", rt.key, e)
            try:
                pblob = self.resolver.picker_for(rt.config).state_dict()
                if pblob:
                    self.ledger.save_learned_state(rt.key, "picker", pblob, round_id=round_id)
            except Exception as e:  # noqa: BLE001
                log.warning("%s: could not save picker state: %s", rt.key, e)

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #

    def prepare_round(self, rnd: RoundConfig, *, session: date, reset_accounts: bool = True) -> None:
        """Reset accounts, deal Round 3 if needed, resolve universes, warm the feed."""
        self.ledger.record_event(
            "round_prepare", f"round {rnd.id} ({rnd.name})",
            data={"mode": self.mode, "universe_mode": rnd.universe_mode,
                  "teams": list(self.teams)},
        )
        for rt in self.teams.values():
            self.ledger.register_team(rt.config)
            rt.rng = random.Random(self.cfg.team_seed(rt.key, rnd.id))
            rt.state = {}
            rt.risk = RiskState(rt.key)
            if reset_accounts:
                try:
                    rt.broker.reset_for_round(self.cfg.starting_cash)
                except BrokerError as e:
                    log.error("%s: could not reset the account: %s", rt.key, e)
            baseline = getattr(rt.broker, "baseline_equity", None)
            if baseline is None:
                baseline = rt.broker.account().equity
            rt.baseline_equity = float(baseline or self.cfg.starting_cash)

        if rnd.universe_mode == "draft" and self.resolver.draft is None:
            draft = self.resolver.run_draft(
                rnd, [rt.config for rt in self.teams.values()],
                seed=self.cfg.fairness.competition_seed + rnd.id,
            )
            self.ledger.record_draft(rnd.id, draft)

        self.refresh_universes(rnd, session=session)

        all_symbols = sorted({s for rt in self.teams.values() for s in rt.universe})
        if all_symbols:
            self.feed.prime(all_symbols)
        self.ledger.record_event(
            "round_ready", f"round {rnd.id} universes resolved",
            data={rt.key: list(rt.universe) for rt in self.teams.values()},
        )

    def refresh_universes(self, rnd: RoundConfig, *, session: date) -> None:
        """Re-run the pickers for a new session (Round 2 and 3, once a day)."""
        for rt in self.teams.values():
            try:
                resolved = self.resolver.resolve(
                    rnd, rt.config, session=session, picker_state=rt.picker_state
                )
            except Exception as e:  # noqa: BLE001
                log.exception("%s: universe resolution failed: %s", rt.key, e)
                rt.errors.append(f"universe: {e}")
                continue
            previous = rt.universe
            rt.universe = resolved.symbols
            rt.universe_source = resolved.source
            self.ledger.record_universe(
                rnd.id, rt.key, session, resolved.symbols,
                source=resolved.source, detail=resolved.detail,
            )
            if previous and set(previous) != set(resolved.symbols):
                dropped = sorted(set(previous) - set(resolved.symbols))
                added = sorted(set(resolved.symbols) - set(previous))
                log.info("%s universe changed: +%s -%s", rt.key, added, dropped)
                # Positions in names that left the universe must be closed --
                # otherwise a team could hold something it is no longer
                # allowed to trade, and could not exit it either.
                self._liquidate_off_universe(rt, dropped)

    def _liquidate_off_universe(self, rt: TeamRuntime, dropped: Sequence[str]) -> None:
        for sym in dropped:
            try:
                if abs(rt.broker.account().qty(sym)) > 1e-9:
                    rt.broker.cancel_all(sym)
                    order = rt.broker.close_position(sym)
                    if order:
                        rt.orders.append(order)
                        log.info("%s: closed %s (left the universe)", rt.key, sym)
            except BrokerError as e:
                log.warning("%s: could not close %s: %s", rt.key, sym, e)

    # ------------------------------------------------------------------ #
    # the tick
    # ------------------------------------------------------------------ #

    def _call_strategy(
        self, rt: TeamRuntime, ctx: StrategyContext
    ) -> tuple[list[OrderIntent], float]:
        """Run `on_tick` with a wall-clock budget, identical for every team."""
        budget = self.cfg.fairness.strategy_timeout_seconds
        started = time.monotonic()
        try:
            intents = list(rt.strategy.on_tick(ctx) or [])
        except Exception as e:  # noqa: BLE001 -- one bad tick must not end the round
            elapsed = time.monotonic() - started
            rt.errors.append(f"{type(e).__name__}: {e}")
            log.error(
                "%s: on_tick raised %s (tick %d)\n%s",
                rt.key, e, rt.tick_count, traceback.format_exc(limit=6),
            )
            self.ledger.record_event(
                "strategy_error", str(e), team_key=rt.key,
                data={"tick": rt.tick_count, "traceback": traceback.format_exc(limit=8)},
                ts=ctx.ts,
            )
            return [], elapsed
        elapsed = time.monotonic() - started
        if elapsed > budget:
            # Not enforced by killing the thread mid-decision (that would risk
            # half-applied strategy state); the overrun is recorded and the
            # intents are dropped, which costs the team the tick.
            rt.timeouts += 1
            log.warning(
                "%s: on_tick took %.2fs, over the %.2fs budget -- tick discarded",
                rt.key, elapsed, budget,
            )
            self.ledger.record_event(
                "strategy_timeout", f"{elapsed:.3f}s > {budget:.2f}s",
                team_key=rt.key, data={"tick": rt.tick_count}, ts=ctx.ts,
            )
            return [], elapsed
        return intents, elapsed

    def _drain_cancels(self, rt: TeamRuntime, ctx: StrategyContext) -> int:
        n = 0
        for order_id in ctx.cancel_requests:
            try:
                rt.broker.cancel(order_id)
                n += 1
            except BrokerError as e:
                log.debug("%s: cancel %s failed: %s", rt.key, order_id, e)
        ctx.cancel_requests.clear()
        return n

    def _submit(
        self, rt: TeamRuntime, intents: Sequence[OrderIntent], ts: datetime
    ) -> list[Order]:
        submitted: list[Order] = []
        rejections: list[Rejection] = []
        for intent in intents:
            if intent.replace_open:
                try:
                    rt.broker.cancel_all(intent.symbol)
                except BrokerError as e:
                    log.debug("%s: cancel_all(%s) failed: %s", rt.key, intent.symbol, e)
            try:
                order = rt.broker.submit(intent)
            except OrderRejected as e:
                rejections.append(Rejection(intent, RejectReason.BROKER_ERROR, str(e)[:300]))
                rt.risk.record_rejection(RejectReason.BROKER_ERROR)
                log.info("%s: broker rejected %s (%s)", rt.key, intent.describe(), e)
                continue
            except BrokerError as e:
                rejections.append(Rejection(intent, RejectReason.BROKER_ERROR, str(e)[:300]))
                rt.risk.record_rejection(RejectReason.BROKER_ERROR)
                log.error("%s: submit failed for %s: %s", rt.key, intent.describe(), e)
                continue
            order.reason = intent.reason
            order.tag = intent.tag
            submitted.append(order)
            rt.orders.append(order)
            if intent.side is Side.BUY:
                rt.entry_marks.setdefault(intent.symbol, rt.broker.account().equity)
        if submitted:
            self.ledger.record_orders(rt.key, submitted)
        if rejections:
            self.ledger.record_rejections(rt.key, ts, rejections)
        return submitted

    def _collect_fills(self, rt: TeamRuntime, ctx: StrategyContext) -> list[Fill]:
        """Pull new fills from the broker (simulator only) and notify the team."""
        fresh: list[Fill] = []
        broker_fills = getattr(rt.broker, "fills", None)
        if broker_fills is None:
            return fresh
        for f in broker_fills:
            fid = f"{f.order_id}:{f.ts.isoformat()}:{f.qty:.6f}"
            if fid in rt.seen_fills:
                continue
            rt.seen_fills.add(fid)
            fresh.append(f)
            rt.fills.append(f)
            try:
                rt.strategy.on_fill(f, ctx)
            except Exception as e:  # noqa: BLE001
                log.debug("%s: on_fill raised %s", rt.key, e)
        if fresh:
            self.ledger.record_fills(rt.key, fresh)
        return fresh

    def _context(
        self, rt: TeamRuntime, snapshot: MarketSnapshot, account: Account,
        rnd: RoundConfig, progress: float, sessions_left: int, final: bool,
    ) -> StrategyContext:
        view = snapshot.restricted_to(rt.universe)
        try:
            open_orders = tuple(rt.broker.open_orders())
        except BrokerError as e:
            log.debug("%s: open_orders failed: %s", rt.key, e)
            open_orders = ()
        return StrategyContext(
            team_key=rt.key,
            round_id=rnd.id,
            snapshot=view,
            account=account,
            universe=rt.universe,
            starting_cash=self.cfg.starting_cash,
            baseline_equity=rt.baseline_equity,
            risk=self.cfg.risk,
            rng=rt.rng,
            state=rt.state,
            log=logging.getLogger(f"competition.strategy.{rt.key}"),
            open_orders=open_orders,
            tick_index=rt.tick_count,
            round_progress=progress,
            sessions_remaining=sessions_left,
            is_final_session=final,
            fractionable=frozenset(s for s in rt.universe if self._is_fractionable(s)),
        )

    def tick(
        self,
        rnd: RoundConfig,
        *,
        now: datetime | None = None,
        progress: float = 0.0,
        sessions_left: int = 0,
        final_session: bool = False,
    ) -> int:
        """Run one engine tick. Returns how many teams acted."""
        now = (now or self.clock()).astimezone(UTC)
        symbols = sorted({s for rt in self.teams.values() for s in rt.universe})
        if not symbols:
            return 0
        snapshot = self.feed.snapshot(symbols, now)
        today = snapshot.session.session_date

        # Rotate so the same team is not always first to act on a snapshot.
        keys = list(self.teams)
        if self.cfg.fairness.rotate_execution_order and keys:
            offset = self._rotation % len(keys)
            keys = keys[offset:] + keys[:offset]
        self._rotation += 1

        acted = 0
        for key in keys:
            rt = self.teams[key]
            if not rt.universe or not rt.due(now):
                continue
            rt.last_tick = now
            rt.tick_count += 1

            try:
                rt.broker.sync(now)
                account = rt.broker.account()
            except BrokerError as e:
                rt.errors.append(f"account: {e}")
                log.error("%s: account read failed: %s", rt.key, e)
                continue

            rolled = rt.risk.roll_session(today, account.equity)
            ctx = self._context(rt, snapshot, account, rnd, progress, sessions_left,
                                final_session)

            if rolled and today not in rt.session_seen:
                rt.session_seen.add(today)
                self._safe_hook(rt, "on_session_start", ctx)

            was_halted = rt.risk.is_halted
            halt = self.guardrails.check_kill_switch(rt.risk, account, today=today)
            if halt:
                # Record the transition, not the state. Gating this on a
                # session roll (as an earlier version did) meant the tick the
                # switch actually tripped on was never written down -- the one
                # event a post-mortem would most want to find.
                if not was_halted:
                    self.ledger.record_event(
                        "kill_switch", halt, team_key=rt.key, ts=now,
                        data={"tick": rt.tick_count, "equity": account.equity,
                              "session_open_equity": rt.risk.session_open_equity},
                    )
                try:
                    rt.broker.cancel_all()
                    flattened = rt.broker.close_all_positions()
                    if flattened:
                        rt.orders.extend(flattened)
                        self.ledger.record_orders(rt.key, flattened)
                except BrokerError as e:
                    log.error("%s: could not flatten after the kill switch: %s", rt.key, e)
                self._collect_fills(rt, ctx)
                continue

            intents, elapsed = self._call_strategy(rt, ctx)
            cancelled = self._drain_cancels(rt, ctx)

            if intents:
                # Re-read the book if anything was cancelled: `ctx.open_orders`
                # was captured before `on_tick` ran, so a strategy that pulls
                # a resting offer and then submits a market exit in the same
                # tick would otherwise be blocked by its own dead order still
                # counting against the position it is trying to sell.
                live_orders = ctx.open_orders
                if cancelled:
                    try:
                        live_orders = tuple(rt.broker.open_orders())
                    except BrokerError as e:
                        log.debug("%s: re-reading open orders failed: %s", rt.key, e)
                validated = self.guardrails.validate(
                    intents, state=rt.risk, account=account, snapshot=ctx.snapshot,
                    universe=rt.universe, open_orders=live_orders, now=now,
                )
                if validated.rejected:
                    self.ledger.record_rejections(rt.key, now, validated.rejected)
                if validated.accepted:
                    self._submit(rt, validated.accepted, now)
                    acted += 1
            self._collect_fills(rt, ctx)

        self._snapshot_equity(now, force=False)
        return acted

    def _safe_hook(self, rt: TeamRuntime, name: str, ctx: StrategyContext) -> None:
        hook = getattr(rt.strategy, name, None)
        if hook is None:
            return
        try:
            hook(ctx)
        except Exception as e:  # noqa: BLE001
            log.warning("%s: %s raised %s", rt.key, name, e)
            rt.errors.append(f"{name}: {e}")

    def _snapshot_equity(self, now: datetime, *, force: bool) -> None:
        if (
            not force
            and self._last_equity_snapshot is not None
            and (now - self._last_equity_snapshot).total_seconds()
            < self.equity_snapshot_seconds
        ):
            return
        self._last_equity_snapshot = now
        for rt in self.teams.values():
            try:
                account = rt.broker.account()
            except BrokerError:
                continue
            rt.equity_curve.append((now, account.equity))
            self.ledger.record_equity(rt.key, now, account)

    # ------------------------------------------------------------------ #
    # round lifecycle
    # ------------------------------------------------------------------ #

    def start_round(self, rnd: RoundConfig) -> None:
        for rt in self.teams.values():
            try:
                account = rt.broker.account()
            except BrokerError:
                continue
            ctx = self._context(rt, self.feed.snapshot(list(rt.universe) or ["AAPL"],
                                                       self.clock()),
                                account, rnd, 0.0, 0, False)
            self._safe_hook(rt, "on_round_start", ctx)

    def finish_round(
        self,
        rnd: RoundConfig,
        *,
        started_at: datetime,
        ticks: int,
        liquidate: bool | None = None,
    ) -> RoundResult:
        """Flatten, score and record the round."""
        now = self.clock()
        do_liquidate = self.cfg.risk.liquidate_at_round_end if liquidate is None else liquidate
        if do_liquidate:
            for rt in self.teams.values():
                try:
                    rt.broker.cancel_all()
                    closed = rt.broker.close_all_positions()
                    if closed:
                        rt.orders.extend(closed)
                        self.ledger.record_orders(rt.key, closed)
                except BrokerError as e:
                    log.error("%s: round-end liquidation failed: %s", rt.key, e)
            # Let the simulator match the closing orders before marking.
            for rt in self.teams.values():
                with contextlib.suppress(BrokerError):
                    rt.broker.sync(now)
                account = rt.broker.account()
                ctx = self._context(
                    rt, self.feed.snapshot(list(rt.universe) or ["AAPL"], now),
                    account, rnd, 1.0, 0, True,
                )
                self._collect_fills(rt, ctx)

        self._snapshot_equity(now, force=True)

        results: list[TeamResult] = []
        for rt in self.teams.values():
            ctx_account = rt.broker.account()
            ctx = self._context(
                rt, self.feed.snapshot(list(rt.universe) or ["AAPL"], now),
                ctx_account, rnd, 1.0, 0, True,
            )
            self._safe_hook(rt, "on_round_end", ctx)

            start = rt.baseline_equity or self.cfg.starting_cash
            end = ctx_account.equity
            curve = [e for _ts, e in rt.equity_curve] or [start, end]
            rets = ind.pct_change(curve)
            results.append(TeamResult(
                team_key=rt.key,
                name=rt.config.name,
                scored=rt.config.scored,
                start_equity=start,
                end_equity=end,
                return_pct=(end / start - 1.0) if start > 0 else 0.0,
                fills=len(rt.fills),
                traded_notional=sum(f.notional for f in rt.fills),
                max_drawdown=ind.max_drawdown(curve),
                sharpe=ind.sharpe(rets, periods_per_year=252 * 78),
                rejections=dict(rt.risk.rejections),
                errors=len(rt.errors),
                timeouts=rt.timeouts,
                universe=rt.universe,
            ))

        self._feedback_to_pickers(results)
        self.save_learned_state(rnd.id)

        draft_payload = self.resolver.draft.to_dict() if self.resolver.draft else None
        return RoundResult(
            round_id=rnd.id,
            round_name=rnd.name,
            started_at=started_at,
            finished_at=now,
            mode=self.mode,
            teams=results,
            ticks=ticks,
            draft=draft_payload,
        )

    def _feedback_to_pickers(self, results: Sequence[TeamResult]) -> None:
        """Tell each picker how its shortlist did, so the bandit can learn.

        Reward is the team's realised round return attributed to every name it
        actually traded. Crude -- it cannot separate a good pick from good
        trading -- but it is the signal the bandit would get in reality, and
        it is applied identically to every picker (only the bandit uses it).
        """
        for res in results:
            rt = self.teams.get(res.team_key)
            if rt is None:
                continue
            traded = {f.symbol for f in rt.fills}
            if not traded:
                continue
            try:
                self.resolver.picker_for(rt.config).on_fills(
                    {sym: res.return_pct for sym in traded}
                )
            except Exception as e:  # noqa: BLE001
                log.debug("%s: picker feedback failed: %s", rt.key, e)

    # ------------------------------------------------------------------ #
    # drivers
    # ------------------------------------------------------------------ #

    def run_replay(
        self,
        rnd: RoundConfig,
        *,
        start: date,
        end: date,
        tick_seconds: int | None = None,
        progress_cb: Callable[[int, int, datetime], None] | None = None,
    ) -> RoundResult:
        """Replay a historical window tick by tick. No network, no waiting."""
        cal = self.feed.calendar
        cadence = tick_seconds or min(
            (rt.strategy.tick_seconds for rt in self.teams.values()), default=300
        )
        stamps = cal.tick_times(start, end, seconds=cadence)
        if not stamps:
            raise ValueError(f"no trading sessions between {start} and {end}")
        sessions = cal.trading_days(start, end)
        started_at = stamps[0]

        self.prepare_round(rnd, session=sessions[0])
        self.start_round(rnd)

        current_session = sessions[0]
        total = len(stamps)
        for i, ts in enumerate(stamps):
            snap_session = cal.session(ts)
            if snap_session.session_date != current_session:
                current_session = snap_session.session_date
                if rnd.picker_enabled and rnd.picker.refresh == "daily":
                    self.refresh_universes(rnd, session=current_session)
                for rt in self.teams.values():
                    eod = getattr(rt.broker, "end_of_day", None)
                    if callable(eod):
                        eod()
            remaining = sum(1 for d in sessions if d > snap_session.session_date)
            self.tick(
                rnd, now=ts,
                progress=i / max(total - 1, 1),
                sessions_left=remaining,
                final_session=(remaining == 0),
            )
            if progress_cb and (i % 100 == 0 or i == total - 1):
                progress_cb(i + 1, total, ts)
        return self.finish_round(rnd, started_at=started_at, ticks=total)

    def run_live(
        self,
        rnd: RoundConfig,
        *,
        start: date,
        end: date,
        poll_seconds: int = 15,
        max_ticks: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
        stop: Callable[[], bool] | None = None,
    ) -> RoundResult:
        """Run against live Alpaca paper accounts until the window closes."""
        cal = self.feed.calendar
        sessions = cal.trading_days(start, end)
        if not sessions:
            raise ValueError(f"no trading sessions between {start} and {end}")
        started_at = self.clock()
        self.prepare_round(rnd, session=sessions[0])
        self.start_round(rnd)
        log.info(
            "round %d live: %s (%d sessions), %d teams",
            rnd.id, cal.describe_window(start, end), len(sessions), len(self.teams),
        )

        ticks = 0
        current_session = None
        last_session_seen = None
        while True:
            if stop is not None and stop():
                log.info("round %d: stop requested", rnd.id)
                break
            if max_ticks is not None and ticks >= max_ticks:
                break
            now = self.clock()
            session = cal.session(now)
            today = session.session_date

            if today > end:
                log.info("round %d: window closed", rnd.id)
                break

            if session.is_trading_day and today != current_session:
                current_session = today
                if today in sessions and rnd.picker_enabled and rnd.picker.refresh == "daily":
                    log.info("round %d: new session %s, refreshing universes", rnd.id, today)
                    self.refresh_universes(rnd, session=today)

            if not session.is_open:
                if last_session_seen is not None and last_session_seen != today:
                    for rt in self.teams.values():
                        account = rt.broker.account()
                        ctx = self._context(
                            rt, self.feed.snapshot(list(rt.universe) or ["AAPL"], now),
                            account, rnd, 1.0, 0, False,
                        )
                        self._safe_hook(rt, "on_session_end", ctx)
                        self._drain_cancels(rt, ctx)
                    last_session_seen = today
                wait = min(max(session.seconds_to_open, poll_seconds), 900.0)
                self._snapshot_equity(now, force=False)
                log.debug("market closed; sleeping %.0fs", wait)
                sleep(wait)
                continue

            last_session_seen = today
            remaining = sum(1 for d in sessions if d > today)
            done = sum(1 for d in sessions if d < today)
            progress = (
                (done + session.minutes_since_open / max(
                    session.minutes_since_open + session.minutes_to_close, 1.0))
                / max(len(sessions), 1)
            )
            final_day = remaining == 0

            # Flatten before the final close so the round is scored on cash.
            if (
                final_day
                and self.cfg.risk.liquidate_at_round_end
                and session.within_minutes_of_close(
                    self.cfg.risk.liquidate_minutes_before_close)
            ):
                log.info("round %d: final close approaching, liquidating", rnd.id)
                break

            self.tick(rnd, now=now, progress=min(progress, 1.0),
                      sessions_left=remaining, final_session=final_day)
            ticks += 1
            sleep(poll_seconds)

        return self.finish_round(rnd, started_at=started_at, ticks=ticks)
