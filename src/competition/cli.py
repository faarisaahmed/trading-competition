"""`comp` -- the command line for running the competition.

    comp doctor                     check config, credentials and accounts
    comp teams                      describe the field
    comp universe                   audit the Round 3 ranked pool
    comp refresh-universe           re-price and re-rank the pool
    comp draft --round 3            deal the hands (and prove they are fair)
    comp verify-draft               re-check a recorded draft from the ledger
    comp pretrain                   train the Q-learner offline on history
    comp backtest --round 1         full dry run, no keys needed
    comp run --round 1              run live against Alpaca paper accounts
    comp score --round 1            score a completed round
    comp leaderboard                season standings
    comp report                     per-team detail from the ledger
    comp explain-news "headline"    show the sentiment scorer's working
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path

from . import __version__
from .broker import AlpacaBroker, AlpacaClient, AlpacaCredentials, AlpacaDataReader
from .broker.base import Broker, BrokerError
from .broker.shared import SharedAccount
from .broker.simulated import SimConfig, SimulatedBroker
from .config import REPO_ROOT, CompetitionConfig, ConfigError, load_config, load_dotenv
from .data.calendar import MarketCalendar
from .data.feed import AlpacaFeed, ReplayFeed
from .data.universe import UniverseProvider, UniverseSnapshot
from .draft import DraftError
from .draft import verify as verify_draft
from .engine import CompetitionEngine, Ledger, UniverseResolver
from .engine.checkpoint import load_round_checkpoint
from .engine.runner import RoundSuspended
from .pickers import load_picker
from .scoring import build_leaderboard, score_round
from .setup import (
    Pair,
    SetupError,
    assign,
    parse_pairs,
    render_env,
    render_template,
    report,
    targets_for,
    write_env,
)

# Both modules export a `verify`: one re-checks a dealt draft, the other
# checks Alpaca credentials. Imported unaliased, the second silently shadows
# the first and `comp draft` calls the wrong function.
from .setup import verify as verify_credentials
from .strategies import load_strategy
from .types import UTC, Bar, utcnow

log = logging.getLogger("competition.cli")

#: `comp run` exited because it ran out of wall-clock time, with the
#: round still in progress. Distinct from success and from failure.
EXIT_SUSPENDED = 75

DEFAULT_LEDGER = REPO_ROOT / "runs" / "competition.sqlite"
DEFAULT_DASHBOARD = REPO_ROOT / "runs" / "dashboard.html"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _setup_logging(verbosity: int, *, quiet: bool = False) -> None:
    level = logging.WARNING if quiet else (
        logging.INFO if verbosity == 0 else logging.DEBUG
    )
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)-34s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    if verbosity < 2:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("competition.alpaca").setLevel(logging.INFO)


def _team_keys(cfg: CompetitionConfig, only: Sequence[str] | None) -> list[str] | None:
    """The --team filter as bare keys, or None when unfiltered."""
    if not only:
        return None
    return [t.key for t in _teams_for(cfg, only)]


def _teams_for(cfg: CompetitionConfig, only: Sequence[str] | None) -> list:
    if not only:
        return list(cfg.teams)
    keys = {k.strip() for part in only for k in part.split(",") if k.strip()}
    unknown = keys - set(cfg.team_keys)
    if unknown:
        raise ConfigError(
            f"unknown team(s): {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(cfg.team_keys)}"
        )
    return [t for t in cfg.teams if t.key in keys]


def _data_reader(cfg: CompetitionConfig) -> AlpacaDataReader | None:
    """The single shared market-data reader, or None if no keys are set."""
    import os

    kid = os.environ.get("ALPACA_DATA_KEY_ID") or ""
    sec = os.environ.get("ALPACA_DATA_SECRET_KEY") or ""
    if not (kid and sec):
        # Fall back to the first team that does have keys -- the data feed is
        # shared, so any working key pair will do.
        for team in cfg.teams:
            k, s = team.credentials()
            if k and s:
                kid, sec = k, s
                log.info("using %s's keys for the shared data feed", team.key)
                break
    if not (kid and sec):
        return None
    creds = AlpacaCredentials.from_env("ALPACA_DATA", feed=cfg.data.feed)
    if not creds.key_id:
        creds = AlpacaCredentials(
            key_id=kid, secret_key=sec, feed=cfg.data.feed,
            trading_url=creds.trading_url, data_url=creds.data_url,
        )
    return AlpacaDataReader(AlpacaClient(creds), feed=cfg.data.feed)


def _round_window(cfg: CompetitionConfig, start: date | None, end: date | None,
                  calendar: MarketCalendar) -> tuple[date, date]:
    if start and end:
        return start, end
    if start:
        return cfg.round_window(0, start)
    # Default: the most recent complete `round_length_days` window that ends
    # on or before yesterday, aligned to start on a Monday where possible.
    today = date.today()
    anchor = calendar.previous_trading_day(today)
    s = anchor - timedelta(days=cfg.round_length_days - 1)
    return s, anchor


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as e:
        raise ConfigError(f"bad date {value!r}; use YYYY-MM-DD") from e


def _provider(cfg: CompetitionConfig, reader, trading_client=None) -> UniverseProvider:
    return UniverseProvider(reader=reader, trading_client=trading_client)


def _attach_dashboard(engine, cfg: CompetitionConfig, args) -> None:
    """Have the engine rewrite the HTML dashboard after each equity snapshot.

    Registered as a callback so the engine never imports the reporting layer,
    and so a rendering failure is logged rather than ending a round.
    """
    target = getattr(args, "dashboard", None)
    if not target:
        return
    from .reporting.dashboard import build_dashboard_data, render_dashboard

    path = Path(target)
    refresh = getattr(args, "dashboard_refresh", 30)

    schedule = getattr(args, "_schedule", None)
    publisher = _publisher(cfg, args)

    def write(eng) -> None:
        data = build_dashboard_data(cfg, engine=eng)
        _stamp_season(data, cfg, schedule)
        html = render_dashboard(cfg, data, refresh=refresh)
        _atomic_write(path, html)
        if publisher is not None:
            publisher(html)

    engine.set_dashboard(write)
    _print(f"  dashboard: {path}  (refreshes every {refresh}s)")


def _atomic_write(path: Path, html: str) -> None:
    """A browser refreshing every 30s will otherwise catch a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(html, encoding="utf-8")
    tmp.replace(path)


def _stamp_season(data, cfg: CompetitionConfig, schedule) -> None:
    """Attach the season clock to a dashboard payload."""
    if schedule is None:
        return
    from .schedule import next_deadline

    state = schedule.state(utcnow())
    label, when = next_deadline(state)
    data.season_phase = state.phase
    data.deadline_label = label
    data.deadline_at = when
    now = utcnow()
    plan = []
    for w in schedule.windows:
        if w.is_past(now):
            st = "done"
        elif w.contains(now):
            st = "live"
        else:
            st = "upcoming"
        plan.append({
            "round_id": w.round_id,
            "name": w.name,
            "window": f"{w.start_date:%a %d %b} - {w.end_date:%a %d %b}",
            "state": st,
        })
    data.season_plan = plan


def _publisher(cfg: CompetitionConfig, args):
    """A rate-limited callable that pushes the page to GitHub Pages."""
    pub = cfg.schedule.publish
    if not pub.enabled or getattr(args, "no_publish", False):
        return None
    from .reporting.publish import PublishError, publish

    repo = Path.cwd()
    state = {"last": 0.0}

    def push(html: str) -> None:
        now = time.monotonic()
        if now - state["last"] < pub.every_seconds:
            return
        state["last"] = now
        try:
            result = publish(html, repo=repo, branch=pub.branch)
            log.info("published dashboard %s (%d bytes)", result.commit,
                     result.bytes_written)
        except PublishError as e:
            # Publishing is cosmetic. A GitHub outage must never stop trading.
            log.warning("could not publish the dashboard: %s", e)

    return push


def _print(text: str) -> None:
    sys.stdout.write(text.rstrip("\n") + "\n")


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #


def cmd_doctor(args, cfg: CompetitionConfig) -> int:
    problems: list[str] = []
    warnings: list[str] = []
    _print(f"comp {__version__}  --  {cfg.name}")
    _print(f"config    : {', '.join(str(p.relative_to(REPO_ROOT)) for p in cfg.source_files)}")
    _print(f"rules hash: {cfg.rules_hash}")
    _print(f"bankroll  : ${cfg.starting_cash:,.2f} per team per round")
    _print("rounds    : " + ", ".join(
        f"{r.id}={r.universe_mode}({r.max_symbols})" for r in cfg.rounds))
    _print(f"teams     : {len(cfg.scored_teams)} scored + "
           f"{len(cfg.teams) - len(cfg.scored_teams)} unscored")

    _print("\n-- strategies and pickers ----------------------------------------")
    for team in cfg.teams:
        try:
            strat = load_strategy(team.strategy)(team)
            picker = load_picker(team.picker)(team)
            _print(f"  OK   {team.key:<14} {strat.__class__.__name__:<14} "
                   f"tick={strat.tick_seconds:>4}s warmup={strat.warmup_bars:>3}  "
                   f"{picker.__class__.__name__}")
        except Exception as e:  # noqa: BLE001
            problems.append(f"{team.key}: {type(e).__name__}: {e}")
            _print(f"  FAIL {team.key:<14} {type(e).__name__}: {e}")

    _print("\n-- market calendar ------------------------------------------------")
    cal = MarketCalendar()
    start, end = _round_window(cfg, _parse_date(args.start), _parse_date(args.end), cal)
    sessions = cal.trading_days(start, end)
    _print(f"  window {start} .. {end}: {cal.describe_window(start, end)}")
    if len(sessions) < 3:
        warnings.append(
            f"only {len(sessions)} trading session(s) in the default window; "
            f"pass --start/--end to pick a full week"
        )

    _print("\n-- Round 3 pool ---------------------------------------------------")
    try:
        snap = UniverseSnapshot.from_csv(REPO_ROOT / "data" / "top500.csv")
        r3 = next((r for r in cfg.rounds if r.universe_mode == "draft"), None)
        _print(f"  {len(snap)} names, as of {snap.as_of}, metric={snap.metric}")
        if r3 and r3.draft:
            need = r3.draft.picks_per_team * len(cfg.scored_teams)
            _print(f"  need {need} unique names for {len(cfg.scored_teams)} hands of "
                   f"{r3.draft.picks_per_team}: "
                   f"{'OK' if len(snap) >= need else 'TOO SMALL'}")
            _print(f"  fair rank sum: {r3.draft.fair_rank_sum} "
                   f"(method={r3.draft.method})")
            if len(snap) < r3.draft.pool_size:
                problems.append(
                    f"pool has {len(snap)} rows but draft.pool_size={r3.draft.pool_size}"
                )
        age_days = (date.today() - date.fromisoformat(snap.as_of)).days if snap.as_of else 999
        if age_days > 45:
            warnings.append(
                f"the ranked pool is {age_days} days old; run "
                f"`comp refresh-universe` before Round 3"
            )
    except (FileNotFoundError, ValueError) as e:
        problems.append(f"Round 3 pool unusable: {e}")

    _print("\n-- credentials ----------------------------------------------------")
    loaded = load_dotenv()
    _print(f"  .env: {'loaded ' + str(len(loaded)) + ' keys' if loaded else 'not found'}")
    have_data = False
    reader = None
    try:
        reader = _data_reader(cfg)
        have_data = reader is not None
    except BrokerError:
        have_data = False
    _print(f"  shared data feed: {'configured' if have_data else 'MISSING'} "
           f"(feed={cfg.data.feed})")
    if not have_data:
        warnings.append(
            "no market-data credentials; `comp backtest --source synthetic` works, "
            "`comp run` does not"
        )

    import os as _os

    from .setup import targets_for

    targets = targets_for(cfg)
    if cfg.accounts.is_shared:
        _print(f"  account mode: shared -- {len(targets)} real account(s) for "
               f"{len(cfg.teams)} teams")
        _print("    teams inside a group share one account, each with its own "
               "virtual book (docs/accounts.md)")
    else:
        _print(f"  account mode: per_team -- {len(targets)} real account(s)")

    missing_targets = []
    for target in targets:
        has = bool(_os.environ.get(f"{target.env_prefix}_KEY_ID")
                   and _os.environ.get(f"{target.env_prefix}_SECRET_KEY"))
        if not has:
            missing_targets.append(target.key)
        _print(f"    {target.key:<14} {target.env_prefix + '_KEY_ID':<26} "
               f"{'keys set' if has else 'no keys':<9} needs ${target.capital:,.2f}")
    if missing_targets:
        warnings.append(
            f"{len(missing_targets)} account(s) without keys "
            f"({', '.join(missing_targets)}); run `comp setup-accounts`"
        )

    if args.check_accounts and not missing_targets:
        _print("\n-- live account check ---------------------------------------------")
        for target in targets:
            try:
                client = AlpacaClient(AlpacaCredentials.from_env(target.env_prefix))
                info = client.verify()
            except BrokerError as e:
                problems.append(f"{target.key}: credential check failed -- {e}")
                _print(f"  {target.key:<14} FAILED: {e}")
                continue
            flags = []
            if not info["paper"]:
                flags.append("LIVE ACCOUNT (!)")
                problems.append(
                    f"{target.key} is pointed at a LIVE account, not paper")
            if info["trading_blocked"]:
                flags.append("trading blocked")
                problems.append(f"{target.key}: trading is blocked")
            equity = info["equity"]
            need = target.capital
            tolerance = max(0.01 * need, 5.0)
            if abs(equity - need) > tolerance:
                flags.append(f"NEEDS ${need:,.2f}")
                problems.append(
                    f"{target.key} holds ${equity:,.2f} but its "
                    f"{len(target.teams)} team(s) need ${need:,.2f} "
                    f"(${cfg.starting_cash:,.2f} each). Every team must start from "
                    f"the same bankroll or the round is not comparable."
                )
            _print(f"  {target.key:<14} #{info['account_number']:<12} "
                   f"equity ${equity:>11,.2f}  needs ${need:>11,.2f}  "
                   f"{' '.join(flags)}")

    _print("\n" + "=" * 68)
    for w in warnings:
        _print(f"  WARN  {w}")
    for p in problems:
        _print(f"  ERROR {p}")
    if not problems:
        _print("  ready to run." if not warnings else "  usable, with warnings above.")
    return 1 if problems else 0


# --------------------------------------------------------------------------- #
# informational commands
# --------------------------------------------------------------------------- #


def cmd_teams(args, cfg: CompetitionConfig) -> int:
    teams = _teams_for(cfg, args.team)
    for team in teams:
        strat = load_strategy(team.strategy)(team)
        picker = load_picker(team.picker)(team)
        _print("=" * 74)
        _print(f"{team.name}  [{team.key}]" + ("" if team.scored else "   (unscored)"))
        if team.tagline:
            _print(f'  "{team.tagline}"')
        if team.philosophy:
            import textwrap
            for line in textwrap.wrap(team.philosophy, 70):
                _print(f"  {line}")
        _print("")
        _print("  STRATEGY")
        for line in strat.describe().splitlines():
            _print("  " + line)
        _print("")
        _print("  PICKER")
        for line in picker.describe().splitlines():
            _print("  " + line)
    _print("=" * 74)
    _print(f"{len(teams)} team(s).")
    return 0


def cmd_universe(args, cfg: CompetitionConfig) -> int:
    reader = None
    trading = None
    if args.live:
        reader = _data_reader(cfg)
        if reader is not None:
            trading = reader.client
    provider = _provider(cfg, reader, trading)
    audit = provider.audit(metric=args.metric, size=args.size)
    _print(json.dumps(audit, indent=2))
    if audit["untradable"]:
        _print(f"\n{len(audit['untradable'])} name(s) flagged untradable; "
               f"`comp draft` drops them and renumbers before dealing.")
    return 0


def cmd_refresh_universe(args, cfg: CompetitionConfig) -> int:
    reader = _data_reader(cfg)
    if reader is None:
        _print("ERROR: refreshing needs market-data credentials (see .env.example).")
        return 1
    provider = _provider(cfg, reader, reader.client)
    before = provider.snapshot()
    refreshed = provider.refresh(metric=args.metric, size=args.size)
    moved = sum(
        1 for r in refreshed.rows
        if (b := before.by_symbol(r.symbol)) and b.rank != r.rank
    )
    _print(f"refreshed {len(refreshed)} names (metric={args.metric}), as of {refreshed.as_of}")
    _print(f"{moved} name(s) changed rank")
    _print(f"top 10: {', '.join(r.symbol for r in refreshed.rows[:10])}")
    _print(f"written to {provider.snapshot_path}")
    return 0


def cmd_lexicon(args, cfg: CompetitionConfig) -> int:
    from .strategies import lexicon

    phrases, words = lexicon.vocabulary_size()
    _print(f"finance sentiment lexicon: {phrases} phrases, {words} words, "
           f"{len(lexicon.NEGATORS)} negators, {len(lexicon.INTENSIFIERS)} intensifiers, "
           f"{len(lexicon.DIMINISHERS)} hedges")
    _print(f"negation window {lexicon.NEGATION_WINDOW} tokens back, "
           f"{lexicon.FORWARD_MODIFIER_WINDOW} forward; "
           f"negation factor {lexicon.NEGATION_FACTOR}")
    return 0


def cmd_explain_news(args, cfg: CompetitionConfig) -> int:
    from .strategies import lexicon

    for text in args.text:
        result = lexicon.explain(text)
        _print(f"\n{text}")
        _print(f"  score: {result['score']:+.4f}  ({result['n_hits']} term(s)"
               f"{', ALLCAPS' if result['allcaps'] else ''})")
        for term, value in result["terms"]:
            _print(f"    {value:+.4f}  {term}")
    return 0


# --------------------------------------------------------------------------- #
# draft
# --------------------------------------------------------------------------- #


def cmd_draft(args, cfg: CompetitionConfig) -> int:
    rnd = cfg.round(args.round)
    if rnd.draft is None:
        _print(f"Round {rnd.id} ({rnd.universe_mode}) has no draft.")
        return 1
    reader = _data_reader(cfg) if args.live else None
    trading = reader.client if reader is not None else None
    provider = _provider(cfg, reader, trading)
    resolver = UniverseResolver(cfg, provider=provider)
    teams = _teams_for(cfg, args.team) or list(cfg.teams)
    seed = args.seed if args.seed is not None else cfg.fairness.competition_seed + rnd.id

    try:
        result = resolver.run_draft(
            rnd, teams, seed=seed, verify_tradable=bool(args.live)
        )
    except DraftError as e:
        _print(f"DRAFT FAILED: {e}")
        return 1

    _print(result.table())
    if result.notes:
        _print("\nnotes:")
        for n in result.notes:
            _print(f"  - {n}")

    _print("\n-- fairness proof -------------------------------------------------")
    sums = {h.team_key: h.rank_sum for h in result.hands}
    _print(f"  every hand's rank sum : {sorted(set(sums.values()))}  "
           f"(target {result.target_sum})")
    _print(f"  every hand's mean rank: "
           f"{sorted({round(h.mean_rank, 4) for h in result.hands})}")
    all_syms = result.all_symbols
    _print(f"  symbols dealt         : {len(all_syms)}, "
           f"{len(set(all_syms))} unique -> "
           f"{'no overlap' if len(all_syms) == len(set(all_syms)) else 'OVERLAP!'}")
    for h in sorted(result.hands, key=lambda x: x.team_key):
        sectors = ", ".join(f"{k} x{v}" for k, v in sorted(
            h.sector_counts().items(), key=lambda kv: -kv[1]))
        _print(f"  {h.team_key:<14} deciles {sorted(h.deciles(result.pool_size))} | {sectors}")

    problems = verify_draft(
        result, provider.snapshot(metric=rnd.draft.metric).head(result.pool_size),
        min_rank_deciles=rnd.draft.min_rank_deciles,
        max_sector_share=rnd.draft.max_sector_share,
        unique_across_teams=rnd.draft.unique_across_teams,
    )
    _print(f"\n  independent verification: {'CLEAN' if not problems else 'FAILED'}")
    for p in problems:
        _print(f"    - {p}")

    if args.write:
        ledger = Ledger(args.ledger)
        ledger.start_run(
            round_id=rnd.id, mode="draft", broker="none", rules_hash=cfg.rules_hash,
            seed=seed, config={"draft": True},
        )
        for team in teams:
            ledger.register_team(team)
        ledger.record_draft(rnd.id, result)
        ledger.finish_run("draft only")
        ledger.close()
        _print(f"\n  recorded to {args.ledger} (run {ledger.run_id})")
    if args.json:
        Path(args.json).write_text(json.dumps(result.to_dict(), indent=2))
        _print(f"  wrote {args.json}")
    return 1 if problems else 0


def cmd_verify_draft(args, cfg: CompetitionConfig) -> int:
    ledger = Ledger(args.ledger)
    payload = ledger.latest_draft(args.round)
    ledger.close()
    if not payload:
        _print(f"no draft recorded for round {args.round} in {args.ledger}")
        return 1
    hands = payload.get("hands", [])
    target = payload.get("target_sum")
    _print(f"draft for round {args.round}: method={payload.get('method')} "
           f"seed={payload.get('seed')} dealt {payload.get('dealt_at')}")
    ok = True
    seen: dict[str, str] = {}
    for hand in hands:
        total = sum(hand["ranks"])
        flag = "OK" if total == target else f"MISMATCH ({total} != {target})"
        if total != target:
            ok = False
        for sym in hand["symbols"]:
            if sym in seen:
                _print(f"  OVERLAP: {sym} dealt to both {seen[sym]} and {hand['team']}")
                ok = False
            seen[sym] = hand["team"]
        _print(f"  {hand['team']:<14} sum={total:<6} {flag}  {', '.join(hand['symbols'])}")
    _print(f"\n{'VERIFIED' if ok else 'FAILED'}: {len(hands)} hands, "
           f"{len(seen)} unique symbols, target {target}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# pretraining
# --------------------------------------------------------------------------- #


def cmd_pretrain(args, cfg: CompetitionConfig) -> int:
    """Train the Q-learner offline. Allowed and expected: it is the coding stage."""
    team = cfg.team(args.team_key)
    if args.only_picker:
        ledger = Ledger(args.ledger)
        rc = _pretrain_picker(args, cfg, team, ledger)
        ledger.close()
        return rc
    strategy = load_strategy(team.strategy)(team)
    if not hasattr(strategy, "pretrain"):
        _print(f"{team.key} has no pretrain step (only the RL entry does).")
        return 1

    symbols = [s.strip().upper() for s in (args.symbols or "").split(",") if s.strip()]
    if not symbols:
        symbols = list(cfg.round(1).symbols) + ["NVDA", "AMZN", "META", "JPM", "XOM", "KO"]

    end = _parse_date(args.end) or date.today() - timedelta(days=1)
    start = _parse_date(args.start) or end - timedelta(days=args.days)
    cal = MarketCalendar()

    if args.source == "alpaca":
        reader = _data_reader(cfg)
        if reader is None:
            _print("ERROR: --source alpaca needs market-data credentials.")
            return 1
        _print(f"downloading {cfg.data.primary_timeframe} bars for "
               f"{len(symbols)} symbols, {start} .. {end}")
        bars = reader.bars(
            symbols, cfg.data.primary_timeframe, limit=10000,
            start=datetime.combine(start, datetime.min.time(), tzinfo=UTC),
            end=datetime.combine(end, datetime.max.time(), tzinfo=UTC),
        )
    else:
        from .data.synthetic import generate_bars

        _print(f"generating synthetic {cfg.data.primary_timeframe} bars for "
               f"{len(symbols)} symbols, {start} .. {end}")
        bars = generate_bars(
            symbols, start, end, seed=args.seed or cfg.fairness.competition_seed,
            calendar=cal, warmup_sessions=5,
        )

    usable = {s: b for s, b in bars.items() if len(b) > strategy.warmup_bars + 20}
    if not usable:
        _print("ERROR: no symbol had enough bars to train on.")
        return 1
    _print(f"training on {len(usable)} symbols, "
           f"{sum(len(b) for b in usable.values()):,} bars, {args.passes} pass(es)")

    ledger = Ledger(args.ledger)
    if args.resume:
        blob = ledger.load_learned_state(team.key, "strategy")
        if blob:
            strategy.load_state(blob)
            _print(f"resumed from {len(getattr(strategy, 'q', {}))} known states")

    import random as _random

    stats = strategy.pretrain(
        usable, passes=args.passes, rng=_random.Random(args.seed or cfg.fairness.competition_seed)
    )
    _print("\n" + json.dumps(stats, indent=2))
    if hasattr(strategy, "policy_table"):
        _print("\n" + strategy.policy_table(args.top))

    if not args.dry_run:
        ledger.save_learned_state(team.key, "strategy", strategy.state_dict(), round_id=0)
        _print(f"\nsaved learned state to {args.ledger}")
    else:
        _print("\n--dry-run: nothing saved")

    # ---- the picker ------------------------------------------------------ #
    # Round 1 has a fixed universe, so the picker is idle until Round 2 -- but
    # every other team's picker is a deterministic screen that works at full
    # strength from its first tick. Leaving the only *learning* picker cold
    # would handicap it for reasons of implementation rather than philosophy.
    if not args.skip_picker:
        rc = _pretrain_picker(args, cfg, team, ledger)
        if rc != 0:
            ledger.close()
            return rc

    ledger.close()
    return 0


def _pretrain_picker(args, cfg: CompetitionConfig, team, ledger) -> int:
    """Train the team's picker offline, if it has a pretrain step."""
    picker = load_picker(team.picker)(team)
    if not hasattr(picker, "pretrain"):
        return 0

    symbols = [s.strip().upper()
               for s in (args.picker_symbols or "").split(",") if s.strip()]
    if not symbols:
        symbols = _pool_symbols(cfg, args.picker_pool)
    if not symbols:
        _print("\nWARNING: no candidate pool for the picker; skipping it.")
        return 0

    end = _parse_date(args.end) or date.today() - timedelta(days=1)
    start = end - timedelta(days=args.picker_days)
    _print("\n-- picker ---------------------------------------------------")
    _print(f"downloading daily bars for {len(symbols)} symbols, {start} .. {end}")

    if args.source == "alpaca":
        reader = _data_reader(cfg)
        if reader is None:
            _print("ERROR: --source alpaca needs market-data credentials.")
            return 1
        bars = reader.bars(
            symbols, cfg.data.slow_timeframe, limit=2000,
            start=datetime.combine(start, datetime.min.time(), tzinfo=UTC),
            end=datetime.combine(end, datetime.max.time(), tzinfo=UTC),
        )
    else:
        from .data.synthetic import generate_daily_history

        # Daily bars directly: generating intraday and collapsing it would
        # cost a hundred times the work for the same series.
        bars = generate_daily_history(
            symbols, end,
            sessions=max(int(args.picker_days * 5 / 7), 120),
            seed=args.seed or cfg.fairness.competition_seed,
            calendar=MarketCalendar(),
        )

    if args.resume:
        blob = ledger.load_learned_state(team.key, "picker")
        if blob:
            picker.load_state(blob)
            _print("resumed the saved picker model")

    max_symbols = cfg.round(2).picker.max_symbols
    stats = picker.pretrain(bars, max_symbols=max_symbols, passes=args.picker_passes)
    _print(json.dumps(stats, indent=2))

    if args.dry_run:
        _print("--dry-run: picker not saved")
        return 0
    ledger.save_learned_state(team.key, "picker", picker.state_dict(), round_id=0)
    _print(f"saved picker state to {args.ledger}")
    return 0


def _pool_symbols(cfg: CompetitionConfig, limit: int) -> list[str]:
    """The most valuable names from the Round 3 pool file, as a training pool.

    A broad cross-section is what the bandit needs: it learns which *feature
    profile* pays, and that is only visible across many names at once.
    """
    path = REPO_ROOT / "data" / "top500.csv"
    if not path.exists():
        return []
    out: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("rank,"):
            continue
        parts = line.split(",")
        if len(parts) > 1 and parts[1]:
            out.append(parts[1].strip().upper())
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------- #
# backtest / run
# --------------------------------------------------------------------------- #


def _symbols_for_round(
    cfg: CompetitionConfig,
    rnd,
    provider: UniverseProvider,
    *,
    pool_size: int,
    resolver: UniverseResolver | None = None,
    teams: Sequence = (),
    seed: int | None = None,
) -> list[str]:
    """Every symbol the round could possibly touch, so data can be pre-built.

    For Round 3 the draft is dealt *first* and only the dealt names get data:
    building bars for all 500 pool members when 80 are dealt is a 6x waste,
    and in a synthetic dry run that is the difference between a minute and
    ten. The dealt result is handed to the resolver so the engine reuses the
    same hands rather than re-dealing.
    """
    if rnd.universe_mode == "fixed":
        return list(rnd.symbols)
    snap = provider.snapshot()
    if rnd.universe_mode == "draft":
        if resolver is not None and teams:
            draft = resolver.run_draft(
                rnd, list(teams),
                seed=seed if seed is not None
                else cfg.fairness.competition_seed + rnd.id,
                verify_tradable=False,
            )
            return sorted(set(draft.all_symbols))
        return list(snap.head(min(rnd.draft.pool_size, len(snap))).symbols)
    return list(snap.head(min(pool_size, len(snap))).symbols)


def cmd_backtest(args, cfg: CompetitionConfig) -> int:
    rnd = cfg.round(args.round)
    cal = MarketCalendar()
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if not start:
        start, end = _round_window(cfg, None, None, cal)
    elif not end:
        start, end = cfg.round_window(rnd.id, start)
    sessions = cal.trading_days(start, end)
    if not sessions:
        _print(f"ERROR: no trading sessions between {start} and {end}.")
        return 1

    teams = _teams_for(cfg, args.team)
    reader = _data_reader(cfg) if args.source == "alpaca" else None
    if args.source == "alpaca" and reader is None:
        _print("ERROR: --source alpaca needs market-data credentials. "
               "Use --source synthetic for a no-key dry run.")
        return 1
    provider = _provider(cfg, reader, reader.client if reader else None)

    # Built here (rather than after the data) so a draft round can deal first
    # and limit data generation to the dealt names.
    resolver = UniverseResolver(cfg, provider=provider, pool_size_override=args.pool_size)
    symbols = _symbols_for_round(
        cfg, rnd, provider, pool_size=args.pool_size,
        resolver=resolver, teams=teams, seed=args.seed,
    )
    _print(f"Round {rnd.id} ({rnd.name}) backtest")
    _print(f"  window   : {start} .. {end}  ({len(sessions)} sessions)")
    _print(f"  teams    : {', '.join(t.key for t in teams)}")
    _print(f"  symbols  : {len(symbols)} ({args.source})")
    _print(f"  bankroll : ${cfg.starting_cash:,.2f} each")

    # ---- build the replay feed --------------------------------------------
    if args.source == "alpaca":
        lookback = datetime.combine(
            cal.previous_trading_day(start), datetime.min.time(), tzinfo=UTC
        ) - timedelta(days=args.history_days)
        feed = ReplayFeed.from_alpaca(
            reader, symbols, lookback,
            datetime.combine(end, datetime.max.time(), tzinfo=UTC),
            primary_timeframe=cfg.data.primary_timeframe,
            fast_timeframe=cfg.data.fast_timeframe,
            calendar=cal, with_news=True,
            history_bars=cfg.data.history_bars_primary,
            spread_bps=args.spread_bps,
        )
        primary = {s: list(feed._store["primary"].get(s, [])) for s in symbols}
        daily_store = {s: tuple(feed._store["daily"].get(s, ())) for s in symbols}
        news_store = {s: tuple(feed._news.get(s, ())) for s in symbols}
    else:
        from .data.synthetic import (
            build_specs,
            generate_bars,
            generate_daily_history,
            generate_news,
            to_daily,
        )

        seed = args.seed if args.seed is not None else cfg.fairness.competition_seed
        snap = provider.snapshot()
        prices = {r.symbol: r.share_price for r in snap.rows if r.share_price > 0}
        specs = build_specs(symbols, seed=seed, prices=prices)
        primary = generate_bars(
            symbols, start, end, timeframe_seconds=300, seed=seed, calendar=cal,
            specs=specs, prices=prices, warmup_sessions=args.warmup_sessions,
        )
        window_daily = to_daily(primary)
        # The pickers screen on daily history (up to 180 days for pair
        # formation). Generating that as intraday bars would be ~1.4M bars for
        # a 60-name pool, so the dry run mirrors live mode instead: a long
        # daily series spliced onto the window, and a short intraday series.
        first_intraday = min(
            (rows[0].ts.date() for rows in primary.values() if rows), default=start
        )
        anchors = {
            s: rows[0].open for s, rows in window_daily.items() if rows
        }
        history = generate_daily_history(
            symbols, first_intraday, sessions=args.daily_history,
            seed=seed, calendar=cal, specs=specs, anchor_prices=anchors,
        )
        daily = {
            s: list(history.get(s, [])) + list(window_daily.get(s, []))
            for s in symbols
        }
        news = generate_news(primary, seed=seed)
        _print(f"  daily    : {args.daily_history} sessions of history + "
               f"{len(window_daily.get(symbols[0], []))} in-window, spliced")
        feed = ReplayFeed(
            primary, fast=primary, daily=daily, news=news, calendar=cal,
            primary_timeframe=cfg.data.primary_timeframe,
            fast_timeframe=cfg.data.primary_timeframe,
            spread_bps=args.spread_bps,
            history_bars=cfg.data.history_bars_primary,
        )
        daily_store = {s: tuple(v) for s, v in daily.items()}
        news_store = {s: tuple(v) for s, v in news.items()}

    have = sum(1 for s in symbols if primary.get(s))
    _print(f"  bars     : {sum(len(v) for v in primary.values()):,} "
           f"across {have}/{len(symbols)} symbols")
    if have == 0:
        _print("ERROR: no bars available for that window.")
        return 1

    # ---- a shared clock so brokers and strategies see the same instant ----
    clock_holder = {"now": datetime.combine(start, datetime.min.time(), tzinfo=UTC)}

    def clock() -> datetime:
        return clock_holder["now"]

    quote_source = feed.quote_source(clock)
    sim_cfg = SimConfig(
        slippage_bps=cfg.fairness.sim_slippage_bps,
        commission_per_share=cfg.fairness.sim_commission_per_share,
        fractional=True,
        allow_short=cfg.risk.allow_short,
    )

    def sim_price(symbol: str) -> float:
        quote = quote_source(symbol)
        return quote.mid if quote is not None else 0.0

    shared_accounts: list[SharedAccount] = []
    brokers: dict[str, Broker] = {}
    if cfg.accounts.is_shared:
        # Mirror the live topology, so the dry run exercises the same broker
        # layer -- internal crossing included -- that the round will use.
        for group in cfg.accounts.groups:
            members = [t.key for t in teams if t.key in group.teams]
            if not members:
                continue
            pooled = SimulatedBroker(
                cfg.starting_cash * len(members), quote_source,
                name=f"sim:{group.label}", config=sim_cfg, clock=clock,
            )
            shared = SharedAccount(
                pooled, members, bankroll=cfg.starting_cash,
                price_of=sim_price, quote_of=quote_source,
                cross_internally=cfg.accounts.cross_internally,
                name=group.label,
            )
            shared_accounts.append(shared)
            for key in members:
                brokers[key] = shared.virtual_broker(key)
        _print(f"  accounts : {len(shared_accounts)} shared group(s), "
               f"{cfg.starting_cash * 3:,.0f} each, internal crossing "
               f"{'on' if cfg.accounts.cross_internally else 'off'}")
    else:
        brokers = {
            t.key: SimulatedBroker(cfg.starting_cash, quote_source,
                                   name=f"sim:{t.key}", config=sim_cfg, clock=clock)
            for t in teams
        }

    ledger = Ledger(args.ledger)
    ledger.start_run(
        round_id=rnd.id, mode="backtest", broker=f"sim/{args.source}",
        rules_hash=cfg.rules_hash,
        seed=args.seed if args.seed is not None else cfg.fairness.competition_seed,
        config={"start": str(start), "end": str(end), "symbols": len(symbols),
                "source": args.source},
        notes=args.notes,
    )

    resolver._daily_bars = lambda syms: {s: daily_store.get(s, ()) for s in syms}
    resolver._news = lambda syms: {s: news_store.get(s, ()) for s in syms}

    engine = CompetitionEngine(
        cfg, feed=feed, brokers=brokers, ledger=ledger, resolver=resolver,
        mode="backtest", teams=teams, clock=clock,
        equity_snapshot_seconds=args.equity_every,
    )
    if shared_accounts:
        engine.shared_accounts = list(shared_accounts)
    if args.resume_learning:
        engine.load_learned_state()
    _attach_dashboard(engine, cfg, args)

    # The engine drives `now`; keep the shared clock in step with it.
    original_tick = engine.tick

    def tick(rnd_, *, now=None, **kw):
        if now is not None:
            clock_holder["now"] = now
        return original_tick(rnd_, now=now, **kw)

    engine.tick = tick  # type: ignore[method-assign]

    def progress(i: int, total: int, ts: datetime) -> None:
        if not args.quiet:
            pct = 100.0 * i / max(total, 1)
            sys.stderr.write(f"\r  {pct:5.1f}%  {ts:%Y-%m-%d %H:%M}  ({i}/{total} ticks)")
            sys.stderr.flush()

    result = engine.run_replay(
        rnd, start=start, end=end, tick_seconds=args.tick_seconds,
        progress_cb=progress,
    )
    if not args.quiet:
        sys.stderr.write("\n")

    scored = score_round(cfg, rnd.id, result.teams, round_name=rnd.name)
    for s in scored.scores:
        ledger.record_result(
            rnd.id, s.team_key, start_equity=s.start_equity, end_equity=s.end_equity,
            return_pct=s.return_pct, place=s.place, points=s.points, scored=s.scored,
            metrics=s.metrics,
        )
    ledger.finish_run(f"backtest round {rnd.id}")

    _print("\n" + scored.table())
    _print("\n-- activity -------------------------------------------------------")
    for tr in sorted(result.teams, key=lambda t: t.team_key):
        rej = ", ".join(f"{k}:{v}" for k, v in
                        sorted(tr.rejections.items(), key=lambda kv: -kv[1])[:4])
        _print(f"  {tr.team_key:<14} fills={tr.fills:<5} "
               f"traded=${tr.traded_notional:>11,.0f} maxDD={tr.max_drawdown:>7.2%} "
               f"errors={tr.errors} timeouts={tr.timeouts}"
               + (f"  [{rej}]" if rej else ""))
    if result.draft:
        _print("\n-- Round 3 hands --------------------------------------------------")
        for hand in result.draft["hands"]:
            _print(f"  {hand['team']:<14} sum={hand['rank_sum']} "
                   f"{', '.join(hand['symbols'])}")
    if shared_accounts:
        _print("\n-- shared accounts -----------------------------------------------")
        for shared in shared_accounts:
            report = shared.reconcile()
            summary = shared.summary()
            _print(f"  {shared.name:<9} {report.describe()}")
            _print(f"  {'':<9} crossed ${summary['crossed_notional']:,.0f} "
                   f"internally, {summary['forced_cancels']} forced cancel(s), "
                   f"{summary['wash_rejections']} wash rejection(s)")
            for key, book in summary["books"].items():
                _print(f"  {'':<9}   {key:<14} crossed "
                       f"${book['crossed_notional']:>10,.0f}")
    if args.dashboard:
        from .reporting import write_dashboard
        out = write_dashboard(cfg, args.dashboard, engine=engine,
                              refresh=args.dashboard_refresh)
        _print(f"dashboard: {out}")
    _print(f"\nledger: {args.ledger} (run {ledger.run_id})")
    ledger.close()
    return 0


def _funded_teams(cfg: CompetitionConfig, teams: Sequence) -> set[str]:
    """Which of `teams` can actually trade, given where the keys live.

    In `per_team` mode a team's credentials are its own. In `shared` mode they
    belong to the account group it sits in, so asking the team directly --
    which is what this used to do -- reports every team unfunded even when all
    three accounts are set up correctly.
    """
    if not cfg.accounts.is_shared:
        return {t.key for t in teams if t.has_credentials}
    wanted = {t.key for t in teams}
    out: set[str] = set()
    for group in cfg.accounts.groups:
        kid = os.environ.get(f"{group.env_prefix}_KEY_ID")
        sec = os.environ.get(f"{group.env_prefix}_SECRET_KEY")
        if kid and sec:
            out |= (set(group.teams) & wanted)
    return out


def cmd_run(args, cfg: CompetitionConfig) -> int:
    rnd = cfg.round(args.round)
    reader = _data_reader(cfg)
    if reader is None:
        _print("ERROR: live running needs market-data credentials. See .env.example.")
        return 1
    cal = MarketCalendar.from_alpaca(reader.client)
    start = _parse_date(args.start) or cal.next_trading_day(date.today(), inclusive=True)
    _, end = cfg.round_window(rnd.id, start)
    if args.end:
        end = _parse_date(args.end)
    sessions = cal.trading_days(start, end)

    teams = _teams_for(cfg, args.team)
    funded = _funded_teams(cfg, teams)
    missing = [t.key for t in teams if t.key not in funded]
    if missing and not args.allow_missing:
        _print(f"ERROR: no Alpaca credentials for: {', '.join(missing)}.")
        _print("Add them to .env, or pass --allow-missing to run only the funded teams,")
        _print("or use `comp backtest` for a no-key dry run.")
        return 1
    teams = [t for t in teams if t.key in funded]
    if not teams:
        _print("ERROR: no team has credentials.")
        return 1

    _print(f"Round {rnd.id} ({rnd.name}) LIVE")
    _print(f"  window  : {start} .. {end}  ({len(sessions)} sessions)")
    _print(f"  teams   : {', '.join(t.key for t in teams)}")
    _print(f"  bankroll: ${cfg.starting_cash:,.2f} each")
    _print(f"  rules   : {cfg.rules_hash}")

    shared_accounts: list[SharedAccount] = []
    brokers: dict[str, Broker] = {}
    price_holder: dict[str, Callable[[str], float]] = {"fn": lambda _s: 0.0}
    # Order sizing must use the actual touch, not the mid, or every book
    # overspends by the half spread. Both are late-bound to the shared feed.
    quote_holder: dict[str, Callable[[str], object]] = {"fn": lambda _s: None}

    if cfg.accounts.is_shared:
        # One real account per group, partitioned into per-team virtual books.
        # The marking price source is late-bound: the shared feed does not
        # exist yet, and is wired in below.
        def price_of(symbol: str) -> float:
            return price_holder["fn"](symbol)

        for group in cfg.accounts.groups:
            members = [t.key for t in teams if t.key in group.teams]
            if not members:
                continue
            try:
                real = AlpacaBroker.from_env(group.env_prefix, name=group.label)
            except BrokerError as e:
                _print(f"ERROR: account group {group.label}: {e}")
                return 1
            shared = SharedAccount(
                real, members, bankroll=cfg.starting_cash, price_of=price_of,
                quote_of=quote_holder["fn"],
                cross_internally=cfg.accounts.cross_internally,
                name=group.label,
            )
            shared_accounts.append(shared)
            for key in members:
                brokers[key] = shared.virtual_broker(key)
        _print(f"  accounts: {len(shared_accounts)} shared "
               f"({', '.join(s.name for s in shared_accounts)}), "
               f"internal crossing "
               f"{'on' if cfg.accounts.cross_internally else 'OFF'}")
    else:
        for team in teams:
            try:
                brokers[team.key] = AlpacaBroker.from_env(team.env_prefix,
                                                          name=team.key)
            except BrokerError as e:
                _print(f"ERROR: {team.key}: {e}")
                return 1

    # Refuse to start with unequal bankrolls -- that is the one condition that
    # invalidates the whole round.
    if shared_accounts:
        # Each shared account must hold its group's whole capital, or the
        # teams inside it cannot all be funded to the same bankroll.
        for shared in shared_accounts:
            need = shared.required_capital()
            try:
                have = shared.broker.account(fresh=True).equity
            except BrokerError as e:
                _print(f"ERROR: {shared.name}: cannot read the account -- {e}")
                return 1
            _print(f"  {shared.name:<9} ${have:>12,.2f} held, ${need:>12,.2f} "
                   f"needed for {len(shared.books)} team(s)")
            if abs(have - need) > max(0.01 * need, 5.0) and not args.force:
                _print(
                    f"\nERROR: {shared.name} holds ${have:,.2f} but its "
                    f"{len(shared.books)} teams need ${need:,.2f} "
                    f"(${cfg.starting_cash:,.2f} each). Recreate that paper "
                    f"account with ${need:,.2f} of funds, or pass --force."
                )
                return 1
    else:
        equities = {}
        for key, broker in brokers.items():
            try:
                equities[key] = broker.account(fresh=True).equity
            except BrokerError as e:
                _print(f"ERROR: {key}: cannot read the account -- {e}")
                return 1
        spread = (max(equities.values()) - min(equities.values())
                  if equities else 0.0)
        tolerance = max(0.01 * cfg.starting_cash, 5.0)
        _print("  equity  : " + ", ".join(f"{k}=${v:,.2f}"
                                          for k, v in equities.items()))
        if spread > tolerance and not args.force:
            _print(f"\nERROR: account equities differ by ${spread:,.2f} "
                   f"(tolerance ${tolerance:,.2f}). Reset the paper accounts so "
                   f"every team starts from the same bankroll, or pass --force.")
            return 1

    # ---- is there an interrupted round to rejoin? ----------------------- #
    probe = Ledger(args.ledger)
    checkpoint = load_round_checkpoint(probe.resumable_round(rnd.id))
    probe.close()
    resume = None
    if checkpoint is not None and not args.fresh:
        ok, why = checkpoint.compatible_with(
            round_id=rnd.id, start=start, end=end, rules_hash=cfg.rules_hash,
            team_keys={t.key for t in teams},
        )
        if ok:
            resume = checkpoint
            _print(f"\n  RESUMING round {rnd.id} from run {checkpoint.run_id}: "
                   f"{checkpoint.ticks} ticks already done, last checkpoint "
                   f"{checkpoint.updated_at:%Y-%m-%d %H:%M} UTC.")
            _print("  Accounts will NOT be reset; positions and strategy state "
                   "are restored from the checkpoint.")
        else:
            _print(f"\n  Found an in-progress round {rnd.id} but cannot resume "
                   f"it: {why}.")
            _print("  Pass --fresh to abandon it and restart the round from "
                   "scratch.")
            return 1
    elif checkpoint is not None and args.fresh:
        _print(f"\n  --fresh: abandoning the in-progress round {rnd.id} "
               f"({checkpoint.ticks} ticks) and restarting from scratch.")

    if args.dry_run:
        _print("\n--dry-run: setup validated, not trading.")
        if resume is not None:
            _print("             (a resume was detected and would be used)")
        return 0

    provider = _provider(cfg, reader, reader.client)
    feed = AlpacaFeed(
        reader, calendar=cal,
        primary_timeframe=cfg.data.primary_timeframe,
        fast_timeframe=cfg.data.fast_timeframe,
        slow_timeframe=cfg.data.slow_timeframe,
        history_bars_primary=cfg.data.history_bars_primary,
        history_bars_fast=cfg.data.history_bars_fast,
        history_days_slow=cfg.data.history_days_slow,
        news_lookback_hours=cfg.data.news_lookback_hours,
        news_limit_per_symbol=cfg.data.news_limit_per_symbol,
        quote_max_age_seconds=cfg.risk.max_quote_age_seconds,
    )

    def daily_bars(syms: Sequence[str]) -> Mapping[str, tuple[Bar, ...]]:
        try:
            rows = reader.bars(syms, cfg.data.slow_timeframe,
                               limit=cfg.data.history_days_slow)
        except Exception as e:  # noqa: BLE001
            log.warning("daily bars for the pickers unavailable: %s", e)
            return {}
        return {s: tuple(v) for s, v in rows.items()}

    def news(syms: Sequence[str]) -> Mapping[str, tuple]:
        try:
            items = reader.news(syms, hours=cfg.data.news_lookback_hours,
                                limit=cfg.data.news_limit_per_symbol)
        except Exception as e:  # noqa: BLE001
            log.warning("news for the pickers unavailable: %s", e)
            return {}
        table: dict[str, list] = {}
        wanted = {s.upper() for s in syms}
        for item in items:
            for sym in item.symbols:
                if sym in wanted:
                    table.setdefault(sym, []).append(item)
        return {k: tuple(v) for k, v in table.items()}

    resolver = UniverseResolver(cfg, provider=provider, daily_bars=daily_bars, news=news)
    # Continue the interrupted run rather than starting a parallel one, so the
    # round has a single unbroken history in the ledger.
    ledger = Ledger(args.ledger, run_id=resume.run_id if resume else None)
    ledger.start_run(
        round_id=rnd.id, mode="live", broker="alpaca", rules_hash=cfg.rules_hash,
        seed=cfg.fairness.competition_seed,
        config={"start": str(start), "end": str(end),
                "teams": [t.key for t in teams]},
        notes=args.notes,
    )

    engine = CompetitionEngine(
        cfg, feed=feed, brokers=brokers, ledger=ledger, resolver=resolver,
        mode="live", teams=teams,
        is_tradable=provider.is_tradable, is_fractionable=provider.is_fractionable,
        equity_snapshot_seconds=args.equity_every,
    )
    if shared_accounts:
        # Mark the virtual books off the same feed every strategy reads, so a
        # team's equity is computed from the same prices it trades on.
        last_price: dict[str, float] = {}

        def feed_price(symbol: str) -> float:
            try:
                price = feed.snapshot([symbol]).price(symbol)
            except Exception:  # noqa: BLE001 -- marking must never raise
                price = 0.0
            if price > 0:
                last_price[symbol] = price
            return price or last_price.get(symbol, 0.0)

        def feed_quote(symbol: str):
            try:
                return feed.snapshot([symbol]).quote(symbol)
            except Exception:  # noqa: BLE001 -- sizing must never raise
                return None

        price_holder["fn"] = feed_price
        for shared in shared_accounts:
            shared.quote_of = feed_quote
        engine.shared_accounts = list(shared_accounts)
        if resume is not None:
            # The per-team split lives only in this process, so unlike real
            # accounts it must be restored from the checkpoint or it is lost.
            for shared in shared_accounts:
                saved = ledger.load_learned_state(shared.name, "shared_books")
                if saved:
                    shared.load(saved)
                    _print(f"  restored {shared.name}: {len(shared.books)} "
                           f"virtual book(s) from the checkpoint")

    engine.load_learned_state()
    _attach_dashboard(engine, cfg, args)

    deadline = getattr(args, "_deadline", None)
    suspend = None
    if deadline is not None:
        def suspend() -> bool:            # noqa: E731 -- a named closure reads better
            return time.monotonic() >= deadline

    try:
        result = engine.run_live(
            rnd, start=start, end=end, poll_seconds=args.poll,
            max_ticks=args.max_ticks, resume=resume, suspend=suspend,
        )
    except RoundSuspended as e:
        # Out of wall-clock time, not out of round. Everything needed to
        # resume is on disk; the next process picks it up.
        _print(f"\nsuspended round {e.round_id} after {e.ticks} ticks "
               f"-- checkpointed, not scored.")
        ledger.close()
        return EXIT_SUSPENDED
    except KeyboardInterrupt:
        _print("\ninterrupted -- flattening and scoring what we have.")
        result = engine.finish_round(rnd, started_at=utcnow(), ticks=0)

    scored = score_round(cfg, rnd.id, result.teams, round_name=rnd.name)
    for s in scored.scores:
        ledger.record_result(
            rnd.id, s.team_key, start_equity=s.start_equity, end_equity=s.end_equity,
            return_pct=s.return_pct, place=s.place, points=s.points, scored=s.scored,
            metrics=s.metrics,
        )
    ledger.finish_run(f"live round {rnd.id}")
    if args.dashboard:
        from .reporting import write_dashboard
        write_dashboard(cfg, args.dashboard, engine=engine, refresh=0)
    _print("\n" + scored.table())
    _print(f"\nledger: {args.ledger} (run {ledger.run_id})")
    ledger.close()
    return 0


# --------------------------------------------------------------------------- #
# the season: run all three rounds unattended
# --------------------------------------------------------------------------- #


def _build_schedule(cfg: CompetitionConfig, cal, *, start: date | None = None):
    from .schedule import SeasonSchedule

    first = start or cfg.schedule.start_date
    if first is None:
        first = cal.next_trading_day(date.today(), inclusive=True)
    return SeasonSchedule.build(
        rounds=[(r.id, r.name) for r in cfg.rounds],
        first_start=first,
        sessions_per_round=cfg.schedule.sessions_per_round,
        calendar=cal,
        gap_sessions=cfg.schedule.gap_sessions,
    )


def _idle_dashboard(cfg, args, schedule, *, note: str = "") -> str | None:
    """Render (and publish) the between-rounds page: a clock and standings."""
    from .reporting.dashboard import build_dashboard_data, render_dashboard

    if not getattr(args, "dashboard", None):
        return None
    ledger = Ledger(args.ledger)
    try:
        runs = ledger.runs()
        if runs:
            ledger.run_id = runs[-1]["run_id"]
        data = build_dashboard_data(cfg, ledger=ledger)
    except Exception as e:  # noqa: BLE001 -- an idle page must never crash a season
        log.warning("could not build the idle dashboard: %s", e)
        ledger.close()
        return None
    ledger.close()
    _stamp_season(data, cfg, schedule)
    html = render_dashboard(cfg, data,
                            refresh=getattr(args, "dashboard_refresh", 30))
    _atomic_write(Path(args.dashboard), html)
    return html


def _publish_now(cfg, args, html: str | None) -> None:
    if html is None or not cfg.schedule.publish.enabled:
        return
    if getattr(args, "no_publish", False):
        return
    from .reporting.publish import PublishError, publish

    try:
        result = publish(html, repo=Path.cwd(), branch=cfg.schedule.publish.branch)
        log.info("published %s", result.commit)
    except PublishError as e:
        log.warning("could not publish: %s", e)


def humanise_secs(secs: float) -> str:
    from .schedule import humanise

    return humanise(secs)


def cmd_season(args, cfg: CompetitionConfig) -> int:
    """Run the whole season start to finish with no human in the loop.

    This is the command that is meant to be left running. It waits for each
    round's opening bell, runs it live, lets the engine flatten and score at
    the close, then rolls straight into the next round. Everything it does is
    derived from the schedule and the market calendar, so a restart at any
    point picks up exactly where it left off.
    """
    reader = _data_reader(cfg)
    if reader is None:
        _print("ERROR: a live season needs market-data credentials. See .env.example.")
        return 1
    cal = MarketCalendar.from_alpaca(reader.client)

    try:
        schedule = _build_schedule(cfg, cal, start=_parse_date(args.start))
    except ValueError as e:
        _print(f"ERROR: {e}")
        return 2

    _print(f"{cfg.name}  --  season {cfg.season}")
    for line in schedule.describe():
        _print(f"  {line}")
    _print(f"  bankroll : ${cfg.starting_cash:,.2f} per team per round")
    _print(f"  teams    : {len(cfg.teams)} ({len(cfg.scored_teams)} scored)")
    pub = cfg.schedule.publish
    if pub.enabled and not args.no_publish:
        from .reporting.publish import pages_url
        url = pages_url(Path.cwd(), pub.branch)
        _print(f"  publish  : {url or pub.branch} every {pub.every_seconds}s")
    _print("")

    if args.dry_run:
        state = schedule.state(utcnow())
        from .schedule import humanise, next_deadline
        label, when = next_deadline(state)
        _print(f"phase now: {state.phase}")
        if when:
            secs = (when - utcnow()).total_seconds()
            _print(f"{label}: {humanise(secs)}  ({when:%a %d %b %H:%M} UTC)")
        _print("\n--dry-run: schedule resolved, not trading.")
        return 0

    # The dashboard hook needs the schedule to render the countdown.
    args._schedule = schedule
    done: set[int] = set()
    # A wall-clock budget, for hosts that cap how long a process may run
    # (a CI job, say). Reaching it suspends the round rather than ending it.
    deadline = None
    if getattr(args, "until", None):
        # An ABSOLUTE wall-clock stop, not a relative budget. A job that
        # starts late must still finish on time, or it overlaps the next one
        # and two processes trade the same accounts.
        try:
            hh, mm = (int(x) for x in args.until.split(":"))
            target = utcnow().replace(hour=hh, minute=mm, second=0, microsecond=0)
        except ValueError:
            _print(f"ERROR: --until wants UTC HH:MM, got {args.until!r}")
            return 2
        secs = (target - utcnow()).total_seconds()
        if secs <= 0:
            _print(f"--until {args.until} UTC has already passed; nothing to do.")
            return 0
        deadline = time.monotonic() + secs
        _print(f"  stopping : {args.until} UTC ({humanise_secs(secs)} from now)")
    elif getattr(args, "max_runtime", 0):
        deadline = time.monotonic() + args.max_runtime

    while True:
        now = utcnow()
        state = schedule.state(now)

        if state.is_finished:
            _print("\nSeason complete.")
            _print("")
            rc = cmd_leaderboard(args, cfg)
            _publish_now(cfg, args, _idle_dashboard(cfg, args, schedule))
            return rc

        if state.current is None or state.current.round_id in done:
            # Between rounds: show the countdown and wait for the bell.
            from .schedule import humanise

            wait = state.seconds_to_start
            if wait is None:
                time.sleep(30)
                continue
            nxt = state.next
            _print(f"waiting for round {nxt.round_id} ({nxt.name}): "
                   f"{humanise(wait)} -- opens {nxt.opens_at:%a %d %b %H:%M} UTC")
            _publish_now(cfg, args, _idle_dashboard(cfg, args, schedule))
            if deadline is not None and time.monotonic() >= deadline:
                _print("out of time for this run; nothing is in progress.")
                return 0
            # Wake up often enough to keep the published clock fresh, and to
            # notice a machine that slept through the open.
            time.sleep(min(wait + 1, max(pub.every_seconds, 60)))
            continue

        # A round is live. Hand off to the existing runner, which blocks until
        # the closing bell, flattens every book, scores and records.
        window = state.current
        _print("")
        _print("=" * 68)
        _print(f"ROUND {window.round_id}: {window.name}   "
               f"{window.start_date:%a %d %b} -> {window.end_date:%a %d %b}")
        _print("=" * 68)

        round_args = _RoundArgs(args, round_id=window.round_id,
                                start=window.start_date, end=window.end_date)
        round_args._deadline = deadline
        try:
            rc = cmd_run(round_args, cfg)
        except KeyboardInterrupt:
            _print("\nseason interrupted.")
            return 130
        if rc == EXIT_SUSPENDED:
            _print("out of time for this run; the round is checkpointed and "
                   "will continue when the season next starts.")
            return 0
        if rc != 0:
            _print(f"\nERROR: round {window.round_id} exited with code {rc}; "
                   f"stopping the season. Fix the problem and re-run "
                   f"`comp season` -- it will resume where it left off.")
            return rc

        done.add(window.round_id)
        _publish_now(cfg, args, _idle_dashboard(cfg, args, schedule))
        if not cfg.schedule.auto_advance:
            _print("\nauto_advance is off; stopping after this round.")
            return 0


class _RoundArgs:
    """`cmd_run`'s argument surface, derived from the season's own args.

    A shim rather than a rebuilt parser namespace, so the two commands cannot
    drift apart: anything `run` grows is inherited here automatically.
    """

    def __init__(self, base, *, round_id: int, start: date, end: date):
        self._base = base
        self.round = round_id
        self.start = start.isoformat()
        self.end = end.isoformat()
        # A season never abandons an interrupted round: resuming is the whole
        # point of being restartable.
        self.fresh = False
        self.dry_run = False

    def __getattr__(self, name):
        return getattr(self._base, name)


# --------------------------------------------------------------------------- #
# publish the dashboard on demand
# --------------------------------------------------------------------------- #


def cmd_publish(args, cfg: CompetitionConfig) -> int:
    """Render the dashboard and push it to GitHub Pages, once."""
    from .reporting.dashboard import build_dashboard_data, render_dashboard
    from .reporting.publish import PublishError, publish

    ledger = Ledger(args.ledger)
    runs = ledger.runs()
    if runs:
        ledger.run_id = runs[-1]["run_id"]
    round_id = None
    pending = ledger.in_progress_rounds()
    if pending:
        round_id = int(pending[0]["round_id"])
    else:
        scored = [r.id for r in cfg.rounds if ledger.results_for_round(r.id)]
        round_id = scored[-1] if scored else None

    schedule = None
    with contextlib.suppress(ValueError, KeyError):
        schedule = _build_schedule(cfg, MarketCalendar())
    data = build_dashboard_data(cfg, ledger=ledger, round_id=round_id)
    _stamp_season(data, cfg, schedule)
    html = render_dashboard(cfg, data, refresh=args.refresh)
    ledger.close()

    if args.out:
        _atomic_write(Path(args.out), html)
        _print(f"wrote {args.out}")
    try:
        result = publish(html, repo=REPO_ROOT, branch=cfg.schedule.publish.branch)
    except PublishError as e:
        _print(f"ERROR: {e}")
        return 1
    _print(f"published {result.commit} -> {result.url or cfg.schedule.publish.branch}")
    return 0


# --------------------------------------------------------------------------- #
# ledger state, for hosts with no persistent disk
# --------------------------------------------------------------------------- #

#: What has to survive between runs: the ledger is the competition's memory --
#: results, equity curves, learned state and the Round 3 draft.
STATE_FILES = ("competition.sqlite",)


def cmd_state(args, cfg: CompetitionConfig) -> int:
    """Carry the ledger between runs on an ephemeral host."""
    from .reporting.publish import PublishError, pull_state, push_state

    runs_dir = Path(args.ledger).parent
    branch = args.branch

    if args.action == "push":
        payload: dict[str, bytes] = {}
        for name in STATE_FILES:
            path = runs_dir / name
            if path.exists():
                payload[name] = path.read_bytes()
        if not payload:
            _print("nothing to push: no ledger on disk yet.")
            return 0
        try:
            commit = push_state(payload, repo=REPO_ROOT, branch=branch)
        except PublishError as e:
            _print(f"ERROR: {e}")
            return 1
        total = sum(len(v) for v in payload.values())
        _print(f"pushed {len(payload)} file(s), {total / 1e6:.1f} MB "
               f"to {branch} ({commit})")
        return 0

    # pull
    try:
        restored = pull_state(repo=REPO_ROOT, branch=branch, dest=runs_dir)
    except PublishError as e:
        _print(f"ERROR: {e}")
        return 1
    if not restored:
        _print(f"no {branch} branch yet -- starting from an empty ledger.")
        return 0
    _print(f"restored {', '.join(restored)} from {branch}")
    return 0


# --------------------------------------------------------------------------- #
# background service
# --------------------------------------------------------------------------- #


def cmd_service(args, cfg: CompetitionConfig) -> int:
    """Install, remove or inspect the background season service."""
    from .service import ServiceError, install, paths, status, uninstall

    repo = REPO_ROOT
    p = paths(repo)

    if args.action == "status":
        st = status(repo)
        _print(f"label    : {st['label']}")
        _print(f"plist    : {st['plist']}"
               f"{'' if st['installed'] else '   (not installed)'}")
        if st["running"]:
            _print(f"state    : RUNNING (pid {st['pid']})")
        elif st["installed"]:
            _print("state    : installed but not running")
            if st["last_exit"] is not None:
                _print(f"last exit: {st['last_exit']}")
        else:
            _print("state    : not installed")
        _print(f"logs     : {p.stdout}")
        _print(f"           {p.stderr}")
        return 0

    if args.action == "logs":
        for path in (p.stdout, p.stderr):
            if not path.exists():
                _print(f"-- {path} (no such file yet)")
                continue
            lines = path.read_text(errors="replace").splitlines()
            _print(f"-- {path}  ({len(lines)} lines)")
            for line in lines[-args.lines:]:
                _print(f"   {line}")
        return 0

    if args.action == "uninstall":
        removed = uninstall(repo)
        _print("service removed." if removed else "no service was installed.")
        return 0

    # install
    extra: list[str] = []
    if args.no_publish:
        extra.append("--no-publish")
    try:
        p = install(repo, extra_args=extra)
    except ServiceError as e:
        _print(f"ERROR: {e}")
        return 1
    _print(f"installed {p.plist}")
    _print("")
    _print("The season now runs in the background. It starts at login, holds")
    _print("the Mac awake while trading, and restarts itself if it exits.")
    _print("")
    _print("  comp service status    what launchd thinks")
    _print("  comp service logs      recent output")
    _print("  comp service uninstall stop and remove it")
    return 0


# --------------------------------------------------------------------------- #
# account setup
# --------------------------------------------------------------------------- #


def cmd_setup_accounts(args, cfg: CompetitionConfig) -> int:
    """Parse, verify and write the nine key pairs."""
    env_path = Path(args.env_out)
    # One credential pair per REAL account: a team in per_team mode, a whole
    # group in shared mode. Counting teams here would demand nine pairs for
    # three accounts.
    _teams_for(cfg, args.team)          # validates the --team keys
    teams = targets_for(cfg, only=_team_keys(cfg, args.team))

    if args.template:
        sys.stdout.write(render_template(cfg))
        return 0

    if args.verify:
        load_dotenv(args.env_file, override=True)
        bound = []
        for team in teams:
            kid, sec = team.credentials()
            if not (kid and sec):
                _print(f"  ----  {team.key:<14} no keys in the environment")
                continue
            bound.append((team, Pair(kid, sec)))
        if not bound:
            _print("Nothing to verify: no team has credentials set. Run "
                   "`comp setup-accounts` first.")
            return 1
        _print(f"Verifying {len(bound)} credential pair(s) against Alpaca…\n")
        results = verify_credentials(bound)
        lines, problems = report(results, bankroll=cfg.starting_cash)
        for line in lines:
            _print(line)
        for problem in problems:
            _print(f"\n  ERROR {problem}")
        return 1 if problems else 0

    # ---- gather the pairs ------------------------------------------------ #
    try:
        if args.from_file:
            text = (sys.stdin.read() if args.from_file == "-"
                    else Path(args.from_file).read_text())
            pairs = parse_pairs(text)
            _print(f"Parsed {len(pairs)} credential pair(s).")
        else:
            pairs = _prompt_for_pairs(teams)
        if not pairs:
            _print("No credentials given; nothing written.")
            return 1
        bound = assign(cfg, pairs, only=_team_keys(cfg, args.team))
    except SetupError as e:
        _print(f"ERROR: {e}")
        return 2
    except OSError as e:
        _print(f"ERROR: cannot read {args.from_file}: {e}")
        return 2

    missing = [t.key for t in teams if t.key not in {b[0].key for b in bound}]
    if missing:
        noun = teams[0].noun if teams else "team"
        _print(f"\n  {len(missing)} {noun}(s) have no credentials: "
               f"{', '.join(missing)}")
        if not args.allow_missing:
            _print("  Pass --allow-missing to write a partial .env anyway, or "
                   "supply the remaining pairs.")
            return 1

    # ---- verify before writing ------------------------------------------ #
    _print(f"\nVerifying {len(bound)} pair(s) against Alpaca…\n")
    results = verify_credentials(bound)
    lines, problems = report(results, bankroll=cfg.starting_cash)
    for line in lines:
        _print(line)

    if problems and not args.force:
        _print("")
        for problem in problems:
            _print(f"  ERROR {problem}")
        _print("\nNothing written. Fix the above, or pass --force to write "
               "anyway (not recommended).")
        return 1

    body = render_env(
        results,
        data_from=args.data_from,
        feed=cfg.data.feed,
    )
    if args.dry_run:
        _print(f"\n--dry-run: would write {len(results)} team(s) to {env_path}. "
               f"Secrets not shown.")
        return 0

    target, backup = write_env(body, env_path)
    _print(f"\nwrote {target} (mode 0600)")
    if backup:
        _print(f"backed up the previous file to {backup}")
    for problem in problems:
        _print(f"  WARNING {problem}")
    _print("\nNext: `comp doctor --check-accounts`")
    return 0


def _prompt_for_pairs(teams: Sequence) -> list[Pair]:
    """Ask for each team's pair, without echoing the secret."""
    import getpass

    _print("Paste each team's Alpaca PAPER credentials. Blank key id skips a "
           "team.")
    _print("Secrets are not echoed. Ctrl-C to abort.\n")
    pairs: list[Pair] = []
    for team in teams:
        try:
            key = input(f"  {team.name} ({team.key}) key id: ").strip()
        except EOFError:
            break
        if not key:
            continue
        secret = getpass.getpass(f"  {team.name} secret (hidden): ").strip()
        if not secret:
            _print("    no secret given; skipping this team")
            continue
        pairs.append(Pair(key, secret, team.key))
    return pairs


# --------------------------------------------------------------------------- #
# dashboard and status
# --------------------------------------------------------------------------- #


def cmd_dashboard(args, cfg: CompetitionConfig) -> int:

    ledger = Ledger(args.ledger)
    if args.run_id:
        ledger.run_id = args.run_id
    else:
        runs = ledger.runs()
        if runs:
            ledger.run_id = runs[-1]["run_id"]
    round_id = args.round
    if round_id is None:
        pending = ledger.in_progress_rounds()
        if pending:
            round_id = int(pending[0]["round_id"])
        else:
            scored = [r.id for r in cfg.rounds if ledger.results_for_round(r.id)]
            round_id = scored[-1] if scored else None
    from .reporting.dashboard import build_dashboard_data, render_dashboard

    # Stamp the season clock so the standalone page carries the same countdown
    # the live one does.
    schedule = None
    try:
        schedule = _build_schedule(cfg, MarketCalendar())
    except (ValueError, KeyError) as e:
        log.debug("no season schedule for the dashboard: %s", e)
    data = build_dashboard_data(cfg, ledger=ledger, round_id=round_id)
    _stamp_season(data, cfg, schedule)
    out = Path(args.out)
    _atomic_write(out, render_dashboard(cfg, data, refresh=args.refresh))
    ledger.close()
    _print(f"wrote {out}")
    if args.round is None and round_id is None:
        _print("No round found in the ledger yet -- the page will be mostly empty. "
               "Run a round, or pass --round N.")
    if args.open:
        import webbrowser
        webbrowser.open(out.resolve().as_uri())
        _print("opened in your browser")
    else:
        _print(f"open it with:  open {out}")
    return 0


def cmd_status(args, cfg: CompetitionConfig) -> int:
    """Where the competition is right now, in one screen of text."""
    cal = MarketCalendar()
    ledger = Ledger(args.ledger)
    runs = ledger.runs()
    if runs:
        ledger.run_id = args.run_id or runs[-1]["run_id"]

    _print(f"{cfg.name}   rules {cfg.rules_hash}")
    session = cal.session(utcnow())
    if session.is_open:
        m = session.minutes_to_close
        clock = f"OPEN, {int(m // 60)}h {int(m % 60):02d}m to the close"
    elif session.is_trading_day and session.open_at and utcnow() < session.open_at:
        clock = f"pre-open, {session.seconds_to_open / 60:.0f}m to the bell"
    else:
        nxt = session.next_open
        clock = f"closed, next open {nxt:%a %d %b %H:%M UTC}" if nxt else "closed"
    _print(f"market   : {clock}   ({session.session_date})")

    pending = ledger.in_progress_rounds()
    _print("")
    for rnd in cfg.rounds:
        row = ledger.round_progress(rnd.id)
        results = ledger.results_for_round(rnd.id)
        if results:
            ranked = sorted(results, key=lambda r: (r["place"] is None, r["place"]))
            winner = ranked[0]
            state = (f"COMPLETE  winner {winner['team_key']} "
                     f"{winner['return_pct'] * 100:+.2f}%")
        elif row is not None and row["status"] == "in_progress":
            start = date.fromisoformat(row["start_date"][:10])
            end = date.fromisoformat(row["end_date"][:10])
            sessions = cal.trading_days(start, end)
            done = sum(1 for d in sessions if d < session.session_date)
            day = max(min((session.session_date - start).days + 1,
                          (end - start).days + 1), 1)
            state = (f"IN PROGRESS  day {day}/{(end - start).days + 1}, "
                     f"session {done + (1 if session.is_open else 0)}/{len(sessions)}, "
                     f"{row['ticks']} ticks, ends {end:%a %d %b}")
        else:
            state = "not started"
        _print(f"round {rnd.id}  {rnd.name:<16} {state}")

    if pending:
        _print("")
        for row in pending:
            _print(f"resumable: round {row['round_id']} from run {row['run_id']} "
                   f"({row['ticks']} ticks, updated {row['updated_at'][:19]}). "
                   f"`comp run --round {row['round_id']}` will rejoin it.")

    scores = _scores_from_ledger(cfg, ledger, [r.id for r in cfg.rounds])
    if scores:
        _print("")
        _print(build_leaderboard(cfg, scores).table(title="STANDINGS SO FAR"))
    ledger.close()
    return 0


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #


def _scores_from_ledger(cfg: CompetitionConfig, ledger: Ledger, rounds: Sequence[int]):
    from .scoring.points import RoundScore, TeamScore

    out = []
    for rid in rounds:
        rows = ledger.results_for_round(rid)
        if not rows:
            continue
        scores = []
        for r in rows:
            try:
                metrics = json.loads(r["metrics_json"])
            except (json.JSONDecodeError, TypeError):
                metrics = {}
            try:
                name = cfg.team(r["team_key"]).name
            except KeyError:
                name = r["team_key"]
            scores.append(TeamScore(
                team_key=r["team_key"], name=name,
                return_pct=float(r["return_pct"]),
                end_equity=float(r["end_equity"]), start_equity=float(r["start_equity"]),
                scored=bool(r["scored"]), place=r["place"],
                points=float(r["points"] or 0.0), metrics=metrics,
            ))
        try:
            rname = cfg.round(rid).name
        except KeyError:
            rname = f"Round {rid}"
        out.append(RoundScore(round_id=rid, round_name=rname, scores=scores))
    return out


def cmd_score(args, cfg: CompetitionConfig) -> int:
    ledger = Ledger(args.ledger)
    rounds = [args.round] if args.round else [r.id for r in cfg.rounds]
    scores = _scores_from_ledger(cfg, ledger, rounds)
    ledger.close()
    if not scores:
        _print(f"no recorded results for round(s) {rounds} in {args.ledger}")
        return 1
    for rs in scores:
        _print(rs.table())
        _print("")
    return 0


def cmd_leaderboard(args, cfg: CompetitionConfig) -> int:
    ledger = Ledger(args.ledger)
    scores = _scores_from_ledger(cfg, ledger, [r.id for r in cfg.rounds])
    ledger.close()
    if not scores:
        _print(f"no recorded results in {args.ledger}. Run a round first.")
        return 1
    lb = build_leaderboard(cfg, scores)
    if args.markdown:
        _print(f"## {cfg.name} -- standings\n")
        _print(lb.markdown())
    elif args.json:
        _print(json.dumps(lb.to_dict(), indent=2))
    else:
        for rs in scores:
            _print(rs.table())
            _print("")
        _print(lb.table())
    if args.write:
        out = Path(args.write)
        out.parent.mkdir(parents=True, exist_ok=True)
        body = (
            f"# {cfg.name}\n\n"
            f"_rules hash `{cfg.rules_hash}` -- generated {utcnow():%Y-%m-%d %H:%M} UTC_\n\n"
            f"## Standings\n\n{lb.markdown()}\n\n"
        )
        for rs in scores:
            body += f"## Round {rs.round_id} -- {rs.round_name}\n\n```\n{rs.table()}\n```\n\n"
        out.write_text(body)
        _print(f"\nwrote {out}")
    return 0


def cmd_report(args, cfg: CompetitionConfig) -> int:
    ledger = Ledger(args.ledger)
    runs = ledger.runs()
    if not runs:
        _print(f"no runs recorded in {args.ledger}")
        ledger.close()
        return 1
    _print(f"ledger: {args.ledger}")
    _print(f"tables: {json.dumps(ledger.stats())}")
    _print("\n-- runs -----------------------------------------------------------")
    for r in runs:
        _print(f"  {r['run_id']}  round {r['round_id']}  {r['mode']:<9} "
               f"{r['broker']:<14} rules={r['rules_hash']}  {r['started_at'][:19]}"
               + (f"  {r['notes']}" if r["notes"] else ""))

    # Report against the most recent run unless told otherwise.
    ledger.run_id = args.run_id or runs[-1]["run_id"]
    teams = _teams_for(cfg, args.team)
    for team in teams:
        fills = ledger.fill_count(team.key)
        if not fills and not args.all:
            continue
        _print(f"\n=== {team.name} [{team.key}] " + "=" * (40 - len(team.name)))
        _print(f"  fills {fills}, traded ${ledger.traded_notional(team.key):,.2f}")
        tally = ledger.rejection_tally(team.key)
        if tally:
            _print("  rejections: " + ", ".join(f"{k}={v}" for k, v in tally.items()))
        curve = ledger.equity_curve(team.key)
        if curve:
            from .util import indicators as ind

            vals = [e for _t, e in curve]
            _print(f"  equity: {vals[0]:,.2f} -> {vals[-1]:,.2f} "
                   f"({vals[-1] / vals[0] - 1:+.2%}), maxDD {ind.max_drawdown(vals):.2%}, "
                   f"{len(vals)} snapshots")
        for rnd in cfg.rounds:
            uni = ledger.universe_for(rnd.id, team.key)
            if uni:
                _print(f"  round {rnd.id} universe ({len(uni)}): {', '.join(uni)}")
        if args.trades:
            rows = ledger.trade_log(team.key, limit=args.trades)
            if rows:
                _print(f"  last {len(rows)} fills:")
                for row in rows:
                    _print(f"    {row['ts'][:19]}  {row['side']:<4} "
                           f"{row['qty']:>10.4f} {row['symbol']:<6} @ {row['price']:>9.4f}"
                           f"  {(row['reason'] or '')[:64]}")
    ledger.close()
    return 0


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="comp",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"comp {__version__}")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="-v for debug logging, -vv for very verbose")
    p.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")
    p.add_argument("--config", default=None, help="config directory (default: ./config)")
    p.add_argument("--env-file", default=None, help="path to a .env file")
    p.add_argument("--ledger", default=str(DEFAULT_LEDGER), help="SQLite ledger path")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="check config, credentials and accounts")
    d.add_argument("--check-accounts", action="store_true",
                   help="hit the Alpaca API to verify every team's account")
    d.add_argument("--start", default=None)
    d.add_argument("--end", default=None)
    d.set_defaults(func=cmd_doctor)

    t = sub.add_parser("teams", help="describe the field")
    t.add_argument("--team", action="append", help="limit to these team keys")
    t.set_defaults(func=cmd_teams)

    u = sub.add_parser("universe", help="audit the Round 3 ranked pool")
    u.add_argument("--metric", default="market_cap", choices=["market_cap", "share_price"])
    u.add_argument("--size", type=int, default=500)
    u.add_argument("--live", action="store_true", help="check tradability against Alpaca")
    u.set_defaults(func=cmd_universe)

    ru = sub.add_parser("refresh-universe", help="re-price and re-rank the pool")
    ru.add_argument("--metric", default="market_cap", choices=["market_cap", "share_price"])
    ru.add_argument("--size", type=int, default=500)
    ru.set_defaults(func=cmd_refresh_universe)

    dr = sub.add_parser("draft", help="deal Round 3's hands")
    dr.add_argument("--round", type=int, default=3)
    dr.add_argument("--seed", type=int, default=None)
    dr.add_argument("--team", action="append")
    dr.add_argument("--live", action="store_true", help="verify tradability first")
    dr.add_argument("--write", action="store_true", help="record the draft to the ledger")
    dr.add_argument("--json", default=None, help="also write the draft to this JSON file")
    dr.set_defaults(func=cmd_draft)

    vd = sub.add_parser("verify-draft", help="re-check a recorded draft")
    vd.add_argument("--round", type=int, default=3)
    vd.set_defaults(func=cmd_verify_draft)

    pt = sub.add_parser("pretrain", help="train the RL entry offline on history")
    pt.add_argument("--team-key", default="q_learner")
    pt.add_argument("--symbols", default=None, help="comma-separated list")
    pt.add_argument("--start", default=None)
    pt.add_argument("--end", default=None)
    pt.add_argument("--days", type=int, default=120)
    pt.add_argument("--passes", type=int, default=4)
    pt.add_argument("--source", default="alpaca", choices=["alpaca", "synthetic"])
    pt.add_argument("--seed", type=int, default=None)
    pt.add_argument("--resume", action="store_true", help="continue from saved state")
    pt.add_argument("--dry-run", action="store_true")
    pt.add_argument("--top", type=int, default=20, help="policy rows to print")
    pt.add_argument("--only-picker", action="store_true",
                    help="train only the picker, leaving the strategy untouched")
    pt.add_argument("--skip-picker", action="store_true",
                    help="train only the strategy, not its stock picker")
    pt.add_argument("--picker-symbols", default=None,
                    help="comma-separated training pool for the picker")
    pt.add_argument("--picker-pool", type=int, default=150,
                    help="how many names from the Round 3 pool to train on")
    pt.add_argument("--picker-days", type=int, default=730,
                    help="calendar days of daily history for the picker")
    pt.add_argument("--picker-passes", type=int, default=1)
    pt.set_defaults(func=cmd_pretrain)

    bt = sub.add_parser("backtest", help="replay a round offline")
    bt.add_argument("--round", type=int, required=True)
    bt.add_argument("--start", default=None)
    bt.add_argument("--end", default=None)
    bt.add_argument("--team", action="append")
    bt.add_argument("--source", default="synthetic", choices=["synthetic", "alpaca"])
    bt.add_argument("--seed", type=int, default=None)
    bt.add_argument("--tick-seconds", type=int, default=None,
                    help="engine cadence (default: the fastest team's)")
    bt.add_argument("--pool-size", type=int, default=80,
                    help="Round 2 candidate pool size for the dry run")
    bt.add_argument("--history-days", type=int, default=90,
                    help="extra history to download before the window (alpaca source)")
    bt.add_argument("--warmup-sessions", type=int, default=40,
                    help="synthetic intraday warm-up sessions before the window")
    bt.add_argument("--daily-history", type=int, default=260,
                    help="synthetic daily sessions of history for the pickers")
    bt.add_argument("--spread-bps", type=float, default=4.0)
    bt.add_argument("--equity-every", type=int, default=300)
    bt.add_argument("--resume-learning", action="store_true",
                    help="load saved Q-table / bandit state first")
    bt.add_argument("--dashboard", nargs="?", const=str(DEFAULT_DASHBOARD),
                    default=None,
                    help="write an HTML dashboard as the backtest runs")
    bt.add_argument("--dashboard-refresh", type=int, default=0,
                    help="auto-refresh seconds for a backtest dashboard (0 = static)")
    bt.add_argument("--notes", default="")
    bt.set_defaults(func=cmd_backtest)

    rn = sub.add_parser("run", help="run a round live on Alpaca paper accounts")
    rn.add_argument("--round", type=int, required=True)
    rn.add_argument("--start", default=None)
    rn.add_argument("--end", default=None)
    rn.add_argument("--team", action="append")
    rn.add_argument("--poll", type=int, default=15, help="seconds between engine polls")
    rn.add_argument("--max-ticks", type=int, default=None)
    rn.add_argument("--equity-every", type=int, default=300)
    rn.add_argument("--dashboard", nargs="?", const=str(DEFAULT_DASHBOARD),
                    default=str(DEFAULT_DASHBOARD),
                    help="write a live HTML dashboard here ('' to disable)")
    rn.add_argument("--dashboard-refresh", type=int, default=30)
    rn.add_argument("--fresh", action="store_true",
                    help="abandon an in-progress round and restart it from scratch")
    rn.add_argument("--dry-run", action="store_true", help="validate setup and stop")
    rn.add_argument("--allow-missing", action="store_true",
                    help="run only the teams that have credentials")
    rn.add_argument("--force", action="store_true",
                    help="start even if account equities differ")
    rn.add_argument("--no-publish", action="store_true",
                    help="do not push the dashboard to GitHub Pages")
    rn.add_argument("--notes", default="")
    rn.set_defaults(func=cmd_run)

    sn = sub.add_parser(
        "season",
        help="run the whole season unattended: all rounds, back to back")
    sn.add_argument("--start", default=None,
                    help="override the first round's start date (YYYY-MM-DD)")
    sn.add_argument("--team", action="append")
    sn.add_argument("--poll", type=int, default=15,
                    help="seconds between engine polls")
    sn.add_argument("--max-ticks", type=int, default=None)
    sn.add_argument("--equity-every", type=int, default=300)
    sn.add_argument("--dashboard", nargs="?", const=str(DEFAULT_DASHBOARD),
                    default=str(DEFAULT_DASHBOARD),
                    help="write a live HTML dashboard here ('' to disable)")
    sn.add_argument("--dashboard-refresh", type=int, default=30)
    sn.add_argument("--no-publish", action="store_true",
                    help="do not push the dashboard to GitHub Pages")
    sn.add_argument("--dry-run", action="store_true",
                    help="resolve the schedule and stop")
    sn.add_argument("--allow-missing", action="store_true",
                    help="run only the teams that have credentials")
    sn.add_argument("--force", action="store_true",
                    help="start even if account equities differ")
    sn.add_argument("--until", default=None, metavar="HH:MM",
                    help="absolute UTC time to suspend and exit")
    sn.add_argument("--max-runtime", type=int, default=0,
                    help="seconds before suspending a round and exiting "
                         "(0 = run until the season ends)")
    sn.add_argument("--notes", default="")
    sn.set_defaults(func=cmd_season)

    sv = sub.add_parser(
        "service", help="run the season in the background (macOS LaunchAgent)")
    sv.add_argument("action", choices=["install", "uninstall", "status", "logs"])
    sv.add_argument("--no-publish", action="store_true",
                    help="do not push the dashboard to GitHub Pages")
    sv.add_argument("--lines", type=int, default=30,
                    help="log lines to show for `logs`")
    sv.set_defaults(func=cmd_service)

    st = sub.add_parser(
        "state", help="carry the ledger between runs on an ephemeral host")
    st.add_argument("action", choices=["push", "pull"])
    st.add_argument("--branch", default="season-state",
                    help="branch the ledger is parked on")
    st.set_defaults(func=cmd_state)

    pb = sub.add_parser("publish", help="render the dashboard and push it to Pages")
    pb.add_argument("--out", default=str(DEFAULT_DASHBOARD),
                    help="also write the HTML here ('' to skip)")
    pb.add_argument("--refresh", type=int, default=30)
    pb.set_defaults(func=cmd_publish)

    sc = sub.add_parser("score", help="score recorded round results")
    sc.add_argument("--round", type=int, default=None)
    sc.set_defaults(func=cmd_score)

    lb = sub.add_parser("leaderboard", help="season standings")
    lb.add_argument("--markdown", action="store_true")
    lb.add_argument("--json", action="store_true")
    lb.add_argument("--write", default=None, help="write a results markdown file")
    lb.set_defaults(func=cmd_leaderboard)

    sa = sub.add_parser("setup-accounts",
                        help="parse, verify and write the teams' Alpaca keys")
    sa.add_argument("--from-file", default=None,
                    help="file of pasted key pairs ('-' for stdin); "
                         "omit to be prompted per team")
    sa.add_argument("--env-out", default=str(REPO_ROOT / ".env"))
    sa.add_argument("--team", action="append", help="limit to these team keys")
    sa.add_argument("--data-from", default=None,
                    help="team whose keys feed the shared market data")
    sa.add_argument("--template", action="store_true",
                    help="print a labelled keys.txt skeleton and exit")
    sa.add_argument("--verify", action="store_true",
                    help="only check the credentials already in .env")
    sa.add_argument("--allow-missing", action="store_true",
                    help="write a partial .env when some teams have no keys")
    sa.add_argument("--force", action="store_true",
                    help="write even if verification found problems")
    sa.add_argument("--dry-run", action="store_true")
    sa.set_defaults(func=cmd_setup_accounts)

    db = sub.add_parser("dashboard", help="render the HTML dashboard")
    db.add_argument("--round", type=int, default=None,
                    help="which round (default: the in-progress or latest one)")
    db.add_argument("--out", default=str(DEFAULT_DASHBOARD))
    db.add_argument("--refresh", type=int, default=30,
                    help="auto-refresh seconds; 0 disables")
    db.add_argument("--run-id", default=None)
    db.add_argument("--open", action="store_true", help="open it in a browser")
    db.set_defaults(func=cmd_dashboard)

    st = sub.add_parser("status", help="where the competition is right now")
    st.add_argument("--run-id", default=None)
    st.set_defaults(func=cmd_status)

    rp = sub.add_parser("report", help="per-team detail from the ledger")
    rp.add_argument("--team", action="append")
    rp.add_argument("--run-id", default=None)
    rp.add_argument("--trades", type=int, default=0, help="show this many recent fills")
    rp.add_argument("--all", action="store_true", help="include teams with no fills")
    rp.set_defaults(func=cmd_report)

    lx = sub.add_parser("lexicon", help="sentiment lexicon statistics")
    lx.set_defaults(func=cmd_lexicon)

    ex = sub.add_parser("explain-news", help="show the sentiment scorer's working")
    ex.add_argument("text", nargs="+")
    ex.set_defaults(func=cmd_explain_news)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose, quiet=args.quiet)
    try:
        cfg = load_config(args.config, env_file=args.env_file)
    except ConfigError as e:
        _print(f"CONFIG ERROR: {e}")
        return 2
    try:
        return int(args.func(args, cfg) or 0)
    except ConfigError as e:
        _print(f"ERROR: {e}")
        return 2
    except BrokerError as e:
        _print(f"BROKER ERROR: {e}")
        return 3
    except KeyboardInterrupt:
        _print("\ninterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
