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
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path

from . import __version__
from .broker import AlpacaBroker, AlpacaClient, AlpacaCredentials, AlpacaDataReader
from .broker.base import Broker, BrokerError
from .broker.simulated import SimConfig, SimulatedBroker
from .config import REPO_ROOT, CompetitionConfig, ConfigError, load_config, load_dotenv
from .data.calendar import MarketCalendar
from .data.feed import AlpacaFeed, ReplayFeed
from .data.universe import UniverseProvider, UniverseSnapshot
from .draft import DraftError, verify
from .engine import CompetitionEngine, Ledger, UniverseResolver
from .pickers import load_picker
from .scoring import build_leaderboard, score_round
from .strategies import load_strategy
from .types import UTC, Bar, utcnow

log = logging.getLogger("competition.cli")

DEFAULT_LEDGER = REPO_ROOT / "runs" / "competition.sqlite"


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

    missing_teams = [t.key for t in cfg.teams if not t.has_credentials]
    ready_teams = [t.key for t in cfg.teams if t.has_credentials]
    for team in cfg.teams:
        state = "keys set" if team.has_credentials else "no keys"
        _print(f"    {team.key:<14} {team.env_prefix + '_KEY_ID':<32} {state}")
    if missing_teams:
        warnings.append(
            f"{len(missing_teams)} team(s) without keys ({', '.join(missing_teams)}); "
            f"they can only run in sim mode"
        )

    if args.check_accounts and ready_teams:
        _print("\n-- live account check ---------------------------------------------")
        equities = {}
        for team in cfg.teams:
            if not team.has_credentials:
                continue
            try:
                client = AlpacaClient(AlpacaCredentials.from_env(team.env_prefix))
                info = client.verify()
                equities[team.key] = info["equity"]
                flags = []
                if not info["paper"]:
                    flags.append("LIVE ACCOUNT (!)")
                    problems.append(f"{team.key} is pointed at a LIVE account, not paper")
                if info["trading_blocked"]:
                    flags.append("trading blocked")
                    problems.append(f"{team.key}: trading is blocked on this account")
                _print(f"  {team.key:<14} #{info['account_number']:<12} "
                       f"equity ${info['equity']:>10,.2f} cash ${info['cash']:>10,.2f} "
                       f"{' '.join(flags)}")
            except BrokerError as e:
                problems.append(f"{team.key}: credential check failed -- {e}")
                _print(f"  {team.key:<14} FAILED: {e}")
        if len(equities) > 1:
            lo, hi = min(equities.values()), max(equities.values())
            spread = hi - lo
            _print(f"\n  equity spread across {len(equities)} accounts: "
                   f"${spread:,.2f} (${lo:,.2f} .. ${hi:,.2f})")
            tolerance = max(0.01 * cfg.starting_cash, 5.0)
            if spread > tolerance:
                problems.append(
                    f"accounts differ by ${spread:,.2f}, more than the ${tolerance:,.2f} "
                    f"tolerance. Unequal bankrolls make the round unfair -- reset the "
                    f"paper accounts before starting."
                )

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

    problems = verify(
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
    ledger.close()
    return 0


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
    brokers: dict[str, Broker] = {
        t.key: SimulatedBroker(cfg.starting_cash, quote_source, name=f"sim:{t.key}",
                               config=sim_cfg, clock=clock)
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
    if args.resume_learning:
        engine.load_learned_state()

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
    _print(f"\nledger: {args.ledger} (run {ledger.run_id})")
    ledger.close()
    return 0


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
    missing = [t.key for t in teams if not t.has_credentials]
    if missing and not args.allow_missing:
        _print(f"ERROR: no Alpaca credentials for: {', '.join(missing)}.")
        _print("Add them to .env, or pass --allow-missing to run only the funded teams,")
        _print("or use `comp backtest` for a no-key dry run.")
        return 1
    teams = [t for t in teams if t.has_credentials]
    if not teams:
        _print("ERROR: no team has credentials.")
        return 1

    _print(f"Round {rnd.id} ({rnd.name}) LIVE")
    _print(f"  window  : {start} .. {end}  ({len(sessions)} sessions)")
    _print(f"  teams   : {', '.join(t.key for t in teams)}")
    _print(f"  bankroll: ${cfg.starting_cash:,.2f} each")
    _print(f"  rules   : {cfg.rules_hash}")

    brokers: dict[str, Broker] = {}
    for team in teams:
        try:
            brokers[team.key] = AlpacaBroker.from_env(team.env_prefix, name=team.key)
        except BrokerError as e:
            _print(f"ERROR: {team.key}: {e}")
            return 1

    # Refuse to start with unequal bankrolls -- that is the one condition that
    # invalidates the whole round.
    equities = {}
    for key, broker in brokers.items():
        try:
            equities[key] = broker.account(fresh=True).equity
        except BrokerError as e:
            _print(f"ERROR: {key}: cannot read the account -- {e}")
            return 1
    spread = max(equities.values()) - min(equities.values()) if equities else 0.0
    tolerance = max(0.01 * cfg.starting_cash, 5.0)
    _print("  equity  : " + ", ".join(f"{k}=${v:,.2f}" for k, v in equities.items()))
    if spread > tolerance and not args.force:
        _print(f"\nERROR: account equities differ by ${spread:,.2f} (tolerance "
               f"${tolerance:,.2f}). Reset the paper accounts so every team starts "
               f"from the same bankroll, or pass --force to accept the difference "
               f"(round P&L is measured per-account either way).")
        return 1

    if args.dry_run:
        _print("\n--dry-run: setup validated, not trading.")
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
    ledger = Ledger(args.ledger)
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
    engine.load_learned_state()

    try:
        result = engine.run_live(
            rnd, start=start, end=end, poll_seconds=args.poll,
            max_ticks=args.max_ticks,
        )
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
    _print("\n" + scored.table())
    _print(f"\nledger: {args.ledger} (run {ledger.run_id})")
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
    rn.add_argument("--dry-run", action="store_true", help="validate setup and stop")
    rn.add_argument("--allow-missing", action="store_true",
                    help="run only the teams that have credentials")
    rn.add_argument("--force", action="store_true",
                    help="start even if account equities differ")
    rn.add_argument("--notes", default="")
    rn.set_defaults(func=cmd_run)

    sc = sub.add_parser("score", help="score recorded round results")
    sc.add_argument("--round", type=int, default=None)
    sc.set_defaults(func=cmd_score)

    lb = sub.add_parser("leaderboard", help="season standings")
    lb.add_argument("--markdown", action="store_true")
    lb.add_argument("--json", action="store_true")
    lb.add_argument("--write", default=None, help="write a results markdown file")
    lb.set_defaults(func=cmd_leaderboard)

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
