"""Resolving each team's tradable universe for a round.

The three rounds differ *only* here, which is deliberate -- the trading loop,
the risk rails and the scoring are identical throughout, so differences in
the results come from the universe rules and the strategies, not from three
subtly different engines.

  Round 1 `fixed`   every team gets the same three tickers. No selection.
  Round 2 `picker`  every team's picker screens the SAME candidate pool and
                    returns its own shortlist (<= picker.max_symbols).
  Round 3 `draft`   the dealer hands each team ten tickers whose ranks sum to
                    the same total; the picker then ranks *within* that hand
                    and may choose to trade fewer than all ten.

The resolved universe is written to the ledger for every session, so "what
was this team allowed to trade on Wednesday" is always answerable.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

from ..config import CompetitionConfig, RoundConfig, TeamConfig
from ..data.universe import UniverseProvider, UniverseSnapshot
from ..draft import DraftResult, deal
from ..pickers import load_picker
from ..pickers.base import Picker, PickerContext, PickResult
from ..types import Bar, NewsItem

log = logging.getLogger("competition.universe")

#: symbols -> {symbol: daily bars}
DailyBarsFn = Callable[[Sequence[str]], Mapping[str, tuple[Bar, ...]]]
#: symbols -> {symbol: news items}
NewsFn = Callable[[Sequence[str]], Mapping[str, tuple[NewsItem, ...]]]


@dataclass
class ResolvedUniverse:
    """One team's tradable set for one session, with its provenance."""

    team_key: str
    symbols: tuple[str, ...]
    source: str
    detail: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.symbols)

    def __iter__(self):
        return iter(self.symbols)


class UniverseResolver:
    """Builds per-team universes for whichever round is being run."""

    def __init__(
        self,
        cfg: CompetitionConfig,
        *,
        provider: UniverseProvider | None = None,
        daily_bars: DailyBarsFn | None = None,
        news: NewsFn | None = None,
        pool_size_override: int | None = None,
    ):
        self.cfg = cfg
        self.provider = provider
        self._daily_bars = daily_bars
        self._news = news
        #: Backtests build data for a subset of the pool; screening 300 names
        #: when only 60 have bars would reject 80% of the field for "no
        #: history" and silently gut Round 2.
        self.pool_size_override = pool_size_override
        self._pickers: dict[str, Picker] = {}
        self._draft: DraftResult | None = None
        self._candidate_cache: tuple[str, ...] | None = None

    # ------------------------------------------------------------------ #

    def picker_for(self, team: TeamConfig) -> Picker:
        if team.key not in self._pickers:
            self._pickers[team.key] = load_picker(team.picker)(team)
        return self._pickers[team.key]

    @property
    def draft(self) -> DraftResult | None:
        return self._draft

    # ------------------------------------------------------------------ #
    # the Round 3 deal
    # ------------------------------------------------------------------ #

    def run_draft(
        self,
        rnd: RoundConfig,
        teams: Sequence[TeamConfig],
        *,
        snapshot: UniverseSnapshot | None = None,
        seed: int | None = None,
        verify_tradable: bool = True,
    ) -> DraftResult:
        """Deal Round 3's hands. Every scored team plus any unscored ones."""
        if rnd.draft is None:
            raise ValueError(f"round {rnd.id} has no draft configuration")
        d = rnd.draft
        if snapshot is None:
            if self.provider is None:
                raise ValueError("a UniverseProvider or an explicit snapshot is required")
            snapshot = self.provider.tradable_pool(
                metric=d.metric, size=d.pool_size, verify=verify_tradable
            )
        result = deal(
            [t.key for t in teams],
            snapshot,
            picks_per_team=d.picks_per_team,
            pool_size=min(d.pool_size, len(snapshot)),
            target_sum=d.rank_sum_target,
            method=d.method,
            unique_across_teams=d.unique_across_teams,
            min_rank_deciles=d.min_rank_deciles,
            max_sector_share=d.max_sector_share,
            seed=seed if seed is not None else self.cfg.fairness.competition_seed,
            max_attempts=d.max_attempts,
            metric=d.metric,
        )
        self._draft = result
        log.info("round %d draft dealt:\n%s", rnd.id, result.table())
        return result

    def set_draft(self, draft: DraftResult) -> None:
        """Reuse a previously dealt (and ledger-recorded) draft."""
        self._draft = draft

    # ------------------------------------------------------------------ #
    # candidate pool (Round 2)
    # ------------------------------------------------------------------ #

    def candidate_pool(self, rnd: RoundConfig, *, refresh: bool = False) -> tuple[str, ...]:
        """The shared screening pool. Built once and reused by every picker.

        Sharing it is the fairness guarantee for Round 2: the teams differ in
        what they *select*, never in what they are shown.
        """
        if self._candidate_cache is not None and not refresh:
            return self._candidate_cache
        if self.provider is None:
            raise ValueError("Round 2 needs a UniverseProvider to build a candidate pool")
        size = self.pool_size_override or rnd.picker.candidate_pool_size
        pool = self.provider.candidate_pool(
            size=size,
            liquidity=rnd.picker.liquidity_filter,
            mode=rnd.picker.candidate_pool,
        )
        self._candidate_cache = tuple(pool)
        log.info(
            "round %d candidate pool: %d names (mode=%s)",
            rnd.id, len(pool), rnd.picker.candidate_pool,
        )
        return self._candidate_cache

    # ------------------------------------------------------------------ #
    # resolution
    # ------------------------------------------------------------------ #

    def resolve(
        self,
        rnd: RoundConfig,
        team: TeamConfig,
        *,
        session: date,
        picker_state: dict | None = None,
        sectors: Mapping[str, str] | None = None,
    ) -> ResolvedUniverse:
        """The team's tradable symbols for this session."""
        if rnd.universe_mode == "fixed":
            return ResolvedUniverse(
                team.key, tuple(rnd.symbols), "fixed",
                {"round": rnd.id, "note": "identical for every team"},
            )

        if rnd.universe_mode == "draft":
            if self._draft is None:
                raise ValueError("run_draft() must be called before resolving Round 3")
            hand = self._draft.hand(team.key)
            picked = self._apply_picker(
                rnd, team, hand.symbols, session=session,
                picker_state=picker_state, sectors=sectors, locked=True,
            )
            # A picker that returns nothing must not silently take the team out
            # of the round; fall back to the full dealt hand.
            symbols = picked.symbols or hand.symbols
            return ResolvedUniverse(
                team.key, tuple(symbols), "draft+picker",
                {
                    "round": rnd.id,
                    "hand": list(hand.symbols),
                    "ranks": list(hand.ranks),
                    "rank_sum": hand.rank_sum,
                    "picker": picked.to_dict(),
                },
            )

        if rnd.universe_mode == "picker":
            pool = self.candidate_pool(rnd)
            picked = self._apply_picker(
                rnd, team, pool, session=session,
                picker_state=picker_state, sectors=sectors, locked=False,
            )
            if not picked.symbols:
                log.warning(
                    "%s: picker returned nothing from a %d-name pool; the team sits out "
                    "this session", team.key, len(pool),
                )
            return ResolvedUniverse(
                team.key, tuple(picked.symbols), "picker",
                {"round": rnd.id, "pool_size": len(pool), "picker": picked.to_dict()},
            )

        raise ValueError(f"unknown universe_mode {rnd.universe_mode!r}")

    def _apply_picker(
        self,
        rnd: RoundConfig,
        team: TeamConfig,
        candidates: Sequence[str],
        *,
        session: date,
        picker_state: dict | None,
        sectors: Mapping[str, str] | None,
        locked: bool,
    ) -> PickResult:
        picker = self.picker_for(team)
        symbols = tuple(dict.fromkeys(s.upper() for s in candidates))
        daily = dict(self._daily_bars(symbols)) if self._daily_bars else {}
        news = dict(self._news(symbols)) if self._news else {}
        if not sectors and self.provider is not None:
            try:
                snap = self.provider.snapshot()
                sectors = {s: snap.sector_of(s) for s in symbols}
            except FileNotFoundError:
                sectors = {}
        ctx = PickerContext(
            round_id=rnd.id,
            candidates=symbols,
            daily_bars=daily,
            max_symbols=rnd.picker.max_symbols,
            log=logging.getLogger(f"competition.picker.{team.key}"),
            news=news,
            sectors=dict(sectors or {}),
            state=picker_state if picker_state is not None else {},
            locked_universe=locked,
        )
        try:
            return picker.pick(ctx)
        except Exception as e:  # noqa: BLE001 -- one broken picker must not stop the round
            log.exception("%s: picker raised %s; falling back to the candidate list",
                          team.key, e)
            fallback = symbols[: rnd.picker.max_symbols]
            return PickResult(fallback, notes={s: "picker error fallback" for s in fallback})
