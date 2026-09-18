"""The stock-picker contract.

A picker is handed daily bars (plus optional news) for a shared candidate
pool and must return a ranked shortlist. The engine calls it once per session
before the open (Round 2) or once to rank within a dealt hand (Round 3).

Round 2: the pool is Alpaca's most-active names unioned with the large-cap
snapshot, liquidity-screened -- identical for every team.
Round 3: the pool *is* the team's ten dealt tickers, so the picker's job
changes from "find something good" to "rank what I was given and decide how
many of them are worth trading at all". A picker that returns fewer than the
maximum is making a real decision: sitting out a bad hand is allowed.
"""

from __future__ import annotations

import abc
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import TeamConfig
from ..strategies.base import Param, Params
from ..types import Bar, NewsItem


@dataclass
class PickerContext:
    """Everything a picker may read."""

    round_id: int
    candidates: tuple[str, ...]
    daily_bars: Mapping[str, tuple[Bar, ...]]
    max_symbols: int
    log: logging.Logger
    news: Mapping[str, tuple[NewsItem, ...]] = field(default_factory=dict)
    intraday_bars: Mapping[str, tuple[Bar, ...]] = field(default_factory=dict)
    sectors: Mapping[str, str] = field(default_factory=dict)
    #: Persistent per-team store, so a learning picker (the bandit) can
    #: remember what it tried and what it earned.
    state: dict[str, Any] = field(default_factory=dict)
    #: Set in Round 3: the picker may only rank within the dealt hand.
    locked_universe: bool = False

    def bars(self, symbol: str) -> tuple[Bar, ...]:
        return self.daily_bars.get(symbol.upper(), ())

    def closes(self, symbol: str, n: int | None = None) -> list[float]:
        vals = [b.close for b in self.bars(symbol)]
        return vals[-n:] if n else vals

    def price(self, symbol: str) -> float:
        series = self.bars(symbol)
        return series[-1].close if series else 0.0

    def has_history(self, symbol: str, n: int) -> bool:
        return len(self.bars(symbol)) >= n

    def avg_dollar_volume(self, symbol: str, n: int = 20) -> float:
        series = self.bars(symbol)[-n:]
        if not series:
            return 0.0
        return sum(b.dollar_volume for b in series) / len(series)

    def sector(self, symbol: str) -> str:
        return self.sectors.get(symbol.upper(), "Unknown")

    def ready(self, min_bars: int) -> list[str]:
        return [s for s in self.candidates if self.has_history(s, min_bars)]


@dataclass
class PickResult:
    """A picker's shortlist plus the reasoning, which goes into the ledger."""

    symbols: tuple[str, ...]
    scores: dict[str, float] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    rejected: dict[str, str] = field(default_factory=dict)
    #: Optional extra payload (the pair picker returns its chosen pairs here).
    extra: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.symbols)

    def __iter__(self):
        return iter(self.symbols)

    def describe(self, limit: int = 12) -> str:
        if not self.symbols:
            return "(no picks)"
        parts = []
        for s in self.symbols[:limit]:
            note = self.notes.get(s, "")
            parts.append(f"{s} {self.scores.get(s, 0.0):+.3f}" + (f" [{note}]" if note else ""))
        return "; ".join(parts)

    def to_dict(self) -> dict:
        return {
            "symbols": list(self.symbols),
            "scores": {k: round(v, 6) for k, v in self.scores.items()},
            "notes": dict(self.notes),
            "n_rejected": len(self.rejected),
            "extra": self.extra,
        }


class Picker(abc.ABC):
    """Base class for every team's universe selector."""

    PARAMS: tuple[Param, ...] = ()
    DESCRIPTION: str = ""
    #: Minimum daily bars a candidate needs before this picker will score it.
    MIN_HISTORY: int = 30

    def __init__(self, team: TeamConfig, params: Mapping[str, Any] | None = None):
        self.team = team
        self.key = team.key
        try:
            self.p = Params(self.PARAMS, params if params is not None else team.picker_params)
        except ValueError as e:
            raise ValueError(f"team {team.key} picker: {e}") from e
        self.log = logging.getLogger(f"competition.picker.{team.key}")

    @abc.abstractmethod
    def pick(self, ctx: PickerContext) -> PickResult:
        """Return the ranked shortlist this team will trade."""

    def on_fills(self, results: Mapping[str, float]) -> None:
        """Feedback hook: symbol -> realised return. Only the bandit uses it."""

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state(self, blob: Mapping[str, Any]) -> None:
        ...

    # -- shared helpers ---------------------------------------------------- #

    def _rank(
        self,
        scored: Sequence[tuple[str, float, str]],
        ctx: PickerContext,
        *,
        descending: bool = True,
        max_per_sector: int | None = None,
    ) -> PickResult:
        """Sort, optionally diversify by sector, and truncate to the cap.

        Shared so every team gets the same selection mechanics -- the teams
        differ in their *scores*, not in how ties and sector limits are
        handled.
        """
        ordered = sorted(scored, key=lambda t: t[1], reverse=descending)
        symbols: list[str] = []
        scores: dict[str, float] = {}
        notes: dict[str, str] = {}
        per_sector: dict[str, int] = {}
        for sym, score, note in ordered:
            if len(symbols) >= ctx.max_symbols:
                break
            if max_per_sector is not None:
                sec = ctx.sector(sym)
                if per_sector.get(sec, 0) >= max_per_sector:
                    continue
                per_sector[sec] = per_sector.get(sec, 0) + 1
            symbols.append(sym)
            scores[sym] = score
            if note:
                notes[sym] = note
        return PickResult(tuple(symbols), scores, notes)

    def describe(self) -> str:
        lines = [f"{self.__class__.__name__} [{self.key}]"]
        if self.DESCRIPTION:
            lines.append("  " + " ".join(self.DESCRIPTION.split()))
        for name, value, doc in self.p.document():
            lines.append(f"    {name:<28} {value!r:<20} {doc}")
        return "\n".join(lines)
