"""Round scoring: return -> place -> points.

The rule from the brief: "at the end of the seven days, each team is ranked
from most money they made to least. the better you did the more points you
get." So ranking is on realised round return, and points come from the
configured table.

Two details that matter and are easy to get wrong:

**Ties.** Two teams with an identical return must not be separated by
dictionary order. Under `tie_policy: average` they share the places they span
and each receives the mean of those places' points -- the standard
competition convention, and the only one that cannot be gamed. The next team
down then takes the place *after* the whole tied block.

**Unscored entries.** The benchmark appears in every table but takes no place
and no points, and does not push a real team down the board.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from ..config import CompetitionConfig


@dataclass
class TeamScore:
    """One team's scored outcome for one round."""

    team_key: str
    name: str
    return_pct: float
    end_equity: float
    start_equity: float
    scored: bool = True
    place: int | None = None
    points: float = 0.0
    tied_with: tuple[str, ...] = ()
    metrics: dict = field(default_factory=dict)

    @property
    def pnl(self) -> float:
        return self.end_equity - self.start_equity

    def to_dict(self) -> dict:
        return {
            "team": self.team_key,
            "name": self.name,
            "return_pct": round(self.return_pct, 6),
            "pnl": round(self.pnl, 2),
            "start_equity": round(self.start_equity, 2),
            "end_equity": round(self.end_equity, 2),
            "place": self.place,
            "points": round(self.points, 3),
            "scored": self.scored,
            "tied_with": list(self.tied_with),
        }


@dataclass
class RoundScore:
    round_id: int
    round_name: str
    scores: list[TeamScore] = field(default_factory=list)

    def scored_only(self) -> list[TeamScore]:
        return [s for s in self.scores if s.scored]

    def winner(self) -> TeamScore | None:
        ranked = [s for s in self.scored_only() if s.place == 1]
        return ranked[0] if ranked else None

    def by_team(self) -> dict[str, TeamScore]:
        return {s.team_key: s for s in self.scores}

    def benchmark(self) -> TeamScore | None:
        for s in self.scores:
            if not s.scored:
                return s
        return None

    def table(self) -> str:
        rows = sorted(
            self.scores,
            key=lambda s: (s.place is None, s.place or 0, -s.return_pct),
        )
        width = max((len(s.name) for s in rows), default=12)
        out = [
            f"Round {self.round_id} -- {self.round_name}",
            f"{'place':<6} {'team':<{width}} {'return':>9} {'P&L':>11} "
            f"{'end equity':>12} {'points':>7}",
            "-" * (width + 50),
        ]
        for s in rows:
            place = f"{s.place}" if s.place else "--"
            tie = "  =" if s.tied_with else ""
            out.append(
                f"{place:<6} {s.name:<{width}} {s.return_pct:>8.2%} {s.pnl:>+11,.2f} "
                f"{s.end_equity:>12,.2f} {s.points:>7.1f}{tie}"
                + ("" if s.scored else "  (unscored)")
            )
        bench = self.benchmark()
        if bench:
            beat = sum(1 for s in self.scored_only() if s.return_pct > bench.return_pct)
            out.append(
                f"\n{beat} of {len(self.scored_only())} teams beat the benchmark "
                f"({bench.return_pct:+.2%})."
            )
        return "\n".join(out)

    def to_dict(self) -> dict:
        return {
            "round_id": self.round_id,
            "round_name": self.round_name,
            "scores": [s.to_dict() for s in self.scores],
        }


def score_round(
    cfg: CompetitionConfig,
    round_id: int,
    results: Iterable,
    *,
    round_name: str = "",
) -> RoundScore:
    """Rank `results` by return and assign places and points.

    `results` may be `TeamResult` objects from the engine or any object with
    `team_key`, `name`, `scored`, `return_pct`, `start_equity`, `end_equity`.
    """
    scores: list[TeamScore] = []
    for r in results:
        scores.append(TeamScore(
            team_key=r.team_key,
            name=getattr(r, "name", r.team_key),
            return_pct=float(r.return_pct),
            end_equity=float(r.end_equity),
            start_equity=float(r.start_equity),
            scored=bool(getattr(r, "scored", True)),
            metrics=r.metrics() if callable(getattr(r, "metrics", None)) else {},
        ))

    contenders = [s for s in scores if s.scored]
    # Sort by return, descending. The secondary key is only there to make the
    # *order* deterministic; ties are then detected on return alone and
    # resolved by the tie policy, so the key never decides who wins.
    contenders.sort(key=lambda s: (-s.return_pct, s.team_key))

    place = 1
    i = 0
    n = len(contenders)
    while i < n:
        # Collect everyone tied at this return (to the cent, to avoid float noise).
        block = [contenders[i]]
        j = i + 1
        while j < n and abs(contenders[j].return_pct - contenders[i].return_pct) < 1e-9:
            block.append(contenders[j])
            j += 1

        places = list(range(place, place + len(block)))
        pot = [cfg.points_for_place(p) for p in places]
        if len(block) == 1:
            awarded = [float(pot[0])]
        elif cfg.tie_policy == "average":
            awarded = [sum(pot) / len(pot)] * len(block)
        elif cfg.tie_policy == "best":
            awarded = [float(max(pot))] * len(block)
        else:                                      # "worst"
            awarded = [float(min(pot))] * len(block)

        keys = tuple(s.team_key for s in block)
        for s, pts in zip(block, awarded, strict=True):
            s.place = places[0]       # a tied block shares the first place number
            s.points = pts
            s.tied_with = tuple(k for k in keys if k != s.team_key)

        place += len(block)
        i = j

    return RoundScore(round_id=round_id, round_name=round_name, scores=scores)
