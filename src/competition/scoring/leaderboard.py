"""Season standings across all three rounds.

Total points decide the competition. The configured `tiebreakers` are applied
in order when points are level -- by default total return, then best single
round, then the shallowest drawdown, so a team that got there with less risk
edges one that got there with more.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..config import CompetitionConfig
from .points import RoundScore


@dataclass
class Standing:
    team_key: str
    name: str
    scored: bool = True
    points: float = 0.0
    rounds_played: int = 0
    total_return: float = 0.0          # compounded across rounds
    returns: dict[int, float] = field(default_factory=dict)
    places: dict[int, int | None] = field(default_factory=dict)
    round_points: dict[int, float] = field(default_factory=dict)
    max_drawdown: float = 0.0
    wins: int = 0
    rank: int | None = None

    @property
    def best_round_return(self) -> float:
        return max(self.returns.values()) if self.returns else 0.0

    @property
    def worst_round_return(self) -> float:
        return min(self.returns.values()) if self.returns else 0.0

    @property
    def average_return(self) -> float:
        return (sum(self.returns.values()) / len(self.returns)) if self.returns else 0.0

    @property
    def average_place(self) -> float:
        got = [p for p in self.places.values() if p]
        return (sum(got) / len(got)) if got else 0.0

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "team": self.team_key,
            "name": self.name,
            "points": round(self.points, 3),
            "wins": self.wins,
            "total_return": round(self.total_return, 6),
            "average_return": round(self.average_return, 6),
            "best_round": round(self.best_round_return, 6),
            "worst_round": round(self.worst_round_return, 6),
            "average_place": round(self.average_place, 2),
            "max_drawdown": round(self.max_drawdown, 6),
            "returns": {k: round(v, 6) for k, v in sorted(self.returns.items())},
            "places": dict(sorted(self.places.items())),
            "round_points": {k: round(v, 3) for k, v in sorted(self.round_points.items())},
            "scored": self.scored,
        }


@dataclass
class Leaderboard:
    standings: list[Standing] = field(default_factory=list)
    rounds: tuple[int, ...] = ()
    tiebreakers: tuple[str, ...] = ()

    @property
    def champion(self) -> Standing | None:
        scored = [s for s in self.standings if s.scored]
        return scored[0] if scored else None

    def by_team(self) -> dict[str, Standing]:
        return {s.team_key: s for s in self.standings}

    def table(self, *, title: str = "FINAL STANDINGS") -> str:
        rows = self.standings
        width = max((len(s.name) for s in rows), default=12)
        header = f"{'#':<4} {'team':<{width}} {'points':>7}"
        for r in self.rounds:
            header += f" {'R' + str(r):>8}"
        header += f" {'total':>9} {'maxDD':>8} {'wins':>5}"
        out = [title, "=" * len(header), header, "-" * len(header)]
        for s in rows:
            line = f"{(s.rank or '--'):<4} {s.name:<{width}} {s.points:>7.1f}"
            for r in self.rounds:
                ret = s.returns.get(r)
                line += f" {ret:>7.2%}" if ret is not None else f" {'--':>8}"
            line += f" {s.total_return:>8.2%} {s.max_drawdown:>8.2%} {s.wins:>5}"
            if not s.scored:
                line += "   (unscored)"
            out.append(line)
        champ = self.champion
        if champ:
            out.append("-" * len(header))
            out.append(
                f"Champion: {champ.name} -- {champ.points:.1f} points, "
                f"{champ.total_return:+.2%} compounded, {champ.wins} round win(s)."
            )
        if self.tiebreakers:
            out.append(f"Tiebreakers applied in order: {', '.join(self.tiebreakers)}.")
        return "\n".join(out)

    def to_dict(self) -> dict:
        return {
            "rounds": list(self.rounds),
            "tiebreakers": list(self.tiebreakers),
            "standings": [s.to_dict() for s in self.standings],
        }

    def markdown(self) -> str:
        """A GitHub-ready results table."""
        cols = ["#", "Team", "Points"] + [f"R{r}" for r in self.rounds] + [
            "Total", "Max DD", "Wins"
        ]
        lines = ["| " + " | ".join(cols) + " |",
                 "|" + "|".join(["---"] * len(cols)) + "|"]
        for s in self.standings:
            cells = [
                str(s.rank or "--"),
                s.name + ("" if s.scored else " *(unscored)*"),
                f"{s.points:.1f}",
            ]
            for r in self.rounds:
                ret = s.returns.get(r)
                cells.append(f"{ret:+.2%}" if ret is not None else "--")
            cells += [f"{s.total_return:+.2%}", f"{s.max_drawdown:.2%}", str(s.wins)]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)


def build_leaderboard(
    cfg: CompetitionConfig, round_scores: Sequence[RoundScore]
) -> Leaderboard:
    """Aggregate per-round scores into season standings."""
    table: dict[str, Standing] = {}
    rounds = tuple(sorted({rs.round_id for rs in round_scores}))

    for rs in round_scores:
        for s in rs.scores:
            st = table.get(s.team_key)
            if st is None:
                st = Standing(team_key=s.team_key, name=s.name, scored=s.scored)
                table[s.team_key] = st
            st.rounds_played += 1
            st.returns[rs.round_id] = s.return_pct
            st.places[rs.round_id] = s.place
            st.round_points[rs.round_id] = s.points
            st.points += s.points
            if s.place == 1 and s.scored:
                st.wins += 1
            dd = float((s.metrics or {}).get("max_drawdown", 0.0) or 0.0)
            st.max_drawdown = min(st.max_drawdown, dd)

    # Compound the round returns: each round restarts from the same bankroll,
    # so the honest "season return" is the product, not the sum.
    for st in table.values():
        total = 1.0
        for r in sorted(st.returns):
            total *= 1.0 + st.returns[r]
        st.total_return = total - 1.0

    key_funcs = {
        "total_points": lambda s: -s.points,
        "total_return": lambda s: -s.total_return,
        "best_round_return": lambda s: -s.best_round_return,
        "max_drawdown": lambda s: s.max_drawdown,          # closer to 0 is better
        "average_place": lambda s: s.average_place or 99,
        "wins": lambda s: -s.wins,
    }
    order = [cfg.tiebreakers[0]] if cfg.tiebreakers else ["total_points"]
    order = [t for t in cfg.tiebreakers if t in key_funcs] or ["total_points"]
    if "total_points" not in order:
        order.insert(0, "total_points")

    standings = sorted(
        table.values(),
        key=lambda s: tuple(key_funcs[t](s) for t in order) + (s.team_key,),
    )
    # Unscored entries are listed last and never take a rank.
    scored = [s for s in standings if s.scored]
    unscored = [s for s in standings if not s.scored]
    for i, s in enumerate(scored, 1):
        s.rank = i
    for s in unscored:
        s.rank = None

    return Leaderboard(
        standings=scored + unscored, rounds=rounds, tiebreakers=tuple(order)
    )
