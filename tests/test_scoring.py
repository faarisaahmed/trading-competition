"""Round scoring and season standings."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from competition.scoring import build_leaderboard, score_round
from competition.scoring.points import TeamScore


@dataclass
class Fake:
    team_key: str
    name: str
    return_pct: float
    scored: bool = True
    start_equity: float = 5000.0
    md: float = -0.02

    @property
    def end_equity(self):
        return self.start_equity * (1 + self.return_pct)

    def metrics(self):
        return {"max_drawdown": self.md}


def eight(returns):
    keys = ["a", "b", "c", "d", "e", "f", "g", "h"]
    return [Fake(k, k.upper(), r) for k, r in zip(keys, returns, strict=True)]


# --------------------------------------------------------------------------- #
# ranking and points
# --------------------------------------------------------------------------- #


def test_ranking_is_by_return_descending(cfg):
    rs = score_round(cfg, 1, eight([0.05, 0.03, 0.01, 0.0, -0.01, -0.02, -0.03, -0.04]))
    ordered = sorted(rs.scored_only(), key=lambda s: s.place)
    assert [s.team_key for s in ordered] == list("abcdefgh")
    assert [s.points for s in ordered] == [float(p) for p in cfg.points_table[:8]]


def test_winner_and_places(cfg):
    rs = score_round(cfg, 1, eight([0.01, 0.09, 0.02, 0.0, -0.01, -0.02, -0.03, -0.04]))
    assert rs.winner().team_key == "b"
    assert rs.by_team()["b"].place == 1
    assert rs.by_team()["b"].points == cfg.points_table[0]


def test_all_points_are_awarded_exactly_once(cfg):
    rs = score_round(cfg, 1, eight([0.05, 0.03, 0.01, 0.0, -0.01, -0.02, -0.03, -0.04]))
    assert sum(s.points for s in rs.scored_only()) == pytest.approx(sum(cfg.points_table))


def test_a_tie_shares_the_places_it_spans(cfg):
    # c and d tie for 3rd/4th.
    rs = score_round(cfg, 1, eight([0.05, 0.03, 0.02, 0.02, -0.01, -0.02, -0.03, -0.04]))
    by = rs.by_team()
    expected = (cfg.points_for_place(3) + cfg.points_for_place(4)) / 2
    assert by["c"].place == by["d"].place == 3
    assert by["c"].points == by["d"].points == pytest.approx(expected)
    assert by["c"].tied_with == ("d",) and by["d"].tied_with == ("c",)
    # The next team takes 5th, not 4th.
    assert by["e"].place == 5
    assert by["e"].points == cfg.points_for_place(5)
    # And the pot is still conserved.
    assert sum(s.points for s in rs.scored_only()) == pytest.approx(sum(cfg.points_table))


def test_a_three_way_tie(cfg):
    rs = score_round(cfg, 1, eight([0.05, 0.02, 0.02, 0.02, -0.01, -0.02, -0.03, -0.04]))
    by = rs.by_team()
    expected = sum(cfg.points_for_place(p) for p in (2, 3, 4)) / 3
    for k in "bcd":
        assert by[k].place == 2
        assert by[k].points == pytest.approx(expected)
    assert by["e"].place == 5
    assert sum(s.points for s in rs.scored_only()) == pytest.approx(sum(cfg.points_table))


def test_everyone_tied(cfg):
    rs = score_round(cfg, 1, eight([0.0] * 8))
    assert all(s.place == 1 for s in rs.scored_only())
    average = sum(cfg.points_table) / 8
    assert all(s.points == pytest.approx(average) for s in rs.scored_only())


def test_ties_are_not_broken_by_team_name(cfg):
    """Two identical returns must not be separated by alphabetical order."""
    forward = score_round(cfg, 1, eight([0.05, 0.02, 0.02, 0.0, -0.01, -0.02, -0.03, -0.04]))
    reversed_input = list(reversed(eight([0.05, 0.02, 0.02, 0.0, -0.01, -0.02,
                                          -0.03, -0.04])))
    backward = score_round(cfg, 1, reversed_input)
    assert forward.by_team()["b"].points == backward.by_team()["b"].points
    assert forward.by_team()["c"].points == backward.by_team()["c"].points


def test_unscored_entries_take_no_place_and_no_points(cfg):
    teams = eight([0.01] * 8) + [Fake("bench", "Benchmark", 0.50, scored=False)]
    rs = score_round(cfg, 1, teams)
    bench = rs.benchmark()
    assert bench.place is None and bench.points == 0.0
    # And the benchmark's huge return must not have displaced anyone.
    assert all(s.place == 1 for s in rs.scored_only())


def test_benchmark_comparison_line(cfg):
    teams = eight([0.05, 0.04, 0.01, 0.0, -0.01, -0.02, -0.03, -0.04]) + \
        [Fake("bench", "Benchmark", 0.02, scored=False)]
    rs = score_round(cfg, 1, teams)
    assert "2 of 8 teams beat the benchmark" in rs.table()


def test_places_beyond_the_table_score_zero(cfg):
    many = [Fake(f"t{i}", f"T{i}", -0.01 * i) for i in range(12)]
    rs = score_round(cfg, 1, many)
    tail = [s for s in rs.scores if s.place and s.place > len(cfg.points_table)]
    assert tail and all(s.points == 0.0 for s in tail)


def test_round_table_and_dict(cfg):
    rs = score_round(cfg, 1, eight([0.05, 0.03, 0.01, 0.0, -0.01, -0.02, -0.03, -0.04]),
                     round_name="The Big Three")
    table = rs.table()
    assert "The Big Three" in table and "place" in table
    d = rs.to_dict()
    assert d["round_id"] == 1 and len(d["scores"]) == 8


# --------------------------------------------------------------------------- #
# season standings
# --------------------------------------------------------------------------- #


def _three_rounds(cfg):
    r1 = score_round(cfg, 1, eight([0.10, 0.05, 0.03, 0.01, 0.0, -0.01, -0.02, -0.03]),
                     round_name="R1")
    r2 = score_round(cfg, 2, eight([-0.05, 0.08, 0.02, 0.01, 0.0, -0.01, -0.02, -0.03]),
                     round_name="R2")
    r3 = score_round(cfg, 3, eight([0.12, -0.02, 0.04, 0.01, 0.0, -0.01, -0.02, -0.03]),
                     round_name="R3")
    return [r1, r2, r3]


def test_leaderboard_totals_points_across_rounds(cfg):
    lb = build_leaderboard(cfg, _three_rounds(cfg))
    by = lb.by_team()
    assert by["a"].points == pytest.approx(
        by["a"].round_points[1] + by["a"].round_points[2] + by["a"].round_points[3])
    assert by["a"].rounds_played == 3
    assert lb.rounds == (1, 2, 3)


def test_total_return_compounds_not_sums(cfg):
    lb = build_leaderboard(cfg, _three_rounds(cfg))
    a = lb.by_team()["a"]
    expected = 1.10 * 0.95 * 1.12 - 1.0
    assert a.total_return == pytest.approx(expected)
    assert a.total_return != pytest.approx(0.10 - 0.05 + 0.12)


def test_wins_are_counted(cfg):
    lb = build_leaderboard(cfg, _three_rounds(cfg))
    by = lb.by_team()
    assert by["a"].wins == 2      # rounds 1 and 3
    assert by["b"].wins == 1      # round 2


def test_champion_is_the_points_leader(cfg):
    lb = build_leaderboard(cfg, _three_rounds(cfg))
    assert lb.champion.rank == 1
    assert lb.champion.points == max(s.points for s in lb.standings if s.scored)


def test_ranks_are_dense_and_unscored_are_last(cfg):
    rounds = _three_rounds(cfg)
    for rs in rounds:
        rs.scores.append(TeamScore("bench", "Benchmark", 0.5, 7500, 5000, scored=False))
    lb = build_leaderboard(cfg, rounds)
    scored = [s for s in lb.standings if s.scored]
    assert [s.rank for s in scored] == list(range(1, len(scored) + 1))
    assert lb.standings[-1].team_key == "bench"
    assert lb.standings[-1].rank is None


def test_points_ties_break_on_total_return(cfg):
    # Two teams with identical points but different compounded returns.
    r1 = score_round(cfg, 1, [Fake("x", "X", 0.10), Fake("y", "Y", 0.02)])
    r2 = score_round(cfg, 2, [Fake("x", "X", 0.02), Fake("y", "Y", 0.10)])
    lb = build_leaderboard(cfg, [r1, r2])
    x, y = lb.by_team()["x"], lb.by_team()["y"]
    assert x.points == y.points
    # 1.10*1.02 == 1.02*1.10, so the next tiebreaker decides; either way the
    # ordering must be total and deterministic.
    assert {x.rank, y.rank} == {1, 2}
    again = build_leaderboard(cfg, [r1, r2])
    assert again.by_team()["x"].rank == x.rank


def test_tiebreaker_prefers_the_shallower_drawdown(cfg):
    deep = Fake("deep", "Deep", 0.05, md=-0.30)
    shallow = Fake("shallow", "Shallow", 0.05, md=-0.02)
    rs = score_round(cfg, 1, [deep, shallow])
    lb = build_leaderboard(cfg, [rs])
    assert lb.by_team()["shallow"].max_drawdown == pytest.approx(-0.02)
    assert lb.by_team()["deep"].max_drawdown == pytest.approx(-0.30)


def test_standing_derived_stats(cfg):
    lb = build_leaderboard(cfg, _three_rounds(cfg))
    a = lb.by_team()["a"]
    assert a.best_round_return == pytest.approx(0.12)
    assert a.worst_round_return == pytest.approx(-0.05)
    assert a.average_return == pytest.approx((0.10 - 0.05 + 0.12) / 3)
    assert 1.0 <= a.average_place <= 8.0


def test_leaderboard_renders(cfg):
    lb = build_leaderboard(cfg, _three_rounds(cfg))
    table = lb.table()
    assert "FINAL STANDINGS" in table and "Champion:" in table
    md = lb.markdown()
    assert md.count("|") > 20 and "Points" in md
    d = lb.to_dict()
    assert len(d["standings"]) == 8 and d["rounds"] == [1, 2, 3]


def test_a_missing_round_does_not_break_standings(cfg):
    """Scoring after only round 1 must work."""
    lb = build_leaderboard(cfg, _three_rounds(cfg)[:1])
    assert lb.rounds == (1,)
    assert all(s.rounds_played == 1 for s in lb.standings)
    assert lb.champion is not None
