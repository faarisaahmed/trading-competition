"""The Round 3 dealer. These are the most important tests in the repo.

If the draft is unfair, nothing downstream matters: the round is decided by
who was dealt the better deck. So the invariants are checked exhaustively
across many seeds, both methods, and several pool/hand shapes -- and checked
by `verify()`, which recomputes them from the result rather than trusting the
dealer's own bookkeeping.
"""

from __future__ import annotations

import statistics
from collections import Counter

import pytest

from competition.draft import (
    DraftError,
    complement_pairs,
    deal,
    feasible_sum_range,
    verify,
)
from competition.draft.rank_sum import fair_target

TEAMS = [f"team_{i}" for i in range(8)]


# --------------------------------------------------------------------------- #
# the arithmetic
# --------------------------------------------------------------------------- #


def test_fair_target_is_the_distribution_midpoint():
    # 10 picks from 1..500 -> 10 * 501/2 = 2505, exactly an integer.
    assert fair_target(500, 10) == 2505
    assert fair_target(500, 10) / 10 == 250.5 == (500 + 1) / 2


def test_feasible_range():
    lo, hi = feasible_sum_range(500, 10)
    assert lo == sum(range(1, 11)) == 55
    assert hi == sum(range(491, 501)) == 4955
    assert lo < fair_target(500, 10) < hi


def test_complement_pairs_all_sum_to_pool_plus_one():
    pairs = complement_pairs(500)
    assert len(pairs) == 250
    assert all(a + b == 501 for a, b in pairs)
    assert sorted(x for p in pairs for x in p) == list(range(1, 501))


# --------------------------------------------------------------------------- #
# the invariants, across many seeds
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["exchange", "complement"])
@pytest.mark.parametrize("seed", [0, 1, 7, 42, 99, 20260101, 123456])
def test_every_hand_has_the_same_rank_sum(pool, method, seed):
    result = deal(TEAMS, pool, method=method, seed=seed,
                  min_rank_deciles=4, max_sector_share=0.40)
    sums = {h.rank_sum for h in result.hands}
    assert len(sums) == 1, f"hands differ: {sums}"
    assert sums.pop() == result.target_sum == 2505


@pytest.mark.parametrize("method", ["exchange", "complement"])
@pytest.mark.parametrize("seed", [0, 3, 11, 77, 2026])
def test_no_symbol_or_rank_is_dealt_twice(pool, method, seed):
    result = deal(TEAMS, pool, method=method, seed=seed)
    ranks = [r for h in result.hands for r in h.ranks]
    syms = [s for h in result.hands for s in h.symbols]
    assert len(ranks) == len(set(ranks)) == 80
    assert len(syms) == len(set(syms)) == 80


@pytest.mark.parametrize("method", ["exchange", "complement"])
@pytest.mark.parametrize("seed", range(12))
def test_verify_is_clean(pool, method, seed):
    result = deal(TEAMS, pool, method=method, seed=seed,
                  min_rank_deciles=4, max_sector_share=0.40)
    problems = verify(result, pool, min_rank_deciles=4, max_sector_share=0.40)
    assert problems == []


@pytest.mark.parametrize("method", ["exchange", "complement"])
def test_every_hand_has_the_same_mean_rank(pool, method):
    result = deal(TEAMS, pool, method=method, seed=5)
    means = {round(h.mean_rank, 9) for h in result.hands}
    assert means == {250.5}


def test_ranks_map_to_the_right_symbols(pool):
    result = deal(TEAMS, pool, seed=8)
    for hand in result.hands:
        for rank, sym in zip(hand.ranks, hand.symbols, strict=True):
            assert pool.by_rank(rank).symbol == sym


def test_side_constraints_are_respected(pool):
    result = deal(TEAMS, pool, seed=13, min_rank_deciles=5, max_sector_share=0.30)
    for hand in result.hands:
        assert len(hand.deciles(result.pool_size)) >= 5
        assert hand.top_sector_share() <= 0.30 + 1e-9


def test_deals_are_reproducible(pool):
    a = deal(TEAMS, pool, seed=31)
    b = deal(TEAMS, pool, seed=31)
    assert [h.ranks for h in a.hands] == [h.ranks for h in b.hands]


def test_different_seeds_give_different_hands(pool):
    a = deal(TEAMS, pool, seed=1)
    b = deal(TEAMS, pool, seed=2)
    assert [h.ranks for h in a.hands] != [h.ranks for h in b.hands]


# --------------------------------------------------------------------------- #
# no team is systematically favoured
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["exchange", "complement"])
def test_no_team_gets_systematically_better_ranks(pool, method):
    """Over many deals, every team's rank distribution must look the same."""
    per_team: dict[str, list[int]] = {t: [] for t in TEAMS}
    for seed in range(60):
        result = deal(TEAMS, pool, method=method, seed=seed)
        for hand in result.hands:
            per_team[hand.team_key].extend(hand.ranks)
    means = [statistics.mean(v) for v in per_team.values()]
    # Fixing the sum fixes the mean exactly, for every team, every time.
    assert all(abs(m - 250.5) < 1e-9 for m in means), means
    # Spread should also be comparable -- no team reliably gets barbell hands.
    spreads = [statistics.pstdev(v) for v in per_team.values()]
    assert max(spreads) - min(spreads) < 0.15 * statistics.mean(spreads)


def test_rank_coverage_is_roughly_uniform(pool):
    """The dealer must not favour particular ranks (an early bug did)."""
    hist: Counter[int] = Counter()
    for seed in range(120):
        for hand in deal(TEAMS, pool, seed=seed).hands:
            hist.update(hand.ranks)
    assert len(hist) == 500, "some ranks were never dealt"
    expected = 120 * 80 / 500
    # Poisson noise alone gives sd ~ sqrt(19.2) ~ 4.4, so allow ~4 sd either way.
    assert max(hist.values()) < expected * 2.6
    assert min(hist.values()) > expected * 0.3


# --------------------------------------------------------------------------- #
# failure modes
# --------------------------------------------------------------------------- #


def test_rejects_a_pool_too_small_for_disjoint_hands(pool):
    with pytest.raises(DraftError, match="disjoint"):
        deal([f"t{i}" for i in range(60)], pool, picks_per_team=10)


def test_rejects_an_unreachable_target(pool):
    with pytest.raises(DraftError, match="outside the achievable range"):
        deal(TEAMS, pool, picks_per_team=10, target_sum=10)


def test_rejects_odd_hand_size_for_complement(pool):
    with pytest.raises(DraftError, match="even"):
        deal(TEAMS, pool, picks_per_team=9, method="complement")


def test_rejects_duplicate_team_keys(pool):
    with pytest.raises(DraftError, match="duplicate"):
        deal(["a", "a"], pool)


def test_rejects_a_snapshot_smaller_than_the_pool_size(pool):
    with pytest.raises(DraftError, match="pool snapshot"):
        deal(TEAMS, pool.head(50), pool_size=500)


def test_verify_catches_a_tampered_hand(pool):
    result = deal(TEAMS, pool, seed=2)
    bad = result.hands[0]
    # Swap one rank for another, breaking the sum.
    tampered = type(bad)(bad.team_key, (bad.ranks[0] + 1,) + bad.ranks[1:],
                         bad.symbols, bad.rows)
    result.hands = (tampered,) + result.hands[1:]
    problems = verify(result, pool)
    assert any("rank sum" in p for p in problems)


# --------------------------------------------------------------------------- #
# other shapes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("picks,teams,size", [(4, 5, 100), (6, 4, 60), (10, 4, 200),
                                              (2, 10, 50), (20, 4, 400)])
def test_other_pool_and_hand_shapes(pool, picks, teams, size):
    keys = [f"t{i}" for i in range(teams)]
    result = deal(keys, pool.head(size), picks_per_team=picks, pool_size=size,
                  seed=17, min_rank_deciles=2, max_sector_share=1.0)
    assert len({h.rank_sum for h in result.hands}) == 1
    assert result.target_sum == fair_target(size, picks)
    assert verify(result, pool.head(size)) == []


def test_odd_product_is_flagged_in_notes(pool):
    """k*(P+1) odd means the exact midpoint is not an integer -- say so."""
    # 3 picks from 100 -> 3 * 101 = 303, odd.
    result = deal(["a", "b"], pool.head(100), picks_per_team=3, pool_size=100,
                  seed=1, min_rank_deciles=1, max_sector_share=1.0)
    assert len({h.rank_sum for h in result.hands}) == 1
    assert any("odd" in n for n in result.notes)
