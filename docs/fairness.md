# Fairness

A competition whose result could be explained by the setup rather than the
strategies is not a competition. This document states what is guaranteed, how,
and what is deliberately left to luck.

## 1. The Round 3 draft

### The constraint

Rank the pool 1..P by value (1 = most valuable). Deal each team `k` tickers.
Require every hand's rank sum to be identical.

### Why that number

For `k` picks drawn from ranks `1..P`, subset sums range over

```
    lo = k(k+1)/2                    (the k best ranks)
    hi = k(2P-k+1)/2                 (the k worst ranks)
```

and the midpoint of the distribution is

```
    T = k(P+1)/2
```

For the shipped configuration (`k=10`, `P=500`) that is `10 × 501 / 2 = 2505`,
an integer — so it is exactly achievable. Every hand therefore has mean rank
`2505/10 = 250.5 = (P+1)/2`, which is the mean rank of the whole pool.

**What this buys:** no team is dealt a systematically larger- or smaller-cap
deck. Rank is a monotone proxy for company size, so equalising mean rank
equalises the central tendency of hand "quality".

**What it does not buy:** equal *shape*. `{1,2,3,4,5,496,497,498,499,500}` and
`{246..255}` both sum to 2505. The first hand is five mega-caps and five
small-caps; the second is ten middling names. They will behave very
differently. That is the intended chaos.

If `k(P+1)` is odd the exact midpoint is not an integer. The dealer then uses
`floor(k(P+1)/2)`, every hand still shares that sum, and the deviation is
reported in `DraftResult.notes` rather than hidden.

### Two dealing methods

**`exchange` (default).** Sample `k` ranks uniformly, then repair the sum with
exchange moves: replace a held rank `h` with a free rank `c`, changing the sum
by `c - h`. A single move usually closes the gap exactly; when it cannot, the
dealer takes the move that gets closest and iterates, which converges
monotonically in `|delta|`.

Repair leaves a fingerprint: the ranks the algorithm reached for to close gaps
get dealt far more often than chance. The first version had a 9x spread in
per-rank deal frequency. The fix is a **mixing walk** — a few hundred
sum-preserving two-for-two swaps (`{a,b} → {c,d}` with `a+b = c+d`) after the
repair. This is an irreducible random walk on the space of `k`-subsets with
fixed sum, so it converges toward uniform over exactly the right space while
preserving the invariant at every step. Measured result: per-rank frequency
spread drops from 9x to 2.2x, against a Poisson floor of about 1.9x for the
same sample size.

**`complement`.** Deal `k/2` complementary pairs `(r, P+1-r)`. Each pair sums
to `P+1`, so a hand of `k/2` pairs sums to `k(P+1)/2 = T` *by construction*.
This method cannot fail, needs no repair, and gives every hand an identical
rank-symmetry profile — the strictest fairness available, at the cost of
variety (every hand is five high ranks and five low ones). Requires even `k`.

### Extra rails

Fixing the sum alone allows degenerate hands. Two bounds are applied, both
repaired with sum-neutral swaps so the invariant survives:

* `min_rank_deciles: 4` — a hand must touch at least four different tenths of
  the pool, so nobody gets ten adjacent ranks.
* `max_sector_share: 0.40` — at most 4 of 10 from one sector, so "rank-sum
  fair" cannot accidentally mean "all ten are banks".

### Verification

`verify()` recomputes every invariant from the `DraftResult` alone —
per-hand sums, disjointness across teams, rank→symbol consistency, decile
coverage, sector caps. It is written without reference to the dealing code, so
a bug in the dealer cannot hide behind a shared helper. `deal()` calls it
before returning and raises if it fails; `comp verify-draft` re-runs it against
the recorded draft at any later time.

The dealer is also seeded and reproducible: the same seed always produces the
same hands, and the seed is recorded in the ledger.

### Statistical check

`tests/test_draft.py` runs 60–120 deals per method and asserts:

* every hand's sum equals the target, every time;
* every team's mean rank is exactly 250.5 across all deals;
* per-team rank *spread* differs by less than 15%, so no team reliably gets
  barbell hands;
* all 500 ranks get dealt, with frequency inside Poisson-plausible bounds;
* infeasible requests raise rather than silently producing an unfair deal.

## 2. Identical information

Exactly one `MarketSnapshot` is constructed per engine tick, from one feed, and
the same object is passed to every team. A strategy receives data structures —
never a network handle — so it *cannot* fetch anything, refresh anything, or
see a print later than another team.

Before a team sees the snapshot it is passed through
`MarketSnapshot.restricted_to(team_universe)`. In Round 3 this means a team
cannot read a price, a bar or a news item for any ticker outside its dealt
hand. Universe enforcement is a property of the data layer, not just a check at
order entry.

In replay mode the same guarantee holds, plus one more: a bar stamped at `t` is
only visible at `t + period`. Without that single line every strategy would see
the future. `tests/test_feed.py` asserts it across every symbol, timeframe and
tick in a four-session window.

## 3. Identical execution plumbing

The things that are easy to get subtly wrong — and which would silently
handicap whichever team's author got them wrong — live once, in
`StrategyContext`, and are shared:

* target weight → dollar delta → share count, with fractional handling;
* the venue minimum notional, and never leaving a sub-minimum remainder that
  can no longer be sold;
* per-tick cash reservation, so filling three position slots in one loop does
  not triple-spend the same dollar;
* crediting same-tick market-sell proceeds (which the engine submits first) but
  *not* resting limit sells, which may never fill;
* flooring, never rounding, every spend to the cent.

Each of these was a real bug caught in the dry run. The `insufficient_cash`
rejection wall that three teams were hitting was one rounding direction:
`round(993.415, 2)` is one cent more than the cash on hand.

## 4. Identical risk rails

Every order from every team passes the same `Guardrails.validate`, driven
entirely by `competition.risk` in the rulebook — universe membership,
tradability, quote freshness, market hours, rate limits, no shorting, no
overselling (including cumulative offers across a batch and against resting
orders), limit-price sanity, notional bounds, buying power, position cap,
leverage cap.

Rejections are counted per team per reason and written to the ledger.
`comp report` prints the tally. A strategy repeatedly attempting something
illegal is visible rather than silently degraded.

## 5. Identical compute

Each `on_tick` gets `fairness.strategy_timeout_seconds` of wall clock. An
overrun is recorded as a timeout and **the tick's intents are discarded** — it
costs that team the tick and affects nobody else. The tick is not killed
mid-decision, because that risks leaving half-applied strategy state.

## 6. Order of action

With `fairness.rotate_execution_order`, the team order rotates each tick, so no
team is permanently first to act on a new snapshot.

## 7. Equal bankroll

`comp doctor --check-accounts` reads all nine live accounts and **refuses to
start a round** if their equities differ by more than 1% of the bankroll.
`comp run` repeats the check and requires `--force` to override.

A live account's balance cannot be set programmatically, so `AlpacaBroker`
flattens the book at round start and records the *baseline equity* it actually
has. Round return is measured against that baseline, so a small residual
difference does not advantage anyone — but it is logged loudly, because the
right fix is to reset the paper accounts.

## 8. Reproducibility

* Every team's RNG is seeded from `(competition_seed, team_key, round_id)`.
* The draft seed is recorded.
* `CompetitionConfig.rules_hash` is a hash of the effective rules, stamped into
  every run, so results are always tied to the exact rules in force.
* The ledger records the dealt hands, each team's universe per session, every
  order with the strategy's own stated reason, every fill, every rejection, and
  equity snapshots on a fixed cadence.

`tests/test_engine.py::test_the_same_seed_reproduces_the_same_result` asserts
byte-identical outcomes across two runs.

## What is *not* equalised, on purpose

* **Hand shape in Round 3.** Discussed above. This is the "luck" the brief
  asked for.
* **Strategy cadence.** The scalper evaluates every 20s, the benchmark every
  15 minutes. That is a design choice each team makes and lives with, not an
  advantage — the fast team pays more spread.
* **Number of positions, concentration, aggression.** All configured per team.
  The Gambler may go 90% into one name; that is its entry.
* **The market itself.** Round 2's pickers may legitimately select different
  names with different luck attached. Selection *is* the skill being measured
  in Round 2.
