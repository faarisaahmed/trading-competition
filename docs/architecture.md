# Architecture

## The tick

One engine tick, in order. The order *is* the fairness argument.

```
1. Build ONE MarketSnapshot from the shared feed.
2. Rotate the team order.
3. For each team whose own cadence is due:
     a. restrict the snapshot to that team's universe
     b. read the account, roll the session, check the kill switch
     c. call strategy.on_tick(ctx) under a wall-clock budget
     d. drain the strategy's cancel requests (and re-read the book)
     e. validate every intent through the shared guardrails
     f. submit survivors; record orders, fills, rejections
4. Snapshot equity for everyone, on the same tick.
```

Live and replay use the **same** loop, the same guardrails and the same
scoring. Only three things differ: the feed (`AlpacaFeed` vs `ReplayFeed`), the
broker (`AlpacaBroker` vs `SimulatedBroker`), and where "now" comes from
(wall clock vs the replay cursor). That is deliberate — it makes a backtest a
dress rehearsal rather than a different program.

## Layers

```
cli.py                    operator interface; wires everything together
  │
engine/runner.py          the tick loop, round lifecycle, learned-state I/O
  ├── engine/universe.py  resolves each team's universe per round/session
  ├── engine/guardrails.py pre-trade risk, identical for all teams
  ├── engine/ledger.py    SQLite record of everything
  └── scoring/            places, points, ties, season standings
  │
strategies/  pickers/     the nine competitors (the only per-team code)
  │
data/                     calendar, snapshot, feeds, synthetic generator
broker/                   Alpaca REST client, deterministic simulator
draft/                    the Round 3 dealer + independent verifier
util/indicators.py        every indicator, once
types.py                  orders, bars, quotes, fills
config.py                 typed rulebook + rules_hash
```

Dependencies point downward only. A strategy imports `types`, `util` and its
own context; it cannot reach the broker, the network, or another team.

## What a strategy can and cannot do

A strategy implements four things:

```python
tick_seconds     # how often it wants to be called
warmup_bars      # history needed before it will trade
PARAMS           # its typed parameter schema (self-documenting)
on_tick(ctx)     # -> list[OrderIntent]
```

The contract is: **read `ctx`, mutate `ctx.state`, return intents.** Do not
submit orders, sleep, or touch the network. A strategy that blocks is timed out
and loses the tick.

`StrategyContext` is the entire world it can see: an immutable snapshot
restricted to its universe, its own account, its own open orders, a seeded RNG,
a persistent state dict, and the order-construction helpers. There is no handle
to anything else.

Unknown parameters are a **hard error**. A typo in `teams.yaml` that silently
left a parameter at its default would be an invisible handicap.

### Shared plumbing

These live once, in `StrategyContext`, because getting them wrong is easy and
would silently penalise one team:

| Helper | Does |
|---|---|
| `rebalance_to_weight(sym, w)` | one order moving a position toward `w` of equity, with a no-churn tolerance |
| `allocate({sym: w})` | whole-book rebalance; sells first, credits their proceeds to the buys |
| `buy_notional / sell_qty / close / limit` | size-checked, cash-reserved, dust-avoiding order builders |
| `reserve / credit` | per-tick cash accounting so one loop cannot spend a dollar twice |
| `sym_state / set_cooldown / bar_clock` | persistent per-symbol scratch space and bar-based (not tick-based) timing |
| `request_cancel*` | pull resting orders; drained by the engine before submission |

`bar_clock` matters: the scalper ticks every 20s on 5-minute bars, so
"cooldown for 6 bars" has to count bars, or the same parameter would mean
different things to different teams.

## Data

`MarketSnapshot` carries, per symbol: primary bars (5Min), fast bars (1Min),
daily bars, a top-of-book quote, and recent news — plus the session state and
the set of symbols the feed could not price.

`AlpacaFeed` is deliberately stingy: nine teams sharing a 200 req/min budget
means bars are refetched only when a bar boundary has passed, quotes are
batched 200 at a time, and news polls on its own slower cadence. Failures log
and serve a stale cache rather than killing the round.

`ReplayFeed` serves history as of `now` with **no lookahead**: a bar stamped at
`t` becomes visible at `t + period`. Windows are memoised on the bisect index —
the engine ticks as fast as its quickest team but bars only change every five
minutes, so the same window was being rebuilt ~15× per bar (≈190M list-element
copies per round on an 80-symbol Round 3). Memoising removed essentially all of
it and changes nothing about what a strategy sees.

## Brokers

`Broker` is a small abstract interface: account, positions, submit, open
orders, cancel, close. Two implementations:

* **`AlpacaBroker`** — written against Alpaca's documented REST API rather than
  the SDK, so retry and rate-limit behaviour is explicit (a per-client token
  bucket at 180/min against the documented 200). Insufficient-funds and
  rejection responses are mapped to typed exceptions the engine handles rather
  than crashes on.
* **`SimulatedBroker`** — deterministic matching. Market orders fill at
  ask/bid ± slippage; limit orders fill when the quote crosses, passing through
  price improvement; `day` orders expire at the close; an optional
  participation cap models partial fills. Every quantity is **floored**, never
  rounded, so an account deploying its last dollar cannot overdraw.

## Ledger

One SQLite file per competition (`runs/competition.sqlite`), WAL mode so
`comp report` can read while a round is still writing. Tables: `runs`, `teams`,
`drafts`, `universes`, `orders`, `fills`, `equity`, `rejections`, `events`,
`results`, `learned_state`.

Two design rules:

* **Every write is best-effort.** A ledger failure logs and continues.
  Recording the race must never be the reason a runner trips.
* **Orders carry the strategy's own stated reason.** `"momentum entry (score
  0.63, adx 40)"`, `"pair rotation AAPL/MSFT z=+1.84 (hold cheap leg)"`. That
  is what makes a post-mortem possible.

`learned_state` is why the RL entries work across rounds: the Q-table and the
bandit model are saved at round end and restored at the next round's start.

## Configuration

`config/competition.yaml` is the rulebook — rounds, points, risk rails,
fairness controls, data settings. `config/teams.yaml` is the field. Nothing is
hard-coded in the engine.

`CompetitionConfig.rules_hash` is a 16-hex-char hash of the effective rules,
stamped into every run. If the rules change mid-season, the results tell you.

Validation is aggressive and happens at startup: unknown keys, an ascending
points table, a points table shorter than the field, two teams sharing an
`env_prefix` (they would trade the same account), a draft pool too small for
disjoint hands, a picker cap above the dealt hand size, a malformed timeframe.
A bad rulebook should fail before the round, not at 3pm on day four.

## Testing

656 tests, all offline.

| File | Covers |
|---|---|
| `test_draft.py` | 72 tests attacking the fairness constraint across many seeds, both methods, several pool shapes, and every failure mode |
| `test_feed.py` | no lookahead, at every tick, for every symbol and timeframe |
| `test_guardrails.py` | every rejection path; long-only and no-overdraw as invariants |
| `test_strategies.py` | all nine teams driven over real synthetic sessions; no exceptions, only legal intents, and each one actually trades |
| `test_pickers.py` | mechanical contract *and* the philosophical one — each picker must select the archetype its strategy trades, on a pool with planted regimes |
| `test_engine.py` | full rounds: equal bankroll, identical snapshots, universe enforcement, a broken strategy cannot stop the round, reproducibility, learned state across rounds |
| `test_scoring.py` | tie handling, points conservation, season tiebreaks |
| `test_indicators.py` | discrimination between regimes, plus every function against empty/constant/NaN input |
| `test_lexicon.py` | sentiment signs, negation reaching phrases, two-sided modifiers |
| `test_calendar.py` | holidays and Easter against published dates |
| `test_broker.py` | fill model, oversell, overdraw, wire-format parsing |
| `test_cli.py` | every subcommand |
| `test_config.py` | rulebook validation |
