# Runbook

## One-time setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # then paste in nine key pairs
comp doctor --check-accounts
```

`doctor` is the gate. It must show no `ERROR` lines before you run anything
live. In particular it fails if:

* any account is a **live** account rather than paper;
* trading is blocked on an account;
* the nine accounts' equities differ by more than 1% of the bankroll.

### Equalising the accounts

Alpaca paper balances cannot be set over the API. In the Alpaca dashboard, use
**Reset Account** on each paper account and set the starting balance to the
same value for all nine. `$5,000` matches the rulebook exactly; any equal value
works, since round return is measured per account from its own baseline.

### Pre-train the RL entry

Once, before Round 1. This is the coding stage, so it is permitted — and it is
what makes an RL entry viable over three weeks rather than three years.

```bash
comp pretrain --source alpaca --days 180 --passes 4
```

Prints the learned policy table and saves it to the ledger. `comp run` restores
it automatically. Re-running with `--resume` continues from the saved state.

---

## Each round

### The day before

```bash
comp doctor --check-accounts        # must be clean
comp backtest --round N --start <last week> --source alpaca
```

The dry run on *real recent bars* is the last chance to catch a data problem.
Check that every team trades and that the `rejections` column is empty or
trivial.

### Round 3 only, before the round

```bash
comp refresh-universe --metric market_cap   # re-price and re-rank the pool
comp universe --live                        # drop untradable names, renumber
comp draft --round 3 --live --write         # deal, verify, record
```

Read the fairness proof it prints. Every hand must show `sum=2505`, the mean
ranks must all be `250.5`, and it must say `no overlap` and
`independent verification: CLEAN`. The deal is now in the ledger and
`comp verify-draft --round 3` can re-check it at any time.

### Starting the round

```bash
comp run --round 1 --start 2026-09-21
```

The process:

* resets every account (cancels orders, flattens positions) and records each
  team's baseline equity;
* resolves universes — fixed for Round 1, pickers for Round 2, the dealt hand
  for Round 3;
* primes the data feed, then ticks until the window closes;
* liquidates 10 minutes before the final close, scores, and writes results.

Run it under something that survives a disconnect:

```bash
nohup comp run --round 1 --start 2026-09-21 -v > runs/round1.log 2>&1 &
```

Or `screen` / `tmux`. It sleeps through nights and weekends on its own, so a
single invocation covers the whole seven days — and because it checkpoints,
losing the process is recoverable rather than fatal (see Recovery below).

### During the round

You should not need to do anything. To watch:

```bash
open runs/dashboard.html             # self-refreshing, leave it open all week
comp status                          # text snapshot: round, day, clock, standings
tail -f runs/round1.log
comp report --trades 20              # per-team detail from the ledger
comp report --team gambler --trades 50
```

`comp dashboard --open` re-renders it from the ledger at any time, including
after a crash or once the round is over.

What to watch:

| Symptom | Meaning |
|---|---|
| a team with **0 fills** after a full session | its picker returned nothing, or its warm-up is not satisfied |
| a rising `rejections` tally | a strategy is repeatedly attempting something illegal |
| `strategy_timeout` events | a strategy is overrunning its budget and losing ticks |
| `kill_switch` event | a team was down 35% intraday and is sitting out until tomorrow |
| repeated `stale_quote` / `no_quote` | a data-feed problem, not a strategy problem |

### After the round

```bash
comp score --round 1
comp leaderboard
comp leaderboard --write RESULTS.md      # commit this
```

---

## Recovery

**The process died mid-round.** Restart it with the same arguments. It will
detect the in-progress round and rejoin it:

```
$ comp run --round 1 --start 2026-09-21

  RESUMING round 1 from run 1af94fad64df4f22: 217 ticks already done,
  last checkpoint 2026-09-24 14:43 UTC.
  Accounts will NOT be reset; positions and strategy state are restored
  from the checkpoint.
```

What comes back: each team's **original baseline equity** (so returns are still
measured from the right zero), its strategy state (trailing stops, entry
prices, cooldowns, the gambler's ladder rung, the pair trader's fitted betas),
its risk state (session-open equity for the kill switch, the daily order
count, whether it is halted), and the set of fills already processed so nothing
is double-counted. Positions are not restored from the checkpoint — they live
in the broker account, which is the authority.

A checkpoint is written on every equity snapshot (five minutes by default), so
a crash costs at most one snapshot interval of engine memory.

**A resume can be refused.** If the rulebook hash, the round window or the team
list has changed since the checkpoint, the resume is declined with a reason
rather than silently producing a different competition:

```
  Found an in-progress round 1 but cannot resume it: the rulebook changed
  since the checkpoint (a7f0635 -> 3b91c2e).
  Pass --fresh to abandon it and restart the round from scratch.
```

**To deliberately restart a round**, `comp run --round N --fresh`. That
flattens every account and starts from zero. It is never the default, because
doing it by accident costs a week.

**What a resume does not fix.** The market moved while you were down. A
position held through a two-hour outage gets whatever price exists on restart,
and its stops did not run in between. The round is salvaged, not rewound —
note the outage alongside the results.

**One team's credentials are broken.** `comp run --allow-missing` runs only the
funded teams. Note it in the results: a round with seven entries is not the same
round.

**A data outage.** The feed logs the failure and serves its stale cache, so
strategies see old bars rather than none. Guardrails refuse to trade on quotes
older than `max_quote_age_seconds` (120s by default), so the practical effect is
that everybody stops trading until data returns. That is the correct behaviour
and it is equal for all teams.

**The results look wrong.** Everything is in the ledger:

```bash
sqlite3 runs/competition.sqlite \
  "SELECT ts, symbol, side, qty, price FROM fills WHERE team_key='gambler' ORDER BY ts"
sqlite3 runs/competition.sqlite \
  "SELECT reason, COUNT(*) FROM rejections WHERE team_key='scalper' GROUP BY reason"
sqlite3 runs/competition.sqlite \
  "SELECT team_key, kind, message FROM events ORDER BY ts"
```

---

## Rules hygiene

`config/*.yaml` is frozen once Round 1 starts. The `rules_hash` in every run
record will show if it changed.

If something genuinely has to change mid-season — a bug that makes a round
unrunnable — change it, note it in `RESULTS.md`, and say which rounds ran under
which hash. Silently retuning a parameter between rounds is how a competition
stops being one.
