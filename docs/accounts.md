# Getting nine Alpaca accounts

The competition needs one paper account per team, so each trades its own
portfolio. Here is the least painful way to get there, and the honest
trade-offs of each route.

## The key fact: 3 paper accounts per login

Alpaca's dashboard lets you create and delete paper accounts under a single
login, **capped at 3**. Alpaca staff have confirmed on their forum that
creating additional logins for more accounts is within the terms of service,
and that native sub-accounts were in beta as of August 2026 — worth asking
support about before you do it the manual way.

So for nine teams:

| Route | Signups | Key pairs | Isolation | Code changes |
|---|---|---|---|---|
| **3 logins × 3 accounts** *(recommended)* | 3 | 9 | full | none |
| 9 logins × 1 account | 9 | 9 | full | none |
| 1 login × 3 accounts, 3 teams | 1 | 3 | full | none (but a 3-team field) |
| 1 account, virtual sub-books | 1 | 1 | software-enforced | a partitioning layer |

**Three logins is the sweet spot.** It is three signups instead of nine, gives
every team a genuinely separate account, and needs no changes to this
codebase. There is a second, less obvious benefit: Alpaca's market-data rate
limits are enforced **per owner, not per account**, so spreading nine accounts
over three logins triples your request headroom. That matters — nine teams
polling one owner's 200 req/min budget is tight, and the scalper alone ticks
every 20 seconds.

## Step by step

### 1. Create the logins

Three sign-ups at [app.alpaca.markets](https://app.alpaca.markets). Use three
email addresses you control. Gmail's `you+r1@gmail.com` aliasing sometimes
works and sometimes gets normalised away; if Alpaca rejects an alias, use
distinct addresses.

### 2. Create three paper accounts under each

In the dashboard, click the paper account number in the upper left and choose
**Open New Paper Account**. Repeat until you have three per login, nine total.

**Set the starting balance when you create each one.** The default is
$100,000, and you cannot change it afterwards without resetting the account.
Two workable choices:

* **$5,000 each** — matches `starting_cash` in the rulebook exactly, and the
  reports read naturally.
* **Leave them at $100,000** — also fine. Every strategy sizes by *weight of
  equity*, not absolute dollars, and the order-size cap is expressed as a
  multiple of the bankroll rather than a fixed figure. Just set
  `competition.starting_cash: 100000` in `config/competition.yaml` so the
  reports match. `comp setup-accounts` will tell you if these disagree.

What matters far more than the value is that **all nine are identical**.
Unequal bankrolls are the one condition that invalidates a round, and both
`comp setup-accounts` and `comp doctor --check-accounts` refuse to proceed
when they differ by more than 1%.

### 3. Name them so you can tell them apart

Nine paper accounts identified only by account number is how keys end up
against the wrong team. Label each one in the dashboard (or just note the
account number beside the name in your scratch file):

| # | Dashboard label | Team | Env prefix |
|---|---|---|---|
| 1 | `comp-1-trend_rider` | Trend Rider | `ALPACA_TREND_RIDER` |
| 2 | `comp-2-mean_reverter` | Mean Reverter | `ALPACA_MEAN_REVERTER` |
| 3 | `comp-3-q_learner` | Q-Learner | `ALPACA_Q_LEARNER` |
| 4 | `comp-4-gambler` | The Gambler | `ALPACA_GAMBLER` |
| 5 | `comp-5-stat_arb` | Stat Arb | `ALPACA_STAT_ARB` |
| 6 | `comp-6-news_hound` | News Hound | `ALPACA_NEWS_HOUND` |
| 7 | `comp-7-vol_breakout` | Vol Breakout | `ALPACA_VOL_BREAKOUT` |
| 8 | `comp-8-scalper` | The Scalper | `ALPACA_SCALPER` |
| 9 | `comp-9-benchmark` | Buy & Hold *(unscored)* | `ALPACA_BENCHMARK` |

The numbering matches the roster order in `config/teams.yaml`, which is also
the order `comp setup-accounts` assigns unlabelled pairs in.

### 4. Generate a key pair per account

Start from a labelled skeleton so misordering is impossible:

```bash
comp setup-accounts --template > keys.txt
```

That writes one `team=` line per team, in order, with the dashboard label in a
comment. Paste each account's pair after the `=`:

```
# 4. The Gambler   dashboard label: comp-4-gambler
gambler=PKAAAA...,secretaaaa...
```

Because each line names its team, **the order of the lines does not matter** —
you cannot paste them one row out, which is the one setup error that is both
easy to make and invisible afterwards.

If you would rather not use the template, a bare file also works:

```
# keys.txt -- delete this file once .env is written
PKAAAA...,secretaaaa...
PKBBBB...,secretbbbb...
PKCCCC...,secretcccc...
...nine lines...
```

Any of these shapes works: `KEY,SECRET`, `KEY SECRET`, `KEY:SECRET`, or
alternating bare `KEY` / `SECRET` lines (which is what copy-pasting straight
from the dashboard tends to give you). `#` comments and blank lines are
ignored.

### 5. Let the tool do the rest

```bash
comp setup-accounts --from-file keys.txt
```

It assigns pairs to teams in roster order, checks every one against Alpaca,
and only then writes `.env` (mode `0600`, backing up any existing file):

```
Parsed 9 credential pair(s).

Verifying 9 pair(s) against Alpaca…

  ok    trend_rider     PKAA…1234  #PA3K8DZQ9    equity $     5,000.00
  ok    mean_reverter   PKBB…5678  #PA7M2XRT1    equity $     5,000.00
  ...
  balances: $5,000.00 .. $5,000.00 (spread $0.00, tolerance $50.00)

wrote /path/to/.env (mode 0600)

Next: `comp doctor --check-accounts`
```

It refuses to write if any pair is a **live** account rather than paper, has
trading blocked, is not active, or fails to authenticate. Secrets are never
printed, logged or echoed; key ids appear masked.

Then delete `keys.txt`.

To assign specific pairs to specific teams rather than in order, prefix each
line: `gambler=PKAAAA...,secret...`. To be prompted instead of using a file,
run `comp setup-accounts` with no arguments — secrets are read without echo.

### 6. Confirm

```bash
comp setup-accounts --verify     # re-check what is in .env
comp doctor --check-accounts     # the full pre-flight
```

`doctor` must show no `ERROR` lines before you run a round.

## Pattern day trading: no longer a problem

This deserves a note because it *would* have been fatal a few months ago.

Under the old FINRA pattern-day-trader rule, an account under $25,000 that
made four or more day trades in five business days was flagged and blocked
from further day trading. Nine $5,000 accounts would every one of them have
been blocked within a day — and it would have hit the strategies unevenly,
destroying the scalper (hundreds of intraday round trips) and the Q-learner
while barely touching the benchmark. That is not a competition; that is a
regulatory artefact deciding the result.

FINRA **retired the PDT rule on 4 June 2026**, and Alpaca replaced it with a
dynamic intraday-margin framework. The $25,000 threshold is gone; the minimum
equity for 4x intraday buying power is now $2,000. Since this competition is
long-only at 1.0x gross leverage, intraday margin is never a binding
constraint.

If you are reading this much later, re-check: `comp doctor --check-accounts`
reports each account's `pattern_day_trader` flag, and the engine logs it.

## The one-account alternative

If nine accounts is still too much friction, the structural alternative is
**one real account holding the whole bankroll, partitioned into nine virtual
books in software** — one custody account, many internal books, which is how
a real multi-strategy desk operates.

It is not currently built. What it would need:

* a `VirtualBroker` wrapping one `AlpacaBroker` and enforcing per-team cash
  and positions;
* fill attribution by `client_order_id`, which is already tagged per team;
* per-team buying power enforced in software rather than by the venue.

The honest trade-offs: one team's bug can no longer bankrupt only itself, the
venue nets positions across teams (harmless here, since long-only means the
aggregate is never short), and the isolation guarantee becomes a property of
my code rather than of Alpaca's. Given that fairness is the whole point of
this project, separate accounts are the better default — but say the word and
I will build it.

## Sources

- [Alpaca paper trading docs](https://docs.alpaca.markets/us/docs/paper-trading)
- [Alpaca forum: more paper trading accounts](https://forum.alpaca.markets/t/feature-request-more-paper-trading-accounts/18125)
- [FINRA retires the PDT rule — Alpaca's new intraday margin framework](https://alpaca.markets/blog/finra-retires-the-pdt-rule-introducing-alpacas-new-intraday-margin-framework/)
