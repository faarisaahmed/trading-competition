# Setting up the Alpaca accounts

Nine teams, **three paper accounts, one email**. Each account holds three
teams' money and is partitioned in software into three virtual books. This
page explains how to create the accounts, why it is safe to share them, and
exactly where the sharing is visible.

## Why three accounts and not nine

Alpaca caps paper accounts at **3 per login** — and that cap includes the
default account you already have. Nine separate accounts therefore means
either three separate logins with three email addresses, or one login and a
partitioning layer.

| Route | Signups | Key pairs | Isolation | Notes |
|---|---|---|---|---|
| **1 login × 3 accounts, 3 teams each** *(what this repo ships)* | 1 | 3 | software-enforced | no extra emails |
| 3 logins × 3 accounts | 3 | 9 | broker-enforced | 3× the data rate limit |
| 9 logins × 1 account | 9 | 9 | broker-enforced | nine signups |

Both routes are supported — the mode is one line of config — but the shipped
default is the single login, because it is the one that needs nothing from you
but a single sign-up.

The cost is real and worth stating plainly: with nine accounts, isolation is
guaranteed by Alpaca; with three, it is guaranteed by `broker/shared.py`.
The next two sections are about making that guarantee hold.

## How the sharing works

`SharedAccount` owns one real Alpaca account and hands each team a
`VirtualBroker` — an object with the ordinary broker interface that sees only
that team's cash and positions. Every order is tagged with its team's
`client_order_id`, so fills attribute unambiguously.

Three things it does that a naive split does not:

**It crosses opposing orders internally.** If the scalper wants to buy 10 AAPL
in the same tick that the mean reverter wants to sell 10, the trade happens
between the two books at the **mid** — the only price that conserves cash
exactly — and never reaches Alpaca. This is not an optimisation. Alpaca
*rejects* opposing same-symbol orders inside one account as wash trades
(HTTP 403, "opposite side market/stop order exists"), so without crossing, one
of the two teams would silently lose its trade. In a full-week shared-mode
backtest, crossing was 2.7% of traded volume ($14.8k of $555k).

**It sizes against the price the order will actually pay.** A market buy fills
at the ask, not the mid. Sizing against the mid overspends by half the spread
every time, which walked the books about $2.50 negative on $5,000 before it
was fixed. `ref_price()` returns the ask for a buy and the bid for a sell.

**It reconciles.** After every flush, the sum of the virtual books must equal
the real account's cash and positions, or the run stops. A shared-mode
backtest reconciled all three groups exactly.

`close_all_positions` on a `VirtualBroker` closes only that team's positions —
it never calls Alpaca's account-wide close, which would wipe its two
room-mates.

## Which teams share an account

The grouping is not arbitrary. The **scalper** is the only team that rests
limit quotes, and therefore the only one that is *penalised* when a wash-trade
conflict forces a cancel. So it shares with the two least active teams:

| Account | Nickname | Teams | Funds |
|---|---|---|---|
| group-a | `ALPACA_GROUP_A` | Scalper, Buy & Hold *(unscored)*, Mean Reverter | 15,000 |
| group-b | `ALPACA_GROUP_B` | Q-Learner, Trend Rider, News Hound | 15,000 |
| group-c | `ALPACA_GROUP_C` | The Gambler, Stat Arb, Vol Breakout | 15,000 |

Each account is funded with **$15,000** = three teams × the $5,000 the rulebook
gives each team per round. The whole-week backtest produced **2 forced cancels
and 0 wash rejections** across all three groups.

## Step by step

### 1. Create the three paper accounts

One login at [app.alpaca.markets](https://app.alpaca.markets). Click the paper
account number in the upper left and choose **Open New Paper Account**.

You already have one paper account, and the cap of 3 includes it — so reset
that one to $15,000 and create two more.

For each, in that dialog:

* **Nickname:** `ALPACA_GROUP_A`, then `_B`, then `_C`. Naming the account
  after the environment variable it fills removes all doubt about which keys
  belong where.
* **Set Funds: 15000.** The default is $100,000, and **it cannot be changed
  afterwards without resetting the account.**
* **Leave "Sync to your live account balance" UNCHECKED.** Ticking it makes
  the paper balance mirror a real account, which would give the three groups
  different starting points and invalidate the round.

Equality is the part that actually matters. Unequal bankrolls are the one
condition that invalidates a round, and both `comp setup-accounts` and
`comp doctor --check-accounts` refuse to proceed when they differ by more
than 1%.

### 2. Generate a key pair per account

**Generate each account's keys immediately after creating it**, while you are
still in that account's dashboard — the secret is shown once and cannot be
retrieved later, only regenerated. Create account, generate keys, paste, move
on to the next.

Start from a labelled skeleton so misordering is impossible:

```bash
comp setup-accounts --template > keys.txt
```

That writes one line per account, with its dashboard nickname and funding in a
comment:

```
# 1. group-a (mean_reverter, scalper, benchmark)
#    Alpaca Nickname: ALPACA_GROUP_A    Set Funds: 15,000
#    3 teams share this account, $5,000 each
group-a=
```

Paste each account's pair after the `=`:

```
group-a=PKAAAA...,secretaaaa...
```

Because each line names its account, **the order of the lines does not
matter** — you cannot paste them one row out, which is the one setup error
that is both easy to make and invisible afterwards.

A bare file without labels also works, assigned in the order above:

```
# keys.txt -- delete this file once .env is written
PKAAAA...,secretaaaa...
PKBBBB...,secretbbbb...
PKCCCC...,secretcccc...
```

Any of these shapes parses: `KEY,SECRET`, `KEY SECRET`, `KEY:SECRET`, or
alternating bare `KEY` / `SECRET` lines (which is what copy-pasting straight
from the dashboard tends to give you). `#` comments and blank lines are
ignored.

### 3. Let the tool do the rest

```bash
comp setup-accounts --from-file keys.txt
```

It binds pairs to accounts, checks every one against Alpaca, and only then
writes `.env` (mode `0600`, backing up any existing file):

```
Parsed 3 credential pair(s).

Verifying 3 pair(s) against Alpaca…

  ok    group-a   PKAA…1234  #PA3K8DZQ9   equity $    15,000.00
  ok    group-b   PKBB…5678  #PA7M2XRT1   equity $    15,000.00
  ok    group-c   PKCC…9012  #PA2N4WQL8   equity $    15,000.00

wrote /path/to/.env (mode 0600)

Next: `comp doctor --check-accounts`
```

It refuses to write if any pair is a **live** account rather than paper, has
trading blocked, is not active, is funded for fewer teams than it holds, or
fails to authenticate. Secrets are never printed, logged or echoed; key ids
appear masked.

Then delete `keys.txt`.

### 4. Confirm

```bash
comp setup-accounts --verify     # re-check what is in .env
comp doctor --check-accounts     # the full pre-flight
```

`doctor` prints the account layout and must show no `ERROR` lines before you
run a round:

```
account mode: shared -- 3 real account(s) for 9 teams
  group-a        ALPACA_GROUP_A_KEY_ID      ok   $15,000.00
  group-b        ALPACA_GROUP_B_KEY_ID      ok   $15,000.00
  group-c        ALPACA_GROUP_C_KEY_ID      ok   $15,000.00
```

## Switching to nine accounts

If you would rather have broker-enforced isolation, make three logins with
three email addresses (Alpaca staff have confirmed on their forum that this is
within the terms of service), create three $5,000 accounts under each, and set:

```yaml
accounts:
  mode: per_team
```

in `config/competition.yaml`. Every command adapts: `--template` emits nine
labelled lines instead of three, `doctor` checks nine accounts, and the engine
gives each team a real `AlpacaBroker` instead of a `VirtualBroker`. Nothing
else changes, and no strategy code is aware of the difference.

There is one genuine advantage beyond isolation: Alpaca's market-data rate
limits are enforced **per owner, not per account**, so spreading the field
over three logins triples your request headroom. With a single login all nine
teams draw on one 200 req/min budget — enough, since quotes are fetched once
per tick and shared across the teams in a group, but with less slack.

## Pattern day trading: no longer a problem

This deserves a note because it *would* have been fatal a few months ago.

Under the old FINRA pattern-day-trader rule, an account under $25,000 that
made four or more day trades in five business days was flagged and blocked
from further day trading. Every account here would have been blocked within a
day — and it would have hit the strategies unevenly, destroying the scalper
(hundreds of intraday round trips) and the Q-learner while barely touching the
benchmark. That is not a competition; that is a regulatory artefact deciding
the result.

FINRA **retired the PDT rule on 4 June 2026**, and Alpaca replaced it with a
dynamic intraday-margin framework. The $25,000 threshold is gone; the minimum
equity for 4x intraday buying power is now $2,000. Since this competition is
long-only at 1.0x gross leverage, intraday margin is never a binding
constraint.

If you are reading this much later, re-check: `comp doctor --check-accounts`
reports each account's `pattern_day_trader` flag, and the engine logs it.

## Sources

- [Alpaca paper trading docs](https://docs.alpaca.markets/us/docs/paper-trading)
- [Alpaca forum: more paper trading accounts](https://forum.alpaca.markets/t/feature-request-more-paper-trading-accounts/18125)
- [FINRA retires the PDT rule — Alpaca's new intraday margin framework](https://alpaca.markets/blog/finra-retires-the-pdt-rule-introducing-alpacas-new-intraday-margin-framework/)
