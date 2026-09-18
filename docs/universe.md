# The Round 3 pool

`data/top500.csv` is the ranked list the Chaos Draft deals from. It is checked
into git on purpose: the ranking is the spine of the fairness constraint, so if
the pool changed between the draft and the audit, "every hand sums to 2505"
would be unverifiable.

## Format

```csv
# as_of,2026-09-15,metric,market_cap,source,scripts/build_top500.py
rank,symbol,name,sector,market_cap_usd,share_price
1,NVDA,NVIDIA,Technology,4400000000000,180.00
2,MSFT,Microsoft,Technology,3700000000000,500.00
...
```

* `rank` — 1 is the most valuable. Must be contiguous from 1.
* `sector` — used by the draft's sector-concentration cap and by the pickers.
* `market_cap_usd` / `share_price` — either can be the ranking metric.

The header comment carries provenance (`as_of`, `metric`, `source`) and is
surfaced by `comp universe` and stamped into the recorded draft.

## "Most expensive": which metric?

The brief says "the top 500 most expensive public companies", which is
ambiguous.

**`market_cap` (default).** Total company value. This is the usual reading of
"most valuable" and gives a pool of recognisable large caps. Downside: Alpaca
does not publish share counts, so caps come from the dated snapshot.
`comp refresh-universe` re-prices every name from Alpaca and scales the stored
cap by the price change since the snapshot — share counts move slowly, prices
do not — then re-ranks. For a hard re-rank, drop in a fresh CSV.

**`share_price`.** Literal price per share. Fully self-contained: computable
from Alpaca alone, no external data, no staleness. But it produces a strange
pool — NVR, Booking, AutoZone, MercadoLibre at the top, and most household
names excluded.

Switch with `draft.metric` in `config/competition.yaml`, or inspect either with:

```bash
comp universe --metric market_cap --size 500
comp universe --metric share_price --size 500
```

## Before Round 3

```bash
comp refresh-universe --metric market_cap     # re-price and re-rank
comp universe --live                          # check tradability against Alpaca
comp draft --round 3 --live --write           # deal, verify, record
```

`--live` verifies every pool member against Alpaca's asset list and **drops
untradable names, then renumbers the ranks**. That matters: a hand containing a
delisted ticker is not a fair hand, and renumbering keeps the rank-sum target
exactly achievable.

`comp doctor` warns if the snapshot is more than 45 days old.

## Regenerating from source

`scripts/build_top500.py` holds the underlying list as reviewable Python — 741
companies with approximate caps, prices and sectors — and emits the top 500
sorted by cap, tie-broken alphabetically for determinism.

```bash
python scripts/build_top500.py --out data/top500.csv --as-of 2026-09-15
```

The 241-name buffer above 500 exists so that delistings and acquisitions can be
absorbed without the pool dropping below the size the draft needs
(`picks_per_team × teams` = 90 unique names for the shipped configuration).

The bundled figures are approximate and hand-maintained. They are a *starting
point*: ranks only need to be a consistent ordering for the draft to be fair,
but the fresher they are, the better "most valuable" matches reality on the
day.

## Round 2's candidate pool

Round 2 does not use this file as the tradable universe. It builds a shared
candidate pool from Alpaca's most-actives screener (by volume and by trade
count) unioned with the top-cap snapshot, then applies the round's liquidity
filter: minimum price, maximum price, average dollar volume, average trade
count, tradability, and an exclusion for leveraged/inverse ETFs.

That pool is built **once** and handed to all nine pickers. The teams differ in
what they *select*, never in what they are shown.
