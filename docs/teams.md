# The field

Nine entries: eight scored, one unscored reference. Each is a **strategy plus a
matching picker** — the brief requires the picker to "use a similar approach to
the actual team", so that coupling is stated for each one and asserted in
`tests/test_pickers.py::test_the_field_does_not_converge_on_one_book`.

Every parameter below is in `config/teams.yaml`, frozen once Round 1 starts.
`comp teams --team KEY` prints them with documentation.

---

## 1. Trend Rider — *momentum*

> "The trend is your friend until the ATR says otherwise."

**Thesis.** Trends persist on horizons between a few hours and a few weeks. The
way to harvest that is not to predict the turn but to own only names whose
advance is *clean*, size by volatility so a quiet name and a wild one
contribute equal risk, and leave via a stop rather than an opinion.

**Signal.** One composite score per name, in ATR units so it means the same
thing for a \$12 stock and a \$5,000 one:

```
score = 0.5 · (EMA₁₂ − EMA₂₆)/ATR  +  0.5 · ROC₂₀/(ATR% · √20)
```

Both terms answer "how many ATRs of drift do I have". Two gates throw out chop:
`ADX ≥ 18` (there is a direction at all) and `R² ≥ 0.25` (the advance is a
line, not a staircase of gaps). The gates are multiplicative rather than
binary, so a name at ADX 17.9 is not treated like one at ADX 4.

**Portfolio.** Top 3 by score, equal-weight at 32%, pyramiding once when a
trade is 1.5 ATR onside — adding to winners is the whole edge in trend
following.

**Exits, checked before entries.** A ratcheting 2.5 ATR trailing stop that
never loosens; a 4 ATR disaster stop from entry; score decay through −0.05. A
stopped-out name is on cooldown for 6 bars, because the most expensive trade in
trend following is re-entering the same failing breakout three times in an
afternoon.

**Picker.** Ranks by volatility-adjusted 21d and 63d return with an R²
trend-quality filter and a proximity-to-high tiebreak. Deliberately blind to
valuation, mean reversion and news — those are other teams' screens.

**Honest weakness.** Whipsaws. In a chopping market it pays the spread
repeatedly for entries that immediately stop out, and the ADX/R² gates only
reduce that, they do not remove it.

---

## 2. Mean Reverter — *statistical dislocation*

> "Everything snaps back. Size the rubber band, not the drama."

**Thesis.** Short-horizon moves overshoot. The edge per trade is small and the
hit rate is high, which means the entire game is (a) not buying things that are
falling for a *reason*, and (b) sizing so the occasional non-bounce does not
undo thirty winners.

**Entry requires all four.** z ≤ −2.0 against its own 20-bar mean; RSI ≤ 32;
**measured OU half-life ≤ 40 bars**; trailing regression slope ≥ −0.004.

The third condition is what separates this from a naive dip-buyer. An AR(1) fit
gives the half-life of the name's own mean reversion; if the series behaves
like a random walk (half-life → ∞) there is nothing to revert *to* and the
trade is declined however oversold it looks. The fourth declines knife-catching
in a genuine downtrend.

**Sizing.** Inverse volatility, targeting ~1.2% per-bar risk per position,
capped at 40%. Entries split into two tranches: the first at z = −2, the second
only if the dislocation *widens* to z = −3. Averaging down is dangerous in
general and correct here, because the thesis is explicitly that a wider gap is
a better trade — the hard z = −4.25 stop is what keeps that honest.

**Exits.** Mean touched (z ≥ −0.2), RSI recovered (≥ 58), thesis broken
(z ≤ −4.25), or 78 bars elapsed (too slow to be worth the capital).

**Picker.** Screens on the *statistics* of reversion — half-life ≤ 12 days,
variance ratio < 0.95, Hurst < 0.48, ADF t ≤ −2.0, with a volatility floor so
the band is worth trading. It is looking for the opposite of what the momentum
picker wants, which is the point.

**Honest weakness.** A regime break. Everything it holds is, by construction,
something that has recently fallen; a market-wide repricing hits it hardest,
and the second tranche makes that worse before the stop fires.

---

## 3. Q-Learner — *reinforcement learning*

> "It doesn't know what a stock is. It knows what worked."

**Thesis.** Don't encode a theory of markets; encode a *reward* and let the
agent find the policy.

**State (300 cells).** Bucketed regression slope (5) × bucketed z-score (5) ×
bucketed ATR% (4) × current exposure level (3). Every feature is scale-free — a
slope in percent, a z-score, an ATR ratio — so **one shared Q-table transfers**
from AAPL to a Round 3 mid-cap it has never seen. A per-symbol table would
never leave its initialisation in a three-week competition.

The bucket edges are calibrated to the 5-minute primary timeframe. Using
daily-scale edges (the first attempt) collapsed every observation into one
bucket and the agent learned nothing — 26 states visited instead of 118.

**Actions.** A target portfolio weight from `{0, 0.15, 0.32}`. Discrete
*exposure*, not discrete buy/sell, so it cannot accidentally learn to churn.

**Reward.**

```
r = prev_exposure · bar_return / ATR%          (risk-normalised P&L)
  − 0.35 · |exposure change|                    (pay for trading)
  − 0.5  · max(0, −position_drawdown)
```

clipped to ±3. Dividing by ATR% lets one table learn from both a sleepy utility
and a meme stock: a 1% move means different things in each, and the agent
should learn about *conditions*.

`bar_return` is close-to-close, not the bar's own open-to-close. The agent
decides at a close and holds to the next one; using the intra-bar return would
credit it for a move it was not positioned for and silently drop every
overnight gap.

**Three anti-churn measures, all of them necessary.** The first version turned
over \$583k on a \$5,000 account in four days — 117× — and gave up 6.5% to
round trips:

1. `turnover_penalty` calibrated to the *real* cost. Reward is P&L/ATR%, so a
   round trip costing ~8bps against a ~25bps ATR is worth ~0.3 reward units.
   The naive first guess of 0.02 was more than ten times too small to affect
   the policy.
2. `action_repeat: 3` — hold an exposure for three bars before re-deciding.
   Standard frame-skip; learning still happens every bar, credited to the
   (state, action) pair actually in force.
3. **Tie-breaking toward the incumbent.** With optimistic initialisation every
   action in an unvisited state has *identical* value, so a random tie-break
   re-rolls exposure at every decision point and pays the spread for it. When
   the agent is genuinely indifferent, the correct action is to do nothing.

And at the portfolio level, **hysteresis**: incumbents keep their slots, and a
challenger only displaces one when its Q exceeds the weakest incumbent's by
more than a round trip costs. Taking the naive top-3-of-12 every bar meant a
different three won each bar and the other nine were flattened — which defeated
`action_repeat` entirely. Fixing it was worth 12.7 percentage points in the dry
run.

**Learning schedule.** Epsilon-greedy, 0.10 → 0.02 over 4,000 updates. The
table **persists across rounds** via the ledger, so the agent that shows up for
Round 3 is the one that learned in Rounds 1 and 2 — which is the whole point of
entering an RL agent. Pre-train once before Round 1 with `comp pretrain`.

**Picker.** A LinUCB contextual bandit — the same "learn from reward" idea one
level up. Arms are candidates described by standardised price/volume features;
one ridge model is shared across all symbols so the learning transfers.
Standardising per call is what makes the weights portable: "one sigma better
than the pool" survives a regime change in a way "+8% over 21 days" does not.

**Honest weakness.** It starts close to ignorant, and 300 states is a coarse
view of a market. Its interest is whether learning across three rounds beats
hand-written rules, not whether it is a good trader on day one.

---

## 4. The Gambler — *deliberate variance*

> "Kelly said bet the edge. Kelly never said how big the edge was."

**Thesis.** Over a three-week, eight-way contest with points weighted toward
the podium, variance is not obviously the enemy. Fifth place at +2% scores 5
points; first at +14% scores 15. If the objective is *points* rather than
risk-adjusted return, swinging for the fence has a real case. This team is that
case, argued honestly and bounded so it cannot break the competition.

**Mechanics.**

1. From the last 60 bars, measure the up-bar frequency `p` and the payoff ratio
   `b` = mean up move / mean down move.
2. Size by Kelly: `f* = p − (1−p)/b`.
3. **Then overbet it** — × 0.85 × 1.6 = 1.36× full Kelly. Overbetting Kelly is
   *known* to reduce long-run growth and raise ruin probability. That is the
   point, and it is labelled as such.
4. **Climb the ladder on losses.** After a loser, step to the next rung of
   `{1.0, 1.45, 1.9}`. After a winner, reset. Lose on the top rung and it sits
   out 8 bars. A *bounded* martingale: three rungs, never more, so the classic
   "double until broke" failure is arithmetically impossible.
5. **Cut fast, run long.** Stop at 2 ATR, target 3 ATR, trail once 2 ATR
   onside. The asymmetry is where the positive expectancy has to come from,
   since the sizing is actively hostile to it.

**Self-imposed guardrail.** Down 28% in a session and it flattens and stops for
the day. The engine's own kill switch is at 35%; this team's job is to be
reckless with position size, not to be the reason a round has to be voided.

**Picker.** The only picker in the field that treats volatility as a feature —
high realised vol, frequent gaps, expanding ranges, high beta. No sector cap:
concentration is the strategy, not a bug.

**Honest weakness.** It will probably lose. Its function is to make the role of
luck in a three-week contest explicit rather than pretending it away.

---

## 5. Stat Arb — *relative value*

> "I don't care where the market goes. I care where the spread goes."

**Thesis.** Two businesses exposed to the same drivers should trade at a stable
ratio. The ratio wanders; the wandering mean-reverts even when neither leg
does. Trade the *spread* and the market's direction stops mattering.

**Pair formation.** For every candidate pair, in order:

1. log-price correlation ≥ 0.65
2. `β` = OLS slope of log y on log x (the hedge ratio), sanity-bounded
3. spread = log y − β·log x
4. Dickey-Fuller t on the spread ≤ −1.8 → the *gap* comes back
5. OU half-life ≤ 60 bars → it comes back *inside the round*

Steps 4 and 5 are what separate this from "these two look correlated".
Correlation says they move together; stationarity says the gap closes;
half-life says it closes soon enough to matter. All three or no trade. Pairs
are re-fitted once per session and ranked by `|t| · corr / half-life`.

**Long-only expression.** A textbook pair trade is long the cheap leg and short
the rich one. On a cash account the short half is unavailable, so the position
is a **rotation**: hold whichever leg the spread says is cheap, and switch when
the spread flips.

```
z ≥ +1.6  →  y is rich  →  hold x
z ≤ −1.6  →  y is cheap →  hold y
|z| ≤ 0.35 →  fair value →  flat
|z| ≥ 3.75 →  relationship broke → flat, blacklist the pair
```

This keeps the relative-value signal — the entry decision is made purely on the
spread — while accepting the market beta the short leg would have hedged. It is
an honest adaptation, not a disguised directional bet.

**Round 3 fallback.** The Chaos Draft can deal ten names with no cointegrated
pair among them. Rather than sit out the week, a bounded relaxed screen
(correlation −0.15, ADF +0.8, half-life ×1.6) trades the best available
relationship at **half size** — weaker evidence, less capital. Relaxed pairs
are flagged in the order reason.

**Picker.** Returns *pairs*, not names: the same four tests run on daily bars
over 180 days, with an in-sector bonus encoding a real prior (cointegration
between two railroads is more likely structural than between a railroad and a
biotech). It then backfills toward the cap with the names most likely to *form*
a pair, so the strategy has room for its own intraday search — returning only
two legs boxed the strategy in when its intraday screen disagreed.

**Honest weakness.** No short leg means it carries market direction it did not
want. And cointegration found in-sample breaks out-of-sample more often than
the t-statistic suggests.

---

## 6. News Hound — *event-driven*

> "The tape tells you what happened. The wire tells you why."

**Thesis.** Price tells you *that* something moved; the wire tells you *why*,
and the why determines whether it continues. A guidance raise reprices over
hours, leaving room between "the story hit" and "the story is in the price".

**Sentiment, without an LLM.** Every article is scored by
`strategies/lexicon.py` — 145 finance phrases, 190 words, negators,
intensifiers and hedges. Rules on top: phrases beat words and consume them
(`"beats estimates"` is one unit, not two); negation flips and dampens within
three tokens *and reaches phrases*, so "did not beat expectations" is negative;
modifiers are detected on **both sides**, because market copy writes both
"sharply higher" and "fell modestly"; ALL-CAPS amplifies; and the score divides
by √(number of hits), so one decisive headline is not drowned out by a long
lukewarm body.

Per-symbol score is a weighted average with three weights: **age** (exponential
decay, 8h half-life), **exclusivity** (`1/√n_tickers`, so "AAPL beats" counts
far more than a ten-ticker listicle), and **source trust** (a small explicit
multiplier for primary wires over aggregators).

**Confirmation, which is the part that matters.** Sentiment alone is a trap: by
the time a retail-visible headline prints, the move has often happened. So an
entry also requires the tape to agree (ROC over 6 bars ≥ +0.05%) **and** the
move not to be spent — price within 4.5% of where it was when the story broke.
Refusing to chase a name already up 7% on the news is what separates reading
the wire from being exit liquidity.

**Exits.** Fixed +5.5% / −2.8% brackets (news trades resolve fast or not at
all), sentiment decay below 0.05, or 30 hours elapsed — after which the story
is not news and the position is an unhedged directional bet the strategy never
intended.

**Picker.** Ranks by the same lexicon score weighted by coverage volume. Will
happily return fewer names than its cap: a name with no coverage is unusable to
this team however good its chart looks.

**Honest weakness.** A fixed lexicon cannot read context, sarcasm, or a story
it has no words for. And the dry-run synthetic wire is generated to *lead*
price, so this team looks better in a backtest than it has any right to.

---

## 7. Vol Breakout — *volatility regime*

> "Quiet, quiet, quiet — then you have to already be in."

**Thesis.** Volatility clusters and mean-reverts. Long quiet stretches resolve
into fast directional moves, large relative to the range that preceded them.
The trade is not "buy new highs" — a coin flip in chop — it is "buy new highs
*that come out of measured quiet*, backed by real participation".

**Three conditions, all required.**

1. **Squeeze.** The Bollinger band sits entirely inside the Keltner channel —
   realised dispersion has fallen below the ATR the name normally covers. This
   is the TTM-squeeze condition, used here in preference to a bandwidth
   percentile for a specific reason: a percentile **saturates with coil
   duration**. A name quiet for sixty bars has a low *median* bandwidth, so it
   can never rank in its own bottom third, and a percentile gate would reject
   the tightest setups on the board. BB-inside-KC is a ratio, so it is
   duration-independent. The squeeze need only have been on within the last 12
   bars, because the breakout bar itself widens the bands.
2. **Break.** Price clears the 20-bar Donchian high (or the session's
   opening-range high) by at least 0.15 ATR. The buffer matters: a break by one
   tick is noise, and one tick above a level everyone is watching is where
   stop-hunting happens.
3. **Expansion.** Volume ≥ 1.35× its 20-bar average **and** bar range ≥ 1.20×
   its recent average. A breakout on no volume is a breakout nobody believes.

**Exits.** A tight 1.75 ATR initial stop (a true breakout should not revisit
its base); a wide 2.75 ATR Chandelier trail from the run-up high (the thesis is
that the move is large); and a **failed-break override** — if price closes back
below the level it broke within 4 bars, exit immediately rather than waiting
for the ATR stop. Failed breakouts reverse hard, and this rule is worth more
than the stop placement.

**Picker.** Hunts coiled springs: low bandwidth percentile, TTM squeeze on
daily bars, consecutive narrow-range days. The one picker in the field that
prefers names where *nothing is happening* — which is exactly why it pairs with
a breakout strategy, since by the time something is happening the entry is
gone.

Its ATR floor is measured over 60 days, not 14: a tight coil has low recent ATR
*by definition*, so gating on that would reject the best setups. What matters
is whether the name moves when it is *not* coiled. An earlier version also
*rewarded* high ATR in the score, which made a 7%-a-day lottery ticket outrank
a genuine coil — exactly backwards.

**Honest weakness.** False breakouts are the dominant failure mode and no
filter removes them. It is a low-hit-rate, high-payoff strategy, which over
only 12–15 sessions may simply not get its setups.

---

## 8. The Scalper — *market microstructure*

> "A penny a hundred times beats a dollar you waited all week for."

**Thesis.** Most of the money in equities is made by people with no directional
view. A market maker earns the spread by being patient on both sides and
managing inventory; the risk is adverse selection. So the craft is: compute a
fair value better than the mid, demand real edge before posting, and get out of
inventory fast.

**Fair value** — because the mid is a lie when the book is lopsided:

```
fair = microprice
     + 0.35 · spread · queue_imbalance
     + 0.45 · short_horizon_drift · price
```

The **microprice** size-weights the two sides: with 100 bid and 900 offered the
true clearing price is nearer the bid, and using the arithmetic mid there means
systematically buying too high. The **drift** term (ROC over 5 fast bars) stops
the strategy posting a static bid into a falling name — the single most
expensive mistake a naive maker makes.

**Quoting.**

```
bid = fair · (1 − edge_bps/10⁴ − inventory_skew)
ask = fair · (1 + edge_bps/10⁴ − inventory_skew)
```

`edge_bps = 6` is the minimum compensation demanded for providing liquidity.
`inventory_skew` shifts *both* quotes down as the book gets long, making the
ask more likely to fill and the bid less — the Avellaneda–Stoikov intuition,
implemented as a linear skew because a three-week competition is not the place
for a stochastic-control solve.

**Long-only adaptation.** A real maker quotes both sides continuously. With no
short leg this runs as a **one-sided maker**: passive bids accumulate
inventory, passive offers distribute it, never below flat. It therefore captures
the spread only on the round trip, not on every fill — a genuine handicap,
stated plainly.

**Discipline.** Spread must be inside [3.5, 60] bps: too tight and there is no
edge, too wide and something is wrong (news, halt, illiquidity) and the maker is
the one being picked off. Quotes re-quote only when fair value moves more than
2.5 bps, and expire after 90 seconds. Inventory is capped at 30% per name and
75% in aggregate, and the book is **flattened 15 minutes before the bell** — a
market maker who holds overnight is a punter with extra steps.

**Picker.** Ranks on trade count, dollar volume and tightness proxies while
*preferring low daily volatility*. Chop, not trend, is where a one-sided maker
earns. True spreads are not in daily bars, so the picker narrows the field and
the strategy re-checks the actual relative spread on every quote.

**Honest weakness.** It is the team most exposed to the gap between simulated
and real fills. The simulator fills a resting limit when the quote crosses it;
reality has queue position, and a real maker gets filled precisely when it
least wants to be.

---

## 9. Buy & Hold — *unscored reference*

Buys an equal-weight basket of its universe on the first tradable tick and does
nothing else. It takes no place and no points and cannot displace a real team.

It is in every report because "+3.1% over the week" means nothing on its own.
If the market ran 4% and the winner made 3%, the interesting fact is that
**nobody beat the basket** — and without this line in the table, nobody would
notice.
