"""Team 6 -- News Hound: event-driven trading off the wire.

Thesis
------
Price tells you *that* something moved; the wire tells you *why*, and the why
determines whether the move continues. A guidance raise is repriced over
hours, not seconds, so there is room between "the story hit" and "the story
is in the price" for a systematic reader to act.

Signal
------
Every article touching a symbol in the last `stale_news_hours` is scored by
`lexicon.score_text` -- a checked-in finance dictionary with negation,
intensifier and phrase handling. No LLM: the scorer is a pure function of the
text, which also means the whole signal is reproducible from the ledger.

The per-symbol score is a weighted average of article scores with three
weights:

    age       exp(-ln2 * hours_old / half_life_hours)   -- a 2-hour-old story
                                                           counts far more than
                                                           a 20-hour-old one
    exclusivity  1 / sqrt(number of tickers tagged)     -- "AAPL beats" is
                                                           about Apple;
                                                           "10 stocks to watch"
                                                           tagging 10 names is not
    source    a small trust multiplier for wires vs. aggregators

Confirmation
------------
Sentiment alone is a trap: by the time a retail-visible headline prints, the
move has often happened. So an entry additionally requires

    * the tape agreeing: ROC over `confirmation_roc_bars` >= `confirmation_min_roc`
    * the move not already being spent: price within `max_chase_pct` of where
      it was when the story broke

The second rule is the one that matters. Refusing to chase a name already up
7% on the news is what separates reading the wire from being exit liquidity.

Exits
-----
Fixed `take_profit_pct` / `stop_loss_pct` brackets (news trades resolve fast
or not at all), sentiment decaying below `sentiment_exit`, or `time_stop_hours`
elapsing -- after which the story is no longer news and the position is just
an unhedged directional bet the strategy never intended to take.
"""

from __future__ import annotations

import math

from ..types import NewsItem, OrderIntent
from ..util import indicators as ind
from . import lexicon
from .base import Param, Strategy, StrategyContext

#: Small, explicit trust multipliers. Primary wires move markets; content
#: aggregators mostly recirculate. Unknown sources sit at 1.0 -- neutral.
SOURCE_WEIGHTS: dict[str, float] = {
    "benzinga": 1.00,
    "reuters": 1.20,
    "bloomberg": 1.20,
    "dow jones": 1.15,
    "the wall street journal": 1.15,
    "wsj": 1.15,
    "cnbc": 1.10,
    "associated press": 1.10,
    "ap": 1.10,
    "barrons": 1.05,
    "marketwatch": 1.00,
    "businesswire": 1.10,
    "business wire": 1.10,
    "globenewswire": 1.10,
    "pr newswire": 1.05,
    "prnewswire": 1.05,
    "seeking alpha": 0.80,
    "zacks": 0.80,
    "insider monkey": 0.70,
    "motley fool": 0.70,
    "investorplace": 0.70,
}


class NewsHound(Strategy):
    DESCRIPTION = (
        "Lexicon-scored news sentiment with age decay, exclusivity weighting and source "
        "trust, gated on tape confirmation and a no-chase rule, held on fixed brackets."
    )

    DEFAULT_TICK_SECONDS = 120

    PARAMS = (
        Param("tick_seconds", 120, "seconds between evaluations", minimum=5),
        Param("sentiment_entry", 0.28, "weighted sentiment needed to buy", minimum=0.0),
        Param("sentiment_exit", 0.05, "sentiment decay that closes a position"),
        Param("min_articles", 2, "distinct articles required to act", minimum=1),
        Param("half_life_hours", 8.0, "age at which an article counts half", minimum=0.1),
        Param("confirmation_roc_bars", 6, "bars for the tape-confirmation check", minimum=1),
        Param("confirmation_min_roc", 0.0005, "minimum ROC for the tape to agree"),
        Param("max_chase_pct", 0.045, "refuse to buy more than this above the news price",
              minimum=0.0),
        Param("max_positions", 3, "concurrent positions", minimum=1),
        Param("weight_per_name", 0.30, "weight per position", minimum=0.0, maximum=1.0),
        Param("take_profit_pct", 0.055, "profit target", minimum=0.001),
        Param("stop_loss_pct", 0.028, "stop loss", minimum=0.001),
        Param("time_stop_hours", 30.0, "hours before an unresolved trade is closed",
              minimum=0.1),
        Param("stale_news_hours", 48.0, "ignore articles older than this", minimum=1.0),
        Param("min_bars_required", 30, "warm-up bars before trading", minimum=5),
    )

    # ------------------------------------------------------------------ #
    # sentiment aggregation
    # ------------------------------------------------------------------ #

    @staticmethod
    def source_weight(source: str) -> float:
        key = (source or "").strip().lower()
        if key in SOURCE_WEIGHTS:
            return SOURCE_WEIGHTS[key]
        for name, w in SOURCE_WEIGHTS.items():
            if name in key:
                return w
        return 1.0

    def article_weight(self, item: NewsItem, now) -> float:
        """Age decay x exclusivity x source trust."""
        hours = max((now - item.ts).total_seconds() / 3600.0, 0.0)
        age = math.exp(-math.log(2.0) * hours / max(self.p.half_life_hours, 0.1))
        exclusivity = 1.0 / math.sqrt(max(len(item.symbols), 1))
        return age * exclusivity * self.source_weight(item.source)

    def sentiment(self, ctx: StrategyContext, symbol: str) -> dict[str, float]:
        """Weighted sentiment for one symbol, plus the diagnostics behind it."""
        items = ctx.snapshot.news_for(symbol, within_hours=self.p.stale_news_hours)
        if not items:
            return {"score": 0.0, "n": 0, "weight": 0.0, "freshest_hours": math.inf}
        num = den = 0.0
        best_abs = 0.0
        headline = ""
        for item in items:
            s = lexicon.score_text(item.text)
            if s == 0.0:
                continue
            w = self.article_weight(item, ctx.ts)
            num += s * w
            den += w
            if abs(s) > best_abs:
                best_abs, headline = abs(s), item.headline
        if den <= 0:
            return {"score": 0.0, "n": len(items), "weight": 0.0, "freshest_hours": math.inf}
        freshest = min((ctx.ts - i.ts).total_seconds() / 3600.0 for i in items)
        return {
            "score": num / den,
            "n": float(len(items)),
            "weight": den,
            "freshest_hours": freshest,
            "headline": headline,          # type: ignore[dict-item]
        }

    # ------------------------------------------------------------------ #
    # confirmation
    # ------------------------------------------------------------------ #

    def tape_confirms(self, ctx: StrategyContext, symbol: str) -> tuple[bool, float]:
        p = self.p
        closes = ctx.closes(symbol, n=p.confirmation_roc_bars + 2)
        if len(closes) < p.confirmation_roc_bars + 1:
            return False, 0.0
        roc = ind.roc(closes, p.confirmation_roc_bars)
        return roc >= p.confirmation_min_roc, roc

    def already_chased(self, ctx: StrategyContext, symbol: str, freshest_hours: float) -> tuple[bool, float]:
        """Has the move already happened since the story broke?

        Compares the current price with the close of the bar nearest to when
        the freshest article printed. If the name has already run more than
        `max_chase_pct`, the edge is gone and this is a bad entry.
        """
        p = self.p
        bars = ctx.bars(symbol)
        if not bars or not math.isfinite(freshest_hours):
            return False, 0.0
        px = ctx.price(symbol)
        if px <= 0:
            return False, 0.0
        # Locate the bar closest to the story's timestamp.
        target = ctx.ts.timestamp() - freshest_hours * 3600.0
        ref = min(bars, key=lambda b: abs(b.ts.timestamp() - target))
        if ref.close <= 0:
            return False, 0.0
        move = px / ref.close - 1.0
        return move > p.max_chase_pct, move

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def on_round_start(self, ctx: StrategyContext) -> None:
        ctx.state.setdefault("_symbols", {})
        phrases, words = lexicon.vocabulary_size()
        ctx.log.info(
            "news_hound: lexicon %d phrases / %d words, entry>=%.2f, no-chase %.1f%%",
            phrases, words, self.p.sentiment_entry, self.p.max_chase_pct * 100,
        )

    def on_tick(self, ctx: StrategyContext) -> list[OrderIntent]:
        p = self.p
        self.sync_bar_clock(ctx)
        intents: list[OrderIntent] = []

        scores = {sym: self.sentiment(ctx, sym) for sym in ctx.universe}

        # ---------------- exits ------------------------------------------- #
        for sym in ctx.held:
            st = ctx.sym_state(sym)
            px = ctx.price(sym)
            entry = float(st.get("entry_price", 0.0))
            if px <= 0 or entry <= 0:
                continue
            move = px / entry - 1.0
            held_hours = (ctx.ts.timestamp() - float(st.get("entry_ts", ctx.ts.timestamp()))) / 3600.0
            cur = scores.get(sym, {}).get("score", 0.0)
            reason = None
            if move >= p.take_profit_pct:
                reason = f"target +{move:.1%}"
            elif move <= -p.stop_loss_pct:
                reason = f"stop {move:.1%}"
            elif cur < p.sentiment_exit:
                reason = f"sentiment decayed to {cur:+.2f}"
            elif held_hours >= p.time_stop_hours:
                reason = f"news stale after {held_hours:.0f}h"
            if reason:
                o = ctx.close(sym, reason=reason, tag="exit")
                if o:
                    intents.append(o)
                    st.clear()
        exiting = {o.symbol for o in intents}

        # ---------------- entries ----------------------------------------- #
        slots = p.max_positions - len([s for s in ctx.held if s not in exiting])
        if slots <= 0:
            return intents

        ranked = []
        for sym, m in scores.items():
            if ctx.holds(sym) or sym in exiting or not ctx.tradable(sym):
                continue
            if ctx.has_open_order(sym):
                continue
            if m["n"] < p.min_articles or m["score"] < p.sentiment_entry:
                continue
            if not ctx.snapshot.has_history(sym, self.warmup_bars):
                continue
            confirmed, roc = self.tape_confirms(ctx, sym)
            if not confirmed:
                continue
            chased, move = self.already_chased(ctx, sym, m["freshest_hours"])
            if chased:
                ctx.log.debug(
                    "news_hound: skipping %s, already +%.1f%% since the story", sym, move * 100
                )
                continue
            # Rank by conviction: sentiment scaled by how much coverage backs it.
            ranked.append((sym, m, roc, m["score"] * math.sqrt(m["weight"])))
        ranked.sort(key=lambda t: -t[3])

        for sym, m, roc, _conv in ranked[:slots]:
            o = ctx.rebalance_to_weight(
                sym, p.weight_per_name, tolerance=0.01,
                reason=(f"news {m['score']:+.2f} from {int(m['n'])} articles "
                        f"(freshest {m['freshest_hours']:.1f}h, roc {roc:+.3%})"),
                tag="entry",
            )
            if not o:
                continue
            intents.append(o)
            ctx.sym_state(sym).update({
                "entry_price": ctx.price(sym),
                "entry_ts": ctx.ts.timestamp(),
                "entry_sentiment": round(m["score"], 4),
                "entry_headline": str(m.get("headline", ""))[:160],
            })
        return intents
