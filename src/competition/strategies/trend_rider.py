"""Team 1 -- Trend Rider: cross-sectional, risk-adjusted momentum.

Thesis
------
Trends persist on horizons between a few hours and a few weeks, and the way
to harvest that is not to predict the turn but to (a) only own names whose
advance is *clean*, (b) size by volatility so a quiet name and a wild one
contribute the same risk, and (c) leave via a stop, never via an opinion.

Signal
------
A single composite score per name, in ATR units so it is comparable across
symbols and across rounds:

    score = w_ema * (ema_fast - ema_slow) / ATR
          + w_roc * ROC(n) / (ATR% * sqrt(n))

Both terms are "how many ATRs of drift do I have", which is the only
momentum measure that means the same thing for a $12 stock and a $5,000 one.
Two gates then throw out chop:

    ADX  >= adx_min   -- there is a direction at all
    R^2  >= r2_min    -- the advance is a line, not a staircase of gaps

Portfolio
---------
Rank the survivors, hold the best `max_positions`, equal-weight at
`target_weight_per_name`. Pyramid once when a trade is `add_on_profit_atr`
ATRs onside -- adding to winners is the whole edge in trend following.

Exits (checked before entries, always)
--------------------------------------
1. Ratcheting `trail_atr_mult` ATR trailing stop -- never loosens.
2. A wider `hard_stop_atr_mult` ATR disaster stop measured from entry.
3. Score decay through `exit_score_max` -- the trend stopped being a trend.
After a stop-out the name is on cooldown for `cooldown_bars_after_stop` bars,
because the most expensive trade in trend following is re-entering the same
failing breakout three times in an afternoon.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..types import Bar, OrderIntent
from ..util import indicators as ind
from .base import Param, Strategy, StrategyContext


class TrendRider(Strategy):
    DESCRIPTION = (
        "Cross-sectional momentum with ATR-normalised scoring, ADX/R^2 regime gates, "
        "pyramiding into winners and a ratcheting ATR trailing stop."
    )

    DEFAULT_TICK_SECONDS = 300

    PARAMS = (
        Param("tick_seconds", 300, "seconds between evaluations", minimum=5),
        Param("fast_ema", 12, "fast EMA period on primary bars", minimum=2),
        Param("slow_ema", 26, "slow EMA period on primary bars", minimum=3),
        Param("roc_period", 20, "rate-of-change lookback in bars", minimum=2),
        Param("adx_period", 14, "ADX period", minimum=2),
        Param("adx_min", 18.0, "minimum ADX to consider a name trending", minimum=0.0),
        Param("r2_min", 0.25, "minimum R^2 of the recent path", minimum=0.0, maximum=1.0),
        Param("max_positions", 3, "concurrent names", minimum=1),
        Param("target_weight_per_name", 0.32, "base weight per name", minimum=0.0, maximum=1.0),
        Param("entry_score_min", 0.15, "composite score needed to enter"),
        Param("exit_score_max", -0.05, "composite score that triggers an exit"),
        Param("atr_period", 14, "ATR period", minimum=2),
        Param("trail_atr_mult", 2.5, "trailing stop distance in ATRs", minimum=0.1),
        Param("hard_stop_atr_mult", 4.0, "disaster stop from entry, in ATRs", minimum=0.1),
        Param("add_on_profit_atr", 1.5, "ATRs of profit before pyramiding", minimum=0.0),
        Param("max_adds", 1, "pyramid steps allowed per position", minimum=0),
        Param("cooldown_bars_after_stop", 6, "bars to sit out after a stop", minimum=0),
        Param("min_bars_required", 60, "warm-up bars before trading", minimum=5),
    )

    # -- scoring ----------------------------------------------------------- #

    def score(self, bars: Sequence[Bar]) -> tuple[float, dict[str, float]]:
        """Composite trend score plus the diagnostics that produced it."""
        p = self.p
        if len(bars) < max(p.slow_ema, p.roc_period, p.adx_period) + 5:
            return 0.0, {}
        closes = [b.close for b in bars]
        price = closes[-1]
        atr = ind.atr(bars, p.atr_period)
        if price <= 0 or atr <= 0:
            return 0.0, {}

        ema_gap_atr = (ind.ema(closes, p.fast_ema) - ind.ema(closes, p.slow_ema)) / atr
        atr_pct = atr / price
        roc = ind.roc(closes, p.roc_period)
        # Normalise the ROC by the volatility it could plausibly have covered
        # over the same window: sigma * sqrt(n). Now both terms are in
        # "standard deviations of drift".
        roc_atr = roc / max(atr_pct * (p.roc_period ** 0.5), 1e-6)

        adx = ind.adx(bars, p.adx_period)
        r2 = ind.r_squared(closes, p.roc_period * 2)
        raw = 0.5 * ema_gap_atr + 0.5 * roc_atr

        diag = {
            "ema_gap_atr": ema_gap_atr, "roc": roc, "roc_atr": roc_atr,
            "adx": adx, "r2": r2, "atr": atr, "atr_pct": atr_pct, "raw": raw,
        }
        # The gates are multiplicative rather than binary so a name at ADX 17.9
        # is not treated identically to one at ADX 4.
        if adx < p.adx_min or r2 < p.r2_min:
            return raw * 0.25, diag
        return raw, diag

    # -- lifecycle --------------------------------------------------------- #

    def on_round_start(self, ctx: StrategyContext) -> None:
        ctx.state.setdefault("_symbols", {})
        ctx.log.info(
            "trend_rider: %d names, holding top %d by ATR-normalised momentum",
            len(ctx.universe), self.p.max_positions,
        )

    def on_tick(self, ctx: StrategyContext) -> list[OrderIntent]:
        p = self.p
        self.sync_bar_clock(ctx)
        intents: list[OrderIntent] = []

        scores: dict[str, float] = {}
        diags: dict[str, dict[str, float]] = {}
        for sym in ctx.candidates(self.warmup_bars):
            s, d = self.score(ctx.bars(sym))
            scores[sym] = s
            diags[sym] = d

        # ---------------- exits first: free the cash, then redeploy -------- #
        for sym in ctx.held:
            d = diags.get(sym, {})
            px = ctx.price(sym)
            if px <= 0:
                continue
            st = ctx.sym_state(sym)
            reason = None

            trail = self.trail_stop(ctx, sym, p.trail_atr_mult, p.atr_period)
            if trail > 0 and px < trail:
                reason = f"trail stop {trail:.2f}"
            elif d and st.get("entry_price") and st.get("entry_atr"):
                hard = float(st["entry_price"]) - p.hard_stop_atr_mult * float(st["entry_atr"])
                if px < hard:
                    reason = f"hard stop {hard:.2f}"
            if reason is None and sym in scores and scores[sym] <= p.exit_score_max:
                reason = f"momentum faded (score {scores[sym]:.2f})"

            if reason:
                o = ctx.close(sym, reason=reason, tag="exit")
                if o:
                    intents.append(o)
                    st.clear()
                    ctx.set_cooldown(sym, p.cooldown_bars_after_stop)
        exiting = {o.symbol for o in intents}

        # ---------------- pyramid into what is already working ------------ #
        for sym in ctx.held:
            if sym in exiting:
                continue
            st = ctx.sym_state(sym)
            entry, atr = float(st.get("entry_price", 0.0)), float(st.get("entry_atr", 0.0))
            adds = int(st.get("adds", 0))
            if entry <= 0 or atr <= 0 or adds >= p.max_adds:
                continue
            px = ctx.price(sym)
            if px - entry < p.add_on_profit_atr * atr:
                continue
            if scores.get(sym, -1.0) < p.entry_score_min:
                continue
            target = min(
                ctx.weight(sym) + p.target_weight_per_name * 0.5,
                p.target_weight_per_name * 1.5,
            )
            o = ctx.rebalance_to_weight(
                sym, target, tolerance=0.03,
                reason=f"pyramid #{adds + 1} ({(px - entry) / atr:.1f} ATR onside)", tag="add",
            )
            if o:
                intents.append(o)
                st["adds"] = adds + 1

        # ---------------- entries ----------------------------------------- #
        slots = p.max_positions - len([s for s in ctx.held if s not in exiting])
        if slots > 0:
            ranked = [
                (s, v) for s, v in sorted(scores.items(), key=lambda kv: -kv[1])
                if v >= p.entry_score_min
                and not ctx.holds(s)
                and s not in exiting
                and not ctx.on_cooldown(s)
                and not ctx.has_open_order(s)
            ]
            for sym, sc in ranked[:slots]:
                o = ctx.rebalance_to_weight(
                    sym, p.target_weight_per_name, tolerance=0.01,
                    reason=f"momentum entry (score {sc:.2f}, adx {diags[sym].get('adx', 0):.0f})",
                    tag="entry",
                )
                if not o:
                    continue
                intents.append(o)
                st = ctx.sym_state(sym)
                st.update({
                    "entry_price": ctx.price(sym),
                    "entry_atr": diags[sym].get("atr", 0.0),
                    "entry_bar": ctx.bar_clock(),
                    "adds": 0,
                })
                st.pop("stop", None)      # fresh trail for a fresh position

        if intents:
            ctx.log.debug(
                "trend_rider tick %d: %s", ctx.tick_index,
                ", ".join(f"{o.describe()} [{o.reason}]" for o in intents),
            )
        return intents
