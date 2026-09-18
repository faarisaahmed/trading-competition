"""Team 2 -- Mean Reverter: buy dislocation, sell the snap back.

Thesis
------
Short-horizon price moves overshoot. A name two standard deviations below its
own twenty-bar mean, with a stretched oscillator and no fundamental reason to
be falling, tends to bounce toward that mean. The edge is small per trade and
the hit rate is high -- which means the entire game is (a) not buying things
that are falling for a *reason*, and (b) sizing so the occasional non-bounce
does not undo thirty winners.

Signal
------
Entry needs all of:
    z-score of price vs its own `band_period` mean  <=  entry_z
    RSI(`rsi_period`)                               <=  rsi_entry_max
    measured OU half-life                           <=  max_half_life_bars
    trailing regression slope                       >=  regime_filter_slope

That third condition is the one that separates this from a naive dip-buyer:
an AR(1) fit gives the half-life of the name's own mean reversion, and if the
series behaves like a random walk (half-life -> infinity) there is nothing to
revert *to* and the trade is declined regardless of how oversold it looks.
The fourth declines knife-catching in a genuine downtrend.

Sizing
------
Inverse volatility: the weight is scaled so every position contributes about
`target_vol_per_name` of per-bar risk, capped at `max_weight_per_name`.
Entries are split into two tranches -- the first at `entry_z`, the second only
if the dislocation *widens* to `second_tranche_z`. Averaging down is
dangerous in general and correct here, because the thesis is explicitly that
a wider gap is a better trade; the hard `stop_z` is what keeps that honest.

Exits
-----
Mean touched (`exit_z`), oscillator recovered (`rsi_exit_min`), the
dislocation blew through `stop_z` (thesis was wrong), or `time_stop_bars`
elapsed (thesis was too slow to be worth the capital).
"""

from __future__ import annotations

from collections.abc import Sequence

from ..types import Bar, OrderIntent
from ..util import indicators as ind
from .base import Param, Strategy, StrategyContext


class MeanReverter(Strategy):
    DESCRIPTION = (
        "Bollinger/RSI dislocation buyer with an AR(1) half-life filter, inverse-vol "
        "sizing, two-tranche scaling and a z-score stop."
    )

    DEFAULT_TICK_SECONDS = 300

    PARAMS = (
        Param("tick_seconds", 300, "seconds between evaluations", minimum=5),
        Param("band_period", 20, "mean/stdev window in bars", minimum=5),
        Param("band_mult", 2.0, "Bollinger multiple (diagnostics)", minimum=0.5),
        Param("entry_z", -2.0, "z-score at which the first tranche buys", maximum=0.0),
        Param("second_tranche_z", -3.0, "z-score for the second tranche", maximum=0.0),
        Param("exit_z", -0.20, "z-score that closes the trade"),
        Param("stop_z", -4.25, "z-score that admits the thesis was wrong", maximum=0.0),
        Param("rsi_period", 14, "RSI period", minimum=2),
        Param("rsi_entry_max", 32.0, "RSI must be at or below this to enter", minimum=0.0),
        Param("rsi_exit_min", 58.0, "RSI recovery that closes the trade", maximum=100.0),
        Param("max_half_life_bars", 40.0, "reject names slower to revert than this", minimum=1.0),
        Param("max_positions", 3, "concurrent names", minimum=1),
        Param("base_weight_per_name", 0.22, "pre-vol-scaling weight", minimum=0.0, maximum=1.0),
        Param("target_vol_per_name", 0.012, "per-bar vol budget per position", minimum=1e-5),
        Param("max_weight_per_name", 0.40, "hard weight cap", minimum=0.0, maximum=1.0),
        Param("time_stop_bars", 78, "bars before an unresolved trade is closed", minimum=1),
        Param("regime_filter_slope", -0.004, "reject names trending down faster than this"),
        Param("cooldown_bars_after_exit", 4, "bars to sit out after closing a name", minimum=0),
        Param("min_bars_required", 60, "warm-up bars before trading", minimum=10),
    )

    # -- measurement ------------------------------------------------------- #

    def assess(self, bars: Sequence[Bar]) -> dict[str, float]:
        """Everything the entry/exit rules need, measured once per name."""
        p = self.p
        if len(bars) < max(p.band_period * 2, p.rsi_period + 2, 30):
            return {}
        closes = [b.close for b in bars]
        price = closes[-1]
        if price <= 0:
            return {}
        lower, mid, upper = ind.bollinger(closes, p.band_period, p.band_mult)
        return {
            "price": price,
            "z": ind.zscore(closes, p.band_period),
            "rsi": ind.rsi(closes, p.rsi_period),
            "mid": mid,
            "lower": lower,
            "upper": upper,
            "half_life": ind.half_life(closes[-min(len(closes), p.band_period * 6):]),
            "slope": ind.linreg_slope(closes, p.band_period * 3),
            "vol": ind.realized_vol(closes, p.band_period),
            "atr_pct": ind.atr_pct(bars),
            "variance_ratio": ind.variance_ratio(closes, 5),
        }

    def target_weight(self, bars: Sequence[Bar], tranches: int) -> float:
        """Inverse-vol weight for `tranches` of the base size."""
        p = self.p
        w = self.inverse_vol_weight(
            bars, p.target_vol_per_name,
            base=p.base_weight_per_name, cap=p.max_weight_per_name, period=p.band_period,
        )
        return min(w * max(tranches, 1), p.max_weight_per_name)

    # -- lifecycle --------------------------------------------------------- #

    def on_round_start(self, ctx: StrategyContext) -> None:
        ctx.state.setdefault("_symbols", {})
        ctx.log.info(
            "mean_reverter: entry z<=%.2f, RSI<=%.0f, half-life<=%.0f bars",
            self.p.entry_z, self.p.rsi_entry_max, self.p.max_half_life_bars,
        )

    def on_tick(self, ctx: StrategyContext) -> list[OrderIntent]:
        p = self.p
        new_bar = self.sync_bar_clock(ctx)
        intents: list[OrderIntent] = []

        stats = {}
        for sym in ctx.candidates(self.warmup_bars):
            m = self.assess(ctx.bars(sym))
            if m:
                stats[sym] = m

        # ---------------- exits ------------------------------------------- #
        for sym in ctx.held:
            m = stats.get(sym)
            st = ctx.sym_state(sym)
            if not m:
                continue
            z, rsi = m["z"], m["rsi"]
            held_bars = ctx.bar_clock() - int(st.get("entry_bar", ctx.bar_clock()))
            reason = None
            if z <= p.stop_z:
                reason = f"dislocation widened past stop (z {z:.2f})"
            elif z >= p.exit_z:
                reason = f"reverted to mean (z {z:.2f})"
            elif rsi >= p.rsi_exit_min:
                reason = f"RSI recovered ({rsi:.0f})"
            elif held_bars >= p.time_stop_bars:
                reason = f"time stop after {held_bars} bars"
            if reason:
                o = ctx.close(sym, reason=reason, tag="exit")
                if o:
                    intents.append(o)
                    st.clear()
                    ctx.set_cooldown(sym, p.cooldown_bars_after_exit)
        exiting = {o.symbol for o in intents}

        # ---------------- second tranche ---------------------------------- #
        for sym in ctx.held:
            if sym in exiting:
                continue
            st = ctx.sym_state(sym)
            m = stats.get(sym)
            if not m or int(st.get("tranches", 1)) >= 2:
                continue
            if m["z"] > p.second_tranche_z:
                continue
            target = self.target_weight(ctx.bars(sym), 2)
            o = ctx.rebalance_to_weight(
                sym, target, tolerance=0.02,
                reason=f"second tranche (z {m['z']:.2f})", tag="scale_in",
            )
            if o:
                intents.append(o)
                st["tranches"] = 2

        # ---------------- entries ----------------------------------------- #
        slots = p.max_positions - len([s for s in ctx.held if s not in exiting])
        if slots > 0:
            eligible = []
            for sym, m in stats.items():
                if ctx.holds(sym) or sym in exiting or ctx.on_cooldown(sym):
                    continue
                if ctx.has_open_order(sym):
                    continue
                if m["z"] > p.entry_z or m["rsi"] > p.rsi_entry_max:
                    continue
                if m["half_life"] > p.max_half_life_bars:
                    continue                       # does not actually revert
                if m["slope"] < p.regime_filter_slope:
                    continue                       # falling knife
                eligible.append((sym, m))
            # Most dislocated first -- the biggest rubber band is the best trade.
            eligible.sort(key=lambda t: t[1]["z"])
            for sym, m in eligible[:slots]:
                target = self.target_weight(ctx.bars(sym), 1)
                o = ctx.rebalance_to_weight(
                    sym, target, tolerance=0.01,
                    reason=(f"oversold entry (z {m['z']:.2f}, RSI {m['rsi']:.0f}, "
                            f"half-life {m['half_life']:.0f}b)"),
                    tag="entry",
                )
                if not o:
                    continue
                intents.append(o)
                ctx.sym_state(sym).update({
                    "entry_price": m["price"],
                    "entry_z": m["z"],
                    "entry_bar": ctx.bar_clock(),
                    "tranches": 1,
                })

        if intents and new_bar:
            ctx.log.debug(
                "mean_reverter bar %d: %s", ctx.bar_clock(),
                ", ".join(f"{o.describe()} [{o.reason}]" for o in intents),
            )
        return intents
