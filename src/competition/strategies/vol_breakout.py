"""Team 7 -- Vol Breakout: buy the range expansion out of a squeeze.

Thesis
------
Volatility clusters and mean-reverts. Long quiet stretches resolve into fast
directional moves, and the resolution is usually large relative to the range
that preceded it. The trade is not "buy new highs" -- that is a coin flip in
chop -- it is "buy new highs *that come out of measured quiet*, and only when
the break is backed by real participation".

The three conditions, all required
----------------------------------
1. **Squeeze.** The Bollinger band sits entirely *inside* the Keltner channel
   -- realised dispersion has fallen below the ATR the name normally covers.
   This is the TTM-squeeze condition, and it is used here in preference to a
   bandwidth percentile for a specific reason: a percentile saturates with
   coil duration. A name quiet for sixty bars has a *median* bandwidth that
   is itself low, so it can never rank in its own bottom third and a
   percentile gate would reject the tightest setups on the board. The
   BB-inside-KC test is a ratio, so it is duration-independent. The squeeze
   need only have been on within the last `squeeze_release_bars` bars, since
   the breakout bar itself widens the bands.
2. **Break.** Price clears the `donchian_period` high by at least
   `breakout_buffer_atr` ATRs. The buffer matters: a break by one tick is
   noise, and being one tick above a level everyone is watching is where
   stop-hunting happens.
3. **Expansion.** Volume >= `volume_expansion_min` x its 20-bar average AND
   the bar's range >= `range_expansion_min` x its recent average. A breakout
   on no volume is a breakout nobody believes.

Opening range
-------------
When `use_opening_range` is on, the high of the first `opening_range_minutes`
of the session is treated as an additional breakout level. Overnight news
resolves in the first half hour and the opening range is the cleanest
intraday level there is.

Exits
-----
* **Initial stop** at `initial_stop_atr` ATRs below entry -- tight, because a
  true breakout should not revisit its base.
* **Chandelier trail** at `chandelier_mult` ATRs below the highest high since
  entry -- wide, because the whole thesis is that the move is large.
* **Failed-break stop.** If price closes back below the level it broke within
  `failed_break_bars` bars, the break failed. Exit immediately rather than
  waiting for the ATR stop; failed breakouts reverse hard and this rule is
  worth more than the stop placement.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

from ..types import Bar, OrderIntent
from ..util import indicators as ind
from .base import Param, Strategy, StrategyContext


class VolBreakout(Strategy):
    DESCRIPTION = (
        "Bollinger-bandwidth squeeze detection followed by a buffered Donchian / "
        "opening-range breakout, confirmed by volume and range expansion, exited on a "
        "Chandelier trail with a failed-break override."
    )

    DEFAULT_TICK_SECONDS = 300

    PARAMS = (
        Param("tick_seconds", 300, "seconds between evaluations", minimum=5),
        Param("donchian_period", 20, "breakout channel lookback in bars", minimum=3),
        Param("squeeze_lookback", 120, "bars of bandwidth history (ranking diagnostic)",
              minimum=20),
        Param("squeeze_period", 20, "period for both the Bollinger and Keltner bands",
              minimum=5),
        Param("bollinger_mult", 2.0, "Bollinger multiple for the squeeze test", minimum=0.5),
        Param("keltner_mult", 1.5, "Keltner ATR multiple; BB inside KC == squeeze",
              minimum=0.5),
        Param("squeeze_release_bars", 12, "how many bars back the squeeze may have been",
              minimum=0),
        Param("breakout_buffer_atr", 0.15, "ATRs the break must clear the channel by",
              minimum=0.0),
        Param("volume_expansion_min", 1.35, "volume vs its 20-bar average", minimum=0.0),
        Param("range_expansion_min", 1.20, "bar range vs its recent average", minimum=0.0),
        Param("atr_period", 14, "ATR period", minimum=2),
        Param("chandelier_mult", 2.75, "trail distance below the run-up high, in ATRs",
              minimum=0.1),
        Param("initial_stop_atr", 1.75, "initial stop distance in ATRs", minimum=0.1),
        Param("max_positions", 3, "concurrent positions", minimum=1),
        Param("weight_per_name", 0.31, "weight per position", minimum=0.0, maximum=1.0),
        Param("opening_range_minutes", 30, "length of the opening range", minimum=1),
        Param("use_opening_range", True, "also break the session's opening-range high"),
        Param("failed_break_bars", 4, "bars in which a break must hold", minimum=1),
        Param("cooldown_bars_after_stop", 8, "bars off after a stop", minimum=0),
        Param("min_bars_required", 130, "warm-up bars before trading", minimum=30),
    )

    # ------------------------------------------------------------------ #
    # setup detection
    # ------------------------------------------------------------------ #

    def bandwidth_series(self, bars: Sequence[Bar], period: int = 20) -> list[float]:
        """Rolling Bollinger bandwidth, one value per bar, oldest first."""
        p = self.p
        closes = [b.close for b in bars]
        n = len(closes)
        window = min(p.squeeze_lookback + p.squeeze_release_bars + 1, n - period)
        if window < 10:
            return []
        return [ind.bandwidth(closes[: n - i], period) for i in range(window, 0, -1)]

    def squeeze_on(self, bars: Sequence[Bar], lag: int = 0) -> bool:
        """TTM squeeze: is the Bollinger band inside the Keltner channel?

        Both are centred on the same 20-bar mean, so this reduces to
        "is 2 x stdev(close) < 1.5 x ATR" -- realised dispersion has dropped
        below the range the name normally travels. Scale-free and
        duration-independent.
        """
        p = self.p
        window = bars[:len(bars) - lag] if lag else bars
        if len(window) < p.squeeze_period + 2:
            return False
        closes = [b.close for b in window]
        bb_lo, _bb_mid, bb_hi = ind.bollinger(closes, p.squeeze_period, p.bollinger_mult)
        kc_lo, _kc_mid, kc_hi = ind.keltner(window, p.squeeze_period, p.keltner_mult)
        if kc_hi <= kc_lo or bb_hi <= bb_lo:
            return False
        return bb_hi < kc_hi and bb_lo > kc_lo

    def squeeze_lag(self, bars: Sequence[Bar]) -> int:
        """How many bars ago the squeeze was last on; -1 if not recently."""
        for lag in range(int(self.p.squeeze_release_bars) + 1):
            if self.squeeze_on(bars, lag):
                return lag
        return -1

    def squeeze_percentile(self, bars: Sequence[Bar]) -> float:
        """Bandwidth percentile -- used only to rank qualifying setups."""
        p = self.p
        series = self.bandwidth_series(bars)
        if len(series) < 12:
            return 1.0
        history = series[max(0, len(series) - 1 - p.squeeze_lookback):-1]
        if len(history) < 10:
            return 1.0
        return ind.percentile_rank(history, series[-1])

    def opening_range_high(self, ctx: StrategyContext, symbol: str) -> float:
        """High of the first N minutes of today's session, 0 if not yet formed."""
        p = self.p
        session = ctx.session
        if not (session.is_trading_day and session.open_at):
            return 0.0
        if session.minutes_since_open < p.opening_range_minutes:
            return 0.0            # range still forming -- no level yet
        cutoff = session.open_at + timedelta(minutes=p.opening_range_minutes)
        highs = [
            b.high for b in ctx.bars(symbol)
            if session.open_at <= b.ts < cutoff
        ]
        return max(highs) if highs else 0.0

    def assess(self, ctx: StrategyContext, symbol: str) -> dict[str, float] | None:
        p = self.p
        bars = ctx.bars(symbol)
        need = max(p.donchian_period + 5, p.atr_period + 5, 40)
        if len(bars) < need:
            return None
        closes = [b.close for b in bars]
        px = ctx.price(symbol)
        atr = ind.atr(bars, p.atr_period)
        if px <= 0 or atr <= 0:
            return None

        # Exclude the current bar from the channel: breaking a level that
        # includes your own high is trivially true.
        channel_high, channel_low = ind.donchian(bars[:-1], p.donchian_period)
        bw_now = ind.bandwidth(closes, p.squeeze_period, p.bollinger_mult)
        squeeze_pct = self.squeeze_percentile(bars)
        sq_lag = self.squeeze_lag(bars)

        vols = [b.volume for b in bars[-21:-1]]
        avg_vol = (sum(vols) / len(vols)) if vols else 0.0
        vol_ratio = (bars[-1].volume / avg_vol) if avg_vol > 0 else 0.0
        ranges = [b.range for b in bars[-21:-1]]
        avg_range = (sum(ranges) / len(ranges)) if ranges else 0.0
        range_ratio = (bars[-1].range / avg_range) if avg_range > 0 else 0.0

        or_high = self.opening_range_high(ctx, symbol) if p.use_opening_range else 0.0
        trigger = channel_high + p.breakout_buffer_atr * atr
        or_trigger = (or_high + p.breakout_buffer_atr * atr) if or_high > 0 else float("inf")
        broke_channel = px > trigger
        broke_or = px > or_trigger

        return {
            "price": px, "atr": atr, "atr_pct": atr / px,
            "channel_high": channel_high, "channel_low": channel_low,
            "trigger": trigger, "or_high": or_high, "or_trigger": or_trigger,
            "bandwidth": bw_now, "squeeze_pct": squeeze_pct,
            "squeeze_lag": float(sq_lag), "squeezed": float(sq_lag >= 0),
            "vol_ratio": vol_ratio, "range_ratio": range_ratio,
            "broke": float(broke_channel or broke_or),
            "broke_channel": float(broke_channel), "broke_or": float(broke_or),
            "level": trigger if broke_channel else (or_trigger if broke_or else trigger),
            # Rank candidates by how emphatic the break is.
            "thrust": (px - channel_high) / atr if atr > 0 else 0.0,
        }

    def qualifies(self, m: dict[str, float]) -> bool:
        p = self.p
        return (
            m["broke"] > 0
            and m["squeezed"] > 0
            and m["vol_ratio"] >= p.volume_expansion_min
            and m["range_ratio"] >= p.range_expansion_min
        )

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def on_round_start(self, ctx: StrategyContext) -> None:
        ctx.state.setdefault("_symbols", {})
        ctx.log.info(
            "vol_breakout: BB(%d,%.1f) inside KC(%d,%.1f) within %db, then a break "
            "+%.2f ATR over the %d-bar high on >=%.2fx volume",
            self.p.squeeze_period, self.p.bollinger_mult, self.p.squeeze_period,
            self.p.keltner_mult, self.p.squeeze_release_bars, self.p.breakout_buffer_atr,
            self.p.donchian_period, self.p.volume_expansion_min,
        )

    def on_tick(self, ctx: StrategyContext) -> list[OrderIntent]:
        p = self.p
        self.sync_bar_clock(ctx)
        intents: list[OrderIntent] = []

        metrics: dict[str, dict[str, float]] = {}
        for sym in ctx.candidates(self.warmup_bars):
            m = self.assess(ctx, sym)
            if m:
                metrics[sym] = m

        # ---------------- manage open breakouts --------------------------- #
        for sym in ctx.held:
            st = ctx.sym_state(sym)
            m = metrics.get(sym)
            px = ctx.price(sym)
            if px <= 0:
                continue
            atr = float(m["atr"]) if m else float(st.get("entry_atr", 0.0))
            entry = float(st.get("entry_price", 0.0))
            level = float(st.get("break_level", 0.0))
            bars_held = ctx.bar_clock() - int(st.get("entry_bar", ctx.bar_clock()))

            # Track the run-up high for the Chandelier trail.
            high_water = max(float(st.get("high_water", entry)), px)
            st["high_water"] = high_water

            reason = None
            if level > 0 and bars_held <= p.failed_break_bars and px < level:
                reason = f"failed break (back under {level:.2f} after {bars_held}b)"
            elif entry > 0 and atr > 0 and px <= entry - p.initial_stop_atr * atr:
                reason = f"initial stop ({(px / entry - 1):.1%})"
            elif atr > 0 and px <= high_water - p.chandelier_mult * atr:
                reason = f"chandelier trail from {high_water:.2f}"
            if reason:
                o = ctx.close(sym, reason=reason, tag="exit")
                if o:
                    intents.append(o)
                    st.clear()
                    ctx.set_cooldown(sym, p.cooldown_bars_after_stop)
        exiting = {o.symbol for o in intents}

        # ---------------- new breakouts ----------------------------------- #
        slots = p.max_positions - len([s for s in ctx.held if s not in exiting])
        if slots <= 0:
            return intents

        ready = [
            (sym, m) for sym, m in metrics.items()
            if self.qualifies(m)
            and not ctx.holds(sym)
            and sym not in exiting
            and not ctx.on_cooldown(sym)
            and not ctx.has_open_order(sym)
        ]
        # Freshest release, tightest coil, strongest thrust.
        ready.sort(key=lambda t: (t[1]["squeeze_lag"], t[1]["squeeze_pct"], -t[1]["thrust"]))

        for sym, m in ready[:slots]:
            o = ctx.rebalance_to_weight(
                sym, p.weight_per_name, tolerance=0.01,
                reason=(f"breakout over {m['level']:.2f} "
                        f"({'OR' if m['broke_or'] and not m['broke_channel'] else 'channel'}), "
                        f"squeeze released {int(m['squeeze_lag'])}b ago "
                        f"({m['squeeze_pct']:.0%}ile bandwidth), vol {m['vol_ratio']:.2f}x, "
                        f"range {m['range_ratio']:.2f}x"),
                tag="entry",
            )
            if not o:
                continue
            intents.append(o)
            ctx.sym_state(sym).update({
                "entry_price": m["price"],
                "entry_atr": m["atr"],
                "entry_bar": ctx.bar_clock(),
                "break_level": m["level"],
                "high_water": m["price"],
            })
        return intents
