"""Team 4 -- The Gambler: deliberately high-variance Kelly betting.

Thesis
------
Over a three-week, eight-way contest with points weighted toward the podium,
variance is not obviously your enemy. Finishing 5th with +2% scores 5 points;
finishing 1st with +14% scores 15. If the goal is *points* rather than
risk-adjusted return, a strategy that swings for the fence has a real case.
This team is that case, argued honestly and bounded so it cannot break the
competition.

Mechanics
---------
1. **Estimate the bet.** From the last `lookback_bars`, measure the up-bar
   frequency `p` and the payoff ratio `b` = mean up move / mean down move.
2. **Size it by Kelly.** f* = p - (1-p)/b, the growth-optimal fraction.
3. **Then overbet it.** Multiply by `kelly_fraction * kelly_multiplier`
   (0.85 * 1.6 = 1.36x full Kelly). Overbetting Kelly is *known* to reduce
   long-run growth and increase ruin probability -- that is the point. This
   is the variance-seeking entry in the field, and it is labelled as such.
4. **Climb the ladder on losses.** After a loser, step to the next rung of
   `martingale_rungs` (1.0 -> 1.45 -> 1.9). After a winner, reset to rung 0.
   Lose on the top rung and it sits out `cool_off_after_rungs` bars. A
   *bounded* martingale: three rungs, never more, so the classic
   "double until you're broke" failure mode is arithmetically impossible.
5. **Cut fast, run long.** Stop at `stop_atr` (2 ATR), target `take_profit_atr`
   (3 ATR), trail once 2 ATR onside. The asymmetry is where the positive
   expectancy has to come from, since the sizing is actively hostile to it.

Guardrails it imposes on itself (tighter than the engine's)
-----------------------------------------------------------
`session_loss_cap_pct` -- down 28% in a session and it flattens and stops for
the day. The engine's own kill switch sits at 35%; this team's job is to be
reckless with position size, not to be the reason a round has to be voided.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..types import Bar, OrderIntent
from ..util import indicators as ind
from .base import Param, Strategy, StrategyContext


class Gambler(Strategy):
    DESCRIPTION = (
        "Overbet fractional Kelly with a bounded three-rung martingale, concentrated in "
        "the highest-variance names, asymmetric ATR exits, and a self-imposed session "
        "loss cap."
    )

    DEFAULT_TICK_SECONDS = 180

    PARAMS = (
        Param("tick_seconds", 180, "seconds between evaluations", minimum=5),
        Param("lookback_bars", 60, "bars used to estimate p and b", minimum=20),
        Param("kelly_fraction", 0.85, "fraction of full Kelly", minimum=0.0, maximum=2.0),
        Param("kelly_multiplier", 1.6, "deliberate overbet multiplier", minimum=0.1, maximum=4.0),
        Param("min_edge", 0.015, "minimum Kelly edge required to bet at all", minimum=0.0),
        Param("max_positions", 2, "concurrent bets", minimum=1),
        Param("max_weight_per_name", 0.90, "hard cap per bet", minimum=0.0, maximum=1.0),
        Param("martingale_rungs", [1.0, 1.45, 1.9], "size multipliers by consecutive losses"),
        Param("cool_off_after_rungs", 8, "bars off after the ladder is exhausted", minimum=0),
        Param("take_profit_atr", 3.0, "profit target in ATRs", minimum=0.1),
        Param("stop_atr", 2.0, "stop distance in ATRs", minimum=0.1),
        Param("trail_after_atr", 2.0, "ATRs of profit before the stop starts trailing",
              minimum=0.0),
        Param("session_loss_cap_pct", 0.28, "self-imposed daily stop", minimum=0.01, maximum=0.9),
        Param("min_bars_required", 45, "warm-up bars before betting", minimum=10),
    )

    # ------------------------------------------------------------------ #
    # the bet
    # ------------------------------------------------------------------ #

    def kelly(self, bars: Sequence[Bar]) -> dict[str, float]:
        """Win probability, payoff ratio, Kelly fraction and a 'juice' score."""
        p = self.p
        if len(bars) < p.lookback_bars:
            return {}
        closes = [b.close for b in bars]
        rets = ind.pct_change(closes)[-p.lookback_bars:]
        if len(rets) < 20:
            return {}
        ups = [r for r in rets if r > 0]
        downs = [-r for r in rets if r < 0]
        if len(ups) < 3 or len(downs) < 3:
            return {}

        win_p = len(ups) / len(rets)
        avg_up = sum(ups) / len(ups)
        avg_dn = sum(downs) / len(downs)
        if avg_dn <= 1e-9:
            return {}
        b = avg_up / avg_dn
        # f* = p - q/b. Negative means the tape says don't bet.
        f = win_p - (1.0 - win_p) / b
        vol = ind.realized_vol(closes, min(len(closes), 40))
        atr_pct = ind.atr_pct(bars)
        # "Juice": how lottery-like the name is right now. Range expansion
        # relative to its own recent normal, times raw volatility.
        recent_range = sum(b_.range for b_ in bars[-5:]) / 5.0
        base_range = sum(b_.range for b_ in bars[-40:]) / 40.0
        expansion = (recent_range / base_range) if base_range > 0 else 1.0
        return {
            "p": win_p, "b": b, "f": f, "edge": f,
            "vol": vol, "atr_pct": atr_pct, "atr": ind.atr(bars),
            "expansion": expansion,
            "juice": max(f, 0.0) * vol * max(expansion, 0.5),
            "price": closes[-1],
        }

    def rung(self, ctx: StrategyContext) -> int:
        return int(ctx.recall("rung", 0))

    def size_multiplier(self, ctx: StrategyContext) -> float:
        rungs = list(self.p.martingale_rungs) or [1.0]
        return float(rungs[min(self.rung(ctx), len(rungs) - 1)])

    def bet_weight(self, ctx: StrategyContext, k: dict[str, float]) -> float:
        p = self.p
        raw = k["f"] * p.kelly_fraction * p.kelly_multiplier * self.size_multiplier(ctx)
        return max(min(raw, p.max_weight_per_name), 0.0)

    # ------------------------------------------------------------------ #
    # session risk
    # ------------------------------------------------------------------ #

    def _session_guard(self, ctx: StrategyContext) -> list[OrderIntent] | None:
        """Flatten and stand down if the self-imposed daily cap is breached."""
        key = ctx.session.session_date.isoformat()
        if ctx.recall("session_date") != key:
            ctx.remember("session_date", key)
            ctx.remember("session_open_equity", ctx.equity)
            ctx.remember("session_halted", False)
        open_eq = float(ctx.recall("session_open_equity", ctx.equity) or ctx.equity)
        if open_eq <= 0:
            return None
        drop = ctx.equity / open_eq - 1.0
        if ctx.recall("session_halted"):
            return ctx.flatten_all(reason="session loss cap -- stood down") or []
        if drop <= -self.p.session_loss_cap_pct:
            ctx.remember("session_halted", True)
            ctx.log.warning(
                "gambler: session down %.1f%%, hitting the self-imposed cap and flattening",
                drop * 100,
            )
            return ctx.flatten_all(reason=f"session loss cap ({drop:.1%})") or []
        return None

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def on_round_start(self, ctx: StrategyContext) -> None:
        ctx.state.setdefault("_symbols", {})
        ctx.remember("rung", 0)
        ctx.remember("wins", 0)
        ctx.remember("losses", 0)
        ctx.log.info(
            "gambler: %.2fx Kelly (%.2f frac x %.2f mult), rungs %s, cap %.0f%% per name",
            self.p.kelly_fraction * self.p.kelly_multiplier, self.p.kelly_fraction,
            self.p.kelly_multiplier, list(self.p.martingale_rungs),
            self.p.max_weight_per_name * 100,
        )

    def on_tick(self, ctx: StrategyContext) -> list[OrderIntent]:
        p = self.p
        self.sync_bar_clock(ctx)

        halted = self._session_guard(ctx)
        if halted is not None:
            return halted

        intents: list[OrderIntent] = []
        metrics = {}
        for sym in ctx.candidates(self.warmup_bars):
            k = self.kelly(ctx.bars(sym))
            if k:
                metrics[sym] = k

        # ---------------- manage open bets -------------------------------- #
        for sym in ctx.held:
            st = ctx.sym_state(sym)
            px = ctx.price(sym)
            entry = float(st.get("entry_price", 0.0))
            atr = float(st.get("entry_atr", 0.0))
            if px <= 0 or entry <= 0 or atr <= 0:
                continue
            gain_atr = (px - entry) / atr
            stop = float(st.get("stop", entry - p.stop_atr * atr))
            if gain_atr >= p.trail_after_atr:
                # Ratchet the stop up to lock in the run.
                stop = max(stop, px - p.stop_atr * atr)
                st["stop"] = stop

            reason = None
            if gain_atr >= p.take_profit_atr:
                reason = f"target hit (+{gain_atr:.1f} ATR)"
            elif px <= stop:
                reason = f"stopped ({gain_atr:+.1f} ATR)"
            if reason:
                o = ctx.close(sym, reason=reason, tag="exit")
                if o:
                    intents.append(o)
                    self._settle(ctx, sym, won=gain_atr > 0)
        exiting = {o.symbol for o in intents}

        # ---------------- place new bets ---------------------------------- #
        if ctx.recall("cool_off_until", 0) and ctx.bar_clock() < int(ctx.recall("cool_off_until")):
            return intents

        slots = p.max_positions - len([s for s in ctx.held if s not in exiting])
        if slots <= 0:
            return intents

        candidates = [
            (sym, k) for sym, k in metrics.items()
            if k["f"] >= p.min_edge
            and not ctx.holds(sym)
            and sym not in exiting
            and not ctx.on_cooldown(sym)
            and not ctx.has_open_order(sym)
        ]
        # Juiciest first: the biggest edge on the wildest name.
        candidates.sort(key=lambda t: -t[1]["juice"])
        for sym, k in candidates[:slots]:
            w = self.bet_weight(ctx, k)
            if w <= 0:
                continue
            o = ctx.rebalance_to_weight(
                sym, w, tolerance=0.01,
                reason=(f"kelly bet w={w:.2f} (p={k['p']:.2f} b={k['b']:.2f} f*={k['f']:.3f} "
                        f"rung {self.rung(ctx)} x{self.size_multiplier(ctx):.2f})"),
                tag="bet",
            )
            if not o:
                continue
            intents.append(o)
            ctx.sym_state(sym).update({
                "entry_price": k["price"],
                "entry_atr": k["atr"],
                "entry_bar": ctx.bar_clock(),
                "stop": k["price"] - p.stop_atr * k["atr"],
                "rung_at_entry": self.rung(ctx),
            })
        return intents

    # ------------------------------------------------------------------ #

    def _settle(self, ctx: StrategyContext, symbol: str, *, won: bool) -> None:
        """Move up or down the ladder after a resolved bet."""
        p = self.p
        rungs = list(p.martingale_rungs) or [1.0]
        st = ctx.sym_state(symbol)
        st.clear()
        if won:
            ctx.remember("wins", int(ctx.recall("wins", 0)) + 1)
            ctx.remember("rung", 0)
            return
        ctx.remember("losses", int(ctx.recall("losses", 0)) + 1)
        nxt = self.rung(ctx) + 1
        if nxt >= len(rungs):
            # Ladder exhausted. Reset and stand down -- this is the bound that
            # keeps a martingale from being a ruin machine.
            ctx.remember("rung", 0)
            ctx.remember("cool_off_until", ctx.bar_clock() + p.cool_off_after_rungs)
            ctx.log.info(
                "gambler: ladder exhausted after %d rungs, cooling off %d bars",
                len(rungs), p.cool_off_after_rungs,
            )
        else:
            ctx.remember("rung", nxt)

    def on_round_end(self, ctx: StrategyContext) -> None:
        w, l = int(ctx.recall("wins", 0)), int(ctx.recall("losses", 0))
        total = w + l
        ctx.log.info(
            "gambler: %d bets settled, %d won (%.0f%%), final rung %d",
            total, w, (w / total * 100) if total else 0.0, self.rung(ctx),
        )
