"""Team 5 -- Stat Arb: relative value between cointegrated names.

Thesis
------
Two businesses exposed to the same drivers should trade at a stable ratio. The
ratio wanders; the wandering is mean-reverting even when neither leg is. Trade
the *spread*, and the market's direction stops mattering.

Pair formation (run once per round, refit every `refit_every_bars`)
-------------------------------------------------------------------
For every candidate pair (y, x) in the universe:

    1. correlation of log prices          >= min_correlation
    2. beta = OLS slope of log y on log x   (the hedge ratio)
    3. spread = log y - beta * log x
    4. Dickey-Fuller t-stat on the spread <= adf_tstat_max   (it's stationary)
    5. OU half-life of the spread         <= max_half_life_bars (it reverts *soon*)

Steps 4 and 5 are what separate this from "these two look correlated".
Correlation says they move together; stationarity says the *gap* comes back;
half-life says it comes back inside the round. All three or no trade.

Long-only expression (this competition has no shorting)
-------------------------------------------------------
A textbook pair trade is long the cheap leg and short the rich one. On a cash
account the short half is unavailable, so the position is expressed as a
**rotation**: hold whichever leg the spread says is cheap, and switch when the
spread flips.

    spread z >= +entry_z   ->  y is rich, x is cheap   ->  hold x
    spread z <= -entry_z   ->  y is cheap              ->  hold y
    |z| <= exit_z          ->  fair value              ->  flat
    |z| >= stop_z          ->  the relationship broke  ->  flat, blacklist

This keeps most of the edge (the relative-value signal) while accepting the
market beta the short leg would have hedged. It is an honest adaptation, not a
disguised directional bet: the entry decision is made purely on the spread.

Risk
----
`stop_z` retires a pair whose spread has run far enough to suggest the
cointegration was spurious or has genuinely broken -- a merger, a guidance
cut, an index change. `rebalance_z_delta` prevents churning the rotation on
every small wiggle of the spread.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Any

from ..types import Bar, OrderIntent
from ..util import indicators as ind
from .base import Param, Strategy, StrategyContext


class StatArb(Strategy):
    DESCRIPTION = (
        "Cointegration-screened pair trading (correlation + OLS hedge ratio + ADF t-stat + "
        "half-life), expressed long-only as a rotation into whichever leg the spread "
        "z-score says is cheap."
    )

    DEFAULT_TICK_SECONDS = 300

    PARAMS = (
        Param("tick_seconds", 300, "seconds between evaluations", minimum=5),
        Param("formation_bars", 240, "bars used to fit and score a pair", minimum=60),
        Param("min_correlation", 0.65, "minimum log-price correlation", minimum=0.0, maximum=1.0),
        Param("max_half_life_bars", 60.0, "spread must revert within this many bars",
              minimum=1.0),
        Param("adf_tstat_max", -1.8, "Dickey-Fuller t-stat ceiling for the spread"),
        Param("entry_z", 1.6, "spread z-score that opens a rotation", minimum=0.1),
        Param("exit_z", 0.35, "spread z-score that flattens", minimum=0.0),
        Param("stop_z", 3.75, "spread z-score that retires the pair", minimum=0.5),
        Param("max_pairs", 2, "concurrent pairs", minimum=1),
        Param("weight_per_pair", 0.42, "equity allocated to a pair's active leg",
              minimum=0.0, maximum=1.0),
        Param("rebalance_z_delta", 0.6, "z move needed before re-sizing", minimum=0.0),
        Param("refit_every_bars", 78, "bars between hedge-ratio refits", minimum=1),
        Param("time_stop_bars", 156, "bars before an unresolved rotation is closed",
              minimum=1),
        Param("min_bars_required", 120, "warm-up bars before trading", minimum=30),
        # Round 3 can deal ten names with no cointegrated pair among them. A
        # relaxed second pass lets the team trade the best relationship
        # available instead of sitting out the whole week -- at half size,
        # because the evidence is weaker.
        Param("allow_relaxed_pairs", True, "fall back to a relaxed screen if none pass"),
        Param("relaxed_correlation_delta", 0.15, "how much to loosen correlation",
              minimum=0.0),
        Param("relaxed_adf_delta", 0.8, "how much to loosen the ADF ceiling", minimum=0.0),
        Param("relaxed_half_life_mult", 1.6, "how much to loosen the half-life cap",
              minimum=1.0),
        Param("relaxed_weight_mult", 0.5, "size multiplier for a relaxed pair",
              minimum=0.0, maximum=1.0),
    )

    # ------------------------------------------------------------------ #
    # formation
    # ------------------------------------------------------------------ #

    def evaluate_pair(
        self, y_bars: Sequence[Bar], x_bars: Sequence[Bar], *, relaxed: bool = False
    ) -> dict[str, float] | None:
        """Fit and score one candidate pair. None if it fails any screen."""
        p = self.p
        min_corr = p.min_correlation - (p.relaxed_correlation_delta if relaxed else 0.0)
        adf_max = p.adf_tstat_max + (p.relaxed_adf_delta if relaxed else 0.0)
        max_hl = p.max_half_life_bars * (p.relaxed_half_life_mult if relaxed else 1.0)
        n = min(len(y_bars), len(x_bars), p.formation_bars)
        if n < max(60, p.formation_bars // 3):
            return None
        y = [b.close for b in y_bars[-n:]]
        x = [b.close for b in x_bars[-n:]]
        if min(y) <= 0 or min(x) <= 0:
            return None
        ly = [math.log(v) for v in y]
        lx = [math.log(v) for v in x]

        corr = ind.correlation(ly, lx)
        if corr < min_corr:
            return None
        beta = ind.hedge_ratio(ly, lx)
        if not (0.05 <= abs(beta) <= 20.0):
            return None          # a hedge ratio that extreme is a fitting artefact
        spread = ind.spread_series(ly, lx, beta)
        if len(spread) < 40:
            return None
        tstat = ind.adf_tstat(spread)
        if tstat > adf_max:
            return None
        hl = ind.half_life(spread)
        if not math.isfinite(hl) or hl <= 0 or hl > max_hl:
            return None
        z = ind.zscore(spread, min(len(spread), p.formation_bars))
        sd = (sum((v - sum(spread) / len(spread)) ** 2 for v in spread) / (len(spread) - 1)) ** 0.5
        if sd <= 1e-8:
            return None
        # Rank by how trustworthy *and* how fast the reversion is.
        quality = abs(tstat) * corr / max(hl, 1.0)
        return {
            "beta": beta, "corr": corr, "tstat": tstat, "half_life": hl,
            "z": z, "sd": sd, "quality": quality, "bars": float(n),
            "relaxed": 1.0 if relaxed else 0.0,
        }

    def form_pairs(self, ctx: StrategyContext) -> list[dict[str, Any]]:
        """Screen every combination in the universe and keep the best."""
        p = self.p
        symbols = ctx.candidates(self.warmup_bars)
        blacklist = set(ctx.recall("blacklist", []) or [])
        out: list[dict[str, Any]] = []
        # Cap the combinatorics: Round 2/3 universes of 10-12 names give 45-66
        # pairs, which is fine, but guard against a pathological config.
        pairs = list(itertools.combinations(sorted(symbols), 2))
        if len(pairs) > 200:
            pairs = pairs[:200]
        for a, b in pairs:
            if f"{a}/{b}" in blacklist:
                continue
            m = self.evaluate_pair(ctx.bars(a), ctx.bars(b))
            if m is None:
                continue
            out.append({"y": a, "x": b, **m})

        # Second pass on a relaxed screen, only if the strict one found
        # nothing. A pairs desk handed ten unrelated names would widen its
        # tolerance and trade smaller, not go flat for a week -- but the
        # relaxation is bounded, recorded on the pair, and halves the size.
        if not out and p.allow_relaxed_pairs:
            for a, b in pairs:
                if f"{a}/{b}" in blacklist:
                    continue
                m = self.evaluate_pair(ctx.bars(a), ctx.bars(b), relaxed=True)
                if m is None:
                    continue
                out.append({"y": a, "x": b, **m})
            if out:
                ctx.log.info(
                    "stat_arb: no pair passed the strict screen; %d passed relaxed "
                    "(corr>=%.2f, adf<=%.2f, hl<=%.0fb) and will trade at %.0f%% size",
                    len(out), p.min_correlation - p.relaxed_correlation_delta,
                    p.adf_tstat_max + p.relaxed_adf_delta,
                    p.max_half_life_bars * p.relaxed_half_life_mult,
                    p.relaxed_weight_mult * 100,
                )
        out.sort(key=lambda d: -d["quality"])

        # Don't let one symbol anchor every selected pair -- if it breaks,
        # the whole book breaks with it.
        selected: list[dict[str, Any]] = []
        used: set[str] = set()
        for cand in out:
            if cand["y"] in used or cand["x"] in used:
                continue
            selected.append(cand)
            used.update((cand["y"], cand["x"]))
            if len(selected) >= p.max_pairs:
                break
        # If disjointness left us short (a 3-name Round 1 universe can only
        # form one disjoint pair), backfill with the best overlapping pairs.
        if len(selected) < p.max_pairs:
            for cand in out:
                if cand in selected:
                    continue
                selected.append(cand)
                if len(selected) >= p.max_pairs:
                    break
        return selected

    def current_z(self, ctx: StrategyContext, pair: dict[str, Any]) -> float | None:
        """Live spread z-score using the pair's fitted beta."""
        y_bars, x_bars = ctx.bars(pair["y"]), ctx.bars(pair["x"])
        n = min(len(y_bars), len(x_bars), self.p.formation_bars)
        if n < 40:
            return None
        y = [b.close for b in y_bars[-n:]]
        x = [b.close for b in x_bars[-n:]]
        if min(y) <= 0 or min(x) <= 0:
            return None
        spread = ind.spread_series(
            [math.log(v) for v in y], [math.log(v) for v in x], float(pair["beta"])
        )
        return ind.zscore(spread, len(spread))

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def on_round_start(self, ctx: StrategyContext) -> None:
        ctx.state.setdefault("_symbols", {})
        ctx.remember("blacklist", [])
        pairs = self.form_pairs(ctx)
        ctx.remember("pairs", pairs)
        ctx.remember("last_fit_bar", ctx.bar_clock())
        if pairs:
            ctx.log.info(
                "stat_arb: formed %d pair(s): %s", len(pairs),
                "; ".join(
                    f"{q['y']}/{q['x']} beta={q['beta']:.3f} corr={q['corr']:.2f} "
                    f"adf={q['tstat']:.2f} hl={q['half_life']:.0f}b"
                    for q in pairs
                ),
            )
        else:
            ctx.log.warning(
                "stat_arb: no pair in %d names passed the cointegration screen; "
                "will retry each refit", len(ctx.universe),
            )

    def on_tick(self, ctx: StrategyContext) -> list[OrderIntent]:
        p = self.p
        self.sync_bar_clock(ctx)
        bar = ctx.bar_clock()

        pairs: list[dict[str, Any]] = list(ctx.recall("pairs", []) or [])
        if not pairs or bar - int(ctx.recall("last_fit_bar", 0)) >= p.refit_every_bars:
            refit = self.form_pairs(ctx)
            if refit:
                # Carry forward the rotation state of pairs that survived.
                prior = {f"{q['y']}/{q['x']}": q for q in pairs}
                for q in refit:
                    old = prior.get(f"{q['y']}/{q['x']}")
                    if old:
                        q["leg"] = old.get("leg")
                        q["entry_bar"] = old.get("entry_bar")
                        q["entry_z"] = old.get("entry_z")
                pairs = refit
                ctx.remember("pairs", pairs)
            ctx.remember("last_fit_bar", bar)

        if not pairs:
            return ctx.flatten_all(reason="no viable pair")

        weights: dict[str, float] = {}
        notes: dict[str, str] = {}
        for pair in pairs:
            z = self.current_z(ctx, pair)
            tag = f"{pair['y']}/{pair['x']}"
            if z is None:
                continue
            pair["z"] = z
            leg = pair.get("leg")
            entry_bar = int(pair.get("entry_bar") or 0)

            # --- retire a broken relationship ---------------------------- #
            if abs(z) >= p.stop_z:
                if leg:
                    ctx.log.info("stat_arb: %s spread hit z=%.2f, retiring the pair", tag, z)
                bl = list(ctx.recall("blacklist", []) or [])
                if tag not in bl:
                    bl.append(tag)
                ctx.remember("blacklist", bl)
                pair["leg"] = None
                continue

            # --- flatten at fair value or on the time stop --------------- #
            if leg and (abs(z) <= p.exit_z or (bar - entry_bar) >= p.time_stop_bars):
                pair["leg"] = None
                continue

            # --- open or maintain the rotation --------------------------- #
            want = None
            if z >= p.entry_z:
                want = pair["x"]          # y rich  -> own the cheap leg, x
            elif z <= -p.entry_z:
                want = pair["y"]          # y cheap -> own y
            elif leg:
                want = leg                # inside the band: hold what we have

            if want is None:
                pair["leg"] = None
                continue
            if leg != want:
                pair["leg"] = want
                pair["entry_bar"] = bar
                pair["entry_z"] = z
            size = p.weight_per_pair * (
                p.relaxed_weight_mult if pair.get("relaxed") else 1.0
            )
            weights[want] = weights.get(want, 0.0) + size
            notes[want] = (
                f"{tag} z={z:+.2f} (hold {'cheap ' if want == pair['x'] else ''}leg, "
                f"beta={pair['beta']:.3f}, hl={pair['half_life']:.0f}b"
                f"{', RELAXED screen' if pair.get('relaxed') else ''})"
            )

        ctx.remember("pairs", pairs)

        # Don't re-trade for a small spread wiggle inside the band.
        last_z = ctx.recall("last_alloc_z", {}) or {}
        cur_z = {f"{q['y']}/{q['x']}": round(float(q.get("z", 0.0)), 3) for q in pairs}
        settled = {
            k for k, v in cur_z.items()
            if k in last_z and abs(v - float(last_z[k])) < p.rebalance_z_delta
        }
        tolerance = 0.06 if len(settled) == len(cur_z) and settled else 0.02
        ctx.remember("last_alloc_z", cur_z)

        intents = ctx.allocate(
            weights, tolerance=tolerance, exit_others=True, reason="pair rotation"
        )
        for o in intents:
            o.reason = notes.get(o.symbol, o.reason or "pair rotation")
            o.tag = "pair"
        return intents

    def on_round_end(self, ctx: StrategyContext) -> None:
        bl = ctx.recall("blacklist", []) or []
        pairs = ctx.recall("pairs", []) or []
        ctx.log.info(
            "stat_arb: ended with %d live pair(s), %d retired: %s",
            len(pairs), len(bl), ", ".join(bl) or "none",
        )
