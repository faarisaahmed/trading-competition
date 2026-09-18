"""Stat Arb's picker: cointegrated pairs, not individual names.

Philosophy match: the strategy trades spreads, so its picker must return
*pairs*. It screens every combination in the candidate pool on correlation,
hedge-ratio sanity, ADF stationarity and half-life -- the same four tests the
strategy applies intraday, run here on daily bars over a longer window. The
returned shortlist is the flattened membership of the best pairs, and the
pairs themselves are handed through `extra["pairs"]` so the strategy does not
have to rediscover them.

The same-sector bonus encodes a real prior: cointegration between two
railroads is far more likely to be structural than cointegration between a
railroad and a biotech, which is more likely to be a coincidence of the
sample window.
"""

from __future__ import annotations

import itertools
import math

from ..strategies.base import Param
from ..util import indicators as ind
from .base import Picker, PickerContext, PickResult


class PairPicker(Picker):
    DESCRIPTION = (
        "Screens candidate pairs on log-price correlation, OLS hedge ratio, ADF t-stat and "
        "OU half-life; returns the members of the best cointegrated pairs plus the pair "
        "definitions."
    )
    MIN_HISTORY = 120

    PARAMS = (
        Param("lookback_days", 180, "formation window in days", minimum=60),
        Param("max_candidates", 60, "pre-filter the pool to this many names first",
              minimum=4),
        Param("min_correlation", 0.70, "log-price correlation floor", minimum=0.0,
              maximum=1.0),
        Param("adf_tstat_max", -2.2, "Dickey-Fuller ceiling for the spread"),
        Param("max_half_life_days", 20.0, "spread must revert within this many days",
              minimum=0.5),
        Param("same_sector_bonus", 0.15, "score bonus for an in-sector pair", minimum=0.0),
        Param("pairs_to_return", 5, "how many pairs to hand to the strategy", minimum=1),
        Param("min_avg_dollar_volume", 15000000.0, "liquidity floor per leg", minimum=0.0),
    )

    def pick(self, ctx: PickerContext) -> PickResult:
        p = self.p
        # Pre-filter: the pair screen is O(n^2), so trim the pool to the most
        # liquid names with enough history before combining.
        eligible = [
            s for s in ctx.candidates
            if ctx.has_history(s, self.MIN_HISTORY)
            and ctx.avg_dollar_volume(s) >= p.min_avg_dollar_volume
        ]
        eligible.sort(key=lambda s: -ctx.avg_dollar_volume(s))
        eligible = eligible[: p.max_candidates]
        if len(eligible) < 2:
            ctx.log.warning(
                "pair_picker: only %d eligible name(s); cannot form a pair", len(eligible)
            )
            return PickResult(tuple(eligible[: ctx.max_symbols]))

        pairs: list[dict] = []
        for a, b in itertools.combinations(sorted(eligible), 2):
            m = self._evaluate(ctx, a, b)
            if m:
                pairs.append(m)
        pairs.sort(key=lambda d: -d["score"])

        # Take the best pairs with disjoint legs where possible, so a single
        # broken relationship cannot take out the whole shortlist.
        chosen: list[dict] = []
        used: set[str] = set()
        for cand in pairs:
            if cand["y"] in used or cand["x"] in used:
                continue
            chosen.append(cand)
            used.update((cand["y"], cand["x"]))
            if len(chosen) >= p.pairs_to_return or len(used) >= ctx.max_symbols:
                break
        for cand in pairs:                        # backfill if we came up short
            if len(used) >= ctx.max_symbols or len(chosen) >= p.pairs_to_return:
                break
            if cand in chosen:
                continue
            chosen.append(cand)
            used.update((cand["y"], cand["x"]))

        symbols: list[str] = []
        scores: dict[str, float] = {}
        notes: dict[str, str] = {}
        for cand in chosen:
            for leg in (cand["y"], cand["x"]):
                if leg not in symbols and len(symbols) < ctx.max_symbols:
                    symbols.append(leg)
                    scores[leg] = cand["score"]
                    notes[leg] = (
                        f"pair {cand['y']}/{cand['x']} corr {cand['corr']:.2f} "
                        f"adf {cand['tstat']:.2f} hl {cand['half_life']:.1f}d"
                    )

        # Backfill toward the cap with the names most likely to *form* a pair,
        # measured as each candidate's highest correlation to anything else in
        # the pool. Returning only the two legs of one pair boxes the strategy
        # in: if its intraday screen disagrees with this daily one, the team is
        # left holding two unusable names instead of ten workable ones. This
        # is still a pair-trading screen, so the picker keeps its philosophy.
        if len(symbols) < ctx.max_symbols:
            best_corr: dict[str, float] = {}
            for cand in pairs:
                for leg in (cand["y"], cand["x"]):
                    best_corr[leg] = max(best_corr.get(leg, 0.0), cand["corr"])
            remaining = sorted(
                (s for s in eligible if s not in symbols),
                key=lambda s: (-best_corr.get(s, 0.0), -ctx.avg_dollar_volume(s)),
            )
            for sym in remaining:
                if len(symbols) >= ctx.max_symbols:
                    break
                symbols.append(sym)
                scores[sym] = best_corr.get(sym, 0.0)
                notes[sym] = (
                    f"backfill (best pairwise corr {best_corr.get(sym, 0.0):.2f}) "
                    f"-- room for the strategy's own intraday pair search"
                )

        result = PickResult(tuple(symbols), scores, notes)
        result.extra["pairs"] = [
            {
                "y": c["y"], "x": c["x"], "beta": round(c["beta"], 6),
                "corr": round(c["corr"], 4), "tstat": round(c["tstat"], 3),
                "half_life_days": round(c["half_life"], 2), "score": round(c["score"], 4),
            }
            for c in chosen
        ]
        result.extra["pairs_screened"] = len(pairs)
        ctx.log.info(
            "pair_picker: %d of %d combinations passed; chose %s",
            len(pairs), len(eligible) * (len(eligible) - 1) // 2,
            ", ".join(f"{c['y']}/{c['x']}" for c in chosen) or "nothing",
        )
        return result

    def _evaluate(self, ctx: PickerContext, a: str, b: str) -> dict | None:
        p = self.p
        ya = ctx.closes(a, p.lookback_days)
        xa = ctx.closes(b, p.lookback_days)
        n = min(len(ya), len(xa))
        if n < self.MIN_HISTORY or min(ya[-n:]) <= 0 or min(xa[-n:]) <= 0:
            return None
        ly = [math.log(v) for v in ya[-n:]]
        lx = [math.log(v) for v in xa[-n:]]
        corr = ind.correlation(ly, lx)
        if corr < p.min_correlation:
            return None
        beta = ind.hedge_ratio(ly, lx)
        if not 0.1 <= abs(beta) <= 10.0:
            return None
        spread = ind.spread_series(ly, lx, beta)
        tstat = ind.adf_tstat(spread)
        if tstat > p.adf_tstat_max:
            return None
        hl = ind.half_life(spread)
        if not math.isfinite(hl) or hl <= 0 or hl > p.max_half_life_days:
            return None
        score = abs(tstat) * corr / max(hl, 1.0)
        if ctx.sector(a) == ctx.sector(b) != "Unknown":
            score *= 1.0 + p.same_sector_bonus
        return {
            "y": a, "x": b, "beta": beta, "corr": corr, "tstat": tstat,
            "half_life": hl, "score": score,
        }
