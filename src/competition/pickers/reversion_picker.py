"""Mean Reverter's picker: find names that actually revert.

Philosophy match: the strategy's edge depends on a name having a *measurable*
tendency to snap back. So the picker screens on the statistics of reversion --
OU half-life, variance ratio, Hurst exponent, ADF t-stat -- rather than on
whether anything happens to be oversold today. It is explicitly looking for
the opposite of what the momentum picker wants, which is the point: the two
teams should not end up trading the same book.
"""

from __future__ import annotations

import math

from ..strategies.base import Param
from ..util import indicators as ind
from .base import Picker, PickerContext, PickResult


class ReversionPicker(Picker):
    DESCRIPTION = (
        "Screens for statistically mean-reverting names: short OU half-life, variance "
        "ratio below 1, Hurst below 0.5, stationary ADF t-stat, with enough volatility "
        "for the bands to be worth trading."
    )
    MIN_HISTORY = 90

    PARAMS = (
        Param("lookback_days", 120, "window for the reversion statistics", minimum=40),
        Param("max_half_life_days", 12.0, "must revert within this many days", minimum=0.5),
        Param("max_variance_ratio", 0.95, "VR(5) must be below this (anti-trending)",
              minimum=0.0),
        Param("max_hurst", 0.48, "Hurst exponent ceiling", minimum=0.0, maximum=1.0),
        Param("min_volatility", 0.015, "daily vol floor -- need a band to trade",
              minimum=0.0),
        Param("adf_tstat_max", -2.0, "Dickey-Fuller ceiling for stationarity"),
        Param("max_per_sector", 4, "sector concentration cap", minimum=1),
        Param("min_price", 5.0, "price floor", minimum=0.0),
    )

    def pick(self, ctx: PickerContext) -> PickResult:
        p = self.p
        scored: list[tuple[str, float, str]] = []
        rejected: dict[str, str] = {}

        for sym in ctx.candidates:
            closes = ctx.closes(sym, p.lookback_days)
            if len(closes) < self.MIN_HISTORY:
                rejected[sym] = f"only {len(closes)} daily bars"
                continue
            if closes[-1] < p.min_price:
                rejected[sym] = f"price {closes[-1]:.2f} below floor"
                continue

            vol = ind.realized_vol(closes, min(len(closes), 21))
            if vol < p.min_volatility:
                rejected[sym] = f"vol {vol:.3f} too low to be worth a band"
                continue
            hl = ind.half_life(closes)
            if not math.isfinite(hl) or hl <= 0 or hl > p.max_half_life_days:
                rejected[sym] = f"half-life {hl:.1f}d -- does not revert in time"
                continue
            vr = ind.variance_ratio(closes, 5)
            if vr > p.max_variance_ratio:
                rejected[sym] = f"variance ratio {vr:.2f} -- trending, not reverting"
                continue
            h = ind.hurst(closes)
            if h > p.max_hurst:
                rejected[sym] = f"Hurst {h:.2f} -- persistent, not reverting"
                continue
            adf = ind.adf_tstat(closes)
            if adf > p.adf_tstat_max:
                rejected[sym] = f"ADF t {adf:.2f} -- not stationary enough"
                continue

            # Prefer fast reversion, strong stationarity, low persistence, and
            # enough volatility that the round trip clears costs.
            score = (
                0.35 * (1.0 / max(hl, 0.5))
                + 0.25 * min(abs(adf) / 4.0, 1.5)
                + 0.20 * (1.0 - vr)
                + 0.10 * (0.5 - h) * 4.0
                + 0.10 * min(vol / 0.03, 2.0)
            )
            scored.append((
                sym, score,
                f"half-life {hl:.1f}d VR {vr:.2f} H {h:.2f} ADF {adf:.2f} vol {vol:.1%}",
            ))

        result = self._rank(scored, ctx, max_per_sector=p.max_per_sector)
        result.rejected = rejected
        ctx.log.info(
            "reversion_picker: %d/%d candidates passed -> %s",
            len(scored), len(ctx.candidates), result.describe(),
        )
        return result
