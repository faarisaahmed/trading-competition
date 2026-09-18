"""Trend Rider's picker: find what is already working, cleanly.

Philosophy match: the strategy buys risk-adjusted momentum with a trend-quality
gate, so the picker screens for exactly that on a daily horizon. It looks for
names whose 21- and 63-day advances are large *relative to their own
volatility* and whose path is close to a straight line. It deliberately does
not look at valuation, mean reversion, or news -- those would be a different
team's screen.
"""

from __future__ import annotations

from ..strategies.base import Param
from ..util import indicators as ind
from .base import Picker, PickerContext, PickResult


class MomentumPicker(Picker):
    DESCRIPTION = (
        "Ranks candidates by volatility-adjusted 21d/63d return with an R^2 trend-quality "
        "filter and a distance-from-high tiebreak."
    )
    MIN_HISTORY = 70

    PARAMS = (
        Param("lookback_days_short", 21, "short momentum window", minimum=2),
        Param("lookback_days_long", 63, "long momentum window", minimum=5),
        Param("vol_lookback_days", 21, "window for the volatility denominator", minimum=5),
        Param("min_trend_r2", 0.20, "minimum path linearity", minimum=0.0, maximum=1.0),
        Param("top_n_multiple", 1.5, "screen this multiple of the slots, keep the best",
              minimum=1.0),
        Param("max_per_sector", 4, "sector concentration cap", minimum=1),
        Param("min_price", 3.0, "skip sub-penny-stock prices", minimum=0.0),
    )

    def pick(self, ctx: PickerContext) -> PickResult:
        p = self.p
        need = max(p.lookback_days_long + 5, self.MIN_HISTORY)
        scored: list[tuple[str, float, str]] = []
        rejected: dict[str, str] = {}

        for sym in ctx.candidates:
            closes = ctx.closes(sym)
            if len(closes) < need:
                rejected[sym] = f"only {len(closes)} daily bars"
                continue
            price = closes[-1]
            if price < p.min_price:
                rejected[sym] = f"price {price:.2f} below floor"
                continue

            vol = ind.realized_vol(closes, p.vol_lookback_days)
            if vol <= 1e-6:
                rejected[sym] = "no measurable volatility"
                continue
            r_short = ind.roc(closes, p.lookback_days_short)
            r_long = ind.roc(closes, p.lookback_days_long)
            r2 = ind.r_squared(closes, p.lookback_days_long)
            if r2 < p.min_trend_r2:
                rejected[sym] = f"path too noisy (R2 {r2:.2f})"
                continue
            if r_short <= 0 or r_long <= 0:
                rejected[sym] = "not advancing on both horizons"
                continue

            # Risk-adjusted momentum: return per unit of the volatility it
            # took to get there, on both horizons, weighted toward the
            # shorter one because the strategy trades intraday-to-weekly.
            sharpe_short = r_short / (vol * (p.lookback_days_short ** 0.5))
            sharpe_long = r_long / (vol * (p.lookback_days_long ** 0.5))
            high = max(closes[-p.lookback_days_long:])
            dist_from_high = (price / high - 1.0) if high > 0 else -1.0

            score = (
                0.55 * sharpe_short
                + 0.30 * sharpe_long
                + 0.15 * r2
                + 0.10 * dist_from_high * 10.0   # reward proximity to the high
            )
            scored.append((
                sym, score,
                f"{r_short:+.1%}/21d {r_long:+.1%}/63d R2={r2:.2f} "
                f"{dist_from_high:+.1%} off high",
            ))

        result = self._rank(scored, ctx, max_per_sector=p.max_per_sector)
        result.rejected = rejected
        ctx.log.info(
            "momentum_picker: %d/%d candidates passed -> %s",
            len(scored), len(ctx.candidates), result.describe(),
        )
        return result
