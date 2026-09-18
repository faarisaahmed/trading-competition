"""The Gambler's picker: maximum variance, by design.

Philosophy match: the strategy overbets Kelly and concentrates, so its edge
per trade must come with a large payoff distribution. This picker hunts
lottery-like names -- high realised volatility, frequent gaps, expanding
ranges, high beta to the market. It is the only picker in the field that
treats volatility as a feature rather than a cost, which is exactly how its
strategy treats it.
"""

from __future__ import annotations

from ..strategies.base import Param
from ..util import indicators as ind
from .base import Picker, PickerContext, PickResult


class LotteryPicker(Picker):
    DESCRIPTION = (
        "Ranks candidates by realised volatility, gap frequency, range expansion and "
        "upside skew -- deliberately selecting the highest-variance names available."
    )
    MIN_HISTORY = 40

    PARAMS = (
        Param("lookback_days", 63, "window for the variance statistics", minimum=20),
        Param("min_volatility", 0.030, "daily vol floor", minimum=0.0),
        Param("min_gap_frequency", 0.10, "fraction of days gapping over the threshold",
              minimum=0.0, maximum=1.0),
        Param("gap_threshold", 0.015, "what counts as a gap", minimum=0.0),
        Param("min_range_expansion", 1.05, "recent range vs its own baseline", minimum=0.0),
        Param("prefer_high_beta", True, "add a beta-to-the-pool term"),
        Param("max_price", 900.0, "keep positions divisible", minimum=1.0),
        Param("min_price", 2.0, "avoid sub-$2 names where spreads eat the edge",
              minimum=0.0),
        Param("min_avg_dollar_volume", 5000000.0, "must still be exitable", minimum=0.0),
    )

    def pick(self, ctx: PickerContext) -> PickResult:
        p = self.p
        scored: list[tuple[str, float, str]] = []
        rejected: dict[str, str] = {}

        # A crude market proxy for beta: the equal-weight mean return of the
        # whole candidate pool, day by day.
        pool_returns = self._pool_returns(ctx, p.lookback_days)

        for sym in ctx.candidates:
            bars = ctx.bars(sym)[-p.lookback_days:]
            closes = [b.close for b in bars]
            if len(closes) < self.MIN_HISTORY:
                rejected[sym] = f"only {len(closes)} daily bars"
                continue
            price = closes[-1]
            if not p.min_price <= price <= p.max_price:
                rejected[sym] = f"price {price:.2f} outside tradable band"
                continue
            if ctx.avg_dollar_volume(sym) < p.min_avg_dollar_volume:
                rejected[sym] = "too thin to exit a concentrated bet"
                continue

            vol = ind.realized_vol(closes, min(len(closes), p.lookback_days))
            if vol < p.min_volatility:
                rejected[sym] = f"vol {vol:.1%} -- not enough variance"
                continue

            gaps = [
                abs(b.open / prev.close - 1.0)
                for prev, b in zip(bars, bars[1:], strict=False) if prev.close > 0
            ]
            gap_freq = (
                sum(1 for g in gaps if g >= p.gap_threshold) / len(gaps) if gaps else 0.0
            )
            if gap_freq < p.min_gap_frequency:
                rejected[sym] = f"gap frequency {gap_freq:.0%} -- too well behaved"
                continue

            recent_range = sum(b.range / b.close for b in bars[-5:] if b.close > 0) / 5.0
            base_range = sum(b.range / b.close for b in bars[-40:] if b.close > 0) / min(
                len(bars[-40:]), 40
            )
            expansion = (recent_range / base_range) if base_range > 0 else 1.0
            if expansion < p.min_range_expansion:
                rejected[sym] = f"range expansion {expansion:.2f}x -- going quiet"
                continue

            rets = ind.pct_change(closes)
            upside = self._upside_skew(rets)
            beta = (
                ind.hedge_ratio(rets, pool_returns[-len(rets):])
                if p.prefer_high_beta and len(pool_returns) >= len(rets) > 10
                else 1.0
            )

            score = (
                0.40 * min(vol / 0.06, 2.5)
                + 0.20 * min(gap_freq / 0.30, 2.0)
                + 0.20 * min(expansion, 2.5)
                + 0.10 * max(min(upside, 2.0), -1.0)
                + 0.10 * max(min(beta, 3.0), 0.0)
            )
            scored.append((
                sym, score,
                f"vol {vol:.1%} gaps {gap_freq:.0%} exp {expansion:.2f}x "
                f"skew {upside:+.2f} beta {beta:.2f}",
            ))

        # No sector cap: concentration is the strategy, not a bug.
        result = self._rank(scored, ctx)
        result.rejected = rejected
        ctx.log.info(
            "lottery_picker: %d/%d candidates passed -> %s",
            len(scored), len(ctx.candidates), result.describe(),
        )
        return result

    @staticmethod
    def _pool_returns(ctx: PickerContext, days: int) -> list[float]:
        """Equal-weight mean daily return of the pool -- a market stand-in."""
        series = []
        for sym in ctx.candidates:
            closes = ctx.closes(sym, days + 1)
            if len(closes) >= days + 1:
                series.append(ind.pct_change(closes))
        if not series:
            return []
        n = min(len(s) for s in series)
        return [sum(s[-n:][i] for s in series) / len(series) for i in range(n)]

    @staticmethod
    def _upside_skew(returns: list[float]) -> float:
        """Mean of the best decile over the absolute mean of the worst decile."""
        if len(returns) < 20:
            return 0.0
        ordered = sorted(returns)
        k = max(len(ordered) // 10, 2)
        worst = -sum(ordered[:k]) / k
        best = sum(ordered[-k:]) / k
        return (best / worst) if worst > 1e-9 else 0.0
