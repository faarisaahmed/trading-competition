"""Vol Breakout's picker: coiled springs, not current winners.

Philosophy match: the strategy buys range expansion out of quiet, so the
picker looks for names that are *currently quiet and range-bound* on a daily
basis -- the pre-condition for the setup, not the setup itself. It uses the
same TTM squeeze test (Bollinger inside Keltner) the strategy uses intraday,
plus a count of consecutive quiet days.

This is the one picker in the field that prefers names where *nothing is
happening*, which is precisely why it pairs with a breakout strategy: by the
time something is happening, the entry is gone.
"""

from __future__ import annotations

from ..strategies.base import Param
from ..util import indicators as ind
from .base import Picker, PickerContext, PickResult


class SqueezePicker(Picker):
    DESCRIPTION = (
        "Screens for daily volatility compression -- Bollinger inside Keltner, low "
        "bandwidth percentile, consecutive narrow-range days -- while requiring enough "
        "underlying ATR that a release is worth trading."
    )
    MIN_HISTORY = 90

    PARAMS = (
        Param("lookback_days", 120, "bandwidth history window", minimum=40),
        Param("bandwidth_percentile_max", 0.30, "bandwidth must be in this bottom fraction",
              minimum=0.0, maximum=1.0),
        Param("squeeze_period", 20, "period for the Bollinger/Keltner test", minimum=5),
        Param("bollinger_mult", 2.0, "Bollinger multiple", minimum=0.5),
        Param("keltner_mult", 1.5, "Keltner ATR multiple", minimum=0.5),
        Param("min_atr_pct", 0.012,
              "floor on the name's NORMAL ATR% (over `normal_atr_days`), so a "
              "release is worth trading", minimum=0.0),
        Param("normal_atr_days", 60,
              "window for the name's normal range; a coiled name's recent ATR is "
              "low by definition, so gating on it would reject the best setups",
              minimum=10),
        Param("min_avg_dollar_volume", 25000000.0, "liquidity floor", minimum=0.0),
        Param("prefer_coiled_days", 5, "consecutive narrow-range days to look for",
              minimum=1),
        Param("max_per_sector", 4, "sector concentration cap", minimum=1),
    )

    def pick(self, ctx: PickerContext) -> PickResult:
        p = self.p
        scored: list[tuple[str, float, str]] = []
        rejected: dict[str, str] = {}

        for sym in ctx.candidates:
            bars = ctx.bars(sym)
            closes = [b.close for b in bars]
            if len(closes) < self.MIN_HISTORY:
                rejected[sym] = f"only {len(closes)} daily bars"
                continue
            if ctx.avg_dollar_volume(sym) < p.min_avg_dollar_volume:
                rejected[sym] = "below the liquidity floor"
                continue
            # Gate on the name's *normal* range, not its current one. A tight
            # coil has a low 14-day ATR by construction, so gating on that
            # would reject precisely the setups this picker exists to find.
            # What matters is whether the name moves enough when it is *not*
            # coiled -- its longer-run ATR.
            atr_pct = ind.atr_pct(bars, 14)
            normal_atr_pct = ind.atr_pct(bars, min(p.normal_atr_days, len(bars) - 1))
            if normal_atr_pct < p.min_atr_pct:
                rejected[sym] = (
                    f"normal ATR {normal_atr_pct:.2%} -- a release would not pay"
                )
                continue

            # TTM squeeze on daily bars: is dispersion below the normal range?
            bb_lo, _mid, bb_hi = ind.bollinger(closes, p.squeeze_period, p.bollinger_mult)
            kc_lo, _kmid, kc_hi = ind.keltner(bars, p.squeeze_period, p.keltner_mult)
            squeezed = bb_hi < kc_hi and bb_lo > kc_lo

            # Bandwidth percentile, as a continuous measure of how tight it is.
            window = min(p.lookback_days, len(closes) - p.squeeze_period)
            history = [
                ind.bandwidth(closes[: len(closes) - i], p.squeeze_period, p.bollinger_mult)
                for i in range(window, 0, -1)
            ]
            bw_now = history[-1] if history else 0.0
            pct = ind.percentile_rank(history[:-1], bw_now) if len(history) > 10 else 1.0

            narrow_days = self._consecutive_narrow(bars, p.prefer_coiled_days)

            if not squeezed and pct > p.bandwidth_percentile_max:
                rejected[sym] = f"not compressed (bandwidth {pct:.0%}ile)"
                continue

            # The score is purely about compression. An earlier version added
            # a term rewarding high ATR, which made a 7%-a-day lottery ticket
            # outrank a genuine coil -- exactly backwards for a squeeze
            # screen. Whether a release pays is a *gate* (above), never a
            # ranking bonus.
            compression = 1.0 - min(atr_pct / max(normal_atr_pct, 1e-9), 1.0)
            score = (
                0.40 * (1.0 - pct)
                + 0.25 * (1.0 if squeezed else 0.0)
                + 0.20 * min(narrow_days / max(p.prefer_coiled_days, 1), 1.0)
                + 0.15 * compression
            )
            scored.append((
                sym, score,
                f"bandwidth {pct:.0%}ile{' +TTM' if squeezed else ''}, "
                f"{narrow_days} quiet days, ATR {atr_pct:.2%} vs normal "
                f"{normal_atr_pct:.2%}",
            ))

        result = self._rank(scored, ctx, max_per_sector=p.max_per_sector)
        result.rejected = rejected
        ctx.log.info(
            "squeeze_picker: %d/%d candidates compressed -> %s",
            len(scored), len(ctx.candidates), result.describe(),
        )
        return result

    @staticmethod
    def _consecutive_narrow(bars, target: int) -> int:
        """How many of the most recent days had a below-average true range."""
        if len(bars) < 25:
            return 0
        ranges = [b.range / b.close for b in bars[-25:] if b.close > 0]
        if len(ranges) < 10:
            return 0
        baseline = sum(ranges) / len(ranges)
        count = 0
        for r in reversed(ranges):
            if r < baseline:
                count += 1
            else:
                break
        return min(count, target * 2)
