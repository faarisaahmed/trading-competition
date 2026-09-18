"""Benchmark's picker: just own the biggest things.

Not a competitor's picker. Takes the most liquid names available and holds
them equal-weight, so the reference line in every report is "what the market
did", not "what a cleverer screen would have done".
"""

from __future__ import annotations

from ..strategies.base import Param
from .base import Picker, PickerContext, PickResult


class MegaCapPicker(Picker):
    DESCRIPTION = "Top names by dollar volume, equal weight. Unscored reference."
    MIN_HISTORY = 5

    PARAMS = (
        Param("top_n_by_value", 10, "basket size", minimum=1),
        Param("min_avg_dollar_volume", 50000000.0, "liquidity floor", minimum=0.0),
    )

    def pick(self, ctx: PickerContext) -> PickResult:
        p = self.p
        scored: list[tuple[str, float, str]] = []
        for sym in ctx.candidates:
            if not ctx.has_history(sym, self.MIN_HISTORY):
                continue
            adv = ctx.avg_dollar_volume(sym)
            if adv < p.min_avg_dollar_volume:
                continue
            scored.append((sym, adv, f"${adv / 1e6:.0f}M ADV"))
        limited = PickerContext(
            round_id=ctx.round_id, candidates=ctx.candidates, daily_bars=ctx.daily_bars,
            max_symbols=min(ctx.max_symbols, p.top_n_by_value), log=ctx.log,
            news=ctx.news, intraday_bars=ctx.intraday_bars, sectors=ctx.sectors,
            state=ctx.state, locked_universe=ctx.locked_universe,
        )
        result = self._rank(scored, limited)
        ctx.log.info("megacap_picker: basket of %d -> %s", len(result), ", ".join(result.symbols))
        return result
