"""Buy & Hold -- the unscored reference line.

Not a competitor. It buys an equal-weight basket of whatever universe it is
given on the first tick it can trade, and then does nothing for the rest of
the week.

Every report prints it alongside the eight scored teams, because "+3.1% over
the week" means nothing on its own. If the market ran 4% and the winner made
3%, the interesting fact is that nobody beat the basket -- and without this
line in the table, nobody would notice.
"""

from __future__ import annotations

from ..types import OrderIntent
from .base import Param, Strategy, StrategyContext


class BuyAndHold(Strategy):
    DESCRIPTION = "Equal-weight buy and hold of the assigned universe. Unscored reference."

    DEFAULT_TICK_SECONDS = 900

    PARAMS = (
        Param("tick_seconds", 900, "seconds between checks", minimum=5),
        Param("max_positions", 10, "names in the basket", minimum=1),
        Param("cash_buffer_pct", 0.02, "cash left unspent to absorb slippage",
              minimum=0.0, maximum=0.5),
        Param("min_bars_required", 1, "no warm-up needed", minimum=0),
    )

    def on_round_start(self, ctx: StrategyContext) -> None:
        ctx.remember("bought", False)

    def on_tick(self, ctx: StrategyContext) -> list[OrderIntent]:
        if ctx.recall("bought"):
            return []
        if not ctx.session.is_open:
            return []
        names = [s for s in ctx.universe if ctx.tradable(s)][: self.p.max_positions]
        if not names:
            return []
        weight = (1.0 - self.p.cash_buffer_pct) / len(names)
        intents = ctx.allocate(
            {s: weight for s in names}, tolerance=0.005, exit_others=False,
            reason="equal-weight buy & hold", cash_buffer=self.p.cash_buffer_pct,
        )
        if intents:
            ctx.remember("bought", True)
            ctx.log.info(
                "benchmark: bought %d names at %.1f%% each", len(names), weight * 100
            )
        return intents
