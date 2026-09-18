"""The Scalper's picker: the cheapest names to trade, not the best to own.

Philosophy match: a market maker's P&L is spread captured minus adverse
selection. It does not care where a stock is going; it cares that the stock
trades constantly, in tiny increments, without trending. So the picker ranks
on trade count, dollar volume, a daily-bar proxy for tightness, and -- unusually
-- *prefers low daily volatility*, because a trending name is where a
one-sided maker gets run over.

Note on spreads: the true bid/ask is not in daily bars, so the picker uses
dollar-volume-per-trade and the close-to-range ratio as proxies, and the
strategy re-checks the actual relative spread on every quote at runtime
(`min_relative_spread` / `max_relative_spread`). The picker narrows the field;
the strategy has the final say tick by tick.
"""

from __future__ import annotations

from ..strategies.base import Param
from ..util import indicators as ind
from .base import Picker, PickerContext, PickResult


class LiquidityPicker(Picker):
    DESCRIPTION = (
        "Ranks candidates on trade count, dollar volume and tightness proxies while "
        "preferring low daily volatility -- chop, not trend, is where a one-sided maker "
        "earns."
    )
    MIN_HISTORY = 25

    PARAMS = (
        Param("lookback_days", 21, "window for the liquidity statistics", minimum=5),
        Param("max_relative_spread", 0.0020, "proxy tightness ceiling", minimum=0.0),
        Param("min_trade_count", 20000, "average daily trades floor", minimum=0),
        Param("min_avg_dollar_volume", 100000000.0, "dollar volume floor", minimum=0.0),
        Param("prefer_low_daily_vol", True, "penalise trending/volatile names"),
        Param("max_daily_vol", 0.045, "hard volatility ceiling", minimum=0.0),
        Param("min_price", 10.0, "price floor -- a 1c tick is a huge relative spread",
              minimum=0.0),
        Param("max_per_sector", 6, "sector concentration cap", minimum=1),
    )

    def pick(self, ctx: PickerContext) -> PickResult:
        p = self.p
        scored: list[tuple[str, float, str]] = []
        rejected: dict[str, str] = {}

        for sym in ctx.candidates:
            bars = ctx.bars(sym)[-p.lookback_days:]
            if len(bars) < min(self.MIN_HISTORY, p.lookback_days):
                rejected[sym] = f"only {len(bars)} daily bars"
                continue
            closes = [b.close for b in bars]
            price = closes[-1]
            if price < p.min_price:
                rejected[sym] = f"price {price:.2f} -- tick size too coarse"
                continue

            adv = sum(b.dollar_volume for b in bars) / len(bars)
            if adv < p.min_avg_dollar_volume:
                rejected[sym] = f"ADV ${adv / 1e6:.1f}M below floor"
                continue
            trades = sum(b.trade_count for b in bars) / len(bars)
            if trades < p.min_trade_count:
                rejected[sym] = f"{trades:.0f} trades/day below floor"
                continue

            vol = ind.realized_vol(closes, len(closes))
            if vol > p.max_daily_vol:
                rejected[sym] = f"daily vol {vol:.1%} -- too fast to make markets in"
                continue

            # Tightness proxy: a name that trades a lot of dollars in a lot of
            # small prints inside a narrow daily range has a tight book.
            avg_range = sum(b.range / b.close for b in bars if b.close > 0) / len(bars)
            dollars_per_trade = adv / max(trades, 1.0)
            spread_proxy = avg_range / max(trades / 1000.0, 1.0)
            if spread_proxy > p.max_relative_spread * 50:
                rejected[sym] = f"tightness proxy {spread_proxy:.5f} too wide"
                continue

            score = (
                0.35 * min(trades / 100_000.0, 2.0)
                + 0.30 * min(adv / 5e8, 2.0)
                + 0.20 * (1.0 / max(spread_proxy * 1000.0, 0.05))
                + (0.15 * (1.0 - min(vol / p.max_daily_vol, 1.0))
                   if p.prefer_low_daily_vol else 0.0)
            )
            scored.append((
                sym, score,
                f"{trades / 1000:.0f}k trades/day ${adv / 1e6:.0f}M ADV "
                f"vol {vol:.1%} ${dollars_per_trade:.0f}/trade",
            ))

        result = self._rank(scored, ctx, max_per_sector=p.max_per_sector)
        result.rejected = rejected
        ctx.log.info(
            "liquidity_picker: %d/%d candidates liquid enough -> %s",
            len(scored), len(ctx.candidates), result.describe(),
        )
        return result
