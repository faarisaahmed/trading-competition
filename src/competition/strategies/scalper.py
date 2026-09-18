"""Team 8 -- The Scalper: one-sided market making on the microprice.

Thesis
------
Most of the money in equities is made by people who never take a directional
view. A market maker earns the spread by being patient on both sides and
managing inventory; the risk is adverse selection -- getting filled by
somebody who knows something. So the whole craft is (a) computing a fair
value better than the mid, (b) demanding a real edge before posting, and
(c) getting out of inventory fast.

Fair value (better than the mid, which is a lie when the book is lopsided)
-------------------------------------------------------------------------
    fair = microprice
         + imbalance_weight * spread * queue_imbalance
         + drift_weight * short_horizon_drift * price

The **microprice** size-weights the two sides: with 100 shares bid and 900
offered, the true clearing price is nearer the bid, and using the arithmetic
mid there means systematically buying too high. The **imbalance** term leans
into queue pressure, and the **drift** term (ROC over `drift_bars` of fast
bars) stops the strategy from posting a static bid into a name that is
trending down -- the single most expensive mistake a naive maker makes.

Quoting
-------
    bid = fair * (1 - edge_bps/10000 - inventory_skew)
    ask = fair * (1 + edge_bps/10000 - inventory_skew)

`edge_bps` is the minimum compensation demanded for providing liquidity.
`inventory_skew` shifts *both* quotes down as the book gets long, which makes
the ask more likely to fill and the bid less -- the standard Avellaneda-Stoikov
intuition, implemented as a linear skew because a three-week competition is
not the place for a stochastic-control solve.

Long-only adaptation
--------------------
A real maker quotes both sides continuously. On a cash account there is no
short leg, so this runs as a **one-sided maker**: it works passive bids to
accumulate inventory and passive offers to distribute it, never going below
flat. It therefore captures the spread only on the round trip, not on every
fill -- a genuine handicap, honestly stated.

Discipline
----------
* Spread must be inside [`min_relative_spread`, `max_relative_spread`]. Too
  tight and there is no edge to capture; too wide and something is wrong
  (news, halt, illiquidity) and the maker is the one being picked off.
* Orders are re-quoted only when fair value has moved more than `requote_bps`,
  and expire after `order_ttl_seconds`. Churning the book costs nothing in
  fees on Alpaca but every cancel/replace is a chance to cross yourself.
* Inventory is capped per name and in aggregate, and the book is flattened
  `flatten_minutes_before_close` minutes before the bell. A market maker who
  holds overnight is not a market maker; he is a punter with extra steps.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..types import Order, OrderIntent, Side, TimeInForce
from ..util import indicators as ind
from .base import Param, Strategy, StrategyContext


class Scalper(Strategy):
    DESCRIPTION = (
        "One-sided passive market making around a microprice fair value with queue-imbalance "
        "and drift adjustments, linear inventory skew, TTL'd quotes and a hard flatten "
        "into the close."
    )

    DEFAULT_TICK_SECONDS = 20

    PARAMS = (
        Param("tick_seconds", 20, "seconds between re-quotes", minimum=5),
        Param("edge_bps", 6.0, "minimum edge vs fair value, in basis points", minimum=0.0),
        Param("min_relative_spread", 0.00035, "skip names tighter than this", minimum=0.0),
        Param("max_relative_spread", 0.0060, "skip names wider than this", minimum=0.0),
        Param("inventory_cap_pct", 0.30, "inventory cap per name, as a weight",
              minimum=0.0, maximum=1.0),
        Param("total_inventory_cap_pct", 0.75, "aggregate inventory cap",
              minimum=0.0, maximum=1.0),
        Param("clip_notional", 240.0, "size of one working order, in dollars", minimum=1.0),
        Param("max_open_orders_per_symbol", 1, "working orders per symbol per side",
              minimum=1),
        Param("skew_per_inventory_unit", 0.65, "quote skew per unit of filled inventory cap",
              minimum=0.0),
        Param("imbalance_weight", 0.35, "how far to lean into queue imbalance", minimum=0.0),
        Param("drift_bars", 5, "fast bars used for the drift estimate", minimum=1),
        Param("drift_weight", 0.45, "how much short-horizon drift shifts fair value",
              minimum=0.0),
        Param("requote_bps", 2.5, "fair-value move needed to re-quote", minimum=0.0),
        Param("order_ttl_seconds", 90, "cancel a resting order after this long", minimum=1),
        Param("take_profit_bps", 12.0, "distribute inventory at this markup", minimum=0.1),
        Param("stop_bps", 45.0, "bail out of inventory at this loss", minimum=0.1),
        Param("max_positions", 4, "names quoted at once", minimum=1),
        Param("flatten_minutes_before_close", 15, "flatten inventory this early",
              minimum=0),
        Param("min_bars_required", 20, "warm-up bars before quoting", minimum=1),
    )

    # ------------------------------------------------------------------ #
    # fair value
    # ------------------------------------------------------------------ #

    def drift(self, ctx: StrategyContext, symbol: str) -> float:
        """Short-horizon drift from fast bars, falling back to the primary series."""
        p = self.p
        for kind in ("fast", "primary"):
            closes = ctx.closes(symbol, kind, n=p.drift_bars + 2)
            if len(closes) >= p.drift_bars + 1:
                return ind.roc(closes, p.drift_bars)
        return 0.0

    def fair_value(self, ctx: StrategyContext, symbol: str) -> dict[str, float] | None:
        p = self.p
        q = ctx.quote(symbol)
        if q is None or not q.is_sane:
            return None
        rel = q.relative_spread
        if not (p.min_relative_spread <= rel <= p.max_relative_spread):
            return None
        micro = q.microprice
        if micro <= 0:
            return None
        drift = self.drift(ctx, symbol)
        fair = (
            micro
            + p.imbalance_weight * q.spread * q.imbalance
            + p.drift_weight * drift * micro
        )
        if fair <= 0:
            return None
        return {
            "fair": fair, "mid": q.mid, "micro": micro, "spread": q.spread,
            "rel_spread": rel, "imbalance": q.imbalance, "drift": drift,
            "bid": q.bid, "ask": q.ask,
        }

    def inventory_skew(self, ctx: StrategyContext, symbol: str) -> float:
        """Fractional price shift applied to both quotes, from inventory."""
        cap = self.p.inventory_cap_pct
        if cap <= 0:
            return 0.0
        filled = min(max(ctx.weight(symbol) / cap, 0.0), 1.5)
        # Skew is expressed in units of the edge, so it scales with the spread
        # the strategy is trying to capture rather than with the price level.
        return filled * self.p.skew_per_inventory_unit * (self.p.edge_bps / 10_000.0)

    # ------------------------------------------------------------------ #
    # quoting
    # ------------------------------------------------------------------ #

    def _needs_requote(self, existing: Sequence[Order], desired_price: float) -> bool:
        """True if there is no live order at (near enough) the desired price."""
        if not existing:
            return True
        threshold = desired_price * self.p.requote_bps / 10_000.0
        return not any(
            o.limit_price and abs(o.limit_price - desired_price) <= threshold
            for o in existing
        )

    def on_round_start(self, ctx: StrategyContext) -> None:
        ctx.state.setdefault("_symbols", {})
        ctx.log.info(
            "scalper: %.1f bps edge, %.0f-dollar clips, inventory cap %.0f%%/name "
            "(%.0f%% total), flat %d min before the close",
            self.p.edge_bps, self.p.clip_notional, self.p.inventory_cap_pct * 100,
            self.p.total_inventory_cap_pct * 100, self.p.flatten_minutes_before_close,
        )

    def on_tick(self, ctx: StrategyContext) -> list[OrderIntent]:
        p = self.p
        self.sync_bar_clock(ctx)

        # ---------------- go flat into the close -------------------------- #
        if ctx.session.within_minutes_of_close(p.flatten_minutes_before_close):
            ctx.request_cancel_all()
            flat = ctx.flatten_all(reason="flatten into the close")
            if flat:
                ctx.log.debug("scalper: flattening %d name(s) into the close", len(flat))
            return flat

        if not ctx.session.is_open:
            ctx.request_cancel_all()
            return []

        # ---------------- housekeeping ------------------------------------ #
        ctx.expire_orders_older_than(p.order_ttl_seconds)

        intents: list[OrderIntent] = []
        quoted = 0
        total_inventory = ctx.gross_weight

        # Quote the tightest, most liquid names first.
        scored: list[tuple[float, str, dict[str, float]]] = []
        for sym in ctx.universe:
            if not ctx.tradable(sym):
                continue
            fv = self.fair_value(ctx, sym)
            if fv is None:
                continue
            scored.append((fv["rel_spread"], sym, fv))
        scored.sort(key=lambda t: t[0])

        for _rel, sym, fv in scored:
            st = ctx.sym_state(sym)
            fair = fv["fair"]
            skew = self.inventory_skew(ctx, sym)
            inventory_w = ctx.weight(sym)

            # ---- distribute inventory we already hold --------------------
            if inventory_w > 1e-6:
                avg = (ctx.position(sym).avg_entry_price if ctx.position(sym) else 0.0)
                px = ctx.price(sym)
                if avg > 0:
                    move_bps = (px / avg - 1.0) * 10_000.0
                    if move_bps <= -p.stop_bps:
                        ctx.request_cancel_symbol(sym)
                        o = ctx.close(sym, reason=f"inventory stop ({move_bps:.0f}bps)",
                                      tag="stop")
                        if o:
                            intents.append(o)
                        continue
                ask = fair * (1.0 + p.edge_bps / 10_000.0 - skew)
                if avg > 0:
                    # Never offer below the level that books a profit.
                    ask = max(ask, avg * (1.0 + p.take_profit_bps / 10_000.0))
                existing_asks = ctx.orders_for(sym, Side.SELL)
                if self._needs_requote(existing_asks, ask):
                    for o in existing_asks:
                        ctx.request_cancel(o.id)
                    # Subtract what is already resting on the offer: without
                    # this the maker repeatedly asks to sell more than it
                    # holds and collects oversell rejections.
                    already_offered = sum(
                        o.leaves_qty for o in ctx.orders_for(sym, Side.SELL)
                        if o.id not in ctx.cancel_requests
                    )
                    qty = min(
                        p.clip_notional / max(ask, 0.01),
                        max(ctx.qty(sym) - already_offered, 0.0),
                    )
                    if qty <= 0:
                        continue
                    intent = ctx.limit(
                        sym, Side.SELL, qty, ask,
                        reason=(f"offer {ask:.2f} vs fair {fair:.2f} "
                                f"(skew {skew * 10_000:.1f}bps, inv {inventory_w:.0%})"),
                        tag="mm_ask", tif=TimeInForce.DAY, replace_open=False,
                    )
                    if intent:
                        intents.append(intent)

            # ---- accumulate inventory ------------------------------------
            room_per_name = inventory_w < p.inventory_cap_pct - 1e-6
            room_total = total_inventory < p.total_inventory_cap_pct - 1e-6
            room_slots = quoted < p.max_positions
            if not (room_per_name and room_total and room_slots):
                continue
            quoted += 1

            bid = fair * (1.0 - p.edge_bps / 10_000.0 - skew)
            # Never bid through the offer, and never pay more than the fair
            # value we just computed -- that would be crossing for no edge.
            bid = min(bid, fv["ask"] * (1 - 1e-6), fair)
            existing_bids = ctx.orders_for(sym, Side.BUY)
            if not self._needs_requote(existing_bids, bid):
                continue
            for o in existing_bids:
                ctx.request_cancel(o.id)
            headroom = min(
                p.clip_notional,
                (p.inventory_cap_pct - inventory_w) * ctx.equity,
                (p.total_inventory_cap_pct - total_inventory) * ctx.equity,
                ctx.buying_power,
            )
            if headroom < ctx.min_notional():
                continue
            intent = ctx.limit(
                sym, Side.BUY, headroom / max(bid, 0.01), bid,
                reason=(f"bid {bid:.2f} vs fair {fair:.2f} "
                        f"(spread {fv['rel_spread'] * 10_000:.1f}bps, "
                        f"imb {fv['imbalance']:+.2f}, drift {fv['drift']:+.3%})"),
                tag="mm_bid", tif=TimeInForce.DAY,
            )
            if intent:
                intents.append(intent)
                st["last_bid"] = bid
                total_inventory += intent.qty * bid / max(ctx.equity, 1.0)

        return intents

    def on_session_end(self, ctx: StrategyContext) -> None:
        ctx.request_cancel_all()

    def on_round_end(self, ctx: StrategyContext) -> None:
        ctx.log.info("scalper: %d fills over the round", len(self._fills))
