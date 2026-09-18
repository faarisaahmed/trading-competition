"""Pre-trade risk checks -- the referee between a strategy and the broker.

Every order intent passes through here before it reaches a venue. The checks
are identical for all eight teams and driven entirely by `competition.risk`
in the rulebook, so no team can be advantaged or handicapped by its own
author's diligence. A rejected intent is *logged*, not silently dropped:
`comp report` prints each team's rejection tally, which is how you find out
that a strategy has been trying to do something illegal all week.

The checks, in order (cheapest and most decisive first):

  1. kill switch active for this team today
  2. malformed size (NaN, zero, negative)
  3. symbol outside the team's assigned universe        <- Round 3 enforcement
  4. asset not tradable / not in the snapshot
  5. no quote, or a quote too stale to trade on
  6. market closed
  7. per-tick and per-day order-count limits
  8. sell exceeding the held quantity (no shorting)
  9. limit price too far from the mid (fat finger)
 10. order notional outside [min, max]
 11. insufficient buying power
 12. resulting position weight over the cap
 13. resulting gross leverage over the cap
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime

from ..config import RiskConfig
from ..data.snapshot import MarketSnapshot
from ..types import (
    Account,
    Order,
    OrderIntent,
    OrderType,
    Rejection,
    RejectReason,
    Side,
)

log = logging.getLogger("competition.guardrails")


@dataclass
class RiskState:
    """Per-team mutable risk bookkeeping for one round."""

    team_key: str
    session_date: date | None = None
    session_open_equity: float = 0.0
    orders_today: int = 0
    halted_until: date | None = None
    halt_reason: str = ""
    rejections: dict[str, int] = field(default_factory=dict)
    peak_equity: float = 0.0

    @property
    def is_halted(self) -> bool:
        return self.halted_until is not None

    def roll_session(self, today: date, equity: float) -> bool:
        """Start a new session if the date changed. Returns True if it rolled."""
        if self.session_date == today:
            return False
        self.session_date = today
        self.session_open_equity = equity
        self.orders_today = 0
        self.peak_equity = max(self.peak_equity, equity)
        if self.halted_until is not None and today > self.halted_until:
            # The kill switch is a one-session timeout, not a disqualification.
            self.halted_until = None
            self.halt_reason = ""
        return True

    def session_drawdown(self, equity: float) -> float:
        if self.session_open_equity <= 0:
            return 0.0
        return equity / self.session_open_equity - 1.0

    def record_rejection(self, reason: RejectReason) -> None:
        self.rejections[reason.value] = self.rejections.get(reason.value, 0) + 1

    def summary(self) -> dict:
        return {
            "team": self.team_key,
            "orders_today": self.orders_today,
            "halted": self.is_halted,
            "halt_reason": self.halt_reason,
            "rejections": dict(sorted(self.rejections.items(), key=lambda kv: -kv[1])),
        }


@dataclass
class ValidationResult:
    accepted: list[OrderIntent] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.accepted)

    @property
    def counts(self) -> tuple[int, int]:
        return len(self.accepted), len(self.rejected)


class Guardrails:
    """Stateless validator driven by the rulebook's risk block."""

    def __init__(
        self,
        risk: RiskConfig,
        *,
        bankroll: float,
        is_tradable: Callable[[str], bool] | None = None,
        is_fractionable: Callable[[str], bool] | None = None,
        allow_closed_market: bool = False,
    ):
        self.risk = risk
        #: The round's per-team bankroll. Order-size limits scale with it, so
        #: the same rulebook works whether an account was created with $5,000
        #: or Alpaca's default $100,000.
        self.bankroll = float(bankroll)
        self._is_tradable = is_tradable or (lambda _s: True)
        self._is_fractionable = is_fractionable or (lambda _s: True)
        self.allow_closed_market = allow_closed_market

    # ------------------------------------------------------------------ #

    def check_kill_switch(
        self, state: RiskState, account: Account, *, today: date
    ) -> str | None:
        """Trip the circuit breaker if the team is down too much today.

        This protects the *competition*, not the team: a strategy with a
        sizing bug can otherwise turn a round into noise for everyone by
        making the leaderboard meaningless. It applies identically to all
        eight entries and lasts one session.
        """
        if state.is_halted:
            return state.halt_reason or "halted"
        dd = state.session_drawdown(account.equity)
        limit = -abs(self.risk.daily_loss_kill_switch_pct)
        if dd <= limit:
            state.halted_until = today
            state.halt_reason = (
                f"session drawdown {dd:.1%} breached the {limit:.0%} kill switch"
            )
            log.warning("%s: KILL SWITCH -- %s", state.team_key, state.halt_reason)
            return state.halt_reason
        return None

    # ------------------------------------------------------------------ #

    def validate(
        self,
        intents: Sequence[OrderIntent],
        *,
        state: RiskState,
        account: Account,
        snapshot: MarketSnapshot,
        universe: Iterable[str],
        open_orders: Sequence[Order] = (),
        now: datetime | None = None,
    ) -> ValidationResult:
        """Filter `intents` down to what may legally be sent."""
        result = ValidationResult()
        allowed = {s.upper() for s in universe}
        now = now or snapshot.ts
        equity = max(account.equity, 1e-9)

        # Running projections, so a batch of intents is checked as a batch and
        # not each one against the pre-batch book.
        cash_left = min(account.buying_power, account.cash)
        for o in open_orders:
            if o.side is Side.BUY:
                cash_left -= (o.limit_price or snapshot.price(o.symbol)) * o.leaves_qty
        cash_left = max(cash_left, 0.0)
        gross = account.gross_exposure
        held = {p.symbol: p.qty for p in account.positions}
        exposure = {p.symbol: abs(p.market_value) for p in account.positions}
        # Shares already committed to resting sells, so two exits in one tick
        # cannot oversell the position.
        committed_sells: dict[str, float] = {}
        for o in open_orders:
            if o.side is Side.SELL:
                committed_sells[o.symbol] = committed_sells.get(o.symbol, 0.0) + o.leaves_qty
        seen: set[tuple[str, str, float]] = set()

        def reject(intent: OrderIntent, reason: RejectReason, detail: str = "") -> None:
            state.record_rejection(reason)
            result.rejected.append(Rejection(intent, reason, detail))
            log.debug("%s rejected %s: %s %s", state.team_key, intent.describe(),
                      reason.value, detail)

        if state.is_halted:
            for intent in intents:
                reject(intent, RejectReason.KILL_SWITCH, state.halt_reason)
            return result

        per_tick = 0
        for intent in intents:
            sym = intent.symbol

            # -- 2. malformed size ---------------------------------------- #
            size = intent.qty if intent.qty is not None else intent.notional
            if size is None or not math.isfinite(size) or size <= 0:
                reject(intent, RejectReason.NAN_SIZE, f"size={size!r}")
                continue

            # -- 3. universe ---------------------------------------------- #
            if sym not in allowed:
                reject(intent, RejectReason.OUTSIDE_UNIVERSE,
                       f"{sym} not in this team's {len(allowed)}-name universe")
                continue

            # -- 4. tradability ------------------------------------------- #
            if not self._is_tradable(sym):
                reject(intent, RejectReason.ASSET_NOT_TRADABLE, sym)
                continue

            # -- 5. quote sanity ------------------------------------------ #
            quote = snapshot.quote(sym)
            price = snapshot.price(sym)
            if price <= 0:
                reject(intent, RejectReason.NO_QUOTE, sym)
                continue
            if quote is None:
                # Exits are allowed to proceed on the last close: refusing to
                # let a team *reduce* risk would be a worse failure than
                # filling at a slightly stale mark.
                if not intent.reduce_only:
                    reject(intent, RejectReason.NO_QUOTE, sym)
                    continue
            else:
                if quote.bid > 0 and quote.ask > 0 and quote.ask < quote.bid:
                    reject(intent, RejectReason.CROSSED_QUOTE,
                           f"bid {quote.bid} > ask {quote.ask}")
                    continue
                age = (now - quote.ts).total_seconds()
                if age > self.risk.max_quote_age_seconds and not intent.reduce_only:
                    reject(intent, RejectReason.STALE_QUOTE, f"{age:.0f}s old")
                    continue
            if sym in snapshot.missing and not intent.reduce_only:
                reject(intent, RejectReason.ASSET_HALTED, f"{sym} missing from the feed")
                continue

            # -- 6. market hours ------------------------------------------ #
            if not snapshot.session.is_open and not self.allow_closed_market:
                reject(intent, RejectReason.MARKET_CLOSED, str(snapshot.session.session_date))
                continue

            # -- 7. rate limits ------------------------------------------- #
            if per_tick >= self.risk.max_orders_per_tick:
                reject(intent, RejectReason.ORDER_RATE_LIMIT,
                       f"{self.risk.max_orders_per_tick}/tick reached")
                continue
            if state.orders_today + per_tick >= self.risk.max_orders_per_day:
                reject(intent, RejectReason.ORDER_RATE_LIMIT,
                       f"{self.risk.max_orders_per_day}/day reached")
                continue

            # -- 8. duplicate in the same batch --------------------------- #
            fingerprint = (sym, intent.side.value, round(intent.limit_price or 0.0, 4))
            if fingerprint in seen:
                reject(intent, RejectReason.DUPLICATE_INTENT, str(fingerprint))
                continue

            # -- size the order in shares and dollars --------------------- #
            ref_price = intent.limit_price or (quote.mid if quote else price) or price
            if intent.qty is not None:
                qty, notional = intent.qty, intent.qty * ref_price
            else:
                notional = intent.notional
                qty = notional / ref_price if ref_price > 0 else 0.0
            if qty <= 0 or not math.isfinite(qty):
                reject(intent, RejectReason.NAN_SIZE, f"qty={qty}")
                continue

            # -- 9. no shorting ------------------------------------------- #
            if intent.side is Side.SELL and not self.risk.allow_short:
                available = held.get(sym, 0.0) - committed_sells.get(sym, 0.0)
                if qty > available + 1e-6:
                    if available <= 1e-6:
                        reject(intent, RejectReason.OVERSELL,
                               f"holds {held.get(sym, 0.0):g}, "
                               f"{committed_sells.get(sym, 0.0):g} already offered")
                        continue
                    # Trim rather than reject: the strategy's intent to reduce
                    # is honoured, just not beyond flat.
                    qty = available
                    notional = qty * ref_price
                    intent = OrderIntent(
                        symbol=sym, side=intent.side, qty=round(qty, 6),
                        order_type=intent.order_type, limit_price=intent.limit_price,
                        tif=intent.tif, reason=intent.reason + " [trimmed to flat]",
                        tag=intent.tag, replace_open=intent.replace_open, reduce_only=True,
                    )

            # -- 10. limit sanity ----------------------------------------- #
            if intent.order_type is OrderType.LIMIT and intent.limit_price:
                mid = quote.mid if quote else price
                if mid > 0:
                    deviation = abs(intent.limit_price / mid - 1.0)
                    if deviation > self.risk.max_limit_deviation_pct:
                        reject(intent, RejectReason.LIMIT_TOO_FAR,
                               f"{deviation:.1%} from mid {mid:.2f}")
                        continue

            # -- 11. notional bounds -------------------------------------- #
            # Closing out an entire position is exempt from the floor. Partial
            # fills leave sub-dollar fractional remainders, and if the venue
            # minimum applied to the close too, that dust would be unsellable
            # forever -- the strategy would retry every tick and the tally
            # would fill with thousands of meaningless rejections. Brokers
            # (Alpaca included) accept a full-position close at any size.
            closing_out = (
                intent.side is Side.SELL
                and intent.reduce_only
                and qty >= held.get(sym, 0.0) - 1e-9
            )
            if notional < self.risk.min_order_notional and not closing_out:
                reject(intent, RejectReason.MIN_NOTIONAL, f"${notional:.2f}")
                continue
            order_ceiling = self.risk.max_order_notional(self.bankroll)
            if notional > order_ceiling + 1e-6:
                reject(intent, RejectReason.MAX_NOTIONAL,
                       f"${notional:.2f} > ${order_ceiling:.2f}")
                continue

            # -- 12/13. buying power, position and leverage caps ---------- #
            if intent.side is Side.BUY:
                if notional > cash_left + 1e-6:
                    reject(intent, RejectReason.INSUFFICIENT_CASH,
                           f"needs ${notional:.2f}, ${cash_left:.2f} uncommitted")
                    continue
                projected_pos = exposure.get(sym, 0.0) + notional
                if projected_pos / equity > self.risk.max_position_pct + 1e-6:
                    reject(intent, RejectReason.POSITION_CAP,
                           f"{projected_pos / equity:.1%} > "
                           f"{self.risk.max_position_pct:.0%}")
                    continue
                projected_gross = gross + notional
                if projected_gross / equity > self.risk.max_gross_leverage + 1e-6:
                    reject(intent, RejectReason.LEVERAGE_CAP,
                           f"{projected_gross / equity:.2f}x > "
                           f"{self.risk.max_gross_leverage:.2f}x")
                    continue
                cash_left -= notional
                gross = projected_gross
                exposure[sym] = projected_pos
            else:
                committed_sells[sym] = committed_sells.get(sym, 0.0) + qty
                gross = max(gross - notional, 0.0)
                exposure[sym] = max(exposure.get(sym, 0.0) - notional, 0.0)
                if intent.order_type is OrderType.MARKET:
                    # A market sell submitted earlier in this same tick frees
                    # its proceeds for a later buy. Credited with a haircut for
                    # slippage. Resting limit sells are NOT credited -- they
                    # may never fill, and spending cash you do not have yet is
                    # how a strategy ends up with a pile of broker rejections.
                    cash_left += notional * 0.995

            seen.add(fingerprint)
            per_tick += 1
            result.accepted.append(intent)

        state.orders_today += per_tick
        return result
