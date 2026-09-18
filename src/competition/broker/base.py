"""The broker interface every execution backend implements.

Two backends ship: `AlpacaBroker` (real paper accounts) and `SimulatedBroker`
(deterministic offline fills). The engine only ever talks to this interface,
so a full three-round competition can be dry-run end to end with no keys and
then re-run live with a one-flag change.
"""

from __future__ import annotations

import abc
from datetime import datetime

from ..types import Account, Order, OrderIntent, Position


class BrokerError(RuntimeError):
    """Any broker-side failure."""


class OrderRejected(BrokerError):
    """The venue refused the order (bad symbol, halted, wash trade, ...)."""


class InsufficientFunds(OrderRejected):
    """Not enough buying power. Common and expected; handled, not crashed on."""


class Broker(abc.ABC):
    """Per-team execution handle. One instance == one portfolio."""

    #: Human label for logs / reports.
    name: str = "broker"

    # -- account state ----------------------------------------------------- #

    @abc.abstractmethod
    def account(self) -> Account:
        """Cash, equity, buying power and open positions, right now."""

    @abc.abstractmethod
    def positions(self) -> list[Position]:
        ...

    # -- order lifecycle --------------------------------------------------- #

    @abc.abstractmethod
    def submit(self, intent: OrderIntent, *, client_order_id: str | None = None) -> Order:
        """Send one order. Raises `OrderRejected` on venue refusal."""

    @abc.abstractmethod
    def open_orders(self, symbol: str | None = None) -> list[Order]:
        ...

    @abc.abstractmethod
    def cancel(self, order_id: str) -> None:
        ...

    @abc.abstractmethod
    def cancel_all(self, symbol: str | None = None) -> int:
        """Cancel every open order (optionally just one symbol). Returns count."""

    def get_order(self, order_id: str) -> Order | None:  # pragma: no cover - optional
        for o in self.open_orders():
            if o.id == order_id:
                return o
        return None

    # -- bulk actions ------------------------------------------------------ #

    @abc.abstractmethod
    def close_position(self, symbol: str) -> Order | None:
        """Market-exit one symbol entirely."""

    @abc.abstractmethod
    def close_all_positions(self, *, cancel_orders: bool = True) -> list[Order]:
        """Flatten the book. Used at round end and by the kill switch."""

    # -- lifecycle hooks --------------------------------------------------- #

    def flush(self) -> list[Order]:
        """Send anything buffered for this tick. No-op for direct brokers.

        Only the shared-account layer buffers: it has to see every team's
        intents for a tick before it can net opposing orders, because Alpaca
        rejects a buy and a sell on the same symbol in one account as a
        potential wash trade.
        """
        return []

    def sync(self, now: datetime | None = None) -> None:
        """Advance internal state (simulator fills, cache invalidation)."""

    def reset_for_round(self, starting_cash: float) -> None:
        """Return the account to a flat `starting_cash` state.

        Live brokers cannot mint cash: `AlpacaBroker` flattens the book,
        cancels everything and records the *baseline equity* it actually has,
        so round P&L is still measured from a common zero. Only the simulator
        can truly reset the balance.
        """

    @property
    def supports_fractional(self) -> bool:
        return True

    @property
    def supports_notional_orders(self) -> bool:
        return True
