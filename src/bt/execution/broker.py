"""Broker protocols shared by the backtest simulator and the live broker."""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from bt.core.events import CancelReason
from bt.core.types import Fill, Order, OrderAck, Step


class Broker(Protocol):
    def submit(self, orders: Sequence[Order]) -> list[OrderAck]:
        """Queue orders for the next step."""
        ...

    def process(self, step: Step) -> list[Fill]:
        """Backtest: fill the queue against the step's bars. Live: poll the exchange."""
        ...

    def cancel(self, order_id: str) -> OrderAck: ...

    def open_orders(self) -> list[Order]: ...


class BacktestBroker(Broker, Protocol):
    """Backtest-only extras the engine needs."""

    def cancel_all(self, reason: CancelReason) -> list[OrderAck]: ...

    def settle_at_close(self, ts: datetime, orders: Sequence[Order]) -> list[Fill]:
        """Fill market orders now at each symbol's last seen close, worsened by slippage,
        with the taker fee."""
        ...
