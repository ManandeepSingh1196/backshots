import pickle
from collections.abc import Mapping
from datetime import UTC, datetime

import pytest

from bt.core.errors import (
    BtError,
    EngineError,
    ExecutionError,
    InvariantError,
    LookAheadError,
    PortfolioError,
    StrategyError,
)
from bt.core.types import OrderAck, OrderStatus, PortfolioView, PositionView


@pytest.mark.parametrize(
    "err", [LookAheadError, StrategyError, ExecutionError, PortfolioError, EngineError]
)
def test_new_errors_are_bt_errors_not_value_errors(err: type[Exception]) -> None:
    assert issubclass(err, BtError)
    assert not issubclass(err, ValueError)


def test_order_ack_coerces_status() -> None:
    ack = OrderAck("o1", "rejected", "below_min_qty")  # pyright: ignore[reportArgumentType]
    assert ack.status is OrderStatus.REJECTED


def test_order_ack_valid_and_invalid() -> None:
    assert OrderAck("o1", OrderStatus.NEW).reason == ""
    assert OrderAck("o1", OrderStatus.CANCELED, "user").reason == "user"
    for args in (
        ("o1", OrderStatus.NEW, "x"),
        ("o1", OrderStatus.FILLED, "x"),
        ("o1", OrderStatus.REJECTED, ""),
        ("o1", OrderStatus.CANCELED, ""),
        ("", OrderStatus.NEW, ""),
        ("o1", "bogus", ""),
    ):
        with pytest.raises(InvariantError):
            OrderAck(*args)  # pyright: ignore[reportArgumentType]


def test_order_ack_pickles() -> None:
    ack = OrderAck("o1", OrderStatus.REJECTED, "below_min_qty")
    assert pickle.loads(pickle.dumps(ack, protocol=5)) == ack


class _Fake:
    @property
    def ts(self) -> datetime | None:
        return None

    @property
    def cash(self) -> float:
        return 0.0

    @property
    def available_cash(self) -> float:
        return 0.0

    @property
    def equity(self) -> float:
        return 0.0

    def position(self, symbol: str) -> PositionView:
        return PositionView(symbol, 0.0, 0.0, 0.0, 0.0)

    def positions(self) -> Mapping[str, PositionView]:
        return {}


def test_portfolio_view_protocol_structural() -> None:
    view: PortfolioView = _Fake()
    assert view.position("BTCUSDT").qty == pytest.approx(0.0, abs=1e-12)
    assert view.ts is None
    assert datetime(2024, 1, 1, tzinfo=UTC)  # keep UTC import used
