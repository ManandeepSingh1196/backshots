from collections.abc import Mapping
from datetime import datetime
from typing import get_args

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bt.core.errors import ExecutionError
from bt.core.events import RejectReason
from bt.core.types import (
    Instrument,
    InstrumentKind,
    Order,
    OrderStatus,
    OrderType,
    PositionView,
    Side,
)
from bt.execution.rules import round_price, round_qty_down, validate_order


def _inst(
    symbol: str, min_qty: float, step: float, kind: InstrumentKind = InstrumentKind.SPOT
) -> Instrument:
    return Instrument(symbol, symbol, kind, "X", "USDT", min_qty, step, 5.0, 0.01, 0.001, 0.001)


AAA = _inst("AAAUSDT", 0.001, 0.001)
BBB = _inst("BBBUSDT", 0.1, 0.1)
PERP = _inst("AAAUSDT-PERP", 0.001, 0.001, InstrumentKind.PERP)


class FakeView:
    def __init__(self, cash: float = 1000.0, held: Mapping[str, float] | None = None) -> None:
        self._cash = cash
        self._held = {"AAAUSDT": 2.0} if held is None else dict(held)

    @property
    def ts(self) -> datetime | None:
        return None

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def available_cash(self) -> float:
        return self._cash

    @property
    def equity(self) -> float:
        return self._cash

    def position(self, symbol: str) -> PositionView:
        q = self._held.get(symbol, 0.0)
        return PositionView(symbol, q, 0.0, 0.0, 0.0)

    def positions(self) -> Mapping[str, PositionView]:
        return {s: self.position(s) for s in sorted(self._held)}


def _mk(
    side: Side,
    qty: float,
    *,
    symbol: str = "AAAUSDT",
    type: OrderType = OrderType.MARKET,
    limit: float | None = None,
    stop: float | None = None,
    reduce_only: bool = False,
) -> Order:
    return Order("o1", symbol, side, qty, type, limit, stop, reduce_only)


def _reason(
    order: Order,
    inst: Instrument = AAA,
    ref: float = 100.0,
    view: FakeView | None = None,
) -> str:
    ack = validate_order(order, inst, ref, view or FakeView())
    assert ack.order_id == order.id
    if ack.status is OrderStatus.NEW:
        assert ack.reason == ""
        return "new"
    assert ack.status is OrderStatus.REJECTED
    assert ack.reason in get_args(RejectReason)
    return ack.reason


@pytest.mark.parametrize(
    ("qty", "inst", "expected"),
    [
        (0.0029999, AAA, 0.002),
        (0.003, AAA, 0.003),
        (0.3, BBB, 0.3),
        (7.95, BBB, 7.9),
        (0.0, AAA, 0.0),
        (1e-12, AAA, 0.0),
    ],
)
def test_round_qty_down(qty: float, inst: Instrument, expected: float) -> None:
    assert round_qty_down(qty, inst) == expected


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_round_qty_down_errors(bad: float) -> None:
    with pytest.raises(ExecutionError):
        round_qty_down(bad, AAA)


def test_round_qty_down_too_large() -> None:
    with pytest.raises(ExecutionError):
        round_qty_down(1e30, AAA)


def test_round_price() -> None:
    assert round_price(100.004, AAA, Side.BUY) == 100.01
    assert round_price(100.004, AAA, Side.SELL) == 100.0
    assert round_price(100.01, AAA, Side.BUY) == 100.01
    assert round_price(100.01, AAA, Side.SELL) == 100.01
    assert round_price(19.8, BBB, Side.SELL) == 19.8
    assert round_price(19.8, BBB, Side.BUY) == 19.8


def test_round_price_errors() -> None:
    with pytest.raises(ExecutionError):
        round_price(0.004, AAA, Side.SELL)
    for bad in (-1.0, 0.0, float("nan")):
        with pytest.raises(ExecutionError):
            round_price(bad, AAA, Side.BUY)


def test_min_notional_boundary() -> None:
    assert _reason(_mk(Side.BUY, 0.05)) == "new"
    assert _reason(_mk(Side.BUY, 0.049)) == "below_min_notional"


def test_min_qty_boundary_and_order() -> None:
    assert _reason(_mk(Side.BUY, 0.001), ref=5000.0) == "new"
    assert _reason(_mk(Side.BUY, 0.0005)) == "below_min_qty"


def test_step() -> None:
    assert _reason(_mk(Side.BUY, 0.0015)) == "qty_off_step"
    assert _reason(_mk(Side.BUY, 0.3, symbol="BBBUSDT"), BBB, 20.0) == "new"


def test_sell_accumulated_float_qty_on_grid() -> None:
    q = 0.1 + 0.1 + 0.1
    assert q != 0.3
    view = FakeView(held={"BBBUSDT": 0.3})
    assert _reason(_mk(Side.SELL, q, symbol="BBBUSDT"), BBB, 20.0, view) == "new"


def test_price_tick_and_limit_notional() -> None:
    assert _reason(_mk(Side.BUY, 1.0, type=OrderType.LIMIT, limit=100.005)) == "price_off_tick"
    assert _reason(_mk(Side.BUY, 1.0, type=OrderType.LIMIT, limit=100.01)) == "new"
    assert _reason(_mk(Side.SELL, 1.0, type=OrderType.STOP, stop=99.999)) == "price_off_tick"
    assert _reason(_mk(Side.BUY, 1.0, type=OrderType.LIMIT, limit=4.99)) == "below_min_notional"


def test_cash() -> None:
    assert _reason(_mk(Side.BUY, 9.99)) == "new"
    assert _reason(_mk(Side.BUY, 9.991)) == "insufficient_cash"
    assert _reason(_mk(Side.BUY, 10.0), view=FakeView(cash=1001.0)) == "new"


def test_holdings() -> None:
    assert _reason(_mk(Side.SELL, 2.0)) == "new"
    assert _reason(_mk(Side.SELL, 2.001)) == "insufficient_holdings"
    assert _reason(_mk(Side.SELL, 0.5), view=FakeView(held={})) == "insufficient_holdings"


def test_reduce_only() -> None:
    assert _reason(_mk(Side.BUY, 0.05, reduce_only=True)) == "reduce_only_increases"
    assert _reason(_mk(Side.SELL, 1.0, reduce_only=True)) == "new"


def test_unsupported_kind() -> None:
    assert _reason(_mk(Side.BUY, 1.0, symbol="AAAUSDT-PERP"), PERP) == "unsupported_kind"


def test_programmer_errors() -> None:
    with pytest.raises(ExecutionError):
        validate_order(_mk(Side.BUY, 1.0, symbol="BBBUSDT"), AAA, 100.0, FakeView())
    for ref in (0.0, float("nan"), -1.0):
        with pytest.raises(ExecutionError):
            validate_order(_mk(Side.BUY, 1.0), AAA, ref, FakeView())


@given(q=st.floats(min_value=0.0, max_value=1e6), inst=st.sampled_from([AAA, BBB]))
def test_round_qty_down_properties(q: float, inst: Instrument) -> None:
    r = round_qty_down(q, inst)
    assert 0 <= q - r < inst.qty_step + 1e-9
    assert round_qty_down(r, inst) == r


def test_round_qty_down_numpy_scalar() -> None:
    import numpy as np

    r = round_qty_down(np.float64(1.2345), AAA)  # pyright: ignore[reportArgumentType]
    assert r == pytest.approx(1.234, abs=1e-12)
    assert type(r) is float


def test_round_qty_down_huge_chains_invariant_error() -> None:
    from bt.core.errors import InvariantError

    with pytest.raises(ExecutionError) as ei:
        round_qty_down(1e300, _inst("HUGE", 1e-8, 1e-8))
    assert isinstance(ei.value.__cause__, InvariantError)
