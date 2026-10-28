import pytest

from bt.core.errors import ExecutionError
from bt.core.types import Instrument, InstrumentKind, Side
from bt.execution.fees import VenueFees


def _inst(symbol: str, min_qty: float, step: float) -> Instrument:
    return Instrument(
        symbol,
        symbol,
        InstrumentKind.SPOT,
        symbol[:3],
        "USDT",
        min_qty,
        step,
        5.0,
        0.01,
        0.001,
        0.001,
    )


AAA = _inst("AAAUSDT", 0.001, 0.001)
BBB = _inst("BBBUSDT", 0.1, 0.1)


def test_fee_default_rates() -> None:
    assert VenueFees().fee(AAA, Side.BUY, 5.0, 102.102, False) == pytest.approx(0.51051, abs=1e-12)
    assert VenueFees().fee(BBB, Side.BUY, 10.0, 19.8, True) == pytest.approx(0.198, abs=1e-12)


def test_fee_overrides() -> None:
    assert VenueFees(taker=0.00075).fee(AAA, Side.SELL, 5.0, 100.0, False) == pytest.approx(
        0.375, abs=1e-12
    )
    assert VenueFees(maker=-0.0001).fee(AAA, Side.BUY, 10.0, 100.0, True) == pytest.approx(
        -0.1, abs=1e-12
    )


def test_fee_zero_and_errors() -> None:
    f = VenueFees()
    assert f.fee(AAA, Side.BUY, 0.0, 100.0, False) == 0.0
    for qty, price in [(-1.0, 100.0), (1.0, 0.0), (float("nan"), 1.0), (1.0, float("inf"))]:
        with pytest.raises(ExecutionError):
            f.fee(AAA, Side.BUY, qty, price, False)


@pytest.mark.parametrize(
    "kwargs",
    [{"taker": -0.001}, {"maker": -0.01}, {"maker": float("nan")}, {"taker": float("inf")}],
)
def test_constructor_errors(kwargs: dict[str, float]) -> None:
    with pytest.raises(ExecutionError):
        VenueFees(**kwargs)
