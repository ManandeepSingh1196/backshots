import pickle

import pytest
from pydantic import BaseModel

from bt.core.errors import StrategyError
from bt.core.types import Fill, Instrument, InstrumentKind, Order, Side
from bt.strategy.base import Intent, NoAction, Orders, Strategy, TargetWeights
from bt.strategy.context import Context


def _inst(symbol: str) -> Instrument:
    return Instrument(
        symbol, symbol, InstrumentKind.SPOT, "X", "USDT", 0.001, 0.001, 5.0, 0.01, 0.001, 0.001
    )


AAA = _inst("AAAUSDT")
BBB = _inst("BBBUSDT")


class Hold(Strategy):
    class Params(BaseModel):
        lookback: int = 20
        vol_target: float = 0.25

    def on_bar(self, ctx: Context) -> Intent:
        return NoAction()


class Other(BaseModel):
    pass


def test_instruments_sorted() -> None:
    s = Hold(Hold.Params(), [BBB, AAA])
    assert s.instruments == (AAA, BBB)
    assert isinstance(s.params, Hold.Params)


def test_bad_params_and_instruments() -> None:
    with pytest.raises(StrategyError):
        Hold(Other(), [AAA])
    with pytest.raises(StrategyError):
        Hold(Hold.Params(), [])
    with pytest.raises(StrategyError):
        Hold(Hold.Params(), [AAA, AAA])


def test_defaults_and_abstract() -> None:
    s = Hold(Hold.Params(), [AAA])
    assert s.warmup_bars() == 0
    fill = Fill(
        "o",
        "AAAUSDT",
        __import__("datetime").datetime.now(__import__("datetime").UTC),
        Side.BUY,
        1,
        1,
        0,
        0,
        False,
    )
    assert s.on_start(None) is None  # type: ignore[arg-type]
    assert s.on_fill(fill, None) is None  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Strategy(Strategy.Params(), [AAA])  # type: ignore[abstract]


def test_orders_copy_and_validation() -> None:
    o1 = Order("1", "AAAUSDT", Side.BUY, 1.0)
    src = [o1]
    o = Orders(src)
    src.append(o1)
    assert o.orders == [o1]
    assert Orders([]).orders == []
    with pytest.raises(StrategyError):
        Orders(["x"])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        hash(o)


def test_target_weights() -> None:
    src = {"AAAUSDT": 0.5}
    tw = TargetWeights(src)
    src["BBBUSDT"] = 1.0
    assert dict(tw.weights) == {"AAAUSDT": 0.5}
    with pytest.raises(TypeError):
        tw.weights["X"] = 1  # type: ignore[index]
    with pytest.raises(StrategyError):
        TargetWeights({"A": float("nan")})
    with pytest.raises(StrategyError):
        TargetWeights({"A": float("inf")})
    with pytest.raises(StrategyError):
        TargetWeights({"A": True})
    with pytest.raises(StrategyError):
        TargetWeights({"": 0.1})
    with pytest.raises(StrategyError):
        TargetWeights({"A": "1"})  # type: ignore[dict-item]
    assert dict(pickle.loads(pickle.dumps(tw)).weights) == {"AAAUSDT": 0.5}
    assert pickle.loads(pickle.dumps(tw)) == tw
    with pytest.raises(TypeError):
        hash(tw)


def test_no_action() -> None:
    assert NoAction() == NoAction()
