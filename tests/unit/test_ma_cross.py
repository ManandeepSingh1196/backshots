"""ma_cross: hand-computed run (see T032) and Params validation."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from bt.core.errors import StrategyError
from bt.core.events import EventLog
from bt.core.types import Bar, Instrument, InstrumentKind, OrderStatus, Side
from bt.engine.loop import EngineConfig, run_engine
from bt.execution.broker_sim import BrokerSim
from bt.execution.fees import VenueFees
from bt.execution.slippage import FixedBps
from bt.portfolio.portfolio import Portfolio
from bt.strategy.registry import build_params, get_strategy
from strategies.ma_cross import MACross, MACrossParams
from tests.support.memory_feed import MemoryFeed

TOL = 1e-9
T0 = datetime(2024, 1, 1, tzinfo=UTC)
INST = Instrument(
    "AAAUSDT", "AAAUSDT", InstrumentKind.SPOT, "AAA", "USDT", 0.001, 0.001, 5.0, 0.01, 0.001, 0.001
)
CLOSES = [10, 10, 10, 12, 14, 13, 11, 9, 10, 13]


def d(i: int) -> datetime:
    return T0 + timedelta(days=i - 1)


def bars() -> list[Bar]:
    out: list[Bar] = []
    prev = 10.0
    for i, c in enumerate(CLOSES, start=1):
        out.append(Bar("AAAUSDT", d(i), prev, max(prev, c), min(prev, c), float(c), 1.0, 1.0))
        prev = float(c)
    return out


def run(fast: int = 2, slow: int = 3):
    log = EventLog()
    pf = Portfolio(1000.0, [INST])
    broker = BrokerSim([INST], VenueFees(), FixedBps(10), pf.view(), log)
    strat = MACross(build_params(MACross, {"fast": fast, "slow": slow}), [INST])
    res = run_engine(strat, MemoryFeed([INST], bars()), broker, pf, log, EngineConfig())
    return res


def test_hand_run() -> None:
    res = run()
    assert [s.equity for s in res.equity] == pytest.approx(
        [
            1000.0,
            1000.0,
            1000.0,
            1000.0,
            1163.677014004,
            1080.844014004,
            915.178014004,
            913.356599167,
            913.356599167,
            913.356599167,
        ],
        abs=TOL,
    )
    buy_fill, sell_fill = res.fills[0], res.fills[1]
    assert (buy_fill.ts, buy_fill.side) == (d(5), Side.BUY)
    assert buy_fill.qty == pytest.approx(82.833, abs=TOL)
    assert buy_fill.price == pytest.approx(12.012, abs=TOL)
    assert (sell_fill.ts, sell_fill.side) == (d(8), Side.SELL)
    assert sell_fill.qty == pytest.approx(82.833, abs=TOL)
    assert sell_fill.price == pytest.approx(10.989, abs=TOL)
    assert len(res.fills) == 2
    last = [r for r in res.orders if r.order.id == "ma-AAAUSDT-202401100000"]
    assert [(r.ts, r.status, r.reason) for r in last] == [
        (d(10), OrderStatus.NEW, ""),
        (d(10), OrderStatus.CANCELED, "end_of_data"),
    ]


def test_first_decision_at_warmup() -> None:
    res = run()
    ids = [r.order.id for r in res.orders if r.status == OrderStatus.NEW]
    assert ids == [
        "ma-AAAUSDT-202401040000",
        "ma-AAAUSDT-202401070000",
        "ma-AAAUSDT-202401100000",
    ]
    assert MACross(MACrossParams(fast=2, slow=3), [INST]).warmup_bars() == 3


@pytest.mark.parametrize("slow", [100, 7])
def test_warmup_bars(slow: int) -> None:
    s = MACross(build_params(MACross, {"fast": 2, "slow": slow}), [INST])
    assert s.warmup_bars() == slow


@pytest.mark.parametrize(
    "bad",
    [
        {"fast": 5, "slow": 5},
        {"fast": 10, "slow": 5},
        {"fast": 0},
        {"fast": True},
        {"fast": 1.5},
        {"alloc": 0},
        {"alloc": True},
        {"slow": True},
        {"bogus": 1},
    ],
)
def test_bad_params(bad: dict[str, object]) -> None:
    with pytest.raises(StrategyError):
        build_params(MACross, bad)


def test_params_defaults_and_frozen() -> None:
    p = build_params(MACross, {})
    assert isinstance(p, MACrossParams)
    assert (p.fast, p.slow, p.alloc) == (20, 100, 0.995)
    with pytest.raises(ValidationError):
        p.fast = 3  # pyright: ignore[reportAttributeAccessIssue]


def test_registry() -> None:
    assert get_strategy("ma_cross") is MACross
