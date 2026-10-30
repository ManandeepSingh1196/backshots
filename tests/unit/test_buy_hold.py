"""buy_hold: hand-computed engine runs, sizing edge cases and Params validation."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from bt.core.errors import StrategyError
from bt.core.events import EventLog
from bt.core.types import Bar, Instrument, InstrumentKind, OrderStatus, Side
from bt.engine.backtest import run_backtest
from bt.engine.loop import EngineConfig, EngineResult, run_engine
from bt.execution.broker_sim import BrokerSim
from bt.execution.fees import VenueFees
from bt.execution.slippage import FixedBps
from bt.portfolio.portfolio import Portfolio
from bt.runs.config import RunConfig
from bt.runs.reader import read_run
from bt.strategy.registry import build_params, get_strategy, list_strategies
from strategies.buy_hold import BuyHold, BuyHoldParams
from tests.support.make_golden_fixture import FIXTURE_DIR
from tests.support.memory_feed import MemoryFeed

TOL = 1e-9
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def d(i: int) -> datetime:
    return T0 + timedelta(days=i - 1)


def inst(sym: str, fee: float = 0.0, min_notional: float = 5.0) -> Instrument:
    return Instrument(
        sym + "USDT",
        sym + "USDT",
        InstrumentKind.SPOT,
        sym,
        "USDT",
        0.001,
        0.001,
        min_notional,
        0.01,
        fee,
        fee,
    )


def flat(sym: str, i: int, p: float = 100.0) -> Bar:
    return Bar(sym + "USDT", d(i), p, p, p, p, 1.0, 1.0)


def run_mem(insts: Sequence[Instrument], bars: Sequence[Bar], alloc: float = 0.995) -> EngineResult:
    log = EventLog()
    pf = Portfolio(1000.0, insts)
    broker = BrokerSim(insts, VenueFees(), FixedBps(0), pf.view(), log)
    strat = BuyHold(build_params(BuyHold, {"alloc": alloc}), insts)
    return run_engine(strat, MemoryFeed(insts, bars), broker, pf, log, EngineConfig())


def test_retry_after_gap(tmp_path: Path) -> None:
    cfg = RunConfig.model_validate(
        {
            "run": {
                "strategy": "buy_hold",
                "start": "2024-01-01",
                "end": "2024-01-05",
                "initial_equity_inr": 85000,
                "inr_per_usdt": 85,
            },
            "universe": {"symbols": ["AAAUSDT"], "kind": "spot"},
            "params": {},
            "costs": {"slippage": {"model": "fixed_bps", "bps": 10}},
        }
    )
    res = run_backtest(cfg, data_dir=FIXTURE_DIR, runs_dir=tmp_path)
    data = read_run(res.run_dir)
    assert data.equity["equity_after_tax"].to_list() == pytest.approx(
        [1000.0, 1000.0, 1016.417158076, 1034.823158076, 1051.131923218], abs=TOL
    )
    orders = data.orders
    rej = orders.filter(orders["status"] == "rejected")
    assert rej["reason"].to_list() == ["insufficient_cash_at_fill"]
    assert rej["qty"].to_list() == pytest.approx([9.94], abs=TOL)
    buys = orders.filter((orders["side"] == "buy") & (orders["status"] == "new"))
    assert buys["qty"].to_list() == pytest.approx([9.94, 9.203], abs=TOL)
    # one order only after the position fills
    assert buys["ts"].to_list() == [d(1), d(2)]
    assert data.trades["price"][0] == pytest.approx(108.108, abs=TOL)


def test_equal_split() -> None:
    insts = [inst("AAA"), inst("BBB")]
    bars = [flat(s, i) for s in ("AAA", "BBB") for i in (1, 2, 3)]
    res = run_mem(insts, bars, alloc=0.5)
    qty = {f.symbol: f.qty for f in res.fills if f.side == Side.BUY}
    assert qty == {"AAAUSDT": pytest.approx(2.5, abs=TOL), "BBBUSDT": pytest.approx(2.5, abs=TOL)}
    news = [r for r in res.orders if r.status == OrderStatus.NEW and r.order.side == Side.BUY]
    assert len(news) == 2


def test_too_small() -> None:
    insts = [inst("AAA", min_notional=5.0), inst("BBB", min_notional=5.0)]
    bars = [flat(s, i) for s in ("AAA", "BBB") for i in (1, 2, 3)]
    res = run_mem(insts, bars, alloc=0.001)
    assert res.orders == ()
    small = [m for m in res.messages if m.message == "buy_hold_too_small"]
    assert len(small) == 6
    assert sorted(str(dict(m.fields)["symbol"]) for m in small[:2]) == ["AAAUSDT", "BBBUSDT"]


def test_missing_bar_single_order() -> None:
    insts = [inst("AAA"), inst("BBB")]
    bars = [flat("AAA", i) for i in (1, 2, 3, 4)] + [flat("BBB", i) for i in (1, 3, 4)]
    res = run_mem(insts, bars)
    b_orders = [r for r in res.orders if r.order.symbol == "BBBUSDT" and r.order.side == Side.BUY]
    assert {r.order.id for r in b_orders} == {"bh-BBBUSDT-202401010000"}
    assert [(r.ts, r.status) for r in b_orders] == [
        (d(1), OrderStatus.NEW),
        (d(3), OrderStatus.FILLED),
    ]
    a_orders = [r for r in res.orders if r.order.symbol == "AAAUSDT" and r.order.side == Side.BUY]
    assert [(r.ts, r.status) for r in a_orders] == [
        (d(1), OrderStatus.NEW),
        (d(2), OrderStatus.FILLED),
    ]


def test_registry() -> None:
    assert get_strategy("buy_hold") is BuyHold
    assert "buy_hold" in [n for n, _ in list_strategies()]
    assert BuyHold(BuyHoldParams(), [inst("AAA")]).warmup_bars() == 0


@pytest.mark.parametrize("bad", [0, 0.0, 1.5, -0.1, float("nan"), True])
def test_bad_alloc(bad: object) -> None:
    with pytest.raises(StrategyError):
        build_params(BuyHold, {"alloc": bad})


def test_params_extra_and_frozen() -> None:
    with pytest.raises(StrategyError):
        build_params(BuyHold, {"alloc": 0.5, "bogus": 1})
    p = build_params(BuyHold, {"alloc": 0.5})
    assert isinstance(p, BuyHoldParams)
    with pytest.raises(ValidationError):
        p.alloc = 0.4  # pyright: ignore[reportAttributeAccessIssue]
    assert BuyHoldParams().alloc == 0.995
