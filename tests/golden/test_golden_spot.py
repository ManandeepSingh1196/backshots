"""M2 golden spot run: every expected number is hand-computed (see ticket T020)."""

import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from bt.core.events import EventLog
from bt.core.types import Fill, Order, OrderStatus, OrderType, Side
from bt.data.catalog import Catalog
from bt.data.feed import ParquetFeed
from bt.engine.loop import EngineConfig, EngineResult, run_engine
from bt.execution.broker_sim import BrokerSim
from bt.execution.fees import VenueFees
from bt.execution.slippage import FixedBps
from bt.portfolio.portfolio import Portfolio
from tests.support.make_golden_fixture import (
    AAA,
    AAA_BARS,
    BBB,
    BBB_BARS,
    FIXTURE_DIR,
    day,
    frame,
)
from tests.support.scripted import ScriptedStrategy

A, B = "AAAUSDT", "BBBUSDT"
CLOSE_ID = "bt-close-AAAUSDT-202401050000"


def _script() -> dict[datetime, list[Order]]:
    return {
        day(1): [
            Order("a1", A, Side.BUY, 5.0),
            Order("b1", B, Side.BUY, 10.0, OrderType.LIMIT, limit_price=19.80),
        ],
        day(2): [Order("a2", A, Side.SELL, 2.0), Order("b2", B, Side.SELL, 10.0)],
        day(3): [Order("a3", A, Side.BUY, 0.040), Order("b3", B, Side.SELL, 10.0)],
        day(4): [Order("a4", A, Side.BUY, 1.0, OrderType.LIMIT, limit_price=112.50)],
    }


def run_golden() -> EngineResult:
    cat = Catalog(FIXTURE_DIR)
    insts, _ = cat.read_instruments()
    feed = ParquetFeed(cat, insts, "1d", day(1), day(6), warmup_bars=0)
    log = EventLog()
    pf = Portfolio(1000.0, insts)
    broker = BrokerSim(insts, VenueFees(), FixedBps(10), pf.view(), log, ttl_bars=3)
    strat = ScriptedStrategy(ScriptedStrategy.Params(), insts, _script())
    return run_engine(strat, feed, broker, pf, log, EngineConfig())


@pytest.fixture(scope="module")
def result() -> EngineResult:
    return run_golden()


def test_fixture_matches_literal_tables() -> None:
    cat = Catalog(FIXTURE_DIR)
    assert cat.read_klines("spot", "1d", A).equals(frame(AAA_BARS))
    assert cat.read_klines("spot", "1d", B).equals(frame(BBB_BARS))
    insts, fetched = cat.read_instruments()
    assert insts == [AAA, BBB]
    assert fetched == datetime(2024, 2, 1, tzinfo=day(1).tzinfo)
    for inst, qty in ((insts[0], 0.001), (insts[1], 0.1)):
        assert (inst.min_qty, inst.qty_step) == (qty, qty)
        assert (inst.min_notional, inst.price_tick) == (5.0, 0.01)
        assert (inst.maker_fee, inst.taker_fee) == (0.001, 0.001)
    first = cat.read_klines("spot", "1d", A)
    assert first["quote_volume"].to_list() == [10.0 * c for c in first["close"].to_list()]
    assert isinstance(first, pl.DataFrame)


@dataclass(frozen=True)
class F:
    oid: str
    sym: str
    d: int
    side: Side
    qty: float
    price: float
    fee: float
    slip: float
    maker: bool


EXPECTED_FILLS = [
    F("a1", A, 2, Side.BUY, 5.0, 102.102, 0.51051, 0.51, False),
    F("b1", B, 2, Side.BUY, 10.0, 19.80, 0.198, 0.0, True),
    F("a2", A, 3, Side.SELL, 2.0, 107.892, 0.215784, 0.216, False),
    F("b2", B, 4, Side.SELL, 10.0, 21.978, 0.21978, 0.22, False),
    F("a4", A, 5, Side.BUY, 1.0, 112.5, 0.1125, 0.0, True),
    F(CLOSE_ID, A, 5, Side.SELL, 4.0, 113.886, 0.455544, 0.456, False),
]


def test_fills(result: EngineResult) -> None:
    assert len(result.fills) == len(EXPECTED_FILLS)
    for got, want in zip(result.fills, EXPECTED_FILLS, strict=True):
        assert isinstance(got, Fill)
        assert got.order_id == want.oid
        assert got.symbol == want.sym
        assert got.ts == day(want.d)
        assert got.side == want.side
        assert got.is_maker is want.maker
        assert got.qty == pytest.approx(want.qty, abs=1e-9)
        assert got.price == pytest.approx(want.price, abs=1e-9)
        assert got.fee == pytest.approx(want.fee, abs=1e-9)
        assert got.slippage_cost == pytest.approx(want.slip, abs=1e-9)


# d: (cash, equity, gross, fees_paid, slippage_paid, realized_pnl)
EXPECTED_EQUITY = {
    1: (1000.0, 1000.0, 0.0, 0.0, 0.0, 0.0),
    2: (290.78149, 1040.78149, 750.0, 0.70851, 0.51, 0.0),
    3: (506.349706, 1046.349706, 540.0, 0.924294, 0.726, 11.58),
    4: (725.909926, 1061.909926, 336.0, 1.144074, 0.946, 33.36),
    5: (1068.385882, 1068.385882, 0.0, 1.712118, 1.402, 70.098),
}


def test_equity(result: EngineResult) -> None:
    assert [e.ts for e in result.equity] == [day(i) for i in range(1, 6)]
    for row, (i, want) in zip(result.equity, EXPECTED_EQUITY.items(), strict=True):
        got = (
            row.cash,
            row.equity,
            row.gross_exposure,
            row.fees_paid,
            row.slippage_paid,
            row.realized_pnl,
        )
        assert got == pytest.approx(want, abs=1e-9), f"day {i}"
        assert row.net_exposure == pytest.approx(row.gross_exposure, abs=1e-9)


def test_order_records(result: EngineResult) -> None:
    new, fl, rj = OrderStatus.NEW, OrderStatus.FILLED, OrderStatus.REJECTED
    expected = [
        (1, "a1", new, ""),
        (1, "b1", new, ""),
        (2, "a1", fl, ""),
        (2, "b1", fl, ""),
        (2, "a2", new, ""),
        (2, "b2", new, ""),
        (3, "a2", fl, ""),
        (3, "a3", rj, "below_min_notional"),
        (3, "b3", rj, "insufficient_holdings"),
        (4, "b2", fl, ""),
        (4, "a4", new, ""),
        (5, "a4", fl, ""),
        (5, CLOSE_ID, new, ""),
        (5, CLOSE_ID, fl, ""),
    ]
    got = [(r.ts, r.order.id, r.status, r.reason) for r in result.orders]
    assert got == [(day(i), oid, st, why) for i, oid, st, why in expected]


def test_identity_and_terminal_state(result: EngineResult) -> None:
    assert pytest.approx(1068.385882, abs=1e-9) == 1000 + 70.098 - 1.712118
    assert result.equity[-1].equity == pytest.approx(1000 + 70.098 - 1.712118, abs=1e-9)
    assert result.final_positions == ()
    assert result.killed_at is None
    assert result.liquidated is False


def test_determinism() -> None:
    a, b = run_golden(), run_golden()
    assert a == b
    assert pickle.dumps(a) == pickle.dumps(b)


def test_fixture_dir_exists() -> None:
    assert Path(FIXTURE_DIR / "manifest.json").is_file()
