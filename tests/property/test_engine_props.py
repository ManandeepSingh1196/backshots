"""Hypothesis accounting invariants for run_engine on random spot scenarios (T020)."""

from dataclasses import dataclass, replace
from datetime import datetime

import numpy as np
import pytest
from hypothesis import event, given, settings
from hypothesis import strategies as st

from bt.core.events import EventLog
from bt.core.types import Bar, Fill, Order, OrderStatus, OrderType, Side
from bt.engine.loop import EngineConfig, EngineResult, run_engine
from bt.execution.broker_sim import BrokerSim
from bt.execution.fees import VenueFees
from bt.execution.slippage import FixedBps
from bt.portfolio.portfolio import Portfolio
from tests.support.make_golden_fixture import AAA, BBB, day
from tests.support.memory_feed import MemoryFeed
from tests.support.scripted import ScriptedStrategy

SYMS = ("AAAUSDT", "BBBUSDT")
STEP = {"AAAUSDT": 0.001, "BBBUSDT": 0.1}
START_PRICE = {"AAAUSDT": 100.0, "BBBUSDT": 20.0}
MAX_UNITS = {"AAAUSDT": 20000, "BBBUSDT": 200}  # qty <= 20 on the step grid
CASH = 1000.0
SETTINGS = settings(derandomize=True, deadline=None, max_examples=100)

order_st = st.tuples(
    st.integers(0, 29),  # step index (mod the number of steps)
    st.sampled_from(SYMS),
    st.sampled_from([Side.BUY, Side.SELL]),
    st.sampled_from([OrderType.MARKET] * 3 + [OrderType.LIMIT, OrderType.STOP]),
    st.one_of(st.floats(0.0, 0.2), st.floats(0.0, 0.2), st.floats(0.0, 0.2), st.floats(0.0, 1.0)),
    # qty fraction of the grid, biased small so more orders pass validation and fill
    st.floats(-0.2, 0.2),  # price offset from the previous close
)


@dataclass(frozen=True)
class Scenario:
    seed: int
    n_bars: int
    maker: float
    taker: float
    bps: float
    kill: float | None
    vol: float
    missing: tuple[tuple[str, int], ...]
    orders: tuple[tuple[int, str, Side, OrderType, float, float], ...]


@st.composite
def scenario_st(draw: st.DrawFn) -> Scenario:
    n = draw(st.integers(5, 30))
    inner = [(s, i) for s in SYMS for i in range(2, n)]  # days 2..n-1, never first or last
    missing = draw(st.lists(st.sampled_from(inner), max_size=8, unique=True))
    return Scenario(
        seed=draw(st.integers(0, 2**32 - 1)),
        n_bars=n,
        maker=draw(st.floats(0.0, 0.002)),
        taker=draw(st.floats(0.0, 0.002)),
        bps=draw(st.floats(0.0, 50.0)),
        kill=draw(st.sampled_from([None, 0.3])),
        vol=draw(st.sampled_from([0.02, 0.15])),
        missing=tuple(sorted(missing)),
        orders=tuple(draw(st.lists(order_st, max_size=40))),
    )


def make_bars(sc: Scenario) -> list[Bar]:
    rng = np.random.default_rng(sc.seed)
    bars: list[Bar] = []
    for sym in SYMS:
        prev = START_PRICE[sym]
        for i in range(1, sc.n_bars + 1):
            o = prev * float(np.exp(rng.normal(0.0, sc.vol / 4)))
            c = o * float(np.exp(rng.normal(0.0, sc.vol)))
            h = max(o, c) * (1 + float(rng.uniform(0.0, sc.vol / 2)))
            lo = min(o, c) * (1 - float(rng.uniform(0.0, sc.vol / 2)))
            prev = c
            if (sym, i) not in sc.missing:
                bars.append(Bar(sym, day(i), o, h, lo, c, 10.0, 10.0 * c))
    return bars


def make_script(sc: Scenario, bars: list[Bar]) -> dict[datetime, list[Order]]:
    steps = sorted({b.ts for b in bars})
    script: dict[datetime, list[Order]] = {}
    bought = {sym: 0.0 for sym in SYMS}  # cumulative qty bought earlier in the script
    ordered = sorted(enumerate(sc.orders), key=lambda e: e[1][0] % len(steps))
    for k, (idx, sym, side, typ, frac, off) in ordered:
        ts = steps[idx % len(steps)]
        closes = [b.close for b in bars if b.symbol == sym and b.ts <= ts]
        qty = round(max(1, round(frac * MAX_UNITS[sym])) * STEP[sym], 3)
        if side == Side.SELL and bought[sym] == 0:
            side = Side.BUY  # nothing held yet: buy instead of selling into a rejection
        if side == Side.SELL:
            qty = max(STEP[sym], min(qty, round(bought[sym], 3)))  # size sells from holdings
        else:
            bought[sym] += qty
        ref = closes[-1]
        px = max(0.01, round(ref * (1 + off), 2))
        if typ == OrderType.LIMIT:
            order = Order(f"o{k}", sym, side, qty, typ, limit_price=px)
        elif typ == OrderType.STOP:
            order = Order(f"o{k}", sym, side, qty, typ, stop_price=px)
        else:
            order = Order(f"o{k}", sym, side, qty)
        script.setdefault(ts, []).append(order)
    return script


def execute(sc: Scenario, bars: list[Bar], script: dict[datetime, list[Order]]) -> EngineResult:
    insts = [
        replace(AAA, maker_fee=sc.maker, taker_fee=sc.taker),
        replace(BBB, maker_fee=sc.maker, taker_fee=sc.taker),
    ]
    log = EventLog()
    pf = Portfolio(CASH, insts)
    broker = BrokerSim(insts, VenueFees(), FixedBps(sc.bps), pf.view(), log)
    strat = ScriptedStrategy(ScriptedStrategy.Params(), insts, script)
    cfg = EngineConfig(close_at_end=True, max_drawdown_kill=sc.kill)
    return run_engine(strat, MemoryFeed(insts, bars), broker, pf, log, cfg)


def tol(x: float) -> float:
    return 1e-9 * max(1.0, abs(x))


def notional(fills: tuple[Fill, ...]) -> float:
    return sum(f.qty * f.price for f in fills)


def label_run(res: EngineResult) -> None:
    n = len(res.fills)
    event("fills=0" if n == 0 else "fills=1-3" if n <= 3 else "fills>=4")
    if res.killed_at is not None:
        event("killed")
    if any(r.status == OrderStatus.REJECTED for r in res.orders):
        event("rejected>0")


@SETTINGS
@given(scenario_st())
def test_engine_invariants(sc: Scenario) -> None:
    bars = make_bars(sc)
    res = execute(sc, bars, make_script(sc, bars))  # (1) never raises
    steps = sorted({b.ts for b in bars})
    label_run(res)

    for row in res.equity:  # (2)
        assert row.equity == pytest.approx(row.cash + row.net_exposure, abs=tol(row.equity))
        assert row.cash >= -1e-6
        assert row.gross_exposure >= 0.0
    assert len(res.equity) == len(steps)

    last = res.equity[-1]
    big = notional(res.fills)
    assert sum(f.fee for f in res.fills) == pytest.approx(last.fees_paid, abs=tol(big))  # (3)
    assert sum(f.slippage_cost for f in res.fills) == pytest.approx(
        last.slippage_paid, abs=tol(big)
    )

    replay = CASH  # (4)
    for f in res.fills:
        gross = f.qty * f.price
        replay += -(gross + f.fee) if f.side == Side.BUY else gross - f.fee
    assert last.cash == pytest.approx(replay, abs=tol(big))

    first_new: dict[str, datetime] = {}  # (5)
    last_status: dict[str, OrderStatus] = {}
    for rec in res.orders:
        last_status.setdefault(rec.order.id, rec.status)
        if rec.status == OrderStatus.NEW:
            first_new.setdefault(rec.order.id, rec.ts)
        last_status[rec.order.id] = rec.status
    for f in res.fills:
        new_ts = first_new[f.order_id]
        if f.order_id.startswith("bt-close"):
            assert f.ts == steps[-1]
        else:
            lag = steps.index(f.ts) - steps.index(new_ts)
            assert 1 <= lag <= 3, (f.order_id, lag)

    terminal = {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED}  # (6)
    assert all(s in terminal for s in last_status.values())
    assert res.final_positions == ()
    assert last.equity == pytest.approx(last.cash, abs=tol(last.equity))


def test_generator_fills_often() -> None:
    seen = [0, 0]  # [examples, examples with at least one fill]

    @settings(derandomize=True, max_examples=100, database=None, deadline=None)
    @given(scenario_st())
    def probe(sc: Scenario) -> None:
        bars = make_bars(sc)
        res = execute(sc, bars, make_script(sc, bars))
        seen[0] += 1
        seen[1] += bool(res.fills)

    probe()
    assert seen[0] >= 50
    assert seen[1] / seen[0] >= 0.60, seen


@SETTINGS
@given(
    st.integers(0, 2**32 - 1),
    st.integers(5, 30),
    st.sampled_from([0.02, 0.15]),
    st.sampled_from([None, 0.3]),
)
def test_zero_costs_empty_script_is_flat(seed: int, n: int, vol: float, kill: float | None) -> None:
    sc = Scenario(seed, n, 0.0, 0.0, 0.0, kill, vol, (), ())
    res = execute(sc, make_bars(sc), {})
    assert [e.equity for e in res.equity] == [1000.0] * n
