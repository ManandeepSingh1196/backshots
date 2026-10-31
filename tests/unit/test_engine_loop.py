"""Tests for bt.engine.loop. All expected numbers are hand-computed."""

import pickle
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

import bt.engine.loop as bt_loop
from bt.core.errors import EngineError, LookAheadError, StrategyError
from bt.core.events import EventLog, OrderRecord, SkippedOrder
from bt.core.types import Bar, Instrument, InstrumentKind, Order, OrderStatus, OrderType, Side
from bt.engine.loop import EngineConfig, EngineResult, run_engine
from bt.execution.broker_sim import BrokerSim
from bt.execution.fees import VenueFees
from bt.execution.slippage import FixedBps, SlippageModel
from bt.portfolio.portfolio import Portfolio
from bt.runs.frames import orders_frame
from bt.strategy.base import Intent, NoAction, Orders, TargetWeights
from bt.strategy.context import Context
from tests.support.memory_feed import MemoryFeed
from tests.support.scripted import ScriptedStrategy

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def d(i: int) -> datetime:
    """Day i, 1-based: d(1) is 2024-01-01."""
    return T0 + timedelta(days=i - 1)


def inst(sym: str = "AAA", fee: float = 0.0, **kw: float) -> Instrument:
    base = dict(
        min_qty=0.001,
        qty_step=0.001,
        min_notional=5.0,
        price_tick=0.01,
        maker_fee=fee,
        taker_fee=fee,
    )
    base.update(kw)
    return Instrument(
        symbol=sym + "USDT",
        venue_symbol=sym + "USDT",
        kind=InstrumentKind.SPOT,
        base=sym,
        quote="USDT",
        **base,
    )


AAA = inst("AAA")
BBB = inst("BBB")
AAAf = inst("AAA", fee=0.001)
A, B = "AAAUSDT", "BBBUSDT"


def bar(sym: str, i: int, o: float, h: float, lo: float, c: float) -> Bar:
    return Bar(sym, d(i), o, h, lo, c, 1.0, 1.0)


def flat(sym: str, i: int, p: float) -> Bar:
    return bar(sym, i, p, p, p, p)


def oc(sym: str, i: int, o: float, c: float) -> Bar:
    return bar(sym, i, o, max(o, c), min(o, c), c)


def buy(oid: str, sym: str = A, qty: float = 1.0) -> Order:
    return Order(oid, sym, Side.BUY, qty)


def sell(oid: str, sym: str = A, qty: float = 1.0) -> Order:
    return Order(oid, sym, Side.SELL, qty)


class Hooked(ScriptedStrategy):
    """ScriptedStrategy with an optional on_bar override and an on_start counter."""

    def __init__(
        self,
        insts: Sequence[Instrument],
        script: Mapping[datetime, Sequence[Order]],
        warmup: int = 0,
        hook: Callable[[Context], Intent | None] | None = None,
    ) -> None:
        super().__init__(ScriptedStrategy.Params(), insts, script, warmup)
        self.hook = hook
        self.starts: list[datetime] = []

    def on_start(self, ctx: Context) -> None:
        self.starts.append(ctx.now)

    def on_bar(self, ctx: Context) -> Intent:
        if self.hook is not None:
            out = self.hook(ctx)
            if out is not None:
                return out
        return super().on_bar(ctx)


@dataclass
class Run:
    result: EngineResult
    strategy: Hooked
    nsteps: int


def check_invariants(r: Run) -> None:
    res = r.result
    assert len(res.equity) == r.nsteps
    by_id: dict[str, list[OrderRecord]] = {}
    for rec in res.orders:
        by_id.setdefault(rec.order.id, []).append(rec)
    for oid, recs in by_id.items():
        assert recs[-1].status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
        ), oid
    last_ts = res.equity[-1].ts
    for f in res.fills:
        new_ts = next(x.ts for x in by_id[f.order_id] if x.status == OrderStatus.NEW)
        tag = by_id[f.order_id][0].order.tag
        if tag == "close_at_end":
            assert f.ts == new_ts == last_ts
        else:
            assert f.ts > new_ts
    for row in res.equity:
        assert row.equity == pytest.approx(row.cash + row.net_exposure, abs=1e-9)


def run(
    bars: Sequence[Bar],
    script: Mapping[datetime, Sequence[Order]],
    *,
    insts: Sequence[Instrument] = (AAA,),
    cash: float = 1000.0,
    config: EngineConfig | None = None,
    slip: SlippageModel | None = None,
    warmup: int = 0,
    hook: Callable[[Context], Intent | None] | None = None,
    check: bool = True,
) -> Run:
    log = EventLog()
    pf = Portfolio(cash, insts)
    broker = BrokerSim(insts, VenueFees(), slip or FixedBps(0), pf.view(), log)
    feed = MemoryFeed(insts, bars)
    strat = Hooked(insts, script, warmup, hook)
    res = run_engine(strat, feed, broker, pf, log, config or EngineConfig())
    r = Run(res, strat, len({b.ts for b in bars}))
    if check:
        check_invariants(r)
    return r


def records(r: Run, oid: str) -> list[tuple[datetime, OrderStatus, str]]:
    return [(x.ts, x.status, x.reason) for x in r.result.orders if x.order.id == oid]


def reason_of(r: Run, oid: str) -> str:
    return records(r, oid)[0][2]


TIMING_BARS = [flat(A, 1, 100), bar(A, 2, 102, 108, 101, 108), flat(A, 3, 110)]


def test_timing() -> None:
    r = run(TIMING_BARS, {d(1): [buy("b1")]})
    (fill, close_fill) = r.result.fills
    assert fill.price == pytest.approx(102.0, abs=1e-12)
    assert fill.ts == d(2)
    assert [e.equity for e in r.result.equity] == pytest.approx([1000, 1006, 1008], abs=1e-9)
    assert records(r, "b1") == [(d(1), OrderStatus.NEW, ""), (d(2), OrderStatus.FILLED, "")]
    assert close_fill.order_id == "bt-close-AAAUSDT-202401030000"
    assert close_fill.price == pytest.approx(110.0, abs=1e-12)
    assert r.strategy.fills_seen == [fill]  # on_fill not called for the close-out


def test_timing_without_close_at_end() -> None:
    r = run(TIMING_BARS, {d(1): [buy("b1")]}, config=EngineConfig(close_at_end=False))
    assert r.result.equity[-1].equity == pytest.approx(1008, abs=1e-9)
    (pos,) = r.result.final_positions
    assert pos.symbol == A
    assert pos.qty == pytest.approx(1.0, abs=1e-12)


KILL_CLOSES = [100, 90, 60, 50, 70]
KILL_OPENS = [100, 100, 90, 55, 50]
KILL_BARS = [
    oc(A, i + 1, o, c) for i, (o, c) in enumerate(zip(KILL_OPENS, KILL_CLOSES, strict=True))
]
KILL_SCRIPT = {d(1): [buy("b1", qty=10)], d(3): [buy("b2")], d(4): [buy("b3")]}


def test_kill_at_exact_boundary() -> None:
    r = run(KILL_BARS, KILL_SCRIPT, config=EngineConfig(max_drawdown_kill=0.40))
    assert r.result.fills[0].price == pytest.approx(100.0, abs=1e-12)
    assert r.result.killed_at == d(3)
    kill_fill = r.result.fills[1]
    assert kill_fill.order_id == "bt-kill-AAAUSDT-202401030000"
    assert kill_fill.ts == d(4)
    assert kill_fill.price == pytest.approx(55.0, abs=1e-12)
    assert [e.equity for e in r.result.equity] == pytest.approx(
        [1000, 900, 600, 550, 550], abs=1e-9
    )
    assert r.strategy.calls == [d(1), d(2)]
    assert r.strategy.fills_seen == [r.result.fills[0]]  # no on_fill after the kill
    (msg,) = [m for m in r.result.messages if m.message == "kill_switch"]
    assert msg.ts == d(3)
    assert dict(msg.fields)["drawdown"] == pytest.approx(0.4, abs=1e-12)
    assert not r.result.liquidated


def test_kill_just_below_boundary_threshold() -> None:
    r = run(KILL_BARS, KILL_SCRIPT, config=EngineConfig(max_drawdown_kill=0.41))
    assert r.result.killed_at == d(4)
    assert r.strategy.calls == [d(1), d(2), d(3)]
    kill_fill = r.result.fills[1]
    assert kill_fill.ts == d(5)
    assert kill_fill.price == pytest.approx(50.0, abs=1e-12)
    assert r.result.equity[-1].equity == pytest.approx(500, abs=1e-9)


@pytest.mark.parametrize("k", [0.51, None])
def test_no_kill(k: float | None) -> None:
    r = run(KILL_BARS, KILL_SCRIPT, config=EngineConfig(max_drawdown_kill=k))
    assert r.result.killed_at is None
    assert r.result.equity[-1].equity == pytest.approx(700, abs=1e-9)
    assert reason_of(r, "b2") == "insufficient_cash"
    assert reason_of(r, "b3") == "insufficient_cash"
    assert r.strategy.calls == [d(i) for i in range(1, 6)]


def test_missing_bar_order_fills_when_bar_returns() -> None:
    seen: list[np.datetime64] = []

    def hook(ctx: Context) -> Intent | None:
        if ctx.now == d(3):
            seen.append(ctx.history(B, 1).ts[-1])
        return None

    bars = [flat(A, i, 100) for i in range(1, 5)] + [flat(B, 1, 50), flat(B, 4, 60)]
    r = run(bars, {d(1): [buy("b1", B)]}, insts=(AAA, BBB), hook=hook)
    (fill,) = [f for f in r.result.fills if f.order_id == "b1"]
    assert fill.ts == d(4)
    assert fill.price == pytest.approx(60.0, abs=1e-12)
    assert d(2) in r.strategy.calls
    assert d(3) in r.strategy.calls
    assert seen == [np.datetime64("2024-01-01T00:00:00")]


def test_missing_bar_keeps_last_mark() -> None:
    bars = [flat(A, i, 100) for i in range(1, 5)]
    bars += [flat(B, 1, 50), oc(B, 2, 50, 60), flat(B, 4, 70)]
    r = run(bars, {d(1): [buy("b1", B)]}, insts=(AAA, BBB))
    # d3 equity: cash 950 + BBB 1 * d2 close 60.
    assert [e.equity for e in r.result.equity] == pytest.approx([1000, 1010, 1010, 1020], abs=1e-9)


def test_ttl_through_engine() -> None:
    bars = [flat(A, i, 100) for i in range(1, 6)] + [flat(B, 1, 50), flat(B, 5, 50)]
    r = run(bars, {d(1): [buy("b1", B)]}, insts=(AAA, BBB))
    assert records(r, "b1") == [
        (d(1), OrderStatus.NEW, ""),
        (d(4), OrderStatus.CANCELED, "ttl_expired"),
    ]
    assert r.result.fills == ()


def test_reservation_within_intent() -> None:
    bars = [flat(A, 1, 100), flat(A, 2, 100)]
    r = run(bars, {d(1): [buy("a", qty=6), buy("b", qty=6)]})
    assert records(r, "a")[0] == (d(1), OrderStatus.NEW, "")
    assert records(r, "b") == [(d(1), OrderStatus.REJECTED, "insufficient_cash")]


def test_reservation_of_holdings() -> None:
    bars = [flat(A, i, 100) for i in range(1, 4)]
    r = run(bars, {d(1): [buy("b0", qty=5)], d(2): [sell("s1", qty=3), sell("s2", qty=3)]})
    assert records(r, "s1")[0][1] == OrderStatus.NEW
    assert records(r, "s2") == [(d(2), OrderStatus.REJECTED, "insufficient_holdings")]


def test_reservation_across_steps() -> None:
    bars = [flat(A, i, 100) for i in range(1, 4)]
    lim = Order("l1", A, Side.BUY, 5.0, OrderType.LIMIT, limit_price=50.0)
    r = run(bars, {d(1): [lim], d(2): [buy("m1", qty=8)]})
    assert records(r, "m1") == [(d(2), OrderStatus.REJECTED, "insufficient_cash")]


def test_id_and_symbol_checks() -> None:
    bars = [flat(A, i, 100) for i in range(1, 4)] + [flat(B, 2, 50), flat(B, 3, 50)]
    script = {
        d(1): [buy("x"), buy("x"), buy("n1", B), buy("z1", "ZZZUSDT")],
        d(2): [buy("x"), buy("bt-x")],
    }
    r = run(bars, script, insts=(AAA, BBB))
    rejected = [
        (x.ts, x.order.id, x.reason) for x in r.result.orders if x.status == OrderStatus.REJECTED
    ]
    assert rejected == [
        (d(1), "x", "duplicate_id"),
        (d(1), "n1", "no_reference_price"),
        (d(1), "z1", "unknown_symbol"),
        (d(2), "x", "duplicate_id"),
        (d(2), "bt-x", "duplicate_id"),
    ]
    assert (d(1), OrderStatus.NEW, "") in records(r, "x")


def test_warmup_and_on_start() -> None:
    bars = [flat(A, i, 100) for i in range(1, 6)]
    r = run(bars, {}, warmup=3)
    assert r.strategy.calls == [d(3), d(4), d(5)]
    assert r.strategy.starts == [d(1)]


def test_intent_errors() -> None:
    bars = [flat(A, 1, 100), flat(A, 2, 100)]

    def boom(ctx: Context) -> Intent | None:
        raise RuntimeError("boom")

    with pytest.raises(StrategyError, match="boom") as ei:
        run(bars, {}, hook=boom, check=False)
    assert isinstance(ei.value.__cause__, RuntimeError)

    class NoneStrategy(Hooked):
        def on_bar(self, ctx: Context) -> Intent:
            return None  # type: ignore[return-value]  # deliberately invalid

    log = EventLog()
    pf = Portfolio(1000.0, (AAA,))
    broker = BrokerSim((AAA,), VenueFees(), FixedBps(0), pf.view(), log)
    with pytest.raises(StrategyError):
        run_engine(
            NoneStrategy((AAA,), {}), MemoryFeed((AAA,), bars), broker, pf, log, EngineConfig()
        )

    def peek(ctx: Context) -> Intent | None:
        raise LookAheadError("peeked")

    with pytest.raises(LookAheadError):
        run(bars, {}, hook=peek, check=False)


def test_wiring_errors() -> None:
    bars = [flat(A, 1, 100)]
    log = EventLog()
    pf = Portfolio(1000.0, (AAA,))
    broker = BrokerSim((AAA,), VenueFees(), FixedBps(0), pf.view(), log)
    cfg = EngineConfig()

    def go(strat_insts: Sequence[Instrument], feed_insts: Sequence[Instrument]) -> None:
        run_engine(Hooked(strat_insts, {}), MemoryFeed(feed_insts, bars), broker, pf, log, cfg)

    with pytest.raises(EngineError):
        go((AAA,), (AAA, BBB))
    with pytest.raises(EngineError):
        go((AAA,), (replace(AAA, min_notional=10.0),))
    perp = replace(AAA, kind=InstrumentKind.PERP)
    with pytest.raises(EngineError):
        go((perp,), (perp,))
    with pytest.raises(EngineError):
        run_engine(Hooked((AAA,), {}), MemoryFeed((AAA,), []), broker, pf, log, cfg)
    for bad in (0.0, 1.5, float("nan")):
        with pytest.raises(EngineError):
            EngineConfig(max_drawdown_kill=bad)
    with pytest.raises(EngineError):
        EngineConfig(rebalance_band=-0.1)
    with pytest.raises(EngineError):
        EngineConfig(global_max_leverage=0.5)
    with pytest.raises(EngineError):
        EngineConfig(global_max_leverage=float("inf"))


def test_end_of_data_cancels_open_orders() -> None:
    bars = [flat(A, 1, 100), flat(A, 2, 100)]
    lim = Order("l1", A, Side.BUY, 1.0, OrderType.LIMIT, limit_price=50.0)
    r = run(bars, {d(2): [lim]})
    assert records(r, "l1") == [
        (d(2), OrderStatus.NEW, ""),
        (d(2), OrderStatus.CANCELED, "end_of_data"),
    ]


def test_costs_flow() -> None:
    r = run(TIMING_BARS, {d(1): [buy("b1")]}, insts=(AAAf,), slip=FixedBps(10))
    f1, f2 = r.result.fills
    assert f1.price == pytest.approx(102.102, abs=1e-9)
    assert f1.fee == pytest.approx(0.102102, abs=1e-9)
    assert f2.price == pytest.approx(109.89, abs=1e-9)
    assert f2.fee == pytest.approx(0.10989, abs=1e-9)
    last = r.result.equity[-1]
    assert last.fees_paid == pytest.approx(0.211992, abs=1e-9)
    assert last.slippage_paid == pytest.approx(0.212, abs=1e-9)
    assert last.equity == pytest.approx(1007.576008, abs=1e-9)


def test_dust_kill_rejected_then_closed_at_end() -> None:
    bars = [flat(A, 1, 125), oc(A, 2, 125, 20), flat(A, 3, 20)]
    r = run(bars, {d(1): [buy("b1", qty=0.04)]}, cash=10.0)
    assert r.result.killed_at == d(2)
    kill_id = "bt-kill-AAAUSDT-202401020000"
    assert records(r, kill_id) == [(d(2), OrderStatus.REJECTED, "below_min_notional")]
    close_id = "bt-close-AAAUSDT-202401030000"
    assert records(r, close_id)[-1][1] == OrderStatus.FILLED
    assert r.result.final_positions == ()
    assert r.result.equity[-1].equity == pytest.approx(5.8, abs=1e-9)


def test_tolerance_parity() -> None:
    from bt.execution import rules
    from bt.portfolio import portfolio

    assert portfolio._QTY_REL_TOL == rules.QTY_REL_TOL  # pyright: ignore[reportPrivateUsage]
    assert portfolio._CASH_ABS_TOL == rules.CASH_ABS_TOL  # pyright: ignore[reportPrivateUsage]


def test_determinism() -> None:
    a = run(KILL_BARS, KILL_SCRIPT)
    b = run(KILL_BARS, KILL_SCRIPT)
    assert a.result == b.result
    assert pickle.dumps(a.result) == pickle.dumps(b.result)


def test_noaction_default_and_single_step() -> None:
    r = run([flat(A, 1, 100)], {})
    assert r.result.fills == ()
    assert [e.equity for e in r.result.equity] == pytest.approx([1000], abs=1e-9)
    assert isinstance(NoAction(), NoAction)


def make_engine(
    strat: ScriptedStrategy, bars: Sequence[Bar], insts: Sequence[Instrument] = (AAA,)
) -> EngineResult:
    log = EventLog()
    pf = Portfolio(1000.0, insts)
    broker = BrokerSim(insts, VenueFees(), FixedBps(0), pf.view(), log)
    return run_engine(strat, MemoryFeed(insts, bars), broker, pf, log, EngineConfig())


class BadWarmup(Hooked):
    def __init__(self, value: object) -> None:
        super().__init__((AAA,), {})
        self.value = value

    def warmup_bars(self) -> int:
        return self.value  # type: ignore[return-value]  # deliberately invalid


class CountingWarmup(Hooked):
    def __init__(self) -> None:
        super().__init__((AAA,), {})
        self.n = 0

    def warmup_bars(self) -> int:
        self.n += 1
        if self.n > 1:
            raise AssertionError("warmup_bars called twice")
        return 0


class RaisingWarmup(Hooked):
    def warmup_bars(self) -> int:
        raise ValueError("no warmup")


@pytest.mark.parametrize("value", [-1, "3"])
def test_bad_warmup_bars(value: object) -> None:
    with pytest.raises(StrategyError, match="warmup_bars"):
        make_engine(BadWarmup(value), [flat(A, 1, 100)])


def test_warmup_bars_called_once_and_guarded() -> None:
    strat = CountingWarmup()
    make_engine(strat, [flat(A, i, 100) for i in range(1, 4)])
    assert strat.n == 1
    with pytest.raises(StrategyError, match="no warmup") as ei:
        make_engine(RaisingWarmup((AAA,), {}), [flat(A, 1, 100)])
    assert isinstance(ei.value.__cause__, ValueError)


class RaisesOnFill(Hooked):
    def on_fill(self, fill: object, ctx: Context) -> None:  # type: ignore[override]
        raise ValueError("fill boom")


class RaisesOnStart(Hooked):
    def on_start(self, ctx: Context) -> None:
        raise ValueError("start boom")


def test_hook_exceptions_become_strategy_errors() -> None:
    bars = [flat(A, i, 100) for i in range(1, 4)]
    with pytest.raises(StrategyError, match="fill boom") as ei:
        make_engine(RaisesOnFill((AAA,), {d(1): [buy("b1")]}), bars)
    assert isinstance(ei.value.__cause__, ValueError)
    with pytest.raises(StrategyError, match="start boom") as ei2:
        make_engine(RaisesOnStart((AAA,), {}), bars)
    assert isinstance(ei2.value.__cause__, ValueError)


def test_kill_cancels_strategy_limit_before_kill_sell() -> None:
    lim = Order("l1", A, Side.BUY, 1.0, OrderType.LIMIT, limit_price=50.0)
    script = {d(1): [buy("b1", qty=9)], d(2): [lim]}
    r = run(KILL_BARS, script, config=EngineConfig(max_drawdown_kill=0.30))
    assert r.result.killed_at == d(3)
    assert records(r, "l1") == [
        (d(2), OrderStatus.NEW, ""),
        (d(3), OrderStatus.CANCELED, "kill_switch"),
    ]
    kill_fill = next(f for f in r.result.fills if f.order_id.startswith("bt-kill"))
    assert kill_fill.ts == d(4)
    d3 = [(x.order.id, x.status, x.reason) for x in r.result.orders if x.ts == d(3)]
    assert d3 == [
        ("l1", OrderStatus.CANCELED, "kill_switch"),
        ("bt-kill-AAAUSDT-202401030000", OrderStatus.NEW, ""),
    ]
    pos = {(x.order.id, x.status): i for i, x in enumerate(r.result.orders)}
    assert pos["l1", OrderStatus.CANCELED] < pos["bt-kill-AAAUSDT-202401030000", OrderStatus.NEW]


def test_at_fill_cash_rejection_after_gap_up() -> None:
    bars = [flat(A, 1, 100), flat(A, 2, 110), flat(A, 3, 110)]
    r = run(bars, {d(1): [buy("b1", qty=9.9)]})
    assert records(r, "b1") == [
        (d(1), OrderStatus.NEW, ""),
        (d(2), OrderStatus.REJECTED, "insufficient_cash_at_fill"),
    ]
    assert r.result.fills == ()
    assert [e.cash for e in r.result.equity] == pytest.approx([1000, 1000, 1000], abs=1e-9)


def test_kill_threshold_rejects_bool() -> None:
    with pytest.raises(EngineError):
        EngineConfig(max_drawdown_kill=True)


def test_memory_feed_no_funding_on_spot() -> None:
    from bt.core.types import FundingEvent

    feed = MemoryFeed((AAA,), [flat(A, 1, 100), flat(A, 2, 100)], [FundingEvent(A, d(1), 0.0001)])
    assert [s.funding for s in feed.steps()] == [(), ()]
    assert feed.funding_history(A, d(5), 10) == ()


# ---- TargetWeights path (T038) ----

AAAr = inst("AAA", fee=0.001)
BBBr = inst("BBB", fee=0.001)
ROT_BARS = [flat(A, i, 100) for i in range(1, 5)] + [flat(B, i, 50) for i in range(1, 5)]
ROT_ID_A = "bt-rb-AAAUSDT-202401010000"
ROT_SELL = "bt-rb-AAAUSDT-202401020000"
ROT_BUY = "bt-rb-BBBUSDT-202401020000"


def weights_hook(
    targets: Mapping[datetime, Mapping[str, float]],
) -> Callable[[Context], Intent | None]:
    return lambda ctx: TargetWeights(dict(targets[ctx.now])) if ctx.now in targets else None


def rotation() -> Run:
    return run(
        ROT_BARS,
        {},
        insts=(AAAr, BBBr),
        slip=FixedBps(5),
        hook=weights_hook({d(1): {A: 1.0}, d(2): {B: 1.0}}),
    )


def test_rotation_sells_and_buys_in_one_bar() -> None:
    r = rotation()
    assert [f.order_id for f in r.result.fills[:1]] == [ROT_ID_A]
    assert r.result.fills[0].qty == pytest.approx(9.935, abs=1e-9)
    assert records(r, ROT_SELL)[0] == (d(2), OrderStatus.NEW, "")
    assert records(r, ROT_BUY)[0] == (d(2), OrderStatus.NEW, "")
    sell_o = next(x.order for x in r.result.orders if x.order.id == ROT_SELL)
    buy_o = next(x.order for x in r.result.orders if x.order.id == ROT_BUY)
    assert sell_o.qty == pytest.approx(9.935, abs=1e-9)
    assert buy_o.qty == pytest.approx(19.712, abs=1e-9)
    assert [f.order_id for f in r.result.fills[1:3]] == [ROT_SELL, ROT_BUY]
    assert r.result.fills[1].ts == r.result.fills[2].ts == d(3)
    assert r.result.fills[1].price == pytest.approx(99.95, abs=1e-9)
    assert r.result.fills[2].price == pytest.approx(50.025, abs=1e-9)
    closing = r.result.fills[3]
    assert closing.price == pytest.approx(49.975, abs=1e-9)
    want = [1000, 998.50925325, 995.5406072, 994.0627]
    assert [e.equity for e in r.result.equity] == pytest.approx(want, abs=1e-6)


def _no_credit(self: object, order: Order, buffer: float) -> None:
    return None


def test_rotation_credit_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    r = rotation()
    assert records(r, ROT_BUY)[0] == (d(2), OrderStatus.NEW, "")
    # at d2 cash is only ~5.009253; the buy leg is NEW only because of credit_sale
    assert r.result.equity[1].cash == pytest.approx(5.009253, abs=1e-6)
    monkeypatch.setattr(bt_loop._ReservingView, "credit_sale", _no_credit)  # pyright: ignore[reportPrivateUsage]
    r2 = rotation()
    assert records(r2, ROT_BUY) == [(d(2), OrderStatus.REJECTED, "insufficient_cash")]


def test_reserving_view_credit_sale() -> None:
    pf = Portfolio(1000.0, (AAAr,))
    view = bt_loop._ReservingView(pf.view(), {A: AAAr}, {A: 100.0})  # pyright: ignore[reportPrivateUsage]
    assert view.available_cash == pytest.approx(1000.0, abs=1e-9)
    view.credit_sale(Order("s", A, Side.SELL, 9.935), 0.0055)
    # 9.935 * 100 * (1 - 0.001) * (1 - 0.0055)
    assert view.available_cash == pytest.approx(1000.0 + 987.04771425, abs=1e-6)


def open_sell_run(limit: bool, patch_buy: bool, monkeypatch: pytest.MonkeyPatch) -> Run:
    """A has no bar on d3, so its d2 sell stays open when d3's target is decided."""
    bars = [flat(A, 1, 100), flat(A, 2, 100), flat(A, 4, 100)]
    bars += [flat(B, i, 50) for i in range(1, 5)]
    lim = Order("lim", A, Side.SELL, 9.935, OrderType.LIMIT, limit_price=1000.0)
    buy_b = Order("bt-rb-BBBUSDT-202401030000", B, Side.BUY, 19.712, tag="rebalance")
    if patch_buy:
        real = bt_loop.weights_to_orders

        def fake(
            weights: Mapping[str, float],
            ctx: Context,
            band: float,
            cap: float,
            *,
            open_orders: Sequence[Order] = (),
            cash_buffer: float = 0.0,
        ) -> tuple[list[Order], list[SkippedOrder]]:
            if ctx.now == d(3):
                return [buy_b], []
            return real(weights, ctx, band, cap, open_orders=open_orders, cash_buffer=cash_buffer)

        monkeypatch.setattr(bt_loop, "weights_to_orders", fake)
    targets: dict[datetime, Mapping[str, float]] = {d(1): {A: 1.0}, d(3): {B: 1.0}}
    if not limit:
        targets[d(2)] = {}
    base = weights_hook(targets)

    def hook(ctx: Context) -> Intent | None:
        if limit and ctx.now == d(2):
            return Orders([lim])
        return base(ctx)

    return run(bars, {}, insts=(AAAr, BBBr), slip=FixedBps(5), hook=hook)


def test_open_market_sell_is_credited(monkeypatch: pytest.MonkeyPatch) -> None:
    r = open_sell_run(False, False, monkeypatch)
    buy_id = "bt-rb-BBBUSDT-202401030000"
    assert records(r, "bt-rb-AAAUSDT-202401020000")[0] == (d(2), OrderStatus.NEW, "")
    assert records(r, buy_id)[0] == (d(3), OrderStatus.NEW, "")
    fill = next(f for f in r.result.fills if f.order_id == buy_id)
    assert fill.ts == d(4)
    assert fill.qty == pytest.approx(19.712, abs=1e-9)


def test_open_limit_sell_is_not_credited(monkeypatch: pytest.MonkeyPatch) -> None:
    r = open_sell_run(True, True, monkeypatch)
    assert records(r, "bt-rb-BBBUSDT-202401030000") == [
        (d(3), OrderStatus.REJECTED, "insufficient_cash")
    ]


def test_orders_path_gets_no_credit() -> None:
    script = {
        d(1): [buy("x1", A, 9.935)],
        d(2): [sell("x2", A, 9.935), buy("x3", B, 19.712)],
    }
    r = run(ROT_BARS, script, insts=(AAAr, BBBr), slip=FixedBps(5))
    assert records(r, "x2")[0] == (d(2), OrderStatus.NEW, "")
    assert records(r, "x3") == [(d(2), OrderStatus.REJECTED, "insufficient_cash")]


def test_orders_with_rebalance_prefix_rejected() -> None:
    r = run([flat(A, 1, 100), flat(A, 2, 100)], {d(1): [buy("bt-rb-AAAUSDT-202401010000")]})
    assert records(r, "bt-rb-AAAUSDT-202401010000") == [
        (d(1), OrderStatus.REJECTED, "duplicate_id")
    ]


def test_skipped_rows_logged() -> None:
    bars = [flat(A, 1, 100), flat(A, 2, 100)]
    r = run(
        bars,
        {},
        config=EngineConfig(rebalance_band=0.0),
        hook=weights_hook({d(1): {A: 0.004}}),
    )
    assert len(r.result.skipped) == 1
    sk = r.result.skipped[0]
    assert (sk.ts, sk.symbol, sk.reason) == (d(1), A, "below_min_notional")
    assert r.result.orders == ()
    row = orders_frame(r.result).row(0, named=True)
    assert row["status"] == "skipped"
    assert row["order_id"] is None
    assert row["reason"] == "below_min_notional"


@pytest.mark.parametrize("weights", [{"ZZZUSDT": 0.1}, {A: -0.1}])
def test_bad_weights_raise_strategy_error(weights: dict[str, float]) -> None:
    bars = [flat(A, 1, 100), flat(A, 2, 100)]
    with pytest.raises(StrategyError):
        run(bars, {}, hook=weights_hook({d(1): weights}), check=False)


def test_unknown_symbol_error_names_it() -> None:
    with pytest.raises(StrategyError, match="ZZZUSDT"):
        run([flat(A, 1, 100)], {}, hook=weights_hook({d(1): {"ZZZUSDT": 0.1}}), check=False)


def test_symbol_without_bar_yet_is_strategy_error() -> None:
    bars = [flat(A, 1, 100), flat(A, 2, 100), flat(B, 2, 50)]
    with pytest.raises(StrategyError, match="BBBUSDT"):
        run(
            bars,
            {},
            insts=(AAA, BBB),
            hook=weights_hook({d(1): {B: 0.5}}),
            check=False,
        )


def test_kill_cancels_pending_rebalance_order() -> None:
    bars = [flat(A, 1, 100), oc(A, 2, 100, 40), flat(A, 3, 40), flat(A, 4, 40)]
    bars += [flat(B, 1, 50), flat(B, 3, 50), flat(B, 4, 50)]  # no B bar on d2
    r = run(
        bars,
        {},
        insts=(AAA, BBB),
        config=EngineConfig(max_drawdown_kill=0.25),
        hook=weights_hook({d(1): {A: 0.5, B: 0.4}}),
    )
    assert r.result.killed_at == d(2)
    rb = "bt-rb-BBBUSDT-202401010000"
    assert records(r, rb) == [
        (d(1), OrderStatus.NEW, ""),
        (d(2), OrderStatus.CANCELED, "kill_switch"),
    ]
    d2 = [(x.order.id, x.status) for x in r.result.orders if x.ts == d(2)]
    d2 = [x for x in d2 if x[1] != OrderStatus.FILLED]
    assert d2[0] == (rb, OrderStatus.CANCELED)
    assert d2[1] == ("bt-kill-AAAUSDT-202401020000", OrderStatus.NEW)


def test_on_fill_fires_for_rebalance_fills() -> None:
    r = rotation()
    ids = [f.order_id for f in r.strategy.fills_seen]
    assert ids[:3] == [ROT_ID_A, ROT_SELL, ROT_BUY]


@pytest.mark.parametrize("bad", [-0.1, 1.0, float("nan"), float("inf"), True, None, "x"])
def test_buy_buffer_validation(bad: object) -> None:
    with pytest.raises(EngineError):
        EngineConfig(buy_buffer=bad)  # type: ignore[arg-type]  # deliberately invalid


def test_buy_buffer_default() -> None:
    assert EngineConfig().buy_buffer == pytest.approx(0.0055, abs=1e-12)
