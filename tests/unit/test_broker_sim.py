from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

import pytest

from bt.core.errors import ExecutionError
from bt.core.events import EventLog, OrderRecord
from bt.core.types import (
    Bar,
    Fill,
    Instrument,
    InstrumentKind,
    Order,
    OrderAck,
    OrderStatus,
    OrderType,
    PositionView,
    Side,
    Step,
)
from bt.execution.broker import BacktestBroker, Broker
from bt.execution.broker_sim import BrokerSim
from bt.execution.fees import VenueFees
from bt.execution.slippage import FixedBps

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _t(n: int) -> datetime:
    return T0 + timedelta(days=n)


def _inst(symbol: str) -> Instrument:
    return Instrument(
        symbol, symbol, InstrumentKind.SPOT, "X", "USDT", 0.001, 0.001, 5.0, 0.01, 0.001, 0.001
    )


AAA = _inst("AAAUSDT")
BBB = _inst("BBBUSDT")


class FakeView:
    def __init__(self, cash: float = 10_000.0, held: Mapping[str, float] | None = None) -> None:
        self.cash_ = cash
        self.held = {"AAAUSDT": 10.0} if held is None else dict(held)

    @property
    def ts(self) -> datetime | None:
        return None

    @property
    def cash(self) -> float:
        return self.cash_

    @property
    def available_cash(self) -> float:
        return self.cash_

    @property
    def equity(self) -> float:
        return self.cash_

    def position(self, symbol: str) -> PositionView:
        return PositionView(symbol, self.held.get(symbol, 0.0), 0.0, 0.0, 0.0)

    def positions(self) -> Mapping[str, PositionView]:
        return {s: self.position(s) for s in sorted(self.held)}


class BadSlip:
    def __init__(self, value: float) -> None:
        self.value = value

    def slippage_frac(self, inst: Instrument, side: Side, qty: float, bar: Bar) -> float:
        return self.value


def bar(sym: str, n: int, o: float = 100, h: float = 105, low: float = 95, c: float = 102) -> Bar:
    return Bar(sym, _t(n), o, h, low, c, 10.0, 1000.0)


def step(n: int, *syms: str) -> Step:
    return Step(_t(n), {s: bar(s, n) for s in syms}, [])


def mk(
    cash: float = 10_000.0, held: Mapping[str, float] | None = None, ttl: int = 3, slip: float = 10
) -> tuple[BrokerSim, EventLog]:
    log = EventLog()
    b = BrokerSim([AAA, BBB], VenueFees(), FixedBps(slip), FakeView(cash, held), log, ttl)
    b.process(Step(T0, {}, []))
    return b, log


def order(
    oid: str,
    side: Side = Side.BUY,
    qty: float = 1.0,
    typ: OrderType = OrderType.MARKET,
    limit: float | None = None,
    stop: float | None = None,
    sym: str = "AAAUSDT",
) -> Order:
    return Order(oid, sym, side, qty, typ, limit, stop)


def fill_one(o: Order) -> list[Fill]:
    b, _ = mk()
    b.submit([o])
    return b.process(step(1, "AAAUSDT"))


def lim(side: Side, p: float) -> Order:
    return order("o", side, typ=OrderType.LIMIT, limit=p)


def stp(side: Side, p: float) -> Order:
    return order("o", side, typ=OrderType.STOP, stop=p)


def test_market_fills() -> None:
    (f,) = fill_one(order("o", Side.BUY))
    assert f.price == pytest.approx(100.1, abs=1e-9)
    assert f.fee == pytest.approx(0.1001, abs=1e-9)
    assert f.slippage_cost == pytest.approx(0.1, abs=1e-9)
    assert not f.is_maker
    assert f.ts == _t(1)
    (g,) = fill_one(order("o", Side.SELL))
    assert g.price == pytest.approx(99.9, abs=1e-9)
    assert g.fee == pytest.approx(0.0999, abs=1e-9)
    assert g.slippage_cost == pytest.approx(0.1, abs=1e-9)


def test_limit_buy() -> None:
    assert fill_one(lim(Side.BUY, 95.00)) == []
    (f,) = fill_one(lim(Side.BUY, 95.01))
    assert f.price == pytest.approx(95.01, abs=1e-9)
    assert f.is_maker
    assert f.fee == pytest.approx(0.09501, abs=1e-9)
    assert f.slippage_cost == pytest.approx(0.0, abs=1e-9)
    (g,) = fill_one(lim(Side.BUY, 101))
    assert g.price == pytest.approx(100.0, abs=1e-9)


def test_limit_sell() -> None:
    assert fill_one(lim(Side.SELL, 105.00)) == []
    (f,) = fill_one(lim(Side.SELL, 104.99))
    assert f.price == pytest.approx(104.99, abs=1e-9)
    assert f.is_maker
    (g,) = fill_one(lim(Side.SELL, 99))
    assert g.price == pytest.approx(100.0, abs=1e-9)


def test_stop_buy() -> None:
    (f,) = fill_one(stp(Side.BUY, 105))
    assert f.price == pytest.approx(105.105, abs=1e-9)
    assert f.fee == pytest.approx(0.105105, abs=1e-9)
    assert f.slippage_cost == pytest.approx(0.105, abs=1e-9)
    assert not f.is_maker
    assert fill_one(stp(Side.BUY, 105.01)) == []
    (g,) = fill_one(stp(Side.BUY, 98))
    assert g.price == pytest.approx(100.1, abs=1e-9)


def test_stop_sell() -> None:
    (f,) = fill_one(stp(Side.SELL, 95))
    assert f.price == pytest.approx(94.905, abs=1e-9)
    assert f.slippage_cost == pytest.approx(0.095, abs=1e-9)
    assert fill_one(stp(Side.SELL, 94.99)) == []
    (g,) = fill_one(stp(Side.SELL, 103))
    assert g.price == pytest.approx(99.9, abs=1e-9)


def _statuses(log: EventLog) -> list[tuple[str, datetime, OrderStatus, str]]:
    return [(r.order.id, r.ts, r.status, r.reason) for r in log.orders()]


def test_ttl_missing_bars() -> None:
    b, log = mk()
    b.submit([order("o")])
    for n in (1, 2, 3):
        assert b.process(step(n, "BBBUSDT")) == []
    assert _statuses(log)[-1] == ("o", _t(3), OrderStatus.CANCELED, "ttl_expired")
    assert b.open_orders() == []
    assert b.process(step(4, "AAAUSDT")) == []


def test_ttl_fill_at_last_chance() -> None:
    b, _ = mk()
    b.submit([order("o")])
    b.process(step(1, "BBBUSDT"))
    b.process(step(2, "BBBUSDT"))
    (f,) = b.process(step(3, "AAAUSDT"))
    assert f.price == pytest.approx(100.1, abs=1e-9)
    assert f.ts == _t(3)


def test_ttl_untriggered_limit() -> None:
    b, log = mk(ttl=1)
    b.submit([order("o", typ=OrderType.LIMIT, limit=50)])
    assert b.process(step(1, "AAAUSDT")) == []
    assert _statuses(log)[-1] == ("o", _t(1), OrderStatus.CANCELED, "ttl_expired")


def test_cash_at_fill() -> None:
    b, log = mk(cash=100.0)
    b.submit([order("o")])
    assert b.process(step(1, "AAAUSDT")) == []
    assert _statuses(log)[-1] == ("o", _t(1), OrderStatus.REJECTED, "insufficient_cash_at_fill")
    assert b.open_orders() == []


def test_sell_funds_buy() -> None:
    b, _ = mk(cash=100.0, held={"AAAUSDT": 1.0})
    b.submit([order("buy", Side.BUY), order("sell", Side.SELL)])
    fills = b.process(step(1, "AAAUSDT"))
    assert [f.order_id for f in fills] == ["sell", "buy"]


def test_holdings_at_fill() -> None:
    b, log = mk(held={"AAAUSDT": 1.0})
    b.submit([order("a", Side.SELL, 0.6), order("b", Side.SELL, 0.6)])
    fills = b.process(step(1, "AAAUSDT"))
    assert [f.order_id for f in fills] == ["a"]
    assert _statuses(log)[-1] == ("b", _t(1), OrderStatus.REJECTED, "insufficient_holdings_at_fill")


def test_submit_rules() -> None:
    log = EventLog()
    fresh = BrokerSim([AAA], VenueFees(), FixedBps(10), FakeView(), log)
    with pytest.raises(ExecutionError):
        fresh.submit([order("o")])
    b, _ = mk()
    acks = b.submit([order("x", sym="ZZZUSDT"), order("o"), order("o")])
    assert acks == [
        OrderAck("x", OrderStatus.REJECTED, "unknown_symbol"),
        OrderAck("o", OrderStatus.NEW),
        OrderAck("o", OrderStatus.REJECTED, "duplicate_id"),
    ]
    b.process(step(1, "AAAUSDT"))
    assert b.submit([order("o")]) == [OrderAck("o", OrderStatus.REJECTED, "duplicate_id")]
    assert b.submit([order("x")]) == [OrderAck("x", OrderStatus.REJECTED, "duplicate_id")]


def test_process_monotonic() -> None:
    b, _ = mk()
    with pytest.raises(ExecutionError):
        b.process(Step(T0, {}, []))
    b.process(step(2))
    with pytest.raises(ExecutionError):
        b.process(step(1))


def test_cancel() -> None:
    b, log = mk()
    b.submit([order("a"), order("b")])
    assert b.cancel("a") == OrderAck("a", OrderStatus.CANCELED, "user")
    assert [o.id for o in b.open_orders()] == ["b"]
    with pytest.raises(ExecutionError):
        b.cancel("nope")
    b.submit([order("c")])
    acks = b.cancel_all("end_of_data")
    assert acks == [
        OrderAck("b", OrderStatus.CANCELED, "end_of_data"),
        OrderAck("c", OrderStatus.CANCELED, "end_of_data"),
    ]
    assert _statuses(log)[-2:] == [
        ("b", T0, OrderStatus.CANCELED, "end_of_data"),
        ("c", T0, OrderStatus.CANCELED, "end_of_data"),
    ]
    assert b.cancel_all("end_of_data") == []


def test_settle_at_close() -> None:
    b, log = mk()
    b.process(step(1, "AAAUSDT"))
    (f,) = b.settle_at_close(_t(1), [order("s", Side.SELL)])
    assert f.price == pytest.approx(101.898, abs=1e-9)
    assert f.fee == pytest.approx(0.101898, abs=1e-9)
    assert f.slippage_cost == pytest.approx(0.102, abs=1e-9)
    assert f.ts == _t(1)
    assert [(r.status) for r in log.orders()] == [OrderStatus.NEW, OrderStatus.FILLED]
    assert b.open_orders() == []
    with pytest.raises(ExecutionError):
        b.settle_at_close(_t(1), [lim(Side.SELL, 100)])
    with pytest.raises(ExecutionError):
        b.settle_at_close(_t(2), [order("t", Side.SELL)])
    with pytest.raises(ExecutionError):
        b.settle_at_close(_t(1), [order("u", Side.SELL, sym="BBBUSDT")])
    with pytest.raises(ExecutionError):
        b.settle_at_close(_t(1), [order("s", Side.SELL)])


def test_settle_at_close_sells_across_symbols_fill_in_input_order() -> None:
    b, _ = mk(held={"AAAUSDT": 10.0, "BBBUSDT": 10.0})
    b.process(step(1, "AAAUSDT", "BBBUSDT"))
    fills = b.settle_at_close(
        _t(1), [order("sb", Side.SELL, sym="BBBUSDT"), order("sa", Side.SELL, sym="AAAUSDT")]
    )
    assert [f.order_id for f in fills] == ["sb", "sa"]


def test_settle_at_close_processes_sells_before_buys_regardless_of_input_order() -> None:
    # Cash 50 cannot fund a buy at ~102.1 unless the sell proceeds (~101.9) are applied first.
    b, log = mk(cash=50.0, held={"AAAUSDT": 10.0})
    b.process(step(1, "AAAUSDT", "BBBUSDT"))
    fills = b.settle_at_close(
        _t(1), [order("buy", Side.BUY, sym="BBBUSDT"), order("sell", Side.SELL, sym="AAAUSDT")]
    )
    assert [f.order_id for f in fills] == ["sell", "buy"]
    assert [(r.order.id, r.status) for r in log.orders()][:2] == [
        ("buy", OrderStatus.NEW),
        ("sell", OrderStatus.NEW),
    ]  # NEW records keep input order; only settlement is sells-first


@pytest.mark.parametrize("bad", [1.0, -0.1, float("nan")])
def test_bad_slippage(bad: float) -> None:
    b = BrokerSim([AAA], VenueFees(), BadSlip(bad), FakeView(), EventLog())
    b.process(Step(T0, {}, []))
    b.submit([order("o")])
    with pytest.raises(ExecutionError):
        b.process(step(1, "AAAUSDT"))


def test_log_sequence() -> None:
    b, log = mk()
    b.submit([order("A"), order("B", typ=OrderType.LIMIT, limit=50)])
    b.process(step(1, "AAAUSDT"))
    b.process(step(2, "AAAUSDT"))
    b.process(step(3, "AAAUSDT"))
    got: list[OrderRecord] = list(log.orders())
    assert [(r.order.id, r.ts, r.status, r.reason) for r in got] == [
        ("A", T0, OrderStatus.NEW, ""),
        ("B", T0, OrderStatus.NEW, ""),
        ("A", _t(1), OrderStatus.FILLED, ""),
        ("B", _t(3), OrderStatus.CANCELED, "ttl_expired"),
    ]
    assert log.fills() == ()


def test_constructor_errors() -> None:
    args = (VenueFees(), FixedBps(10), FakeView(), EventLog())
    with pytest.raises(ExecutionError):
        BrokerSim([], *args)
    with pytest.raises(ExecutionError):
        BrokerSim([AAA, AAA], *args)
    with pytest.raises(ExecutionError):
        BrokerSim([AAA], *args, ttl_bars=0)


def test_protocol_conformance() -> None:
    b: BacktestBroker = BrokerSim([AAA], VenueFees(), FixedBps(10), FakeView(), EventLog())
    b2: Broker = BrokerSim([AAA], VenueFees(), FixedBps(10), FakeView(), EventLog())
    assert b is not None
    assert b2 is not None
