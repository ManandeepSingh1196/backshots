from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import pytest

from bt.core.errors import EngineError, StrategyError
from bt.core.events import SkippedOrder
from bt.core.types import (
    FundingEvent,
    Instrument,
    InstrumentKind,
    Order,
    OrderType,
    PositionView,
    Side,
)
from bt.engine.rebalance import REBALANCE_TAG, rebalance_order_id, weights_to_orders
from bt.strategy.context import HistoryView

TS = datetime(2024, 1, 2, tzinfo=UTC)


def _inst(
    sym: str,
    step: float = 0.001,
    min_qty: float = 0.001,
    kind: InstrumentKind = InstrumentKind.SPOT,
) -> Instrument:
    return Instrument(sym, sym, kind, sym, "USDT", min_qty, step, 5.0, 0.01, 0.001, 0.001, 1.0, 0.0)


@dataclass
class _Hist:
    close: np.ndarray


@dataclass
class FakeCtx:
    equity: float = 1000.0
    cash: float = 1000.0
    held: Mapping[str, float] = field(default_factory=dict[str, float])
    refs: Mapping[str, float | None] = field(default_factory=lambda: {"A": 100.0, "B": 50.0})
    insts: Mapping[str, Instrument] = field(
        default_factory=lambda: {"A": _inst("A"), "B": _inst("B")}
    )
    now: datetime = TS

    def history(self, symbol: str, n: int) -> HistoryView:
        r = self.refs.get(symbol)
        arr = np.array([] if r is None else [r], dtype=np.float64)
        return _Hist(arr)  # pyright: ignore[reportReturnType]

    def funding_history(self, symbol: str, n: int) -> Sequence[FundingEvent]:
        return ()

    def position(self, symbol: str) -> PositionView:
        return PositionView(symbol, self.held.get(symbol, 0.0), 0.0, 0.0, 0.0)

    def instrument(self, symbol: str) -> Instrument:
        return self.insts[symbol]

    def symbols(self) -> Sequence[str]:
        return tuple(self.insts)

    def log(self, msg: str, **fields: object) -> None:
        pass


BUF = 0.006


def run(
    weights: Mapping[str, float],
    ctx: FakeCtx | None = None,
    band: float = 0.02,
    cap: float = 2.0,
    open_orders: Sequence[Order] = (),
    buf: float = BUF,
) -> tuple[list[Order], list[SkippedOrder]]:
    return weights_to_orders(
        weights, ctx or FakeCtx(), band, cap, open_orders=open_orders, cash_buffer=buf
    )


def simple(orders: list[Order]) -> list[tuple[str, str, float]]:
    return [(o.symbol, o.side.value, o.qty) for o in orders]


def check(orders: list[Order], expected: list[tuple[str, Side, float]]) -> None:
    assert [(o.symbol, o.side, o.qty) for o in orders] == [
        (s, sd, pytest.approx(q, abs=1e-9)) for s, sd, q in expected
    ]


def mkt(side: Side, sym: str, qty: float, oid: str = "x1") -> Order:
    return Order(oid, sym, side, qty)


def test_plain() -> None:
    orders, skipped = run({"A": 0.5})
    check(orders, [("A", Side.BUY, 5.0)])
    assert skipped == []
    assert orders[0].type == OrderType.MARKET
    assert orders[0].tag == REBALANCE_TAG


def test_near_one_scaled() -> None:
    orders, _ = run({"A": 1.0})
    check(orders, [("A", Side.BUY, 9.93)])
    k = 1000 / (1000 * 1.001 * 1.006)
    assert k == pytest.approx(0.9930427425, abs=1e-9)


def test_buffer_proves_itself() -> None:
    def fill(qty: float, open_: float) -> float:
        # next open with 10 bps slippage, then 10 bps fee
        return qty * open_ * 1.001 * 1.001

    assert fill(9.93, 100.5) == pytest.approx(999.961928, abs=1e-4)
    assert fill(9.93, 100.5) <= 1000
    assert fill(9.93, 100.6) == pytest.approx(1000.956915, abs=1e-4)
    assert fill(9.93, 100.6) > 1000
    orders, _ = run({"A": 1.0}, buf=0.0)
    check(orders, [("A", Side.BUY, 9.99)])
    assert fill(9.99, 100.5) == pytest.approx(1006.003994, abs=1e-4)
    assert fill(9.99, 100.5) > 1000


def test_rotation() -> None:
    orders, _ = run({"B": 1.0}, FakeCtx(cash=0.0, held={"A": 10.0}))
    check(orders, [("A", Side.SELL, 10.0), ("B", Side.BUY, 19.721)])


def test_band_exact_trades() -> None:
    orders, skipped = run({"A": 0.12}, FakeCtx(cash=900.0, held={"A": 1.0}))
    check(orders, [("A", Side.BUY, 0.2)])
    assert skipped == []


def test_band_below() -> None:
    assert run({"A": 0.1199}, FakeCtx(cash=900.0, held={"A": 1.0})) == ([], [])


def test_band_applies_to_exits() -> None:
    assert run({}, FakeCtx(cash=985.0, held={"A": 0.15})) == ([], [])


def test_skip_below_min_notional_buy() -> None:
    orders, skipped = run({"A": 0.104}, FakeCtx(cash=900.0, held={"A": 1.0}), band=0.0)
    assert orders == []
    assert skipped == [SkippedOrder(TS, "A", Side.BUY, 0.04, 100.0, "below_min_notional")]


def test_skip_below_min_notional_sell() -> None:
    ctx = FakeCtx(equity=100.0, cash=100.0, held={"A": 0.04})
    orders, skipped = run({}, ctx, band=0.0)
    assert orders == []
    assert skipped == [SkippedOrder(TS, "A", Side.SELL, 0.04, 100.0, "below_min_notional")]


def test_skip_zero_qty() -> None:
    ctx = FakeCtx(insts={"A": _inst("A", step=1.0, min_qty=1.0), "B": _inst("B")})
    orders, skipped = run({"A": 0.05}, ctx)
    assert orders == []
    assert skipped == [SkippedOrder(TS, "A", Side.BUY, 0.0, 100.0, "zero_qty")]


def test_skip_below_min_qty() -> None:
    ctx = FakeCtx(
        equity=100.0,
        cash=100.0,
        insts={"A": _inst("A", step=0.01, min_qty=0.1), "B": _inst("B")},
    )
    _, skipped = run({"A": 0.05}, ctx)
    assert skipped == [SkippedOrder(TS, "A", Side.BUY, 0.05, 100.0, "below_min_qty")]


def test_exactly_min_notional() -> None:
    orders, skipped = run({"B": 0.005}, band=0.0)
    check(orders, [("B", Side.BUY, 0.1)])
    assert skipped == []


def test_gross_cap_fits() -> None:
    orders, _ = run({"A": 3, "B": 1}, FakeCtx(cash=3000.0))
    check(orders, [("A", Side.BUY, 15.0), ("B", Side.BUY, 10.0)])


def test_gross_cap_scaled_by_cash() -> None:
    orders, _ = run({"A": 3, "B": 1})
    check(orders, [("A", Side.BUY, 7.447), ("B", Side.BUY, 4.965)])


def test_gross_exactly_cap_not_scaled() -> None:
    orders, _ = run({"A": 1.5, "B": 0.5}, FakeCtx(cash=3000.0))
    check(orders, [("A", Side.BUY, 15.0), ("B", Side.BUY, 10.0)])


def test_open_buy_netted() -> None:
    orders, _ = run({"A": 0.5}, open_orders=[mkt(Side.BUY, "A", 2.0)])
    check(orders, [("A", Side.BUY, 3.0)])


def test_open_buy_reserve_shrinks_budget() -> None:
    orders, _ = run({"A": 1.0}, open_orders=[mkt(Side.BUY, "A", 2.0)])
    check(orders, [("A", Side.BUY, 7.93)])


def test_open_sell_nets_to_target() -> None:
    ctx = FakeCtx(cash=500.0, held={"A": 5.0})
    assert run({"A": 0.2}, ctx, open_orders=[mkt(Side.SELL, "A", 3.0)]) == ([], [])


def test_sell_capped_at_holdings() -> None:
    ctx = FakeCtx(cash=700.0, held={"A": 3.0})
    orders, _ = run({}, ctx, open_orders=[mkt(Side.BUY, "A", 5.0)])
    check(orders, [("A", Side.SELL, 3.0)])


def test_open_limit_buy_reserved_no_skip() -> None:
    lim = Order("l1", "A", Side.BUY, 2.0, OrderType.LIMIT, limit_price=90.0)
    orders, skipped = run({"B": 0.5}, open_orders=[lim])
    check(orders, [("B", Side.BUY, 10.0)])
    assert skipped == []


def test_open_market_sell_credited() -> None:
    ctx = FakeCtx(cash=0.0, held={"A": 10.0})
    orders, _ = run({"B": 1.0}, ctx, open_orders=[mkt(Side.SELL, "A", 10.0)])
    check(orders, [("B", Side.BUY, 19.721)])


def test_open_limit_sell_not_credited() -> None:
    ctx = FakeCtx(cash=0.0, held={"A": 10.0})
    lim = Order("l1", "A", Side.SELL, 10.0, OrderType.LIMIT, limit_price=101.0)
    orders, skipped = run({"B": 1.0}, ctx, open_orders=[lim])
    assert orders == []
    assert skipped == [SkippedOrder(TS, "B", Side.BUY, 0.0, 50.0, "zero_qty")]


def test_full_exit_unrounded() -> None:
    held = 0.30000000000000004
    ctx = FakeCtx(
        equity=100.0,
        cash=70.0,
        insts={"A": _inst("A", step=0.1), "B": _inst("B")},
        held={"A": held},
    )
    orders, _ = run({}, ctx)
    assert orders[0].qty == held


def test_partial_sell_rounds_down() -> None:
    orders, _ = run({"A": 0.5}, FakeCtx(cash=0.0, held={"A": 10.0}))
    check(orders, [("A", Side.SELL, 5.0)])


@pytest.mark.parametrize(
    "weights", [{"A": -0.1}, {"A": None}, {"A": float("nan")}, {"A": float("inf")}, {"A": True}]
)
def test_bad_weights(weights: dict[str, object]) -> None:
    with pytest.raises(StrategyError, match="'A'"):
        run(weights)  # pyright: ignore[reportArgumentType]


def test_unknown_weight_symbol() -> None:
    with pytest.raises(StrategyError, match="'Z'"):
        run({"Z": 0.1})


def test_np_bool_weight_rejected() -> None:
    with pytest.raises(StrategyError):
        run({"A": np.bool_(True)})  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("band", [-0.1, float("nan"), None, True])
def test_bad_band(band: object) -> None:
    with pytest.raises(EngineError):
        run({"A": 0.5}, band=band)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("cap", [0, float("inf"), True])
def test_bad_cap(cap: object) -> None:
    with pytest.raises(EngineError):
        run({"A": 0.5}, cap=cap)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("buf", [-0.1, 1.0, float("nan")])
def test_bad_buffer(buf: float) -> None:
    with pytest.raises(EngineError):
        run({"A": 0.5}, buf=buf)


def test_perp_rejected() -> None:
    ctx = FakeCtx(insts={"A": _inst("A", kind=InstrumentKind.PERP), "B": _inst("B")})
    with pytest.raises(EngineError):
        run({"B": 0.5}, ctx)


def test_no_history() -> None:
    ctx = FakeCtx(refs={"A": None, "B": 50.0})
    with pytest.raises(StrategyError):
        run({"A": 0.5}, ctx)
    assert run({"B": 0.0}, ctx) == ([], [])
    assert run({}, ctx) == ([], [])


def test_zero_equity() -> None:
    assert run({"A": 0.5}, FakeCtx(equity=0.0, cash=0.0)) == ([], [])


def test_numpy_inputs_and_plain_floats() -> None:
    orders, _ = run({"A": np.float32(0.5), "B": np.int64(0)})  # pyright: ignore[reportArgumentType]
    check(orders, [("A", Side.BUY, 5.0)])
    ctx = FakeCtx(
        refs={"A": np.float64(100.0), "B": np.float64(50.0)},  # pyright: ignore[reportArgumentType]
        held={"A": np.float64(0.04)},  # pyright: ignore[reportArgumentType]
    )
    orders2, skipped = run({"B": np.int64(1)}, ctx, band=0.0)  # pyright: ignore[reportArgumentType]
    for o in orders + orders2:
        assert type(o.qty) is float
    assert skipped
    for s in skipped:
        assert type(s.ref_price) is float
        assert type(s.qty) is float


def test_deterministic_and_ids() -> None:
    a = run({"A": 0.3, "B": 0.3})
    b = run({"B": 0.3, "A": 0.3})
    assert a == b
    assert rebalance_order_id("BTCUSDT", TS) == "bt-rb-BTCUSDT-202401020000"
    assert all(o.tag == "rebalance" for o in a[0])


def test_gross_cap_counterexample_dust_stays() -> None:
    """Cap 0.5: B buys to the capped target while sub-minimum dust in C stays (INTERFACES 15.5)."""
    insts = {s: _inst(s) for s in "ABC"}
    ctx = FakeCtx(
        equity=10.001,
        cash=10.0,
        held={"C": 0.001},
        refs={"A": 1.0, "B": 1.0, "C": 1.0},
        insts=insts,
    )
    orders, skipped = run({"B": 1.0}, ctx, band=0.0, cap=0.5, buf=0.0)
    check(orders, [("B", Side.BUY, 5.0)])
    assert skipped == [SkippedOrder(TS, "C", Side.SELL, 0.001, 1.0, "below_min_notional")]


def test_unknown_open_order_symbol() -> None:
    with pytest.raises(EngineError):
        run({"A": 0.5}, open_orders=[mkt(Side.BUY, "Z", 1.0)])


@pytest.mark.parametrize("big", [10**400, -(10**400)])
def test_overflowing_ints(big: int) -> None:
    with pytest.raises(StrategyError):
        run({"A": big})
    with pytest.raises(EngineError):
        run({"A": 0.5}, band=big)


def test_gross_sum_overflow() -> None:
    with pytest.raises(StrategyError):
        run({"A": 1e308, "B": 1e308})
