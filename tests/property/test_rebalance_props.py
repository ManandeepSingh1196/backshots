from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from bt.core.types import (
    FundingEvent,
    Instrument,
    InstrumentKind,
    Order,
    OrderStatus,
    PositionView,
    Side,
)
from bt.engine.rebalance import weights_to_orders
from bt.execution.rules import CASH_ABS_TOL, QTY_REL_TOL, round_qty_down, validate_order
from bt.strategy.context import HistoryView

TS = datetime(2024, 1, 2, tzinfo=UTC)
SYMS = ("A", "B", "C")
STEP = 0.001
RATE = 0.001


def _inst(sym: str) -> Instrument:
    return Instrument(
        sym, sym, InstrumentKind.SPOT, sym, "USDT", 0.002, STEP, 5.0, 0.01, RATE, RATE, 1.0, 0.0
    )


INSTS = {s: _inst(s) for s in SYMS}


@dataclass
class _Hist:
    close: np.ndarray


class Ctx:
    def __init__(self, cash: float, held: Mapping[str, float], refs: Mapping[str, float]) -> None:
        self.cash = cash
        self.held = held
        self.refs = refs
        self.now = TS
        self.equity = cash + sum(held[s] * refs[s] for s in SYMS)

    def history(self, symbol: str, n: int) -> HistoryView:
        return _Hist(np.array([self.refs[symbol]]))  # pyright: ignore[reportReturnType]

    def funding_history(self, symbol: str, n: int) -> Sequence[FundingEvent]:
        return ()

    def position(self, symbol: str) -> PositionView:
        return PositionView(symbol, self.held[symbol], 0.0, 0.0, 0.0)

    def instrument(self, symbol: str) -> Instrument:
        return INSTS[symbol]

    def symbols(self) -> Sequence[str]:
        return SYMS

    def log(self, msg: str, **fields: object) -> None:
        pass


class View:
    def __init__(self, cash: float, held: Mapping[str, float]) -> None:
        self.cash = cash
        self.available_cash = cash
        self.equity = cash
        self.ts = None
        self._held = held

    def position(self, symbol: str) -> PositionView:
        return PositionView(symbol, self._held[symbol], 0.0, 0.0, 0.0)

    def positions(self) -> Mapping[str, PositionView]:
        return {s: self.position(s) for s in SYMS}


@st.composite
def scenarios(draw: st.DrawFn):
    refs = {s: draw(st.floats(1.0, 500.0)) for s in SYMS}
    held = {s: draw(st.integers(0, 20_000)) * STEP for s in SYMS}
    weights = {s: draw(st.floats(0.0, 1.5)) for s in SYMS if draw(st.booleans())}
    cash = draw(st.floats(0.0, 5000.0))
    band = draw(st.sampled_from([0.0, 0.02, 0.1]))
    cap = draw(st.sampled_from([0.5, 1.0, 2.0]))
    buf = draw(st.sampled_from([0.0, 0.006]))
    return refs, held, weights, cash, band, cap, buf


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(scenarios())
def test_rebalance_properties(
    sc: tuple[dict[str, float], dict[str, float], dict[str, float], float, float, float, float],
) -> None:
    refs, held, weights, cash, band, cap, buf = sc
    ctx = Ctx(cash, held, refs)
    orders, skipped = weights_to_orders(weights, ctx, band, cap, cash_buffer=buf)

    sides = [o.side for o in orders]
    assert sides == sorted(sides, key=lambda x: x != Side.SELL)
    for side in (Side.SELL, Side.BUY):
        syms = [o.symbol for o in orders if o.side == side]
        assert syms == sorted(syms)
    assert len({o.symbol for o in orders}) == len(orders)
    assert all(s.qty >= 0 for s in skipped)

    proceeds = 0.0
    buy_cost = 0.0
    after = dict(held)
    for o in orders:
        ref = refs[o.symbol]
        assert o.qty >= 0.002 * (1 - QTY_REL_TOL)
        assert o.qty * ref >= 5.0 * (1 - QTY_REL_TOL)
        if o.side == Side.SELL:
            assert o.qty <= held[o.symbol] + 1e-12
            if o.qty != held[o.symbol]:
                assert round_qty_down(o.qty, INSTS[o.symbol]) == o.qty
            proceeds += o.qty * ref * (1 - RATE) * (1 - buf)
            after[o.symbol] -= o.qty
        else:
            assert round_qty_down(o.qty, INSTS[o.symbol]) == o.qty
            buy_cost += o.qty * ref * (1 + RATE) * (1 + buf)
            after[o.symbol] += o.qty
    assert buy_cost <= cash + proceeds + CASH_ABS_TOL

    view = View(cash + proceeds, held)
    for o in orders:
        ack = validate_order(o, INSTS[o.symbol], refs[o.symbol], view)
        assert ack.status == OrderStatus.NEW, (o, ack)

    before = sum(held[s] * refs[s] for s in SYMS)
    gross_after = sum(after[s] * refs[s] for s in SYMS)
    # Ticket bound is max(cap * equity, before) * (1 + 1e-9). Positions left alone by the band
    # or by the min-notional filter (each worth < band * equity or < min_notional) sit on top of
    # the capped targets, so the bound needs that slack. See the T037 report.
    slack = len(SYMS) * (band * ctx.equity + 5.0)
    assert gross_after <= max(cap * ctx.equity + slack, before) * (1 + 1e-9)


def test_order_type_is_plain() -> None:
    ctx = Ctx(1000.0, dict.fromkeys(SYMS, 0.0), dict.fromkeys(SYMS, 100.0))
    orders, _ = weights_to_orders({"A": 0.5}, ctx, 0.02, 2.0)
    assert all(isinstance(o, Order) for o in orders)
