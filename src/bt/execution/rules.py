"""Exchange order rules shared by backtest and live: rounding and order validation."""

import math
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Final, get_args

from bt.core.errors import ExecutionError
from bt.core.events import RejectReason
from bt.core.types import (
    Instrument,
    InstrumentKind,
    Order,
    OrderAck,
    OrderStatus,
    OrderType,
    PortfolioView,
    Side,
)

QTY_REL_TOL: Final = 1e-9
CASH_ABS_TOL: Final = 1e-6
_GRID_ABS_TOL: Final = 1e-6


def _dec(x: float) -> Decimal:
    return Decimal(repr(x))


def _decimals(step: float) -> int:
    exp = _dec(step).normalize().as_tuple().exponent
    assert isinstance(exp, int)
    return max(0, -exp)


def _on_grid(value: float, step: float) -> bool:
    """Tolerant float check: within abs 1e-6 or rel QTY_REL_TOL of a grid point."""
    x = value / step
    dist = abs(x - round(x))
    return dist <= _GRID_ABS_TOL or dist <= QTY_REL_TOL * max(1.0, abs(x))


def round_qty_down(qty: float, inst: Instrument) -> float:
    """Floor qty to the instrument's qty_step with exact Decimal arithmetic.

    May return less than min_qty; the caller decides what to do with that.
    """
    if not math.isfinite(qty) or qty < 0:
        raise ExecutionError(f"round_qty_down: qty must be finite and >= 0, got {qty!r}")
    step = _dec(inst.qty_step)
    k = _dec(qty) // step
    return float(k * step)


def round_price(price: float, inst: Instrument, side: Side) -> float:
    """Round to the tick toward the less favorable side: BUY up, SELL down."""
    if not math.isfinite(price) or price <= 0:
        raise ExecutionError(f"round_price: price must be finite and > 0, got {price!r}")
    tick = inst.price_tick
    x = price / tick
    nearest = round(x)
    if abs(x - nearest) <= QTY_REL_TOL * max(1.0, x):
        k = Decimal(nearest)
    elif side == Side.BUY:
        k = _dec(x).to_integral_value(rounding=ROUND_CEILING)
    else:
        k = _dec(x).to_integral_value(rounding=ROUND_FLOOR)
    result = round(float(k * _dec(tick)), _decimals(tick))
    if result <= 0:
        raise ExecutionError(f"round_price: {price!r} rounds to {result!r} on tick {tick!r}")
    return result


def _reject(order: Order, reason: RejectReason) -> OrderAck:
    assert reason in get_args(RejectReason)
    return OrderAck(order.id, OrderStatus.REJECTED, reason)


def validate_order(
    order: Order, inst: Instrument, ref_price: float, portfolio: PortfolioView
) -> OrderAck:
    """Pure validation of an order against exchange rules and the portfolio view.

    Returns a NEW ack or a REJECTED ack carrying the first failing RejectReason.
    Raises ExecutionError only for programmer errors (wrong instrument, bad ref_price).
    """
    if order.symbol != inst.symbol:
        raise ExecutionError(f"order {order.id} is for {order.symbol}, not {inst.symbol}")
    if not math.isfinite(ref_price) or ref_price <= 0:
        raise ExecutionError(f"ref_price must be finite and > 0, got {ref_price!r}")
    if inst.kind != InstrumentKind.SPOT:
        return _reject(order, "unsupported_kind")
    qty = order.qty
    if qty < inst.min_qty * (1 - QTY_REL_TOL):
        return _reject(order, "below_min_qty")
    if not _on_grid(qty, inst.qty_step):
        return _reject(order, "qty_off_step")
    if order.type == OrderType.LIMIT:
        assert order.limit_price is not None
        p = order.limit_price
    elif order.type == OrderType.STOP:
        assert order.stop_price is not None
        p = order.stop_price
    else:
        p = ref_price
    if order.type != OrderType.MARKET and not _on_grid(p, inst.price_tick):
        return _reject(order, "price_off_tick")
    if qty * p < inst.min_notional * (1 - QTY_REL_TOL):
        return _reject(order, "below_min_notional")
    if order.side == Side.BUY:
        if order.reduce_only:
            return _reject(order, "reduce_only_increases")
        rate = max(inst.maker_fee, inst.taker_fee, 0.0)
        if qty * p * (1 + rate) > portfolio.available_cash + CASH_ABS_TOL:
            return _reject(order, "insufficient_cash")
    else:
        held = portfolio.position(order.symbol).qty
        if qty > held + QTY_REL_TOL * inst.qty_step:
            return _reject(order, "insufficient_holdings")
    return OrderAck(order.id, OrderStatus.NEW)
