"""Sizing and data helpers for strategies; import only bt.core and the Context."""

import math
from typing import Final

import numpy as np

from bt.core.errors import InvariantError, StrategyError
from bt.core.rounding import floor_to_step
from bt.core.types import Instrument
from bt.strategy.context import Context

# Mirrors bt.execution.rules.QTY_REL_TOL (strategy may not import bt.execution).
_MIN_REL_TOL: Final = 1e-9


def _finite(name: str, x: object) -> float:
    if isinstance(x, bool):
        raise StrategyError(f"affordable_qty: {name} must be a number, got bool")
    try:
        f = float(x)  # pyright: ignore[reportArgumentType]
    except (TypeError, ValueError) as exc:
        raise StrategyError(f"affordable_qty: {name} must be a number, got {x!r}") from exc
    if not math.isfinite(f):
        raise StrategyError(f"affordable_qty: {name} must be finite, got {f!r}")
    return f


def affordable_qty(notional: float, price: float, inst: Instrument) -> float:
    """Largest grid quantity buyable with `notional` including the reserved fee, or 0.0.

    Returns 0.0 if the quantity is below min_qty or its notional below min_notional.
    """
    n = _finite("notional", notional)
    p = _finite("price", price)
    if n < 0:
        raise StrategyError(f"affordable_qty: notional must be >= 0, got {n!r}")
    if p <= 0:
        raise StrategyError(f"affordable_qty: price must be > 0, got {p!r}")
    rate = max(inst.maker_fee, inst.taker_fee, 0.0)
    try:
        qty = floor_to_step(n / (p * (1.0 + rate)), inst.qty_step)
    except InvariantError as exc:
        raise StrategyError(f"affordable_qty: cannot size {n!r} at {p!r}: {exc}") from exc
    keep = 1.0 - _MIN_REL_TOL
    if qty < inst.min_qty * keep or qty * p < inst.min_notional * keep:
        return 0.0
    return float(qty)


def has_bar_now(ctx: Context, symbol: str) -> bool:
    """True when the last bar of ctx.history(symbol, 1) has ts == ctx.now."""
    ts = ctx.history(symbol, 1).ts
    if len(ts) == 0:
        return False
    now64 = np.datetime64(ctx.now.replace(tzinfo=None), "us")
    return bool(ts[-1] == now64)
