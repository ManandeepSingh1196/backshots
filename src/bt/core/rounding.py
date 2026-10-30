"""Exact grid rounding shared by execution and strategies. Stdlib only."""

import math
from decimal import Decimal, InvalidOperation

from bt.core.errors import InvariantError


def _coerce(name: str, x: object) -> float:
    if isinstance(x, bool) or type(x).__name__ in ("bool", "bool_"):  # incl. numpy bools
        raise InvariantError(f"floor_to_step: {name} must be a number, got bool {x!r}")
    try:
        f = float(x)  # pyright: ignore[reportArgumentType]
    except (TypeError, ValueError) as exc:
        raise InvariantError(f"floor_to_step: {name} must be a number, got {x!r}") from exc
    if not math.isfinite(f):
        raise InvariantError(f"floor_to_step: {name} must be finite, got {f!r}")
    return f


def floor_to_step(value: float, step: float) -> float:
    """Exact Decimal floor of value to a multiple of step.

    Coerces with float() first, so numpy scalars work. Always returns a Python float.
    value must be finite and >= 0, step finite and > 0, neither a bool; else InvariantError.
    """
    v = _coerce("value", value)
    s = _coerce("step", step)
    if v < 0:
        raise InvariantError(f"floor_to_step: value must be >= 0, got {v!r}")
    if s <= 0:
        raise InvariantError(f"floor_to_step: step must be > 0, got {s!r}")
    try:
        dstep = Decimal(repr(s))
        return float((Decimal(repr(v)) // dstep) * dstep)
    except InvalidOperation as exc:
        raise InvariantError(f"floor_to_step: {v!r} is too large for step {s!r}") from exc
