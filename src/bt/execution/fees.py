"""Fee model: venue fee rates applied to notional."""

import math
from typing import Protocol

from bt.core.errors import ExecutionError
from bt.core.types import Instrument, Side


class FeeModel(Protocol):
    """Computes the fee in quote currency for one fill."""

    def fee(self, inst: Instrument, side: Side, qty: float, price: float, is_maker: bool) -> float:
        """Fee in USDT; negative for a rebate."""
        ...


class VenueFees:
    """Fee = qty * price * rate, with rate taken from the instrument unless overridden.

    Invariants: overrides are finite; taker override >= 0; maker override > -0.01.
    Stateless after construction; fee() is pure and raises ExecutionError on
    negative or non-finite qty, or non-positive or non-finite price.
    """

    def __init__(self, maker: float | None = None, taker: float | None = None) -> None:
        for name, value in (("maker", maker), ("taker", taker)):
            if value is not None and not math.isfinite(value):
                raise ExecutionError(f"VenueFees.{name} must be finite, got {value!r}")
        if taker is not None and taker < 0:
            raise ExecutionError(f"VenueFees.taker must be >= 0, got {taker!r}")
        if maker is not None and maker <= -0.01:
            raise ExecutionError(f"VenueFees.maker must be > -0.01, got {maker!r}")
        self._maker = maker
        self._taker = taker

    def fee(self, inst: Instrument, side: Side, qty: float, price: float, is_maker: bool) -> float:
        if not math.isfinite(qty) or qty < 0:
            raise ExecutionError(f"fee: qty must be finite and >= 0, got {qty!r}")
        if not math.isfinite(price) or price <= 0:
            raise ExecutionError(f"fee: price must be finite and > 0, got {price!r}")
        if qty == 0:
            return 0.0
        if is_maker:
            rate = inst.maker_fee if self._maker is None else self._maker
        else:
            rate = inst.taker_fee if self._taker is None else self._taker
        return qty * price * rate
