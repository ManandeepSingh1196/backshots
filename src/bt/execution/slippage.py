"""Slippage models."""

import math
from typing import Protocol

from bt.core.errors import ExecutionError
from bt.core.types import Bar, Instrument, Side


class SlippageModel(Protocol):
    """Adverse price move as a fraction of price for one fill."""

    def slippage_frac(self, inst: Instrument, side: Side, qty: float, bar: Bar) -> float:
        """Fraction >= 0 by which the fill price is worse than the reference."""
        ...


class FixedBps:
    """Constant slippage of bps / 10_000 for every fill.

    Invariants: 0 <= bps < 10_000 and finite. Stateless; result never depends on arguments.
    """

    def __init__(self, bps: float = 5.0) -> None:
        if not math.isfinite(bps) or not 0 <= bps < 10_000:
            raise ExecutionError(f"FixedBps.bps must be in [0, 10000), got {bps!r}")
        self._frac = bps / 10_000

    def slippage_frac(self, inst: Instrument, side: Side, qty: float, bar: Bar) -> float:
        return self._frac
