from datetime import UTC, datetime

import pytest

from bt.core.errors import ExecutionError
from bt.core.types import Bar, Instrument, InstrumentKind, Side
from bt.execution.slippage import FixedBps

INST = Instrument(
    "AAAUSDT", "AAAUSDT", InstrumentKind.SPOT, "AAA", "USDT", 0.001, 0.001, 5.0, 0.01, 0.001, 0.001
)
BAR = Bar("AAAUSDT", datetime(2024, 1, 1, tzinfo=UTC), 100.0, 101.0, 99.0, 100.0, 10.0, 1000.0)


def test_fixed_bps() -> None:
    assert FixedBps(5).slippage_frac(INST, Side.BUY, 1.0, BAR) == pytest.approx(0.0005, abs=1e-15)
    assert FixedBps().slippage_frac(INST, Side.SELL, 99.0, BAR) == pytest.approx(0.0005, abs=1e-15)
    assert FixedBps(0).slippage_frac(INST, Side.BUY, 1.0, BAR) == 0.0


@pytest.mark.parametrize("bps", [-1.0, 10_000.0, float("nan"), float("inf")])
def test_fixed_bps_invalid(bps: float) -> None:
    with pytest.raises(ExecutionError):
        FixedBps(bps)
