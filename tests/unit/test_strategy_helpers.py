from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from bt.core.errors import StrategyError
from bt.core.events import EventLog
from bt.core.types import Bar, Instrument, InstrumentKind
from bt.strategy.context import StepContext
from bt.strategy.helpers import affordable_qty, has_bar_now
from tests.support.memory_feed import MemoryFeed

INST = Instrument(
    "AAAUSDT", "AAAUSDT", InstrumentKind.SPOT, "AAA", "USDT", 0.001, 0.001, 5.0, 0.01, 0.001, 0.001
)


def test_basic() -> None:
    r = affordable_qty(995.0, np.float64(100.0), INST)  # pyright: ignore[reportArgumentType]
    assert r == pytest.approx(9.94, abs=1e-12)
    assert type(r) is float


def test_larger_fee_used() -> None:
    inst = replace(INST, maker_fee=0.002)
    assert affordable_qty(995.0, 100.0, inst) == pytest.approx(9.93, abs=1e-12)


def test_minimums() -> None:
    assert affordable_qty(4.0, 100.0, INST) == 0.0
    inst = replace(INST, min_notional=0.0, qty_step=0.01, min_qty=0.1)
    assert affordable_qty(5.0, 100.0, inst) == 0.0
    assert affordable_qty(5.005, 100.0, INST) == pytest.approx(0.05, abs=1e-12)


@pytest.mark.parametrize(
    ("n", "p"), [(100.0, 0.0), (100.0, float("nan")), (-1.0, 100.0), (float("inf"), 100.0)]
)
def test_bad_input(n: float, p: float) -> None:
    with pytest.raises(StrategyError):
        affordable_qty(n, p, INST)


class _Pf:
    ts = None
    cash = 1000.0
    available_cash = 1000.0
    equity = 1000.0

    def position(self, symbol: str) -> object:
        raise NotImplementedError

    def positions(self) -> dict[str, object]:
        return {}


def test_has_bar_now() -> None:
    other = replace(INST, symbol="BBBUSDT", venue_symbol="BBBUSDT", base="BBB")
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    bars = [
        Bar("AAAUSDT", t0 + timedelta(days=i), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0) for i in range(2)
    ] + [Bar("BBBUSDT", t0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)]
    feed = MemoryFeed([INST, other], bars)
    seen: list[tuple[bool, bool]] = []
    for step in feed.steps():
        ctx = StepContext(step.ts, feed, _Pf(), [INST, other], EventLog())  # pyright: ignore[reportArgumentType]
        seen.append((has_bar_now(ctx, "AAAUSDT"), has_bar_now(ctx, "BBBUSDT")))
    assert seen == [(True, True), (True, False)]


def test_has_bar_now_empty_history() -> None:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    feed = MemoryFeed([INST], [Bar("AAAUSDT", t0 + timedelta(days=5), 1, 1, 1, 1, 1, 1)])
    ctx = StepContext(t0, feed, _Pf(), [INST], EventLog())  # pyright: ignore[reportArgumentType]
    assert has_bar_now(ctx, "AAAUSDT") is False
