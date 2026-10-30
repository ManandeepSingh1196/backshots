"""buy_hold_curve and benchmark_metrics against hand-computed numbers."""

import math
import statistics
from datetime import UTC, datetime, timedelta
from itertools import pairwise

import numpy as np
import pytest

from bt.analytics.benchmark import benchmark_metrics, buy_hold_curve
from bt.analytics.metrics import BENCHMARK_KEYS
from bt.core.errors import InvariantError
from bt.core.types import Instrument, InstrumentKind

TOL = 1e-9
OPEN = [100.0, 100.2, 108.0, 111.0, 113.0]
CLOSE = [100.0, 108.0, 110.0, 112.0, 114.0]


def inst(fee: float = 0.001) -> Instrument:
    return Instrument(
        symbol="BTCUSDT",
        venue_symbol="BTCUSDT",
        kind=InstrumentKind.SPOT,
        base="BTC",
        quote="USDT",
        min_qty=0.001,
        qty_step=0.001,
        min_notional=5.0,
        price_tick=0.01,
        maker_fee=fee,
        taker_fee=fee,
    )


def curve(**kw: object) -> list[float]:
    args: dict[str, object] = {
        "open_": OPEN,
        "close": CLOSE,
        "inst": inst(),
        "initial_equity": 1000.0,
        "slippage_frac": 0.001,
        "close_at_end": True,
    }
    args.update(kw)
    return buy_hold_curve(**args)  # type: ignore[arg-type]


CURVE = [1000.0, 1075.539028012, 1095.419028012, 1115.299028012, 1132.913841172]


def test_cash_and_quantity() -> None:
    c = curve(close_at_end=False)
    # row i = cash + 9.94 * close[i]; cash = 2.019028012
    assert c[1] - 9.94 * 108.0 == pytest.approx(2.019028012, abs=TOL)


def test_close_at_end() -> None:
    assert curve() == pytest.approx(CURVE, abs=TOL)


def test_no_close_at_end() -> None:
    assert curve(close_at_end=False) == pytest.approx([*CURVE[:4], 1135.179028012], abs=TOL)


def test_costs_off() -> None:
    c = curve(inst=inst(0.0), slippage_frac=0.0, alloc=1.0, close_at_end=False)
    assert c == pytest.approx([1000, 1077.844, 1097.804, 1117.764, 1137.724], abs=TOL)


def test_clamp() -> None:
    c = curve(open_=[100.0, 102.0, 108.0, 111.0, 113.0])
    exp = [1000, 1056.707066032, 1076.275066032, 1095.843066032, 1113.181429408]
    assert c == pytest.approx(exp, abs=TOL)
    assert min(c) > 0


def test_one_bar_and_types() -> None:
    assert curve(open_=[100.0], close=[100.0]) == [1000.0]
    c = curve(open_=np.array(OPEN), close=np.array(CLOSE), initial_equity=np.float64(1000.0))
    assert all(type(v) is float for v in c)
    assert c == pytest.approx(CURVE, abs=TOL)


@pytest.mark.parametrize(
    "kw",
    [
        {"open_": [], "close": []},
        {"open_": OPEN[:4]},
        {"open_": [100.0, math.nan, 1, 1, 1]},
        {"close": [100.0, 0.0, 1, 1, 1]},
        {"close": [100.0, -1.0, 1, 1, 1]},
        {"open_": [100.0, math.inf, 1, 1, 1]},
        {"initial_equity": 0.0},
        {"slippage_frac": -0.1},
        {"slippage_frac": 1.0},
        {"alloc": 0.0},
        {"alloc": 1.5},
    ],
)
def test_bad_input(kw: dict[str, object]) -> None:
    with pytest.raises(InvariantError):
        curve(**kw)


TS = [datetime(2024, 1, d, tzinfo=UTC) for d in range(1, 6)]
RUN = [1000.0, 1010.0, 1005.0, 1030.0, 1040.0]


def test_metrics_basic() -> None:
    m = benchmark_metrics(TS, RUN, TS, CURVE, "1d")
    assert set(m) == set(BENCHMARK_KEYS)
    assert all(type(v) is float for v in m.values())
    assert m["benchmark_total_return"] == pytest.approx(0.132913841172, abs=TOL)
    assert m["benchmark_max_drawdown"] == pytest.approx(0.0, abs=TOL)
    r = [b / a - 1 for a, b in pairwise(CURVE)]
    assert m["benchmark_sharpe"] == pytest.approx(
        statistics.mean(r) / statistics.stdev(r) * math.sqrt(365), rel=1e-9
    )
    assert m["benchmark_correlation"] == pytest.approx(0.000696226541, abs=TOL)


def test_partial_overlap() -> None:
    keep = [0, 1, 3, 4]
    m = benchmark_metrics([TS[i] for i in keep], [RUN[i] for i in keep], TS, CURVE, "1d")
    assert m["benchmark_correlation"] == pytest.approx(-0.140746356687, abs=TOL)


def test_nan_correlations() -> None:
    assert math.isnan(benchmark_metrics(TS, [1000.0] * 5, TS, CURVE, "1d")["benchmark_correlation"])
    two = benchmark_metrics(TS[:2], RUN[:2], TS, CURVE, "1d")
    assert math.isnan(two["benchmark_correlation"])
    assert not math.isnan(two["benchmark_total_return"])


def test_metrics_bad_input() -> None:
    with pytest.raises(InvariantError):
        benchmark_metrics(TS, RUN, TS, CURVE[:3], "1d")
    naive = [t.replace(tzinfo=None) + timedelta(0) for t in TS]
    with pytest.raises(InvariantError):
        benchmark_metrics(naive, RUN, TS, CURVE, "1d")
