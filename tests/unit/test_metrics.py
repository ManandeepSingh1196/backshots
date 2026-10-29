"""Tests for bt.analytics.metrics. All expected numbers are hand-computed."""

import math
import pickle
from datetime import UTC, datetime, timedelta

import pytest

from bt.analytics.metrics import (
    BENCHMARK_KEYS,
    PERIODS_PER_YEAR,
    CurveStats,
    compute_metrics,
    curve_stats,
    drawdown_series,
    yearly_returns,
)
from bt.core.errors import InvariantError
from bt.core.events import EquitySnapshot, OrderRecord, SkippedOrder
from bt.core.types import Fill, Order, OrderStatus, Side
from bt.data.validate import Timeframe
from bt.engine.loop import EngineResult

D1 = datetime(2024, 1, 1, tzinfo=UTC)


def days(n: int, step: timedelta = timedelta(days=1)) -> list[datetime]:
    return [D1 + i * step for i in range(n)]


def test_periods_per_year() -> None:
    assert dict(PERIODS_PER_YEAR) == {"1d": 365, "4h": 2190}


def test_curve_a() -> None:
    e = [100.0, 110.0, 99.0, 108.9, 119.79]
    s = curve_stats(days(5), e, "1d")
    assert s.total_return == pytest.approx(0.1979, rel=1e-12)
    assert s.vol_annual == pytest.approx(0.1 * math.sqrt(365), rel=1e-9)
    assert s.sharpe == pytest.approx(0.5 * math.sqrt(365), rel=1e-9)
    assert s.sortino == pytest.approx(1.0 * math.sqrt(365), rel=1e-9)
    assert s.max_drawdown == pytest.approx(0.1, rel=1e-12)
    assert s.max_drawdown_days == 3.0
    cagr = 1.1979 ** (365 / 4) - 1
    assert s.cagr == pytest.approx(cagr, rel=1e-9)
    assert s.calmar == pytest.approx(cagr / 0.1, rel=1e-9)
    assert drawdown_series(e) == pytest.approx([0, 0, -0.1, -0.01, 0], abs=1e-12)


def test_unrecovered() -> None:
    s = curve_stats(days(4), [100, 120, 90, 96], "1d")
    assert s.max_drawdown == pytest.approx(0.25, rel=1e-12)
    assert s.max_drawdown_days == 2.0


def test_two_spells() -> None:
    s = curve_stats(days(7), [100, 90, 100, 100, 80, 85, 90], "1d")
    assert s.max_drawdown_days == 3.0
    assert s.max_drawdown == pytest.approx(0.2, rel=1e-12)


def test_cagr_exact_one_year() -> None:
    e = [100.0 + 100.0 * i / 365 for i in range(366)]
    s = curve_stats(days(366), e, "1d")
    assert s.cagr == pytest.approx(1.0, rel=1e-9)
    e4 = [100.0 + 100.0 * i / 2190 for i in range(2191)]
    s4 = curve_stats(days(2191, timedelta(hours=4)), e4, "4h")
    assert s4.cagr == pytest.approx(1.0, rel=1e-9)


def test_cagr_years_value() -> None:
    m = compute_metrics(_result(), "1d", 1000.0, 85.0)
    assert m["years"] == pytest.approx(3 / 365, rel=1e-12)
    ts = days(2191, timedelta(hours=4))
    r = _result(equity=[100.0 + 100.0 * i / 2190 for i in range(2191)], ts=ts)
    assert compute_metrics(r, "4h", 100.0, 85.0)["years"] == pytest.approx(1.0, rel=1e-12)


def test_4h_sharpe() -> None:
    e = [100.0, 110.0, 99.0, 108.9, 119.79]
    s = curve_stats(days(5, timedelta(hours=4)), e, "4h")
    assert s.sharpe == pytest.approx(0.5 * math.sqrt(2190), rel=1e-9)


def test_flat_curve() -> None:
    s = curve_stats(days(3), [100, 100, 100], "1d")
    assert math.isnan(s.sharpe)
    assert s.vol_annual == 0.0
    assert math.isnan(s.sortino)
    assert s.max_drawdown == 0.0
    assert math.isnan(s.calmar)
    assert s.cagr == pytest.approx(0.0, abs=1e-12)


def test_single_row() -> None:
    s = curve_stats(days(1), [100], "1d")
    assert s.total_return == 0.0
    assert math.isnan(s.cagr)
    assert math.isnan(s.sharpe)
    assert math.isnan(s.sortino)
    assert s.max_drawdown_days == 0.0


@pytest.mark.parametrize("g", [1.1, 1.01, 1.003, 1.07])
def test_constant_growth_has_zero_spread(g: float) -> None:
    # Constant-growth returns differ only by float noise; treated as zero spread.
    s = curve_stats(days(30), [100.0 * g**i for i in range(30)], "1d")
    assert math.isnan(s.sharpe)
    assert s.vol_annual == 0.0
    assert math.isnan(s.sortino)


def test_invalid_inputs() -> None:
    with pytest.raises(InvariantError):
        curve_stats(days(2), [0, 1], "1d")
    with pytest.raises(InvariantError):
        curve_stats(days(2), [100, math.nan], "1d")
    with pytest.raises(InvariantError):
        curve_stats([D1, D1], [100, 100], "1d")
    with pytest.raises(InvariantError):
        curve_stats(days(3), [100, 100], "1d")
    with pytest.raises(InvariantError):
        curve_stats([datetime(2024, 1, 1), datetime(2024, 1, 2)], [100, 101], "1d")  # noqa: DTZ001


def test_non_positive_later_equity() -> None:
    s = curve_stats(days(3), [100, 50, 0], "1d")
    assert s.cagr == -1.0
    r = [-0.5, -1.0]
    mean = -0.75
    sd = math.sqrt(((r[0] - mean) ** 2 + (r[1] - mean) ** 2) / 1)
    assert s.sharpe == pytest.approx(mean / sd * math.sqrt(365), rel=1e-9)


def test_yearly_returns() -> None:
    ts = [
        datetime(2023, 12, 30, tzinfo=UTC),
        datetime(2023, 12, 31, tzinfo=UTC),
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 6, 1, tzinfo=UTC),
    ]
    out = yearly_returns(ts, [100, 110, 121, 99])
    assert [y for y, _ in out] == [2023, 2024]
    assert out[0][1] == pytest.approx(0.1, rel=1e-12)
    assert out[1][1] == pytest.approx(-0.1, rel=1e-12)


# ---- compute_metrics ----


def _fill(oid: str, i: int, side: Side, price: float) -> Fill:
    return Fill(oid, "AAA", D1 + timedelta(days=i), side, 1.0, price, 0.0, 0.0, False)


def _result(
    equity: list[float] | None = None,
    ts: list[datetime] | None = None,
    killed_at: datetime | None = None,
    tax: list[float] | None = None,
) -> EngineResult:
    if equity is None:
        equity = [1000.0, 1010.0, 990.0, 1002.0]
        gross = [0.0, 505.0, 495.0, 0.0]
    else:
        gross = [0.0] * len(equity)
    n = len(equity)
    taxes = tax or [0.0] * n
    snaps = tuple(
        EquitySnapshot(
            ts=(ts[i] if ts else D1 + timedelta(days=i)),
            equity=equity[i],
            cash=equity[i] - gross[i],
            gross_exposure=gross[i],
            net_exposure=gross[i],
            realized_pnl=0.0,
            fees_paid=1.5 if i == n - 1 else 0.0,
            slippage_paid=0.5 if i == n - 1 else 0.0,
            tax_paid=taxes[i],
        )
        for i in range(n)
    )
    fills = (
        _fill("b1", 1, Side.BUY, 100.0),
        _fill("b2", 1, Side.BUY, 110.0),
        _fill("s1", 2, Side.SELL, 120.0),
        _fill("s2", 3, Side.SELL, 100.0),
    )
    orders = (
        OrderRecord(D1, Order("o1", "AAA", Side.BUY, 1.0), OrderStatus.NEW),
        OrderRecord(D1, Order("o1", "AAA", Side.BUY, 1.0), OrderStatus.FILLED),
        OrderRecord(D1, Order("o2", "AAA", Side.BUY, 1.0), OrderStatus.REJECTED, "below_min_qty"),
        OrderRecord(D1, Order("o3", "AAA", Side.BUY, 1.0), OrderStatus.CANCELED, "user"),
    )
    skipped = (SkippedOrder(D1, "AAA", Side.BUY, 0.0, 100.0, "zero_qty"),)
    return EngineResult(snaps, fills, orders, skipped, (), (), killed_at, False)


def _eq(a: float, b: float) -> bool:
    return a == b or (math.isnan(a) and math.isnan(b))


def test_compute_metrics_hand_built() -> None:
    m = compute_metrics(_result(), "1d", 1000.0, 85.0)
    exp = {
        "cost_fees_usdt": 1.5,
        "cost_fees_inr": 127.5,
        "cost_fees_frac": 0.0015,
        "cost_slippage_inr": 42.5,
        "cost_total_usdt": 2.0,
        "cost_total_inr": 170.0,
        "cost_funding_usdt": 0.0,
        "cost_funding_inr": 0.0,
        "cost_funding_frac": 0.0,
        "cost_tax_usdt": 0.0,
        "cost_tax_inr": 0.0,
        "cost_tax_frac": 0.0,
        "n_fills": 4.0,
        "n_buys": 2.0,
        "n_sells": 2.0,
        "win_rate": 0.5,
        "avg_trade_pnl_usdt": 5.0,
        "traded_notional_usdt": 430.0,
        "exposure_time": 0.5,
        "avg_gross_leverage": 0.25,
        "final_equity_inr_after_tax": 85170.0,
        "n_orders": 3.0,
        "n_rejected": 1.0,
        "n_canceled": 1.0,
        "n_skipped": 1.0,
        "killed": 0.0,
        "liquidated": 0.0,
        "n_bars": 4.0,
        "initial_equity_usdt": 1000.0,
        "initial_equity_inr": 85000.0,
        "inr_per_usdt": 85.0,
        "missing_bars": 0.0,
    }
    for k, v in exp.items():
        assert m[k] == pytest.approx(v, rel=1e-12, abs=1e-12), k
    # avg_gross_leverage: (0 + 505/1010 + 495/990 + 0)/4
    assert m["years"] == pytest.approx(3 / 365, rel=1e-12)
    assert m["turnover_annual"] == pytest.approx(430 / 1000.5 / (3 / 365), rel=1e-12)
    for k, v in m.items():
        if k.endswith("_pre_tax"):
            assert _eq(v, m[k.replace("_pre_tax", "_after_tax")]), k


def test_pre_after_differ_with_tax() -> None:
    r = _result(equity=[1000.0, 1010.0, 990.0, 1002.0], tax=[0.0, 0.0, 5.0, 5.0])
    m = compute_metrics(r, "1d", 1000.0, 85.0)
    assert m["total_return_pre_tax"] - m["total_return_after_tax"] == pytest.approx(
        5 / 1000, abs=1e-12
    )
    assert m["final_equity_usdt_pre_tax"] == pytest.approx(1007.0, rel=1e-12)
    assert m["final_equity_usdt_after_tax"] == pytest.approx(1002.0, rel=1e-12)
    assert m["cost_tax_usdt"] == 5.0


_CURVE_FIELDS = [
    "total_return",
    "cagr",
    "vol_annual",
    "sharpe",
    "sortino",
    "max_drawdown",
    "max_drawdown_days",
    "calmar",
    "final_equity_usdt",
    "final_equity_inr",
]
_EXPECTED_KEYS = (
    {f"{f}_{s}" for f in _CURVE_FIELDS for s in ("pre_tax", "after_tax")}
    | {
        "initial_equity_usdt",
        "initial_equity_inr",
        "inr_per_usdt",
        "n_bars",
        "years",
        "missing_bars",
    }
    | {
        f"cost_{c}_{u}"
        for c in ("fees", "slippage", "funding", "tax", "total")
        for u in ("usdt", "inr", "frac")
    }
    | {"traded_notional_usdt", "turnover_annual", "exposure_time", "avg_gross_leverage"}
    | {"n_fills", "n_buys", "n_sells", "win_rate", "avg_trade_pnl_usdt"}
    | {"n_orders", "n_rejected", "n_canceled", "n_skipped"}
    | {"killed", "liquidated"}
    | {
        "benchmark_total_return",
        "benchmark_cagr",
        "benchmark_sharpe",
        "benchmark_max_drawdown",
        "benchmark_correlation",
    }
)


def test_key_set_and_types() -> None:
    m = compute_metrics(_result(), "1d", 1000.0, 85.0)
    assert set(m) == _EXPECTED_KEYS
    assert set(BENCHMARK_KEYS) == {k for k in _EXPECTED_KEYS if k.startswith("benchmark_")}
    assert all(math.isnan(m[k]) for k in BENCHMARK_KEYS)
    assert all(type(v) is float for v in m.values())


def test_killed() -> None:
    m = compute_metrics(_result(killed_at=D1 + timedelta(days=2)), "1d", 1000.0, 85.0)
    assert m["killed"] == 1.0


def test_purity_and_pickle() -> None:
    r = _result()
    a = compute_metrics(r, "1d", 1000.0, 85.0)
    b = compute_metrics(r, "1d", 1000.0, 85.0)
    assert a.keys() == b.keys()
    assert all(_eq(a[k], b[k]) for k in a)
    s = curve_stats(days(5), [100, 110, 99, 108.9, 119.79], "1d")
    t = pickle.loads(pickle.dumps(s))
    assert isinstance(t, CurveStats)
    assert t == s


def test_timeframe_type_is_literal() -> None:
    tf: Timeframe = "4h"
    assert tf in PERIODS_PER_YEAR
