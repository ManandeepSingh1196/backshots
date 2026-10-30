"""Backtest metrics: curve statistics, costs, activity and pre/after-tax keys.

Everything here is pure: no clock, no I/O, no global state.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from itertools import pairwise
from typing import Final

from bt.core.clock import ensure_utc
from bt.core.errors import InvariantError, TimezoneError
from bt.core.types import OrderStatus, Side
from bt.data.validate import Timeframe
from bt.engine.loop import EngineResult

PERIODS_PER_YEAR: Final[Mapping[Timeframe, int]] = {"1d": 365, "4h": 2190}
EXPOSURE_ABS_TOL: Final = 1e-6
BENCHMARK_KEYS: Final[tuple[str, ...]] = (
    "benchmark_total_return",
    "benchmark_cagr",
    "benchmark_sharpe",
    "benchmark_max_drawdown",
    "benchmark_correlation",
)

# Spread at or below this fraction of |mean return| is float noise, i.e. a constant return.
_ZERO_SPREAD_REL_TOL: Final = 1e-12
_NAN: Final = float("nan")
_DAY: Final = timedelta(days=1)
_DAYS_PER_YEAR: Final = 365.0
_TIMEFRAME_DAYS: Final[Mapping[Timeframe, float]] = {"1d": 1.0, "4h": 1.0 / 6.0}


@dataclass(frozen=True, slots=True)
class CurveStats:
    """Summary statistics of one equity curve.

    Invariants: max_drawdown >= 0 and max_drawdown_days >= 0; every field is a Python float;
    undefined values are NaN.
    """

    total_return: float
    cagr: float
    vol_annual: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_days: float
    calmar: float


def drawdown_series(equity: Sequence[float]) -> list[float]:
    """Return e / running_max - 1 for each row (always <= 0)."""
    out: list[float] = []
    peak = -math.inf
    for e in equity:
        peak = max(peak, e)
        out.append(float(e / peak - 1.0) if peak > 0 else 0.0)
    return out


def _validate(
    ts: Sequence[datetime], equity: Sequence[float]
) -> tuple[list[datetime], list[float]]:
    if len(ts) != len(equity):
        raise InvariantError(f"ts and equity lengths differ: {len(ts)} vs {len(equity)}")
    if len(equity) == 0:
        raise InvariantError("equity curve must have at least one row")
    try:
        stamps = [ensure_utc(t, "metrics ts") for t in ts]
    except TimezoneError as exc:
        raise InvariantError(str(exc)) from exc
    if any(b <= a for a, b in pairwise(stamps)):
        raise InvariantError("ts must be strictly increasing")
    values = [float(e) for e in equity]
    if not all(math.isfinite(e) for e in values):
        raise InvariantError("equity must be finite")
    if values[0] <= 0:
        raise InvariantError("equity[0] must be > 0")
    return stamps, values


def _returns(e: Sequence[float]) -> list[float]:
    out: list[float] = []
    for prev, cur in pairwise(e):
        out.append(cur / prev - 1.0)
        if cur <= 0:
            break
    return out


def _mean(x: Sequence[float]) -> float:
    return math.fsum(x) / len(x)


def _std(x: Sequence[float]) -> float:
    m = _mean(x)
    return math.sqrt(math.fsum((v - m) ** 2 for v in x) / (len(x) - 1))


def _max_dd_days(stamps: Sequence[datetime], e: Sequence[float]) -> float:
    best = timedelta(0)
    peak = e[0]
    peak_ts = stamps[0]
    underwater = False
    for i in range(1, len(e)):
        if e[i] >= peak:
            if underwater:
                best = max(best, stamps[i] - peak_ts)
                underwater = False
            peak = e[i]
            peak_ts = stamps[i]
        else:
            underwater = True
    if underwater:
        best = max(best, stamps[-1] - peak_ts)
    return best / _DAY


def curve_stats(
    ts: Sequence[datetime], equity: Sequence[float], timeframe: Timeframe
) -> CurveStats:
    """Compute CurveStats for one curve; raises InvariantError on malformed input."""
    stamps, e = _validate(ts, equity)
    p = float(PERIODS_PER_YEAR[timeframe])
    years = (len(e) - 1) * _TIMEFRAME_DAYS[timeframe] / _DAYS_PER_YEAR
    total_return = e[-1] / e[0] - 1.0
    if years == 0:
        cagr = _NAN
    elif e[-1] <= 0:
        cagr = -1.0
    else:
        cagr = (e[-1] / e[0]) ** (1.0 / years) - 1.0

    r = _returns(e)
    vol = sharpe = sortino = _NAN
    if len(r) >= 2:
        sd = _std(r)
        if sd <= _ZERO_SPREAD_REL_TOL * max(abs(_mean(r)), 1e-300):
            sd = 0.0
        vol = sd * math.sqrt(p)
        if sd != 0:
            sharpe = _mean(r) / sd * math.sqrt(p)
    if r:
        downside = math.sqrt(math.fsum(min(v, 0.0) ** 2 for v in r) / len(r))
        if downside <= _ZERO_SPREAD_REL_TOL * max(abs(_mean(r)), 1e-300):
            downside = 0.0
        if downside != 0:
            sortino = _mean(r) / downside * math.sqrt(p)

    max_dd = -min(drawdown_series(e))
    calmar = _NAN if max_dd == 0 or math.isnan(cagr) else cagr / max_dd
    return CurveStats(
        total_return=float(total_return),
        cagr=float(cagr),
        vol_annual=float(vol),
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_drawdown=float(max_dd),
        max_drawdown_days=float(_max_dd_days(stamps, e)),
        calmar=float(calmar),
    )


def yearly_returns(ts: Sequence[datetime], equity: Sequence[float]) -> list[tuple[int, float]]:
    """Calendar-year returns: last equity of the year over the last of the previous year."""
    stamps, e = _validate(ts, equity)
    last: dict[int, float] = {}
    for t, v in zip(stamps, e, strict=True):
        last[t.year] = v
    out: list[tuple[int, float]] = []
    base = e[0]
    for year in sorted(last):
        out.append((year, last[year] / base - 1.0))
        base = last[year]
    return out


def _trade_stats(result: EngineResult) -> tuple[int, int, list[float]]:
    qty: dict[str, float] = {}
    avg: dict[str, float] = {}
    n_buys = 0
    pnls: list[float] = []
    for f in result.fills:
        q = qty.get(f.symbol, 0.0)
        if f.side == Side.BUY:
            n_buys += 1
            avg[f.symbol] = (q * avg.get(f.symbol, 0.0) + f.qty * f.price) / (q + f.qty)
            qty[f.symbol] = q + f.qty
        else:
            # A SELL with no prior position (avg=0) cannot happen on spot.
            pnls.append((f.price - avg.get(f.symbol, 0.0)) * f.qty)
            qty[f.symbol] = max(q - f.qty, 0.0)
    return n_buys, len(pnls), pnls


def compute_metrics(
    result: EngineResult, timeframe: Timeframe, initial_equity: float, inr_per_usdt: float
) -> dict[str, float]:
    """Return every DESIGN section 8 metric as a flat dict of floats (NaN when undefined)."""
    snaps = result.equity
    ts = [s.ts for s in snaps]
    after = [s.equity for s in snaps]
    pre = [s.equity + s.tax_paid for s in snaps]
    m: dict[str, float] = {}

    for sfx, curve in (("pre_tax", pre), ("after_tax", after)):
        st = curve_stats(ts, curve, timeframe)
        for fld in fields(st):
            m[f"{fld.name}_{sfx}"] = float(getattr(st, fld.name))
        m[f"final_equity_usdt_{sfx}"] = float(curve[-1])
        m[f"final_equity_inr_{sfx}"] = float(curve[-1] * inr_per_usdt)

    n = len(snaps)
    years = (n - 1) * _TIMEFRAME_DAYS[timeframe] / _DAYS_PER_YEAR
    m["initial_equity_usdt"] = float(initial_equity)
    m["initial_equity_inr"] = float(initial_equity * inr_per_usdt)
    m["inr_per_usdt"] = float(inr_per_usdt)
    m["n_bars"] = float(n)
    m["years"] = float(years)
    m["missing_bars"] = 0.0

    last = snaps[-1]
    lines = {
        "fees": last.fees_paid,
        "slippage": last.slippage_paid,
        "funding": last.funding_paid,
        "tax": last.tax_paid,
    }
    for name, usdt in lines.items():
        m[f"cost_{name}_usdt"] = float(usdt)
        m[f"cost_{name}_inr"] = float(usdt * inr_per_usdt)
        m[f"cost_{name}_frac"] = float(usdt / initial_equity)
    total = math.fsum(lines.values())
    m["cost_total_usdt"] = float(total)
    m["cost_total_inr"] = float(total * inr_per_usdt)
    m["cost_total_frac"] = float(total / initial_equity)

    notional = math.fsum(f.qty * f.price for f in result.fills)
    m["traded_notional_usdt"] = float(notional)
    m["turnover_annual"] = float(notional / _mean(after) / years) if years > 0 else _NAN
    m["exposure_time"] = float(sum(s.gross_exposure > EXPOSURE_ABS_TOL for s in snaps) / n)
    # Leverage is undefined at equity <= 0, so those rows are skipped.
    lev = [s.gross_exposure / s.equity for s in snaps if s.equity > 0]
    m["avg_gross_leverage"] = float(_mean(lev)) if lev else _NAN

    n_buys, n_sells, pnls = _trade_stats(result)
    m["n_fills"] = float(len(result.fills))
    m["n_buys"] = float(n_buys)
    m["n_sells"] = float(n_sells)
    m["win_rate"] = float(sum(p > 0 for p in pnls) / n_sells) if pnls else _NAN
    m["avg_trade_pnl_usdt"] = float(_mean(pnls)) if pnls else _NAN

    m["n_orders"] = float(len({r.order.id for r in result.orders}))
    m["n_rejected"] = float(sum(r.status == OrderStatus.REJECTED for r in result.orders))
    m["n_canceled"] = float(sum(r.status == OrderStatus.CANCELED for r in result.orders))
    m["n_skipped"] = float(len(result.skipped))
    m["killed"] = 1.0 if result.killed_at else 0.0
    m["liquidated"] = float(result.liquidated)

    for key in BENCHMARK_KEYS:  # M4: run_backtest overwrites via analytics/benchmark.py
        m[key] = _NAN
    return m
