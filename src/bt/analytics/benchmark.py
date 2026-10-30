"""BTC buy-and-hold benchmark (INTERFACES section 15.4): same timing and costs as buy_hold."""

import math
from collections.abc import Sequence
from datetime import datetime
from itertools import pairwise
from typing import Final

from bt.analytics.metrics import curve_stats
from bt.core.clock import ensure_utc
from bt.core.errors import InvariantError, TimezoneError
from bt.core.rounding import floor_to_step
from bt.core.types import Instrument
from bt.data.validate import Timeframe

BENCHMARK_SYMBOL: Final = "BTCUSDT"
BENCHMARK_ALLOC: Final = 0.995  # equals BuyHold.Params.alloc default

_ZERO_VAR_REL_TOL: Final = 1e-12
_NAN: Final = float("nan")


def _finite_positive(name: str, values: Sequence[float]) -> list[float]:
    out = [float(v) for v in values]
    if not all(math.isfinite(v) and v > 0 for v in out):
        raise InvariantError(f"buy_hold_curve: {name} must be finite and > 0")
    return out


def buy_hold_curve(
    open_: Sequence[float],
    close: Sequence[float],
    inst: Instrument,
    initial_equity: float,
    slippage_frac: float,
    close_at_end: bool,
    alloc: float = BENCHMARK_ALLOC,
) -> list[float]:
    """One equity value per bar for buying at bar 1's open and holding.

    Invariants: row 0 is initial_equity; cash never goes negative (the fill quantity is
    clamped to what cash affords); bad input raises InvariantError.
    """
    n = len(close)
    if n == 0 or len(open_) != n:
        raise InvariantError(f"buy_hold_curve: need equal non-empty bars, got {len(open_)}/{n}")
    o = _finite_positive("open", open_)
    c = _finite_positive("close", close)
    e0, s, a = float(initial_equity), float(slippage_frac), float(alloc)
    if isinstance(initial_equity, bool) or not (math.isfinite(e0) and e0 > 0):
        raise InvariantError(f"buy_hold_curve: initial_equity must be > 0, got {initial_equity!r}")
    if not (math.isfinite(s) and 0 <= s < 1):
        raise InvariantError(f"buy_hold_curve: slippage_frac must be in [0, 1), got {s!r}")
    if not (math.isfinite(a) and 0 < a <= 1):
        raise InvariantError(f"buy_hold_curve: alloc must be in (0, 1], got {a!r}")
    if n == 1:
        return [e0]

    fee, step = inst.taker_fee, inst.qty_step
    q_plan = floor_to_step(e0 * a / (c[0] * (1 + max(inst.maker_fee, fee, 0.0))), step)
    p = o[1] * (1 + s)
    q = min(q_plan, floor_to_step(e0 / (p * (1 + fee)), step))
    cash = e0 - q * p * (1 + fee)
    curve = [e0] + [cash + q * c[i] for i in range(1, n)]
    if close_at_end:
        curve[-1] = cash + q * c[-1] * (1 - s) * (1 - fee)
    return [float(v) for v in curve]


def _correlation(a: Sequence[float], b: Sequence[float]) -> float:
    ra = [y / x - 1.0 for x, y in pairwise(a)] if min(a) > 0 else []
    rb = [y / x - 1.0 for x, y in pairwise(b)] if min(b) > 0 else []
    n = len(ra)
    if n < 2 or len(rb) != n:
        return _NAN
    ma, mb = math.fsum(ra) / n, math.fsum(rb) / n
    va = math.fsum((x - ma) ** 2 for x in ra) / (n - 1)
    vb = math.fsum((x - mb) ** 2 for x in rb) / (n - 1)
    if va <= _ZERO_VAR_REL_TOL * max(ma * ma, 1e-300) or vb <= _ZERO_VAR_REL_TOL * max(
        mb * mb, 1e-300
    ):
        return _NAN
    cov = math.fsum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True)) / (n - 1)
    return float(max(-1.0, min(1.0, cov / math.sqrt(va * vb))))


def benchmark_metrics(
    run_ts: Sequence[datetime],
    run_equity: Sequence[float],
    bench_ts: Sequence[datetime],
    bench_equity: Sequence[float],
    timeframe: Timeframe,
) -> dict[str, float]:
    """Return exactly BENCHMARK_KEYS as floats; correlation uses the ts common to both curves."""
    if len(run_ts) != len(run_equity):
        raise InvariantError(
            f"run ts and equity lengths differ: {len(run_ts)} vs {len(run_equity)}"
        )
    stats = curve_stats(bench_ts, bench_equity, timeframe)
    try:
        run_stamps = [ensure_utc(t, "benchmark run ts") for t in run_ts]
        bench_stamps = [ensure_utc(t, "benchmark ts") for t in bench_ts]
    except TimezoneError as exc:
        raise InvariantError(str(exc)) from exc
    run_by_ts = {t: float(e) for t, e in zip(run_stamps, run_equity, strict=True)}
    bench_by_ts = {t: float(e) for t, e in zip(bench_stamps, bench_equity, strict=True)}
    shared = sorted(run_by_ts.keys() & bench_by_ts.keys())
    corr = _correlation([run_by_ts[t] for t in shared], [bench_by_ts[t] for t in shared])
    out = {
        "benchmark_total_return": stats.total_return,
        "benchmark_cagr": stats.cagr,
        "benchmark_sharpe": stats.sharpe,
        "benchmark_max_drawdown": stats.max_drawdown,
        "benchmark_correlation": corr,
    }
    return out
