"""Terminal report: header, metrics, costs, activity, yearly returns, ASCII equity, warnings."""

import json
import math
import re
import textwrap
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Final

import plotext  # pyright: ignore[reportMissingTypeStubs]
import polars as pl

from bt.analytics.metrics import yearly_returns
from bt.data.validate import TIMEFRAME_DELTA
from bt.runs.config import RunConfig
from bt.runs.meta import RunMeta

SURVIVORSHIP_NOTE: Final = (
    "Survivorship bias: fixed universe of today's majors; dead coins excluded."
)
SKIP_WARN_FRAC: Final = 0.05
WARNING_PREFIX: Final = "WARNING: "

_MIN_WIDTH: Final = 80
_CHART_HEIGHT: Final = 12
_NA: Final = "n/a"
_NO_POINTS: Final = "(not enough points to plot)"
_ANSI: Final = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_DAY: Final = timedelta(days=1)


def _fin(x: float) -> bool:
    return math.isfinite(x)


def _pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f}%" if _fin(x) else _NA


def _num(x: float, fmt: str) -> str:
    return format(x, fmt) if _fin(x) else _NA


def _inr(x: float) -> str:
    return f"₹{x:,.2f}" if _fin(x) else _NA


def _usdt(x: float) -> str:
    return f"{x:,.2f}" if _fin(x) else _NA


def _count(x: float) -> str:
    return str(int(x)) if _fin(x) else _NA


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Fixed-width table: first column left-aligned, the rest right-aligned."""
    grid = [list(header), *[list(r) for r in rows]]
    widths = [max(len(r[i]) for r in grid) for i in range(len(header))]

    def fmt(r: Sequence[str]) -> str:
        cells = [r[0].ljust(widths[0])] + [
            c.rjust(w) for c, w in zip(r[1:], widths[1:], strict=True)
        ]
        return " | ".join(cells).rstrip()

    rule = "-+-".join("-" * w for w in widths)
    return [fmt(grid[0]), rule, *[fmt(r) for r in grid[1:]]]


def _wrap(text: str, width: int) -> list[str]:
    return [ln.rstrip() for ln in textwrap.wrap(text, width=width, subsequent_indent="  ")]


def _ascii_equity(values: Sequence[float], width: int, height: int = _CHART_HEIGHT) -> str:
    if len(values) < 2:
        return _NO_POINTS
    f: Any = plotext.figure  # plotext ships no type hints
    try:
        f.clear()
        f.plot_size(width, height)
        f.draw(f.signal(list(values)).lines())
        raw = str(f.build().string(colorless=True))
    finally:
        f.clear()
    text = _ANSI.sub("", raw)
    return "\n".join(ln.rstrip() for ln in text.splitlines())


def _header(
    config: RunConfig, metrics: Mapping[str, float], meta: RunMeta, width: int
) -> list[str]:
    run = config.run
    params = ", ".join(f"{k}={json.dumps(v)}" for k, v in config.params.items()) or "(none)"
    commit = (meta.git_commit or "unknown")[:10] + (" (dirty)" if meta.git_dirty else "")
    cost = config.costs
    out = [f"backshots run {meta.run_id}"]
    out += _wrap(f"strategy {run.strategy}  params {params}", width)
    out += _wrap(
        f"window {run.start} → {run.end} ({_count(metrics['n_bars'])} bars, {run.timeframe})",
        width,
    )
    out += _wrap(f"universe {', '.join(config.universe.symbols)} ({config.universe.kind})", width)
    out += _wrap(
        f"capital ₹{run.initial_equity_inr:,.2f} = {_usdt(metrics['initial_equity_usdt'])} USDT"
        f" @ {run.inr_per_usdt:g} INR/USDT",
        width,
    )
    out += _wrap(
        f"costs fees={cost.fees}  slippage={cost.slippage.model} {cost.slippage.bps:g}"
        f"  tax={cost.tax.model}",
        width,
    )
    out.append(f"code {commit}  hash {meta.hash8}")
    out.append(SURVIVORSHIP_NOTE)
    return out


def _performance(config: RunConfig, m: Mapping[str, float]) -> list[str]:
    bench = {
        "total_return": m["benchmark_total_return"],
        "cagr": m["benchmark_cagr"],
        "sharpe": m["benchmark_sharpe"],
        "max_drawdown": m["benchmark_max_drawdown"],
    }

    def row(label: str, key: str, fmt: str, bkey: str | None = None) -> list[str]:
        cells: list[str] = []
        for sfx in ("pre_tax", "after_tax"):
            v = m[f"{key}_{sfx}"]
            if fmt == "pct":
                cells.append(_pct(v))
            elif fmt == "inr":
                cells.append(_inr(v))
            else:
                cells.append(_num(v, fmt))
        b = ""
        if bkey is not None:
            b = _pct(bench[bkey]) if fmt == "pct" else _num(bench[bkey], fmt)
        return [label, *cells, b]

    rows = [
        row("Total return", "total_return", "pct", "total_return"),
        row("CAGR", "cagr", "pct", "cagr"),
        row("Volatility (ann.)", "vol_annual", "pct"),
        row("Sharpe", "sharpe", ".2f", "sharpe"),
        row("Sortino", "sortino", ".2f"),
        row("Max drawdown", "max_drawdown", "pct", "max_drawdown"),
        row("Max DD length (days)", "max_drawdown_days", ".0f"),
        row("Calmar", "calmar", ".2f"),
        row("Final equity ₹", "final_equity_inr", "inr"),
    ]
    out = ["Performance"]
    out += _table(["metric", "pre-tax", "after-tax", "BTC buy&hold"], rows)
    out.append(f"BTC correlation: {_num(m['benchmark_correlation'], '.2f')}")
    return out


def _costs(config: RunConfig, m: Mapping[str, float]) -> list[str]:
    rows = [
        [
            label,
            _inr(m[f"cost_{k}_inr"]),
            _usdt(m[f"cost_{k}_usdt"]),
            _pct(m[f"cost_{k}_frac"], 3),
        ]
        for label, k in (
            ("Fees", "fees"),
            ("Slippage", "slippage"),
            ("Funding", "funding"),
            ("Tax", "tax"),
            ("Total", "total"),
        )
    ]
    out = ["Costs"]
    out += _table(["cost", "₹", "USDT", "% of initial"], rows)
    if config.costs.tax.model == "none":
        out.append("Tax model: none — after-tax equals pre-tax.")
    return out


def _activity(m: Mapping[str, float], width: int) -> list[str]:
    out = ["Activity"]
    lines = [
        f"fills {_count(m['n_fills'])} (buys {_count(m['n_buys'])}, sells {_count(m['n_sells'])})"
        f"  win rate {_pct(m['win_rate'])}  avg trade pnl USDT {_usdt(m['avg_trade_pnl_usdt'])}",
        f"traded notional USDT {_usdt(m['traded_notional_usdt'])}"
        f"  turnover (ann.) {_num(m['turnover_annual'], '.2f')}"
        f"  time exposed {_pct(m['exposure_time'])}"
        f"  avg gross leverage {_num(m['avg_gross_leverage'], '.2f')}",
        f"orders {_count(m['n_orders'])}  rejected {_count(m['n_rejected'])}"
        f"  canceled {_count(m['n_canceled'])}  skipped {_count(m['n_skipped'])}",
    ]
    for ln in lines:
        out += _wrap(ln, width)
    return out


def _yearly(ts: list[datetime], pre: list[float], after: list[float]) -> list[str]:
    a = dict(yearly_returns(ts, pre))
    b = dict(yearly_returns(ts, after))
    rows = [[str(y), _pct(a[y]), _pct(b.get(y, math.nan))] for y in sorted(a)]
    return ["Yearly returns", *_table(["year", "pre-tax", "after-tax"], rows)]


def _chart(ts: list[datetime], after_inr: list[float], width: int) -> list[str]:
    out = ["Equity curve (after tax, ₹)"]
    chart = _ascii_equity(after_inr, width)
    out += chart.split("\n")
    first, last = ts[0].strftime("%Y-%m-%d"), ts[-1].strftime("%Y-%m-%d")
    out.append((first + " " * max(width - len(first) - len(last), 1) + last).rstrip())
    return out


def _stamp(t: datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M")


def _short_stamp(t: datetime) -> str:
    return t.strftime("%Y-%m-%d") if t.time() == time(0) else _stamp(t)


def _midnight(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def _warnings(
    config: RunConfig, m: Mapping[str, float], ts: list[datetime], meta: RunMeta, width: int
) -> list[str]:
    msgs: list[str] = []
    if m["liquidated"] == 1.0:
        msgs.append("liquidation occurred; the run counts as failed.")
    if meta.killed_at is not None:
        msgs.append(
            f"drawdown kill switch fired at {_stamp(meta.killed_at)}; positions were flattened."
        )
    r = m["n_rejected"] + m["n_skipped"]
    d = m["n_orders"] + m["n_skipped"]
    if r > 0 and r / max(d, 1) >= SKIP_WARN_FRAC:
        msgs.append(
            f"{_count(r)} of {_count(d)} orders rejected or skipped ({_pct(r / max(d, 1))})."
        )
    for cov in sorted(meta.coverage, key=lambda c: c.symbol):
        if cov.missing > 0:
            msgs.append(
                f"{cov.symbol} has {cov.missing} missing bar(s) in the window; "
                "its orders wait and its mark carries forward."
            )
    run = config.run
    first_ok = _midnight(run.start)
    last_ok = _midnight(run.end) + _DAY - TIMEFRAME_DELTA[run.timeframe]
    if ts and (ts[0] != first_ok or ts[-1] < last_ok):
        msgs.append(
            f"data covers {_short_stamp(ts[0])} → {_short_stamp(ts[-1])}, "
            f"not the configured window {run.start} → {run.end}."
        )
    if meta.git_dirty is True:
        msgs.append("working tree was dirty; this run is not reproducible from the commit alone.")
    out = ["Warnings"]
    if not msgs:
        return [*out, "No warnings."]
    for msg in msgs:
        out += _wrap(WARNING_PREFIX + msg, width)
    return out


def render_report(
    config: RunConfig,
    metrics: Mapping[str, float],
    equity: pl.DataFrame,
    meta: RunMeta,
    *,
    width: int = 100,
) -> str:
    """Render the DESIGN section 9 report as deterministic plain text (no ANSI, no clock)."""
    if width < _MIN_WIDTH:
        raise ValueError(f"width must be >= {_MIN_WIDTH}, got {width}")
    ts: list[datetime] = equity["ts"].to_list()
    pre: list[float] = equity["equity_pre_tax"].to_list()
    after: list[float] = equity["equity_after_tax"].to_list()
    inr = config.run.inr_per_usdt
    sections = [
        _header(config, metrics, meta, width),
        _performance(config, metrics),
        _costs(config, metrics),
        _activity(metrics, width),
        _yearly(ts, pre, after),
        _chart(ts, [v * inr for v in after], width),
        _warnings(config, metrics, ts, meta, width),
    ]
    lines: list[str] = []
    for i, sec in enumerate(sections):
        if i:
            lines.append("")
        lines += [ln.rstrip() for ln in sec]
    return "\n".join(lines) + "\n"
