"""Tests for bt.analytics.report."""

import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from pydantic import BaseModel

from bt.analytics.metrics import BENCHMARK_KEYS
from bt.analytics.report import (
    SURVIVORSHIP_NOTE,
    WARNING_PREFIX,
    _ascii_equity,  # pyright: ignore[reportPrivateUsage]
    render_report,
)
from bt.runs.config import RunConfig
from bt.runs.fingerprint import SymbolCoverage
from bt.runs.meta import RunMeta
from bt.strategy.base import Intent, NoAction, Strategy
from bt.strategy.context import Context
from bt.strategy.registry import register

D1 = datetime(2024, 1, 1, tzinfo=UTC)


@register("rpttest")
class RptTest(Strategy):
    class Params(BaseModel):
        a: int = 1

    def on_bar(self, ctx: Context) -> Intent:
        return NoAction()


def make_config(end: str = "2024-01-05") -> RunConfig:
    return RunConfig.model_validate(
        {
            "run": {
                "strategy": "rpttest",
                "start": "2024-01-01",
                "end": end,
                "initial_equity_inr": 85000,
                "inr_per_usdt": 85,
            },
            "universe": {"symbols": ["AAAUSDT", "BBBUSDT"]},
            "params": {"a": 3},
            "costs": {"slippage": {"bps": 10}},
        }
    )


def make_equity() -> pl.DataFrame:
    eq = [1000.0, 1028.97949, 1038.97949, 1048.97949, 1057.84006]
    n = len(eq)
    ts = [D1 + timedelta(days=i) for i in range(n)]
    z = [0.0] * n
    return pl.DataFrame(
        {
            "ts": ts,
            "close_ts": [t + timedelta(days=1) for t in ts],
            "equity_pre_tax": eq,
            "equity_after_tax": eq,
            "cash": eq,
            "gross_exposure": z,
            "net_exposure": z,
            "drawdown": z,
            "realized_pnl": z,
            "fees_paid": z,
            "slippage_paid": z,
            "funding_paid": z,
            "tax_paid": z,
        }
    ).with_columns(pl.col("ts", "close_ts").dt.cast_time_unit("us").dt.replace_time_zone("UTC"))


def make_metrics() -> dict[str, float]:
    m: dict[str, float] = {}
    for sfx in ("pre_tax", "after_tax"):
        m.update(
            {
                f"total_return_{sfx}": 0.05784006,
                f"cagr_{sfx}": 1.5,
                f"vol_annual_{sfx}": 0.2,
                f"sharpe_{sfx}": 2.0,
                f"sortino_{sfx}": 3.0,
                f"max_drawdown_{sfx}": -0.01,
                f"max_drawdown_days_{sfx}": 2.0,
                f"calmar_{sfx}": 4.0,
                f"final_equity_inr_{sfx}": 89916.4051,
            }
        )
    m.update(
        {
            "initial_equity_usdt": 1000.0,
            "initial_equity_inr": 85000.0,
            "n_bars": 5.0,
            "cost_fees_inr": 91.7949,
            "cost_fees_usdt": 1.07994,
            "cost_fees_frac": 0.00107994,
            "cost_slippage_inr": 91.8,
            "cost_slippage_usdt": 1.08,
            "cost_slippage_frac": 0.00108,
            "cost_funding_inr": 0.0,
            "cost_funding_usdt": 0.0,
            "cost_funding_frac": 0.0,
            "cost_tax_inr": 0.0,
            "cost_tax_usdt": 0.0,
            "cost_tax_frac": 0.0,
            "cost_total_inr": 183.5949,
            "cost_total_usdt": 2.15994,
            "cost_total_frac": 0.00215994,
            "traded_notional_usdt": 2000.0,
            "turnover_annual": 100.0,
            "exposure_time": 0.5,
            "avg_gross_leverage": 0.5,
            "n_fills": 2.0,
            "n_buys": 1.0,
            "n_sells": 1.0,
            "win_rate": 1.0,
            "avg_trade_pnl_usdt": 57.8,
            "n_orders": 10.0,
            "n_rejected": 0.0,
            "n_canceled": 0.0,
            "n_skipped": 0.0,
            "liquidated": 0.0,
        }
    )
    for k in BENCHMARK_KEYS:
        m[k] = math.nan
    return m


def make_meta(**kw: object) -> RunMeta:
    base = RunMeta(
        run_id="20240101-rpttest-abcd1234",
        strategy="rpttest",
        hash8="abcd1234",
        config_sha256="a" * 64,
        data_sha256="b" * 64,
        git_commit="0123456789abcdef",
        git_dirty=False,
        versions=(("python", "3.12"),),
        started_at=D1,
        finished_at=D1,
        coverage=(
            SymbolCoverage("AAAUSDT", 5, 0, D1, D1 + timedelta(days=4)),
            SymbolCoverage("BBBUSDT", 4, 1, D1, D1 + timedelta(days=4)),
        ),
        killed_at=None,
        liquidated=False,
    )
    return replace(base, **kw)  # pyright: ignore[reportArgumentType]


def render(
    m: dict[str, float] | None = None,
    meta: RunMeta | None = None,
    config: RunConfig | None = None,
    equity: pl.DataFrame | None = None,
    width: int = 100,
) -> str:
    return render_report(
        config or make_config(),
        m or make_metrics(),
        make_equity() if equity is None else equity,
        meta or make_meta(),
        width=width,
    )


def warning_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.startswith(WARNING_PREFIX)]


def test_contents() -> None:
    text = render()
    lines = text.splitlines()
    assert sum("5.78%" in ln for ln in lines if ln.startswith("Total return")) == 1
    total = next(ln for ln in lines if ln.startswith("Total return"))
    assert total.count("5.78%") == 2
    for s in ("₹91.79", "1.08", "0.108%", "₹89,916.41", "n/a", SURVIVORSHIP_NOTE):
        assert s in text
    assert "Tax model: none — after-tax equals pre-tax." in lines
    assert "capital ₹85,000.00 = 1,000.00 USDT @ 85 INR/USDT" in lines
    assert "strategy rpttest  params a=3" in lines
    assert "code 0123456789  hash abcd1234" in lines


def test_section_order() -> None:
    text = render()
    titles = [
        "backshots run ",
        "Performance",
        "Costs",
        "Activity",
        "Yearly returns",
        "Equity curve (after tax, ₹)",
        "Warnings",
    ]
    pos = [text.index("\n" + t if i else t) for i, t in enumerate(titles)]
    assert pos == sorted(pos)


def test_cost_table_rows() -> None:
    lines = render().splitlines()
    for label in ("Fees", "Slippage", "Funding", "Tax", "Total"):
        assert sum(ln.startswith(label) for ln in lines) >= 1
    for label in ("Funding", "Tax"):
        row = next(ln for ln in lines if ln.startswith(label + " "))
        assert "₹0.00" in row
        assert "0.000%" in row


def test_yearly_one_row() -> None:
    lines = render().splitlines()
    i = lines.index("Yearly returns")
    block = lines[i + 1 : lines.index("", i)]
    rows = block[2:]
    assert len(rows) == 1
    assert rows[0].startswith("2024")
    assert rows[0].count("5.78%") == 2


def test_default_warning_is_missing_bar_only() -> None:
    w = warning_lines(render())
    assert len(w) == 1
    assert w[0].startswith("WARNING: BBBUSDT has 1 missing bar(s) in the window; ")


def test_all_warning_kinds_in_order() -> None:
    m = make_metrics()
    m["liquidated"] = 1.0
    m["n_rejected"] = 1.0
    meta = make_meta(killed_at=D1 + timedelta(days=2), git_dirty=True)
    text = render(m, meta)
    w = warning_lines(text)
    assert len(w) == 5
    assert w[0] == "WARNING: liquidation occurred; the run counts as failed."
    assert w[1] == (
        "WARNING: drawdown kill switch fired at 2024-01-03 00:00; positions were flattened."
    )
    assert "1 of 10 orders rejected or skipped (10.00%)" in w[2]
    assert "BBBUSDT" in w[3]
    assert "dirty" in w[4]


def test_no_warnings() -> None:
    meta = make_meta(coverage=())
    text = render(meta=meta)
    assert warning_lines(text) == []
    assert text.splitlines()[-1] == "No warnings."


def test_reject_fraction_below_threshold() -> None:
    m = make_metrics()
    m["n_rejected"] = 1.0
    m["n_orders"] = 40.0
    assert not any("rejected" in w for w in warning_lines(render(m, make_meta(coverage=()))))


def test_skipped_counts_in_denominator() -> None:
    m = make_metrics()
    m["n_skipped"] = 2.0
    w = warning_lines(render(m, make_meta(coverage=())))
    assert w == ["WARNING: 2 of 12 orders rejected or skipped (16.67%)."]


def test_window_warning() -> None:
    w = warning_lines(render(meta=make_meta(coverage=()), config=make_config(end="2024-01-06")))
    assert w == [
        "WARNING: data covers 2024-01-01 → 2024-01-05, "
        "not the configured window 2024-01-01 → 2024-01-06."
    ]


def test_window_warning_late_start() -> None:
    eq = make_equity().slice(1)
    w = warning_lines(render(meta=make_meta(coverage=()), equity=eq))
    assert len(w) == 1
    assert "data covers 2024-01-02 →" in w[0]


@pytest.mark.parametrize("width", [80, 100, 120])
def test_line_rules(width: int) -> None:
    m = make_metrics()
    m["liquidated"] = 1.0
    meta = make_meta(killed_at=D1, git_dirty=True)
    text = render(m, meta, width=width)
    assert "\x1b" not in text
    assert text.endswith("\n")
    for ln in text.splitlines():
        assert len(ln) <= width
        assert ln == ln.rstrip()


def test_width_too_small() -> None:
    with pytest.raises(ValueError, match="width"):
        render(width=79)


def test_ascii_chart() -> None:
    out = _ascii_equity([100, 110, 99, 108.9, 119.79], 60)
    lines = out.splitlines()
    assert len(lines) == 12
    assert all(len(ln) <= 60 for ln in lines)
    assert "119.8" in out
    assert out == _ascii_equity([100, 110, 99, 108.9, 119.79], 60)


def test_ascii_chart_too_few_points() -> None:
    assert _ascii_equity([1.0], 60) == "(not enough points to plot)"
    assert _ascii_equity([], 60) == "(not enough points to plot)"


def test_ascii_chart_leaves_figure_cleared() -> None:
    _ascii_equity([100, 110, 99, 108.9, 119.79], 60)
    out = _ascii_equity([5000, 5010, 5020], 60)
    assert "119.8" not in out
    assert "5020" in out


def test_determinism_and_no_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    a = render()
    b = render()
    assert a == b
    assert capsys.readouterr().out == ""


def test_nan_renders_na() -> None:
    m = make_metrics()
    m["sharpe_after_tax"] = math.nan
    row = next(ln for ln in render(m).splitlines() if ln.startswith("Sharpe"))
    cells = [c.strip() for c in row.split("|")]
    assert cells[1] == "2.00"
    assert cells[2] == "n/a"
