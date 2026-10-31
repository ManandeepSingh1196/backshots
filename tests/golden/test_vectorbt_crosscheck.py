"""M4 exit check: buy_hold and ma_cross through run_backtest match vectorbt (ticket T035).

Zero fees, slippage and tax. vectorbt is imported lazily (about 6 s) and offline.
"""

import ast
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from bt.analytics.benchmark import buy_hold_curve
from bt.core.rounding import floor_to_step
from bt.data.catalog import Catalog
from bt.engine.backtest import run_backtest
from bt.runs.config import load_config
from bt.runs.reader import RunData, read_run
from tests.support.make_crosscheck_fixture import CROSSCHECK_DIR

EQUITY0 = 100000.0
ALLOC = 0.995
STEP = 0.001
FAST, SLOW = 5, 20
RTOL = 1e-3
ROOT = Path(__file__).resolve().parents[2]


def bars() -> pl.DataFrame:
    return Catalog(CROSSCHECK_DIR).read_klines("spot", "1d", "BTCUSDT").sort("ts")


def qty_for(equity: float, price: float) -> float:
    return floor_to_step(equity * ALLOC / price, STEP)


def expected_orders(strategy: str, open_: list[float], close: list[float]) -> dict[int, float]:
    """Decision-bar index -> signed size (+buy, -sell), from the fixture alone."""
    n = len(close)
    if strategy == "buy_hold":
        return {0: qty_for(EQUITY0, close[0])}
    cash, pos = EQUITY0, 0.0
    pending: tuple[int, float] | None = None
    out: dict[int, float] = {}
    for t in range(n):
        if pending is not None:  # decided on t-1, fills at this bar's open
            cash -= pending[1] * open_[t]
            pos += pending[1]
            pending = None
        if t < SLOW - 1:
            continue
        fast = float(np.mean(close[t - FAST + 1 : t + 1]))
        slow = float(np.mean(close[t - SLOW + 1 : t + 1]))
        long = fast > slow
        size = 0.0
        if long and pos == 0.0:
            size = qty_for(cash + pos * close[t], close[t])
        elif not long and pos > 0.0:
            size = -pos
        if size != 0.0 and t + 1 < n:
            out[t] = size
            pending = (t, size)
    return out


def run_ours(strategy: str, tmp: Path) -> RunData:
    params = ["params.fast=5", "params.slow=20"] if strategy == "ma_cross" else []
    last = bars()["ts"][-1]
    end = (last + (last - bars()["ts"][-2])).strftime("%Y-%m-%d")
    cfg = load_config(
        None,
        [
            f"run.strategy={strategy}",
            "run.start=2023-01-01",
            f"run.end={end}",
            "run.initial_equity_inr=100000",
            "run.inr_per_usdt=1",
            "run.close_at_end=false",
            "costs.slippage.bps=0",
            'costs.tax={model="none"}',
            'universe.symbols=["BTCUSDT"]',
            *params,
        ],
    )
    return read_run(run_backtest(cfg, data_dir=CROSSCHECK_DIR, runs_dir=tmp).run_dir)


@dataclass(frozen=True)
class VbtResult:
    value: np.ndarray[Any, Any]
    n_orders: int
    total_return: float


def vbt_portfolio(orders: dict[int, float], shift: int) -> VbtResult:
    import pandas as pd  # pyright: ignore[reportMissingTypeStubs]
    import vectorbt as vbt  # pyright: ignore[reportMissingTypeStubs]  # lazy: slow import

    df = bars()
    idx = pd.DatetimeIndex(df["ts"].to_list())
    open_ = pd.Series(df["open"].to_list(), index=idx)
    close = pd.Series(df["close"].to_list(), index=idx)
    size = pd.Series(np.nan, index=idx)
    for t, s in orders.items():
        size.iloc[t + shift] = s
    pf: Any = vbt.Portfolio.from_orders(  # pyright: ignore[reportUnknownMemberType]
        close=close,
        price=open_,
        size=size,
        size_type="amount",
        fees=0.0,
        slippage=0.0,
        init_cash=EQUITY0,
        direction="longonly",
    )
    return VbtResult(
        value=np.asarray(pf.value(), dtype=float),
        n_orders=int(pf.orders.count()),
        total_return=float(pf.total_return()),
    )


def max_rel_diff(ours: np.ndarray[Any, Any], theirs: np.ndarray[Any, Any]) -> float:
    return float(np.max(np.abs(ours - theirs) / np.abs(theirs)))


@pytest.mark.parametrize("strategy", ["buy_hold", "ma_cross"])
def test_matches_vectorbt(strategy: str, tmp_path: Path) -> None:
    df = bars()
    open_, close = df["open"].to_list(), df["close"].to_list()
    orders = expected_orders(strategy, open_, close)
    run = run_ours(strategy, tmp_path)
    pf = vbt_portfolio(orders, shift=1)

    n_ours, n_theirs = run.trades.height, pf.n_orders
    assert n_ours == n_theirs, f"trade count: ours {n_ours}, vectorbt {n_theirs}"
    if strategy == "buy_hold":
        assert n_ours == 1
    else:
        assert n_ours >= 6

    ours = run.equity["equity_after_tax"].to_numpy()
    theirs = pf.value
    assert len(ours) == len(theirs) == 400
    diff = max_rel_diff(ours, theirs)
    assert abs(ours[-1] - theirs[-1]) / theirs[-1] <= RTOL, f"final: max rel diff {diff:.3e}"
    assert np.allclose(ours, theirs, rtol=RTOL, atol=0), f"max rel diff {diff:.3e}"
    print(f"{strategy}: max relative difference {diff:.3e}")

    bench = pf.total_return if strategy == "buy_hold" else None
    if bench is not None:
        assert run.metrics["benchmark_total_return"] == pytest.approx(bench, abs=1e-3)
    else:
        bh = vbt_portfolio(expected_orders("buy_hold", open_, close), shift=1)
        assert run.metrics["benchmark_total_return"] == pytest.approx(bh.total_return, abs=1e-3)


def test_buy_hold_matches_benchmark_curve(tmp_path: Path) -> None:
    df = bars()
    inst = {i.symbol: i for i in Catalog(CROSSCHECK_DIR).read_instruments()[0]}["BTCUSDT"]
    ref = buy_hold_curve(df["open"].to_list(), df["close"].to_list(), inst, EQUITY0, 0.0, False)
    ours = run_ours("buy_hold", tmp_path).equity["equity_after_tax"].to_numpy()
    assert np.allclose(ours, np.asarray(ref), rtol=1e-9, atol=0)


def test_sensitivity_zero_bar_shift_is_detected() -> None:
    df = bars()
    orders = expected_orders("ma_cross", df["open"].to_list(), df["close"].to_list())
    right = vbt_portfolio(orders, shift=1).value
    wrong = vbt_portfolio(orders, shift=0).value
    assert max_rel_diff(wrong, right) > RTOL


def test_collecting_module_does_not_import_vectorbt() -> None:
    code = (
        "import sys, tests.golden.test_vectorbt_crosscheck;"
        "sys.exit(1 if 'vectorbt' in sys.modules else 0)"
    )
    res = subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=False)
    assert res.returncode == 0
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    top = [n for n in tree.body if isinstance(n, ast.Import | ast.ImportFrom)]
    assert not any("vectorbt" in ast.dump(n) for n in top)


def test_src_and_strategies_do_not_mention_vectorbt() -> None:
    for d in ("src", "strategies"):
        for p in (ROOT / d).rglob("*.py"):
            assert "vectorbt" not in p.read_text(encoding="utf-8"), p
