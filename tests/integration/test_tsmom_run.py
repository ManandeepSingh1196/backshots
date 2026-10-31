"""TSMOM end to end on the committed crosscheck fixture (ticket T040)."""

import math
import shutil
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from bt.cli.main import app
from bt.data.catalog import Catalog
from bt.engine.backtest import run_backtest
from bt.runs.config import load_config
from bt.runs.reader import RunData, read_run
from bt.runs.writer import DETERMINISTIC_FILES, RUN_FILES
from tests.support.make_crosscheck_fixture import CROSSCHECK_DIR

runner = CliRunner()
SKIP_REASONS = {"zero_qty", "below_min_qty", "below_min_notional"}
BASE = [
    "run.start=2023-03-15",
    "run.end=2024-02-04",
    'universe.symbols=["BTCUSDT", "ETHUSDT"]',
    "universe.kind=spot",
    "params.lookbacks=[20,60]",
    "params.vol_window=20",
    "costs.slippage.model=fixed_bps",
]


def sets(*extra: str, bps: float = 5.0) -> list[str]:
    return [*BASE, f"costs.slippage.bps={bps}", *extra]


def cli(
    tmp: Path, *extra: str, bps: float = 5.0, data: Path = CROSSCHECK_DIR, tag: str = "r"
) -> RunData:
    args = ["run", "tsmom", "--data-dir", str(data), "--runs-dir", str(tmp / tag)]
    for s in sets(*extra, bps=bps):
        args += ["--set", s]
    res = runner.invoke(app, args)
    assert res.exit_code == 0, res.output
    (d,) = [p for p in (tmp / tag).iterdir() if not p.name.startswith(".")]
    return read_run(d)


def test_runs(tmp_path: Path) -> None:
    run = cli(tmp_path)
    for name in RUN_FILES:
        assert (run.run_dir / name).stat().st_size > 0
    assert run.metrics["n_fills"] > 0
    assert math.isfinite(run.metrics["benchmark_total_return"])
    assert run.metrics["killed"] == 0.0


def test_determinism(tmp_path: Path) -> None:
    a = load_config(None, ["run.strategy=tsmom", *sets()])
    r1 = run_backtest(a, data_dir=CROSSCHECK_DIR, runs_dir=tmp_path / "a")
    r2 = run_backtest(a, data_dir=CROSSCHECK_DIR, runs_dir=tmp_path / "b")
    for name in DETERMINISTIC_FILES:
        assert (r1.run_dir / name).read_bytes() == (r2.run_dir / name).read_bytes(), name


def test_no_cash_rejects(tmp_path: Path) -> None:
    run = cli(tmp_path)
    bad = run.orders.filter(
        pl.col("reason").is_in(["insufficient_cash_at_fill", "insufficient_cash"])
    )
    assert bad.height == 0
    assert run.metrics["n_rejected"] == 0.0


def test_decision_before_fill(tmp_path: Path) -> None:
    run = cli(tmp_path)
    last = datetime(2024, 2, 4, tzinfo=UTC)
    # close_at_end: engine end-of-run flatten, not a strategy decision; last bar only
    end = run.trades.filter(pl.col("tag") == "close_at_end")
    assert (end["ts"] == last).all()
    strat = run.trades.filter(pl.col("tag") != "close_at_end")
    assert strat.height > 0
    new = run.orders.filter(pl.col("status") == "new").select("order_id", pl.col("ts").alias("nts"))
    joined = strat.join(new, on="order_id", how="left")
    assert joined["nts"].null_count() == 0
    assert (joined["ts"] > joined["nts"]).all()


def test_band_works(tmp_path: Path) -> None:
    wide = cli(tmp_path, "engine.rebalance_band=0.02", tag="w")
    zero = cli(tmp_path, "engine.rebalance_band=0.0", tag="z")
    assert zero.metrics["n_fills"] > wide.metrics["n_fills"]


def test_costs_matter(tmp_path: Path) -> None:
    on = cli(tmp_path, bps=5.0, tag="on")
    off = cli(tmp_path, bps=0.0, tag="off")
    assert off.metrics["n_fills"] > 0
    assert off.metrics["cost_slippage_usdt"] == pytest.approx(0.0, abs=1e-12)
    assert on.metrics["cost_slippage_usdt"] > 0.0
    assert on.equity["equity_after_tax"][-1] < off.equity["equity_after_tax"][-1]
    text = (on.run_dir / "report.txt").read_text().lower()
    assert "before-tax" in text or "before tax" in text or "pre-tax" in text
    assert "after-tax" in text or "after tax" in text


def max_lev(run: RunData) -> float:
    return float((run.equity["gross_exposure"] / run.equity["equity_after_tax"]).max())  # pyright: ignore[reportArgumentType]


def test_vol_target_caps_exposure(tmp_path: Path) -> None:
    capped = cli(tmp_path, "params.vol_target=5.0", "params.max_leverage=0.3", tag="c")
    free = cli(tmp_path, "params.vol_target=5.0", "params.max_leverage=1.0", tag="f")
    drift = 0.05
    assert max_lev(capped) <= 0.3 + drift
    assert max_lev(free) > 0.3 + drift
    assert max_lev(capped) < max_lev(free)


def test_lookahead_canary(tmp_path: Path) -> None:
    day = datetime(2023, 9, 1, tzinfo=UTC)
    alt = tmp_path / "alt"
    shutil.copytree(CROSSCHECK_DIR, alt)
    cat = Catalog(alt)
    for sym in ("BTCUSDT", "ETHUSDT"):
        df = cat.read_klines("spot", "1d", sym)
        after = pl.col("ts") > day
        df = df.with_columns(
            [
                pl.when(after).then(pl.col(c) * 1.7).otherwise(pl.col(c)).alias(c)
                for c in ("open", "high", "low", "close")
            ]
        )
        cat.write_klines("spot", "1d", sym, df, ("canary",), {}, datetime(2024, 3, 1, tzinfo=UTC))
    base = cli(tmp_path, tag="base")
    mod = cli(tmp_path, data=alt, tag="mod")
    b = base.orders.filter(pl.col("ts") <= day)
    m = mod.orders.filter(pl.col("ts") <= day)
    assert b.height > 0
    assert b.equals(m)
    # sanity: the change is visible after day D
    assert not base.orders.equals(mod.orders)


def test_skip_reasons(tmp_path: Path) -> None:
    run = cli(tmp_path, "engine.rebalance_band=0.0")
    skipped = run.orders.filter(pl.col("status") == "skipped")
    assert set(skipped["reason"].to_list()) <= SKIP_REASONS
    run2 = cli(tmp_path, tag="b2")
    sk2 = run2.orders.filter(pl.col("status") == "skipped")
    assert "below_band" not in set(sk2["reason"].to_list())
