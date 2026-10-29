"""M3 golden run through run_backtest: every expected number is hand-computed (ticket T028)."""

import json
import math
import re
import statistics
from itertools import pairwise
from pathlib import Path

import pytest

import tests.support.buy_once  # noqa: F401  (registers "test_buy_once")
from bt.engine.backtest import run_backtest
from bt.runs.config import RunConfig, resolve_config
from bt.runs.reader import read_run
from bt.runs.writer import RUN_FILES
from tests.support.make_golden_fixture import FIXTURE_DIR, day

EQUITY = [1000.0, 1028.97949, 1038.97949, 1048.97949, 1057.84006]
TOL = 1e-9


def golden_config() -> RunConfig:
    return RunConfig.model_validate(
        {
            "run": {
                "strategy": "test_buy_once",
                "start": "2024-01-01",
                "end": "2024-01-05",
                "initial_equity_inr": 85000,
                "inr_per_usdt": 85,
            },
            "universe": {"symbols": ["BBBUSDT", "AAAUSDT"], "kind": "spot"},
            "params": {"buys": {"AAAUSDT": 5.0}},
            "costs": {"slippage": {"model": "fixed_bps", "bps": 10}},
        }
    )


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    runs = tmp_path_factory.mktemp("runs")
    return run_backtest(golden_config(), data_dir=FIXTURE_DIR, runs_dir=runs).run_dir


def test_equity_rows(run_dir: Path) -> None:
    eq = read_run(run_dir).equity
    assert eq["equity_after_tax"].to_list() == pytest.approx(EQUITY, abs=TOL)
    assert eq["equity_pre_tax"].to_list() == eq["equity_after_tax"].to_list()
    assert eq["ts"].to_list() == [day(i) for i in range(1, 6)]
    assert eq["close_ts"].to_list() == [day(i) for i in range(2, 7)]


def test_initial_cash_and_buy_fill(run_dir: Path) -> None:
    eq = read_run(run_dir).equity
    assert eq["cash"][0] == pytest.approx(1000.0, abs=TOL)
    # d2: cash 1000 - 510.51 - 0.51051 = 488.97949
    assert eq["cash"][1] == pytest.approx(488.97949, abs=TOL)
    fill = read_run(run_dir).trades.row(0, named=True)
    assert fill["price"] == pytest.approx(102.102, abs=TOL)
    assert fill["notional"] == pytest.approx(510.51, abs=TOL)
    assert fill["fee"] == pytest.approx(0.51051, abs=TOL)
    assert fill["slippage_cost"] == pytest.approx(0.51, abs=TOL)
    sell = read_run(run_dir).trades.row(1, named=True)
    assert sell["price"] == pytest.approx(113.886, abs=TOL)
    assert sell["notional"] == pytest.approx(569.43, abs=TOL)
    assert sell["fee"] == pytest.approx(0.56943, abs=TOL)
    assert sell["slippage_cost"] == pytest.approx(0.57, abs=TOL)


def test_metrics(run_dir: Path) -> None:
    m = read_run(run_dir).metrics
    exact = {
        "total_return_after_tax": 0.05784006,
        "total_return_pre_tax": 0.05784006,
        "max_drawdown_after_tax": 0.0,
        "cost_fees_usdt": 1.07994,
        "cost_fees_inr": 91.7949,
        "cost_slippage_usdt": 1.08,
        "cost_funding_usdt": 0.0,
        "cost_tax_usdt": 0.0,
        "traded_notional_usdt": 1079.94,
        "n_fills": 2.0,
        "n_buys": 1.0,
        "n_sells": 1.0,
        "win_rate": 1.0,
        "exposure_time": 0.6,
        "n_bars": 5.0,
        "years": 4 / 365,
        "missing_bars": 1.0,
        "final_equity_inr_after_tax": 89916.4051,
        "killed": 0.0,
    }
    for key, want in exact.items():
        assert m[key] == pytest.approx(want, abs=TOL), key
    r = [b / a - 1 for a, b in pairwise(EQUITY)]
    sharpe = statistics.mean(r) / statistics.stdev(r) * math.sqrt(365)
    assert m["sharpe_after_tax"] == pytest.approx(sharpe, rel=1e-9)
    bench = [k for k in m if k.startswith("benchmark_")]
    assert bench
    assert all(math.isnan(m[k]) for k in bench)


def test_trades(run_dir: Path) -> None:
    t = read_run(run_dir).trades
    assert t.height == 2
    assert t["order_id"].to_list() == ["buy-AAAUSDT", "bt-close-AAAUSDT-202401050000"]
    assert t["tag"][1] == "close_at_end"


def test_messages(run_dir: Path) -> None:
    msgs = read_run(run_dir).messages
    assert msgs.height == 1
    assert msgs["message"][0] == "buy_once"
    assert msgs["ts"][0] == day(1)
    assert msgs["fields"][0] == '{"n":1}'


def test_directory_contents(run_dir: Path) -> None:
    for name in RUN_FILES:
        assert (run_dir / name).is_file(), name
    data = read_run(run_dir)
    assert data.config == resolve_config(golden_config())
    assert data.config.params == {"buys": {"AAAUSDT": 5.0}, "warmup": 0}
    assert data.config.universe.symbols == ("AAAUSDT", "BBBUSDT")
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d{4}_test_buy_once_[0-9a-f]{8}", run_dir.name)
    meta = json.loads((run_dir / "meta.json").read_text())
    assert run_dir.name.endswith("_" + meta["hash8"])


def test_report(run_dir: Path) -> None:
    text = (run_dir / "report.txt").read_text(encoding="utf-8")
    assert "WARNING: BBBUSDT has 1 missing bar(s)" in text
    assert "Survivorship bias" in text
    assert "₹91.79" in text
