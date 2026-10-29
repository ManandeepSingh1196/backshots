"""M3 exit check: the same config gives byte-identical deterministic files (ticket T029).

Never skip or xfail (CLAUDE.md section 8).
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

import bt.engine.backtest as bt_backtest
from bt.engine.backtest import run_backtest
from bt.runs.config import RunConfig, load_config
from bt.runs.reader import read_run
from bt.runs.writer import DETERMINISTIC_FILES
from bt.strategy.registry import get_strategy
from tests.support.buy_once import BuyOnce
from tests.support.make_golden_fixture import FIXTURE_DIR

ROOT = Path(__file__).resolve().parents[2]
CONFIG = """
[run]
strategy = "test_buy_once"
start = 2024-01-01
end = 2024-01-05
initial_equity_inr = 85000
inr_per_usdt = 85

[universe]
symbols = ["BBBUSDT", "AAAUSDT"]
kind = "spot"

[params]
buys = { AAAUSDT = 5.0 }

[costs.slippage]
model = "fixed_bps"
bps = 10
"""


def test_buy_once_registered() -> None:
    assert get_strategy("test_buy_once") is BuyOnce


FIXED = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)


def config(buys: str = "5.0") -> RunConfig:
    return load_config(None, [f"params.buys={{AAAUSDT={buys}}}", *_BASE])


_BASE = [
    "run.strategy=test_buy_once",
    "run.start=2024-01-01",
    "run.end=2024-01-05",
    "run.initial_equity_inr=85000",
    "run.inr_per_usdt=85",
    'universe.symbols=["BBBUSDT", "AAAUSDT"]',
    "costs.slippage.bps=10",
]


def go(runs: Path, cfg: RunConfig | None = None, data_dir: Path = FIXTURE_DIR) -> Path:
    return run_backtest(cfg or config(), data_dir=data_dir, runs_dir=runs).run_dir


def hash8(d: Path) -> str:
    return json.loads((d / "meta.json").read_text())["hash8"]


def assert_same_bytes(a: Path, b: Path) -> None:
    for name in DETERMINISTIC_FILES:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_same_process_two_runs(tmp_path: Path) -> None:
    a, b = go(tmp_path / "r1"), go(tmp_path / "r2")
    assert_same_bytes(a, b)
    assert hash8(a) == hash8(b)


def test_same_runs_dir_gets_suffix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bt_backtest, "_utcnow", lambda: FIXED)
    a, b = go(tmp_path), go(tmp_path)
    assert b.name == a.name + "-2"
    assert_same_bytes(a, b)


def test_cross_process_hash_seed(tmp_path: Path) -> None:
    cfg_path = tmp_path / "run.toml"
    cfg_path.write_text(CONFIG)
    sub_runs = tmp_path / "sub"
    code = (
        "import sys; from pathlib import Path\n"
        "import tests.support.buy_once\n"
        "from bt.engine.backtest import run_backtest\n"
        "from bt.runs.config import load_config\n"
        "cfg = load_config(Path(sys.argv[1]))\n"
        "r = run_backtest(cfg, data_dir=Path(sys.argv[2]), runs_dir=Path(sys.argv[3]))\n"
        "print(r.run_dir)\n"
    )
    env = {**os.environ, "PYTHONHASHSEED": "123"}
    out = subprocess.run(
        [sys.executable, "-c", code, str(cfg_path), str(FIXTURE_DIR), str(sub_runs)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    sub_dir = Path(out.stdout.strip().splitlines()[-1])
    here = go(tmp_path / "here", load_config(cfg_path))
    assert_same_bytes(sub_dir, here)
    assert hash8(sub_dir) == hash8(here)


def test_sensitivity_to_params(tmp_path: Path) -> None:
    a, b = go(tmp_path / "r1"), go(tmp_path / "r2", config("4.0"))
    assert hash8(a) != hash8(b)
    assert (a / "metrics.json").read_bytes() != (b / "metrics.json").read_bytes()


def _copy_fixture(tmp_path: Path) -> Path:
    dst = tmp_path / "data"
    shutil.copytree(FIXTURE_DIR, dst)
    return dst


def test_sensitivity_to_instrument_fee(tmp_path: Path) -> None:
    data = _copy_fixture(tmp_path)
    path = data / "instruments.json"
    doc = json.loads(path.read_text())
    for inst in doc["instruments"]:
        if inst["symbol"] == "BBBUSDT":
            inst["taker_fee"] = 0.002
    path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    assert hash8(go(tmp_path / "r1")) != hash8(go(tmp_path / "r2", data_dir=data))


def test_ingested_at_is_not_hashed(tmp_path: Path) -> None:
    data = _copy_fixture(tmp_path)
    path = data / "manifest.json"
    doc = json.loads(path.read_text())
    for entry in doc["files"].values():
        entry["ingested_at"] = "2025-06-01T00:00:00+00:00"
    path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    a, b = go(tmp_path / "r1"), go(tmp_path / "r2", data_dir=data)
    assert hash8(a) == hash8(b)
    assert_same_bytes(a, b)


def test_meta_and_report_differ_only_in_time_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    times = iter(datetime(2024, 3, 1, 12, m, tzinfo=UTC) for m in range(10))
    monkeypatch.setattr(bt_backtest, "_utcnow", lambda: next(times))
    a, b = go(tmp_path / "r1"), go(tmp_path / "r2")
    ma = json.loads((a / "meta.json").read_text())
    mb = json.loads((b / "meta.json").read_text())
    assert ma["run_id"] != mb["run_id"]
    ignore = {"run_id", "started_at", "finished_at", "wall_seconds"}
    assert {k: v for k, v in ma.items() if k not in ignore} == {
        k: v for k, v in mb.items() if k not in ignore
    }
    ra = (a / "report.txt").read_text().splitlines()
    rb = (b / "report.txt").read_text().splitlines()
    assert ra[0] != rb[0]
    assert ra[1:] == rb[1:]
    assert read_run(a).meta.hash8 == read_run(b).meta.hash8
