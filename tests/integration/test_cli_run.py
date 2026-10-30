"""bt run / bt show through typer's CliRunner (ticket T029)."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

import bt.cli.run as cli_run
import bt.engine.backtest as bt_backtest
from bt.cli.main import app
from bt.core.errors import StrategyError
from bt.runs.reader import read_run
from bt.runs.writer import RUN_FILES
from bt.strategy.registry import get_strategy
from tests.support.buy_once import BuyOnce
from tests.support.make_golden_fixture import FIXTURE_DIR

TOL = 1e-9
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

runner = CliRunner()


def test_buy_once_registered() -> None:
    assert get_strategy("test_buy_once") is BuyOnce


@pytest.fixture
def cfg_file(tmp_path: Path) -> Path:
    p = tmp_path / "run.toml"
    p.write_text(CONFIG)
    return p


def visible(runs: Path) -> list[Path]:
    return [p for p in runs.iterdir() if not p.name.startswith(".")] if runs.exists() else []


def run_args(cfg: Path, tmp: Path, *extra: str) -> list[str]:
    return [
        "run",
        "--config",
        str(cfg),
        "--data-dir",
        str(FIXTURE_DIR),
        "--runs-dir",
        str(tmp / "runs"),
        *extra,
    ]


def test_run_from_config(cfg_file: Path, tmp_path: Path) -> None:
    res = runner.invoke(app, run_args(cfg_file, tmp_path))
    assert res.exit_code == 0, res.output
    dirs = visible(tmp_path / "runs")
    assert len(dirs) == 1
    for name in RUN_FILES:
        assert (dirs[0] / name).stat().st_size > 0
    report = (dirs[0] / "report.txt").read_text()
    assert report.rstrip("\n") in res.stdout
    assert res.stdout.rstrip("\n").splitlines()[-1] == f"run dir: {dirs[0]}"
    metrics = json.loads((dirs[0] / "metrics.json").read_text())
    assert metrics["total_return_after_tax"] == pytest.approx(0.05784006, abs=TOL)


def test_overrides_and_positional_strategy(tmp_path: Path) -> None:
    res = runner.invoke(
        app,
        [
            "run",
            "test_buy_once",
            "--set",
            "run.start=2024-01-01",
            "--set",
            "run.end=2024-01-05",
            "--set",
            'universe.symbols=["AAAUSDT"]',
            "--set",
            "params.buys={AAAUSDT=5.0}",
            "--set",
            "run.initial_equity_inr=85000",
            "--set",
            "run.inr_per_usdt=85",
            "--set",
            "costs.slippage.bps=10",
            "--data-dir",
            str(FIXTURE_DIR),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert res.exit_code == 0, res.output
    (d,) = visible(tmp_path / "runs")
    assert 'symbols = ["AAAUSDT"]' in (d / "config.toml").read_text()
    data = read_run(d)
    assert data.equity["equity_after_tax"].to_list() == pytest.approx(
        [1000.0, 1028.97949, 1038.97949, 1048.97949, 1057.84006], abs=TOL
    )
    assert data.metrics["final_equity_inr_after_tax"] == pytest.approx(89916.4051, abs=1e-4)
    assert data.metrics["missing_bars"] == pytest.approx(0.0, abs=TOL)


@pytest.mark.parametrize(
    "extra",
    [
        ["--set", "run.foo=1"],
        ["--set", "nonsense"],
        ["nope"],
        ["--set", "params.zz=1"],
        ["--config", "MISSING"],
        ["--set", "engine.max_drawdown_kill=2"],
    ],
)
def test_exit_2(extra: list[str], cfg_file: Path, tmp_path: Path) -> None:
    extra = [str(tmp_path / "missing.toml") if x == "MISSING" else x for x in extra]
    args = run_args(cfg_file, tmp_path)
    if "--config" in extra:  # drop the valid --config pair so the missing file is the only --config
        args = args[:1] + args[3:]
    res = runner.invoke(app, [*args, *extra])
    assert res.exit_code == 2, res.output
    assert "error:" in res.output
    assert visible(tmp_path / "runs") == []


def test_exit_1_empty_data_dir(cfg_file: Path, tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    args = run_args(cfg_file, tmp_path)
    args[args.index("--data-dir") + 1] = str(empty)
    res = runner.invoke(app, args)
    assert res.exit_code == 1, res.output
    assert "error:" in res.output
    assert visible(tmp_path / "runs") == []


def test_exit_1_unknown_symbol(cfg_file: Path, tmp_path: Path) -> None:
    res = runner.invoke(app, run_args(cfg_file, tmp_path, "--set", 'universe.symbols=["CCCUSDT"]'))
    assert res.exit_code == 1, res.output
    assert "error:" in res.output
    assert visible(tmp_path / "runs") == []


def test_show(cfg_file: Path, tmp_path: Path) -> None:
    assert runner.invoke(app, run_args(cfg_file, tmp_path)).exit_code == 0
    runs = tmp_path / "runs"
    (d,) = visible(runs)
    report = (d / "report.txt").read_text().rstrip("\n")
    hash8 = d.name.split("_")[-1][:8]
    for ref in (str(d), hash8, "latest"):
        res = runner.invoke(app, ["show", ref, "--runs-dir", str(runs)])
        assert res.exit_code == 0, res.output
        assert report in res.stdout
        assert "differs" not in res.stdout
    res = runner.invoke(app, ["show", "zzzz", "--runs-dir", str(runs)])
    assert res.exit_code == 2
    assert "error:" in res.output


def test_show_notes_stale_report(cfg_file: Path, tmp_path: Path) -> None:
    assert runner.invoke(app, run_args(cfg_file, tmp_path)).exit_code == 0
    (d,) = visible(tmp_path / "runs")
    (d / "report.txt").write_text("old report\n")
    res = runner.invoke(app, ["show", str(d)])
    assert res.exit_code == 0, res.output
    assert "report.txt differs from re-render" in res.stdout


def test_show_latest_picks_later_run(
    cfg_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    times = iter([datetime(2024, 3, 1, tzinfo=UTC), datetime(2024, 3, 2, tzinfo=UTC)] * 2)
    monkeypatch.setattr(bt_backtest, "_utcnow", lambda: next(times))
    runs = tmp_path / "runs"
    assert runner.invoke(app, run_args(cfg_file, tmp_path)).exit_code == 0
    res2 = runner.invoke(app, run_args(cfg_file, tmp_path, "--set", "params.buys={AAAUSDT=4.0}"))
    assert res2.exit_code == 0, res2.output
    first, second = sorted(visible(runs), key=lambda p: p.name)
    h1, h2 = first.name.split("_")[-1][:4], second.name.split("_")[-1][:4]
    assert h1 != h2
    res = runner.invoke(app, ["show", "latest", "--runs-dir", str(runs)])
    assert res.exit_code == 0
    assert (second / "report.txt").read_text().rstrip("\n") in res.stdout


def _spec_none(name: str, package: str | None = None) -> None:
    return None


def _spec_found(name: str, package: str | None = None) -> object:
    return object()


def test_ensure_strategies_importable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(cli_run.importlib.util, "find_spec", _spec_none)
    empty = tmp_path / "empty"
    empty.mkdir()
    before = list(sys.path)
    cli_run._ensure_strategies_importable(empty)  # pyright: ignore[reportPrivateUsage]
    assert sys.path == before

    pkg = tmp_path / "proj"
    (pkg / "strategies").mkdir(parents=True)
    (pkg / "strategies" / "__init__.py").write_text("")
    cli_run._ensure_strategies_importable(pkg)  # pyright: ignore[reportPrivateUsage]
    assert sys.path[-1] == str(pkg)
    assert sys.path[0] != str(pkg)
    cli_run._ensure_strategies_importable(pkg)  # pyright: ignore[reportPrivateUsage]
    assert sys.path.count(str(pkg)) == 1


def test_ensure_strategies_importable_noop_when_importable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(cli_run.importlib.util, "find_spec", _spec_found)
    (tmp_path / "strategies").mkdir()
    (tmp_path / "strategies" / "__init__.py").write_text("")
    before = list(sys.path)
    cli_run._ensure_strategies_importable(tmp_path)  # pyright: ignore[reportPrivateUsage]
    assert sys.path == before


def test_help_lists_commands() -> None:
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    for name in ("data", "run", "show"):
        assert name in res.stdout


def test_list_strategies() -> None:
    res = runner.invoke(app, ["list", "strategies"])
    assert res.exit_code == 0, res.output
    lines = res.output.splitlines()
    assert 'test_buy_once  {"buys":{},"warmup":0}' in lines
    names = [ln.split("  ", 1)[0] for ln in lines]
    assert names == sorted(names)


def test_list_strategies_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> list[tuple[str, type]]:
        raise StrategyError("boom")

    monkeypatch.setattr(cli_run, "list_strategies", boom)
    res = runner.invoke(app, ["list", "strategies"])
    assert res.exit_code == 1
    assert "boom" in res.output


def test_list_no_subcommand() -> None:
    res = runner.invoke(app, ["list"])
    assert "strategies" in res.output
    assert "Traceback" not in res.output
