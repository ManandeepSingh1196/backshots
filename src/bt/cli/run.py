"""`bt run` and `bt show`: backtest from a config, and re-print a past run. No clock, no network."""

import importlib.util
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape

from bt.analytics.report import WARNING_PREFIX, render_report
from bt.core.errors import BtError, ConfigError, RunError, StrategyError
from bt.engine.backtest import run_backtest
from bt.runs.config import RunConfig, load_config, resolve_config
from bt.runs.reader import read_run, resolve_run_ref

_EXIT_FAILED = 1
_EXIT_USAGE = 2
_WIDE = 140

_RunsDirOpt = Annotated[Path, typer.Option("--runs-dir", help="Runs root directory.")]


def _console() -> Console:
    """Plain rich console; wide when not a terminal so report lines are never wrapped."""
    console = Console(highlight=False, emoji=False)
    if not console.is_terminal:
        console = Console(highlight=False, emoji=False, width=_WIDE)
    return console


def _ensure_strategies_importable(cwd: Path) -> None:
    """Append cwd to sys.path (never at index 0) if it holds a `strategies` package that is not
    already importable."""
    if importlib.util.find_spec("strategies") is not None:
        return
    if (cwd / "strategies" / "__init__.py").is_file():
        entry = str(cwd)
        if entry not in sys.path:
            sys.path.append(entry)


def _fail(console: Console, error: BaseException, code: int) -> typer.Exit:
    console.print(f"[red]error: {escape(type(error).__name__)}: {escape(str(error))}[/red]")
    return typer.Exit(code)


def _print_report(console: Console, text: str) -> None:
    for line in text.splitlines():
        style = "red" if line.startswith(WARNING_PREFIX) else None
        console.print(line, style=style, markup=False, highlight=False, emoji=False, soft_wrap=True)


def run_cmd(
    strategy: Annotated[str | None, typer.Argument(help="Strategy name (run.strategy).")] = None,
    config: Annotated[Path | None, typer.Option("--config", help="Run config TOML.")] = None,
    set_values: Annotated[
        list[str] | None,
        typer.Option("--set", help="Override KEY=VALUE, e.g. run.start=2024-01-01. Repeatable."),
    ] = None,
    data_dir: Annotated[Path, typer.Option("--data-dir", help="Data root directory.")] = Path(
        "data"
    ),
    runs_dir: _RunsDirOpt = Path("runs"),
) -> None:
    """Backtest a strategy and write a run directory."""
    console = _console()
    overrides = ([f"run.strategy={strategy}"] if strategy else []) + list(set_values or [])
    _ensure_strategies_importable(Path.cwd())
    try:
        cfg: RunConfig = load_config(config, overrides)
        resolve_config(cfg)
    except (ConfigError, StrategyError) as e:
        raise _fail(console, e, _EXIT_USAGE) from e
    except BtError as e:
        raise _fail(console, e, _EXIT_FAILED) from e
    try:
        result = run_backtest(cfg, data_dir=data_dir, runs_dir=runs_dir)
        data = read_run(result.run_dir)
    except ConfigError as e:
        raise _fail(console, e, _EXIT_USAGE) from e
    except BtError as e:
        raise _fail(console, e, _EXIT_FAILED) from e
    _print_report(console, data.report)
    console.print(f"run dir: {escape(str(result.run_dir))}", markup=False, soft_wrap=True)


def show_cmd(
    run: Annotated[str, typer.Argument(help="Run directory, name, hash8 prefix or 'latest'.")],
    runs_dir: _RunsDirOpt = Path("runs"),
) -> None:
    """Re-print the report of a past run."""
    console = _console()
    try:
        run_dir = resolve_run_ref(runs_dir, run)
    except RunError as e:
        raise _fail(console, e, _EXIT_USAGE) from e
    try:
        data = read_run(run_dir)
    except RunError as e:
        raise _fail(console, e, _EXIT_FAILED) from e
    text = render_report(data.config, data.metrics, data.equity, data.meta)
    _print_report(console, text)
    if text != data.report:
        console.print("[yellow]report.txt differs from re-render (older code version?)[/yellow]")
