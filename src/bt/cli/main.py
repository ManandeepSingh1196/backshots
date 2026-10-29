"""The `bt` command-line entry point."""

import typer

from bt.cli.data import data_app
from bt.cli.run import run_cmd, show_cmd

app = typer.Typer(no_args_is_help=True, help="Crypto backtest and trading engine.")
app.add_typer(data_app, name="data", help="Download, validate and inspect market data.")
app.command("run")(run_cmd)
app.command("show")(show_cmd)


@app.callback()
def main() -> None:
    """Crypto backtest and trading engine."""
