"""The `bt` command-line entry point."""

import typer

from bt.cli.data import data_app

app = typer.Typer(no_args_is_help=True, help="Crypto backtest and trading engine.")
app.add_typer(data_app, name="data", help="Download, validate and inspect market data.")


@app.callback()
def main() -> None:
    """Crypto backtest and trading engine."""
