"""The `bt` command-line entry point."""

import typer

app = typer.Typer(no_args_is_help=True, help="Crypto backtest and trading engine.")


@app.callback()
def main() -> None:
    """Crypto backtest and trading engine."""
