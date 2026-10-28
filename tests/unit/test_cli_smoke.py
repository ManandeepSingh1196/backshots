from typer.testing import CliRunner

from bt.cli.main import app


def test_help_exits_zero() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Crypto backtest and trading engine" in result.output
