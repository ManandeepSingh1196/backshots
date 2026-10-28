"""Integration tests for `bt data`: fake archive server + fake ccxt exchange. No network."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from bt.cli import data as cli_data
from bt.cli.main import app
from bt.data.catalog import Catalog
from bt.data.fetch_binance import ArchiveFile, funding_archive_files, kline_archive_files
from tests.support.binance_archive import (
    ArchiveServer,
    Response,
    build_checksum,
    build_csv_zip,
)
from tests.support.fake_exchange import (
    FakeExchange,
    epoch_ms,
    funding_row,
    kline_row,
    perp_market,
    spot_market,
)

NOW = datetime(2024, 3, 10, 1, 0, tzinfo=UTC)
TODAY = NOW.date()
H8 = timedelta(hours=8)
START = date(2024, 1, 1)


def at(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def days(first: date, last: date) -> list[date]:
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def price(d: date) -> float:
    return 100.0 + (d - START).days


def kline_csv(ds: Iterable[date]) -> bytes:
    lines: list[str] = []
    for d in ds:
        p = price(d)
        open_ms = epoch_ms(at(d))
        close_ms = open_ms + 86_400_000 - 1
        ohlc = f"{p:.2f},{p + 1:.2f},{p - 1:.2f},{p:.2f}"
        lines.append(f"{open_ms},{ohlc},10.0,{close_ms},1000.0,42,5.0,500.0,0")
    return ("\n".join(lines) + "\n").encode()


def funding_csv(first: datetime, last: datetime) -> bytes:
    lines = ["calc_time,funding_interval_hours,last_funding_rate"]
    t = first
    while t <= last:
        lines.append(f"{epoch_ms(t)},8,0.00010000")
        t += H8
    return ("\n".join(lines) + "\n").encode()


def publish(responses: dict[str, Response], file: ArchiveFile, csv: bytes) -> None:
    zip_bytes = build_csv_zip(csv)
    responses[file.url] = (200, zip_bytes)
    responses[file.checksum_url] = (200, build_checksum(zip_bytes, file.url.rsplit("/", 1)[1]))


def kline_files(symbol: str) -> dict[str, ArchiveFile]:
    files = kline_archive_files("perp", "1d", symbol, START, TODAY, TODAY)
    return {f.period: f for f in files}


def publish_klines(
    responses: dict[str, Response], symbol: str, jan_csv: bytes | None = None
) -> None:
    files = kline_files(symbol)
    publish(responses, files["2024-01"], jan_csv or kline_csv(days(START, date(2024, 1, 31))))
    publish(responses, files["2024-02"], kline_csv(days(date(2024, 2, 1), date(2024, 2, 29))))
    for d in days(date(2024, 3, 1), date(2024, 3, 8)):
        publish(responses, files[f"{d:%Y-%m-%d}"], kline_csv([d]))


def publish_funding(responses: dict[str, Response], symbol: str) -> None:
    files = {f.period: f for f in funding_archive_files(symbol, START, TODAY, TODAY)}
    jan = funding_csv(at(START), datetime(2024, 1, 31, 16, tzinfo=UTC))
    feb = funding_csv(at(date(2024, 2, 1)), datetime(2024, 2, 29, 16, tzinfo=UTC))
    publish(responses, files["2024-01"], jan)
    publish(responses, files["2024-02"], feb)


def make_exchange(symbols: Iterable[str]) -> FakeExchange:
    fake = FakeExchange(markets=[spot_market(), perp_market()])
    for symbol in symbols:
        rows = [
            kline_row(epoch_ms(at(d)), epoch_ms(at(d)) + 86_399_999, price(d))
            for d in days(date(2024, 3, 9), TODAY)
        ]
        fake.set_klines("fapiPublicGetKlines", symbol, "1d", rows)
        t = datetime(2024, 3, 1, tzinfo=UTC)
        events: list[dict[str, object]] = []
        while t <= datetime(2024, 3, 10, 8, tzinfo=UTC):
            events.append(funding_row(epoch_ms(t), "0.0002", "42000.5"))
            t += H8
        fake.set_funding(symbol, events)
    return fake


class Env:
    def __init__(self, tmp_path: Path, responses: dict[str, Response], fake: FakeExchange) -> None:
        self.data_dir = tmp_path / "data"
        self.server = ArchiveServer(responses)
        self.fake = fake

    def invoke(self, *args: str, data_dir: bool = True) -> tuple[int, str]:
        extra = ["--data-dir", str(self.data_dir)] if data_dir else []
        result = CliRunner().invoke(app, ["data", *args, *extra])
        return result.exit_code, result.output


MakeEnv = Callable[..., Env]


@pytest.fixture
def make_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> MakeEnv:
    def build(responses: dict[str, Response], symbols: Iterable[str] = ("BTCUSDT",)) -> Env:
        env = Env(tmp_path, responses, make_exchange(symbols))
        monkeypatch.setattr(cli_data, "_make_client", env.server.client)
        monkeypatch.setattr(cli_data, "_make_exchange", lambda: env.fake)
        monkeypatch.setattr(cli_data, "_now", lambda: NOW)
        return env

    return build


def good_responses() -> dict[str, Response]:
    responses: dict[str, Response] = {}
    publish_klines(responses, "BTCUSDT")
    publish_funding(responses, "BTCUSDT")
    return responses


FETCH = ["fetch", "--symbols", "BTCUSDT", "--markets", "perp", "--timeframes", "1d"]


def fetch_args(*extra: str) -> list[str]:
    return [*FETCH, "--start", "2024-01-01", *extra]


# ----------------------------------------------------------------------------- fetch


def test_fetch_stores_klines_and_funding(make_env: MakeEnv) -> None:
    env = make_env(good_responses())
    code, out = env.invoke(*fetch_args())
    assert code == 0, out
    assert (env.data_dir / "binance/perp/klines/1d/BTCUSDT.parquet").is_file()
    assert (env.data_dir / "binance/perp/funding/BTCUSDT.parquet").is_file()
    assert len(Catalog(env.data_dir).manifest()) == 2
    assert "perp/klines/1d/BTCUSDT" in out
    assert "69" in out
    assert "perp/funding/BTCUSDT" in out
    assert "survivorship" in out


def test_fetch_no_funding_stores_one_entry(make_env: MakeEnv) -> None:
    env = make_env(good_responses())
    code, out = env.invoke(*fetch_args("--no-funding"))
    assert code == 0, out
    assert len(Catalog(env.data_dir).manifest()) == 1
    assert not any("fundingRate" in u for u in env.server.requested_urls)


def test_fetch_spot_only_never_requests_funding(make_env: MakeEnv) -> None:
    responses: dict[str, Response] = {}
    files = kline_archive_files("spot", "1d", "BTCUSDT", START, TODAY, TODAY)
    for f in files:
        d = f.period
        if len(d) == 7:
            first = date.fromisoformat(d + "-01")
            last = (first + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            publish(responses, f, kline_csv(days(first, last)))
        elif date.fromisoformat(d) <= date(2024, 3, 8):
            publish(responses, f, kline_csv([date.fromisoformat(d)]))
    env = make_env(responses)
    env.fake.set_klines(
        "publicGetKlines",
        "BTCUSDT",
        "1d",
        [
            kline_row(epoch_ms(at(d)), epoch_ms(at(d)) + 86_399_999, price(d))
            for d in days(date(2024, 3, 9), TODAY)
        ],
    )
    code, out = env.invoke(
        "fetch", "--symbols", "btcusdt", "--markets", "spot", "--timeframes", "1d",
        "--start", "2024-01-01",
    )  # fmt: skip
    assert code == 0, out
    assert "spot/klines/1d/BTCUSDT" in out
    assert list(Catalog(env.data_dir).manifest()) == ["binance/spot/klines/1d/BTCUSDT.parquet"]
    assert not any("fundingRate" in u for u in env.server.requested_urls)
    assert env.fake.calls_to("fapiPublicGetFundingRate") == []


def test_fetch_failed_validation_exits_1_and_continues(make_env: MakeEnv) -> None:
    responses: dict[str, Response] = {}
    publish_klines(responses, "BTCUSDT")
    dup = kline_csv(days(START, date(2024, 1, 31))) + kline_csv([START]).replace(
        b",100.00,", b",150.00,"
    )
    publish_klines(responses, "ETHUSDT", jan_csv=dup)
    env = make_env(responses, ("BTCUSDT", "ETHUSDT"))
    code, out = env.invoke(
        "fetch", "--symbols", "ETHUSDT,BTCUSDT", "--markets", "perp", "--timeframes", "1d",
        "--start", "2024-01-01", "--no-funding",
    )  # fmt: skip
    assert code == 1, out
    assert list(Catalog(env.data_dir).manifest()) == ["binance/perp/klines/1d/BTCUSDT.parquet"]
    assert "perp/klines/1d/ETHUSDT" in out
    assert "FAILED" in out
    assert "duplicate ts" in out


def test_fetch_checksum_mismatch_exits_1_and_continues(make_env: MakeEnv) -> None:
    responses: dict[str, Response] = {}
    publish_klines(responses, "BTCUSDT")
    publish_klines(responses, "ETHUSDT")
    jan = kline_files("ETHUSDT")["2024-01"]
    responses[jan.checksum_url] = (200, build_checksum(b"other", "x.zip"))
    env = make_env(responses, ("BTCUSDT", "ETHUSDT"))
    code, out = env.invoke(
        "fetch", "--symbols", "ETHUSDT,BTCUSDT", "--markets", "perp", "--timeframes", "1d",
        "--start", "2024-01-01", "--no-funding",
    )  # fmt: skip
    assert code == 1, out
    assert list(Catalog(env.data_dir).manifest()) == ["binance/perp/klines/1d/BTCUSDT.parquet"]
    assert "ChecksumMismatchError" in out
    assert "FAILED" in out


def test_fetch_lists_missing_archives_and_gaps_without_failing(make_env: MakeEnv) -> None:
    responses = good_responses()
    feb = kline_files("BTCUSDT")["2024-02"]
    del responses[feb.url], responses[feb.checksum_url]
    env = make_env(responses)
    code, out = env.invoke(*fetch_args("--no-funding"))
    assert code == 0, out
    assert "missing archive" in out
    assert "gap: after" in out
    assert "missing 29" in out


# ----------------------------------------------------------------------- bad arguments


@pytest.mark.parametrize(
    "bad",
    [
        ["--markets", "futures"],
        ["--timeframes", "1h"],
        ["--start", "2024-13-01"],
        ["--end", "soon"],
        ["--symbols", "../x"],
        ["--symbols", ","],
        ["--start", "2024-02-01", "--end", "2024-01-01"],
        ["--start", "2024-03-11"],
    ],
)
def test_fetch_bad_arguments_exit_2(make_env: MakeEnv, bad: list[str]) -> None:
    env = make_env(good_responses())
    code, out = env.invoke("fetch", "--start", "2024-01-01", *bad)
    assert code == 2, out
    assert env.server.requested_urls == []
    assert not env.data_dir.exists()


def test_instruments_bad_arguments_exit_2(make_env: MakeEnv) -> None:
    env = make_env({})
    assert env.invoke("instruments", "--markets", "futures")[0] == 2
    assert env.invoke("instruments", "--symbols", "../x")[0] == 2


# ----------------------------------------------------------------------------- check


def test_check_after_good_fetch_exits_0(make_env: MakeEnv) -> None:
    env = make_env(good_responses())
    assert env.invoke(*fetch_args())[0] == 0
    code, out = env.invoke("check")
    assert code == 0, out
    assert "perp/klines/1d/BTCUSDT" in out
    assert "perp/funding/BTCUSDT" in out


def test_check_after_tampering_exits_1(make_env: MakeEnv) -> None:
    env = make_env(good_responses())
    assert env.invoke(*fetch_args("--no-funding"))[0] == 0
    path = env.data_dir / "binance/perp/klines/1d/BTCUSDT.parquet"
    pl.read_parquet(path).slice(0, 68).write_parquet(path)
    code, out = env.invoke("check")
    assert code == 1, out
    assert "manifest mismatch" in out
    assert "FAILED" in out


def test_check_empty_dir_prints_no_datasets(make_env: MakeEnv, tmp_path: Path) -> None:
    env = make_env({})
    code, out = env.invoke("check")
    assert code == 0
    assert "no datasets" in out


def test_check_corrupt_manifest_exits_1(make_env: MakeEnv) -> None:
    env = make_env({})
    env.data_dir.mkdir()
    (env.data_dir / "manifest.json").write_text("{not json")
    code, out = env.invoke("check")
    assert code == 1, out
    assert "manifest" in out


# ------------------------------------------------------------------------ instruments


def test_instruments_writes_snapshot_and_lists_both(make_env: MakeEnv) -> None:
    env = make_env({})
    code, out = env.invoke("instruments", "--symbols", "BTCUSDT")
    assert code == 0, out
    raw = (env.data_dir / "instruments.json").read_text()
    assert "BTCUSDT-PERP" in raw
    assert {i["symbol"] for i in json.loads(raw)["instruments"]} == {"BTCUSDT", "BTCUSDT-PERP"}
    assert "BTCUSDT-PERP" in out
    assert "spot" in out
    assert "perp" in out


def test_instruments_unknown_symbol_exits_1(make_env: MakeEnv) -> None:
    env = make_env({})
    code, out = env.invoke("instruments", "--symbols", "NOPEUSDT")
    assert code == 1, out
    assert not (env.data_dir / "instruments.json").exists()


# ---------------------------------------------------------------------------- help


def test_data_help_lists_commands() -> None:
    result = CliRunner().invoke(app, ["data", "--help"])
    assert result.exit_code == 0
    for name in ("fetch", "check", "instruments"):
        assert name in result.output
