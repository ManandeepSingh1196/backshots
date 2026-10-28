from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import polars as pl
import pytest

from bt.core.errors import ArchiveError, ChecksumMismatchError, DataError
from bt.data import fetch_binance
from bt.data.fetch_binance import (
    daily_kline_archive_files,
    download_archive,
    funding_archive_files,
    kline_archive_files,
    parse_funding_csv,
    parse_klines_csv,
)
from bt.data.validate import FUNDING_SCHEMA, KLINE_SCHEMA
from tests.support.binance_archive import (
    ArchiveServer,
    build_checksum,
    build_csv_zip,
    build_zip,
    sha256_hex,
)

BASE = "https://data.binance.vision"
TODAY = date(2024, 3, 10)


def test_perp_4h_listing_monthly_then_daily() -> None:
    files = kline_archive_files("perp", "4h", "BTCUSDT", date(2023, 11, 15), TODAY, today=TODAY)
    assert len(files) == 13
    assert [f.period for f in files[:4]] == ["2023-11", "2023-12", "2024-01", "2024-02"]
    assert files[4].period == "2024-03-01"
    assert (
        files[0].url == f"{BASE}/data/futures/um/monthly/klines/BTCUSDT/4h/BTCUSDT-4h-2023-11.zip"
    )
    assert (
        files[-1].url == f"{BASE}/data/futures/um/daily/klines/BTCUSDT/4h/BTCUSDT-4h-2024-03-09.zip"
    )
    assert files[-1].period == "2024-03-09"
    assert all(f.checksum_url == f.url + ".CHECKSUM" for f in files)
    assert sum("/daily/" in f.url for f in files) == 9


def test_spot_url_root() -> None:
    files = kline_archive_files(
        "spot", "1d", "ETHUSDT", date(2024, 1, 5), date(2024, 1, 20), today=TODAY
    )
    assert [f.url for f in files] == [
        f"{BASE}/data/spot/monthly/klines/ETHUSDT/1d/ETHUSDT-1d-2024-01.zip"
    ]


def test_end_clamp_and_edges() -> None:
    files = kline_archive_files(
        "perp", "1d", "BTCUSDT", date(2023, 11, 15), date(2024, 1, 20), today=TODAY
    )
    assert [f.period for f in files] == ["2023-11", "2023-12", "2024-01"]
    assert kline_archive_files("perp", "1d", "BTCUSDT", TODAY, TODAY, today=TODAY) == []
    with pytest.raises(ValueError, match="after end"):
        kline_archive_files(
            "perp", "1d", "BTCUSDT", date(2024, 1, 2), date(2024, 1, 1), today=TODAY
        )


def test_end_beyond_today_is_clamped() -> None:
    files = kline_archive_files(
        "perp", "1d", "BTCUSDT", date(2024, 3, 8), date(2030, 1, 1), today=TODAY
    )
    assert [f.period for f in files] == ["2024-03-08", "2024-03-09"]


def test_start_in_future_gives_empty() -> None:
    assert (
        kline_archive_files(
            "perp", "1d", "BTCUSDT", date(2025, 1, 1), date(2025, 2, 1), today=TODAY
        )
        == []
    )


def test_daily_month_listing_leap_year() -> None:
    files = daily_kline_archive_files("spot", "1d", "BTCUSDT", date(2024, 2, 17))
    assert len(files) == 29
    assert files[0].period == "2024-02-01"
    assert files[-1].period == "2024-02-29"
    assert files[-1].url.endswith("/data/spot/daily/klines/BTCUSDT/1d/BTCUSDT-1d-2024-02-29.zip")


def test_funding_listing() -> None:
    files = funding_archive_files("BTCUSDT", date(2024, 1, 1), TODAY, today=TODAY)
    assert len(files) == 2
    assert files[0].url == (
        f"{BASE}/data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2024-01.zip"
    )
    assert files[0].checksum_url == files[0].url + ".CHECKSUM"


def test_invalid_symbol_raises_data_error() -> None:
    with pytest.raises(DataError):
        kline_archive_files("perp", "1d", "btc/usdt", date(2024, 1, 1), TODAY, today=TODAY)
    with pytest.raises(DataError):
        daily_kline_archive_files("perp", "1d", "../x", date(2024, 1, 1))
    with pytest.raises(DataError):
        funding_archive_files("B", date(2024, 1, 1), TODAY, today=TODAY)


# --- download ---------------------------------------------------------------------------

CSV = b"1,2,3\n"
FILE = kline_archive_files(
    "spot", "1d", "BTCUSDT", date(2024, 1, 1), date(2024, 1, 31), today=TODAY
)[0]
FNAME = "BTCUSDT-1d-2024-01.zip"
REL = "data/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip"


def _serve(zip_bytes: bytes | None = None) -> tuple[ArchiveServer, bytes]:
    z = zip_bytes if zip_bytes is not None else build_csv_zip(CSV, "BTCUSDT-1d-2024-01.csv")
    server = ArchiveServer({FILE.url: (200, z), FILE.checksum_url: (200, build_checksum(z, FNAME))})
    return server, z


def test_download_happy_path_and_cache_layout(tmp_path: Path) -> None:
    server, z = _serve()
    with server.client() as c:
        data = download_archive(c, FILE, tmp_path)
    assert data is not None
    assert data.csv == CSV
    assert data.sha256 == sha256_hex(z)
    assert data.file == FILE
    assert (tmp_path / REL).read_bytes() == z
    assert (tmp_path / (REL + ".CHECKSUM")).exists()
    assert server.requested_urls == [FILE.checksum_url, FILE.url]


def test_second_call_served_from_cache(tmp_path: Path) -> None:
    server, _ = _serve()
    with server.client() as c:
        first = download_archive(c, FILE, tmp_path)
    with ArchiveServer.failing().client() as c:
        second = download_archive(c, FILE, tmp_path)
    assert first == second


def test_corrupted_cache_is_redownloaded(tmp_path: Path) -> None:
    server, z = _serve()
    with server.client() as c:
        download_archive(c, FILE, tmp_path)
    (tmp_path / REL).write_bytes(z + b"x")
    server2, _ = _serve(z)
    with server2.client() as c:
        data = download_archive(c, FILE, tmp_path)
    assert server2.request_count == 2
    assert data is not None
    assert data.csv == CSV
    assert (tmp_path / REL).read_bytes() == z


def test_half_cached_entry_is_redownloaded(tmp_path: Path) -> None:
    server, z = _serve()
    with server.client() as c:
        download_archive(c, FILE, tmp_path)
    (tmp_path / (REL + ".CHECKSUM")).unlink()
    server2, _ = _serve(z)
    with server2.client() as c:
        assert download_archive(c, FILE, tmp_path) is not None
    assert server2.request_count == 2


def test_checksum_mismatch_caches_nothing(tmp_path: Path) -> None:
    z = build_csv_zip(CSV)
    server = ArchiveServer(
        {FILE.url: (200, z), FILE.checksum_url: (200, build_checksum(z + b"?", FNAME))}
    )
    with server.client() as c, pytest.raises(ChecksumMismatchError) as exc:
        download_archive(c, FILE, tmp_path)
    assert FILE.url in str(exc.value)
    assert sha256_hex(z) in str(exc.value)
    assert sha256_hex(z + b"?") in str(exc.value)
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


def test_checksum_404_returns_none(tmp_path: Path) -> None:
    server = ArchiveServer({})
    with server.client() as c:
        assert download_archive(c, FILE, tmp_path) is None
    assert server.request_count == 1


def test_zip_404_with_checksum_raises(tmp_path: Path) -> None:
    z = build_csv_zip(CSV)
    server = ArchiveServer({FILE.checksum_url: (200, build_checksum(z, FNAME))})
    with server.client() as c, pytest.raises(ArchiveError, match="404"):
        download_archive(c, FILE, tmp_path)


# --- retries and error handling --------------------------------------------------------


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch_binance, "_BACKOFF_SECONDS", 0.0)


def _flaky(statuses: list[int], tmp_path: Path) -> tuple[ArchiveServer, bytes]:
    z = build_csv_zip(CSV)
    server = ArchiveServer(
        {
            FILE.checksum_url: (200, build_checksum(z, FNAME)),
            FILE.url: [*[(s, b"") for s in statuses], (200, z)],
        }
    )
    return server, z


@pytest.mark.usefixtures("no_backoff")
def test_two_503_then_success(tmp_path: Path) -> None:
    server, _ = _flaky([503, 503], tmp_path)
    with server.client() as c:
        assert download_archive(c, FILE, tmp_path) is not None
    assert server.request_count == 4  # checksum + 3 zip attempts


@pytest.mark.usefixtures("no_backoff")
def test_429_is_retried(tmp_path: Path) -> None:
    server, _ = _flaky([429], tmp_path)
    with server.client() as c:
        assert download_archive(c, FILE, tmp_path) is not None


@pytest.mark.usefixtures("no_backoff")
def test_three_503_raises_archive_error(tmp_path: Path) -> None:
    server, _ = _flaky([503, 503, 503], tmp_path)
    with server.client() as c, pytest.raises(ArchiveError, match="3 attempts"):
        download_archive(c, FILE, tmp_path)
    assert server.requested_urls.count(FILE.url) == 3


@pytest.mark.usefixtures("no_backoff")
def test_403_fails_immediately(tmp_path: Path) -> None:
    server = ArchiveServer({FILE.checksum_url: (403, b"")})
    with server.client() as c, pytest.raises(ArchiveError, match="403"):
        download_archive(c, FILE, tmp_path)
    assert server.request_count == 1


@pytest.mark.usefixtures("no_backoff")
def test_transport_error_retried_then_chained(tmp_path: Path) -> None:
    calls = 0

    def boom(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("down", request=request)

    with (
        httpx.Client(transport=httpx.MockTransport(boom)) as c,
        pytest.raises(ArchiveError) as exc,
    ):
        download_archive(c, FILE, tmp_path)
    assert calls == 3
    assert exc.value.__cause__ is not None
    assert isinstance(exc.value.__cause__.__cause__, httpx.ConnectError)


def test_malformed_checksum_raises(tmp_path: Path) -> None:
    z = build_csv_zip(CSV)
    server = ArchiveServer({FILE.url: (200, z), FILE.checksum_url: (200, b"not-a-digest  file\n")})
    with server.client() as c, pytest.raises(ArchiveError, match="malformed"):
        download_archive(c, FILE, tmp_path)


def test_checksum_without_trailing_newline_ok(tmp_path: Path) -> None:
    z = build_csv_zip(CSV)
    server = ArchiveServer(
        {
            FILE.url: (200, z),
            FILE.checksum_url: (200, build_checksum(z, FNAME, trailing_newline=False)),
        }
    )
    with server.client() as c:
        assert download_archive(c, FILE, tmp_path) is not None


def _download_zip(tmp_path: Path, z: bytes) -> None:
    server, _ = _serve(z)
    with server.client() as c:
        download_archive(c, FILE, tmp_path)


def test_zip_with_two_csv_members_raises(tmp_path: Path) -> None:
    with pytest.raises(ArchiveError, match="exactly one CSV"):
        _download_zip(tmp_path, build_zip({"a.csv": b"1\n", "b.csv": b"2\n"}))
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


def test_zip_with_no_csv_member_raises(tmp_path: Path) -> None:
    with pytest.raises(ArchiveError, match="exactly one CSV"):
        _download_zip(tmp_path, build_zip({"a.txt": b"1\n"}))


def test_corrupt_zip_raises(tmp_path: Path) -> None:
    with pytest.raises(ArchiveError, match="corrupt zip"):
        _download_zip(tmp_path, b"this is not a zip")


# --- kline parsing ---------------------------------------------------------------------

ROW_MS = (
    "1704067200000,42283.58,44184.10,42180.77,44179.55,27174.29903,"
    "1704153599999,1162870513.34,1043852,13517.74,578460929.45,0"
)
ROW_US = ROW_MS.replace("1704067200000", "1735689600000000", 1)
HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore"
)


def _row_ts(day: int) -> str:
    return f"{1704067200000 + day * 86_400_000}" + ROW_MS[len("1704067200000") :]


def test_headerless_kline_row() -> None:
    df = parse_klines_csv((ROW_MS + "\n").encode())
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["ts"] == datetime(2024, 1, 1, tzinfo=UTC)
    assert row["open"] == pytest.approx(42283.58, abs=1e-9)
    assert row["high"] == pytest.approx(44184.10, abs=1e-9)
    assert row["low"] == pytest.approx(42180.77, abs=1e-9)
    assert row["close"] == pytest.approx(44179.55, abs=1e-9)
    assert row["volume"] == pytest.approx(27174.29903, abs=1e-9)
    assert row["quote_volume"] == pytest.approx(1162870513.34, abs=1e-9)
    assert row["trades"] == 1043852
    assert df.schema == pl.Schema(dict(KLINE_SCHEMA))


def test_microsecond_row_and_mixed_file() -> None:
    us = parse_klines_csv(ROW_US.encode())
    assert us["ts"][0] == datetime(2025, 1, 1, tzinfo=UTC)
    mixed = parse_klines_csv(f"{ROW_MS}\n{ROW_US}\n".encode())
    assert mixed["ts"].to_list() == [
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 1, tzinfo=UTC),
    ]


def test_threshold_boundary() -> None:
    just_below = ROW_MS.replace("1704067200000", str(10**14 - 1), 1)
    at = ROW_MS.replace("1704067200000", str(10**14), 1)
    ts = parse_klines_csv(f"{just_below}\n{at}".encode())["ts"].dt.epoch("us").to_list()
    assert ts == [10**14, (10**14 - 1) * 1000]  # sorted by ts; ms value scaled to us


def test_header_row_gives_identical_output() -> None:
    body = f"{ROW_MS}\n{ROW_US}\n"
    assert parse_klines_csv(f"{HEADER}\n{body}".encode()).equals(parse_klines_csv(body.encode()))


def test_kline_rows_sorted_and_duplicates_kept() -> None:
    raw = f"{_row_ts(2)}\n{_row_ts(0)}\n{_row_ts(1)}\n{_row_ts(1)}\n".encode()
    df = parse_klines_csv(raw)
    assert df.height == 4
    days = [t.day for t in df["ts"].to_list()]
    assert days == [1, 2, 2, 3]
    assert df.schema == pl.Schema(dict(KLINE_SCHEMA))


def test_empty_klines() -> None:
    for raw in (b"", HEADER.encode() + b"\n"):
        df = parse_klines_csv(raw)
        assert df.height == 0
        assert df.schema == pl.Schema(dict(KLINE_SCHEMA))


def test_short_row_mentions_line_number() -> None:
    short = ",".join(ROW_MS.split(",")[:11])
    with pytest.raises(ArchiveError, match="line 2"):
        parse_klines_csv(f"{ROW_MS}\n{short}\n".encode())


def test_unparseable_number_mentions_line_number() -> None:
    bad = ROW_MS.replace("42283.58", "abc", 1)
    with pytest.raises(ArchiveError, match="line 3"):
        parse_klines_csv(f"{HEADER}\n{ROW_MS}\n{bad}\n".encode())


# --- funding parsing -------------------------------------------------------------------

FUND_HEADER = "calc_time,funding_interval_hours,last_funding_rate"
FUND_ROW = "1704067200000,8,0.00010000"


def test_funding_with_and_without_header() -> None:
    with_header = parse_funding_csv(f"{FUND_HEADER}\n{FUND_ROW}\n".encode())
    headerless = parse_funding_csv(f"{FUND_ROW}\n".encode())
    for df in (with_header, headerless):
        row = df.row(0, named=True)
        assert row["ts"] == datetime(2024, 1, 1, tzinfo=UTC)
        assert row["rate"] == pytest.approx(0.0001, abs=1e-12)
        assert row["mark_price"] is None
        assert df.schema == pl.Schema(dict(FUNDING_SCHEMA))
    assert with_header.equals(headerless)


def test_funding_header_locates_columns_by_name() -> None:
    raw = "last_funding_rate,calc_time\n0.0002,1704096000000\n"
    df = parse_funding_csv(raw.encode())
    assert df["rate"].to_list() == [pytest.approx(0.0002, abs=1e-12)]
    assert df["ts"][0] == datetime(2024, 1, 1, 8, tzinfo=UTC)


def test_funding_sorted_micros_and_empty() -> None:
    raw = "1735718400000000,8,0.0002\n1704067200000,8,0.0001\n"
    df = parse_funding_csv(raw.encode())
    assert df["ts"].to_list() == [
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 1, 8, tzinfo=UTC),
    ]
    empty = parse_funding_csv(FUND_HEADER.encode())
    assert empty.height == 0
    assert empty.schema == pl.Schema(dict(FUNDING_SCHEMA))


def test_funding_bad_rows_raise() -> None:
    with pytest.raises(ArchiveError, match="line 2"):
        parse_funding_csv(f"{FUND_ROW}\n1704067200000,8\n".encode())
    with pytest.raises(ArchiveError, match="line 1"):
        parse_funding_csv(b"1704067200000,8,oops\n")
    with pytest.raises(ArchiveError, match="header"):
        parse_funding_csv(b"a,b,c\n1,2,3\n")
