from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from bt.core.errors import ArchiveError, ChecksumMismatchError, DataError
from bt.data.fetch_binance import (
    daily_kline_archive_files,
    download_archive,
    funding_archive_files,
    kline_archive_files,
)
from tests.support.binance_archive import (
    ArchiveServer,
    build_checksum,
    build_csv_zip,
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
