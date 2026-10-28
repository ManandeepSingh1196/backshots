from __future__ import annotations

from datetime import date

import pytest

from bt.core.errors import DataError
from bt.data.fetch_binance import (
    daily_kline_archive_files,
    funding_archive_files,
    kline_archive_files,
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
