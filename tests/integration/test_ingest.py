"""Integration tests for bt.data.ingest: fake archive server + fake ccxt exchange. No network."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from bt.core.errors import ChecksumMismatchError, TimezoneError
from bt.data.catalog import Catalog
from bt.data.fetch_binance import (
    ArchiveFile,
    daily_kline_archive_files,
    funding_archive_files,
    kline_archive_files,
)
from bt.data.ingest import (
    IngestResult,
    check_catalog,
    ingest_funding,
    ingest_instruments,
    ingest_klines,
)
from bt.data.validate import Gap
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

DAY = timedelta(days=1)
NOW = datetime(2024, 3, 10, 1, 0, tzinfo=UTC)
KLINES_TAG = "ccxt:binance:klines"


def at(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def days(first: date, last: date) -> list[date]:
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def price(d: date) -> float:
    return 100.0 + (d - date(2024, 1, 1)).days


def kline_csv(ds: Iterable[date], close_adjust: float = 0.0) -> bytes:
    lines: list[str] = []
    for d in ds:
        p = price(d) + close_adjust
        open_ms = epoch_ms(at(d))
        close_ms = open_ms + 86_400_000 - 1
        ohlc = f"{p:.2f},{p + 1:.2f},{p - 1:.2f},{p:.2f}"
        lines.append(f"{open_ms},{ohlc},10.0,{close_ms},1000.0,42,5.0,500.0,0")
    return ("\n".join(lines) + "\n").encode()


def publish(responses: dict[str, Response], file: ArchiveFile, csv: bytes) -> None:
    zip_bytes = build_csv_zip(csv)
    responses[file.url] = (200, zip_bytes)
    responses[file.checksum_url] = (200, build_checksum(zip_bytes, file.url.rsplit("/", 1)[1]))


def monthly(start: date, end: date, today: date) -> dict[str, ArchiveFile]:
    files = kline_archive_files("perp", "1d", "BTCUSDT", start, end, today)
    return {f.period: f for f in files}


def daily_feb() -> dict[str, ArchiveFile]:
    files = daily_kline_archive_files("perp", "1d", "BTCUSDT", date(2024, 2, 1))
    return {f.period: f for f in files}


def exchange_with(ds: Iterable[date]) -> FakeExchange:
    fake = FakeExchange()
    fake.set_klines(
        "fapiPublicGetKlines",
        "BTCUSDT",
        "1d",
        [kline_row(epoch_ms(at(d)), epoch_ms(at(d)) + 86_399_999, price(d)) for d in ds],
    )
    return fake


def standard_responses(
    today: date = date(2024, 3, 10), daily_until: date = date(2024, 3, 8)
) -> dict[str, Response]:
    """Jan + Feb monthly and daily files for 03-01..daily_until."""
    files = monthly(date(2024, 1, 1), today, today)
    responses: dict[str, Response] = {}
    publish(responses, files["2024-01"], kline_csv(days(date(2024, 1, 1), date(2024, 1, 31))))
    publish(responses, files["2024-02"], kline_csv(days(date(2024, 2, 1), date(2024, 2, 29))))
    for d in days(date(2024, 3, 1), daily_until):
        publish(responses, files[f"{d:%Y-%m-%d}"], kline_csv([d]))
    return responses


@pytest.fixture
def catalog(tmp_path: Path) -> Catalog:
    return Catalog(tmp_path / "data")


def run(
    catalog: Catalog,
    responses: dict[str, Response],
    fake: FakeExchange,
    *,
    start: date = date(2024, 1, 1),
    end: date | None = None,
    now: datetime = NOW,
) -> tuple[IngestResult, ArchiveServer]:
    server = ArchiveServer(responses)
    with server.client() as client:
        result = ingest_klines(catalog, client, fake, "perp", "1d", "BTCUSDT", start, end, now)
    return result, server


def test_klines_archive_plus_ccxt_tail(catalog: Catalog) -> None:
    fake = exchange_with(days(date(2024, 3, 9), date(2024, 3, 10)))
    result, _ = run(catalog, standard_responses(), fake)
    entry = result.entry
    assert entry is not None
    assert result.dataset == "perp/klines/1d/BTCUSDT"
    assert result.report.ok
    assert entry.rows == 69
    assert entry.first_ts == at(date(2024, 1, 1))
    assert entry.last_ts == at(date(2024, 3, 9))
    assert entry.ingested_at == NOW
    assert result.missing_archives == ()
    assert entry.sources[-1] == KLINES_TAG
    assert len(entry.sources) == 11
    assert len(entry.checksums) == 10
    assert KLINES_TAG not in entry.checksums
    assert result.report.gaps == ()
    assert catalog.read_klines("perp", "1d", "BTCUSDT").height == 69


def test_pre_listing_404s_are_skipped_silently(catalog: Catalog) -> None:
    fake = exchange_with(days(date(2024, 3, 9), date(2024, 3, 10)))
    result, _ = run(catalog, standard_responses(), fake, start=date(2023, 10, 1))
    assert result.entry is not None
    assert result.entry.rows == 69
    assert result.entry.first_ts == at(date(2024, 1, 1))
    assert result.missing_archives == ()
    assert len(result.entry.checksums) == 10


def test_monthly_404_falls_back_to_daily(catalog: Catalog) -> None:
    responses = standard_responses()
    feb = monthly(date(2024, 1, 1), date(2024, 3, 10), date(2024, 3, 10))["2024-02"]
    del responses[feb.url], responses[feb.checksum_url]
    for d, f in daily_feb().items():
        publish(responses, f, kline_csv([date.fromisoformat(d)]))
    fake = exchange_with(days(date(2024, 3, 9), date(2024, 3, 10)))
    result, _ = run(catalog, responses, fake)
    assert result.entry is not None
    assert result.entry.rows == 69
    assert result.missing_archives == ()
    assert feb.url not in result.entry.sources
    assert len(result.entry.checksums) == 1 + 29 + 8


def test_daily_fallback_missing_day_is_reported_and_gap(catalog: Catalog) -> None:
    responses = standard_responses()
    feb = monthly(date(2024, 1, 1), date(2024, 3, 10), date(2024, 3, 10))["2024-02"]
    del responses[feb.url], responses[feb.checksum_url]
    dailies = daily_feb()
    for d, f in dailies.items():
        if d != "2024-02-14":
            publish(responses, f, kline_csv([date.fromisoformat(d)]))
    fake = exchange_with(days(date(2024, 3, 9), date(2024, 3, 10)))
    result, _ = run(catalog, responses, fake)
    assert result.entry is not None
    assert result.entry.rows == 68
    assert result.missing_archives == (dailies["2024-02-14"].url,)
    assert len(result.report.gaps) == 1
    gap = result.report.gaps[0]
    assert gap == Gap(at(date(2024, 2, 13)), at(date(2024, 2, 15)), 1)


def test_checksum_mismatch_propagates_and_stores_nothing(catalog: Catalog) -> None:
    responses = standard_responses()
    jan = monthly(date(2024, 1, 1), date(2024, 3, 10), date(2024, 3, 10))["2024-01"]
    responses[jan.checksum_url] = (200, build_checksum(b"other", "x.zip"))
    fake = exchange_with(days(date(2024, 3, 9), date(2024, 3, 10)))
    with pytest.raises(ChecksumMismatchError):
        run(catalog, responses, fake)
    assert not catalog.klines_path("perp", "1d", "BTCUSDT").exists()
    assert catalog.manifest() == {}


def test_duplicate_disagreement_is_not_stored_and_keeps_old_file(catalog: Catalog) -> None:
    seed_df = pl.DataFrame(
        {
            "ts": [at(date(2024, 1, 1))],
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [1.0],
            "quote_volume": [1.0],
            "trades": [1],
        },
        schema={
            "ts": pl.Datetime("us", "UTC"),
            "open": pl.Float64(),
            "high": pl.Float64(),
            "low": pl.Float64(),
            "close": pl.Float64(),
            "volume": pl.Float64(),
            "quote_volume": pl.Float64(),
            "trades": pl.Int64(),
        },
    )
    catalog.write_klines("perp", "1d", "BTCUSDT", seed_df, [], {}, NOW)
    path = catalog.klines_path("perp", "1d", "BTCUSDT")
    before = path.read_bytes()
    manifest_before = catalog.manifest()

    files = monthly(date(2024, 1, 1), date(2024, 2, 29), date(2024, 3, 10))
    responses: dict[str, Response] = {}
    jan_days = [*days(date(2024, 1, 1), date(2024, 1, 31)), date(2024, 2, 1)]
    publish(responses, files["2024-01"], kline_csv(jan_days, close_adjust=50.0))
    publish(responses, files["2024-02"], kline_csv(days(date(2024, 2, 1), date(2024, 2, 29))))
    result, _ = run(catalog, responses, FakeExchange(), end=date(2024, 2, 29))
    assert result.entry is None
    assert any(e.startswith("duplicate ts") for e in result.report.errors)
    assert path.read_bytes() == before
    assert catalog.manifest() == manifest_before


def test_end_before_tail_makes_no_daily_or_ccxt_calls(catalog: Catalog) -> None:
    fake = exchange_with(days(date(2024, 2, 1), date(2024, 3, 10)))
    result, server = run(catalog, standard_responses(), fake, end=date(2024, 1, 31))
    assert result.entry is not None
    assert result.entry.rows == 31
    assert result.entry.last_ts == at(date(2024, 1, 31))
    assert not any("/daily/" in u for u in server.requested_urls)
    assert fake.calls_to("fapiPublicGetKlines") == []
    assert KLINES_TAG not in result.entry.sources


def test_rerun_is_cached_and_byte_identical(catalog: Catalog) -> None:
    responses = standard_responses(daily_until=date(2024, 3, 9))
    fake = exchange_with(days(date(2024, 3, 9), date(2024, 3, 10)))
    first, server1 = run(catalog, responses, fake)
    assert first.entry is not None
    assert server1.request_count > 0
    path = catalog.klines_path("perp", "1d", "BTCUSDT")
    bytes1 = path.read_bytes()
    second, server2 = run(catalog, responses, fake)
    assert second.entry is not None
    assert server2.request_count == 0
    assert path.read_bytes() == bytes1
    assert second.entry.sha256 == first.entry.sha256


def test_lag_hole_is_filled_by_ccxt_and_later_archives_unused(catalog: Catalog) -> None:
    now = datetime(2024, 3, 2, 1, 0, tzinfo=UTC)
    today = now.date()
    files = monthly(date(2024, 1, 1), today, today)
    dailies = daily_feb()
    responses: dict[str, Response] = {}
    publish(responses, files["2024-01"], kline_csv(days(date(2024, 1, 1), date(2024, 1, 31))))
    for d in days(date(2024, 2, 1), date(2024, 2, 28)):
        publish(responses, dailies[f"{d:%Y-%m-%d}"], kline_csv([d]))
    publish(responses, files["2024-03-01"], kline_csv([date(2024, 3, 1)], close_adjust=7.0))
    fake = exchange_with(days(date(2024, 2, 29), date(2024, 3, 2)))
    result, _ = run(catalog, responses, fake, now=now)
    assert result.entry is not None
    assert result.missing_archives == ()
    assert result.report.gaps == ()
    assert files["2024-03-01"].url not in result.entry.sources
    assert result.entry.sources[-1] == KLINES_TAG
    assert result.entry.rows == 31 + 29 + 1
    assert result.entry.last_ts == at(date(2024, 3, 1))
    df = catalog.read_klines("perp", "1d", "BTCUSDT")
    mar1 = df.filter(pl.col("ts") == at(date(2024, 3, 1)))["close"][0]
    assert mar1 == pytest.approx(price(date(2024, 3, 1)), abs=1e-9)


def test_no_data_at_all_is_empty_and_not_stored(catalog: Catalog) -> None:
    result, _ = run(catalog, {}, FakeExchange())
    assert result.entry is None
    assert "empty" in result.report.errors
    assert catalog.manifest() == {}


def test_naive_now_raises(catalog: Catalog) -> None:
    with pytest.raises(TimezoneError):
        run(catalog, {}, FakeExchange(), now=datetime(2024, 3, 10, 1, 0))  # noqa: DTZ001


# --------------------------------------------------------------------------- funding

FUNDING_TAG = "ccxt:binance:fundingRate"
H8 = timedelta(hours=8)


def events(first: datetime, last: datetime) -> list[datetime]:
    out: list[datetime] = []
    t = first
    while t <= last:
        out.append(t)
        t += H8
    return out


def funding_csv(ts: Iterable[datetime]) -> bytes:
    lines = ["calc_time,funding_interval_hours,last_funding_rate"]
    lines += [f"{epoch_ms(t)},8,0.00010000" for t in ts]
    return ("\n".join(lines) + "\n").encode()


def funding_exchange(ts: Iterable[datetime]) -> FakeExchange:
    fake = FakeExchange()
    fake.set_funding("BTCUSDT", [funding_row(epoch_ms(t), "0.0002", "42000.5") for t in ts])
    return fake


def funding_responses(today: date, feb: bool = True) -> dict[str, Response]:
    files = {f.period: f for f in funding_archive_files("BTCUSDT", date(2024, 1, 1), today, today)}
    responses: dict[str, Response] = {}
    jan = events(datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 31, 16, tzinfo=UTC))
    publish(responses, files["2024-01"], funding_csv(jan))
    if feb:
        feb_ev = events(datetime(2024, 2, 1, tzinfo=UTC), datetime(2024, 2, 29, 16, tzinfo=UTC))
        publish(responses, files["2024-02"], funding_csv(feb_ev))
    return responses


def run_funding(
    catalog: Catalog,
    responses: dict[str, Response],
    fake: FakeExchange,
    now: datetime = NOW,
    end: date | None = None,
) -> tuple[IngestResult, ArchiveServer]:
    server = ArchiveServer(responses)
    with server.client() as client:
        result = ingest_funding(catalog, client, fake, "BTCUSDT", date(2024, 1, 1), end, now)
    return result, server


def test_funding_archive_plus_ccxt_tail(catalog: Catalog) -> None:
    tail = events(datetime(2024, 3, 1, tzinfo=UTC), datetime(2024, 3, 10, 8, tzinfo=UTC))
    assert len(tail) == 29  # 28 up to 03-10T00:00 plus 03-10T08:00 (after now)
    result, _ = run_funding(catalog, funding_responses(NOW.date()), funding_exchange(tail))
    entry = result.entry
    assert entry is not None
    assert result.dataset == "perp/funding/BTCUSDT"
    assert entry.rows == 93 + 87 + 28 == 208
    assert entry.last_ts == datetime(2024, 3, 10, tzinfo=UTC)
    assert entry.sources[-1] == FUNDING_TAG
    assert len(entry.checksums) == 2
    assert result.missing_archives == ()
    df = catalog.read_funding("BTCUSDT")
    assert df.filter(pl.col("ts") < at(date(2024, 3, 1)))["mark_price"].null_count() == 180
    assert df.filter(pl.col("ts") >= at(date(2024, 3, 1)))["mark_price"].null_count() == 0


def test_funding_monthly_lag_404_is_filled_by_ccxt(catalog: Catalog) -> None:
    now = datetime(2024, 3, 2, 1, 0, tzinfo=UTC)
    responses = funding_responses(now.date(), feb=False)
    tail = events(datetime(2024, 2, 1, tzinfo=UTC), datetime(2024, 3, 2, 0, tzinfo=UTC))
    result, _ = run_funding(catalog, responses, funding_exchange(tail), now=now)
    assert result.entry is not None
    assert result.missing_archives == ()
    assert result.report.gaps == ()
    assert result.entry.rows == 93 + 87 + 3 + 1  # Feb, 03-01 x3, 03-02T00:00
    assert len(result.entry.checksums) == 1


def test_funding_monthly_404_not_lag_is_missing(catalog: Catalog) -> None:
    responses = funding_responses(NOW.date(), feb=False)
    files = {
        f.period: f
        for f in funding_archive_files("BTCUSDT", date(2024, 1, 1), NOW.date(), NOW.date())
    }
    tail = events(datetime(2024, 3, 1, tzinfo=UTC), datetime(2024, 3, 10, tzinfo=UTC))
    result, _ = run_funding(catalog, responses, funding_exchange(tail))
    assert result.entry is not None
    assert result.missing_archives == (files["2024-02"].url,)
    assert len(result.report.gaps) == 1


def test_funding_end_excludes_tail_events_after_end(catalog: Catalog) -> None:
    fake = funding_exchange(
        events(datetime(2024, 3, 1, tzinfo=UTC), datetime(2024, 3, 9, tzinfo=UTC))
    )
    result, _ = run_funding(catalog, funding_responses(NOW.date()), fake, end=date(2024, 2, 29))
    assert result.entry is not None
    assert result.entry.rows == 180
    assert result.entry.last_ts == datetime(2024, 2, 29, 16, tzinfo=UTC)
    assert FUNDING_TAG not in result.entry.sources
    # the tail is bounded at end + 1 day, so the 03-01 events are never even requested
    assert len(fake.calls_to("fapiPublicGetFundingRate")) == 1


def test_funding_naive_now_raises(catalog: Catalog) -> None:
    with pytest.raises(TimezoneError):
        run_funding(catalog, {}, FakeExchange(), now=datetime(2024, 3, 10, 1, 0))  # noqa: DTZ001


# --------------------------------------------------------------------------- instruments


def test_ingest_instruments_writes_sorted_snapshot(catalog: Catalog) -> None:
    fake = FakeExchange(markets=[spot_market(), perp_market()])
    out = ingest_instruments(catalog, fake, ["BTCUSDT"], ["spot", "perp"], NOW)
    assert [i.symbol for i in out] == ["BTCUSDT", "BTCUSDT-PERP"]
    stored, fetched_at = catalog.read_instruments()
    assert [i.symbol for i in stored] == ["BTCUSDT", "BTCUSDT-PERP"]
    assert fetched_at == NOW


def test_ingest_instruments_naive_now_raises(catalog: Catalog) -> None:
    fake = FakeExchange(markets=[spot_market()])
    with pytest.raises(TimezoneError):
        ingest_instruments(catalog, fake, ["BTCUSDT"], ["spot"], datetime(2024, 3, 10))  # noqa: DTZ001
    assert not (catalog.root / "instruments.json").exists()


# --------------------------------------------------------------------------- check_catalog

SCHEMA = {
    "ts": pl.Datetime("us", "UTC"),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Float64(),
    "quote_volume": pl.Float64(),
    "trades": pl.Int64(),
}


def small_klines(n: int) -> pl.DataFrame:
    ds = days(date(2024, 1, 1), date(2024, 1, 1) + timedelta(days=n - 1))
    return pl.DataFrame(
        {
            "ts": [at(d) for d in ds],
            "open": [10.0] * n,
            "high": [12.0] * n,
            "low": [9.0] * n,
            "close": [11.0] * n,
            "volume": [1.0] * n,
            "quote_volume": [11.0] * n,
            "trades": [3] * n,
        },
        schema=SCHEMA,
    )


@pytest.fixture
def two_datasets(catalog: Catalog) -> Catalog:
    catalog.write_klines("perp", "1d", "BTCUSDT", small_klines(5), ["s"], {"s": "0" * 64}, NOW)
    catalog.write_klines("spot", "1d", "ETHUSDT", small_klines(4), [], {}, NOW)
    return catalog


def test_check_catalog_clean(two_datasets: Catalog) -> None:
    reports = check_catalog(two_datasets)
    assert [r.dataset for r in reports] == ["perp/klines/1d/BTCUSDT", "spot/klines/1d/ETHUSDT"]
    assert all(r.ok for r in reports)
    assert [r.rows for r in reports] == [5, 4]


def test_check_catalog_empty_root(catalog: Catalog) -> None:
    assert check_catalog(catalog) == []


def test_check_catalog_row_dropped_behind_its_back(two_datasets: Catalog) -> None:
    path = two_datasets.klines_path("perp", "1d", "BTCUSDT")
    small_klines(5).head(4).write_parquet(path)
    reports = check_catalog(two_datasets)
    bad = reports[0]
    assert "manifest mismatch: rows 5 != 4" in bad.errors
    assert any(e.startswith("manifest mismatch: last_ts") for e in bad.errors)
    assert any(e.startswith("manifest mismatch: sha256") for e in bad.errors)
    assert reports[1].ok


def test_check_catalog_detects_changed_close_by_sha(two_datasets: Catalog) -> None:
    path = two_datasets.klines_path("perp", "1d", "BTCUSDT")
    df = pl.read_parquet(path)
    df.with_columns(
        pl.when(pl.col("ts") == df["ts"][2]).then(11.5).otherwise(pl.col("close")).alias("close")
    ).write_parquet(path)
    bad = check_catalog(two_datasets)[0]
    assert len(bad.errors) == 1
    assert bad.errors[0].startswith("manifest mismatch: sha256")


def test_check_catalog_missing_file(two_datasets: Catalog) -> None:
    two_datasets.klines_path("spot", "1d", "ETHUSDT").unlink()
    reports = check_catalog(two_datasets)
    assert reports[0].ok
    assert reports[1].errors == ("missing file",)


def test_check_catalog_stray_parquet(two_datasets: Catalog) -> None:
    stray = two_datasets.klines_path("perp", "4h", "SOLUSDT")
    stray.parent.mkdir(parents=True)
    small_klines(2).write_parquet(stray)
    reports = check_catalog(two_datasets)
    assert [r.dataset for r in reports] == [
        "perp/klines/1d/BTCUSDT",
        "perp/klines/4h/SOLUSDT",
        "spot/klines/1d/ETHUSDT",
    ]
    assert reports[1].errors == ("not in manifest",)
    assert reports[0].ok
    assert reports[2].ok


def test_check_catalog_validates_content(catalog: Catalog) -> None:
    catalog.write_klines("perp", "1d", "BTCUSDT", small_klines(3), [], {}, NOW)
    path = catalog.klines_path("perp", "1d", "BTCUSDT")
    broken = small_klines(3).with_columns(pl.lit(-1.0).alias("open"))
    broken.write_parquet(path)
    (report,) = check_catalog(catalog)
    assert any(e.startswith("non-positive open") for e in report.errors)
