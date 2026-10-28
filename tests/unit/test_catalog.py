"""Tests for bt.data.catalog."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from bt.core.errors import DataError, DataValidationError, InvariantError, TimezoneError
from bt.core.types import Instrument, InstrumentKind
from bt.data.catalog import Catalog, ManifestEntry
from bt.data.validate import FUNDING_SCHEMA, KLINE_SCHEMA

D0 = datetime(2024, 1, 1, tzinfo=UTC)
T0 = datetime(2024, 3, 10, 12, tzinfo=UTC)
SRC = ("https://data.binance.vision/x.zip",)
SUMS = {"https://data.binance.vision/x.zip": "ab" * 32}


def D(n: int) -> datetime:  # noqa: N802
    return D0 + timedelta(days=n)


def klines_df(n: int = 5, step: timedelta = timedelta(days=1)) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": [D0 + i * step for i in range(n)],
            "open": [100.0 + i for i in range(n)],
            "high": [110.0 + i for i in range(n)],
            "low": [90.0 + i for i in range(n)],
            "close": [105.0 + i for i in range(n)],
            "volume": [10.0] * n,
            "quote_volume": [1000.0] * n,
            "trades": list(range(1, n + 1)),
        },
        schema=dict(KLINE_SCHEMA),
    )


def put(cat: Catalog, df: pl.DataFrame | None = None, at: datetime = T0) -> ManifestEntry:
    return cat.write_klines(
        "perp", "1d", "BTCUSDT", klines_df() if df is None else df, SRC, SUMS, at
    )


def test_paths(tmp_path: Path) -> None:
    cat = Catalog(tmp_path)
    assert cat.root == tmp_path
    assert cat.klines_path("perp", "1d", "BTCUSDT") == (
        tmp_path / "binance/perp/klines/1d/BTCUSDT.parquet"
    )
    assert cat.funding_path("BTCUSDT") == tmp_path / "binance/perp/funding/BTCUSDT.parquet"
    assert cat.cache_dir() == tmp_path / "cache"


@pytest.mark.parametrize("symbol", ["../etc", "btcusdt", "", "A/B", "X"])
def test_bad_symbols_rejected(tmp_path: Path, symbol: str) -> None:
    cat = Catalog(tmp_path)
    with pytest.raises(DataError):
        cat.klines_path("perp", "1d", symbol)
    with pytest.raises(DataError):
        cat.funding_path(symbol)


def test_bad_market_and_timeframe_rejected(tmp_path: Path) -> None:
    cat = Catalog(tmp_path)
    with pytest.raises(DataError):
        cat.klines_path(cast(Any, "futures"), "1d", "BTCUSDT")
    with pytest.raises(DataError):
        cat.klines_path("perp", cast(Any, "1h"), "BTCUSDT")


def test_klines_round_trip(tmp_path: Path) -> None:
    cat = Catalog(tmp_path)
    df = klines_df()
    put(cat, df)
    back = cat.read_klines("perp", "1d", "BTCUSDT")
    assert back.equals(df)
    assert list(back.schema.items()) == list(KLINE_SCHEMA.items())


def test_entry_and_manifest(tmp_path: Path) -> None:
    cat = Catalog(tmp_path)
    entry = put(cat)
    path = cat.klines_path("perp", "1d", "BTCUSDT")
    assert entry.path == "binance/perp/klines/1d/BTCUSDT.parquet"
    assert entry.rows == 5
    assert entry.first_ts == D(0)
    assert entry.last_ts == D(4)
    assert entry.ingested_at == T0
    assert entry.sources == SRC
    assert dict(entry.checksums) == SUMS
    assert entry.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert cat.manifest()[entry.path] == entry


def test_second_dataset_keeps_first(tmp_path: Path) -> None:
    cat = Catalog(tmp_path)
    first = put(cat)
    cat.write_klines("spot", "4h", "ETHUSDT", klines_df(step=timedelta(hours=4)), (), {}, T0)
    m = cat.manifest()
    assert len(m) == 2
    assert m[first.path] == first
    assert "binance/spot/klines/4h/ETHUSDT.parquet" in m


def test_invalid_frame_touches_nothing(tmp_path: Path) -> None:
    cat = Catalog(tmp_path)
    put(cat)
    path = cat.klines_path("perp", "1d", "BTCUSDT")
    manifest = tmp_path / "manifest.json"
    before = (path.read_bytes(), manifest.read_bytes())
    bad = pl.concat([klines_df(), klines_df(1)])
    with pytest.raises(DataValidationError) as ei:
        put(cat, bad)
    assert "perp/klines/1d/BTCUSDT" in str(ei.value)
    assert "duplicate" in str(ei.value)
    assert (path.read_bytes(), manifest.read_bytes()) == before


def test_invalid_frame_on_fresh_root_creates_nothing(tmp_path: Path) -> None:
    cat = Catalog(tmp_path / "root")
    with pytest.raises(DataValidationError):
        put(cat, pl.concat([klines_df(), klines_df(1)]))
    assert not (tmp_path / "root").exists()


def test_atomic_write_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cat = Catalog(tmp_path)
    put(cat)
    path = cat.klines_path("perp", "1d", "BTCUSDT")
    manifest = tmp_path / "manifest.json"
    before = (path.read_bytes(), manifest.read_bytes())

    def boom(self: pl.DataFrame, file: Path, *args: object, **kwargs: object) -> None:
        Path(file).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", boom)
    with pytest.raises(OSError, match="disk full"):
        put(cat, klines_df(3))
    assert (path.read_bytes(), manifest.read_bytes()) == before
    assert sorted(p.name for p in path.parent.iterdir()) == ["BTCUSDT.parquet"]
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == ["manifest.json"]


def test_naive_ingested_at(tmp_path: Path) -> None:
    with pytest.raises(TimezoneError):
        put(Catalog(tmp_path), at=datetime(2024, 3, 10, 12))  # noqa: DTZ001


def test_ingested_at_normalised_before_storing(tmp_path: Path) -> None:
    cat = Catalog(tmp_path)
    entry = put(cat, at=datetime(2024, 3, 10, 12, tzinfo=ZoneInfo("Europe/London")))
    assert entry.ingested_at.tzinfo is UTC
    with pytest.raises(TimezoneError):
        put(cat, at=datetime(2024, 3, 10, 12, tzinfo=timezone(timedelta(hours=2))))


def test_determinism(tmp_path: Path) -> None:
    a, b = Catalog(tmp_path / "a"), Catalog(tmp_path / "b")
    ea, eb = put(a), put(b)
    assert a.klines_path("perp", "1d", "BTCUSDT").read_bytes() == (
        b.klines_path("perp", "1d", "BTCUSDT").read_bytes()
    )
    assert (tmp_path / "a/manifest.json").read_bytes() == (
        tmp_path / "b/manifest.json"
    ).read_bytes()
    assert ea.sha256 == eb.sha256


def test_manifest_json_format(tmp_path: Path) -> None:
    cat = Catalog(tmp_path)
    entry = put(cat)
    text = (tmp_path / "manifest.json").read_text()
    assert text.endswith("}\n")
    doc = json.loads(text)
    assert doc["version"] == 1
    f = doc["files"][entry.path]
    assert f["first_ts"] == "2024-01-01T00:00:00+00:00"
    assert f["sha256"] == entry.sha256
    assert f["sources"] == list(SRC)


def test_missing_and_empty(tmp_path: Path) -> None:
    cat = Catalog(tmp_path)
    assert cat.manifest() == {}
    with pytest.raises(DataError, match="BTCUSDT"):
        cat.read_klines("perp", "1d", "BTCUSDT")


def test_manifest_unknown_version(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text('{"version": 2, "files": {}}\n')
    with pytest.raises(DataError, match="version"):
        Catalog(tmp_path).manifest()


@pytest.mark.parametrize(
    "text",
    [
        "{",
        "[]",
        '{"version": 1}',
        '{"version": 1, "files": {"a": {"path": "a"}}}',
        '{"version": 1, "files": {"a": 3}}',
    ],
)
def test_manifest_corrupt(tmp_path: Path, text: str) -> None:
    (tmp_path / "manifest.json").write_text(text)
    with pytest.raises(DataError, match=r"manifest\.json"):
        Catalog(tmp_path).manifest()


def test_manifest_bad_entry_types(tmp_path: Path) -> None:
    cat = Catalog(tmp_path)
    entry = put(cat)
    doc = json.loads((tmp_path / "manifest.json").read_text())
    doc["files"][entry.path]["rows"] = "5"
    (tmp_path / "manifest.json").write_text(json.dumps(doc))
    with pytest.raises(DataError):
        cat.manifest()
    doc["files"][entry.path]["rows"] = 5
    doc["files"][entry.path]["first_ts"] = "2024-01-01T00:00:00"
    (tmp_path / "manifest.json").write_text(json.dumps(doc))
    with pytest.raises(DataError):
        cat.manifest()


def test_read_klines_wrong_schema(tmp_path: Path) -> None:
    cat = Catalog(tmp_path)
    put(cat)
    path = cat.klines_path("perp", "1d", "BTCUSDT")
    pl.read_parquet(path).with_columns(pl.col("ts").dt.cast_time_unit("ms")).write_parquet(path)
    with pytest.raises(DataError, match=r"BTCUSDT\.parquet"):
        cat.read_klines("perp", "1d", "BTCUSDT")


def test_read_klines_garbage_file(tmp_path: Path) -> None:
    cat = Catalog(tmp_path)
    path = cat.klines_path("perp", "1d", "BTCUSDT")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not parquet")
    with pytest.raises(DataError):
        cat.read_klines("perp", "1d", "BTCUSDT")


def funding_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": [D0 + timedelta(hours=8 * i) for i in range(4)],
            "rate": [0.0001, -0.0002, 0.0001, 0.00005],
            "mark_price": [42000.5, None, 42100.0, None],
        },
        schema=dict(FUNDING_SCHEMA),
    )


def test_funding_round_trip_with_null_mark_price(tmp_path: Path) -> None:
    cat = Catalog(tmp_path)
    df = funding_df()
    entry = cat.write_funding("BTCUSDT", df, ("ccxt:binance:fapiPublicGetFundingRate",), {}, T0)
    assert entry.path == "binance/perp/funding/BTCUSDT.parquet"
    assert entry.rows == 4
    assert entry.last_ts == D0 + timedelta(hours=24)
    back = cat.read_funding("BTCUSDT")
    assert back.equals(df)
    assert back["mark_price"].null_count() == 2
    assert cat.manifest()[entry.path] == entry


def test_funding_invalid_and_missing(tmp_path: Path) -> None:
    cat = Catalog(tmp_path)
    with pytest.raises(DataValidationError, match=r"perp/funding/BTCUSDT"):
        cat.write_funding("BTCUSDT", funding_df().head(0), (), {}, T0)
    with pytest.raises(DataError):
        cat.read_funding("BTCUSDT")
    with pytest.raises(DataError):
        cat.write_funding("btc", funding_df(), (), {}, T0)


def instruments() -> list[Instrument]:
    perp = Instrument(
        symbol="ETHUSDT-PERP",
        venue_symbol="ETHUSDT",
        kind=InstrumentKind.PERP,
        base="ETH",
        quote="USDT",
        min_qty=0.001,
        qty_step=0.001,
        min_notional=5.0,
        price_tick=0.01,
        maker_fee=0.0002,
        taker_fee=0.0005,
        max_leverage=20.0,
        maint_margin_rate=0.005,
    )
    spot = Instrument(
        symbol="BTCUSDT",
        venue_symbol="BTCUSDT",
        kind=InstrumentKind.SPOT,
        base="BTC",
        quote="USDT",
        min_qty=0.00001,
        qty_step=0.00001,
        min_notional=5.0,
        price_tick=0.01,
        maker_fee=0.001,
        taker_fee=0.001,
    )
    return [perp, spot]


def test_instruments_round_trip(tmp_path: Path) -> None:
    cat = Catalog(tmp_path)
    perp, spot = instruments()
    cat.write_instruments([perp, spot], T0)
    doc = json.loads((tmp_path / "instruments.json").read_text())
    assert [i["symbol"] for i in doc["instruments"]] == ["BTCUSDT", "ETHUSDT-PERP"]
    assert doc["instruments"][1]["kind"] == "perp"
    assert doc["fetched_at"] == "2024-03-10T12:00:00+00:00"
    got, fetched = cat.read_instruments()
    assert got == [spot, perp]
    assert fetched == T0


def test_instruments_naive_fetched_at(tmp_path: Path) -> None:
    with pytest.raises(TimezoneError):
        Catalog(tmp_path).write_instruments(instruments(), datetime(2024, 3, 10, 12))  # noqa: DTZ001
    assert not (tmp_path / "instruments.json").exists()


def test_instruments_missing(tmp_path: Path) -> None:
    with pytest.raises(DataError, match=r"instruments"):
        Catalog(tmp_path).read_instruments()


def test_instruments_invariant_violation_is_data_error(tmp_path: Path) -> None:
    cat = Catalog(tmp_path)
    cat.write_instruments(instruments(), T0)
    path = tmp_path / "instruments.json"
    doc = json.loads(path.read_text())
    doc["instruments"][0]["qty_step"] = 0
    path.write_text(json.dumps(doc))
    with pytest.raises(DataError, match=r"instruments\.json") as ei:
        cat.read_instruments()
    assert isinstance(ei.value.__cause__, InvariantError)


@pytest.mark.parametrize("text", ["{", "[]", '{"version": 1}', '{"version": 9, "instruments": []}'])
def test_instruments_corrupt(tmp_path: Path, text: str) -> None:
    (tmp_path / "instruments.json").write_text(text)
    with pytest.raises(DataError, match=r"instruments\.json"):
        Catalog(tmp_path).read_instruments()


def test_instruments_bad_field_types(tmp_path: Path) -> None:
    cat = Catalog(tmp_path)
    cat.write_instruments(instruments(), T0)
    path = tmp_path / "instruments.json"
    doc = json.loads(path.read_text())
    doc["instruments"][0]["kind"] = "future"
    path.write_text(json.dumps(doc))
    with pytest.raises(DataError):
        cat.read_instruments()
    doc["instruments"][0]["kind"] = "spot"
    doc["instruments"][0]["min_qty"] = "1"
    path.write_text(json.dumps(doc))
    with pytest.raises(DataError):
        cat.read_instruments()


def test_instruments_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cat = Catalog(tmp_path)
    cat.write_instruments(instruments(), T0)
    before = (tmp_path / "instruments.json").read_bytes()

    def boom(self: Path, *args: object, **kwargs: object) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", boom)
    with pytest.raises(OSError, match="disk full"):
        cat.write_instruments(instruments()[:1], T0)
    assert (tmp_path / "instruments.json").read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["instruments.json"]
