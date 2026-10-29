"""Tests for bt.runs.fingerprint on the golden_m2 fixture."""

import json
import pickle
import shutil
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bt.core.errors import DataError, InvariantError
from bt.core.types import Instrument
from bt.data.catalog import Catalog
from bt.data.feed import ParquetFeed
from bt.runs.fingerprint import (
    canonical_json,
    data_fingerprint,
    window_coverage,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "golden_m2"
SHA_A = "9e3b517c3d5c4071da7ef1f9a2420e1ce6a0528e7e3ff22d68c95b540054b126"
SHA_B = "0632efa4ef62e9717a8a9d3d6c319b20385481ad4d25e5187c06393ca94fda3d"


def D(day: int) -> datetime:  # noqa: N802
    return datetime(2024, 1, day, tzinfo=UTC)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    dest = tmp_path / "cat"
    shutil.copytree(FIXTURE, dest)
    return dest


def insts(cat: Catalog) -> list[Instrument]:
    items, _ = cat.read_instruments()
    return items


def test_fingerprint_files_and_order(root: Path) -> None:
    cat = Catalog(root)
    a, b = insts(cat)
    fp = data_fingerprint(cat, [a, b], "spot", "1d")
    assert [f.path for f in fp.files] == [
        "binance/spot/klines/1d/AAAUSDT.parquet",
        "binance/spot/klines/1d/BBBUSDT.parquet",
    ]
    assert [f.sha256 for f in fp.files] == [SHA_A, SHA_B]
    assert [f.rows for f in fp.files] == [5, 4]
    assert len(fp.sha256) == 64
    assert fp.sha256 == fp.sha256.lower()
    int(fp.sha256, 16)
    assert data_fingerprint(cat, [b, a], "spot", "1d") == fp
    assert data_fingerprint(cat, [a, b], "spot", "1d") == fp


def test_ingested_at_does_not_change_hash(root: Path) -> None:
    cat = Catalog(root)
    a, b = insts(cat)
    before = data_fingerprint(cat, [a, b], "spot", "1d")
    mp = root / "manifest.json"
    doc = json.loads(mp.read_text())
    for e in doc["files"].values():
        e["ingested_at"] = "2025-05-05T00:00:00+00:00"
        e["sources"] = ["other"]
    mp.write_text(json.dumps(doc))
    assert data_fingerprint(cat, [a, b], "spot", "1d").sha256 == before.sha256


def test_fee_change_changes_hash(root: Path) -> None:
    cat = Catalog(root)
    a, b = insts(cat)
    base = data_fingerprint(cat, [a, b], "spot", "1d")
    other = data_fingerprint(cat, [a, replace(b, taker_fee=0.002)], "spot", "1d")
    assert other.sha256 != base.sha256


def test_modified_file_raises(root: Path) -> None:
    cat = Catalog(root)
    a, b = insts(cat)
    with (root / "binance/spot/klines/1d/AAAUSDT.parquet").open("ab") as f:
        f.write(b"\0")
    with pytest.raises(DataError, match="changed since ingest"):
        data_fingerprint(cat, [a, b], "spot", "1d")


def test_missing_manifest_entry_raises(root: Path) -> None:
    cat = Catalog(root)
    a, b = insts(cat)
    mp = root / "manifest.json"
    doc = json.loads(mp.read_text())
    del doc["files"]["binance/spot/klines/1d/AAAUSDT.parquet"]
    mp.write_text(json.dumps(doc))
    with pytest.raises(DataError, match="not in manifest"):
        data_fingerprint(cat, [a, b], "spot", "1d")


def test_missing_file_empty_and_duplicate(root: Path) -> None:
    cat = Catalog(root)
    a, b = insts(cat)
    with pytest.raises(DataError):
        data_fingerprint(cat, [], "spot", "1d")
    with pytest.raises(DataError):
        data_fingerprint(cat, [a, a], "spot", "1d")
    (root / "binance/spot/klines/1d/BBBUSDT.parquet").unlink()
    with pytest.raises(DataError):
        data_fingerprint(cat, [a, b], "spot", "1d")


def test_canonical_json() -> None:
    assert canonical_json({"b": 1, "a": [1.5, "x"]}) == '{"a":[1.5,"x"],"b":1}'
    with pytest.raises(InvariantError):
        canonical_json({"x": float("nan")})


def test_coverage_full_window() -> None:
    cat = Catalog(FIXTURE)
    a, b = insts(cat)
    cov = window_coverage(cat, [b, a], "spot", "1d", D(1), D(6))
    assert [c.symbol for c in cov] == ["AAAUSDT", "BBBUSDT"]
    assert (cov[0].bars, cov[0].missing) == (5, 0)
    assert (cov[1].bars, cov[1].missing) == (4, 1)
    assert (cov[1].first_ts, cov[1].last_ts) == (D(1), D(5))


def test_coverage_sub_windows() -> None:
    cat = Catalog(FIXTURE)
    a, b = insts(cat)
    cov = window_coverage(cat, [a, b], "spot", "1d", D(2), D(4))
    assert [(c.bars, c.missing) for c in cov] == [(2, 0), (1, 1)]
    cov = window_coverage(cat, [a, b], "spot", "1d", D(3), D(4))
    assert (cov[1].bars, cov[1].missing, cov[1].first_ts, cov[1].last_ts) == (0, 1, None, None)


def test_coverage_bad_window() -> None:
    cat = Catalog(FIXTURE)
    a, _ = insts(cat)
    with pytest.raises(DataError):
        window_coverage(cat, [a], "spot", "1d", D(3), D(3))


def test_coverage_matches_feed_steps() -> None:
    cat = Catalog(FIXTURE)
    a, b = insts(cat)
    steps = list(ParquetFeed(cat, [a, b], "1d", D(1), D(6)).steps())
    cov = window_coverage(cat, [a, b], "spot", "1d", D(1), D(6))
    assert len(steps) == 5
    assert all(c.bars + c.missing == len(steps) for c in cov)


def test_pickle_roundtrip() -> None:
    cat = Catalog(FIXTURE)
    a, b = insts(cat)
    fp = data_fingerprint(cat, [a, b], "spot", "1d")
    cov = window_coverage(cat, [a, b], "spot", "1d", D(1), D(6))
    for x in (fp, fp.files[0], cov[0]):
        assert pickle.loads(pickle.dumps(x)) == x
