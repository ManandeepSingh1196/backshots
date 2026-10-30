import itertools
from datetime import UTC, timedelta
from pathlib import Path

import polars as pl

from bt.data.catalog import Catalog
from tests.support.make_crosscheck_fixture import CROSSCHECK_DIR, build_crosscheck_fixture

SYMBOLS = ("BTCUSDT", "ETHUSDT")


def _klines(root: Path, sym: str) -> pl.DataFrame:
    return Catalog(root).read_klines("spot", "1d", sym)


def test_regenerated_equals_committed(tmp_path: Path) -> None:
    build_crosscheck_fixture(tmp_path)
    for sym in SYMBOLS:
        assert _klines(tmp_path, sym).equals(_klines(CROSSCHECK_DIR, sym))
    assert (tmp_path / "instruments.json").read_bytes() == (
        CROSSCHECK_DIR / "instruments.json"
    ).read_bytes()


def test_bars_contiguous_and_consistent() -> None:
    for sym in SYMBOLS:
        df = _klines(CROSSCHECK_DIR, sym)
        assert df.height == 400
        ts = df["ts"].to_list()
        assert all(t.tzinfo is not None and t.utcoffset() == timedelta(0) for t in ts)
        assert ts[0] == ts[0].replace(tzinfo=UTC)
        assert all((b - a).total_seconds() == 86400 for a, b in itertools.pairwise(ts))
        top = df.select(pl.max_horizontal("open", "close")).to_series()
        bot = df.select(pl.min_horizontal("open", "close")).to_series()
        assert (df["high"] >= top).all()
        assert (df["low"] <= bot).all()


def test_open_gap_within_buffer() -> None:
    for sym in SYMBOLS:
        df = _klines(CROSSCHECK_DIR, sym)
        gap = (df["open"] / df["close"].shift(1) - 1.0).drop_nulls().abs().max()
        assert isinstance(gap, float)
        assert gap <= 0.002


def test_instruments_zero_fee() -> None:
    insts, _ = Catalog(CROSSCHECK_DIR).read_instruments()
    assert {i.symbol for i in insts} == set(SYMBOLS)
    for i in insts:
        assert i.taker_fee == 0.0
        assert i.maker_fee == 0.0


def test_ma_cross_state_changes() -> None:
    for sym in SYMBOLS:
        close = _klines(CROSSCHECK_DIR, sym)["close"]
        fast = close.rolling_mean(5)
        slow = close.rolling_mean(20)
        state = (fast > slow).to_list()[19:]
        changes = sum(1 for a, b in itertools.pairwise(state) if a != b)
        assert changes >= 6, (sym, changes)
