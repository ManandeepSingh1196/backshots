"""Tests for bt.data.catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from bt.core.errors import DataError
from bt.data.catalog import Catalog


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
