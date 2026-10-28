"""On-disk catalog: validated Parquet datasets, a manifest and an instruments snapshot."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import get_args

from bt.core.errors import DataError
from bt.data.validate import Market, Timeframe, check_symbol


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """Manifest record for one stored Parquet file.

    Invariants: `path` is posix and relative to the data root; datetimes are UTC;
    `sha256` is the hex digest of the Parquet bytes exactly as stored.
    """

    path: str
    sources: tuple[str, ...]
    checksums: Mapping[str, str]
    rows: int
    first_ts: datetime
    last_ts: datetime
    ingested_at: datetime
    sha256: str


def _check_market(market: str) -> None:
    if market not in get_args(Market):
        raise DataError(f"invalid market {market!r}: expected one of {get_args(Market)}")


def _check_timeframe(timeframe: str) -> None:
    if timeframe not in get_args(Timeframe):
        raise DataError(f"invalid timeframe {timeframe!r}: expected one of {get_args(Timeframe)}")


class Catalog:
    """Owns one data root and the INTERFACES section 9 layout beneath it.

    Invariants: every data path is built from a validated market, timeframe and
    venue symbol, so no path escapes the root; nothing here reads the system clock.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def klines_path(self, market: Market, timeframe: Timeframe, symbol: str) -> Path:
        _check_market(market)
        _check_timeframe(timeframe)
        check_symbol(symbol)
        return self._root / "binance" / market / "klines" / timeframe / f"{symbol}.parquet"

    def funding_path(self, symbol: str) -> Path:
        check_symbol(symbol)
        return self._root / "binance" / "perp" / "funding" / f"{symbol}.parquet"

    def cache_dir(self) -> Path:
        return self._root / "cache"
