"""On-disk catalog: validated Parquet datasets, a manifest and an instruments snapshot."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Final, cast, get_args

import polars as pl

from bt.core.clock import ensure_utc
from bt.core.errors import DataError, DataValidationError, InvariantError
from bt.core.types import Instrument, InstrumentKind
from bt.data.validate import (
    FUNDING_SCHEMA,
    KLINE_SCHEMA,
    Market,
    Timeframe,
    ValidationReport,
    check_symbol,
    validate_funding,
    validate_klines,
)

_MANIFEST_VERSION: Final = 1
_INSTRUMENTS_VERSION: Final = 1
_FILE_MODE: Final = 0o644


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


class _UnknownVersionError(Exception):
    """Internal: manifest version is not supported."""


def _check_market(market: str) -> None:
    if market not in get_args(Market):
        raise DataError(f"invalid market {market!r}: expected one of {get_args(Market)}")


def _check_timeframe(timeframe: str) -> None:
    if timeframe not in get_args(Timeframe):
        raise DataError(f"invalid timeframe {timeframe!r}: expected one of {get_args(Timeframe)}")


def _atomic_write(path: Path, write: Callable[[Path], None]) -> None:
    """Run `write(tmp)` on a temp file beside `path`, then os.replace it into place.

    On any exception the temp file is removed and `path` is left as it was.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        write(tmp)
        os.chmod(tmp, _FILE_MODE)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write(path, lambda tmp: _write_text(tmp, text))


def _entry_to_json(entry: ManifestEntry) -> dict[str, object]:
    return {
        "path": entry.path,
        "sources": list(entry.sources),
        "checksums": dict(sorted(entry.checksums.items())),
        "rows": entry.rows,
        "first_ts": entry.first_ts.isoformat(),
        "last_ts": entry.last_ts.isoformat(),
        "ingested_at": entry.ingested_at.isoformat(),
        "sha256": entry.sha256,
    }


def _str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {value!r}")
    return value


def _dt(value: object, name: str) -> datetime:
    return ensure_utc(datetime.fromisoformat(_str(value, name)), name)


def _obj(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object, got {type(value).__name__}")
    items = cast(dict[object, object], value)
    return {_str(k, f"{name} key"): v for k, v in items.items()}


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list, got {type(value).__name__}")
    return list(cast(list[object], value))


def _entry_from_json(raw: object) -> ManifestEntry:
    d = _obj(raw, "manifest entry")
    rows = d["rows"]
    if not isinstance(rows, int) or isinstance(rows, bool):
        raise TypeError(f"rows must be an int, got {rows!r}")
    return ManifestEntry(
        path=_str(d["path"], "path"),
        sources=tuple(_str(x, "sources[]") for x in _list(d["sources"], "sources")),
        checksums={k: _str(v, "checksums value") for k, v in _obj(d["checksums"], "checksums").items()},
        rows=rows,
        first_ts=_dt(d["first_ts"], "first_ts"),
        last_ts=_dt(d["last_ts"], "last_ts"),
        ingested_at=_dt(d["ingested_at"], "ingested_at"),
        sha256=_str(d["sha256"], "sha256"),
    )


def _instrument_to_json(inst: Instrument) -> dict[str, object]:
    out: dict[str, object] = {f.name: getattr(inst, f.name) for f in fields(Instrument)}
    out["kind"] = inst.kind.value
    return out


def _num(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number, got {value!r}")
    return float(value)


def _instrument_from_json(raw: object) -> Instrument:
    d = _obj(raw, "instrument")
    return Instrument(
        symbol=_str(d["symbol"], "symbol"),
        venue_symbol=_str(d["venue_symbol"], "venue_symbol"),
        kind=InstrumentKind(_str(d["kind"], "kind")),
        base=_str(d["base"], "base"),
        quote=_str(d["quote"], "quote"),
        min_qty=_num(d["min_qty"], "min_qty"),
        qty_step=_num(d["qty_step"], "qty_step"),
        min_notional=_num(d["min_notional"], "min_notional"),
        price_tick=_num(d["price_tick"], "price_tick"),
        maker_fee=_num(d["maker_fee"], "maker_fee"),
        taker_fee=_num(d["taker_fee"], "taker_fee"),
        max_leverage=_num(d["max_leverage"], "max_leverage"),
        maint_margin_rate=_num(d["maint_margin_rate"], "maint_margin_rate"),
    )


def _schema_matches(df: pl.DataFrame, expected: Mapping[str, pl.DataType]) -> bool:
    return list(df.schema.items()) == list(expected.items())


def _read_parquet(path: Path, expected: Mapping[str, pl.DataType]) -> pl.DataFrame:
    if not path.is_file():
        raise DataError(f"dataset not found: {path}")
    try:
        df = pl.read_parquet(path)
    except (pl.exceptions.PolarsError, OSError) as e:
        raise DataError(f"cannot read {path}: {e}") from e
    if not _schema_matches(df, expected):
        raise DataError(f"schema of {path} does not match the expected schema: {dict(df.schema)}")
    return df


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

    def manifest(self) -> Mapping[str, ManifestEntry]:
        path = self._manifest_path()
        if not path.exists():
            return {}
        try:
            raw: object = json.loads(path.read_text(encoding="utf-8"))
            top = _obj(raw, "manifest")
            version = top["version"]
            files = top["files"]
            if version != _MANIFEST_VERSION or isinstance(version, bool):
                raise _UnknownVersionError(f"unsupported manifest version {version!r}")
            entries = [_entry_from_json(v) for v in _obj(files, "files").values()]
        except _UnknownVersionError as e:
            raise DataError(f"{path}: {e}") from e
        except (KeyError, TypeError, ValueError, OSError) as e:
            raise DataError(f"cannot parse manifest {path}: {e!r}") from e
        return {e.path: e for e in entries}

    def write_klines(
        self,
        market: Market,
        timeframe: Timeframe,
        symbol: str,
        df: pl.DataFrame,
        sources: Sequence[str],
        checksums: Mapping[str, str],
        ingested_at: datetime,
    ) -> ManifestEntry:
        """Validate and store klines atomically, then update the manifest.

        The Parquet footer embeds the polars version (`created_by`), so `sha256` is
        stable within one install but may change across polars versions.
        """
        path = self.klines_path(market, timeframe, symbol)
        dataset = f"{market}/klines/{timeframe}/{symbol}"
        report = validate_klines(df, timeframe, dataset)
        return self._store(path, df, report, sources, checksums, ingested_at)

    def read_klines(self, market: Market, timeframe: Timeframe, symbol: str) -> pl.DataFrame:
        return _read_parquet(self.klines_path(market, timeframe, symbol), KLINE_SCHEMA)

    def write_funding(
        self,
        symbol: str,
        df: pl.DataFrame,
        sources: Sequence[str],
        checksums: Mapping[str, str],
        ingested_at: datetime,
    ) -> ManifestEntry:
        path = self.funding_path(symbol)
        report = validate_funding(df, f"perp/funding/{symbol}")
        return self._store(path, df, report, sources, checksums, ingested_at)

    def read_funding(self, symbol: str) -> pl.DataFrame:
        return _read_parquet(self.funding_path(symbol), FUNDING_SCHEMA)

    def write_instruments(self, instruments: Sequence[Instrument], fetched_at: datetime) -> None:
        fetched = ensure_utc(fetched_at, "fetched_at")
        doc = {
            "version": _INSTRUMENTS_VERSION,
            "fetched_at": fetched.isoformat(),
            "instruments": [
                _instrument_to_json(i) for i in sorted(instruments, key=lambda i: i.symbol)
            ],
        }
        _atomic_write_text(
            self._instruments_path(), json.dumps(doc, indent=2, sort_keys=True) + "\n"
        )

    def read_instruments(self) -> tuple[list[Instrument], datetime]:
        path = self._instruments_path()
        if not path.is_file():
            raise DataError(f"instruments snapshot not found: {path}")
        try:
            top = _obj(json.loads(path.read_text(encoding="utf-8")), "instruments file")
            version = top["version"]
            if isinstance(version, bool) or version != _INSTRUMENTS_VERSION:
                raise _UnknownVersionError(f"unsupported instruments version {version!r}")
            fetched = _dt(top["fetched_at"], "fetched_at")
            items = [_instrument_from_json(x) for x in _list(top["instruments"], "instruments")]
        except _UnknownVersionError as e:
            raise DataError(f"{path}: {e}") from e
        except (KeyError, TypeError, ValueError, OSError, InvariantError) as e:
            raise DataError(f"cannot parse instruments file {path}: {e!r}") from e
        return items, fetched

    def _instruments_path(self) -> Path:
        return self._root / "instruments.json"

    def _manifest_path(self) -> Path:
        return self._root / "manifest.json"

    def _store(
        self,
        path: Path,
        df: pl.DataFrame,
        report: ValidationReport,
        sources: Sequence[str],
        checksums: Mapping[str, str],
        ingested_at: datetime,
    ) -> ManifestEntry:
        """Validate, then replace the data file, then replace the manifest.

        The two replaces are separate steps: if the manifest write fails (e.g. disk
        full) after the data replace, file and manifest disagree until the next
        successful write; the manifest's sha256 lets a later check detect that.
        """
        ingested = ensure_utc(ingested_at, "ingested_at")
        if not report.ok:
            raise DataValidationError(f"{report.dataset} failed validation: " + "; ".join(report.errors))
        assert report.first_ts is not None and report.last_ts is not None
        # Load the manifest first: a corrupt one must fail before the data file changes.
        existing = self.manifest()
        digest: list[str] = []

        def write(tmp: Path) -> None:
            df.write_parquet(tmp, compression="zstd", statistics=True)
            digest.append(hashlib.sha256(tmp.read_bytes()).hexdigest())

        _atomic_write(path, write)
        entry = ManifestEntry(
            path=path.relative_to(self._root).as_posix(),
            sources=tuple(sources),
            checksums=dict(sorted(checksums.items())),
            rows=report.rows,
            first_ts=report.first_ts,
            last_ts=report.last_ts,
            ingested_at=ingested,
            sha256=digest[-1],
        )
        files = {p: _entry_to_json(e) for p, e in existing.items()}
        files[entry.path] = _entry_to_json(entry)
        doc = {"version": _MANIFEST_VERSION, "files": files}
        _atomic_write_text(
            self._manifest_path(), json.dumps(doc, indent=2, sort_keys=True) + "\n"
        )
        return entry
