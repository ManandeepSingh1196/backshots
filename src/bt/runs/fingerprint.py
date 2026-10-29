"""Data fingerprint and window coverage: what data a run used and how complete it was."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, fields
from datetime import datetime

from pydantic import JsonValue

from bt.core.clock import ensure_utc
from bt.core.errors import DataError, InvariantError
from bt.core.types import Instrument
from bt.data.catalog import Catalog
from bt.data.validate import Market, Timeframe


def canonical_json(value: JsonValue) -> str:
    """Deterministic compact JSON; NaN and infinities raise InvariantError."""
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        )
    except ValueError as exc:
        raise InvariantError(f"value is not canonical-JSON serialisable: {exc}") from exc


@dataclass(frozen=True, slots=True)
class FileFingerprint:
    """One verified catalog file. Invariants: `sha256` was re-computed from the file's bytes."""

    path: str
    sha256: str
    rows: int
    first_ts: datetime
    last_ts: datetime


@dataclass(frozen=True, slots=True)
class DataFingerprint:
    """The data a run used. Invariants: files sorted by path, instruments by symbol;
    `sha256` covers content and instrument rules only, never ingest provenance."""

    files: tuple[FileFingerprint, ...]
    instruments: tuple[Instrument, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class SymbolCoverage:
    """Bars of one symbol inside a window. Invariant: missing = window steps - bars."""

    symbol: str
    bars: int
    missing: int
    first_ts: datetime | None
    last_ts: datetime | None


def _check_instruments(instruments: Sequence[Instrument]) -> tuple[Instrument, ...]:
    if not instruments:
        raise DataError("instruments must be non-empty")
    symbols = [i.symbol for i in instruments]
    if len(set(symbols)) != len(symbols):
        raise DataError(f"duplicate instrument symbols: {sorted(symbols)}")
    return tuple(sorted(instruments, key=lambda i: i.symbol))


def _instrument_payload(inst: Instrument) -> JsonValue:
    out: dict[str, JsonValue] = {}
    for f in fields(Instrument):
        v = getattr(inst, f.name)
        out[f.name] = inst.kind.value if f.name == "kind" else v
    return out


def _file_payload(f: FileFingerprint) -> JsonValue:
    return {
        "path": f.path,
        "sha256": f.sha256,
        "rows": f.rows,
        "first_ts": f.first_ts.isoformat(),
        "last_ts": f.last_ts.isoformat(),
    }


def data_fingerprint(
    catalog: Catalog,
    instruments: Sequence[Instrument],
    market: Market,
    timeframe: Timeframe,
) -> DataFingerprint:
    """Hash the klines file of every instrument against the manifest and fingerprint the set."""
    insts = _check_instruments(instruments)
    manifest = catalog.manifest()
    files: list[FileFingerprint] = []
    for inst in insts:
        path = catalog.klines_path(market, timeframe, inst.venue_symbol)
        key = path.relative_to(catalog.root).as_posix()
        entry = manifest.get(key)
        if entry is None:
            raise DataError(f"{key} not in manifest")
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise DataError(f"cannot read {key}: {exc}") from exc
        if digest != entry.sha256:
            raise DataError(f"{key} changed since ingest")
        files.append(FileFingerprint(key, digest, entry.rows, entry.first_ts, entry.last_ts))
        # M6: add funding_path(...) for perps
    files.sort(key=lambda f: f.path)
    payload: JsonValue = {
        "files": [_file_payload(f) for f in files],
        "instruments": [_instrument_payload(i) for i in insts],
    }
    digest = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    return DataFingerprint(tuple(files), insts, digest)


def window_coverage(
    catalog: Catalog,
    instruments: Sequence[Instrument],
    market: Market,
    timeframe: Timeframe,
    start: datetime,
    end: datetime,
) -> tuple[SymbolCoverage, ...]:
    """Per-symbol bar counts inside [start, end) against the union of all symbols' ts."""
    start = ensure_utc(start, "start")
    end = ensure_utc(end, "end")
    if start >= end:
        raise DataError(f"start {start.isoformat()} must be before end {end.isoformat()}")
    insts = _check_instruments(instruments)
    kept: dict[str, list[datetime]] = {}
    for inst in insts:
        df = catalog.read_klines(market, timeframe, inst.venue_symbol)
        kept[inst.symbol] = sorted(
            t for t in (ensure_utc(x, "kline ts") for x in df["ts"].to_list()) if start <= t < end
        )
    steps = {t for ts in kept.values() for t in ts}
    return tuple(
        SymbolCoverage(
            symbol=sym,
            bars=len(ts),
            missing=len(steps) - len(ts),
            first_ts=ts[0] if ts else None,
            last_ts=ts[-1] if ts else None,
        )
        for sym, ts in sorted(kept.items())
    )
