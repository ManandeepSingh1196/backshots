"""Ingest pipeline: cached, checksummed archives plus a ccxt tail, validated before storing.

Nothing here reads the system clock: `now` is always an argument. Data is never repaired:
a dataset that fails validation is reported and left unstored.
"""

from __future__ import annotations

import calendar
import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import httpx
import polars as pl

from bt.core.clock import ensure_utc
from bt.core.types import Instrument
from bt.data.catalog import Catalog, ManifestEntry
from bt.data.fetch_binance import (
    ArchiveData,
    ArchiveFile,
    daily_kline_archive_files,
    download_archive,
    funding_archive_files,
    kline_archive_files,
    parse_funding_csv,
    parse_klines_csv,
)
from bt.data.fetch_ccxt import (
    CcxtExchange,
    fetch_instruments,
    fetch_recent_funding,
    fetch_recent_klines,
)
from bt.data.validate import (
    FUNDING_SCHEMA,
    KLINE_SCHEMA,
    TIMEFRAME_DELTA,
    Market,
    Timeframe,
    ValidationReport,
    check_symbol,
    validate_funding,
    validate_klines,
)

DEFAULT_SYMBOLS: Final[tuple[str, ...]] = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "LINKUSDT",
)
DEFAULT_START: Final[date] = date(2019, 1, 1)
PUBLICATION_LAG_DAYS: Final[int] = 2

_KLINES_TAG: Final = "ccxt:binance:klines"
_FUNDING_TAG: Final = "ccxt:binance:fundingRate"
_MS: Final = timedelta(milliseconds=1)
_DAY: Final = timedelta(days=1)


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Outcome of ingesting one dataset.

    Invariants: `entry` is None iff nothing was stored (then `report` is not ok); `missing_archives` holds only 404s that were not publication lag.
    """

    dataset: str
    entry: ManifestEntry | None
    report: ValidationReport
    missing_archives: tuple[str, ...]


def _midnight(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def _last_day(file: ArchiveFile) -> date:
    """Last calendar day an archive covers (period is "YYYY-MM" or "YYYY-MM-DD")."""
    if len(file.period) == 7:
        year, month = int(file.period[:4]), int(file.period[5:])
        return date(year, month, calendar.monthrange(year, month)[1])
    return date.fromisoformat(file.period)


@dataclass(frozen=True, slots=True)
class _Archives:
    used: tuple[ArchiveData, ...]
    missing: tuple[str, ...]


def _collect(
    client: httpx.Client,
    cache_dir: Path,
    files: Sequence[ArchiveFile],
    today: date,
    fallback: Callable[[ArchiveFile], Sequence[ArchiveFile]] | None,
) -> _Archives:
    """Download `files` in order.

    404s before the first success are skipped (not listed yet). After it, a 404 whose last
    covered day is within the publication lag ends the run (later files are not used and
    the tail fills the hole); any other 404 is reported missing. A monthly 404 tries
    `fallback` first, if given; the monthly URL itself is then not reported.
    """
    lag_cut = today - timedelta(days=PUBLICATION_LAG_DAYS)
    used: list[ArchiveData] = []
    missing: list[str] = []

    def fetch(batch: Sequence[ArchiveFile], allow_fallback: bool) -> bool:
        """Process `batch`; return True when a lag-classified 404 ends the run."""
        for f in batch:
            data = download_archive(client, f, cache_dir)
            if data is not None:
                used.append(data)
            elif used and allow_fallback and fallback is not None and len(f.period) == 7:
                if fetch(fallback(f), False):
                    return True
            elif _last_day(f) >= lag_cut:
                return True
            elif not used:
                continue
            else:
                missing.append(f.url)
        return False

    fetch(files, True)
    return _Archives(tuple(used), tuple(missing))


def _bounded(df: pl.DataFrame, start: date, end_d: date) -> pl.DataFrame:
    lo, hi = _midnight(start), _midnight(end_d + _DAY)
    return df.filter((pl.col("ts") >= lo) & (pl.col("ts") < hi))


def _last_ts(frames: Sequence[pl.DataFrame]) -> datetime | None:
    stamps = [t for t in (f["ts"].max() for f in frames if f.height) if isinstance(t, datetime)]
    return max(stamps) if stamps else None


def _result(
    dataset: str,
    report: ValidationReport,
    missing: tuple[str, ...],
    store: Callable[[], ManifestEntry],
) -> IngestResult:
    entry = store() if report.ok else None
    return IngestResult(dataset, entry, report, missing)


def ingest_klines(
    catalog: Catalog,
    client: httpx.Client,
    exchange: CcxtExchange,
    market: Market,
    timeframe: Timeframe,
    symbol: str,
    start: date,
    end: date | None,
    now: datetime,
) -> IngestResult:
    """Rebuild one klines dataset from archives plus a ccxt tail; store it only if valid."""
    now = ensure_utc(now, "now")
    today = now.date()
    end_d = today if end is None else end
    delta = TIMEFRAME_DELTA[timeframe]
    dataset = f"{market}/klines/{timeframe}/{symbol}"

    first_day, last_day = start, min(end_d, today - _DAY)

    def fallback(monthly: ArchiveFile) -> list[ArchiveFile]:
        month = date.fromisoformat(monthly.period + "-01")
        days = daily_kline_archive_files(market, timeframe, symbol, month)
        return [f for f in days if first_day <= date.fromisoformat(f.period) <= last_day]

    files = kline_archive_files(market, timeframe, symbol, start, end_d, today)
    archives = _collect(client, catalog.cache_dir(), files, today, fallback)
    frames = [parse_klines_csv(a.csv) for a in archives.used]

    tail_now = min(now, _midnight(end_d + _DAY))
    last_archive = _last_ts(frames)
    since = _midnight(start) if last_archive is None else last_archive + delta
    tail = pl.DataFrame(schema=dict(KLINE_SCHEMA))
    if since < tail_now:
        tail = fetch_recent_klines(exchange, market, timeframe, symbol, since, tail_now)
        if last_archive is not None:
            tail = tail.filter(pl.col("ts") > last_archive)

    def finish(df: pl.DataFrame) -> pl.DataFrame:
        closed = _bounded(df, start, end_d).filter(pl.col("ts") + delta <= now)
        return closed.unique(maintain_order=True)

    tail_rows = finish(tail)
    df = pl.concat([finish(pl.concat(frames)) if frames else tail_rows.clear(), tail_rows])
    sources = [a.file.url for a in archives.used]
    if tail_rows.height:
        sources.append(_KLINES_TAG)
    checksums = {a.file.url: a.sha256 for a in archives.used}
    report = validate_klines(df, timeframe, dataset)
    return _result(
        dataset,
        report,
        archives.missing,
        lambda: catalog.write_klines(market, timeframe, symbol, df, sources, checksums, now),
    )


def ingest_funding(
    catalog: Catalog,
    client: httpx.Client,
    exchange: CcxtExchange,
    symbol: str,
    start: date,
    end: date | None,
    now: datetime,
) -> IngestResult:
    """Rebuild one funding dataset from monthly archives plus a ccxt tail; store it if valid."""
    now = ensure_utc(now, "now")
    today = now.date()
    end_d = today if end is None else end
    dataset = f"perp/funding/{symbol}"

    files = funding_archive_files(symbol, start, end_d, today)
    archives = _collect(client, catalog.cache_dir(), files, today, None)
    frames = [parse_funding_csv(a.csv) for a in archives.used]

    tail_now = min(now, _midnight(end_d + _DAY))
    last_archive = _last_ts(frames)
    since = _midnight(start) if last_archive is None else last_archive + _MS
    tail = pl.DataFrame(schema=dict(FUNDING_SCHEMA))
    if since < tail_now:
        tail = fetch_recent_funding(exchange, symbol, since, tail_now)
        if last_archive is not None:
            tail = tail.filter(pl.col("ts") > last_archive)

    def finish(df: pl.DataFrame) -> pl.DataFrame:
        return _bounded(df, start, end_d).filter(pl.col("ts") <= now).unique(maintain_order=True)

    tail_rows = finish(tail)
    df = pl.concat([finish(pl.concat(frames)) if frames else tail_rows.clear(), tail_rows])
    sources = [a.file.url for a in archives.used]
    if tail_rows.height:
        sources.append(_FUNDING_TAG)
    checksums = {a.file.url: a.sha256 for a in archives.used}
    report = validate_funding(df, dataset)
    return _result(
        dataset,
        report,
        archives.missing,
        lambda: catalog.write_funding(symbol, df, sources, checksums, now),
    )


def ingest_instruments(
    catalog: Catalog,
    exchange: CcxtExchange,
    symbols: Sequence[str],
    markets: Sequence[Market],
    now: datetime,
) -> list[Instrument]:
    """Fetch instrument rules for `symbols` on each market and store one snapshot at `now`."""
    now = ensure_utc(now, "now")
    symbols = list(dict.fromkeys(symbols))
    for symbol in symbols:
        check_symbol(symbol)
    found: list[Instrument] = []
    for market in markets:
        found.extend(fetch_instruments(exchange, market, symbols))
    found.sort(key=lambda i: i.symbol)
    catalog.write_instruments(found, fetched_at=now)
    return found


def _failed(dataset: str, error: str) -> ValidationReport:
    return ValidationReport(dataset, 0, None, None, (error,), (), ())


def _dataset_name(rel: str) -> str:
    """"binance/perp/klines/1d/BTCUSDT.parquet" -> "perp/klines/1d/BTCUSDT"."""
    return rel.removeprefix("binance/").removesuffix(".parquet")


def _validate_stored(rel: str, df: pl.DataFrame) -> ValidationReport:
    dataset = _dataset_name(rel)
    parts = dataset.split("/")
    if len(parts) == 4 and parts[1] == "klines" and parts[2] in TIMEFRAME_DELTA:
        timeframe: Timeframe = "1d" if parts[2] == "1d" else "4h"
        return validate_klines(df, timeframe, dataset)
    if len(parts) == 3 and parts[1] == "funding":
        return validate_funding(df, dataset)
    return _failed(dataset, "unrecognized dataset path")


def _check_entry(root: Path, entry: ManifestEntry) -> ValidationReport:
    dataset = _dataset_name(entry.path)
    path = root / entry.path
    if not path.is_file():
        return _failed(dataset, "missing file")
    try:
        raw = path.read_bytes()
        df = pl.read_parquet(path)
    except (pl.exceptions.PolarsError, OSError) as e:
        return _failed(dataset, f"cannot read file: {e}")
    report = _validate_stored(entry.path, df)
    checks: list[tuple[str, object, object]] = [
        ("rows", entry.rows, report.rows),
        ("first_ts", entry.first_ts, report.first_ts),
        ("last_ts", entry.last_ts, report.last_ts),
        ("sha256", entry.sha256, hashlib.sha256(raw).hexdigest()),
    ]
    mismatches = tuple(
        f"manifest mismatch: {name} {stored} != {actual}"
        for name, stored, actual in checks
        if stored != actual
    )
    return replace(report, errors=report.errors + mismatches)


def check_catalog(catalog: Catalog) -> list[ValidationReport]:
    """Re-validate every manifest entry against its file; flag Parquet files not in the manifest."""
    manifest = catalog.manifest()
    reports = [_check_entry(catalog.root, e) for e in manifest.values()]
    data_root = catalog.root / "binance"
    if data_root.is_dir():
        for path in data_root.rglob("*.parquet"):
            rel = path.relative_to(catalog.root).as_posix()
            if rel not in manifest:
                reports.append(_failed(_dataset_name(rel), "not in manifest"))
    return sorted(reports, key=lambda r: r.dataset)
