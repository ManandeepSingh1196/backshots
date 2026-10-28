"""Binance bulk archive (data.binance.vision): file listing, checksummed download, CSV parsing.

Functions are pure or take an injected `httpx.Client`; nothing here builds its own client.
"""

from __future__ import annotations

import calendar
import hashlib
import io
import os
import re
import zipfile
from zlib import error as zlib_error
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

import httpx
from tenacity import RetryError, Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from bt.core.errors import ArchiveError, ChecksumMismatchError
from bt.data.validate import Market, Timeframe, check_symbol

ARCHIVE_BASE: Final = "https://data.binance.vision"
MICROS_THRESHOLD: Final = 10**14  # raw ts >= this is microseconds, else milliseconds

_MAX_ATTEMPTS: Final = 3
_BACKOFF_SECONDS: float = 1.0  # exponential-backoff multiplier; tests patch this to 0
_CHECKSUM_LINE: Final = re.compile(r"([0-9a-fA-F]{64})\s+\*?(\S.*)")


@dataclass(frozen=True, slots=True)
class ArchiveFile:
    """One archive zip and its checksum file; `checksum_url == url + ".CHECKSUM"`."""

    url: str
    checksum_url: str
    period: str  # "2024-01" (monthly) or "2024-03-05" (daily)


def _file(url: str, period: str) -> ArchiveFile:
    return ArchiveFile(url=url, checksum_url=url + ".CHECKSUM", period=period)


def _root(market: Market) -> str:
    return "data/spot" if market == "spot" else "data/futures/um"


def _months(start: date, end: date) -> list[date]:
    """First days of every calendar month overlapping [start, end]."""
    out: list[date] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append(date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _kline_url(market: Market, symbol: str, tf: Timeframe, kind: str, stamp: str) -> str:
    return f"{ARCHIVE_BASE}/{_root(market)}/{kind}/klines/{symbol}/{tf}/{symbol}-{tf}-{stamp}.zip"


def _check_range(start: date, end: date, today: date) -> date:
    """Validate the range and return `end` clamped to `today`."""
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    return min(end, today)


def _complete_months(start: date, end: date, today: date) -> list[date]:
    """Months overlapping [start, end] that are strictly before today's month."""
    this_month = today.replace(day=1)
    if start > end:
        return []
    return [m for m in _months(start, end) if m < this_month]


def kline_archive_files(
    market: Market, timeframe: Timeframe, symbol: str, start: date, end: date, today: date
) -> list[ArchiveFile]:
    """Monthly files for complete months, then daily files for the current month so far."""
    check_symbol(symbol)
    end = _check_range(start, end, today)
    files = [
        _file(_kline_url(market, symbol, timeframe, "monthly", f"{m:%Y-%m}"), f"{m:%Y-%m}")
        for m in _complete_months(start, end, today)
    ]
    first = today.replace(day=1)
    if end >= first:
        day = max(start, first)
        last = min(end, today - timedelta(days=1))
        while day <= last:
            files.append(
                _file(_kline_url(market, symbol, timeframe, "daily", f"{day:%Y-%m-%d}"), f"{day:%Y-%m-%d}")
            )
            day += timedelta(days=1)
    return files


def daily_kline_archive_files(
    market: Market, timeframe: Timeframe, symbol: str, month: date
) -> list[ArchiveFile]:
    """One daily file for every day of `month`'s calendar month (`month.day` is ignored)."""
    check_symbol(symbol)
    n = calendar.monthrange(month.year, month.month)[1]
    days = [date(month.year, month.month, d) for d in range(1, n + 1)]
    return [
        _file(_kline_url(market, symbol, timeframe, "daily", f"{d:%Y-%m-%d}"), f"{d:%Y-%m-%d}")
        for d in days
    ]


def funding_archive_files(symbol: str, start: date, end: date, today: date) -> list[ArchiveFile]:
    """Monthly funding files for complete months only (Binance has no daily funding archive)."""
    check_symbol(symbol)
    end = _check_range(start, end, today)
    return [
        _file(
            f"{ARCHIVE_BASE}/data/futures/um/monthly/fundingRate/{symbol}/"
            f"{symbol}-fundingRate-{m:%Y-%m}.zip",
            f"{m:%Y-%m}",
        )
        for m in _complete_months(start, end, today)
    ]


@dataclass(frozen=True, slots=True)
class ArchiveData:
    """A verified archive: `sha256` is the digest of the zip, `csv` its single member."""

    file: ArchiveFile
    sha256: str
    csv: bytes


class _Retryable(Exception):
    """Internal marker for a failure worth retrying (transport error, 5xx, 429)."""


def _get(client: httpx.Client, url: str) -> httpx.Response:
    """GET once. Returns 200 or 404; raises _Retryable or ArchiveError otherwise."""
    try:
        resp = client.get(url)
    except httpx.TransportError as exc:
        raise _Retryable(f"{url}: {exc!r}") from exc
    code = resp.status_code
    if code in (200, 404):
        return resp
    if code >= 500 or code == 429:
        raise _Retryable(f"{url}: HTTP {code}")
    raise ArchiveError(f"{url}: HTTP {code}")


def _get_with_retries(client: httpx.Client, url: str) -> httpx.Response:
    retrying = Retrying(
        stop=stop_after_attempt(_MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=_BACKOFF_SECONDS),
        retry=retry_if_exception_type(_Retryable),
        reraise=False,
    )
    try:
        return retrying(_get, client, url)
    except RetryError as err:
        cause = err.last_attempt.exception()
        raise ArchiveError(f"{url}: failed after {_MAX_ATTEMPTS} attempts: {cause}") from cause


def _parse_checksum(text: bytes, url: str) -> str:
    try:
        decoded = text.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ArchiveError(f"{url}: checksum file is not UTF-8") from exc
    match = _CHECKSUM_LINE.fullmatch(decoded)
    if match is None:
        raise ArchiveError(f"{url}: malformed checksum file {decoded[:80]!r}")
    return match.group(1).lower()


def _extract_csv(zip_bytes: bytes, url: str) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = [i.filename for i in zf.infolist() if i.filename.lower().endswith(".csv")]
            if len(names) != 1:
                raise ArchiveError(f"{url}: expected exactly one CSV member, found {len(names)}")
            return zf.read(names[0])
    except (zipfile.BadZipFile, zlib_error, EOFError, NotImplementedError) as exc:
        raise ArchiveError(f"{url}: corrupt zip: {exc}") from exc


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _cache_path(cache_dir: Path, url: str) -> Path:
    path = (cache_dir / urlparse(url).path.lstrip("/")).resolve()
    if not path.is_relative_to(cache_dir.resolve()):
        raise ArchiveError(f"{url}: path escapes the cache directory")
    return path


def _from_cache(file: ArchiveFile, zip_path: Path, sum_path: Path) -> ArchiveData | None:
    """Return the cached archive if both files exist and agree; otherwise clear them."""
    if not (zip_path.exists() and sum_path.exists()):
        zip_path.unlink(missing_ok=True)
        sum_path.unlink(missing_ok=True)
        return None
    zip_bytes = zip_path.read_bytes()
    actual = hashlib.sha256(zip_bytes).hexdigest()
    try:
        expected = _parse_checksum(sum_path.read_bytes(), file.checksum_url)
    except ArchiveError:
        expected = ""
    if actual != expected:
        zip_path.unlink()
        sum_path.unlink()
        return None
    return ArchiveData(file=file, sha256=actual, csv=_extract_csv(zip_bytes, file.url))


def download_archive(client: httpx.Client, file: ArchiveFile, cache_dir: Path) -> ArchiveData | None:
    """Download `file` with SHA-256 verification and an on-disk cache.

    Returns None if Binance does not publish it (checksum 404). A verified cache hit makes
    no HTTP request; a cache entry that fails verification is deleted and re-downloaded.
    """
    zip_path = _cache_path(cache_dir, file.url)
    sum_path = _cache_path(cache_dir, file.checksum_url)
    cached = _from_cache(file, zip_path, sum_path)
    if cached is not None:
        return cached
    sum_resp = _get_with_retries(client, file.checksum_url)
    if sum_resp.status_code == 404:
        return None
    expected = _parse_checksum(sum_resp.content, file.checksum_url)
    zip_resp = _get_with_retries(client, file.url)
    if zip_resp.status_code == 404:
        raise ArchiveError(f"{file.url}: checksum published but zip is missing (404)")
    zip_bytes = zip_resp.content
    actual = hashlib.sha256(zip_bytes).hexdigest()
    if actual != expected:
        raise ChecksumMismatchError(f"{file.url}: expected sha256 {expected}, got {actual}")
    csv = _extract_csv(zip_bytes, file.url)
    _atomic_write(zip_path, zip_bytes)
    _atomic_write(sum_path, sum_resp.content)
    return ArchiveData(file=file, sha256=actual, csv=csv)
