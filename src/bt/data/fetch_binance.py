"""Binance bulk archive (data.binance.vision): file listing, checksummed download, CSV parsing.

Functions are pure or take an injected `httpx.Client`; nothing here builds its own client.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Final

from bt.data.validate import Market, Timeframe, check_symbol

ARCHIVE_BASE: Final = "https://data.binance.vision"
MICROS_THRESHOLD: Final = 10**14  # raw ts >= this is microseconds, else milliseconds


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
