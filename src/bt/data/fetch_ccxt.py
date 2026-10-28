"""Thin, fake-able wrappers over Binance's public endpoints via ccxt.

Builds `Instrument`s from market metadata and fetches closed klines and funding events
after the archive's last row. Public endpoints only: no credentials, no environment reads.
Every datetime is checked with `ensure_utc`; unclosed bars are never returned.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Final, Protocol, TypeVar, cast

import ccxt  # pyright: ignore[reportMissingTypeStubs]
import polars as pl

from bt.core.clock import ensure_utc
from bt.core.errors import DataError, InvariantError
from bt.core.types import Instrument, InstrumentKind
from bt.data.validate import (
    FUNDING_SCHEMA,
    KLINE_SCHEMA,
    TIMEFRAME_DELTA,
    Market,
    Timeframe,
    check_symbol,
)

PAGE_LIMIT: Final = 1000
MAX_ATTEMPTS: Final = 3
_RETRY_WAIT_S: float = 1.0  # tests patch this to 0

_T = TypeVar("_T")


class CcxtExchange(Protocol):
    """The subset of ccxt.binance used here; tests use a fake."""

    def load_markets(self) -> Mapping[str, Mapping[str, object]]: ...
    def publicGetKlines(self, params: Mapping[str, object]) -> list[list[object]]: ...  # noqa: N802
    def fapiPublicGetKlines(self, params: Mapping[str, object]) -> list[list[object]]: ...  # noqa: N802
    def fapiPublicGetFundingRate(  # noqa: N802
        self, params: Mapping[str, object]
    ) -> list[Mapping[str, object]]: ...


def make_exchange() -> CcxtExchange:
    """ccxt.binance for public endpoints only (spot + linear markets); no credentials, no I/O."""
    exchange = ccxt.binance(
        {
            "enableRateLimit": True,
            "options": {"fetchMarkets": {"types": ["spot", "linear"]}},
        }
    )
    return cast(CcxtExchange, exchange)  # ccxt ships no type information


# --------------------------------------------------------------------------- instruments


def _sub(obj: object, *path: str) -> object:
    cur = obj
    for key in path:
        if not isinstance(cur, Mapping):
            return None
        cur = cast(Mapping[str, object], cur).get(key)
    return cur


def _number(market: Mapping[str, object], venue_symbol: str, *path: str) -> float:
    value = _sub(market, *path)
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise DataError(f"{venue_symbol}: market field {'.'.join(path)} is missing or None")
    try:
        return float(value)
    except ValueError as exc:
        raise DataError(f"{venue_symbol}: market field {'.'.join(path)} is not numeric") from exc


def _matches(market: Mapping[str, object], kind: Market) -> bool:
    if kind == "spot":
        return market.get("spot") is True
    return market.get("swap") is True and market.get("linear") is True


def _build_instrument(market: Mapping[str, object], kind: Market) -> Instrument:
    venue_symbol = str(market["id"])
    if kind == "spot":
        symbol, ikind, leverage, mmr = venue_symbol, InstrumentKind.SPOT, 1.0, 0.0
    else:
        symbol, ikind, leverage = f"{venue_symbol}-PERP", InstrumentKind.PERP, 1.0
        mmr = _number(market, venue_symbol, "info", "maintMarginPercent") / 100
    base, quote = market.get("base"), market.get("quote")
    if not isinstance(base, str) or not isinstance(quote, str):
        raise DataError(f"{venue_symbol}: market field base/quote is missing or None")
    try:
        return Instrument(
            symbol=symbol,
            venue_symbol=venue_symbol,
            kind=ikind,
            base=base,
            quote=quote,
            min_qty=_number(market, venue_symbol, "limits", "amount", "min"),
            qty_step=_number(market, venue_symbol, "precision", "amount"),
            min_notional=_number(market, venue_symbol, "limits", "cost", "min"),
            price_tick=_number(market, venue_symbol, "precision", "price"),
            maker_fee=_number(market, venue_symbol, "maker"),
            taker_fee=_number(market, venue_symbol, "taker"),
            max_leverage=leverage,
            maint_margin_rate=mmr,
        )
    except InvariantError as exc:
        raise DataError(f"{venue_symbol}: invalid instrument data: {exc}") from exc


def fetch_instruments(
    exchange: CcxtExchange, market: Market, venue_symbols: Sequence[str]
) -> list[Instrument]:
    """Instruments for `venue_symbols` in input order; one DataError lists all missing symbols.

    Perp `max_leverage` is a 1.0 placeholder: public market data carries no leverage limits.
    """
    markets = _call(exchange.load_markets)
    by_id: dict[str, Mapping[str, object]] = {}
    for m in markets.values():
        mid = m.get("id")
        if isinstance(mid, str) and _matches(m, market) and mid not in by_id:
            by_id[mid] = m
    missing = [s for s in venue_symbols if s not in by_id]
    if missing:
        raise DataError(f"{market} markets not found on exchange: {', '.join(missing)}")
    return [_build_instrument(by_id[s], market) for s in venue_symbols]


# --------------------------------------------------------------------------- retry / paging


def _call(fn: Callable[[], _T]) -> _T:
    """Run `fn`, retrying ccxt.NetworkError up to MAX_ATTEMPTS; ccxt errors become DataError."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn()
        except ccxt.NetworkError as exc:
            if attempt == MAX_ATTEMPTS:
                raise DataError(f"network error after {MAX_ATTEMPTS} attempts: {exc}") from exc
            time.sleep(_RETRY_WAIT_S)
        except ccxt.BaseError as exc:
            raise DataError(f"exchange error: {exc}") from exc
    raise AssertionError("unreachable")  # pragma: no cover


def _ms(ts: datetime) -> int:
    delta = ts - datetime(1970, 1, 1, tzinfo=UTC)
    return (delta.days * 86_400 + delta.seconds) * 1000 + delta.microseconds // 1000


def _to_int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise DataError(f"malformed {what}: {value!r}")
    try:
        return int(value)
    except ValueError as exc:
        raise DataError(f"malformed {what}: {value!r}") from exc


def _to_float(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise DataError(f"malformed {what}: {value!r}")
    try:
        return float(value)
    except ValueError as exc:
        raise DataError(f"malformed {what}: {value!r}") from exc


def _paginate(
    fetch: Callable[[int], list[_T]], open_ms: Callable[[_T], int], since_ms: int, now_ms: int
) -> list[_T]:
    """Collect pages from `since_ms` until a short page or a row at/after `now_ms`."""
    rows: list[_T] = []
    start = since_ms
    while True:
        page = _call(lambda s=start: fetch(s))
        if not page:
            break
        rows.extend(page)
        last = open_ms(page[-1])
        if len(page) < PAGE_LIMIT or last >= now_ms:
            break
        if last < start:
            raise DataError("pagination did not advance")
        start = last + 1
    return rows


def _ts_frame(values: list[int]) -> pl.Series:
    return (
        pl.Series("ts", values, dtype=pl.Int64)
        .cast(pl.Datetime("ms"))
        .dt.replace_time_zone("UTC")
        .dt.cast_time_unit("us")
    )


# --------------------------------------------------------------------------- klines


def fetch_recent_klines(
    exchange: CcxtExchange,
    market: Market,
    timeframe: Timeframe,
    venue_symbol: str,
    since: datetime,
    now: datetime,
) -> pl.DataFrame:
    """Closed bars with `since <= ts` and `ts + timeframe <= now`, unique, sorted, KLINE_SCHEMA."""
    since = ensure_utc(since, "since")
    now = ensure_utc(now, "now")
    check_symbol(venue_symbol)
    if since > now:
        return pl.DataFrame(schema=dict(KLINE_SCHEMA))
    delta = TIMEFRAME_DELTA[timeframe]
    endpoint = exchange.publicGetKlines if market == "spot" else exchange.fapiPublicGetKlines

    def fetch(start: int) -> list[list[object]]:
        params: dict[str, object] = {
            "symbol": venue_symbol,
            "interval": timeframe,
            "startTime": start,
            "limit": PAGE_LIMIT,
        }
        return endpoint(params)

    def open_ms(row: list[object]) -> int:
        if len(row) < 9:
            raise DataError(f"{venue_symbol}: malformed kline row {row!r}")
        return _to_int(row[0], "kline open time")

    raw = _paginate(fetch, open_ms, _ms(since), _ms(now))
    since_ms, last_open_ms = _ms(since), _ms(now - delta)
    seen: dict[int, list[object]] = {}
    for row in raw:
        t = open_ms(row)
        if since_ms <= t <= last_open_ms and t not in seen:
            seen[t] = row
    keys = sorted(seen)
    rows = [seen[k] for k in keys]
    df = pl.DataFrame(
        {
            "ts": _ts_frame(keys),
            "open": [_to_float(r[1], "open") for r in rows],
            "high": [_to_float(r[2], "high") for r in rows],
            "low": [_to_float(r[3], "low") for r in rows],
            "close": [_to_float(r[4], "close") for r in rows],
            "volume": [_to_float(r[5], "volume") for r in rows],
            "quote_volume": [_to_float(r[7], "quote_volume") for r in rows],
            "trades": [_to_int(r[8], "trades") for r in rows],
        },
        schema=dict(KLINE_SCHEMA),
    )
    return df


# --------------------------------------------------------------------------- funding


def fetch_recent_funding(
    exchange: CcxtExchange, venue_symbol: str, since: datetime, now: datetime
) -> pl.DataFrame:
    """Funding events with `since <= ts <= now`, unique, sorted, FUNDING_SCHEMA."""
    since = ensure_utc(since, "since")
    now = ensure_utc(now, "now")
    check_symbol(venue_symbol)
    if since > now:
        return pl.DataFrame(schema=dict(FUNDING_SCHEMA))

    def fetch(start: int) -> list[Mapping[str, object]]:
        params: dict[str, object] = {
            "symbol": venue_symbol,
            "startTime": start,
            "limit": PAGE_LIMIT,
        }
        return exchange.fapiPublicGetFundingRate(params)

    def open_ms(row: Mapping[str, object]) -> int:
        return _to_int(row.get("fundingTime"), "fundingTime")

    raw = _paginate(fetch, open_ms, _ms(since), _ms(now))
    since_ms, now_ms = _ms(since), _ms(now)
    seen: dict[int, Mapping[str, object]] = {}
    for row in raw:
        t = open_ms(row)
        if since_ms <= t <= now_ms and t not in seen:
            seen[t] = row
    keys = sorted(seen)
    rows = [seen[k] for k in keys]

    def mark(row: Mapping[str, object]) -> float | None:
        value = row.get("markPrice")
        if value is None or value == "":
            return None
        return _to_float(value, "markPrice")

    return pl.DataFrame(
        {
            "ts": _ts_frame(keys),
            "rate": [_to_float(r.get("fundingRate"), "fundingRate") for r in rows],
            "mark_price": [mark(r) for r in rows],
        },
        schema=dict(FUNDING_SCHEMA),
    )
