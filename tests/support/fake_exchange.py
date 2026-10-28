"""In-memory `CcxtExchange` for tests: canned markets, kline rows, funding rows, a call log.

Serves pages the way Binance does: rows with open time >= `startTime`, ascending, at most
`limit` rows (default 500, max 1000). No network, ever.

    fake = FakeExchange(markets=[spot_market(), perp_market()])
    fake.set_klines("fapiPublicGetKlines", "BTCUSDT", "1d", [kline_row(ms), ...])
    fake.set_funding("BTCUSDT", [funding_row(ms, 0.0001, "42300.5")])
    fake.fail_network(2)            # next 2 data calls raise ccxt.NetworkError
    fake.calls                      # [(method, params), ...] in call order
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Final

import ccxt  # pyright: ignore[reportMissingTypeStubs]

KLINE_ENDPOINTS: Final = ("publicGetKlines", "fapiPublicGetKlines")
_DEFAULT_LIMIT: Final = 500
_MAX_LIMIT: Final = 1000


def epoch_ms(ts: datetime) -> int:
    """Milliseconds since the Unix epoch of an aware datetime."""
    return int(ts.timestamp() * 1000)


def spot_market(
    venue_symbol: str = "BTCUSDT", base: str = "BTC", quote: str = "USDT"
) -> dict[str, object]:
    """A ccxt-shaped spot market with plausible Binance rules."""
    return {
        "id": venue_symbol,
        "symbol": f"{base}/{quote}",
        "spot": True,
        "swap": False,
        "linear": None,
        "base": base,
        "quote": quote,
        "limits": {
            "amount": {"min": 1e-5},
            "cost": {"min": 5.0},
            "leverage": {"max": None},
        },
        "precision": {"amount": 1e-5, "price": 0.01},
        "maker": 0.001,
        "taker": 0.001,
        "info": {},
    }


def perp_market(
    venue_symbol: str = "BTCUSDT", base: str = "BTC", quote: str = "USDT"
) -> dict[str, object]:
    """A ccxt-shaped linear swap market (note ccxt gives no leverage limit)."""
    return {
        "id": venue_symbol,
        "symbol": f"{base}/{quote}:{quote}",
        "spot": False,
        "swap": True,
        "linear": True,
        "base": base,
        "quote": quote,
        "limits": {
            "amount": {"min": 0.001},
            "cost": {"min": 100.0},
            "leverage": {"max": None},
        },
        "precision": {"amount": 0.001, "price": 0.1},
        "maker": 0.0002,
        "taker": 0.0005,
        "info": {"maintMarginPercent": "2.5000"},
    }


def kline_row(open_ms: int, close_ms: int, price: float = 100.0) -> list[object]:
    """A Binance 12-field kline array (strings, like the real API) with a flat-ish bar."""
    p = f"{price:.2f}"
    return [
        open_ms,
        p,
        f"{price + 1:.2f}",
        f"{price - 1:.2f}",
        p,
        "10.0",
        close_ms,
        "1000.0",
        42,
        "5.0",
        "500.0",
        "0",
    ]


def funding_row(funding_ms: int, rate: str | float, mark_price: str | None) -> dict[str, object]:
    """A Binance fundingRate record."""
    row: dict[str, object] = {"symbol": "BTCUSDT", "fundingTime": funding_ms, "fundingRate": rate}
    if mark_price is not None:
        row["markPrice"] = mark_price
    return row


class FakeExchange:
    """Implements `CcxtExchange` over canned data; records every call in `calls`."""

    def __init__(self, markets: Iterable[Mapping[str, object]] = ()) -> None:
        self._markets: list[Mapping[str, object]] = list(markets)
        self._klines: dict[tuple[str, str, str], list[list[object]]] = {}
        self._funding: dict[str, list[Mapping[str, object]]] = {}
        self._network_failures = 0
        self.calls: list[tuple[str, dict[str, object]]] = []

    # -- configuration -------------------------------------------------------------

    def set_klines(
        self, endpoint: str, symbol: str, interval: str, rows: Iterable[list[object]]
    ) -> None:
        """Canned rows for (endpoint, symbol, interval); endpoint is a method name."""
        if endpoint not in KLINE_ENDPOINTS:
            raise ValueError(f"unknown kline endpoint {endpoint!r}")
        self._klines[(endpoint, symbol, interval)] = list(rows)

    def set_funding(self, symbol: str, rows: Iterable[Mapping[str, object]]) -> None:
        """Canned funding records for a symbol."""
        self._funding[symbol] = list(rows)

    def fail_network(self, times: int) -> None:
        """Make the next `times` data calls (not load_markets) raise ccxt.NetworkError."""
        self._network_failures = times

    def calls_to(self, method: str) -> list[dict[str, object]]:
        """Params of every call to `method`, in order."""
        return [p for m, p in self.calls if m == method]

    # -- CcxtExchange ----------------------------------------------------------------

    def load_markets(self) -> Mapping[str, Mapping[str, object]]:
        self.calls.append(("load_markets", {}))
        out: dict[str, Mapping[str, object]] = {}
        for i, m in enumerate(self._markets):
            out[str(m.get("symbol", i))] = m
        return out

    def publicGetKlines(self, params: Mapping[str, object]) -> list[list[object]]:  # noqa: N802
        return self._klines_page("publicGetKlines", params)

    def fapiPublicGetKlines(self, params: Mapping[str, object]) -> list[list[object]]:  # noqa: N802
        return self._klines_page("fapiPublicGetKlines", params)

    def fapiPublicGetFundingRate(  # noqa: N802
        self, params: Mapping[str, object]
    ) -> list[Mapping[str, object]]:
        self._record("fapiPublicGetFundingRate", params)
        rows = self._funding.get(str(params["symbol"]), [])
        start, limit = _start_limit(params)
        page = sorted((r for r in rows if _as_int(r["fundingTime"]) >= start), key=_funding_key)
        return page[:limit]

    # -- internals -------------------------------------------------------------------

    def _record(self, method: str, params: Mapping[str, object]) -> None:
        self.calls.append((method, dict(params)))
        if self._network_failures > 0:
            self._network_failures -= 1
            raise ccxt.NetworkError(f"fake network failure in {method}")

    def _klines_page(self, method: str, params: Mapping[str, object]) -> list[list[object]]:
        self._record(method, params)
        rows = self._klines.get((method, str(params["symbol"]), str(params["interval"])), [])
        start, limit = _start_limit(params)
        page = sorted((r for r in rows if _as_int(r[0]) >= start), key=lambda r: _as_int(r[0]))
        return page[:limit]


def _as_int(value: object) -> int:
    assert isinstance(value, (int, str))
    return int(value)


def _funding_key(row: Mapping[str, object]) -> int:
    return _as_int(row["fundingTime"])


def _start_limit(params: Mapping[str, object]) -> tuple[int, int]:
    start = _as_int(params.get("startTime", 0))
    limit = min(_as_int(params.get("limit", _DEFAULT_LIMIT)), _MAX_LIMIT)
    return start, limit
