"""Tests for bt.data.fetch_ccxt against the in-memory fake exchange. No network."""

from __future__ import annotations

import socket
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import cast

import ccxt  # pyright: ignore[reportMissingTypeStubs]
import pytest

import bt.data.fetch_ccxt as fc
from bt.core.errors import DataError, TimezoneError
from bt.core.types import Instrument, InstrumentKind
from bt.data.fetch_ccxt import (
    fetch_instruments,
    fetch_recent_funding,
    fetch_recent_klines,
    make_exchange,
)
from bt.data.validate import FUNDING_SCHEMA, KLINE_SCHEMA
from tests.support.fake_exchange import (
    FakeExchange,
    epoch_ms,
    funding_row,
    kline_row,
    perp_market,
    spot_market,
)

DAY = timedelta(days=1)
H4 = timedelta(hours=4)


def D(day: int) -> datetime:  # noqa: N802
    return datetime(2024, 3, day, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _no_retry_wait(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    monkeypatch.setattr(fc, "_RETRY_WAIT_S", 0.0)


def bars(start: datetime, n: int, step: timedelta) -> list[list[object]]:
    return [
        kline_row(epoch_ms(start + i * step), epoch_ms(start + (i + 1) * step) - 1, 100.0 + i)
        for i in range(n)
    ]


def mutate(market: dict[str, object], fn: Callable[[dict[str, object]], None]) -> dict[str, object]:
    fn(market)
    return market


# ------------------------------------------------------------------ instruments


def test_spot_instrument_mapping() -> None:
    out = fetch_instruments(FakeExchange([spot_market()]), "spot", ["BTCUSDT"])
    assert out == [
        Instrument(
            "BTCUSDT", "BTCUSDT", InstrumentKind.SPOT, "BTC", "USDT",
            1e-5, 1e-5, 5.0, 0.01, 0.001, 0.001, 1.0, 0.0,
        )
    ]  # fmt: skip


def test_perp_instrument_mapping() -> None:
    (inst,) = fetch_instruments(FakeExchange([perp_market()]), "perp", ["BTCUSDT"])
    assert inst.symbol == "BTCUSDT-PERP"
    assert inst.venue_symbol == "BTCUSDT"
    assert inst.kind is InstrumentKind.PERP
    assert inst.min_qty == pytest.approx(0.001, abs=1e-12)
    assert inst.qty_step == pytest.approx(0.001, abs=1e-12)
    assert inst.min_notional == pytest.approx(100.0, abs=1e-9)
    assert inst.price_tick == pytest.approx(0.1, abs=1e-12)
    assert inst.maker_fee == pytest.approx(0.0002, abs=1e-12)
    assert inst.taker_fee == pytest.approx(0.0005, abs=1e-12)
    assert inst.max_leverage == 1.0
    assert inst.maint_margin_rate == pytest.approx(0.025, abs=1e-12)


def test_perp_ignores_ccxt_leverage_limit() -> None:
    m = perp_market()
    m["limits"] = {
        "amount": {"min": 0.001},
        "cost": {"min": 100.0},
        "leverage": {"max": 125},
    }
    (inst,) = fetch_instruments(FakeExchange([m]), "perp", ["BTCUSDT"])
    assert inst.max_leverage == 1.0


def test_market_kind_selects_matching_market() -> None:
    fake = FakeExchange([spot_market(), perp_market()])
    (perp,) = fetch_instruments(fake, "perp", ["BTCUSDT"])
    (spot,) = fetch_instruments(fake, "spot", ["BTCUSDT"])
    assert perp.kind is InstrumentKind.PERP
    assert perp.symbol == "BTCUSDT-PERP"
    assert spot.kind is InstrumentKind.SPOT
    assert spot.symbol == "BTCUSDT"


def test_missing_symbols_listed_together() -> None:
    fake = FakeExchange([spot_market()])
    with pytest.raises(DataError) as ei:
        fetch_instruments(fake, "spot", ["BTCUSDT", "FOOUSDT", "BARUSDT"])
    assert "FOOUSDT" in str(ei.value)
    assert "BARUSDT" in str(ei.value)


def test_symbol_only_on_other_market_is_missing() -> None:
    with pytest.raises(DataError, match="BTCUSDT"):
        fetch_instruments(FakeExchange([spot_market()]), "perp", ["BTCUSDT"])


def test_perp_missing_maint_margin_field() -> None:
    m = perp_market()
    m["info"] = {}
    with pytest.raises(DataError, match="maintMarginPercent"):
        fetch_instruments(FakeExchange([m]), "perp", ["BTCUSDT"])


def test_precision_amount_none() -> None:
    m = spot_market()
    m["precision"] = {"amount": None, "price": 0.01}
    with pytest.raises(DataError, match=r"BTCUSDT.*precision\.amount"):
        fetch_instruments(FakeExchange([m]), "spot", ["BTCUSDT"])


@pytest.mark.parametrize("field", ["maker", "taker"])
def test_missing_fee_field(field: str) -> None:
    m = spot_market()
    del m[field]
    with pytest.raises(DataError, match=field):
        fetch_instruments(FakeExchange([m]), "spot", ["BTCUSDT"])


def test_min_notional_none() -> None:
    m = spot_market()
    m["limits"] = {"amount": {"min": 1e-5}, "cost": {"min": None}}
    with pytest.raises(DataError, match=r"cost\.min"):
        fetch_instruments(FakeExchange([m]), "spot", ["BTCUSDT"])


def test_invalid_precision_price_zero_is_data_error() -> None:
    m = spot_market()
    m["precision"] = {"amount": 1e-5, "price": 0}
    with pytest.raises(DataError, match="BTCUSDT") as ei:
        fetch_instruments(FakeExchange([m]), "spot", ["BTCUSDT"])
    assert ei.value.__cause__ is not None


def test_output_order_follows_input_and_markets_loaded_once() -> None:
    fake = FakeExchange([spot_market("BTCUSDT"), spot_market("ETHUSDT", "ETH")])
    out = fetch_instruments(fake, "spot", ["ETHUSDT", "BTCUSDT"])
    assert [i.venue_symbol for i in out] == ["ETHUSDT", "BTCUSDT"]
    assert len(fake.calls_to("load_markets")) == 1


# ------------------------------------------------------------------ klines


def test_recent_klines_drops_unclosed_bar_and_uses_perp_endpoint() -> None:
    fake = FakeExchange()
    fake.set_klines("fapiPublicGetKlines", "BTCUSDT", "1d", bars(D(1), 10, DAY))
    df = fetch_recent_klines(
        fake, "perp", "1d", "BTCUSDT", D(9), datetime(2024, 3, 10, 1, tzinfo=UTC)
    )
    assert df["ts"].to_list() == [D(9)]
    assert [m for m, _ in fake.calls] == ["fapiPublicGetKlines"]
    assert fake.calls[0][1]["startTime"] == epoch_ms(D(9))
    assert fake.calls[0][1]["interval"] == "1d"
    assert fake.calls[0][1]["limit"] == 1000


def test_recent_klines_spot_endpoint() -> None:
    fake = FakeExchange()
    fake.set_klines("publicGetKlines", "BTCUSDT", "1d", bars(D(1), 10, DAY))
    df = fetch_recent_klines(
        fake, "spot", "1d", "BTCUSDT", D(8), datetime(2024, 3, 10, 1, tzinfo=UTC)
    )
    assert df["ts"].to_list() == [D(8), D(9)]
    assert [m for m, _ in fake.calls] == ["publicGetKlines"]


def test_bar_closing_exactly_at_now_is_kept() -> None:
    fake = FakeExchange()
    fake.set_klines("publicGetKlines", "BTCUSDT", "1d", bars(D(1), 10, DAY))
    df = fetch_recent_klines(fake, "spot", "1d", "BTCUSDT", D(9), D(10))
    assert df["ts"].to_list() == [D(9)]


def test_pagination_over_2500_bars() -> None:
    start = datetime(2023, 1, 1, tzinfo=UTC)
    fake = FakeExchange()
    rows = bars(start, 2500, H4)
    fake.set_klines("fapiPublicGetKlines", "BTCUSDT", "4h", rows)
    now = start + 2500 * H4 + H4
    df = fetch_recent_klines(fake, "perp", "4h", "BTCUSDT", start, now)
    assert df.height == 2500
    ts = df["ts"].to_list()
    assert ts == sorted(set(ts))
    calls = fake.calls_to("fapiPublicGetKlines")
    assert len(calls) == 3
    assert calls[0]["startTime"] == epoch_ms(start)
    assert calls[1]["startTime"] == epoch_ms(start + 999 * H4) + 1
    assert calls[2]["startTime"] == epoch_ms(start + 1999 * H4) + 1


def test_pagination_stops_when_row_reaches_now() -> None:
    start = datetime(2023, 1, 1, tzinfo=UTC)
    fake = FakeExchange()
    fake.set_klines("fapiPublicGetKlines", "BTCUSDT", "4h", bars(start, 2500, H4))
    now = start + 1500 * H4  # inside the second page
    df = fetch_recent_klines(fake, "perp", "4h", "BTCUSDT", start, now)
    assert df.height == 1500
    assert len(fake.calls) == 2


def test_kline_field_mapping_and_schema() -> None:
    raw: list[object] = [
        1704067200000, "42283.58", "44184.10", "42180.77", "44179.55", "27174.29903",
        1704153599999, "1162870513.34", 1043852, "13517.74", "578460929.45", "0",
    ]  # fmt: skip
    fake = FakeExchange()
    fake.set_klines("publicGetKlines", "BTCUSDT", "1d", [raw])
    df = fetch_recent_klines(
        fake,
        "spot",
        "1d",
        "BTCUSDT",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 3, tzinfo=UTC),
    )
    assert dict(df.schema) == dict(KLINE_SCHEMA)
    row = df.row(0, named=True)
    assert row["ts"] == datetime(2024, 1, 1, tzinfo=UTC)
    assert row["open"] == pytest.approx(42283.58, abs=1e-9)
    assert row["high"] == pytest.approx(44184.10, abs=1e-9)
    assert row["low"] == pytest.approx(42180.77, abs=1e-9)
    assert row["close"] == pytest.approx(44179.55, abs=1e-9)
    assert row["volume"] == pytest.approx(27174.29903, abs=1e-9)
    assert row["quote_volume"] == pytest.approx(1162870513.34, abs=1e-6)
    assert row["trades"] == 1043852


def test_klines_deduplicated_and_sorted() -> None:
    rows = bars(D(1), 4, DAY)
    fake = FakeExchange()
    fake.set_klines(
        "publicGetKlines", "BTCUSDT", "1d", [rows[2], rows[0], rows[1], rows[1], rows[3]]
    )
    df = fetch_recent_klines(fake, "spot", "1d", "BTCUSDT", D(1), D(10))
    assert df["ts"].to_list() == [D(1), D(2), D(3), D(4)]


def test_klines_empty_result_has_schema() -> None:
    df = fetch_recent_klines(FakeExchange(), "spot", "1d", "BTCUSDT", D(1), D(10))
    assert df.height == 0
    assert dict(df.schema) == dict(KLINE_SCHEMA)


def test_since_after_now_is_empty_without_calls() -> None:
    fake = FakeExchange()
    k = fetch_recent_klines(fake, "spot", "1d", "BTCUSDT", D(10), D(1))
    f = fetch_recent_funding(fake, "BTCUSDT", D(10), D(1))
    assert k.height == 0
    assert dict(k.schema) == dict(KLINE_SCHEMA)
    assert f.height == 0
    assert dict(f.schema) == dict(FUNDING_SCHEMA)
    assert fake.calls == []


def test_naive_datetimes_rejected() -> None:
    fake = FakeExchange()
    naive = datetime(2024, 3, 10)  # noqa: DTZ001
    with pytest.raises(TimezoneError):
        fetch_recent_klines(fake, "spot", "1d", "BTCUSDT", D(1), naive)
    with pytest.raises(TimezoneError):
        fetch_recent_klines(fake, "spot", "1d", "BTCUSDT", naive, D(2))
    with pytest.raises(TimezoneError):
        fetch_recent_funding(fake, "BTCUSDT", D(1), naive)
    assert fake.calls == []


def test_bad_symbol_rejected() -> None:
    with pytest.raises(DataError):
        fetch_recent_klines(FakeExchange(), "spot", "1d", "btc/usdt", D(1), D(2))


def test_malformed_kline_row_is_data_error() -> None:
    fake = FakeExchange()
    fake.set_klines("publicGetKlines", "BTCUSDT", "1d", [[epoch_ms(D(1)), 2, 3]])
    with pytest.raises(DataError):
        fetch_recent_klines(fake, "spot", "1d", "BTCUSDT", D(1), D(10))


# ------------------------------------------------------------------ funding


def test_funding_mapping_and_null_mark_price() -> None:
    t0 = 1704067200000
    fake = FakeExchange()
    fake.set_funding(
        "BTCUSDT",
        [
            funding_row(t0, "0.00010000", "42300.5"),
            funding_row(t0 + 8 * 3600_000, "-0.00020000", ""),
            funding_row(t0 + 16 * 3600_000, "0.00005000", None),
        ],
    )
    df = fetch_recent_funding(
        fake, "BTCUSDT", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)
    )
    assert dict(df.schema) == dict(FUNDING_SCHEMA)
    assert df["rate"].to_list() == pytest.approx([0.0001, -0.0002, 0.00005], abs=1e-12)
    marks = df["mark_price"].to_list()
    assert marks[0] == pytest.approx(42300.5, abs=1e-9)
    assert marks[1] is None
    assert marks[2] is None
    assert df["ts"][0] == datetime(2024, 1, 1, tzinfo=UTC)
    params = fake.calls[0][1]
    assert fake.calls[0][0] == "fapiPublicGetFundingRate"
    assert params == {"symbol": "BTCUSDT", "startTime": t0, "limit": 1000}


def test_funding_range_dedupe_sorted_and_paginated() -> None:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    step = timedelta(hours=8)
    rows = [funding_row(epoch_ms(start + i * step), "0.0001", "100") for i in range(1500)]
    fake = FakeExchange()
    fake.set_funding("BTCUSDT", [*reversed(rows), rows[0]])
    since = start + 10 * step
    now = start + 1400 * step
    df = fetch_recent_funding(fake, "BTCUSDT", since, now)
    assert df["ts"][0] == since
    assert df["ts"][-1] == now
    assert df.height == 1391
    assert df["ts"].is_sorted()
    assert df["ts"].n_unique() == df.height
    calls = fake.calls_to("fapiPublicGetFundingRate")
    assert len(calls) == 2
    assert calls[1]["startTime"] == epoch_ms(since + 999 * step) + 1


# ------------------------------------------------------------------ errors / retry


def test_network_error_retried_then_succeeds() -> None:
    fake = FakeExchange()
    fake.set_klines("publicGetKlines", "BTCUSDT", "1d", bars(D(1), 3, DAY))
    fake.fail_network(2)
    df = fetch_recent_klines(fake, "spot", "1d", "BTCUSDT", D(1), D(10))
    assert df.height == 3
    assert len(fake.calls) == 3


def test_network_error_three_times_is_data_error() -> None:
    fake = FakeExchange()
    fake.set_klines("publicGetKlines", "BTCUSDT", "1d", bars(D(1), 3, DAY))
    fake.fail_network(3)
    with pytest.raises(DataError) as ei:
        fetch_recent_klines(fake, "spot", "1d", "BTCUSDT", D(1), D(10))
    assert isinstance(ei.value.__cause__, ccxt.NetworkError)
    assert len(fake.calls) == 3


def test_other_ccxt_error_is_immediate_data_error() -> None:
    class Bad(FakeExchange):
        def publicGetKlines(self, params: object) -> list[list[object]]:  # noqa: N802
            self.calls.append(("publicGetKlines", {}))
            raise ccxt.BadSymbol("nope")

    fake = Bad()
    with pytest.raises(DataError) as ei:
        fetch_recent_klines(fake, "spot", "1d", "BTCUSDT", D(1), D(10))
    assert isinstance(ei.value.__cause__, ccxt.BadSymbol)
    assert len(fake.calls) == 1


# ------------------------------------------------------------------ make_exchange


def test_make_exchange_is_offline_and_keyless(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_a: object, **_k: object) -> None:
        raise AssertionError("socket connect attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    ex: object = make_exchange()
    assert not getattr(ex, "apiKey", None)
    assert not getattr(ex, "secret", None)
    assert getattr(ex, "enableRateLimit", False)
    options = cast(dict[str, object], getattr(ex, "options"))  # noqa: B009
    assert cast(dict[str, object], options["fetchMarkets"])["types"] == ["spot", "linear"]
