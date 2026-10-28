"""Tests for bt.data.validate: schemas, errors, gaps and flags."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl

from bt.data.validate import (
    FUNDING_INTERVAL,
    FUNDING_SCHEMA,
    KLINE_SCHEMA,
    TIMEFRAME_DELTA,
    BarFlag,
    Gap,
    validate_funding,
    validate_klines,
)

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def D(n: int) -> datetime:  # noqa: N802
    return T0 + timedelta(days=n)


def H(n: float) -> datetime:  # noqa: N802
    return T0 + timedelta(hours=n)


def klines(rows: list[dict[str, Any]], *, ts_dtype: pl.DataType | None = None) -> pl.DataFrame:
    base: dict[str, Any] = {
        "open": 100.0,
        "high": 110.0,
        "low": 90.0,
        "close": 105.0,
        "volume": 1.0,
        "quote_volume": 100.0,
        "trades": 10,
    }
    full = [{**base, **r} for r in rows]
    schema = dict(KLINE_SCHEMA)
    if ts_dtype is not None:
        schema["ts"] = ts_dtype
    return pl.DataFrame(full, schema=schema)


def daily(days: list[int], **over: float) -> pl.DataFrame:
    return klines([{"ts": D(d), **over} for d in days])


def funding(rows: list[dict[str, Any]]) -> pl.DataFrame:
    full = [{"rate": 0.0001, "mark_price": 40000.0, **r} for r in rows]
    return pl.DataFrame(full, schema=dict(FUNDING_SCHEMA))


def fund_ts(hours: list[float]) -> pl.DataFrame:
    return funding([{"ts": H(h)} for h in hours])


def test_constants() -> None:
    assert TIMEFRAME_DELTA["1d"] == timedelta(days=1)
    assert TIMEFRAME_DELTA["4h"] == timedelta(hours=4)
    assert timedelta(hours=8) == FUNDING_INTERVAL


def test_clean_klines() -> None:
    r = validate_klines(daily(list(range(10))), "1d", "perp/klines/1d/BTCUSDT")
    assert r.ok
    assert r.dataset == "perp/klines/1d/BTCUSDT"
    assert r.rows == 10
    assert r.first_ts == D(0)
    assert r.last_ts == D(9)
    assert r.errors == ()
    assert r.gaps == ()
    assert r.flags == ()


def test_daily_gap() -> None:
    r = validate_klines(daily([0, 1, 4]), "1d", "x")
    assert r.ok
    assert r.gaps == (Gap(after_ts=D(1), before_ts=D(4), missing=2),)


def test_4h_gap_and_grid() -> None:
    df = klines([{"ts": H(h)} for h in (0, 4, 12)])
    r = validate_klines(df, "4h", "x")
    assert r.ok
    assert r.gaps == (Gap(after_ts=H(4), before_ts=H(12), missing=1),)
    off = klines([{"ts": H(h)} for h in (0, 1)])
    r2 = validate_klines(off, "4h", "x")
    assert not r2.ok
    assert any("grid" in e for e in r2.errors)
    assert r2.gaps == ()  # no wrong grid -> no gaps computed beyond diff <= delta


def test_duplicate_and_unsorted() -> None:
    r = validate_klines(daily([0, 1, 2, 2, 3]), "1d", "x")
    assert not r.ok
    assert r.errors == ("duplicate ts: 2 rows, first 2024-01-03T00:00:00+00:00",)
    assert r.gaps == ()
    r2 = validate_klines(daily([1, 0]), "1d", "x")
    assert not r2.ok
    assert len(r2.errors) == 1
    assert "unsorted" in r2.errors[0]


def test_schema_errors() -> None:
    ms = klines([{"ts": D(0)}], ts_dtype=pl.Datetime("ms", "UTC"))
    r = validate_klines(ms, "1d", "x")
    assert len(r.errors) == 1
    assert "schema" in r.errors[0]
    no_trades = daily([0, 1]).drop("trades")
    assert "schema" in validate_klines(no_trades, "1d", "x").errors[0]
    naive = klines([{"ts": D(0).replace(tzinfo=None)}], ts_dtype=pl.Datetime("us"))
    assert "schema" in validate_klines(naive, "1d", "x").errors[0]
    reordered = daily([0]).select(
        ["open", "ts", "high", "low", "close", "volume", "quote_volume", "trades"]
    )
    assert "schema" in validate_klines(reordered, "1d", "x").errors[0]


def test_empty() -> None:
    r = validate_klines(daily([]), "1d", "x")
    assert r.errors == ("empty",)
    assert r.rows == 0
    assert r.first_ts is None
    assert r.last_ts is None


def test_price_errors() -> None:
    assert not validate_klines(daily([0], high=104.0), "1d", "x").ok
    assert not validate_klines(daily([0], low=106.0), "1d", "x").ok
    assert not validate_klines(daily([0], close=0.0), "1d", "x").ok
    assert not validate_klines(daily([0], volume=-1.0), "1d", "x").ok
    assert not validate_klines(daily([0], quote_volume=-1.0), "1d", "x").ok
    assert not validate_klines(daily([0], trades=-1), "1d", "x").ok
    # high < low with open/close inside neither bound
    assert not validate_klines(daily([0], high=80.0, low=95.0, open=90.0, close=90.0), "1d", "x").ok


def test_null_close_named() -> None:
    df = klines([{"ts": D(0)}, {"ts": D(1), "close": None}])
    r = validate_klines(df, "1d", "x")
    assert not r.ok
    assert any("close" in e and "1" in e for e in r.errors)


def _closes(closes: list[float]) -> pl.DataFrame:
    return klines(
        [{"ts": D(i), "open": c, "high": c, "low": c, "close": c} for i, c in enumerate(closes)]
    )


def test_jump_boundaries() -> None:
    assert validate_klines(_closes([100.0, 150.0]), "1d", "x").flags == ()
    assert validate_klines(_closes([100.0, 151.0]), "1d", "x").flags == (BarFlag(D(1), "jump"),)
    assert validate_klines(_closes([100.0, 50.0]), "1d", "x").flags == ()
    assert validate_klines(_closes([100.0, 49.0]), "1d", "x").flags == (BarFlag(D(1), "jump"),)


def test_wide_range_boundaries() -> None:
    ok = validate_klines(daily([0], low=50.0, high=100.0, open=60.0, close=70.0), "1d", "x")
    assert ok.flags == ()
    wide = validate_klines(daily([0], low=50.0, high=101.0, open=60.0, close=70.0), "1d", "x")
    assert wide.ok
    assert wide.flags == (BarFlag(D(0), "wide_range"),)


def test_zero_volume_flag_sorted() -> None:
    df = klines(
        [
            {"ts": D(0), "volume": 0.0, "low": 10.0, "high": 110.0, "open": 50.0, "close": 60.0},
            {"ts": D(1), "volume": 0.0, "low": 10.0, "open": 50.0, "close": 60.0},
        ]
    )
    r = validate_klines(df, "1d", "x")
    assert r.ok
    assert r.flags == (
        BarFlag(D(0), "wide_range"),
        BarFlag(D(0), "zero_volume"),
        BarFlag(D(1), "wide_range"),
        BarFlag(D(1), "zero_volume"),
    )


def test_funding_clean_and_gaps() -> None:
    r = validate_funding(fund_ts([8.0 * i for i in range(9)]), "perp/funding/BTCUSDT")
    assert r.ok
    assert r.rows == 9
    assert r.first_ts == H(0)
    assert r.last_ts == H(64)
    assert r.gaps == ()
    assert r.flags == ()
    hours = [8.0 * i for i in range(9)]
    del hours[4]
    g = validate_funding(fund_ts(hours), "x")
    assert g.ok
    assert g.gaps == (Gap(after_ts=H(24), before_ts=H(40), missing=1),)


def test_funding_tolerance_and_4h() -> None:
    jitter = funding([{"ts": T0 + timedelta(hours=8 * i, seconds=30 * i)} for i in range(5)])
    assert validate_funding(jitter, "x").gaps == ()
    mixed = fund_ts([0.0, 8.0, 12.0, 16.0, 20.0, 28.0])
    assert validate_funding(mixed, "x").gaps == ()


def test_funding_rate_flags_and_errors() -> None:
    def one(**kw: float | None) -> pl.DataFrame:
        return funding([{"ts": H(0), **kw}])

    assert validate_funding(one(rate=0.01), "x").flags == ()
    assert validate_funding(one(rate=0.0101), "x").flags == (BarFlag(H(0), "extreme_rate"),)
    assert validate_funding(one(rate=-0.0101), "x").flags == (BarFlag(H(0), "extreme_rate"),)
    assert not validate_funding(one(rate=None), "x").ok
    assert not validate_funding(one(mark_price=0.0), "x").ok
    assert validate_funding(one(mark_price=None), "x").ok


def test_funding_schema_empty_order() -> None:
    assert "schema" in validate_funding(fund_ts([0.0]).drop("rate"), "x").errors[0]
    assert validate_funding(funding([]), "x").errors == ("empty",)
    r = validate_funding(fund_ts([8.0, 0.0]), "x")
    assert "unsorted" in r.errors[0]
    d = validate_funding(fund_ts([0.0, 0.0]), "x")
    assert "duplicate" in d.errors[0]


def test_inputs_not_mutated() -> None:
    k = daily([0, 1, 4])
    f = fund_ts([0.0, 8.0])
    k0, f0 = k.clone(), f.clone()
    validate_klines(k, "1d", "x")
    validate_funding(f, "x")
    assert k.equals(k0)
    assert f.equals(f0)
