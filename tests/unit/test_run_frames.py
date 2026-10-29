"""Tests for bt.runs.frames."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from bt.core.events import LogRecord, SkippedOrder
from bt.core.types import Side
from bt.engine.loop import EngineResult
from bt.runs.frames import (
    EQUITY_SCHEMA,
    MESSAGES_SCHEMA,
    ORDERS_SCHEMA,
    TRADES_SCHEMA,
    equity_frame,
    messages_frame,
    orders_frame,
    trades_frame,
)
from tests.golden.test_golden_spot import run_golden
from tests.support.make_golden_fixture import day


@pytest.fixture(scope="module")
def result() -> EngineResult:
    return run_golden()


def _empty(r: EngineResult) -> EngineResult:
    return replace(r, equity=(), fills=(), orders=(), skipped=(), messages=())


def test_equity_frame_golden(result: EngineResult) -> None:
    df = equity_frame(result, "1d")
    assert df.height == 5
    assert df["ts"].to_list() == [day(i) for i in range(1, 6)]
    assert df["close_ts"].to_list() == [day(i) for i in range(2, 7)]
    assert df["equity_pre_tax"].to_list() == df["equity_after_tax"].to_list()
    assert all(d <= 0 for d in df["drawdown"].to_list())
    assert df["drawdown"][0] == 0.0
    assert dict(df.schema) == dict(EQUITY_SCHEMA)
    assert df.columns == list(EQUITY_SCHEMA)


def test_equity_frame_pre_tax_adds_tax(result: EngineResult) -> None:
    snaps = tuple(replace(s, tax_paid=2.5) for s in result.equity)
    df = equity_frame(replace(result, equity=snaps), "4h")
    for pre, post in zip(df["equity_pre_tax"], df["equity_after_tax"], strict=True):
        assert pre == pytest.approx(post + 2.5, abs=1e-9)
    assert df["close_ts"][0] - df["ts"][0] == timedelta(hours=4)


def test_trades_frame(result: EngineResult) -> None:
    df = trades_frame(result)
    assert df.height == len(result.fills)
    assert dict(df.schema) == dict(TRADES_SCHEMA)
    f0 = result.fills[0]
    row = df.row(0, named=True)
    assert row["event"] == "fill"
    assert row["order_id"] == f0.order_id
    assert row["notional"] == pytest.approx(f0.qty * f0.price, abs=1e-9)
    assert row["cash_flow"] == pytest.approx(-(row["notional"] + f0.fee), abs=1e-9)
    assert f0.side == Side.BUY
    assert df["tag"][-1] == "close_at_end"
    assert df["side"].to_list() == [f.side.value for f in result.fills]


def test_trades_frame_sell_cash_flow(result: EngineResult) -> None:
    df = trades_frame(result)
    for f, row in zip(result.fills, df.iter_rows(named=True), strict=True):
        n = f.qty * f.price
        want = -(n + f.fee) if f.side == Side.BUY else n - f.fee
        assert row["cash_flow"] == pytest.approx(want, abs=1e-9)


def test_orders_frame(result: EngineResult) -> None:
    df = orders_frame(result)
    assert df.height == len(result.orders)
    assert dict(df.schema) == dict(ORDERS_SCHEMA)
    assert df["status"].to_list() == [r.status.value for r in result.orders]
    assert "new" in df["status"].to_list()
    assert 19.80 in df["limit_price"].to_list()
    assert df["limit_price"].null_count() > 0


def test_orders_frame_skipped(result: EngineResult) -> None:
    sk = SkippedOrder(day(2), "AAAUSDT", Side.BUY, 0.0, 100.0, "zero_qty")
    df = orders_frame(replace(result, skipped=(sk,)))
    assert df.height == len(result.orders) + 1
    last = df.row(-1, named=True)
    assert last["status"] == "skipped"
    assert last["order_id"] is None
    assert last["type"] is None
    assert last["reason"] == "zero_qty"
    assert dict(df.schema) == dict(ORDERS_SCHEMA)


def test_empty_frames(result: EngineResult) -> None:
    e = _empty(result)
    for df, schema in (
        (equity_frame(e, "1d"), EQUITY_SCHEMA),
        (trades_frame(e), TRADES_SCHEMA),
        (orders_frame(e), ORDERS_SCHEMA),
        (messages_frame(e), MESSAGES_SCHEMA),
    ):
        assert df.height == 0
        assert df.columns == list(schema)
        assert dict(df.schema) == dict(schema)


def test_messages_frame(result: EngineResult) -> None:
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    rec = LogRecord(ts, "m", (("a", 1), ("b", "x")))
    df = messages_frame(replace(result, messages=(rec,)))
    assert df["fields"].to_list() == ['{"a":1,"b":"x"}']
    assert df["message"].to_list() == ["m"]
    assert df["ts"].to_list() == [ts]
    assert dict(df.schema) == dict(MESSAGES_SCHEMA)
    assert isinstance(df, pl.DataFrame)
