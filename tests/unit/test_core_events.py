import pickle
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from bt.core.errors import InvariantError, TimezoneError
from bt.core.events import (
    EquitySnapshot,
    EventLog,
    LogRecord,
    OrderRecord,
    SkippedOrder,
)
from bt.core.types import Fill, Order, OrderStatus, Side

T = datetime(2024, 1, 1, tzinfo=UTC)
LONDON = datetime(2024, 1, 1, tzinfo=ZoneInfo("Europe/London"))
ORDER = Order("o1", "BTCUSDT", Side.BUY, 1.0)


def _snap(**kw: object) -> EquitySnapshot:
    base: dict[str, Any] = dict(
        ts=T,
        equity=1000.0,
        cash=1000.0,
        gross_exposure=0.0,
        net_exposure=0.0,
        realized_pnl=0.0,
        fees_paid=0.0,
        slippage_paid=0.0,
    )
    base.update(kw)
    return EquitySnapshot(**base)


def _skip(**kw: object) -> SkippedOrder:
    base: dict[str, Any] = dict(
        ts=T, symbol="BTCUSDT", side=Side.BUY, qty=0.0, ref_price=100.0, reason="zero_qty"
    )
    base.update(kw)
    return SkippedOrder(**base)


def test_order_record_ok_and_bad() -> None:
    r = OrderRecord(T, ORDER, OrderStatus.CANCELED, "ttl_expired")
    assert r.reason == "ttl_expired"
    assert OrderRecord(T, ORDER, OrderStatus.NEW).reason == ""
    assert OrderRecord(T, ORDER, "rejected", "below_min_qty").status is OrderStatus.REJECTED  # pyright: ignore[reportArgumentType]
    for status, reason in (
        (OrderStatus.CANCELED, "below_min_qty"),
        (OrderStatus.REJECTED, "ttl_expired"),
        (OrderStatus.REJECTED, ""),
        (OrderStatus.NEW, "user"),
        (OrderStatus.FILLED, "user"),
        (OrderStatus.PARTIAL, ""),
    ):
        with pytest.raises(InvariantError):
            OrderRecord(T, ORDER, status, reason)


def test_naive_ts_raises() -> None:
    naive = T.replace(tzinfo=None)
    with pytest.raises(TimezoneError):
        OrderRecord(naive, ORDER, OrderStatus.NEW)
    with pytest.raises(TimezoneError):
        _snap(ts=naive)
    with pytest.raises(TimezoneError):
        _skip(ts=naive)
    with pytest.raises(TimezoneError):
        LogRecord(naive, "m")


def test_ts_normalised_to_utc() -> None:
    assert OrderRecord(LONDON, ORDER, OrderStatus.NEW).ts.tzinfo is UTC
    assert _snap(ts=LONDON).ts.tzinfo is UTC
    assert _skip(ts=LONDON).ts.tzinfo is UTC
    assert LogRecord(LONDON, "m").ts.tzinfo is UTC


def test_skipped_order() -> None:
    assert _skip().qty == pytest.approx(0.0, abs=1e-12)
    for kw in (
        {"qty": -0.001},
        {"qty": float("nan")},
        {"ref_price": 0.0},
        {"ref_price": float("inf")},
        {"reason": "nope"},
        {"side": "hold"},
    ):
        with pytest.raises(InvariantError):
            _skip(**kw)


def test_equity_snapshot() -> None:
    s = _snap()
    assert s.funding_paid == pytest.approx(0.0, abs=1e-12)
    assert s.tax_paid == pytest.approx(0.0, abs=1e-12)
    assert _snap(fees_paid=-1.0).fees_paid == pytest.approx(-1.0, abs=1e-12)
    _snap(gross_exposure=100.0, net_exposure=-100.0 - 1e-8)  # within tolerance
    for kw in (
        {"gross_exposure": 100.0, "net_exposure": -150.0},
        {"equity": float("nan")},
        {"gross_exposure": -1.0},
        {"slippage_paid": -0.1},
        {"tax_paid": -0.1},
        {"cash": float("inf")},
    ):
        with pytest.raises(InvariantError):
            _snap(**kw)


def test_log_record() -> None:
    r = LogRecord(T, "signal", (("a", 1), ("b", "x")))
    assert hash(r) == hash(LogRecord(T, "signal", (("a", 1), ("b", "x"))))
    assert LogRecord(T, "m").fields == ()
    bad: list[Any] = [
        (T, "signal", (("a", [1]),)),
        (T, "signal", (("a", object()),)),
        (T, "signal", (("b", 1), ("a", 2))),
        (T, "signal", (("a", 1), ("a", 2))),
        (T, "signal", ((1, 1),)),
        (T, "", ()),
    ]
    for args in bad:
        with pytest.raises(InvariantError):
            LogRecord(*args)


def test_event_log_order_and_snapshots() -> None:
    log = EventLog()
    o1 = OrderRecord(T, ORDER, OrderStatus.NEW)
    o2 = OrderRecord(T, ORDER, OrderStatus.FILLED)
    fill = Fill("o1", "BTCUSDT", T, Side.BUY, 1.0, 100.0, 0.1, 0.0, False)
    sk = _skip()
    e1, e2 = _snap(), _snap(equity=1001.0)
    msg = LogRecord(T, "hi")
    log.record_order(o1)
    log.record_order(o2)
    log.record_fill(fill)
    log.record_skipped(sk)
    log.record_equity(e1)
    log.record_equity(e2)
    log.record_message(msg)
    before = log.orders()
    assert before == (o1, o2)
    assert log.fills() == (fill,)
    assert log.skipped() == (sk,)
    assert log.equity() == (e1, e2)
    assert log.messages() == (msg,)
    log.record_order(o1)
    assert len(before) == 2
    assert len(log.orders()) == 3


def test_event_logs_do_not_share_state() -> None:
    a, b = EventLog(), EventLog()
    a.record_message(LogRecord(T, "x"))
    assert b.messages() == ()


def test_pickle_roundtrip() -> None:
    items = [
        OrderRecord(T, ORDER, OrderStatus.REJECTED, "below_min_qty"),
        _skip(),
        _snap(),
        LogRecord(T, "m", (("a", 1.5), ("b", None))),
    ]
    for item in items:
        assert pickle.loads(pickle.dumps(item, protocol=5)) == item
