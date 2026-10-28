import dataclasses
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from bt.core.errors import (
    ArchiveError,
    BtError,
    ChecksumMismatchError,
    DataError,
    DataValidationError,
    InvariantError,
    TimezoneError,
)
from bt.core.types import (
    Bar,
    Fill,
    FundingEvent,
    Instrument,
    InstrumentKind,
    Order,
    OrderStatus,
    OrderType,
    Position,
    PositionView,
    Side,
    Step,
)

T = datetime(2024, 1, 1, tzinfo=UTC)
NAIVE = T.replace(tzinfo=None)
NAN = float("nan")
INF = float("inf")


def make_bar(**kw: object) -> Bar:
    args: dict[str, object] = {
        "symbol": "BTCUSDT",
        "ts": T,
        "open": 100,
        "high": 110,
        "low": 90,
        "close": 105,
        "volume": 1.5,
        "quote_volume": 150.0,
    }
    args.update(kw)
    return Bar(**args)  # type: ignore[arg-type]


def make_instrument(**kw: object) -> Instrument:
    args: dict[str, object] = {
        "symbol": "BTCUSDT-PERP",
        "venue_symbol": "BTCUSDT",
        "kind": InstrumentKind.PERP,
        "base": "BTC",
        "quote": "USDT",
        "min_qty": 0.001,
        "qty_step": 0.001,
        "min_notional": 5.0,
        "price_tick": 0.1,
        "maker_fee": 0.0002,
        "taker_fee": 0.0005,
    }
    args.update(kw)
    return Instrument(**args)  # type: ignore[arg-type]


def make_order(**kw: object) -> Order:
    args: dict[str, object] = {"id": "o1", "symbol": "BTCUSDT", "side": Side.BUY, "qty": 0.5}
    args.update(kw)
    return Order(**args)  # type: ignore[arg-type]


def make_fill(**kw: object) -> Fill:
    args: dict[str, object] = {
        "order_id": "o1",
        "symbol": "BTCUSDT",
        "ts": T,
        "side": Side.BUY,
        "qty": 0.1,
        "price": 42000.0,
        "fee": 0.84,
        "slippage_cost": 0.21,
        "is_maker": False,
    }
    args.update(kw)
    return Fill(**args)  # type: ignore[arg-type]


# ---- Bar ----


def test_bar_constructs() -> None:
    bar = Bar("BTCUSDT", T, 100, 110, 90, 105, 1.5, 150.0)
    assert bar.close == pytest.approx(105, abs=1e-12)


def test_bar_flat_is_valid() -> None:
    make_bar(open=100, high=100, low=100, close=100)


def test_bar_boundaries_valid() -> None:
    make_bar(low=100, open=100)  # low == open
    make_bar(low=90, close=90, open=95, high=95)  # low == close, high == open
    make_bar(volume=0, quote_volume=0)


@pytest.mark.parametrize(
    "kw",
    [
        {"high": 104},  # below close
        {"high": 99},  # below open
        {"low": 0},
        {"low": -1},
        {"low": 101},  # above open
        {"low": 106, "high": 110},  # above close and open
        {"volume": -1},
        {"quote_volume": -1},
        {"open": NAN},
        {"high": INF},
        {"low": -INF},
        {"close": NAN},
        {"volume": INF},
        {"quote_volume": NAN},
    ],
)
def test_bar_invalid(kw: dict[str, object]) -> None:
    with pytest.raises(InvariantError):
        make_bar(**kw)


def test_bar_naive_ts_raises_timezone_error() -> None:
    with pytest.raises(TimezoneError):
        make_bar(ts=NAIVE)


# ---- FundingEvent ----


def test_funding_event_ok_and_negative_rate() -> None:
    assert FundingEvent("BTCUSDT", T, -0.0001).rate == pytest.approx(-0.0001, abs=1e-15)


def test_funding_event_rejects_naive_and_nonfinite() -> None:
    with pytest.raises(TimezoneError):
        FundingEvent("BTCUSDT", NAIVE, 0.0001)
    with pytest.raises(InvariantError):
        FundingEvent("BTCUSDT", T, NAN)
    with pytest.raises(InvariantError):
        FundingEvent("BTCUSDT", T, INF)


# ---- Order ----


def test_order_default_is_market() -> None:
    o = Order("o1", "BTCUSDT", Side.BUY, 0.5)
    assert o.type is OrderType.MARKET
    assert o.limit_price is None
    assert o.stop_price is None
    assert o.reduce_only is False
    assert o.tag == ""


@pytest.mark.parametrize(
    "kw",
    [
        {"qty": 0},
        {"qty": -1},
        {"qty": NAN},
        {"qty": INF},
        {"type": OrderType.LIMIT},
        {"type": OrderType.LIMIT, "limit_price": 100, "stop_price": 90},
        {"type": OrderType.LIMIT, "limit_price": 0},
        {"type": OrderType.LIMIT, "limit_price": -5},
        {"type": OrderType.LIMIT, "limit_price": NAN},
        {"type": OrderType.LIMIT, "limit_price": INF},
        {"type": OrderType.STOP},
        {"type": OrderType.STOP, "stop_price": 0},
        {"type": OrderType.STOP, "stop_price": -1},
        {"type": OrderType.STOP, "stop_price": 90, "limit_price": 100},
        {"type": OrderType.STOP, "stop_price": NAN},
        {"type": OrderType.MARKET, "limit_price": 100},
        {"type": OrderType.MARKET, "stop_price": 100},
        {"id": ""},
        {"symbol": ""},
    ],
)
def test_order_invalid(kw: dict[str, object]) -> None:
    with pytest.raises(InvariantError):
        make_order(**kw)


def test_order_valid_limit_and_stop() -> None:
    assert make_order(type=OrderType.LIMIT, limit_price=100.0).limit_price == pytest.approx(
        100.0, abs=1e-12
    )
    assert make_order(type=OrderType.STOP, stop_price=90.0, reduce_only=True).reduce_only


# ---- Instrument ----


def test_perp_instrument_constructs() -> None:
    inst = make_instrument(max_leverage=20, maint_margin_rate=0.004)
    assert inst.max_leverage == pytest.approx(20, abs=1e-12)


def test_spot_defaults_construct() -> None:
    make_instrument(kind=InstrumentKind.SPOT)


@pytest.mark.parametrize(
    "kw",
    [
        {"kind": InstrumentKind.SPOT, "max_leverage": 2.0},
        {"kind": InstrumentKind.SPOT, "maint_margin_rate": 0.01},
        {"qty_step": 0},
        {"qty_step": -0.1},
        {"price_tick": 0},
        {"min_qty": -0.001},
        {"min_notional": -1},
        {"max_leverage": 0.99},
        {"maint_margin_rate": 1.0},
        {"maint_margin_rate": -0.001},
        {"taker_fee": -0.0001},
        {"maker_fee": NAN},
        {"taker_fee": INF},
        {"qty_step": NAN},
        {"price_tick": INF},
        {"min_qty": NAN},
        {"min_notional": INF},
        {"max_leverage": INF},
        {"maint_margin_rate": NAN},
    ],
)
def test_instrument_invalid(kw: dict[str, object]) -> None:
    with pytest.raises(InvariantError):
        make_instrument(**kw)


def test_instrument_boundaries_valid() -> None:
    make_instrument(
        min_qty=0, min_notional=0, max_leverage=1.0, maint_margin_rate=0.0, taker_fee=0.0
    )
    make_instrument(maker_fee=-0.0001)  # rebate
    make_instrument(maint_margin_rate=0.999)


# ---- Fill ----


def test_fill_constructs() -> None:
    f = Fill("o1", "BTCUSDT", T, Side.BUY, 0.1, 42000.0, 0.84, 0.21, False)
    assert f.price == pytest.approx(42000.0, abs=1e-9)


def test_fill_negative_fee_and_zero_slippage_ok() -> None:
    make_fill(fee=-0.5, slippage_cost=0.0)


@pytest.mark.parametrize(
    "kw",
    [
        {"price": 0},
        {"price": -1},
        {"qty": 0},
        {"qty": -0.1},
        {"slippage_cost": -0.01},
        {"fee": NAN},
        {"fee": INF},
        {"price": NAN},
        {"qty": INF},
        {"slippage_cost": NAN},
    ],
)
def test_fill_invalid(kw: dict[str, object]) -> None:
    with pytest.raises(InvariantError):
        make_fill(**kw)


def test_fill_naive_ts() -> None:
    with pytest.raises(TimezoneError):
        make_fill(ts=NAIVE)


# ---- Position / PositionView ----


def test_position_is_mutable() -> None:
    p = Position("BTCUSDT", 1.0, 100.0)
    p.qty = -1.0
    assert p.qty == pytest.approx(-1.0, abs=1e-12)
    assert p.realized_pnl == pytest.approx(0.0, abs=1e-12)
    assert p.posted_margin == pytest.approx(0.0, abs=1e-12)


def test_position_view_frozen_and_finite() -> None:
    v = PositionView("BTCUSDT", 1.0, 100.0, 105.0, 5.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.qty = 2.0  # type: ignore[misc]
    with pytest.raises(InvariantError):
        PositionView("BTCUSDT", 1.0, NAN, 105.0, 5.0)


# ---- Step ----


def test_step_constructs() -> None:
    s = Step(ts=T, bars={"BTCUSDT": make_bar()}, funding=[])
    assert s.ts == T


def test_step_empty_bars_ok() -> None:
    Step(ts=T, bars={}, funding=())


def test_step_rejects_later_bar() -> None:
    with pytest.raises(InvariantError):
        Step(ts=T, bars={"BTCUSDT": make_bar(ts=T + timedelta(days=1))}, funding=[])


def test_step_rejects_earlier_bar() -> None:
    with pytest.raises(InvariantError):
        Step(ts=T, bars={"BTCUSDT": make_bar(ts=T - timedelta(days=1))}, funding=[])


def test_step_rejects_one_microsecond_later_bar() -> None:
    with pytest.raises(InvariantError):
        Step(ts=T, bars={"BTCUSDT": make_bar(ts=T + timedelta(microseconds=1))}, funding=[])


def test_step_rejects_key_symbol_mismatch() -> None:
    with pytest.raises(InvariantError):
        Step(ts=T, bars={"ETHUSDT": make_bar()}, funding=[])


def test_step_funding_boundaries() -> None:
    Step(ts=T, bars={}, funding=[FundingEvent("BTCUSDT", T, 0.0001)])
    Step(ts=T, bars={}, funding=[FundingEvent("BTCUSDT", T - timedelta(hours=8), 0.0001)])
    with pytest.raises(InvariantError):
        Step(
            ts=T, bars={}, funding=[FundingEvent("BTCUSDT", T + timedelta(milliseconds=1), 0.0001)]
        )


def test_step_naive_ts() -> None:
    with pytest.raises(TimezoneError):
        Step(ts=NAIVE, bars={}, funding=[])


def test_step_one_bad_bar_among_good() -> None:
    eth = make_bar(symbol="ETHUSDT", ts=T + timedelta(days=1))
    with pytest.raises(InvariantError):
        Step(ts=T, bars={"BTCUSDT": make_bar(), "ETHUSDT": eth}, funding=[])


# ---- frozen / slots ----

FROZEN: list[Callable[[], object]] = [
    make_bar,
    make_instrument,
    make_order,
    make_fill,
    lambda: FundingEvent("BTCUSDT", T, 0.0),
    lambda: PositionView("BTCUSDT", 1.0, 1.0, 1.0, 0.0),
    lambda: Step(T, {}, []),
]


@pytest.mark.parametrize("factory", FROZEN)
def test_frozen_types_have_no_dict_and_reject_assignment(factory: Callable[[], object]) -> None:
    obj = factory()
    assert not hasattr(obj, "__dict__")
    field = dataclasses.fields(obj)[0].name  # type: ignore[arg-type]
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(obj, field, getattr(obj, field))


def test_bar_close_assignment_raises() -> None:
    bar = make_bar()
    with pytest.raises(dataclasses.FrozenInstanceError):
        bar.close = 1.0  # type: ignore[misc]


def test_position_has_no_dict() -> None:
    assert not hasattr(Position("X", 0.0, 0.0), "__dict__")


# ---- enums / errors ----


def test_enums_are_str() -> None:
    assert Side.BUY == "buy"
    assert str(Side.BUY) == "buy"
    assert str(InstrumentKind.PERP) == "perp"
    assert str(OrderType.STOP) == "stop"
    assert str(OrderStatus.PARTIAL) == "partial"
    assert {s.value for s in OrderStatus} == {"new", "filled", "partial", "canceled", "rejected"}


def test_error_hierarchy() -> None:
    assert issubclass(ChecksumMismatchError, DataError)
    assert issubclass(ArchiveError, DataError)
    assert issubclass(DataValidationError, DataError)
    assert issubclass(DataError, BtError)
    assert issubclass(TimezoneError, BtError)
    assert issubclass(InvariantError, BtError)
    assert issubclass(TimezoneError, ValueError)
    assert issubclass(InvariantError, ValueError)
    assert not issubclass(DataError, ValueError)


def test_no_any_in_core() -> None:
    import ast
    from pathlib import Path

    core = Path(__file__).resolve().parents[2] / "src" / "bt" / "core"
    for path in core.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            assert not (isinstance(node, ast.Name) and node.id == "Any"), path
            assert not (isinstance(node, ast.Attribute) and node.attr == "Any"), path
            if isinstance(node, ast.ImportFrom):
                assert all(a.name != "Any" for a in node.names), path


def test_timestamps_normalised_to_utc_on_all_types() -> None:
    london = datetime(2024, 1, 1, tzinfo=ZoneInfo("Europe/London"))
    assert make_bar(ts=london).ts.tzinfo is UTC
    assert FundingEvent("BTCUSDT", london, 0.0).ts.tzinfo is UTC
    assert make_fill(ts=london).ts.tzinfo is UTC
    assert Step(london, {}, []).ts.tzinfo is UTC
    assert make_bar(ts=datetime(2024, 1, 1, tzinfo=ZoneInfo("UTC"))).ts.tzinfo is UTC


def test_london_summer_rejected_by_types() -> None:
    summer = datetime(2024, 7, 1, tzinfo=ZoneInfo("Europe/London"))
    with pytest.raises(TimezoneError):
        make_bar(ts=summer)


def test_step_snapshots_inputs() -> None:
    d = {"BTCUSDT": make_bar()}
    f: list[FundingEvent] = []
    step = Step(T, d, f)
    d["ETHUSDT"] = make_bar(symbol="ETHUSDT", ts=T + timedelta(days=1))
    f.append(FundingEvent("BTCUSDT", T + timedelta(milliseconds=1), 0.0001))
    assert set(step.bars) == {"BTCUSDT"}
    assert len(step.funding) == 0
    with pytest.raises(TypeError):
        step.bars["X"] = make_bar()  # type: ignore[index]


def test_order_unknown_type_rejected() -> None:
    with pytest.raises(InvariantError, match="unknown order type"):
        make_order(type="bogus")
