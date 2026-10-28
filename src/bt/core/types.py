"""Shared value types. Frozen types validate their invariants at construction."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType

from bt.core.clock import ensure_utc
from bt.core.errors import InvariantError


class InstrumentKind(StrEnum):
    """Spot or perpetual. Members compare equal to their lowercase string values."""

    SPOT = "spot"
    PERP = "perp"


class Side(StrEnum):
    """Order direction. Members compare equal to their lowercase string values."""

    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    """Order type. Members compare equal to their lowercase string values."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderStatus(StrEnum):
    """Order lifecycle state. Members compare equal to their lowercase string values."""

    NEW = "new"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELED = "canceled"
    REJECTED = "rejected"


def _finite(owner: str, **values: float | None) -> None:
    for name, value in values.items():
        if value is not None and not math.isfinite(value):
            raise InvariantError(f"{owner}.{name} must be finite, got {value!r}")


def _coerce[E: StrEnum](enum_cls: type[E], value: object, owner: str, field: str, label: str) -> E:
    """Return the exact (case-sensitive) enum member for value, or raise InvariantError."""
    try:
        return enum_cls(value)
    except ValueError as exc:
        raise InvariantError(f"{owner}.{field}: unknown {label} {value!r}") from exc


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise InvariantError(message)


@dataclass(frozen=True, slots=True)
class Instrument:
    """A tradable instrument.

    Invariants: all floats finite; qty_step > 0; price_tick > 0; min_qty >= 0;
    min_notional >= 0; taker_fee >= 0 (maker_fee may be negative for rebates);
    max_leverage >= 1; 0 <= maint_margin_rate < 1; SPOT implies
    max_leverage == 1 and maint_margin_rate == 0.
    """

    symbol: str
    venue_symbol: str
    kind: InstrumentKind
    base: str
    quote: str
    min_qty: float
    qty_step: float
    min_notional: float
    price_tick: float
    maker_fee: float
    taker_fee: float
    max_leverage: float = 1.0
    maint_margin_rate: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            _coerce(
                InstrumentKind, self.kind, f"Instrument {self.symbol}", "kind", "instrument kind"
            ),
        )
        _finite(
            "Instrument",
            min_qty=self.min_qty,
            qty_step=self.qty_step,
            min_notional=self.min_notional,
            price_tick=self.price_tick,
            maker_fee=self.maker_fee,
            taker_fee=self.taker_fee,
            max_leverage=self.max_leverage,
            maint_margin_rate=self.maint_margin_rate,
        )
        _require(self.qty_step > 0, f"Instrument {self.symbol}: qty_step must be > 0")
        _require(self.price_tick > 0, f"Instrument {self.symbol}: price_tick must be > 0")
        _require(self.min_qty >= 0, f"Instrument {self.symbol}: min_qty must be >= 0")
        _require(self.min_notional >= 0, f"Instrument {self.symbol}: min_notional must be >= 0")
        _require(self.taker_fee >= 0, f"Instrument {self.symbol}: taker_fee must be >= 0")
        _require(self.max_leverage >= 1.0, f"Instrument {self.symbol}: max_leverage must be >= 1")
        _require(
            0 <= self.maint_margin_rate < 1,
            f"Instrument {self.symbol}: maint_margin_rate must be in [0, 1)",
        )
        if self.kind == InstrumentKind.SPOT:
            _require(
                self.max_leverage == 1.0, f"Instrument {self.symbol}: spot max_leverage must be 1"
            )
            _require(
                self.maint_margin_rate == 0.0,
                f"Instrument {self.symbol}: spot maint_margin_rate must be 0",
            )


@dataclass(frozen=True, slots=True)
class Bar:
    """One closed OHLCV bar stamped with its open time.

    Invariants: ts is UTC; all floats finite; 0 < low <= min(open, close);
    high >= max(open, close); volume >= 0; quote_volume >= 0.
    """

    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "ts", ensure_utc(self.ts, "Bar.ts"))
        _finite(
            "Bar",
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            quote_volume=self.quote_volume,
        )
        ctx = f"Bar {self.symbol} @ {self.ts.isoformat()}"
        _require(self.low > 0, f"{ctx}: low must be > 0")
        _require(self.low <= min(self.open, self.close), f"{ctx}: low must be <= open and close")
        _require(self.high >= max(self.open, self.close), f"{ctx}: high must be >= open and close")
        _require(self.volume >= 0, f"{ctx}: volume must be >= 0")
        _require(self.quote_volume >= 0, f"{ctx}: quote_volume must be >= 0")


@dataclass(frozen=True, slots=True)
class FundingEvent:
    """A funding payment. Invariants: ts is UTC; rate finite. rate > 0 means longs pay shorts."""

    symbol: str
    ts: datetime
    rate: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "ts", ensure_utc(self.ts, "FundingEvent.ts"))
        _finite("FundingEvent", rate=self.rate)


@dataclass(frozen=True, slots=True)
class Order:
    """An order request.

    Invariants: id and symbol non-empty; qty > 0 and finite; MARKET carries no
    prices; LIMIT carries limit_price > 0 and no stop_price; STOP carries
    stop_price > 0 and no limit_price.
    """

    id: str
    symbol: str
    side: Side
    qty: float
    type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    reduce_only: bool = False
    tag: str = ""

    def __post_init__(self) -> None:
        owner = f"Order {self.id}"
        object.__setattr__(self, "side", _coerce(Side, self.side, owner, "side", "side"))
        object.__setattr__(self, "type", _coerce(OrderType, self.type, owner, "type", "order type"))
        _finite("Order", qty=self.qty, limit_price=self.limit_price, stop_price=self.stop_price)
        _require(bool(self.id), "Order.id must be non-empty")
        _require(bool(self.symbol), "Order.symbol must be non-empty")
        _require(self.qty > 0, f"Order {self.id}: qty must be > 0")
        if self.type == OrderType.MARKET:
            _require(
                self.limit_price is None and self.stop_price is None,
                f"Order {self.id}: market order must not carry limit_price or stop_price",
            )
        elif self.type == OrderType.LIMIT:
            _require(
                self.limit_price is not None and self.limit_price > 0 and self.stop_price is None,
                f"Order {self.id}: limit order needs limit_price > 0 and no stop_price",
            )
        elif self.type == OrderType.STOP:
            _require(
                self.stop_price is not None and self.stop_price > 0 and self.limit_price is None,
                f"Order {self.id}: stop order needs stop_price > 0 and no limit_price",
            )
        else:
            raise InvariantError(f"Order {self.id}: unknown order type {self.type!r}")


@dataclass(frozen=True, slots=True)
class Fill:
    """An execution.

    Invariants: ts is UTC; all floats finite; qty > 0; price > 0; slippage_cost >= 0.
    """

    order_id: str
    symbol: str
    ts: datetime
    side: Side
    qty: float
    price: float
    fee: float
    slippage_cost: float
    is_maker: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "ts", ensure_utc(self.ts, "Fill.ts"))
        object.__setattr__(
            self, "side", _coerce(Side, self.side, f"Fill {self.order_id}", "side", "side")
        )
        _finite(
            "Fill", qty=self.qty, price=self.price, fee=self.fee, slippage_cost=self.slippage_cost
        )
        _require(self.qty > 0, f"Fill {self.order_id}: qty must be > 0")
        _require(self.price > 0, f"Fill {self.order_id}: price must be > 0")
        _require(self.slippage_cost >= 0, f"Fill {self.order_id}: slippage_cost must be >= 0")


@dataclass(slots=True)
class Position:
    """Mutable position, owned and mutated only by Portfolio. qty is signed (+ long, - short).

    No construction-time checks: Portfolio is responsible for its own consistency.
    """

    symbol: str
    qty: float
    avg_entry: float
    realized_pnl: float = 0.0
    posted_margin: float = 0.0


@dataclass(frozen=True, slots=True)
class PositionView:
    """Read-only position snapshot handed to strategies. Invariant: all floats finite."""

    symbol: str
    qty: float
    avg_entry: float
    market_value: float
    unrealized_pnl: float

    def __post_init__(self) -> None:
        _finite(
            "PositionView",
            qty=self.qty,
            avg_entry=self.avg_entry,
            market_value=self.market_value,
            unrealized_pnl=self.unrealized_pnl,
        )


@dataclass(frozen=True, slots=True)
class Step:
    """Everything the feed delivers for one timestamp.

    Inputs are snapshotted (bars becomes a read-only mapping, funding a tuple) so later
    mutation of the caller's containers cannot change the step.

    Invariants: ts is UTC; every bars key equals its bar's symbol and every bar
    is stamped exactly ts; every funding event has ts <= Step.ts. A step can
    therefore never carry a later bar (no look-ahead).
    """

    ts: datetime
    bars: Mapping[str, Bar]
    funding: Sequence[FundingEvent]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ts", ensure_utc(self.ts, "Step.ts"))
        object.__setattr__(self, "bars", MappingProxyType(dict(self.bars)))
        object.__setattr__(self, "funding", tuple(self.funding))
        for key, bar in self.bars.items():
            _require(
                key == bar.symbol,
                f"Step {self.ts.isoformat()}: key {key!r} != bar.symbol {bar.symbol!r}",
            )
            _require(
                bar.ts == self.ts,
                f"Step {self.ts.isoformat()}: bar {key} has ts {bar.ts.isoformat()} (look-ahead)",
            )
        for ev in self.funding:
            _require(
                ev.ts <= self.ts,
                f"Step {self.ts.isoformat()}: funding {ev.symbol} at {ev.ts.isoformat()} is later",
            )
