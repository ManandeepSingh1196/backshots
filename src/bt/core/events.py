"""Run event records and the append-only EventLog."""

from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from typing import Final, Literal, get_args

from bt.core.clock import ensure_utc
from bt.core.types import (
    Fill,
    Order,
    OrderStatus,
    Side,
    _coerce,  # pyright: ignore[reportPrivateUsage]
    _finite,  # pyright: ignore[reportPrivateUsage]
    _require,  # pyright: ignore[reportPrivateUsage]
)

RejectReason = Literal[
    "unknown_symbol",
    "duplicate_id",
    "unsupported_kind",
    "no_reference_price",
    "below_min_qty",
    "qty_off_step",
    "below_min_notional",
    "price_off_tick",
    "reduce_only_increases",
    "insufficient_holdings",
    "insufficient_cash",
    "insufficient_holdings_at_fill",
    "insufficient_cash_at_fill",
]
CancelReason = Literal["ttl_expired", "end_of_data", "kill_switch", "user"]
SkipReason = Literal["below_band", "zero_qty", "below_min_qty", "below_min_notional"]

LogValue = str | int | float | bool | None

_EXPOSURE_REL_TOL: Final = 1e-9


@dataclass(frozen=True, slots=True)
class OrderRecord:
    """One status change of one order.

    Invariants: ts is UTC; status is NEW, FILLED, CANCELED or REJECTED (never PARTIAL);
    reason is "" for NEW/FILLED, a RejectReason for REJECTED, a CancelReason for CANCELED.
    """

    ts: datetime
    order: Order
    status: OrderStatus
    reason: str = ""

    def __post_init__(self) -> None:
        owner = f"OrderRecord {self.order.id}"
        object.__setattr__(self, "ts", ensure_utc(self.ts, "OrderRecord.ts"))
        status = _coerce(OrderStatus, self.status, owner, "status", "order status")
        object.__setattr__(self, "status", status)
        _require(status != OrderStatus.PARTIAL, f"{owner}: partial fills are not supported")
        if status == OrderStatus.REJECTED:
            _require(
                self.reason in get_args(RejectReason),
                f"{owner}: bad reject reason {self.reason!r}",
            )
        elif status == OrderStatus.CANCELED:
            _require(
                self.reason in get_args(CancelReason),
                f"{owner}: bad cancel reason {self.reason!r}",
            )
        else:
            _require(self.reason == "", f"{owner}: reason must be empty for {status.value}")


@dataclass(frozen=True, slots=True)
class SkippedOrder:
    """An order the rebalancer chose not to create; it never reaches the broker.

    Invariants: ts is UTC; qty finite and >= 0; ref_price finite and > 0; reason is a SkipReason.
    """

    ts: datetime
    symbol: str
    side: Side
    qty: float
    ref_price: float
    reason: SkipReason

    def __post_init__(self) -> None:
        owner = f"SkippedOrder {self.symbol}"
        object.__setattr__(self, "ts", ensure_utc(self.ts, "SkippedOrder.ts"))
        object.__setattr__(self, "side", _coerce(Side, self.side, owner, "side", "side"))
        _finite("SkippedOrder", qty=self.qty, ref_price=self.ref_price)
        _require(self.qty >= 0, f"{owner}: qty must be >= 0")
        _require(self.ref_price > 0, f"{owner}: ref_price must be > 0")
        _require(self.reason in get_args(SkipReason), f"{owner}: bad reason {self.reason!r}")


@dataclass(frozen=True, slots=True)
class EquitySnapshot:
    """One per step, after marking at the bar's close.

    Invariants: ts is UTC; all floats finite; gross_exposure, slippage_paid and tax_paid
    >= 0 (fees_paid may be negative for rebates); gross_exposure >= |net_exposure| within
    a relative tolerance.
    """

    ts: datetime
    equity: float
    cash: float
    gross_exposure: float
    net_exposure: float
    realized_pnl: float
    fees_paid: float
    slippage_paid: float
    funding_paid: float = 0.0
    tax_paid: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "ts", ensure_utc(self.ts, "EquitySnapshot.ts"))
        _finite(
            "EquitySnapshot",
            equity=self.equity,
            cash=self.cash,
            gross_exposure=self.gross_exposure,
            net_exposure=self.net_exposure,
            realized_pnl=self.realized_pnl,
            fees_paid=self.fees_paid,
            slippage_paid=self.slippage_paid,
            funding_paid=self.funding_paid,
            tax_paid=self.tax_paid,
        )
        _require(self.gross_exposure >= 0, "EquitySnapshot: gross_exposure must be >= 0")
        _require(self.slippage_paid >= 0, "EquitySnapshot: slippage_paid must be >= 0")
        _require(self.tax_paid >= 0, "EquitySnapshot: tax_paid must be >= 0")
        _require(
            self.gross_exposure
            >= abs(self.net_exposure) - _EXPOSURE_REL_TOL * max(1.0, self.gross_exposure),
            "EquitySnapshot: gross_exposure must be >= |net_exposure|",
        )


@dataclass(frozen=True, slots=True)
class LogRecord:
    """Context.log output.

    Invariants: ts is UTC; message non-empty; fields is a tuple of (str, LogValue) pairs
    with unique keys sorted ascending. Hashable whenever its values are.
    """

    ts: datetime
    message: str
    fields: tuple[tuple[str, LogValue], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "ts", ensure_utc(self.ts, "LogRecord.ts"))
        _require(bool(self.message), "LogRecord.message must be non-empty")
        pairs = tuple((k, v) for k, v in self.fields)
        object.__setattr__(self, "fields", pairs)
        for key, value in pairs:
            _require(isinstance(key, str), f"LogRecord: field key {key!r} must be a str")  # pyright: ignore[reportUnnecessaryIsInstance]
            _require(
                value is None or isinstance(value, str | int | float | bool),  # pyright: ignore[reportUnnecessaryIsInstance]
                f"LogRecord: field {key!r} has unsupported value type {type(value).__name__}",
            )
        keys = [k for k, _ in pairs]
        _require(
            all(a < b for a, b in pairwise(keys)),
            "LogRecord: field keys must be unique and sorted",
        )


class EventLog:
    """Append-only in-memory record of one run (append order is event order; nothing is removed).

    Invariants: each record_* appends exactly one item; each getter returns a fresh tuple
    in append order; instances share no state.
    """

    def __init__(self) -> None:
        self._orders: list[OrderRecord] = []
        self._fills: list[Fill] = []
        self._skipped: list[SkippedOrder] = []
        self._equity: list[EquitySnapshot] = []
        self._messages: list[LogRecord] = []

    def record_order(self, record: OrderRecord) -> None:
        self._orders.append(record)

    def record_fill(self, fill: Fill) -> None:
        self._fills.append(fill)

    def record_skipped(self, skipped: SkippedOrder) -> None:
        self._skipped.append(skipped)

    def record_equity(self, snapshot: EquitySnapshot) -> None:
        self._equity.append(snapshot)

    def record_message(self, record: LogRecord) -> None:
        self._messages.append(record)

    def orders(self) -> tuple[OrderRecord, ...]:
        return tuple(self._orders)

    def fills(self) -> tuple[Fill, ...]:
        return tuple(self._fills)

    def skipped(self) -> tuple[SkippedOrder, ...]:
        return tuple(self._skipped)

    def equity(self) -> tuple[EquitySnapshot, ...]:
        return tuple(self._equity)

    def messages(self) -> tuple[LogRecord, ...]:
        return tuple(self._messages)
