"""Simulated broker: fills queued orders against the next bar, pessimistically."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from bt.core.clock import ensure_utc
from bt.core.errors import ExecutionError
from bt.core.events import CancelReason, EventLog, OrderRecord, RejectReason
from bt.core.types import (
    Bar,
    Fill,
    Instrument,
    Order,
    OrderAck,
    OrderStatus,
    OrderType,
    PortfolioView,
    Side,
    Step,
)
from bt.execution.fees import FeeModel
from bt.execution.rules import CASH_ABS_TOL, QTY_REL_TOL
from bt.execution.slippage import SlippageModel


@dataclass(slots=True)
class _Open:
    order: Order
    age: int = 0


@dataclass(frozen=True, slots=True)
class _Exec:
    price: float
    base: float
    is_maker: bool


class BrokerSim:
    """Queues orders decided on bar t and fills them against bar t+1 (DESIGN section 5).

    Invariants: open orders keep submission order; an order id is accepted at most once
    ever; process() timestamps strictly increase; each order gets exactly ttl_bars
    fill chances; sells are walked before buys within a step and fills are checked
    against running cash and holdings budgets; no partial fills; deterministic.
    Fills are returned, not recorded: the engine records them (the log gets order records).
    """

    def __init__(
        self,
        instruments: Sequence[Instrument],
        fees: FeeModel,
        slippage: SlippageModel,
        portfolio: PortfolioView,
        log: EventLog,
        ttl_bars: int = 3,
    ) -> None:
        if not instruments:
            raise ExecutionError("BrokerSim: instruments must be non-empty")
        self._inst: dict[str, Instrument] = {}
        for inst in instruments:
            if inst.symbol in self._inst:
                raise ExecutionError(f"BrokerSim: duplicate instrument {inst.symbol}")
            self._inst[inst.symbol] = inst
        if isinstance(ttl_bars, bool) or not isinstance(ttl_bars, int) or ttl_bars < 1:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ExecutionError(f"BrokerSim: ttl_bars must be an int >= 1, got {ttl_bars!r}")
        self._fees = fees
        self._slippage = slippage
        self._portfolio = portfolio
        self._log = log
        self._ttl = ttl_bars
        self._open: list[_Open] = []
        self._seen: set[str] = set()
        self._last_ts: datetime | None = None
        self._last_bar: dict[str, Bar] = {}

    # -- public API ---------------------------------------------------------------

    def submit(self, orders: Sequence[Order]) -> list[OrderAck]:
        ts = self._last_ts
        if ts is None:
            raise ExecutionError("submit before first process")
        acks: list[OrderAck] = []
        for order in orders:
            if order.symbol not in self._inst:
                ack = OrderAck(order.id, OrderStatus.REJECTED, "unknown_symbol")
            elif order.id in self._seen:
                ack = OrderAck(order.id, OrderStatus.REJECTED, "duplicate_id")
            else:
                ack = OrderAck(order.id, OrderStatus.NEW)
                self._open.append(_Open(order))
            self._seen.add(order.id)
            self._log.record_order(OrderRecord(ts, order, ack.status, ack.reason))
            acks.append(ack)
        return acks

    def process(self, step: Step) -> list[Fill]:
        if self._last_ts is not None and step.ts <= self._last_ts:
            raise ExecutionError(f"process: step {step.ts} is not after {self._last_ts}")
        self._last_ts = step.ts
        for sym, bar in step.bars.items():
            self._last_bar[sym] = bar
        cash = self._portfolio.available_cash
        held = {s: self._portfolio.position(s).qty for s in self._inst}
        fills: list[Fill] = []
        done: set[int] = set()
        for entry in self._walk_order():
            bar = step.bars.get(entry.order.symbol)
            if bar is None:
                entry.age += 1
                continue
            ex = self._try_fill(entry.order, bar)
            if ex is None:
                entry.age += 1
                continue
            done.add(id(entry))
            fill, cash = self._settle(entry.order, ex, step.ts, cash, held)
            if fill is not None:
                fills.append(fill)
        self._open = [e for e in self._open if id(e) not in done]
        expired = [e for e in self._open if e.age >= self._ttl]
        for e in expired:
            self._log.record_order(
                OrderRecord(step.ts, e.order, OrderStatus.CANCELED, "ttl_expired")
            )
        self._open = [e for e in self._open if e.age < self._ttl]
        return fills

    def cancel(self, order_id: str) -> OrderAck:
        for i, e in enumerate(self._open):
            if e.order.id == order_id:
                del self._open[i]
                return self._cancelled(e.order, "user")
        raise ExecutionError(f"cancel: order {order_id!r} is not open")

    def open_orders(self) -> list[Order]:
        return [e.order for e in self._open]

    def cancel_all(self, reason: CancelReason) -> list[OrderAck]:
        entries, self._open = self._open, []
        return [self._cancelled(e.order, reason) for e in entries]

    def settle_at_close(self, ts: datetime, orders: Sequence[Order]) -> list[Fill]:
        ts = ensure_utc(ts, "settle_at_close.ts")
        if self._last_ts is None or ts != self._last_ts:
            raise ExecutionError(f"settle_at_close: ts {ts} != last processed ts {self._last_ts}")
        ids: set[str] = set()
        for o in orders:
            if o.type != OrderType.MARKET:
                raise ExecutionError(f"settle_at_close: order {o.id} is not a market order")
            if o.symbol not in self._inst:
                raise ExecutionError(f"settle_at_close: unknown symbol {o.symbol}")
            if o.id in self._seen or o.id in ids:
                raise ExecutionError(f"settle_at_close: duplicate order id {o.id}")
            if o.symbol not in self._last_bar:
                raise ExecutionError(f"settle_at_close: no bar seen for {o.symbol}")
            ids.add(o.id)
        self._seen |= ids
        for o in orders:
            self._log.record_order(OrderRecord(ts, o, OrderStatus.NEW))
        cash = self._portfolio.available_cash
        held = {s: self._portfolio.position(s).qty for s in self._inst}
        fills: list[Fill] = []
        for o in [o for o in orders if o.side == Side.SELL] + [
            o for o in orders if o.side == Side.BUY
        ]:
            bar = self._last_bar[o.symbol]
            s = self._slip(o, bar)
            base = bar.close
            price = base * (1 + s) if o.side == Side.BUY else base * (1 - s)
            fill, cash = self._settle(o, _Exec(price, base, False), ts, cash, held)
            if fill is not None:
                fills.append(fill)
        return fills

    # -- internals ----------------------------------------------------------------

    def _walk_order(self) -> list[_Open]:
        return [e for e in self._open if e.order.side == Side.SELL] + [
            e for e in self._open if e.order.side == Side.BUY
        ]

    def _cancelled(self, order: Order, reason: CancelReason) -> OrderAck:
        assert self._last_ts is not None
        self._log.record_order(OrderRecord(self._last_ts, order, OrderStatus.CANCELED, reason))
        return OrderAck(order.id, OrderStatus.CANCELED, reason)

    def _slip(self, order: Order, bar: Bar) -> float:
        inst = self._inst[order.symbol]
        s = self._slippage.slippage_frac(inst, order.side, order.qty, bar)
        if not math.isfinite(s) or not 0 <= s < 1:
            raise ExecutionError(f"slippage for order {order.id} must be in [0, 1), got {s!r}")
        return s

    def _try_fill(self, order: Order, bar: Bar) -> _Exec | None:
        buy = order.side == Side.BUY
        if order.type == OrderType.MARKET:
            s = self._slip(order, bar)
            base = bar.open
            return _Exec(base * (1 + s) if buy else base * (1 - s), base, False)
        if order.type == OrderType.LIMIT:
            assert order.limit_price is not None
            lim = order.limit_price
            if buy and bar.low < lim:
                return _Exec(min(lim, bar.open), min(lim, bar.open), True)
            if not buy and bar.high > lim:
                return _Exec(max(lim, bar.open), max(lim, bar.open), True)
            return None
        assert order.stop_price is not None
        stop = order.stop_price
        if buy and bar.high >= stop:
            base = max(stop, bar.open)
        elif not buy and bar.low <= stop:
            base = min(stop, bar.open)
        else:
            return None
        s = self._slip(order, bar)
        return _Exec(base * (1 + s) if buy else base * (1 - s), base, False)

    def _settle(
        self, order: Order, ex: _Exec, ts: datetime, cash: float, held: dict[str, float]
    ) -> tuple[Fill | None, float]:
        """Apply fill-time funds checks on the running budgets; log; return (fill, cash)."""
        inst = self._inst[order.symbol]
        qty = order.qty
        fee = self._fees.fee(inst, order.side, qty, ex.price, ex.is_maker)
        if order.side == Side.SELL:
            if qty > held[order.symbol] + QTY_REL_TOL * inst.qty_step:
                self._reject(order, ts, "insufficient_holdings_at_fill")
                return None, cash
            held[order.symbol] -= qty
            cash += qty * ex.price - fee
        else:
            cost = qty * ex.price + fee
            if cost > cash + CASH_ABS_TOL:
                self._reject(order, ts, "insufficient_cash_at_fill")
                return None, cash
            cash -= cost
        self._log.record_order(OrderRecord(ts, order, OrderStatus.FILLED))
        fill = Fill(
            order.id,
            order.symbol,
            ts,
            order.side,
            qty,
            ex.price,
            fee,
            abs(ex.price - ex.base) * qty,
            ex.is_maker,
        )
        return fill, cash

    def _reject(self, order: Order, ts: datetime, reason: RejectReason) -> None:
        self._log.record_order(OrderRecord(ts, order, OrderStatus.REJECTED, reason))
