"""Spot portfolio: cash, positions, fees, mark-to-market and a read-only view."""

import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from types import MappingProxyType

from bt.core.clock import ensure_utc
from bt.core.errors import PortfolioError
from bt.core.events import EquitySnapshot
from bt.core.types import (
    Fill,
    Instrument,
    InstrumentKind,
    PortfolioView,
    Position,
    PositionView,
    Side,
)

# Private copies of the bt.execution.rules tolerances (portfolio may import only bt.core).
# T019 asserts they stay equal.
_QTY_REL_TOL = 1e-9
_CASH_ABS_TOL = 1e-6


class Portfolio:
    """Spot-only portfolio.

    Invariants: no shorting (every qty >= 0); cash >= -1e-6 after any accepted fill;
    equity == cash + sum(qty * last mark) summed in ascending symbol order;
    equity - initial_cash == realized + unrealized - fees_paid (fees are never part of
    avg_entry); apply_fill and mark are atomic (a PortfolioError leaves state unchanged);
    timestamps never go backwards; no state is shared between instances.
    """

    def __init__(self, initial_cash: float, instruments: Sequence[Instrument]) -> None:
        if not math.isfinite(initial_cash) or initial_cash <= 0:
            raise PortfolioError(f"initial_cash must be finite and > 0, got {initial_cash!r}")
        if not instruments:
            raise PortfolioError("instruments must be non-empty")
        symbols = [i.symbol for i in instruments]
        if len(set(symbols)) != len(symbols):
            raise PortfolioError(f"duplicate instrument symbols: {sorted(symbols)}")
        for inst in instruments:
            if inst.kind != InstrumentKind.SPOT:
                raise PortfolioError(f"{inst.symbol}: only spot is supported; perps arrive in M6")
        self._initial_cash = initial_cash
        self._steps: dict[str, float] = {i.symbol: i.qty_step for i in instruments}
        self._order: tuple[str, ...] = tuple(sorted(symbols))
        self._pos: dict[str, Position] = {s: Position(s, 0.0, 0.0) for s in self._order}
        self._marks: dict[str, float] = {}
        self._cash = initial_cash
        self._ts: datetime | None = None
        self._last_event_ts: datetime | None = None  # latest accepted fill or mark
        self._fees_paid = 0.0
        self._slippage_paid = 0.0
        self._realized = 0.0

    @property
    def ts(self) -> datetime | None:
        """Last mark time; None before the first mark."""
        return self._ts

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def available_cash(self) -> float:
        """Equals cash in M2 (spot, no margin)."""
        return self._cash

    @property
    def equity(self) -> float:
        total = self._cash
        for s in self._order:
            q = self._pos[s].qty
            if q != 0.0:
                total += q * self._marks[s]
        return total

    def position(self, symbol: str) -> PositionView:
        pos = self._pos.get(symbol)
        if pos is None:
            raise PortfolioError(f"unknown symbol {symbol!r}")
        if pos.qty == 0.0:
            return PositionView(symbol, 0.0, 0.0, 0.0, 0.0)
        mark = self._marks[symbol]
        return PositionView(
            symbol, pos.qty, pos.avg_entry, pos.qty * mark, pos.qty * (mark - pos.avg_entry)
        )

    def positions(self) -> Mapping[str, PositionView]:
        out = {s: self.position(s) for s in self._order if self._pos[s].qty != 0.0}
        return MappingProxyType(out)

    def apply_fill(self, fill: Fill) -> None:
        """Apply a fill atomically; raises PortfolioError and changes nothing on violation."""
        pos = self._pos.get(fill.symbol)
        if pos is None:
            raise PortfolioError(f"fill {fill.order_id}: unknown symbol {fill.symbol!r}")
        if self._last_event_ts is not None and fill.ts < self._last_event_ts:
            raise PortfolioError(
                f"fill {fill.order_id}: ts {fill.ts.isoformat()} before last event "
                f"{self._last_event_ts.isoformat()}"
            )
        step = self._steps[fill.symbol]
        notional = fill.qty * fill.price
        if fill.side == Side.BUY:
            new_cash = self._cash - (notional + fill.fee)
            if new_cash < -_CASH_ABS_TOL:
                raise PortfolioError(
                    f"fill {fill.order_id}: buy would leave cash {new_cash!r} (overspend)"
                )
            new_qty = pos.qty + fill.qty
            new_avg = (pos.qty * pos.avg_entry + notional) / new_qty
            new_realized = self._realized
            trade_pnl = 0.0
        else:
            new_qty = pos.qty - fill.qty
            tol = _QTY_REL_TOL * max(step, abs(pos.qty)) + math.ulp(pos.qty)
            if new_qty < -tol:
                raise PortfolioError(
                    f"fill {fill.order_id}: sell {fill.qty!r} exceeds holding {pos.qty!r}"
                )
            new_cash = self._cash + notional - fill.fee
            new_avg = pos.avg_entry
            trade_pnl = (fill.price - pos.avg_entry) * fill.qty
            new_realized = self._realized + trade_pnl
            if abs(new_qty) <= tol:
                new_qty = 0.0
                new_avg = 0.0
        # All checks passed: mutate.
        self._cash = new_cash
        pos.qty = new_qty
        pos.avg_entry = new_avg
        self._realized = new_realized
        pos.realized_pnl += trade_pnl
        self._last_event_ts = fill.ts
        self._fees_paid += fill.fee
        self._slippage_paid += fill.slippage_cost
        self._marks.setdefault(fill.symbol, fill.price)

    def mark(self, ts: datetime, prices: Mapping[str, float]) -> None:
        """Mark to market; symbols missing from prices keep their previous mark."""
        ts = ensure_utc(ts, "mark ts")
        if self._last_event_ts is not None and ts < self._last_event_ts:
            raise PortfolioError(
                f"mark ts {ts.isoformat()} before last event {self._last_event_ts.isoformat()}"
            )
        for sym, price in prices.items():
            if sym not in self._pos:
                raise PortfolioError(f"mark: unknown symbol {sym!r}")
            if not math.isfinite(price) or price <= 0:
                raise PortfolioError(f"mark: bad price for {sym}: {price!r}")
        self._marks.update({sym: float(price) for sym, price in prices.items()})
        self._ts = ts
        self._last_event_ts = ts

    def snapshot(self) -> EquitySnapshot:
        """Equity snapshot at the last mark; raises PortfolioError before the first mark."""
        if self._ts is None:
            raise PortfolioError("snapshot requires a prior mark")
        values = [self._pos[s].qty * self._marks[s] for s in self._order if self._pos[s].qty != 0.0]
        return EquitySnapshot(
            ts=self._ts,
            equity=self.equity,
            cash=self._cash,
            gross_exposure=sum(abs(v) for v in values),
            net_exposure=sum(values),
            realized_pnl=self._realized,
            fees_paid=self._fees_paid,
            slippage_paid=self._slippage_paid,
        )

    def view(self) -> PortfolioView:
        """Read-only live proxy without mutators."""
        return _PortfolioViewProxy(self)


class _PortfolioViewProxy:
    """Read-only live window onto a Portfolio; exposes exactly the PortfolioView members."""

    __slots__ = ("_p",)  # private reference to the live Portfolio; strategies must not use it

    def __init__(self, portfolio: Portfolio) -> None:
        self._p = portfolio

    @property
    def ts(self) -> datetime | None:
        return self._p.ts

    @property
    def cash(self) -> float:
        return self._p.cash

    @property
    def available_cash(self) -> float:
        return self._p.available_cash

    @property
    def equity(self) -> float:
        return self._p.equity

    def position(self, symbol: str) -> PositionView:
        return self._p.position(symbol)

    def positions(self) -> Mapping[str, PositionView]:
        return self._p.positions()
