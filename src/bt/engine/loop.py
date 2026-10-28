"""The engine loop: DESIGN section 5 event order, order validation with reservations,
the max-drawdown kill switch and the close at end of data."""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final, cast

from bt.core.errors import BtError, EngineError, StrategyError
from bt.core.events import (
    EquitySnapshot,
    EventLog,
    LogRecord,
    OrderRecord,
    SkippedOrder,
)
from bt.core.types import (
    Fill,
    Instrument,
    InstrumentKind,
    Order,
    OrderStatus,
    OrderType,
    PortfolioView,
    PositionView,
    Side,
)
from bt.data.feed import DataFeed
from bt.execution.broker import BacktestBroker
from bt.execution.rules import validate_order
from bt.portfolio.portfolio import Portfolio
from bt.strategy.base import NoAction, Orders, Strategy, TargetWeights
from bt.strategy.context import StepContext

_KILL_REL_TOL: Final = 1e-9
_RESERVED_PREFIX: Final = "bt-"


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Engine switches.

    Invariants: max_drawdown_kill is None or in (0, 1]; rebalance_band >= 0;
    global_max_leverage >= 1; every number is finite.
    """

    close_at_end: bool = True
    max_drawdown_kill: float | None = 0.40
    rebalance_band: float = 0.02  # unused until M5
    global_max_leverage: float = 2.0  # unused until M5

    def __post_init__(self) -> None:
        k = self.max_drawdown_kill
        if k is not None and (isinstance(k, bool) or not (math.isfinite(k) and 0 < k <= 1)):
            raise EngineError(f"max_drawdown_kill must be None or in (0, 1], got {k!r}")
        if not math.isfinite(self.rebalance_band) or self.rebalance_band < 0:
            raise EngineError(
                f"rebalance_band must be finite and >= 0, got {self.rebalance_band!r}"
            )
        if not math.isfinite(self.global_max_leverage) or self.global_max_leverage < 1:
            raise EngineError(
                f"global_max_leverage must be finite and >= 1, got {self.global_max_leverage!r}"
            )


@dataclass(frozen=True, slots=True)
class EngineResult:
    """Everything a run produced, in memory.

    Invariants: `equity` has exactly one snapshot per step (the last one after the
    close-out); all sequences are in event order; final_positions are non-flat and sorted by
    symbol; liquidated is always False in M2.
    """

    equity: tuple[EquitySnapshot, ...]
    fills: tuple[Fill, ...]
    orders: tuple[OrderRecord, ...]
    skipped: tuple[SkippedOrder, ...]
    messages: tuple[LogRecord, ...]
    final_positions: tuple[PositionView, ...]
    killed_at: datetime | None
    liquidated: bool


class _ReservingView:
    """PortfolioView that holds back cash and holdings already promised to pending orders.

    Invariants: available_cash == base.available_cash - reserved cash; position(sym).qty ==
    base qty - reserved sells (never below zero); everything else delegates to the base view.
    """

    def __init__(
        self,
        base: PortfolioView,
        instruments: Mapping[str, Instrument],
        last_close: Mapping[str, float],
    ) -> None:
        self._base = base
        self._inst = instruments
        self._last_close = last_close
        self._cash = 0.0
        self._sells: dict[str, float] = {}

    def reserve(self, order: Order) -> None:
        if order.side == Side.SELL:
            self._sells[order.symbol] = self._sells.get(order.symbol, 0.0) + order.qty
            return
        inst = self._inst[order.symbol]
        if order.type == OrderType.LIMIT:
            if order.limit_price is None:
                raise EngineError(f"limit order {order.id} has no limit_price")
            p = order.limit_price
        elif order.type == OrderType.STOP:
            if order.stop_price is None:
                raise EngineError(f"stop order {order.id} has no stop_price")
            p = order.stop_price
        else:
            p = self._last_close[order.symbol]
        self._cash += order.qty * p * (1 + max(inst.maker_fee, inst.taker_fee, 0.0))

    @property
    def ts(self) -> datetime | None:
        return self._base.ts

    @property
    def cash(self) -> float:
        return self._base.cash

    @property
    def available_cash(self) -> float:
        return self._base.available_cash - self._cash

    @property
    def equity(self) -> float:
        return self._base.equity

    def position(self, symbol: str) -> PositionView:
        pv = self._base.position(symbol)
        reserved = self._sells.get(symbol, 0.0)
        if reserved == 0.0 or pv.qty == 0.0:
            return pv
        qty = max(pv.qty - reserved, 0.0)
        mark = self._last_close[symbol]
        return PositionView(symbol, qty, pv.avg_entry, qty * mark, qty * (mark - pv.avg_entry))

    def positions(self) -> Mapping[str, PositionView]:
        return self._base.positions()


def _guard[**P, T](
    what: str, ts: datetime | None, fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> T:
    """Run strategy code; BtErrors pass through, anything else becomes a StrategyError."""
    try:
        return fn(*args, **kwargs)
    except BtError:
        raise
    except Exception as exc:
        where = "" if ts is None else f" at {ts.isoformat()}"
        raise StrategyError(f"{what} raised {type(exc).__name__}: {exc}{where}") from exc


def _apply(fills: Sequence[Fill], portfolio: Portfolio, log: EventLog) -> None:
    for fill in fills:
        portfolio.apply_fill(fill)
        log.record_fill(fill)
        # M8: tax hook here


def _check_wiring(strategy: Strategy, feed: DataFeed) -> tuple[tuple[Instrument, ...], int]:
    feed_insts = tuple(sorted(feed.instruments(), key=lambda i: i.symbol))
    if tuple(sorted(strategy.instruments, key=lambda i: i.symbol)) != feed_insts:
        raise EngineError("strategy instruments differ from feed instruments")
    for inst in feed_insts:
        if inst.kind != InstrumentKind.SPOT:
            raise EngineError(f"{inst.symbol}: only spot is supported; perps arrive in M6")
    warmup = cast("object", _guard("warmup_bars", None, strategy.warmup_bars))
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise StrategyError(f"warmup_bars must be an int >= 0, got {warmup!r}")
    return feed_insts, warmup


def _market_sells(portfolio: Portfolio, prefix: str, ts: datetime, tag: str) -> list[Order]:
    return [
        Order(f"{prefix}-{sym}-{ts:%Y%m%d%H%M}", sym, Side.SELL, pv.qty, OrderType.MARKET, tag=tag)
        for sym, pv in sorted(portfolio.positions().items())
    ]


def run_engine(
    strategy: Strategy,
    feed: DataFeed,
    broker: BacktestBroker,
    portfolio: Portfolio,
    log: EventLog,
    config: EngineConfig,
) -> EngineResult:
    """Run the DESIGN section 5 loop over every step of `feed` and return the result."""
    instruments, warmup = _check_wiring(strategy, feed)
    inst_by_sym = {i.symbol: i for i in instruments}
    peak = portfolio.equity
    killed_at: datetime | None = None
    seen: set[str] = set()
    last_close: dict[str, float] = {}
    ready = False
    started = False
    pending: EquitySnapshot | None = None
    last_ts: datetime | None = None

    for step in feed.steps():
        if pending is not None:
            log.record_equity(pending)
        ts = step.ts
        last_ts = ts
        # M6: apply funding here
        fills = broker.process(step)
        _apply(fills, portfolio, log)
        # M6: liquidation here
        for sym, bar in step.bars.items():
            last_close[sym] = bar.close
        portfolio.mark(ts, {sym: bar.close for sym, bar in step.bars.items()})
        ctx = StepContext(ts, feed, portfolio.view(), instruments, log)
        if killed_at is None:
            for fill in fills:
                _guard("on_fill", ts, strategy.on_fill, fill, ctx)

        k = config.max_drawdown_kill
        if k is not None and killed_at is None:
            equity = portfolio.equity
            peak = max(peak, equity)
            if equity <= peak * (1 - k) + _KILL_REL_TOL * peak:
                killed_at = ts
                log.record_message(
                    LogRecord(
                        ts,
                        "kill_switch",
                        (("drawdown", 1 - equity / peak), ("equity", equity), ("peak", peak)),
                    )
                )
                broker.cancel_all("kill_switch")
                view = _ReservingView(portfolio.view(), inst_by_sym, last_close)
                accepted: list[Order] = []
                for order in _market_sells(portfolio, "bt-kill", ts, "kill_switch"):
                    seen.add(order.id)
                    ack = validate_order(
                        order, inst_by_sym[order.symbol], last_close[order.symbol], view
                    )
                    if ack.status == OrderStatus.REJECTED:
                        log.record_order(OrderRecord(ts, order, OrderStatus.REJECTED, ack.reason))
                    else:
                        view.reserve(order)
                        accepted.append(order)
                broker.submit(accepted)

        if killed_at is None:
            if not started:
                started = True
                _guard("on_start", ts, strategy.on_start, ctx)
            if not ready:
                ready = all(len(ctx.history(s, warmup).ts) >= warmup for s in ctx.symbols())
            if ready:
                raw: object = _guard("on_bar", ts, strategy.on_bar, ctx)
                _submit_intent(raw, ts, broker, log, inst_by_sym, last_close, seen, portfolio)
        pending = portfolio.snapshot()

    if last_ts is None:
        raise EngineError("feed yielded no steps")
    broker.cancel_all("end_of_data")
    if config.close_at_end:
        orders = _market_sells(portfolio, "bt-close", last_ts, "close_at_end")
        _apply(broker.settle_at_close(last_ts, orders), portfolio, log)
    log.record_equity(portfolio.snapshot())
    return EngineResult(
        equity=log.equity(),
        fills=log.fills(),
        orders=log.orders(),
        skipped=(),
        messages=log.messages(),
        final_positions=tuple(portfolio.positions().values()),
        killed_at=killed_at,
        liquidated=False,
    )


def _submit_intent(
    raw: object,
    ts: datetime,
    broker: BacktestBroker,
    log: EventLog,
    inst_by_sym: Mapping[str, Instrument],
    last_close: Mapping[str, float],
    seen: set[str],
    portfolio: Portfolio,
) -> None:
    if isinstance(raw, NoAction):
        return
    if isinstance(raw, TargetWeights):
        raise StrategyError(f"TargetWeights needs the M5 rebalancer (at {ts.isoformat()})")
    if not isinstance(raw, Orders):
        raise StrategyError(f"on_bar returned {type(raw).__name__}, not an Intent")
    view = _ReservingView(portfolio.view(), inst_by_sym, last_close)
    for o in broker.open_orders():
        view.reserve(o)
    accepted: list[Order] = []
    for order in raw.orders:
        reason = ""
        if order.id in seen or order.id.startswith(_RESERVED_PREFIX):
            reason = "duplicate_id"
        elif order.symbol not in inst_by_sym:
            reason = "unknown_symbol"
        elif order.symbol not in last_close:
            reason = "no_reference_price"
        seen.add(order.id)
        if not reason:
            ack = validate_order(order, inst_by_sym[order.symbol], last_close[order.symbol], view)
            reason = ack.reason
        if reason:
            log.record_order(_rejected(ts, order, reason))
        else:
            view.reserve(order)
            accepted.append(order)
    broker.submit(accepted)


def _rejected(ts: datetime, order: Order, reason: str) -> OrderRecord:
    return OrderRecord(ts, order, OrderStatus.REJECTED, reason)
