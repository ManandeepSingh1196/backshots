"""Target weights to exchange-valid market orders (spot). Pure: reads the world only via ctx."""

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from bt.core.errors import EngineError, InvariantError, StrategyError
from bt.core.events import SkippedOrder, SkipReason
from bt.core.rounding import floor_to_step
from bt.core.types import Instrument, InstrumentKind, Order, OrderType, Side
from bt.execution.rules import QTY_REL_TOL
from bt.strategy.context import Context

REBALANCE_TAG: Final = "rebalance"


def rebalance_order_id(symbol: str, ts: datetime) -> str:
    """Deterministic id of a rebalance order: one per symbol per bar close (minute resolution)."""
    return f"bt-rb-{symbol}-{ts:%Y%m%d%H%M}"


@dataclass(frozen=True, slots=True)
class _Cand:
    symbol: str
    inst: Instrument
    side: Side
    qty: float
    ref: float
    delta: float
    rate: float


def _real(x: object) -> float | None:
    """float(x) for a finite real non-bool number, else None."""
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        return None
    try:
        f = float(x)
    except OverflowError:
        return None
    return f if math.isfinite(f) else None


def _check_args(band: object, cap: object, buf: object) -> tuple[float, float, float]:
    b, c, cb = _real(band), _real(cap), _real(buf)
    if b is None or b < 0:
        raise EngineError(f"weights_to_orders: band must be finite and >= 0, got {band!r}")
    if c is None or c <= 0:
        raise EngineError(f"weights_to_orders: max_gross_leverage must be > 0, got {cap!r}")
    if cb is None or not 0 <= cb < 1:
        raise EngineError(f"weights_to_orders: cash_buffer must be in [0, 1), got {buf!r}")
    return b, c, cb


def _check_weights(weights: Mapping[str, float], symbols: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for sym in sorted(weights, key=str):
        if sym not in symbols:
            raise StrategyError(f"weights_to_orders: unknown symbol {sym!r} in weights")
        w = _real(weights[sym])
        if w is None:
            raise StrategyError(
                f"weights_to_orders: weight for {sym!r} must be a finite real number, "
                f"got {weights[sym]!r}"
            )
        if w < 0:
            raise StrategyError(f"weights_to_orders: negative weight for {sym!r} on spot: {w!r}")
        out[sym] = w
    return out


def _ref_price(ctx: Context, symbol: str) -> float:
    close = ctx.history(symbol, 1).close
    if len(close) == 0:
        raise StrategyError(f"weights_to_orders: no price history for {symbol!r}")
    ref = float(close[-1])
    if not math.isfinite(ref) or ref <= 0:
        raise StrategyError(f"weights_to_orders: bad reference price {ref!r} for {symbol!r}")
    return ref


def _filter(inst: Instrument, qty: float, ref: float) -> SkipReason | None:
    if qty == 0:
        return "zero_qty"
    if qty < inst.min_qty * (1 - QTY_REL_TOL):
        return "below_min_qty"
    if qty * ref < inst.min_notional * (1 - QTY_REL_TOL):
        return "below_min_notional"
    return None


def weights_to_orders(
    weights: Mapping[str, float],
    ctx: Context,
    band: float,
    max_gross_leverage: float,
    *,
    open_orders: Sequence[Order] = (),
    cash_buffer: float = 0.0,
) -> tuple[list[Order], list[SkippedOrder]]:
    """Turn target weights into sorted market orders plus logged skips (spot only).

    Invariants: output depends only on the argument values (never on mapping order); sells
    precede buys and each side is sorted by symbol; buys are scaled to fit the cash budget
    (cash + haircut sell proceeds - reserves of open buys); sells never exceed holdings
    net of pending sells; every order passes the exchange minimums.
    """
    band, max_gross_leverage, cash_buffer = _check_args(band, max_gross_leverage, cash_buffer)
    symbols = sorted(ctx.symbols())
    insts = {s: ctx.instrument(s) for s in symbols}
    for s in symbols:
        if insts[s].kind != InstrumentKind.SPOT:
            raise EngineError(f"weights_to_orders: {s} is not spot (perps are M6)")
    w_in = _check_weights(weights, symbols)
    equity = float(ctx.equity)
    if not math.isfinite(equity):
        raise EngineError(f"weights_to_orders: equity must be finite, got {equity!r}")
    if equity <= 0:
        return [], []
    cash = float(ctx.cash)

    try:
        gross = math.fsum(abs(v) for v in w_in.values())
    except OverflowError as exc:
        raise StrategyError("weights_to_orders: sum of |weights| overflows") from exc
    if gross > max_gross_leverage:
        scale = max_gross_leverage / gross
        w_in = {s: v * scale for s, v in w_in.items()}

    for o in open_orders:
        if o.symbol not in insts:
            raise EngineError(f"weights_to_orders: open order {o.id} for unknown {o.symbol!r}")

    refs: dict[str, float] = {}
    sells: list[_Cand] = []
    buys: list[_Cand] = []
    for sym in symbols:
        w = w_in.get(sym, 0.0)
        held = float(ctx.position(sym).qty)
        pend_buy = math.fsum(o.qty for o in open_orders if o.symbol == sym and o.side == Side.BUY)
        pend_sell = math.fsum(o.qty for o in open_orders if o.symbol == sym and o.side == Side.SELL)
        if w == 0 and held == 0 and pend_buy == 0 and pend_sell == 0:
            continue
        ref = _ref_price(ctx, sym)
        refs[sym] = ref
        inst = insts[sym]
        rate = max(inst.maker_fee, inst.taker_fee, 0.0)
        delta = w * equity - (held + pend_buy - pend_sell) * ref
        if delta == 0 or abs(delta) < band * equity:
            continue
        if delta < 0:
            sellable = held - pend_sell
            if sellable <= 0:
                continue
            if w == 0 and pend_buy == 0:
                qty = sellable
            else:
                qty = min(_floor(-delta / ref, inst), sellable)
            sells.append(_Cand(sym, inst, Side.SELL, qty, ref, delta, rate))
        else:
            buys.append(_Cand(sym, inst, Side.BUY, 0.0, ref, delta, rate))

    orders: list[Order] = []
    skipped: list[SkippedOrder] = []
    proceeds: list[float] = []
    for c in sells:
        reason = _filter(c.inst, c.qty, c.ref)
        if reason is not None:
            skipped.append(_skip(ctx.now, c, c.qty, reason))
            continue
        orders.append(_order(ctx.now, c, c.qty))
        proceeds.append(c.qty * c.ref * (1 - c.rate) * (1 - cash_buffer))

    reserves: list[float] = []
    for o in open_orders:
        inst = insts[o.symbol]
        rate = max(inst.maker_fee, inst.taker_fee, 0.0)
        if o.side == Side.SELL:
            if o.type == OrderType.MARKET:
                proceeds.append(o.qty * refs[o.symbol] * (1 - rate) * (1 - cash_buffer))
            continue
        if o.type == OrderType.LIMIT and o.limit_price is not None:
            p = o.limit_price
        elif o.type == OrderType.STOP and o.stop_price is not None:
            p = o.stop_price
        else:
            p = refs[o.symbol]
        reserves.append(o.qty * p * (1 + rate) * (1 + cash_buffer))
    budget = max(0.0, cash + math.fsum(proceeds) - math.fsum(reserves))

    if buys:
        need = math.fsum(c.delta * (1 + c.rate) * (1 + cash_buffer) for c in buys)
        k = min(1.0, budget / need)
        for c in buys:
            qty = _floor(k * c.delta / c.ref, c.inst)
            reason = _filter(c.inst, qty, c.ref)
            if reason is not None:
                skipped.append(_skip(ctx.now, c, qty, reason))
            else:
                orders.append(_order(ctx.now, c, qty))
    return orders, skipped


def _floor(value: float, inst: Instrument) -> float:
    try:
        return floor_to_step(value, inst.qty_step)
    except InvariantError as exc:
        raise StrategyError(f"weights_to_orders: cannot round {value!r}: {exc}") from exc


def _order(ts: datetime, c: _Cand, qty: float) -> Order:
    try:
        return Order(
            rebalance_order_id(c.symbol, ts),
            c.symbol,
            c.side,
            float(qty),
            OrderType.MARKET,
            tag=REBALANCE_TAG,
        )
    except InvariantError as exc:
        raise StrategyError(f"weights_to_orders: bad order for {c.symbol}: {exc}") from exc


def _skip(ts: datetime, c: _Cand, qty: float, reason: SkipReason) -> SkippedOrder:
    try:
        return SkippedOrder(ts, c.symbol, c.side, float(qty), c.ref, reason)
    except InvariantError as exc:
        raise StrategyError(f"weights_to_orders: bad skip for {c.symbol}: {exc}") from exc
