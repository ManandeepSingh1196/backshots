"""Run-file frames: an EngineResult as typed polars DataFrames (pure, no I/O).

Every frame has exactly its `*_SCHEMA` columns, in order, even when empty. Timestamps are
`Datetime("us", "UTC")`; enums are written as their string value.

Row order in `orders_frame`: log order (OrderRecord) first, then skipped orders. Within one
step, engine rejections precede broker NEW records, so row order is NOT intent order.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

import polars as pl

from bt.analytics.metrics import drawdown_series
from bt.core.types import Side
from bt.data.validate import TIMEFRAME_DELTA, Timeframe
from bt.engine.loop import EngineResult
from bt.runs.fingerprint import canonical_json

_TS = pl.Datetime("us", "UTC")
_F = pl.Float64()
_S = pl.String()
_B = pl.Boolean()

EQUITY_SCHEMA: Final[Mapping[str, pl.DataType]] = MappingProxyType(
    {
        "ts": _TS,
        "close_ts": _TS,
        "equity_pre_tax": _F,
        "equity_after_tax": _F,
        "cash": _F,
        "gross_exposure": _F,
        "net_exposure": _F,
        "drawdown": _F,
        "realized_pnl": _F,
        "fees_paid": _F,
        "slippage_paid": _F,
        "funding_paid": _F,
        "tax_paid": _F,
    }
)
TRADES_SCHEMA: Final[Mapping[str, pl.DataType]] = MappingProxyType(
    {
        "event": _S,
        "ts": _TS,
        "symbol": _S,
        "order_id": _S,
        "side": _S,
        "qty": _F,
        "price": _F,
        "notional": _F,
        "fee": _F,
        "slippage_cost": _F,
        "is_maker": _B,
        "cash_flow": _F,
        "tag": _S,
    }
)
ORDERS_SCHEMA: Final[Mapping[str, pl.DataType]] = MappingProxyType(
    {
        "ts": _TS,
        "order_id": _S,
        "symbol": _S,
        "side": _S,
        "type": _S,
        "qty": _F,
        "limit_price": _F,
        "stop_price": _F,
        "reduce_only": _B,
        "tag": _S,
        "status": _S,
        "reason": _S,
    }
)
MESSAGES_SCHEMA: Final[Mapping[str, pl.DataType]] = MappingProxyType(
    {"ts": _TS, "message": _S, "fields": _S}
)


def _frame(columns: Mapping[str, Sequence[object]], schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(columns, schema=dict(schema))


def equity_frame(result: EngineResult, timeframe: Timeframe) -> pl.DataFrame:
    """One row per EquitySnapshot; ts is the bar open, close_ts = ts + timeframe."""
    snaps = result.equity
    delta = TIMEFRAME_DELTA[timeframe]
    after = [s.equity for s in snaps]
    cols: Mapping[str, Sequence[object]] = {
        "ts": [s.ts for s in snaps],
        "close_ts": [s.ts + delta for s in snaps],
        "equity_pre_tax": [s.equity + s.tax_paid for s in snaps],
        "equity_after_tax": list(after),
        "cash": [s.cash for s in snaps],
        "gross_exposure": [s.gross_exposure for s in snaps],
        "net_exposure": [s.net_exposure for s in snaps],
        "drawdown": list(drawdown_series(after)),
        "realized_pnl": [s.realized_pnl for s in snaps],
        "fees_paid": [s.fees_paid for s in snaps],
        "slippage_paid": [s.slippage_paid for s in snaps],
        "funding_paid": [s.funding_paid for s in snaps],
        "tax_paid": [s.tax_paid for s in snaps],
    }
    return _frame(cols, EQUITY_SCHEMA)


def trades_frame(result: EngineResult) -> pl.DataFrame:
    """One row per fill, in order; cash_flow is signed (BUY pays notional + fee)."""
    tags = {r.order.id: r.order.tag for r in result.orders}
    fills = result.fills
    notional = [f.qty * f.price for f in fills]
    cash_flow = [
        -(n + f.fee) if f.side == Side.BUY else n - f.fee for f, n in zip(fills, notional)
    ]
    cols: Mapping[str, Sequence[object]] = {
        "event": ["fill" for _ in fills],
        "ts": [f.ts for f in fills],
        "symbol": [f.symbol for f in fills],
        "order_id": [f.order_id for f in fills],
        "side": [f.side.value for f in fills],
        "qty": [f.qty for f in fills],
        "price": [f.price for f in fills],
        "notional": notional,
        "fee": [f.fee for f in fills],
        "slippage_cost": [f.slippage_cost for f in fills],
        "is_maker": [f.is_maker for f in fills],
        "cash_flow": cash_flow,
        "tag": [tags.get(f.order_id, "") for f in fills],
    }
    return _frame(cols, TRADES_SCHEMA)


def orders_frame(result: EngineResult) -> pl.DataFrame:
    """OrderRecords in log order, then SkippedOrders (order_id and type null)."""
    recs, skips = result.orders, result.skipped
    cols: Mapping[str, Sequence[object]] = {
        "ts": [r.ts for r in recs] + [s.ts for s in skips],
        "order_id": [r.order.id for r in recs] + [None for _ in skips],
        "symbol": [r.order.symbol for r in recs] + [s.symbol for s in skips],
        "side": [r.order.side.value for r in recs] + [s.side.value for s in skips],
        "type": [r.order.type.value for r in recs] + [None for _ in skips],
        "qty": [r.order.qty for r in recs] + [s.qty for s in skips],
        "limit_price": [r.order.limit_price for r in recs] + [None for _ in skips],
        "stop_price": [r.order.stop_price for r in recs] + [None for _ in skips],
        "reduce_only": [r.order.reduce_only for r in recs] + [False for _ in skips],
        "tag": [r.order.tag for r in recs] + ["" for _ in skips],
        "status": [r.status.value for r in recs] + ["skipped" for _ in skips],
        "reason": [r.reason for r in recs] + [s.reason for s in skips],
    }
    return _frame(cols, ORDERS_SCHEMA)


def messages_frame(result: EngineResult) -> pl.DataFrame:
    """One row per LogRecord; fields is a canonical JSON object."""
    msgs = result.messages
    cols: Mapping[str, Sequence[object]] = {
        "ts": [m.ts for m in msgs],
        "message": [m.message for m in msgs],
        "fields": [canonical_json(dict(m.fields)) for m in msgs],
    }
    return _frame(cols, MESSAGES_SCHEMA)
