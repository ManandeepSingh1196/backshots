"""Look-ahead canary strategies and a deliberately leaky feed. Test support only."""

from collections.abc import Sequence
from datetime import datetime, timedelta
from itertools import pairwise

import numpy as np
from pydantic import BaseModel

from bt.core.errors import StrategyError
from bt.core.types import Bar, Instrument, Order, Side
from bt.data.feed import DataFeed
from bt.strategy.base import Intent, NoAction, Orders, Strategy
from bt.strategy.context import Context, HistoryView
from tests.support.memory_feed import MemoryFeed

_BIG = 10**6
Snapshot = tuple[datetime, tuple[list[object], ...]]


def _snapshot(now: datetime, h: HistoryView) -> Snapshot:
    cols = (h.ts, h.open, h.high, h.low, h.close, h.volume)
    return (now, tuple(c.tolist() for c in cols))


class _OneSymbol(Strategy):
    """Shared plumbing: a single symbol, unique order ids, recorded history snapshots."""

    def __init__(self, params: BaseModel, instruments: Sequence[Instrument]) -> None:
        super().__init__(params, instruments)
        self.sym = self.instruments[0].symbol
        self.snapshots: list[Snapshot] = []
        self._n = 0

    def _order(self, side: Side, qty: float) -> Orders:
        self._n += 1
        return Orders([Order(f"c{self._n}", self.sym, side, qty)])


class TrendCanary(_OneSymbol):
    """Buys 5 after an up bar when flat; sells everything after a down bar when long.

    Reads only `ctx.history(sym, 1)`, so on an alternating dataset it is always on the wrong
    side of the next bar and earns nothing."""

    def on_bar(self, ctx: Context) -> Intent:
        self.snapshots.append(_snapshot(ctx.now, ctx.history(self.sym, _BIG)))
        last = ctx.history(self.sym, 1)
        if len(last.ts) == 0:
            return NoAction()
        qty = ctx.position(self.sym).qty
        if last.close[-1] > last.open[-1] and qty == 0.0:
            return self._order(Side.BUY, 5.0)
        if last.close[-1] < last.open[-1] and qty > 0.0:
            return self._order(Side.SELL, qty)
        return NoAction()


class OracleCanary(_OneSymbol):
    """Handed the bars up front (foresight): trades on whether the NEXT bar is up or down."""

    def __init__(
        self, params: BaseModel, instruments: Sequence[Instrument], bars: Sequence[Bar]
    ) -> None:
        super().__init__(params, instruments)
        ordered = sorted(bars, key=lambda b: b.ts)
        self._next = {a.ts: b for a, b in pairwise(ordered)}

    def on_bar(self, ctx: Context) -> Intent:
        nxt = self._next.get(ctx.now)
        if nxt is None:
            return NoAction()
        qty = ctx.position(self.sym).qty
        if nxt.close > nxt.open and qty == 0.0:
            return self._order(Side.BUY, 5.0)
        if nxt.close < nxt.open and qty > 0.0:
            return self._order(Side.SELL, qty)
        return NoAction()


class SneakyStrategy(_OneSymbol):
    """Bypasses Context and asks the feed for the bar after `now`."""

    def __init__(
        self, params: BaseModel, instruments: Sequence[Instrument], feed: DataFeed
    ) -> None:
        super().__init__(params, instruments)
        self.feed = feed

    def on_bar(self, ctx: Context) -> Intent:
        self.feed.history(self.sym, ctx.now + timedelta(days=1), 1)
        return NoAction()


class ProbeStrategy(_OneSymbol):
    """Records, on every call, whether the Context contract holds."""

    def __init__(self, params: BaseModel, instruments: Sequence[Instrument]) -> None:
        super().__init__(params, instruments)
        self.checks: list[tuple[str, bool]] = []

    def on_bar(self, ctx: Context) -> Intent:
        self._n += 1
        h = ctx.history(self.sym, _BIG)
        now64 = np.datetime64(ctx.now.replace(tzinfo=None), "us")
        self.checks.append(("last_ts_is_now", bool(h.ts[-1] == now64)))
        self.checks.append(("length_is_steps_so_far", len(h.ts) == self._n))
        for name in ("ts", "open", "high", "low", "close", "volume"):
            arr: np.ndarray = getattr(h, name)
            try:
                arr[0] = arr[0]
            except ValueError:
                self.checks.append((f"{name}_readonly", True))
            else:
                self.checks.append((f"{name}_readonly", False))
        self.checks.append(("no_funding", ctx.funding_history(self.sym, _BIG) == ()))
        try:
            ctx.history(self.sym, -1)
        except StrategyError:
            self.checks.append(("negative_n_rejected", True))
        else:
            self.checks.append(("negative_n_rejected", False))
        return NoAction()


class LeakyFeed(MemoryFeed):
    """A MemoryFeed whose history() returns bars through the step AFTER `end`, with no guard."""

    def history(self, symbol: str, end: datetime, n: int) -> HistoryView:
        later = [s.ts for s in self._steps if s.ts > end]
        if not later:
            return super().history(symbol, end, n)
        saved = self._next_ts
        self._next_ts = None  # skip the look-ahead guard
        try:
            return super().history(symbol, later[0], n)
        finally:
            self._next_ts = saved
