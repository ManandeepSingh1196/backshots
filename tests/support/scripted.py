"""Test-support strategy that emits a fixed script of orders."""

from collections.abc import Mapping, Sequence
from datetime import datetime

from pydantic import BaseModel

from bt.core.types import Fill, Instrument, Order
from bt.strategy.base import Intent, NoAction, Orders, Strategy
from bt.strategy.context import Context


class ScriptedStrategy(Strategy):
    """Returns `script[ctx.now]` as Orders when present, else NoAction; records its calls."""

    def __init__(
        self,
        params: BaseModel,
        instruments: Sequence[Instrument],
        script: Mapping[datetime, Sequence[Order]],
        warmup: int = 0,
    ) -> None:
        super().__init__(params, instruments)
        self.script = {ts: list(orders) for ts, orders in script.items()}
        self.warmup = warmup
        self.calls: list[datetime] = []
        self.fills_seen: list[Fill] = []

    def warmup_bars(self) -> int:
        return self.warmup

    def on_bar(self, ctx: Context) -> Intent:
        self.calls.append(ctx.now)
        if ctx.now in self.script:
            return Orders(list(self.script[ctx.now]))
        return NoAction()

    def on_fill(self, fill: Fill, ctx: Context) -> None:
        self.fills_seen.append(fill)
