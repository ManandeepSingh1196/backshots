"""Moving-average crossover: long while the fast mean of closes is above the slow mean."""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bt.core.types import Order, Side
from bt.strategy.base import Intent, NoAction, Orders, Strategy
from bt.strategy.context import Context
from bt.strategy.helpers import affordable_qty, has_bar_now
from bt.strategy.registry import register


class MACrossParams(BaseModel):
    """fast >= 1, slow > fast, alloc in (0, 1]."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fast: int = Field(default=20, ge=1)
    slow: int = 100
    alloc: float = Field(default=0.995, gt=0.0, le=1.0)

    @field_validator("fast", "slow", "alloc", mode="before")
    @classmethod
    def _no_bool(cls, v: object) -> object:
        if isinstance(v, bool):
            raise ValueError("must be a number, not a bool")
        return v

    @model_validator(mode="after")
    def _slow_gt_fast(self) -> "MACrossParams":
        if self.slow <= self.fast:
            raise ValueError(f"slow ({self.slow}) must be > fast ({self.fast})")
        return self


@register("ma_cross")
class MACross(Strategy):
    """Spot long/flat on a level signal (fast_ma > slow_ma), per symbol.

    Invariants: stateless (decisions read positions); never shorts; sells the whole position;
    SELLs precede BUYs in one Orders; order ids are `ma-<sym>-<ts>`.
    """

    Params = MACrossParams

    def warmup_bars(self) -> int:
        p = self.params
        assert isinstance(p, MACrossParams)
        return p.slow

    def on_bar(self, ctx: Context) -> Intent:
        p = self.params
        assert isinstance(p, MACrossParams)
        symbols = ctx.symbols()
        budget = ctx.equity * p.alloc / len(symbols)
        sells: list[Order] = []
        buys: list[Order] = []
        for sym in symbols:
            if not has_bar_now(ctx, sym):
                continue
            close = ctx.history(sym, p.slow).close
            long_signal = float(np.mean(close[-p.fast :])) > float(np.mean(close))
            held = ctx.position(sym).qty
            oid = f"ma-{sym}-{ctx.now:%Y%m%d%H%M}"
            if held > 0 and not long_signal:
                sells.append(Order(oid, sym, Side.SELL, held))
            elif held == 0 and long_signal:
                qty = affordable_qty(budget, float(close[-1]), ctx.instrument(sym))
                if qty == 0.0:
                    ctx.log("ma_cross_too_small", symbol=sym)
                    continue
                buys.append(Order(oid, sym, Side.BUY, qty))
        orders = sells + buys
        return Orders(orders) if orders else NoAction()
