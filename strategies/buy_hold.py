"""Buy and hold: split `alloc` of equity equally over the universe, buy once, never sell."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from bt.core.types import Order, Side
from bt.strategy.base import Intent, NoAction, Orders, Strategy
from bt.strategy.context import Context
from bt.strategy.helpers import affordable_qty, has_bar_now
from bt.strategy.registry import register


class BuyHoldParams(BaseModel):
    """alloc in (0, 1]: fraction of equity deployed, split equally over the universe."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    alloc: float = Field(default=0.995, gt=0.0, le=1.0)

    @field_validator("alloc", mode="before")
    @classmethod
    def _no_bool(cls, v: object) -> object:
        if isinstance(v, bool):
            raise ValueError("alloc must be a number, not a bool")
        return v


@register("buy_hold")
class BuyHold(Strategy):
    """Buys every flat symbol with an equal share of `alloc * equity`; holds forever.

    Invariants: stateless (decisions read positions); order ids are `bh-<sym>-<ts>`; never sells.
    """

    Params = BuyHoldParams

    def on_bar(self, ctx: Context) -> Intent:
        p = self.params
        assert isinstance(p, BuyHoldParams)
        symbols = ctx.symbols()
        budget = ctx.equity * p.alloc / len(symbols)
        orders: list[Order] = []
        for sym in symbols:
            if not has_bar_now(ctx, sym) or ctx.position(sym).qty != 0:
                continue
            close = float(ctx.history(sym, 1).close[-1])
            qty = affordable_qty(budget, close, ctx.instrument(sym))
            if qty == 0.0:
                ctx.log("buy_hold_too_small", symbol=sym)
                continue
            orders.append(Order(f"bh-{sym}-{ctx.now:%Y%m%d%H%M}", sym, Side.BUY, qty))
        return Orders(orders) if orders else NoAction()
