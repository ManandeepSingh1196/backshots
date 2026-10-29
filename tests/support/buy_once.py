"""Test-support strategy: market-buys fixed quantities on the first bar only."""

from collections.abc import Sequence

from pydantic import BaseModel

from bt.core.errors import StrategyError
from bt.core.types import Instrument, Order, Side
from bt.strategy.base import Intent, NoAction, Orders, Strategy
from bt.strategy.context import Context
from bt.strategy.registry import register


class BuyOnceParams(BaseModel):
    """Params of BuyOnce: symbol -> qty to buy, and the warmup it asks for."""

    buys: dict[str, float] = {}
    warmup: int = 0


@register("test_buy_once")
class BuyOnce(Strategy):
    """Emits market BUYs for `buys` on the first on_bar call, then does nothing."""

    Params = BuyOnceParams

    def __init__(self, params: BaseModel, instruments: Sequence[Instrument]) -> None:
        super().__init__(params, instruments)
        p = self.params
        if not isinstance(p, BuyOnceParams):
            raise StrategyError("BuyOnce: wrong params type")
        self._buys: dict[str, float] = dict(p.buys)
        self._warmup: int = p.warmup
        self._done = False

    def warmup_bars(self) -> int:
        return self._warmup

    def on_bar(self, ctx: Context) -> Intent:
        if self._done:
            return NoAction()
        self._done = True
        buys = self._buys
        ctx.log("buy_once", n=len(buys))
        return Orders([Order(f"buy-{s}", s, Side.BUY, q) for s, q in sorted(buys.items())])
