"""Test-support strategy: returns a scripted TargetWeights per on_bar call, then NoAction."""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from bt.core.errors import StrategyError
from bt.core.types import Instrument
from bt.strategy.base import Intent, NoAction, Strategy, TargetWeights
from bt.strategy.context import Context
from bt.strategy.registry import register


class WeightsParams(BaseModel):
    """Params of TestWeights: one target-weights dict per on_bar call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    targets: list[dict[str, float]] = []


@register("test_weights")
class TestWeights(Strategy):
    """The i-th on_bar call returns TargetWeights(targets[i]); later calls return NoAction."""

    __test__ = False  # not a pytest class

    Params = WeightsParams

    def __init__(self, params: BaseModel, instruments: Sequence[Instrument]) -> None:
        super().__init__(params, instruments)
        p = self.params
        if not isinstance(p, WeightsParams):
            raise StrategyError("TestWeights: wrong params type")
        self._targets = [dict(t) for t in p.targets]
        self._i = 0

    def warmup_bars(self) -> int:
        return 0

    def on_bar(self, ctx: Context) -> Intent:
        i = self._i
        self._i += 1
        if i < len(self._targets):
            return TargetWeights(self._targets[i])
        return NoAction()
