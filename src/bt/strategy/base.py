"""Strategy base class and the three intents a strategy may return."""

import math
import numbers
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict

from bt.core.errors import StrategyError
from bt.core.types import Fill, Instrument, Order
from bt.strategy.context import Context


@dataclass(frozen=True, slots=True)
class Orders:
    """Explicit orders to place at the next bar's open.

    Invariants: `orders` is a private list copy; every element is an Order. An empty list
    means nothing to do. Not hashable, because it holds a list.
    """

    orders: list[Order]

    def __post_init__(self) -> None:
        items = list(self.orders)
        for o in items:
            if not isinstance(o, Order):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise StrategyError(f"Orders: expected Order, got {type(o).__name__}")
        object.__setattr__(self, "orders", items)


def _to_float(key: str, v: object) -> float:
    if isinstance(v, bool | np.bool_) or not isinstance(v, numbers.Real | np.integer | np.floating):
        raise StrategyError(f"TargetWeights: weight for {key!r} must be a real number, got {v!r}")
    try:
        f = float(v)  # pyright: ignore[reportUnknownArgumentType]
    except OverflowError as e:
        raise StrategyError(f"TargetWeights: weight for {key!r} overflows float, got {v!r}") from e
    if not math.isfinite(f):
        raise StrategyError(f"TargetWeights: weight for {key!r} must be finite, got {v!r}")
    return f


@dataclass(frozen=True, slots=True)
class TargetWeights:
    """Desired signed fraction of equity per symbol; a missing symbol means 0.

    Invariants: `weights` is a read-only snapshot; every key is a non-empty str and every
    value a finite Python float (ints and numpy reals are converted; bools are rejected).
    Not hashable, because it holds a Mapping.
    """

    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        snap: dict[str, float] = {}
        for k, v in dict(self.weights).items():
            if not isinstance(k, str) or not k:  # pyright: ignore[reportUnnecessaryIsInstance]
                raise StrategyError(f"TargetWeights: key must be a non-empty str, got {k!r}")
            snap[k] = _to_float(k, v)
        object.__setattr__(self, "weights", MappingProxyType(snap))

    def __reduce__(self) -> tuple[type["TargetWeights"], tuple[dict[str, float]]]:
        # MappingProxyType is not picklable; rebuild through __init__ so invariants re-run.
        return (TargetWeights, (dict(self.weights),))


@dataclass(frozen=True, slots=True)
class NoAction:
    """Do nothing this bar."""


Intent = Orders | TargetWeights | NoAction


class _DefaultParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Strategy(ABC):
    """Base class of every strategy.

    Invariants after __init__: `params` is an instance of type(self).Params; `instruments` is
    a non-empty tuple sorted by symbol with unique symbols. Subclasses that override __init__
    must call super().__init__. Subclasses override `Params`; extra="forbid", frozen=True is
    recommended (not enforced). Every Params field must have a default.
    """

    Params: ClassVar[type[BaseModel]] = _DefaultParams

    params: BaseModel
    instruments: tuple[Instrument, ...]

    def __init__(self, params: BaseModel, instruments: Sequence[Instrument]) -> None:
        if not isinstance(params, type(self).Params):
            raise StrategyError(
                f"{type(self).__name__}: params must be a {type(self).Params.__qualname__}, "
                f"got {type(params).__name__}"
            )
        insts = tuple(sorted(instruments, key=lambda i: i.symbol))
        if not insts:
            raise StrategyError(f"{type(self).__name__}: instruments must be non-empty")
        symbols = [i.symbol for i in insts]
        if len(set(symbols)) != len(symbols):
            raise StrategyError(f"{type(self).__name__}: duplicate instrument symbols {symbols}")
        self.params = params
        self.instruments = insts

    def warmup_bars(self) -> int:
        """History needed before the first decision."""
        return 0

    def on_start(self, ctx: Context) -> None:  # noqa: B027
        """Called once before the first bar."""

    @abstractmethod
    def on_bar(self, ctx: Context) -> Intent:
        """Decide on the bar just closed."""

    def on_fill(self, fill: Fill, ctx: Context) -> None:  # noqa: B027
        """Called for each fill."""
