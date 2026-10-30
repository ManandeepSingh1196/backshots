"""Daily vol-targeted, long-or-flat time-series momentum on spot, returned as TargetWeights."""

import math
from collections.abc import Sequence
from typing import Final, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bt.strategy.base import Intent, Strategy, TargetWeights
from bt.strategy.context import Context
from bt.strategy.registry import register

_ZERO_VOL: Final = 1e-12
_YEAR_SECONDS: Final = 365.0 * 86400.0


class TSMOMParams(BaseModel):
    """lookbacks unique, each >= 1; vol_window >= 2; vol_target > 0; max_leverage in (0, 2]."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lookbacks: tuple[int, ...] = (20, 60, 120)
    vol_window: int = Field(default=30, ge=2)
    vol_target: float = Field(default=0.25, gt=0.0, allow_inf_nan=False)
    max_leverage: float = Field(default=1.0, gt=0.0, le=2.0)

    @field_validator("lookbacks", "vol_window", "vol_target", "max_leverage", mode="before")
    @classmethod
    def _no_bool(cls, v: object) -> object:
        if isinstance(v, bool):
            raise ValueError("must be a number, not a bool")
        if isinstance(v, list | tuple) and any(
            isinstance(x, bool) for x in cast("Sequence[object]", v)
        ):
            raise ValueError("lookbacks must be ints, not bools")
        return cast("object", v)

    @model_validator(mode="after")
    def _check_lookbacks(self) -> "TSMOMParams":
        if not self.lookbacks:
            raise ValueError("lookbacks must be non-empty")
        if len(set(self.lookbacks)) != len(self.lookbacks):
            raise ValueError("lookbacks must be unique")
        if any(lb < 1 for lb in self.lookbacks):
            raise ValueError("each lookback must be >= 1")
        return self


@register("tsmom")
class TSMOM(Strategy):
    """Long-or-flat TSMOM: weight = signal * (vol_target / n) / vol, capped at max_leverage gross.

    Invariants: stateless; reads only `ctx.history`; returns a TargetWeights covering every
    symbol of the universe on every bar (zeros included, so held positions get flattened);
    weights are Python floats in sorted symbol order; never short; gross <= max_leverage.
    """

    Params = TSMOMParams

    def warmup_bars(self) -> int:
        p = self.params
        assert isinstance(p, TSMOMParams)
        return max(max(p.lookbacks), p.vol_window) + 1

    def on_bar(self, ctx: Context) -> Intent:
        p = self.params
        assert isinstance(p, TSMOMParams)
        warmup = self.warmup_bars()
        symbols = sorted(ctx.symbols())
        n = len(symbols)
        raw: list[float] = []
        for sym in symbols:
            h = ctx.history(sym, warmup)
            c = np.asarray(h.close, dtype=np.float64)
            if len(c) < warmup:
                raw.append(0.0)
                continue
            signal = sum(1 for lb in p.lookbacks if c[-1] > c[-1 - lb]) / len(p.lookbacks)
            if signal == 0.0:
                raw.append(0.0)
                continue
            spacing = float(np.median(np.diff(h.ts)) / np.timedelta64(1, "s"))
            periods_per_year = _YEAR_SECONDS / spacing
            r = np.diff(np.log(c[-p.vol_window - 1 :]))
            vol = float(np.std(r, ddof=1)) * math.sqrt(periods_per_year)
            if not vol > _ZERO_VOL:
                raw.append(0.0)
            else:
                raw.append(signal * (p.vol_target / n) / vol)
        gross = math.fsum(raw)
        scale = p.max_leverage / gross if gross > p.max_leverage else 1.0
        return TargetWeights({s: float(w * scale) for s, w in zip(symbols, raw, strict=True)})
