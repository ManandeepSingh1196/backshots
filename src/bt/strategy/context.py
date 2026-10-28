"""The strategy's only window on the world: read-only history, pinned to the bar just closed."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, cast

import numpy as np

from bt.core.clock import ensure_utc
from bt.core.errors import EngineError, InvariantError, LookAheadError, StrategyError
from bt.core.events import EventLog, LogRecord, LogValue
from bt.core.types import FundingEvent, Instrument, PortfolioView, PositionView


class HistoryView(Protocol):
    """Numpy arrays, flags.writeable = False, oldest first."""

    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray


class Context(Protocol):
    """What a strategy may see. Nothing here reaches past `now`."""

    @property
    def now(self) -> datetime:
        """Timestamp of the bar just closed."""
        ...

    @property
    def equity(self) -> float: ...

    @property
    def cash(self) -> float: ...

    def history(self, symbol: str, n: int) -> HistoryView:
        """Last n closed bars, ending at now."""
        ...

    def funding_history(self, symbol: str, n: int) -> Sequence[FundingEvent]: ...

    def position(self, symbol: str) -> PositionView: ...

    def instrument(self, symbol: str) -> Instrument: ...

    def symbols(self) -> Sequence[str]: ...

    def log(self, msg: str, **fields: object) -> None: ...


def _frozen_copy(x: object, dtype: str | type[np.float64]) -> np.ndarray:
    arr = np.array(x, dtype=dtype, copy=True)
    arr.flags.writeable = False
    return arr


def _plain(value: object) -> object:
    """Convert a numpy scalar to the equivalent Python scalar; pass anything else through."""
    if isinstance(value, np.generic):
        return cast("object", value.item())
    return value


@dataclass(slots=True, eq=False)
class BarHistory:
    """Concrete HistoryView.

    Invariants: every array is 1-D, the same length, owns its data (base is None) and is
    read-only; ts is datetime64[us] and strictly increasing; the others are float64.
    Not frozen (the protocol declares writable attributes), but every history() call builds
    a fresh object from fresh copies, so reassigning a field affects only the caller.
    """

    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __post_init__(self) -> None:
        self.ts = _frozen_copy(self.ts, "datetime64[us]")
        self.open = _frozen_copy(self.open, np.float64)
        self.high = _frozen_copy(self.high, np.float64)
        self.low = _frozen_copy(self.low, np.float64)
        self.close = _frozen_copy(self.close, np.float64)
        self.volume = _frozen_copy(self.volume, np.float64)
        arrays = (self.ts, self.open, self.high, self.low, self.close, self.volume)
        if any(a.ndim != 1 for a in arrays):
            raise InvariantError("BarHistory: every array must be 1-D")
        if len({a.shape[0] for a in arrays}) != 1:
            raise InvariantError("BarHistory: arrays must all have the same length")
        if np.isnat(self.ts).any():
            raise InvariantError("BarHistory: ts contains NaT")
        if len(self.ts) > 1 and not bool((np.diff(self.ts) > np.timedelta64(0, "us")).all()):
            raise InvariantError("BarHistory: ts must be strictly increasing")

    def __len__(self) -> int:
        return int(self.ts.shape[0])

    @classmethod
    def empty(cls) -> "BarHistory":
        return cls(*(np.empty(0, dtype=np.float64) for _ in range(6)))


class HistorySource(Protocol):
    """What StepContext reads history from; DataFeed satisfies it."""

    def history(self, symbol: str, end: datetime, n: int) -> HistoryView: ...

    def funding_history(self, symbol: str, end: datetime, n: int) -> Sequence[FundingEvent]: ...


class StepContext:
    """The engine's Context implementation; one instance per step.

    Invariants: `now` is UTC and fixed at construction; every history read is bounded by
    `now` and re-checked here (LookAheadError on any later ts); history arrays handed out are
    fresh, owned, read-only copies; no method takes a timestamp or reads the clock.
    """

    __slots__ = ("_instruments", "_log", "_now", "_portfolio", "_source")

    def __init__(
        self,
        now: datetime,
        source: HistorySource,
        portfolio: PortfolioView,
        instruments: Sequence[Instrument],
        log: EventLog,
    ) -> None:
        self._now = ensure_utc(now, "StepContext.now")
        if not instruments:
            raise EngineError("StepContext: instruments must be non-empty")
        mapping: dict[str, Instrument] = {}
        for inst in instruments:
            if inst.symbol in mapping:
                raise EngineError(f"StepContext: duplicate instrument {inst.symbol!r}")
            mapping[inst.symbol] = inst
        self._instruments = mapping
        self._source = source
        self._portfolio = portfolio
        self._log = log

    @property
    def now(self) -> datetime:
        return self._now

    @property
    def equity(self) -> float:
        return self._portfolio.equity

    @property
    def cash(self) -> float:
        return self._portfolio.cash

    def _check_args(self, symbol: str, n: int) -> None:
        if symbol not in self._instruments:
            raise StrategyError(f"unknown symbol {symbol!r}")
        if isinstance(n, bool) or not isinstance(n, int):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise StrategyError(f"n must be an int, got {type(n).__name__}")
        if n < 0:
            raise StrategyError(f"n must be >= 0, got {n}")

    def history(self, symbol: str, n: int) -> HistoryView:
        self._check_args(symbol, n)
        if n == 0:
            return BarHistory.empty()
        raw = self._source.history(symbol, self._now, n)
        ts = np.asarray(raw.ts).astype("datetime64[us]")
        now64 = np.datetime64(self._now.replace(tzinfo=None), "us")
        if ts.size and bool((ts > now64).any()):
            raise LookAheadError(
                f"history({symbol!r}) returned ts after now={self._now.isoformat()}"
            )
        if ts.shape[0] > n:
            raise EngineError(f"history({symbol!r}) returned {ts.shape[0]} rows for n={n}")
        arrays = (raw.ts, raw.open, raw.high, raw.low, raw.close, raw.volume)
        if any(bool(np.asarray(a).flags.writeable) for a in arrays):
            raise EngineError(f"history({symbol!r}) returned a writeable array")
        return BarHistory(raw.ts, raw.open, raw.high, raw.low, raw.close, raw.volume)

    def funding_history(self, symbol: str, n: int) -> Sequence[FundingEvent]:
        self._check_args(symbol, n)
        if n == 0:
            return ()
        events = tuple(self._source.funding_history(symbol, self._now, n))
        if any(ev.ts > self._now for ev in events):
            raise LookAheadError(
                f"funding_history({symbol!r}) returned an event after now={self._now.isoformat()}"
            )
        if len(events) > n:
            raise EngineError(f"funding_history({symbol!r}) returned {len(events)} events, n={n}")
        return events

    def position(self, symbol: str) -> PositionView:
        if symbol not in self._instruments:
            raise StrategyError(f"unknown symbol {symbol!r}")
        return self._portfolio.position(symbol)

    def instrument(self, symbol: str) -> Instrument:
        try:
            return self._instruments[symbol]
        except KeyError:
            raise StrategyError(f"unknown symbol {symbol!r}") from None

    def symbols(self) -> Sequence[str]:
        return tuple(sorted(self._instruments))

    def log(self, msg: str, **fields: object) -> None:
        clean: dict[str, LogValue] = {}
        for key, value in fields.items():
            item = _plain(value)
            if item is not None and not isinstance(item, str | int | float | bool):
                raise StrategyError(f"log field {key!r} has unsupported type {type(item).__name__}")
            clean[key] = item
        try:
            record = LogRecord(self._now, msg, tuple(sorted(clean.items())))
        except InvariantError as exc:
            raise StrategyError(f"bad log record: {exc}") from exc
        self._log.record_message(record)
