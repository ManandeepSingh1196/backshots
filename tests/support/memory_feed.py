"""In-memory DataFeed with the same look-ahead guard as ParquetFeed (funding: perps only)."""

from collections.abc import Iterator, Sequence
from datetime import datetime

import numpy as np

from bt.core.clock import ensure_utc
from bt.core.errors import DataError, LookAheadError
from bt.core.types import Bar, FundingEvent, Instrument, InstrumentKind, Step
from bt.strategy.context import BarHistory, HistoryView


class MemoryFeed:
    """DataFeed over in-memory bars. steps() is one-shot; history()/funding_history() raise
    LookAheadError unless `end` is strictly before the next undelivered step. No coverage check."""

    def __init__(
        self,
        instruments: Sequence[Instrument],
        bars: Sequence[Bar],
        funding: Sequence[FundingEvent] = (),
    ) -> None:
        self._instruments = tuple(sorted(instruments, key=lambda i: i.symbol))
        symbols = {i.symbol for i in self._instruments}
        perps = {i.symbol for i in self._instruments if i.kind == InstrumentKind.PERP}
        by_sym: dict[str, dict[datetime, Bar]] = {s: {} for s in sorted(symbols)}
        for bar in bars:
            if bar.symbol not in symbols:
                raise DataError(f"bar for unknown symbol {bar.symbol!r}")
            if bar.ts in by_sym[bar.symbol]:
                raise DataError(f"duplicate bar {bar.symbol} at {bar.ts.isoformat()}")
            by_sym[bar.symbol][bar.ts] = bar
        self._bars = {s: [b[t] for t in sorted(b)] for s, b in by_sym.items()}
        self._funding = {
            s: tuple(
                sorted((e for e in funding if e.symbol == s and s in perps), key=lambda e: e.ts)
            )
            for s in sorted(symbols)
        }
        step_ts = sorted({b.ts for lst in self._bars.values() for b in lst})
        steps: list[Step] = []
        prev: datetime | None = None
        for ts in step_ts:
            chunk = [
                e
                for evs in self._funding.values()
                for e in evs
                if e.ts <= ts and (prev is None or e.ts > prev)
            ]
            steps.append(Step(ts, {s: by_sym[s][ts] for s in by_sym if ts in by_sym[s]}, chunk))
            prev = ts
        self._steps = tuple(steps)
        self._next_ts: datetime | None = step_ts[0] if step_ts else None
        self._started = False

    def instruments(self) -> Sequence[Instrument]:
        return self._instruments

    def steps(self) -> Iterator[Step]:
        if self._started:
            raise DataError("steps() may be called only once")
        self._started = True
        return self._iterate()

    def _iterate(self) -> Iterator[Step]:
        for i, step in enumerate(self._steps):
            self._next_ts = self._steps[i + 1].ts if i + 1 < len(self._steps) else None
            yield step

    def _check(self, kind: str, symbol: str, end: datetime, n: int) -> datetime:
        if symbol not in self._bars:
            raise DataError(f"{kind}: unknown symbol {symbol!r}")
        if isinstance(n, bool) or not isinstance(n, int):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise DataError(f"{kind}: n must be an int, got {type(n).__name__}")
        if n < 0:
            raise DataError(f"{kind}: n must be >= 0, got {n}")
        end = ensure_utc(end, "end")
        if self._next_ts is not None and end >= self._next_ts:
            raise LookAheadError(
                f"{kind}({symbol!r}): end {end.isoformat()} is not before the next "
                f"undelivered step {self._next_ts.isoformat()}"
            )
        return end

    def history(self, symbol: str, end: datetime, n: int) -> HistoryView:
        end = self._check("history", symbol, end, n)
        rows = [b for b in self._bars[symbol] if b.ts <= end]
        rows = rows[len(rows) - n :] if n < len(rows) else rows
        return BarHistory(
            np.array([b.ts.replace(tzinfo=None) for b in rows], dtype="datetime64[us]"),
            np.array([b.open for b in rows], dtype=np.float64),
            np.array([b.high for b in rows], dtype=np.float64),
            np.array([b.low for b in rows], dtype=np.float64),
            np.array([b.close for b in rows], dtype=np.float64),
            np.array([b.volume for b in rows], dtype=np.float64),
        )

    def funding_history(self, symbol: str, end: datetime, n: int) -> Sequence[FundingEvent]:
        end = self._check("funding_history", symbol, end, n)
        evs = [e for e in self._funding[symbol] if e.ts <= end]
        return tuple(evs[len(evs) - n :] if n < len(evs) else evs)
