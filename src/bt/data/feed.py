"""The backtest DataFeed: reads a validated catalog and never serves a step it has not delivered."""

from bisect import bisect_left
from collections.abc import Iterator, Sequence
from datetime import datetime
from typing import Protocol

import numpy as np
import polars as pl

from bt.core.clock import ensure_utc
from bt.core.errors import DataError, InvariantError, LookAheadError, TimezoneError
from bt.core.types import Bar, FundingEvent, Instrument, InstrumentKind, Step
from bt.data.catalog import Catalog
from bt.data.validate import TIMEFRAME_DELTA, Timeframe
from bt.strategy.context import BarHistory, HistoryView


class DataFeed(Protocol):
    """Swappable data edge (INTERFACES section 5)."""

    def instruments(self) -> Sequence[Instrument]: ...

    def steps(self) -> Iterator[Step]:
        """Backtest: whole window; live: latest closed step."""
        ...

    def history(self, symbol: str, end: datetime, n: int) -> HistoryView: ...

    def funding_history(self, symbol: str, end: datetime, n: int) -> Sequence[FundingEvent]: ...


class _Series:
    """One symbol's kept bars as read-only numpy columns (window plus warmup)."""

    __slots__ = ("close", "high", "low", "open", "ts", "volume")

    def __init__(self, df: pl.DataFrame) -> None:
        self.ts = df["ts"].dt.epoch("us").to_numpy().astype("datetime64[us]")
        self.open = df["open"].to_numpy().astype(np.float64)
        self.high = df["high"].to_numpy().astype(np.float64)
        self.low = df["low"].to_numpy().astype(np.float64)
        self.close = df["close"].to_numpy().astype(np.float64)
        self.volume = df["volume"].to_numpy().astype(np.float64)


class ParquetFeed:
    """DataFeed over a catalog for one backtest window.

    Invariants: everything is loaded at construction and rows with ts >= end are never kept;
    steps() can be iterated once; history() and funding_history() raise LookAheadError unless
    `end` is strictly before the next undelivered step (no limit once exhausted), and return
    fresh copies, never views of feed storage; the step sequence is deterministic.
    """

    def __init__(
        self,
        catalog: Catalog,
        instruments: Sequence[Instrument],
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        warmup_bars: int = 0,
    ) -> None:
        start = ensure_utc(start, "start")
        end = ensure_utc(end, "end")
        if start >= end:
            raise DataError(f"start {start.isoformat()} must be before end {end.isoformat()}")
        if timeframe not in TIMEFRAME_DELTA:
            raise DataError(f"unsupported timeframe {timeframe!r}")
        if isinstance(warmup_bars, bool) or not isinstance(warmup_bars, int):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise DataError(f"warmup_bars must be an int, got {type(warmup_bars).__name__}")
        if warmup_bars < 0:
            raise DataError(f"warmup_bars must be >= 0, got {warmup_bars}")
        if not instruments:
            raise DataError("instruments must be non-empty")
        symbols = [i.symbol for i in instruments]
        if len(set(symbols)) != len(symbols):
            raise DataError(f"duplicate instrument symbols: {sorted(symbols)}")

        self._instruments = tuple(sorted(instruments, key=lambda i: i.symbol))
        self._series: dict[str, _Series] = {}
        self._funding: dict[str, tuple[FundingEvent, ...]] = {}
        self._funding_ts: dict[str, np.ndarray] = {}
        window_bars: dict[str, dict[datetime, Bar]] = {}
        for inst in self._instruments:
            df = catalog.read_klines(inst.kind.value, timeframe, inst.venue_symbol)
            kept, window = self._load_bars(inst, df, start, end, warmup_bars)
            self._series[inst.symbol] = _Series(kept)
            window_bars[inst.symbol] = window
            events = self._load_funding(catalog, inst, end)
            self._funding[inst.symbol] = events
            self._funding_ts[inst.symbol] = np.array(
                [ev.ts.replace(tzinfo=None) for ev in events], dtype="datetime64[us]"
            )

        step_ts = sorted({ts for bars in window_bars.values() for ts in bars})
        for sym, bars in window_bars.items():
            if not bars:
                raise DataError(f"{sym} has no bars in [{start.isoformat()}, {end.isoformat()})")
        for label, ts in (("first", step_ts[0]), ("last", step_ts[-1])):
            for sym in sorted(window_bars):
                if ts not in window_bars[sym]:
                    raise DataError(
                        f"{sym} does not cover the window: no bar at the {label} step "
                        f"{ts.isoformat()}"
                    )

        self._steps = self._build_steps(step_ts, window_bars, start)
        self._next_ts: datetime | None = step_ts[0]  # None means +infinity (exhausted)
        self._started = False

    @staticmethod
    def _load_bars(
        inst: Instrument, df: pl.DataFrame, start: datetime, end: datetime, warmup: int
    ) -> tuple[pl.DataFrame, dict[datetime, Bar]]:
        ts_list = [ensure_utc(t, "kline ts") for t in df["ts"].to_list()]
        hi = bisect_left(ts_list, end)  # rows at or after end are dropped here
        lo = bisect_left(ts_list, start)
        first = max(0, lo - warmup)
        kept = df.slice(first, hi - first)
        window: dict[datetime, Bar] = {}
        for i in range(first, hi):
            row = df.row(i, named=True)
            try:
                bar = Bar(
                    inst.symbol,
                    ts_list[i],
                    row["open"],
                    row["high"],
                    row["low"],
                    row["close"],
                    row["volume"],
                    row["quote_volume"],
                )
            except (InvariantError, TimezoneError) as exc:
                raise DataError(
                    f"bad bar for {inst.symbol} at {ts_list[i].isoformat()}: {exc}"
                ) from exc
            if i >= lo:
                window[ts_list[i]] = bar
        return kept, window

    @staticmethod
    def _load_funding(
        catalog: Catalog, inst: Instrument, end: datetime
    ) -> tuple[FundingEvent, ...]:
        if inst.kind is not InstrumentKind.PERP:
            return ()
        df = catalog.read_funding(inst.venue_symbol)
        events: list[FundingEvent] = []
        for ts, rate in zip(df["ts"].to_list(), df["rate"].to_list(), strict=True):
            ts = ensure_utc(ts, "funding ts")
            if ts >= end:
                continue
            try:
                events.append(FundingEvent(inst.symbol, ts, rate))
            except (InvariantError, TimezoneError) as exc:
                raise DataError(
                    f"bad funding event for {inst.symbol} at {ts.isoformat()}: {exc}"
                ) from exc
        return tuple(events)

    def _build_steps(
        self, step_ts: list[datetime], window_bars: dict[str, dict[datetime, Bar]], start: datetime
    ) -> tuple[Step, ...]:
        all_events = sorted(
            (ev for evs in self._funding.values() for ev in evs), key=lambda e: (e.ts, e.symbol)
        )
        steps: list[Step] = []
        cursor = 0
        prev: datetime | None = None
        for ts in step_ts:
            chunk: list[FundingEvent] = []
            # Events after the last step ts are never delivered: a known M6 timing question.
            while cursor < len(all_events) and all_events[cursor].ts <= ts:
                ev = all_events[cursor]
                cursor += 1
                if (ev.ts > prev) if prev is not None else (ev.ts >= start):
                    chunk.append(ev)
            bars = {s: window_bars[s][ts] for s in sorted(window_bars) if ts in window_bars[s]}
            steps.append(Step(ts, bars, chunk))
            prev = ts
        return tuple(steps)

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
        if not isinstance(symbol, str) or symbol not in self._series:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise DataError(f"{kind}: unknown symbol {symbol!r}")
        if isinstance(n, bool) or not isinstance(n, int):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise DataError(f"{kind}: n must be an int, got {type(n).__name__}")
        if n < 0:
            raise DataError(f"{kind}: n must be >= 0, got {n}")
        end = ensure_utc(end, "end")
        limit = self._next_ts
        if limit is not None and end >= limit:
            raise LookAheadError(
                f"{kind}({symbol!r}): end {end.isoformat()} is not before the next "
                f"undelivered step {limit.isoformat()}"
            )
        return end

    def history(self, symbol: str, end: datetime, n: int) -> HistoryView:
        end = self._check("history", symbol, end, n)
        s = self._series[symbol]
        hi = int(np.searchsorted(s.ts, np.datetime64(end.replace(tzinfo=None), "us"), "right"))
        lo = max(0, hi - n)
        return BarHistory(
            s.ts[lo:hi], s.open[lo:hi], s.high[lo:hi], s.low[lo:hi], s.close[lo:hi], s.volume[lo:hi]
        )

    def funding_history(self, symbol: str, end: datetime, n: int) -> Sequence[FundingEvent]:
        end = self._check("funding_history", symbol, end, n)
        ts = self._funding_ts[symbol]
        hi = int(np.searchsorted(ts, np.datetime64(end.replace(tzinfo=None), "us"), "right"))
        lo = max(0, hi - n)
        return self._funding[symbol][lo:hi]
