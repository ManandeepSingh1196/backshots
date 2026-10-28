from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from bt.core.errors import (
    EngineError,
    InvariantError,
    LookAheadError,
    StrategyError,
    TimezoneError,
)
from bt.core.events import EventLog, LogRecord
from bt.core.types import FundingEvent, Instrument, InstrumentKind, PortfolioView, PositionView
from bt.strategy.context import BarHistory, Context, HistorySource, HistoryView, StepContext

D1 = datetime(2024, 1, 1, tzinfo=UTC)
NOW = datetime(2024, 1, 3, tzinfo=UTC)
CLOSES = [10.0, 11.0, 12.0, 13.0, 14.0]


def _hist(closes: Sequence[float], start: int = 0) -> BarHistory:
    c = np.array(closes, dtype=np.float64)
    ts = np.array(
        [
            np.datetime64((D1 + timedelta(days=start + i)).replace(tzinfo=None), "us")
            for i in range(len(c))
        ]
    )
    return BarHistory(ts, c - 0.5, c + 1, c - 1, c, np.full(len(c), 100.0))


class FakeSource:
    def __init__(self) -> None:
        self.calls = 0
        self.all = _hist(CLOSES)

    def history(self, symbol: str, end: datetime, n: int) -> HistoryView:
        self.calls += 1
        e = np.datetime64(end.replace(tzinfo=None), "us")
        idx = np.nonzero(self.all.ts <= e)[0][-n:]
        a = self.all
        return BarHistory(
            a.ts[idx], a.open[idx], a.high[idx], a.low[idx], a.close[idx], a.volume[idx]
        )

    def funding_history(self, symbol: str, end: datetime, n: int) -> Sequence[FundingEvent]:
        return [FundingEvent(symbol, D1, 0.0001)]


class FakePortfolio:
    ts = None
    cash = 1000.0
    available_cash = 1000.0
    equity = 1234.5

    def position(self, symbol: str) -> PositionView:
        return PositionView(symbol, 0.0, 0.0, 0.0, 0.0)

    def positions(self) -> Mapping[str, PositionView]:
        return {}


def _inst(symbol: str = "AAAUSDT") -> Instrument:
    return Instrument(
        symbol,
        symbol,
        InstrumentKind.SPOT,
        symbol[:3],
        "USDT",
        0.001,
        0.001,
        5.0,
        0.01,
        0.001,
        0.001,
    )


def _ctx(
    source: HistorySource | None = None,
    symbols: Sequence[str] = ("AAAUSDT",),
    log: EventLog | None = None,
) -> StepContext:
    return StepContext(
        NOW, source or FakeSource(), FakePortfolio(), [_inst(s) for s in symbols], log or EventLog()
    )


def test_history_last_n() -> None:
    h = _ctx().history("AAAUSDT", 2)
    assert h.close.tolist() == pytest.approx([11.0, 12.0], abs=1e-12)
    expected = np.array(["2024-01-02", "2024-01-03"], dtype="datetime64[us]")
    assert h.ts.dtype == np.dtype("datetime64[us]")
    assert (h.ts == expected).all()


def test_history_more_than_exists_and_zero() -> None:
    src = FakeSource()
    ctx = _ctx(src)
    assert len(ctx.history("AAAUSDT", 100).close) == 3
    before = src.calls
    assert len(ctx.history("AAAUSDT", 0).close) == 0
    assert src.calls == before


def test_arrays_read_only_and_owned() -> None:
    h = _ctx().history("AAAUSDT", 2)
    with pytest.raises(ValueError, match="read-only"):
        h.close[0] = 1.0
    for arr in (h.ts, h.open, h.high, h.low, h.close, h.volume):
        assert arr.base is None
        assert not arr.flags.writeable


def test_mutating_returned_array_does_not_affect_later_calls() -> None:
    ctx = _ctx()
    h = ctx.history("AAAUSDT", 2)
    h.close.flags.writeable = True
    h.close[0] = -5.0
    assert ctx.history("AAAUSDT", 2).close[0] == pytest.approx(11.0, abs=1e-12)


def test_rebuilt_even_if_source_returns_same_object() -> None:
    shared = _hist(CLOSES[:3])

    class Same:
        def history(self, symbol: str, end: datetime, n: int) -> HistoryView:
            return shared

        def funding_history(self, symbol: str, end: datetime, n: int) -> Sequence[FundingEvent]:
            return ()

    h = _ctx(Same()).history("AAAUSDT", 3)
    assert h is not shared
    assert h.close is not shared.close
    assert h.close.base is None


class LeakySource(FakeSource):
    def history(self, symbol: str, end: datetime, n: int) -> HistoryView:
        return _hist(CLOSES[:4])

    def funding_history(self, symbol: str, end: datetime, n: int) -> Sequence[FundingEvent]:
        return [FundingEvent(symbol, datetime(2024, 1, 3, 8, tzinfo=UTC), 0.0001)]


def test_leaky_sources_raise_lookahead() -> None:
    ctx = _ctx(LeakySource())
    with pytest.raises(LookAheadError):
        ctx.history("AAAUSDT", 10)
    with pytest.raises(LookAheadError):
        ctx.funding_history("AAAUSDT", 10)


def test_ts_exactly_now_is_allowed() -> None:
    h = _ctx().history("AAAUSDT", 1)
    assert h.ts[-1] == np.datetime64("2024-01-03", "us")


def test_too_many_rows_engine_error_but_leak_checked_first() -> None:
    class Rows:
        def __init__(self, k: int) -> None:
            self.k = k

        def history(self, symbol: str, end: datetime, n: int) -> HistoryView:
            return _hist(CLOSES[: self.k])

        def funding_history(self, symbol: str, end: datetime, n: int) -> Sequence[FundingEvent]:
            return [FundingEvent(symbol, D1, 0.0)] * 3

    with pytest.raises(EngineError):
        _ctx(Rows(3)).history("AAAUSDT", 2)
    with pytest.raises(LookAheadError):
        _ctx(Rows(4)).history("AAAUSDT", 2)  # 4 rows, last after now, also too many
    with pytest.raises(EngineError):
        _ctx(Rows(3)).funding_history("AAAUSDT", 2)


def test_writeable_result_engine_error() -> None:
    @dataclass
    class Loose:
        ts: np.ndarray
        open: np.ndarray
        high: np.ndarray
        low: np.ndarray
        close: np.ndarray
        volume: np.ndarray

    class Src:
        def history(self, symbol: str, end: datetime, n: int) -> HistoryView:
            h = _hist(CLOSES[:2])
            return Loose(
                h.ts.copy(),
                h.open.copy(),
                h.high.copy(),
                h.low.copy(),
                h.close.copy(),
                h.volume.copy(),
            )

        def funding_history(self, symbol: str, end: datetime, n: int) -> Sequence[FundingEvent]:
            return ()

    with pytest.raises(EngineError):
        _ctx(Src()).history("AAAUSDT", 2)


def test_funding_history_ok_and_zero() -> None:
    ctx = _ctx()
    assert ctx.funding_history("AAAUSDT", 0) == ()
    out = ctx.funding_history("AAAUSDT", 5)
    assert isinstance(out, tuple)
    assert out == (FundingEvent("AAAUSDT", D1, 0.0001),)
    assert all(ev.ts <= NOW for ev in out)
    with pytest.raises(StrategyError):
        ctx.funding_history("ZZZUSDT", 1)
    with pytest.raises(StrategyError):
        ctx.funding_history("AAAUSDT", -1)


@pytest.mark.parametrize(("symbol", "n"), [("ZZZUSDT", 1), ("AAAUSDT", -1), ("AAAUSDT", True)])
def test_bad_history_args(symbol: str, n: int) -> None:
    with pytest.raises(StrategyError):
        _ctx().history(symbol, n)


def test_non_int_n() -> None:
    with pytest.raises(StrategyError):
        _ctx().history("AAAUSDT", 1.5)  # pyright: ignore[reportArgumentType]


def test_portfolio_passthrough_and_symbols() -> None:
    ctx = _ctx()
    assert ctx.equity == pytest.approx(1234.5, abs=1e-12)
    assert ctx.cash == pytest.approx(1000.0, abs=1e-12)
    assert ctx.position("AAAUSDT").qty == pytest.approx(0.0, abs=1e-12)
    assert ctx.symbols() == ("AAAUSDT",)
    assert _ctx(symbols=["BBBUSDT", "AAAUSDT"]).symbols() == ("AAAUSDT", "BBBUSDT")
    assert ctx.instrument("AAAUSDT").symbol == "AAAUSDT"
    with pytest.raises(StrategyError):
        ctx.position("ZZZUSDT")
    with pytest.raises(StrategyError):
        ctx.instrument("ZZZUSDT")
    assert ctx.now == NOW


def test_now_read_only_and_no_public_source() -> None:
    ctx = _ctx()
    with pytest.raises(AttributeError):
        ctx.now = D1  # pyright: ignore[reportAttributeAccessIssue]
    assert not hasattr(ctx, "source")
    assert not hasattr(ctx, "__dict__")


def test_log() -> None:
    log = EventLog()
    ctx = _ctx(log=log)
    ctx.log("signal", b=1, a=2)
    assert log.messages()[-1] == LogRecord(NOW, "signal", (("a", 2), ("b", 1)))
    ctx.log("x", v=np.float64(1.5), w=np.bool_(True), z=None)
    fields = dict(log.messages()[-1].fields)
    assert type(fields["v"]) is float
    assert fields["v"] == pytest.approx(1.5, abs=1e-12)
    assert fields["w"] is True
    with pytest.raises(StrategyError):
        ctx.log("")
    with pytest.raises(StrategyError):
        ctx.log("x", v=[1])
    with pytest.raises(StrategyError):
        ctx.log("x", v=float("nan"))
    assert len(log.messages()) == 2


def test_bar_history_invariants() -> None:
    a = np.arange(3, dtype=np.float64)
    ts = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[us]")
    with pytest.raises(InvariantError):
        BarHistory(ts, a[:2], a, a, a, a)
    with pytest.raises(InvariantError):
        BarHistory(ts, a.reshape(3, 1), a, a, a, a)
    with pytest.raises(InvariantError):
        BarHistory(ts[[0, 2, 1]], a, a, a, a, a)
    with pytest.raises(InvariantError):
        BarHistory(ts[[0, 0, 1]], a, a, a, a, a)
    ok = BarHistory(ts, a, a, a, a, a)
    assert len(ok) == 3
    a[0] = 99.0  # producer's array must not alias
    assert ok.close[0] == pytest.approx(0.0, abs=1e-12)


def test_bar_history_empty() -> None:
    e = BarHistory.empty()
    assert len(e) == 0
    assert e.close.dtype == np.float64
    assert e.ts.dtype == np.dtype("datetime64[us]")
    assert not e.close.flags.writeable


def test_step_context_construction_errors() -> None:
    naive = datetime(2024, 1, 3)  # noqa: DTZ001
    with pytest.raises(TimezoneError):
        StepContext(naive, FakeSource(), FakePortfolio(), [_inst()], EventLog())
    with pytest.raises(EngineError):
        _ctx(symbols=["AAAUSDT", "AAAUSDT"])
    with pytest.raises(EngineError):
        _ctx(symbols=[])


def test_protocol_assignments() -> None:
    ctx_typed: Context = _ctx()
    hv: HistoryView = BarHistory.empty()
    pv: PortfolioView = FakePortfolio()
    assert ctx_typed.now == NOW
    assert len(hv.close) == 0
    assert pv.equity > 0


class _Src:
    def __init__(self, hist: object = None, funding: Sequence[FundingEvent] = ()) -> None:
        self.hist = hist
        self.funding = funding

    def history(self, symbol: str, end: datetime, n: int) -> HistoryView:
        return self.hist  # pyright: ignore[reportReturnType]

    def funding_history(self, symbol: str, end: datetime, n: int) -> Sequence[FundingEvent]:
        return self.funding


def _ro(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    a.flags.writeable = False
    return a


def test_toctou_ts_property_read_once() -> None:
    day3 = _ro(np.array(["2024-01-03"], dtype="datetime64[us]"))
    day4 = _ro(np.array(["2024-01-04"], dtype="datetime64[us]"))
    one = _ro(np.array([1.0]))

    class Flip:
        def __init__(self) -> None:
            self.reads = 0

        @property
        def ts(self) -> np.ndarray:
            self.reads += 1
            return day3 if self.reads == 1 else day4

        open = high = low = close = volume = one

    out = _ctx(_Src(Flip())).history("AAAUSDT", 1)
    assert out.ts[0] == np.datetime64("2024-01-03", "us")


def test_non_datetime_ts_rejected() -> None:
    h = _hist(CLOSES[:2])

    @dataclass
    class Ints:
        ts: np.ndarray
        open: np.ndarray
        high: np.ndarray
        low: np.ndarray
        close: np.ndarray
        volume: np.ndarray

    bad = Ints(_ro(np.array([1, 2], dtype=np.int64)), h.open, h.high, h.low, h.close, h.volume)
    with pytest.raises(EngineError):
        _ctx(_Src(bad)).history("AAAUSDT", 2)


def test_lossy_ts_cast_rejected() -> None:
    h = _hist(CLOSES[:1])

    @dataclass
    class Ns:
        ts: np.ndarray
        open: np.ndarray
        high: np.ndarray
        low: np.ndarray
        close: np.ndarray
        volume: np.ndarray

    ts = _ro(np.array([np.datetime64("2024-01-03T00:00:00.000000999", "ns")]))
    bad = Ns(ts, h.open, h.high, h.low, h.close, h.volume)
    with pytest.raises(EngineError):
        _ctx(_Src(bad)).history("AAAUSDT", 1)
    ok = Ns(_ro(h.ts.astype("datetime64[ns]")), h.open, h.high, h.low, h.close, h.volume)
    assert len(_ctx(_Src(ok)).history("AAAUSDT", 1).ts) == 1


def test_funding_wrong_symbol_or_unsorted() -> None:
    other = [FundingEvent("BBBUSDT", D1, 0.0)]
    with pytest.raises(EngineError):
        _ctx(_Src(funding=other)).funding_history("AAAUSDT", 5)
    later = FundingEvent("AAAUSDT", D1 + timedelta(hours=8), 0.0)
    with pytest.raises(EngineError):
        _ctx(_Src(funding=[later, FundingEvent("AAAUSDT", D1, 0.0)])).funding_history("AAAUSDT", 5)


def test_unhashable_symbol_is_strategy_error() -> None:
    ctx = _ctx()
    bad: object = ["AAAUSDT"]
    for call in (ctx.history, ctx.funding_history):
        with pytest.raises(StrategyError):
            call(bad, 1)  # pyright: ignore[reportArgumentType]
    with pytest.raises(StrategyError):
        ctx.position(bad)  # pyright: ignore[reportArgumentType]
    with pytest.raises(StrategyError):
        ctx.instrument(bad)  # pyright: ignore[reportArgumentType]
