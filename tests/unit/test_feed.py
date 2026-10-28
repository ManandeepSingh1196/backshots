"""Tests for bt.data.feed (ParquetFeed and its look-ahead guard)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from bt.core.errors import DataError, InvariantError, LookAheadError, TimezoneError
from bt.core.types import Instrument, InstrumentKind
from bt.data.catalog import Catalog
from bt.data.feed import DataFeed, ParquetFeed
from bt.data.validate import FUNDING_SCHEMA, KLINE_SCHEMA, Market, Timeframe

T0 = datetime(2024, 3, 10, tzinfo=UTC)
NAIVE = datetime(2024, 1, 5)  # noqa: DTZ001


def D(day: int, hour: int = 0) -> datetime:  # noqa: N802
    return datetime(2024, 1, day, hour, tzinfo=UTC)


def inst(venue: str, kind: InstrumentKind = InstrumentKind.SPOT) -> Instrument:
    symbol = venue if kind is InstrumentKind.SPOT else f"{venue}-PERP"
    lev = 1.0 if kind is InstrumentKind.SPOT else 5.0
    return Instrument(
        symbol, venue, kind, venue[:3], "USDT", 0.0, 0.001, 0.0, 0.01, 0.001, 0.001, lev
    )


AAA = inst("AAAUSDT")
BBB = inst("BBBUSDT")
AAA_PERP = inst("AAAUSDT", InstrumentKind.PERP)


def klines(first_close: float, days: range, skip: tuple[int, ...] = ()) -> pl.DataFrame:
    days_kept = [d for d in days if d not in skip]
    closes = [first_close + (d - days.start) for d in days_kept]
    n = len(days_kept)
    return pl.DataFrame(
        {
            "ts": [D(d) for d in days_kept],
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [10.0] * n,
            "quote_volume": [1000.0] * n,
            "trades": [5] * n,
        },
        schema=dict(KLINE_SCHEMA),
    )


def funding(n: int = 15) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": [D(1) + timedelta(hours=8 * i) for i in range(n)],
            "rate": [0.0001] * n,
            "mark_price": [None] * n,
        },
        schema=dict(FUNDING_SCHEMA),
    )


def put(cat: Catalog, market: Market, venue: str, df: pl.DataFrame) -> None:
    cat.write_klines(market, "1d", venue, df, (), {}, T0)


@pytest.fixture
def cat(tmp_path: Path) -> Catalog:
    c = Catalog(tmp_path)
    put(c, "spot", "AAAUSDT", klines(100.0, range(1, 7)))
    put(c, "spot", "BBBUSDT", klines(20.0, range(1, 7), skip=(3,)))
    return c


def make(cat: Catalog, start: int = 2, end: int = 6, warmup: int = 2) -> ParquetFeed:
    return ParquetFeed(cat, [AAA, BBB], "1d", D(start), D(end), warmup_bars=warmup)


def test_steps_and_missing_bar(cat: Catalog) -> None:
    steps = list(make(cat).steps())
    assert [s.ts for s in steps] == [D(2), D(3), D(4), D(5)]
    assert list(steps[1].bars) == ["AAAUSDT"]
    for s in (steps[0], steps[2], steps[3]):
        assert list(s.bars) == ["AAAUSDT", "BBBUSDT"]
    assert all(s.funding == () for s in steps)


def test_keys_sorted_regardless_of_input_order(cat: Catalog) -> None:
    feed = ParquetFeed(cat, [BBB, AAA], "1d", D(2), D(6))
    assert [i.symbol for i in feed.instruments()] == ["AAAUSDT", "BBBUSDT"]
    assert list(next(feed.steps()).bars) == ["AAAUSDT", "BBBUSDT"]


def test_guard_before_iterating(cat: Catalog) -> None:
    feed = make(cat)
    assert list(feed.history("AAAUSDT", D(1), 5).close) == [100.0]
    with pytest.raises(LookAheadError):
        feed.history("AAAUSDT", D(2), 5)


def test_guard_before_iterating_applies_even_after_steps_called(cat: Catalog) -> None:
    feed = make(cat)
    it = feed.steps()  # not advanced yet
    with pytest.raises(LookAheadError):
        feed.history("AAAUSDT", D(2), 5)
    next(it)


def test_guard_after_first_step(cat: Catalog) -> None:
    feed = make(cat)
    it = feed.steps()
    next(it)
    assert list(feed.history("AAAUSDT", D(2), 5).close) == [100.0, 101.0]
    with pytest.raises(LookAheadError, match="2024-01-03"):
        feed.history("AAAUSDT", D(3), 1)
    assert list(feed.history("BBBUSDT", D(2), 1).close) == [21.0]


def test_guard_boundary_is_exact(cat: Catalog) -> None:
    feed = make(cat)
    it = feed.steps()
    next(it)
    just_before = D(3) - timedelta(microseconds=1)
    assert list(feed.history("AAAUSDT", just_before, 1).close) == [101.0]
    with pytest.raises(LookAheadError):
        feed.history("AAAUSDT", D(3), 1)
    with pytest.raises(LookAheadError):
        feed.history("AAAUSDT", D(3) + timedelta(microseconds=1), 1)


def test_guard_message_has_end_and_next(cat: Catalog) -> None:
    feed = make(cat)
    next(feed.steps())
    with pytest.raises(LookAheadError) as e:
        feed.history("AAAUSDT", D(4), 1)
    assert D(4).isoformat() in str(e.value)
    assert D(3).isoformat() in str(e.value)


def test_guard_advances_with_missing_bar_step(cat: Catalog) -> None:
    feed = make(cat)
    it = feed.steps()
    next(it)
    next(it)  # step 01-03, BBB has no bar
    assert list(feed.history("BBBUSDT", D(3), 2).close) == [20.0, 21.0]
    with pytest.raises(LookAheadError):
        feed.history("BBBUSDT", D(4), 1)


def test_guard_on_every_step(cat: Catalog) -> None:
    feed = make(cat)
    steps = list(feed.steps())  # exhausted: no limit
    del steps
    feed = make(cat)
    it = feed.steps()
    all_ts = [D(2), D(3), D(4), D(5)]
    for i, step in enumerate(it):
        assert step.ts == all_ts[i]
        feed.history("AAAUSDT", step.ts, 1)
        if i + 1 < len(all_ts):
            with pytest.raises(LookAheadError):
                feed.history("AAAUSDT", all_ts[i + 1], 1)


def test_after_exhaustion_no_limit_and_end_row_never_loaded(cat: Catalog) -> None:
    feed = make(cat)
    list(feed.steps())
    h = feed.history("AAAUSDT", datetime(2030, 1, 1, tzinfo=UTC), 100)
    assert len(h.close) == 5  # 01-01 .. 01-05; 01-06 (== end) was never loaded
    assert h.close[-1] == pytest.approx(104.0, abs=1e-12)


def test_no_rows_at_or_after_end_loaded_even_when_not_exhausted(cat: Catalog) -> None:
    feed = make(cat, end=5)
    list(feed.steps())
    h = feed.history("AAAUSDT", datetime(2030, 1, 1, tzinfo=UTC), 100)
    assert h.close[-1] == pytest.approx(103.0, abs=1e-12)
    assert len(h.close) == 4


def test_history_arguments(cat: Catalog) -> None:
    feed = make(cat)
    list(feed.steps())
    end = D(5)
    assert len(feed.history("AAAUSDT", end, 0).close) == 0
    with pytest.raises(DataError):
        feed.history("AAAUSDT", end, -1)
    with pytest.raises(DataError):
        feed.history("AAAUSDT", end, True)
    with pytest.raises(DataError):
        feed.history("ZZZUSDT", end, 1)
    with pytest.raises(TimezoneError):
        feed.history("AAAUSDT", NAIVE, 1)
    with pytest.raises(DataError):
        feed.funding_history("AAAUSDT", end, -1)
    with pytest.raises(DataError):
        feed.funding_history("AAAUSDT", end, True)
    with pytest.raises(DataError):
        feed.funding_history("ZZZUSDT", end, 1)
    with pytest.raises(TimezoneError):
        feed.funding_history("AAAUSDT", NAIVE, 1)


def test_history_is_readonly_copy(cat: Catalog) -> None:
    feed = make(cat)
    list(feed.steps())
    h = feed.history("AAAUSDT", D(5), 3)
    assert not h.close.flags.writeable
    again = feed.history("AAAUSDT", D(5), 3)
    assert list(again.close) == [102.0, 103.0, 104.0]
    assert h.close.base is None


def test_steps_once_checked_at_call_time(cat: Catalog) -> None:
    feed = make(cat)
    feed.steps()  # never iterated
    with pytest.raises(DataError):
        feed.steps()


def test_coverage_first_step(cat: Catalog) -> None:
    with pytest.raises(DataError, match="BBBUSDT"):
        make(cat, start=3)


def test_coverage_last_step(tmp_path: Path) -> None:
    c = Catalog(tmp_path)
    put(c, "spot", "AAAUSDT", klines(100.0, range(1, 7), skip=(5,)))
    put(c, "spot", "BBBUSDT", klines(20.0, range(1, 7)))
    with pytest.raises(DataError, match="AAAUSDT does not cover the window"):
        make(c)


def test_symbol_without_window_rows(tmp_path: Path) -> None:
    c = Catalog(tmp_path)
    put(c, "spot", "AAAUSDT", klines(100.0, range(1, 7)))
    put(c, "spot", "BBBUSDT", klines(20.0, range(1, 3)))
    with pytest.raises(DataError, match="BBBUSDT"):
        make(c, start=4)


def test_bad_arguments(cat: Catalog) -> None:
    with pytest.raises(DataError):
        make(cat, start=5, end=5)
    with pytest.raises(DataError):
        make(cat, start=6, end=2)
    with pytest.raises(DataError):
        make(cat, warmup=-1)
    with pytest.raises(DataError):
        ParquetFeed(cat, [AAA], "1h", D(2), D(6))  # pyright: ignore[reportArgumentType]
    with pytest.raises(DataError):
        ParquetFeed(cat, [AAA, AAA], "1d", D(2), D(6))
    with pytest.raises(DataError):
        ParquetFeed(cat, [], "1d", D(2), D(6))
    with pytest.raises(DataError):
        ParquetFeed(cat, [AAA], "1d", D(2), D(6), warmup_bars=True)
    with pytest.raises(TimezoneError):
        ParquetFeed(cat, [AAA], "1d", NAIVE, D(6))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DataError):
        ParquetFeed(Catalog(tmp_path), [AAA], "1d", D(2), D(6))


def test_bad_row_becomes_data_error(cat: Catalog) -> None:
    class Bad(Catalog):
        def read_klines(self, market: Market, timeframe: Timeframe, symbol: str) -> pl.DataFrame:
            df = super().read_klines(market, timeframe, symbol)
            return df.with_columns(low=pl.col("open") + 1.0)

    bad = Bad(cat.root)
    with pytest.raises(DataError) as e:
        ParquetFeed(bad, [AAA], "1d", D(2), D(6))
    assert isinstance(e.value.__cause__, InvariantError)
    assert "AAAUSDT" in str(e.value)


def test_bad_warmup_row_also_rejected(cat: Catalog) -> None:
    class Bad(Catalog):
        def read_klines(self, market: Market, timeframe: Timeframe, symbol: str) -> pl.DataFrame:
            df = super().read_klines(market, timeframe, symbol)
            return df.with_columns(
                low=pl.when(pl.col("ts") == D(1))
                .then(pl.col("open") + 1.0)
                .otherwise(pl.col("low"))
            )

    with pytest.raises(DataError):
        ParquetFeed(Bad(cat.root), [AAA], "1d", D(2), D(6), warmup_bars=1)


@pytest.fixture
def perp_cat(tmp_path: Path) -> Catalog:
    c = Catalog(tmp_path)
    put(c, "perp", "AAAUSDT", klines(100.0, range(1, 7)))
    c.write_funding("AAAUSDT", funding(), (), {}, T0)
    return c


def test_perp_funding_per_step(perp_cat: Catalog) -> None:
    feed = ParquetFeed(perp_cat, [AAA_PERP], "1d", D(2), D(6))
    steps = list(feed.steps())
    assert [e.ts for e in steps[0].funding] == [D(2)]
    assert [e.ts for e in steps[1].funding] == [D(2, 8), D(2, 16), D(3)]
    assert [e.ts for e in steps[3].funding] == [D(4, 8), D(4, 16), D(5)]
    for s in steps:
        assert all(e.symbol == "AAAUSDT-PERP" for e in s.funding)
        assert all(b.symbol == "AAAUSDT-PERP" for b in s.bars.values())
        assert list(s.bars) == ["AAAUSDT-PERP"]
    assert steps[0].funding[0].rate == pytest.approx(0.0001, abs=1e-12)


def test_perp_funding_history_guarded(perp_cat: Catalog) -> None:
    feed = ParquetFeed(perp_cat, [AAA_PERP], "1d", D(2), D(6))
    with pytest.raises(LookAheadError):
        feed.funding_history("AAAUSDT-PERP", D(2), 10)
    it = feed.steps()
    next(it)
    got = feed.funding_history("AAAUSDT-PERP", D(2), 10)
    assert len(got) == 4
    assert got[-1].ts == D(2)
    assert len(feed.funding_history("AAAUSDT-PERP", D(2), 2)) == 2
    assert feed.funding_history("AAAUSDT-PERP", D(2), 0) == ()
    assert len(feed.funding_history("AAAUSDT-PERP", D(2, 8), 10)) == 5  # still before step 01-03
    with pytest.raises(LookAheadError):
        feed.funding_history("AAAUSDT-PERP", D(3), 10)


def test_funding_at_or_after_end_not_kept(perp_cat: Catalog) -> None:
    feed = ParquetFeed(perp_cat, [AAA_PERP], "1d", D(2), D(4))
    list(feed.steps())
    got = feed.funding_history("AAAUSDT-PERP", datetime(2030, 1, 1, tzinfo=UTC), 100)
    assert max(e.ts for e in got) == D(3, 16)
    assert len(got) == 9  # 01-01T00:00 .. 01-03T16:00


def test_funding_before_start_not_delivered_in_steps(perp_cat: Catalog) -> None:
    feed = ParquetFeed(perp_cat, [AAA_PERP], "1d", D(3), D(5))
    first = next(feed.steps())
    assert [e.ts for e in first.funding] == [D(3)]


def test_spot_funding_history_empty(cat: Catalog) -> None:
    feed = make(cat)
    next(feed.steps())
    assert feed.funding_history("AAAUSDT", D(2), 10) == ()
    with pytest.raises(LookAheadError):
        feed.funding_history("AAAUSDT", D(3), 10)


def test_deterministic(perp_cat: Catalog) -> None:
    a = list(ParquetFeed(perp_cat, [AAA_PERP], "1d", D(2), D(6), 2).steps())
    b = list(ParquetFeed(perp_cat, [AAA_PERP], "1d", D(2), D(6), 2).steps())
    assert a == b


def test_satisfies_protocol(cat: Catalog) -> None:
    f: DataFeed = make(cat)
    assert len(f.instruments()) == 2
