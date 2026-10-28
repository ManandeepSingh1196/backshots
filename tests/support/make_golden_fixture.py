"""Writes the committed golden_m2 catalog.

Run once: `uv run python -m tests.support.make_golden_fixture`.

The literal tables below are the single source of truth; the golden test asserts that the
committed Parquet files read back equal to them.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import polars as pl

from bt.core.types import Instrument, InstrumentKind
from bt.data.catalog import Catalog
from bt.data.validate import KLINE_SCHEMA

FIXTURE_DIR: Final = Path(__file__).resolve().parents[1] / "fixtures" / "golden_m2"
INGESTED_AT: Final = datetime(2024, 2, 1, tzinfo=UTC)
D1: Final = datetime(2024, 1, 1, tzinfo=UTC)

# (day number, open, high, low, close); volume 10, quote_volume 10*close, trades 1.
AAA_BARS: Final[tuple[tuple[int, float, float, float, float], ...]] = (
    (1, 100.0, 105.0, 95.0, 100.0),
    (2, 102.0, 110.0, 101.0, 108.0),
    (3, 108.0, 112.0, 104.0, 110.0),
    (4, 111.0, 115.0, 109.0, 112.0),
    (5, 113.0, 114.0, 111.0, 114.0),
)
BBB_BARS: Final[tuple[tuple[int, float, float, float, float], ...]] = (
    (1, 20.0, 21.0, 19.0, 20.0),
    (2, 20.0, 22.0, 19.5, 21.0),
    (4, 22.0, 23.0, 21.0, 22.5),
    (5, 23.0, 24.0, 22.0, 24.0),
)


def _instrument(sym: str, min_qty: float) -> Instrument:
    return Instrument(
        symbol=f"{sym}USDT",
        venue_symbol=f"{sym}USDT",
        kind=InstrumentKind.SPOT,
        base=sym,
        quote="USDT",
        min_qty=min_qty,
        qty_step=min_qty,
        min_notional=5.0,
        price_tick=0.01,
        maker_fee=0.001,
        taker_fee=0.001,
    )


AAA: Final = _instrument("AAA", 0.001)
BBB: Final = _instrument("BBB", 0.1)
INSTRUMENTS: Final = (AAA, BBB)


def day(i: int) -> datetime:
    """Day i, 1-based: day(1) is 2024-01-01 UTC."""
    return D1 + timedelta(days=i - 1)


def frame(bars: tuple[tuple[int, float, float, float, float], ...]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": [day(b[0]) for b in bars],
            "open": [b[1] for b in bars],
            "high": [b[2] for b in bars],
            "low": [b[3] for b in bars],
            "close": [b[4] for b in bars],
            "volume": [10.0] * len(bars),
            "quote_volume": [10.0 * b[4] for b in bars],
            "trades": [1] * len(bars),
        },
        schema=dict(KLINE_SCHEMA),
    )


def write_fixture(root: Path) -> Catalog:
    cat = Catalog(root)
    cat.write_klines("spot", "1d", "AAAUSDT", frame(AAA_BARS), ("golden",), {}, INGESTED_AT)
    cat.write_klines("spot", "1d", "BBBUSDT", frame(BBB_BARS), ("golden",), {}, INGESTED_AT)
    cat.write_instruments(INSTRUMENTS, INGESTED_AT)
    return cat


if __name__ == "__main__":
    write_fixture(FIXTURE_DIR)
