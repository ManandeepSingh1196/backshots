"""Writes the committed crosscheck catalog (two spot symbols, 400 daily bars each).

Run once: `uv run python -m tests.support.make_crosscheck_fixture`.

The generator is the source of truth; the unit test regenerates and compares to the
committed files. Zero fees so an external cross-check needs no cost modelling.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

from bt.core.types import Instrument, InstrumentKind
from bt.data.catalog import Catalog
from bt.data.validate import KLINE_SCHEMA

CROSSCHECK_DIR: Final = Path(__file__).resolve().parents[1] / "fixtures" / "crosscheck"
INGESTED_AT: Final = datetime(2024, 3, 1, tzinfo=UTC)
START: Final = datetime(2023, 1, 1, tzinfo=UTC)
N_BARS: Final = 400
MAX_GAP: Final = 0.002

# symbol -> (seed, start price, sine period in bars, drift amplitude, noise sigma)
_SPECS: Final[dict[str, tuple[int, float, float, float, float]]] = {
    "BTCUSDT": (41, 20000.0, 70.0, 0.012, 0.012),
    "ETHUSDT": (42, 1500.0, 55.0, 0.014, 0.015),
}


def _bars(seed: int, p0: float, period: float, amp: float, sigma: float) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    gaps = rng.uniform(-MAX_GAP, MAX_GAP, N_BARS)
    noise = rng.normal(0.0, sigma, N_BARS)
    wicks = rng.uniform(0.0, 0.01, (N_BARS, 2))
    vols = rng.uniform(100.0, 1000.0, N_BARS)
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    prev = p0
    for t in range(N_BARS):
        o = prev if t == 0 else round(prev * (1.0 + float(gaps[t])), 6)
        c = round(prev * float(np.exp(amp * np.sin(2 * np.pi * t / period) + noise[t])), 6)
        if t == 0:
            c = o  # first open equals first close
        h = round(max(o, c) * (1.0 + float(wicks[t, 0])), 6)
        lo = round(min(o, c) * (1.0 - float(wicks[t, 1])), 6)
        opens.append(o)
        highs.append(h)
        lows.append(lo)
        closes.append(c)
        prev = c
    for t in range(N_BARS):
        assert highs[t] >= max(opens[t], closes[t])
        assert lows[t] <= min(opens[t], closes[t])
        assert min(opens[t], highs[t], lows[t], closes[t]) >= 50.0
        if t > 0:
            assert abs(opens[t] / closes[t - 1] - 1.0) <= MAX_GAP
    assert opens[0] == closes[0]
    return pl.DataFrame(
        {
            "ts": [START + timedelta(days=t) for t in range(N_BARS)],
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [round(float(v), 3) for v in vols],
            "quote_volume": [round(float(v) * c, 3) for v, c in zip(vols, closes, strict=True)],
            "trades": [100] * N_BARS,
        },
        schema=dict(KLINE_SCHEMA),
    )


def _instrument(symbol: str) -> Instrument:
    return Instrument(
        symbol=symbol,
        venue_symbol=symbol,
        kind=InstrumentKind.SPOT,
        base=symbol.removesuffix("USDT"),
        quote="USDT",
        min_qty=0.001,
        qty_step=0.001,
        min_notional=5.0,
        price_tick=0.01,
        maker_fee=0.0,
        taker_fee=0.0,
    )


def build_crosscheck_fixture(root: Path) -> None:
    """Write the crosscheck catalog under `root`; identical bytes on every call."""
    cat = Catalog(root)
    for sym, spec in _SPECS.items():
        cat.write_klines("spot", "1d", sym, _bars(*spec), ("crosscheck",), {}, INGESTED_AT)
    cat.write_instruments(tuple(_instrument(s) for s in _SPECS), INGESTED_AT)


if __name__ == "__main__":
    build_crosscheck_fixture(CROSSCHECK_DIR)
