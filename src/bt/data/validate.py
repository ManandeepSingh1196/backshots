"""Schemas and validation for stored klines and funding.

Validation is pure: it never raises on bad data, never fixes it, and returns a report.
Errors block storage; gaps are reported and never filled; flags mark suspicious bars
and never delete them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Final, Literal

import polars as pl

Market = Literal["spot", "perp"]
Timeframe = Literal["1d", "4h"]

TIMEFRAME_DELTA: Final[Mapping[Timeframe, timedelta]] = MappingProxyType(
    {"1d": timedelta(days=1), "4h": timedelta(hours=4)}
)
_TS = pl.Datetime("us", "UTC")
KLINE_SCHEMA: Final[Mapping[str, pl.DataType]] = MappingProxyType(
    {
        "ts": _TS,
        "open": pl.Float64(),
        "high": pl.Float64(),
        "low": pl.Float64(),
        "close": pl.Float64(),
        "volume": pl.Float64(),
        "quote_volume": pl.Float64(),
        "trades": pl.Int64(),
    }
)
FUNDING_SCHEMA: Final[Mapping[str, pl.DataType]] = MappingProxyType(
    {"ts": _TS, "rate": pl.Float64(), "mark_price": pl.Float64()}
)
FUNDING_INTERVAL: Final[timedelta] = timedelta(hours=8)

_FUNDING_TOLERANCE: Final[timedelta] = timedelta(seconds=60)
_JUMP_LIMIT: Final[float] = 0.5
_RANGE_LIMIT: Final[float] = 1.0
_EXTREME_RATE: Final[float] = 0.01
_US: Final[timedelta] = timedelta(microseconds=1)


@dataclass(frozen=True, slots=True)
class Gap:
    """Missing stretch between two present timestamps; `missing` is a positive-or-zero count."""

    after_ts: datetime
    before_ts: datetime
    missing: int


@dataclass(frozen=True, slots=True)
class BarFlag:
    """A suspicious bar or funding event: kept in the data, only reported."""

    ts: datetime
    reason: str


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Outcome of validating one dataset.

    Invariants: `ok` is True iff `errors` is empty; `flags` are sorted by (ts, reason);
    `rows`, `first_ts` and `last_ts` are filled whenever the schema was correct.
    """

    dataset: str
    rows: int
    first_ts: datetime | None
    last_ts: datetime | None
    errors: tuple[str, ...]
    gaps: tuple[Gap, ...]
    flags: tuple[BarFlag, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _schema_error(df: pl.DataFrame, expected: Mapping[str, pl.DataType]) -> str | None:
    actual = list(df.schema.items())
    want = list(expected.items())
    if len(actual) == len(want) and all(
        a_name == w_name and a_type == w_type
        for (a_name, a_type), (w_name, w_type) in zip(actual, want, strict=True)
    ):
        return None
    got = ", ".join(f"{n}: {t}" for n, t in actual)
    exp = ", ".join(f"{n}: {t}" for n, t in want)
    return f"schema mismatch: expected [{exp}], got [{got}]"


def _count_first(df: pl.DataFrame, cond: pl.Expr) -> tuple[int, datetime | None]:
    """Rows where `cond` is true (null counts as false) and the first such ts."""
    bad = df.filter(cond.fill_null(False))
    if bad.height == 0:
        return 0, None
    first: datetime = bad["ts"][0]
    return bad.height, first


def _rule(df: pl.DataFrame, cond: pl.Expr, message: str) -> list[str]:
    n, first = _count_first(df, cond)
    if n == 0 or first is None:
        return []
    return [f"{message}: {n} rows, first {first.isoformat()}"]


def _null_errors(df: pl.DataFrame, columns: list[str]) -> list[str]:
    errors: list[str] = []
    for col in columns:
        n = df[col].null_count()
        if n:
            errors.append(f"nulls in {col}: {n}")
    return errors


def _order_errors(df: pl.DataFrame) -> tuple[list[str], bool]:
    """Duplicate / unsorted errors, and whether ts is strictly increasing."""
    errors: list[str] = []
    dup = df.filter(pl.col("ts").is_duplicated())
    if dup.height:
        first: datetime = dup["ts"][0]
        errors.append(f"duplicate ts: {dup.height} rows, first {first.isoformat()}")
    dec = df.filter(pl.col("ts").diff().dt.total_microseconds() < 0)
    if dec.height:
        first_dec: datetime = dec["ts"][0]
        errors.append(f"unsorted ts: {dec.height} decreases, first at {first_dec.isoformat()}")
    return errors, not errors


def _diffs(df: pl.DataFrame) -> pl.DataFrame:
    """Frame of consecutive ts pairs with `d_us`, the difference in microseconds."""
    return df.select(
        pl.col("ts").shift(1).alias("after_ts"),
        pl.col("ts").alias("before_ts"),
        pl.col("ts").diff().dt.total_microseconds().alias("d_us"),
    ).filter(pl.col("d_us").is_not_null())


def _bounds(df: pl.DataFrame) -> tuple[datetime | None, datetime | None]:
    if df.height == 0:
        return None, None
    return df["ts"].min(), df["ts"].max()  # pyright: ignore[reportReturnType]


def _flags(df: pl.DataFrame, cond: pl.Expr, reason: str) -> list[BarFlag]:
    hits = df.filter(cond.fill_null(False))
    return [BarFlag(ts=t, reason=reason) for t in hits["ts"].to_list()]


def _sorted_flags(flags: list[BarFlag]) -> tuple[BarFlag, ...]:
    return tuple(sorted(flags, key=lambda f: (f.ts, f.reason)))


def validate_klines(df: pl.DataFrame, timeframe: Timeframe, dataset: str) -> ValidationReport:
    """Validate a klines frame against KLINE_SCHEMA and the timeframe grid."""
    schema_err = _schema_error(df, KLINE_SCHEMA)
    if schema_err is not None:
        return ValidationReport(dataset, df.height, None, None, (schema_err,), (), ())
    first_ts, last_ts = _bounds(df)
    if df.height == 0:
        return ValidationReport(dataset, 0, None, None, ("empty",), (), ())

    delta_us = TIMEFRAME_DELTA[timeframe] // _US
    errors = _null_errors(df, list(KLINE_SCHEMA))
    ts_clean = df["ts"].null_count() == 0
    ordered = False
    if ts_clean:
        order_errors, ordered = _order_errors(df)
        errors += order_errors
        errors += _rule(
            df,
            pl.col("ts").dt.epoch("us") % delta_us != 0,
            f"ts off {timeframe} grid",
        )

    o, h, lo, c = (pl.col(n) for n in ("open", "high", "low", "close"))
    for name in ("open", "high", "low", "close"):
        errors += _rule(df, pl.col(name) <= 0, f"non-positive {name}")
    errors += _rule(df, h < pl.max_horizontal(o, c), "high below open/close")
    errors += _rule(df, lo > pl.min_horizontal(o, c), "low above open/close")
    errors += _rule(df, h < lo, "high below low")
    for name in ("volume", "quote_volume", "trades"):
        errors += _rule(df, pl.col(name) < 0, f"negative {name}")

    gaps: list[Gap] = []
    flags = _flags(df, pl.col("volume") == 0, "zero_volume")
    flags += _flags(df, (h - lo) / lo > _RANGE_LIMIT, "wide_range")
    if ordered:
        d = _diffs(df).filter(pl.col("d_us") > delta_us)
        gaps = [
            Gap(a, b, d_us // delta_us - 1)
            for a, b, d_us in d.iter_rows()
        ]
        flags += _flags(df, (c / c.shift(1) - 1).abs() > _JUMP_LIMIT, "jump")
    return ValidationReport(
        dataset, df.height, first_ts, last_ts, tuple(errors), tuple(gaps), _sorted_flags(flags)
    )


def validate_funding(df: pl.DataFrame, dataset: str) -> ValidationReport:
    """Validate a funding frame against FUNDING_SCHEMA; no grid check, 8h/4h intervals."""
    schema_err = _schema_error(df, FUNDING_SCHEMA)
    if schema_err is not None:
        return ValidationReport(dataset, df.height, None, None, (schema_err,), (), ())
    first_ts, last_ts = _bounds(df)
    if df.height == 0:
        return ValidationReport(dataset, 0, None, None, ("empty",), (), ())

    errors = _null_errors(df, ["ts", "rate"])
    ordered = False
    if df["ts"].null_count() == 0:
        order_errors, ordered = _order_errors(df)
        errors += order_errors
    errors += _rule(df, pl.col("mark_price") <= 0, "non-positive mark_price")

    interval_us = FUNDING_INTERVAL // _US
    limit_us = (FUNDING_INTERVAL + _FUNDING_TOLERANCE) // _US
    gaps: list[Gap] = []
    if ordered:
        d = _diffs(df).filter(pl.col("d_us") > limit_us)
        gaps = [Gap(a, b, max(1, round(d_us / interval_us) - 1)) for a, b, d_us in d.iter_rows()]
    flags = _flags(df, pl.col("rate").abs() > _EXTREME_RATE, "extreme_rate")
    return ValidationReport(
        dataset, df.height, first_ts, last_ts, tuple(errors), tuple(gaps), _sorted_flags(flags)
    )
