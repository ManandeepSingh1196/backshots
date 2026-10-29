"""Read a run directory back into typed data, and resolve run references."""

import json
import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import polars as pl

from bt.core.errors import BtError, RunError
from bt.runs.config import RunConfig
from bt.runs.frames import EQUITY_SCHEMA, MESSAGES_SCHEMA, ORDERS_SCHEMA, TRADES_SCHEMA
from bt.runs.meta import RunMeta, meta_from_json
from bt.runs.writer import RUN_FILES

_MIN_HASH_REF = 4


@dataclass(frozen=True, slots=True)
class RunData:
    """One run directory, parsed. Invariants: every RUN_FILES entry existed and parsed; frames
    match their *_SCHEMA; metrics null values are NaN."""

    run_dir: Path
    config: RunConfig
    meta: RunMeta
    metrics: Mapping[str, float]
    equity: pl.DataFrame
    trades: pl.DataFrame
    orders: pl.DataFrame
    messages: pl.DataFrame
    report: str


def _text(run_dir: Path, name: str) -> str:
    try:
        return (run_dir / name).read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RunError(f"cannot read {name} in {run_dir}: {exc}") from exc


def _metrics(text: str) -> dict[str, float]:
    try:
        raw: object = json.loads(text)
        if not isinstance(raw, dict):
            raise TypeError("metrics must be a JSON object")
        out: dict[str, float] = {}
        for k, v in cast("dict[str, object]", raw).items():
            if v is None:
                out[k] = math.nan
            elif isinstance(v, bool) or not isinstance(v, int | float):
                raise TypeError(f"metric {k!r} must be a number or null, got {v!r}")
            else:
                out[k] = float(v)
        return out
    except (ValueError, TypeError, RecursionError) as exc:
        raise RunError(f"cannot parse metrics.json: {exc!r}") from exc


def _frame(run_dir: Path, name: str, schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    try:
        df = pl.read_parquet(run_dir / name)
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise RunError(f"cannot read {name} in {run_dir}: {exc}") from exc
    if df.columns != list(schema) or dict(df.schema) != dict(schema):
        raise RunError(f"{name} in {run_dir} does not match its schema: {dict(df.schema)}")
    return df


def read_run(run_dir: Path) -> RunData:
    """Parse a run directory; any missing or malformed file raises RunError."""
    for name in RUN_FILES:
        if not (run_dir / name).is_file():
            raise RunError(f"run {run_dir} is missing {name}")
    try:
        config = RunConfig.model_validate(tomllib.loads(_text(run_dir, "config.toml")))
    except (tomllib.TOMLDecodeError, ValueError, BtError) as exc:
        raise RunError(f"cannot parse config.toml in {run_dir}: {exc}") from exc
    return RunData(
        run_dir=run_dir,
        config=config,
        meta=meta_from_json(_text(run_dir, "meta.json")),
        metrics=_metrics(_text(run_dir, "metrics.json")),
        equity=_frame(run_dir, "equity.parquet", EQUITY_SCHEMA),
        trades=_frame(run_dir, "trades.parquet", TRADES_SCHEMA),
        orders=_frame(run_dir, "orders.parquet", ORDERS_SCHEMA),
        messages=_frame(run_dir, "messages.parquet", MESSAGES_SCHEMA),
        report=_text(run_dir, "report.txt"),
    )


def _visible(runs_dir: Path) -> list[Path]:
    try:
        return sorted(p for p in runs_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    except OSError:
        return []


def _latest(runs_dir: Path) -> Path | None:
    best: tuple[datetime, str] | None = None
    best_path: Path | None = None
    for p in _visible(runs_dir):
        try:
            started = meta_from_json(_text(p, "meta.json")).started_at
        except RunError:
            continue
        key = (started, p.name)
        if best is None or key > best:
            best, best_path = key, p
    return best_path


def resolve_run_ref(runs_dir: Path, ref: str) -> Path:
    """Resolve a path, "latest", an exact run name, or a unique hash8 prefix (>= 4 chars)."""
    direct = Path(ref)
    if direct.is_dir() and (direct / "meta.json").is_file():
        return direct
    if ref == "latest":
        latest = _latest(runs_dir)
        if latest is None:
            raise RunError(f"no run found in {runs_dir}")
        return latest
    visible = _visible(runs_dir)
    for p in visible:
        if p.name == ref:
            return p
    if len(ref) >= _MIN_HASH_REF:
        hits = [p for p in visible if p.name.split("_")[-1][:8].startswith(ref)]
        if len(hits) == 1:
            return hits[0]
        if hits:
            names = ", ".join(p.name for p in hits)
            raise RunError(f"ambiguous run reference {ref!r}: {names}")
    raise RunError(f"no run matches {ref!r} in {runs_dir}")
