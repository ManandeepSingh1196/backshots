"""Atomic run-directory writer: RunArtifacts -> `<runs_dir>/<run_id>/` (DESIGN §9, INTERFACES §14.7).

The directory is built under `.<run_id>.tmp` and renamed into place, so a visible run
directory is always complete. Nothing existing is ever overwritten.

Determinism: the files in DETERMINISTIC_FILES are byte-identical for identical artifacts. This
holds only within one install: polars/pyarrow versions can change Parquet bytes. The versions
used are recorded in `meta.json`.
"""

import hashlib
import json
import math
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl
from pydantic import JsonValue

from bt.core.clock import ensure_utc
from bt.core.errors import RunError
from bt.runs.config import RunConfig, config_to_toml
from bt.runs.fingerprint import DataFingerprint, canonical_json
from bt.runs.frames import EQUITY_SCHEMA, MESSAGES_SCHEMA, ORDERS_SCHEMA, TRADES_SCHEMA
from bt.runs.meta import RunMeta, meta_to_json

RUN_FILES: Final[tuple[str, ...]] = (
    "config.toml",
    "meta.json",
    "metrics.json",
    "equity.parquet",
    "trades.parquet",
    "orders.parquet",
    "messages.parquet",
    "report.txt",
)
DETERMINISTIC_FILES: Final[tuple[str, ...]] = tuple(
    f for f in RUN_FILES if f not in ("meta.json", "report.txt")
)
_HASH_VERSION: Final = 1
_EQUITY_METADATA: Final = {"bt.ts": "bar open time; values hold at close_ts"}


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """Everything one run directory holds.

    Invariants: config is the resolved config; frames follow their *_SCHEMA (checked by
    write_run, not here); nothing time-dependent lives outside `meta` and `report`.
    """

    config: RunConfig
    meta: RunMeta
    metrics: Mapping[str, float]
    equity: pl.DataFrame
    trades: pl.DataFrame
    orders: pl.DataFrame
    messages: pl.DataFrame
    report: str


def run_hash(config: RunConfig, data: DataFingerprint) -> str:
    """sha256 hex of the resolved config and the data fingerprint (config is not resolved here)."""
    payload: dict[str, JsonValue] = {"config": config.model_dump(mode="json"), "data": data.sha256, "v": _HASH_VERSION}
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def next_run_id(runs_dir: Path, started_at: datetime, strategy: str, hash8: str) -> str:
    """`<YYYY-MM-DDTHHMM>_<strategy>_<hash8>`, with -2, -3, ... when the name (or its tmp) is taken."""
    if started_at.tzinfo is None:
        ensure_utc(started_at, "started_at")  # raises TimezoneError for naive input
    started_at = started_at.astimezone(UTC)
    base = f"{started_at:%Y-%m-%dT%H%M}_{strategy}_{hash8}"

    def taken(name: str) -> bool:
        return (runs_dir / name).exists() or (runs_dir / f".{name}.tmp").exists()

    if not taken(base):
        return base
    n = 2
    while taken(f"{base}-{n}"):
        n += 1
    return f"{base}-{n}"


def metrics_to_json(metrics: Mapping[str, float]) -> str:
    """Sorted-key, indent-2 JSON; non-finite values become null; values are floats."""
    obj = {k: (float(v) if math.isfinite(v) else None) for k, v in metrics.items()}
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def _check_schema(name: str, df: pl.DataFrame, schema: Mapping[str, pl.DataType]) -> None:
    if df.columns != list(schema) or dict(df.schema) != dict(schema):
        raise RunError(f"{name} frame does not match its schema: {dict(df.schema)}")


def _write_parquet(df: pl.DataFrame, path: Path, metadata: Mapping[str, str] | None = None) -> None:
    df.write_parquet(
        path,
        compression="zstd",
        compression_level=3,
        statistics=True,
        metadata=dict(metadata) if metadata else None,
    )


def write_run(runs_dir: Path, run_id: str, artifacts: RunArtifacts) -> Path:
    """Write all RUN_FILES atomically and return the final directory; never overwrites."""
    if artifacts.meta.run_id != run_id:
        raise RunError(f"meta.run_id {artifacts.meta.run_id!r} != run_id {run_id!r}")
    _check_schema("equity", artifacts.equity, EQUITY_SCHEMA)
    _check_schema("trades", artifacts.trades, TRADES_SCHEMA)
    _check_schema("orders", artifacts.orders, ORDERS_SCHEMA)
    _check_schema("messages", artifacts.messages, MESSAGES_SCHEMA)
    final = runs_dir / run_id
    tmp = runs_dir / f".{run_id}.tmp"
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        if final.exists():
            raise RunError(f"run directory already exists: {final}")
        try:
            tmp.mkdir(exist_ok=False)
        except FileExistsError as exc:
            raise RunError(f"temporary run directory already exists: {tmp}") from exc
    except OSError as exc:
        raise RunError(f"cannot create run directory in {runs_dir}: {exc}") from exc
    try:
        (tmp / "config.toml").write_bytes(config_to_toml(artifacts.config).encode("utf-8"))
        (tmp / "meta.json").write_bytes(meta_to_json(artifacts.meta).encode("utf-8"))
        (tmp / "metrics.json").write_bytes(metrics_to_json(artifacts.metrics).encode("utf-8"))
        _write_parquet(artifacts.equity, tmp / "equity.parquet", _EQUITY_METADATA)
        _write_parquet(artifacts.trades, tmp / "trades.parquet")
        _write_parquet(artifacts.orders, tmp / "orders.parquet")
        _write_parquet(artifacts.messages, tmp / "messages.parquet")
        (tmp / "report.txt").write_bytes(artifacts.report.encode("utf-8"))
        if final.exists():  # os.rename would silently replace an empty directory
            raise RunError(f"run directory already exists: {final}")
        os.rename(tmp, final)
    except RunError:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    except (OSError, pl.exceptions.PolarsError) as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RunError(f"cannot write run {run_id}: {exc}") from exc
    return final
