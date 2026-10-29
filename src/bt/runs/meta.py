"""Run metadata: git state, package versions, timing and data coverage (meta.json)."""

import json
import platform
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final, cast

from bt.core.clock import ensure_utc
from bt.core.errors import BtError, InvariantError, RunError
from bt.runs.fingerprint import SymbolCoverage

_SCHEMA: Final = 1
_HEX = re.compile(r"^[0-9a-f]+$")
_PACKAGES: Final = ("backshots", "numpy", "polars", "pyarrow", "pydantic")


@dataclass(frozen=True, slots=True)
class RunMeta:
    """Provenance of one run. Invariants: datetimes are UTC; finished_at >= started_at;
    hash8 is 8 lowercase hex chars; all hashes are lowercase hex; versions sorted by name."""

    run_id: str
    strategy: str
    hash8: str
    config_sha256: str
    data_sha256: str
    git_commit: str | None
    git_dirty: bool | None
    versions: tuple[tuple[str, str], ...]
    started_at: datetime
    finished_at: datetime
    coverage: tuple[SymbolCoverage, ...]
    killed_at: datetime | None
    liquidated: bool

    def __post_init__(self) -> None:
        started = ensure_utc(self.started_at, "started_at")
        finished = ensure_utc(self.finished_at, "finished_at")
        object.__setattr__(self, "started_at", started)
        object.__setattr__(self, "finished_at", finished)
        if self.killed_at is not None:
            object.__setattr__(self, "killed_at", ensure_utc(self.killed_at, "killed_at"))
        if finished < started:
            raise InvariantError("finished_at is before started_at")
        if len(self.hash8) != 8:
            raise InvariantError(f"hash8 must have 8 characters, got {self.hash8!r}")
        hashes = [("hash8", self.hash8), ("config_sha256", self.config_sha256)]
        hashes.append(("data_sha256", self.data_sha256))
        for name, h in hashes:
            if not _HEX.match(h):
                raise InvariantError(f"{name} must be lowercase hex, got {h!r}")
        if self.git_commit is not None and not _HEX.match(self.git_commit):
            raise InvariantError(f"git_commit must be lowercase hex, got {self.git_commit!r}")
        if list(self.versions) != sorted(self.versions):
            raise InvariantError("versions must be sorted")


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout if proc.returncode == 0 else None


def git_info(repo_root: Path) -> tuple[str | None, bool | None]:
    """(HEAD commit, tracked-files-dirty) from the local git; (None, None) on any failure."""
    head = _git(repo_root, "rev-parse", "HEAD")
    if head is None:
        return None, None
    status = _git(repo_root, "status", "--porcelain", "--untracked-files=no")
    if status is None:
        return None, None
    return head.strip(), bool(status.strip())


def package_versions() -> tuple[tuple[str, str], ...]:
    """Sorted (name, version) for the key packages plus python."""
    out: list[tuple[str, str]] = []
    for name in _PACKAGES:
        try:
            out.append((name, version(name)))
        except PackageNotFoundError:
            out.append((name, "unknown"))
    out.append(("python", platform.python_version()))
    return tuple(sorted(out))


def _dt_json(ts: datetime | None) -> str | None:
    return None if ts is None else ts.isoformat()


def meta_to_json(meta: RunMeta) -> str:
    """Deterministic JSON (sorted keys, indent 2, trailing newline)."""
    obj: dict[str, object] = {
        "schema": _SCHEMA,
        "run_id": meta.run_id,
        "strategy": meta.strategy,
        "hash8": meta.hash8,
        "config_sha256": meta.config_sha256,
        "data_sha256": meta.data_sha256,
        "git_commit": meta.git_commit,
        "git_dirty": meta.git_dirty,
        "versions": dict(meta.versions),
        "started_at": meta.started_at.isoformat(),
        "finished_at": meta.finished_at.isoformat(),
        "wall_seconds": (meta.finished_at - meta.started_at).total_seconds(),
        "coverage": [
            {
                "symbol": c.symbol,
                "bars": c.bars,
                "missing": c.missing,
                "first_ts": _dt_json(c.first_ts),
                "last_ts": _dt_json(c.last_ts),
            }
            for c in meta.coverage
        ],
        "killed_at": _dt_json(meta.killed_at),
        "liquidated": meta.liquidated,
    }
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def _str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {value!r}")
    return value


def _opt_str(value: object, name: str) -> str | None:
    return None if value is None else _str(value, name)


def _int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {value!r}")
    return value


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {value!r}")
    return value


def _opt_bool(value: object, name: str) -> bool | None:
    return None if value is None else _bool(value, name)


def _dt(value: object, name: str) -> datetime:
    return ensure_utc(datetime.fromisoformat(_str(value, name)), name)


def _opt_dt(value: object, name: str) -> datetime | None:
    return None if value is None else _dt(value, name)


def _obj(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object, got {type(value).__name__}")
    items = cast(dict[object, object], value)
    return {_str(k, f"{name} key"): v for k, v in items.items()}


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list, got {type(value).__name__}")
    return list(cast(list[object], value))


def _coverage(raw: object) -> SymbolCoverage:
    d = _obj(raw, "coverage entry")
    return SymbolCoverage(
        symbol=_str(d["symbol"], "symbol"),
        bars=_int(d["bars"], "bars"),
        missing=_int(d["missing"], "missing"),
        first_ts=_opt_dt(d["first_ts"], "first_ts"),
        last_ts=_opt_dt(d["last_ts"], "last_ts"),
    )


def meta_from_json(text: str) -> RunMeta:
    """Inverse of meta_to_json; any malformed input raises RunError."""
    try:
        d = _obj(json.loads(text), "meta")
        if _int(d["schema"], "schema") != _SCHEMA:
            raise ValueError(f"unsupported meta schema {d['schema']!r}")
        versions = tuple((k, _str(v, "version")) for k, v in _obj(d["versions"], "versions").items())
        return RunMeta(
            run_id=_str(d["run_id"], "run_id"),
            strategy=_str(d["strategy"], "strategy"),
            hash8=_str(d["hash8"], "hash8"),
            config_sha256=_str(d["config_sha256"], "config_sha256"),
            data_sha256=_str(d["data_sha256"], "data_sha256"),
            git_commit=_opt_str(d["git_commit"], "git_commit"),
            git_dirty=_opt_bool(d["git_dirty"], "git_dirty"),
            versions=tuple(sorted(versions)),
            started_at=_dt(d["started_at"], "started_at"),
            finished_at=_dt(d["finished_at"], "finished_at"),
            coverage=tuple(_coverage(c) for c in _list(d["coverage"], "coverage")),
            killed_at=_opt_dt(d["killed_at"], "killed_at"),
            liquidated=_bool(d["liquidated"], "liquidated"),
        )
    except (KeyError, TypeError, ValueError, RecursionError, BtError) as exc:
        raise RunError(f"cannot parse meta.json: {exc!r}") from exc
