"""Tests for bt.runs.meta."""

import json
import pickle
import subprocess
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from bt.core.errors import InvariantError, RunError, TimezoneError
from bt.runs.fingerprint import SymbolCoverage
from bt.runs.meta import (
    RunMeta,
    git_info,
    meta_from_json,
    meta_to_json,
    package_versions,
)

T0 = datetime(2024, 3, 1, 12, 0, 0, tzinfo=UTC)


def make_meta() -> RunMeta:
    return RunMeta(
        run_id="20240301-tsmom-abcd1234",
        strategy="tsmom",
        hash8="abcd1234",
        config_sha256="a" * 64,
        data_sha256="b" * 64,
        git_commit=None,
        git_dirty=None,
        versions=(("numpy", "2.0"), ("python", "3.12.1")),
        started_at=T0,
        finished_at=T0 + timedelta(seconds=12.5),
        coverage=(
            SymbolCoverage("AAAUSDT", 5, 0, T0, T0 + timedelta(days=4)),
            SymbolCoverage("BBBUSDT", 0, 5, None, None),
        ),
        killed_at=T0 + timedelta(seconds=5),
        liquidated=True,
    )


def run_git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def test_git_info_no_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert git_info(tmp_path) == (None, None)


def test_git_info_clean_dirty_untracked(tmp_path: Path) -> None:
    run_git(tmp_path, "init", "-q")
    f = tmp_path / "a.txt"
    f.write_text("one")
    run_git(tmp_path, "add", "a.txt")
    run_git(tmp_path, "commit", "-q", "-m", "init")
    commit, dirty = git_info(tmp_path)
    assert commit is not None
    assert len(commit) == 40
    int(commit, 16)
    assert dirty is False
    (tmp_path / "new.txt").write_text("x")
    assert git_info(tmp_path) == (commit, False)
    f.write_text("two")
    assert git_info(tmp_path) == (commit, True)


def test_git_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")
    assert git_info(tmp_path) == (None, None)


def test_package_versions() -> None:
    vs = package_versions()
    names = [n for n, _ in vs]
    assert names == sorted(names)
    assert {"python", "polars", "backshots"} <= set(names)
    assert all(v for _, v in vs)


def test_round_trip() -> None:
    m = make_meta()
    text = meta_to_json(m)
    assert meta_from_json(text) == m
    assert text.endswith("\n")
    obj = json.loads(text)
    assert list(obj) == sorted(obj)
    assert obj["wall_seconds"] == pytest.approx(12.5, abs=1e-9)
    assert obj["schema"] == 1
    m2 = replace(m, git_commit="f" * 40, git_dirty=False, killed_at=None, liquidated=False)
    assert meta_from_json(meta_to_json(m2)) == m2


def test_timezone_handling() -> None:
    ist = timezone(timedelta(hours=5, minutes=30))
    with pytest.raises(TimezoneError):
        replace(make_meta(), started_at=T0.astimezone(ist))
    with pytest.raises(TimezoneError):
        replace(make_meta(), started_at=datetime(2024, 3, 1))  # noqa: DTZ001
    zero = timezone(timedelta(0), "X")
    m = replace(make_meta(), started_at=T0.astimezone(zero))
    assert m.started_at.tzinfo is UTC


def test_invariants() -> None:
    m = make_meta()
    with pytest.raises(InvariantError):
        replace(m, finished_at=m.started_at - timedelta(seconds=1))
    with pytest.raises(InvariantError):
        replace(m, hash8="ABC")
    with pytest.raises(InvariantError):
        replace(m, config_sha256="G" * 64)
    with pytest.raises(InvariantError):
        replace(m, versions=(("python", "3"), ("numpy", "2")))


def test_from_json_errors() -> None:
    obj = json.loads(meta_to_json(make_meta()))
    no_strategy = {k: v for k, v in obj.items() if k != "strategy"}
    for text in (json.dumps(no_strategy), json.dumps({**obj, "schema": 2}), "{not json"):
        with pytest.raises(RunError):
            meta_from_json(text)
    with pytest.raises(RunError):
        meta_from_json(json.dumps({**obj, "liquidated": "yes"}))


def test_from_json_deeply_nested_raises_run_error() -> None:
    with pytest.raises(RunError):
        meta_from_json("[" * 100_000 + "]" * 100_000)


def test_pickle() -> None:
    m = make_meta()
    assert pickle.loads(pickle.dumps(m)) == m
