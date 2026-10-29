"""Tests for bt.runs.writer and bt.runs.reader."""

import math
import shutil
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from bt.core.errors import RunError, TimezoneError
from bt.engine.loop import EngineResult
from bt.runs.config import RunConfig, RunSection
from bt.runs.fingerprint import DataFingerprint
from bt.runs.frames import equity_frame, messages_frame, orders_frame, trades_frame
from bt.runs.meta import RunMeta, meta_to_json
from bt.runs.reader import read_run, resolve_run_ref
from bt.runs.writer import (
    DETERMINISTIC_FILES,
    RUN_FILES,
    RunArtifacts,
    metrics_to_json,
    next_run_id,
    run_hash,
    write_run,
)
from tests.golden.test_golden_spot import run_golden

T0 = datetime(2024, 3, 5, 14, 7, 59, tzinfo=UTC)
RID = "2024-03-05T1407_test_buy_once_abcd1234"


def make_config(**params: float) -> RunConfig:
    run = RunSection(strategy="test_buy_once", start=date(2024, 1, 1), end=date(2024, 1, 5))
    return RunConfig(run=run, params=dict(params))


def make_fp(sha: str = "d" * 64) -> DataFingerprint:
    return DataFingerprint(files=(), instruments=(), sha256=sha)


def make_meta(run_id: str = RID, started: datetime = T0) -> RunMeta:
    return RunMeta(
        run_id=run_id,
        strategy="test_buy_once",
        hash8="abcd1234",
        config_sha256="a" * 64,
        data_sha256="b" * 64,
        git_commit=None,
        git_dirty=None,
        versions=(("polars", "1"),),
        started_at=started,
        finished_at=started + timedelta(seconds=3),
        coverage=(),
        killed_at=None,
        liquidated=False,
    )


@pytest.fixture(scope="module")
def result() -> EngineResult:
    return run_golden()


def make_artifacts(result: EngineResult, run_id: str = RID) -> RunArtifacts:
    return RunArtifacts(
        config=make_config(a=1.5),
        meta=make_meta(run_id),
        metrics={"sharpe": 1.25, "calmar": float("nan"), "n": 3.0},
        equity=equity_frame(result, "1d"),
        trades=trades_frame(result),
        orders=orders_frame(result),
        messages=messages_frame(result),
        report="report text é\nline 2\n",
    )


def test_run_hash() -> None:
    h = run_hash(make_config(a=1.0), make_fp())
    assert len(h) == 64
    int(h, 16)
    assert h == run_hash(make_config(a=1.0), make_fp())
    assert h != run_hash(make_config(a=2.0), make_fp())
    assert h != run_hash(make_config(a=1.0), make_fp("e" * 64))


def test_next_run_id(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    assert next_run_id(runs, T0, "test_buy_once", "abcd1234") == RID  # missing runs_dir
    runs.mkdir()
    (runs / RID).mkdir()
    assert next_run_id(runs, T0, "test_buy_once", "abcd1234") == RID + "-2"
    (runs / (RID + "-2")).mkdir()
    assert next_run_id(runs, T0, "test_buy_once", "abcd1234") == RID + "-3"


def test_next_run_id_tmp_blocks(tmp_path: Path) -> None:
    (tmp_path / f".{RID}.tmp").mkdir()
    assert next_run_id(tmp_path, T0, "test_buy_once", "abcd1234") == RID + "-2"


def test_next_run_id_timezones(tmp_path: Path) -> None:
    ist = timezone(timedelta(hours=5, minutes=30))
    local = datetime(2024, 3, 5, 19, 37, 59, tzinfo=ist)
    assert next_run_id(tmp_path, local, "test_buy_once", "abcd1234") == RID
    with pytest.raises(TimezoneError):
        next_run_id(tmp_path, datetime(2024, 3, 5, 14, 7), "s", "abcd1234")


def test_metrics_to_json() -> None:
    text = metrics_to_json({"b": 1.0, "a": float("nan"), "c": 2})
    assert text == '{\n  "a": null,\n  "b": 1.0,\n  "c": 2.0\n}\n'
    assert '"x": null' in metrics_to_json({"x": float("inf")})


def test_write_then_read(tmp_path: Path, result: EngineResult) -> None:
    art = make_artifacts(result)
    out = write_run(tmp_path / "runs", RID, art)
    assert out == tmp_path / "runs" / RID
    assert sorted(p.name for p in out.iterdir()) == sorted(RUN_FILES)
    assert [p.name for p in (tmp_path / "runs").iterdir()] == [RID]
    data = read_run(out)
    assert data.run_dir == out
    assert data.config == art.config
    assert data.meta == art.meta
    assert set(data.metrics) == set(art.metrics)
    assert math.isnan(data.metrics["calmar"])
    assert data.metrics["sharpe"] == pytest.approx(1.25, abs=1e-12)
    assert data.equity.equals(art.equity)
    assert data.trades.equals(art.trades)
    assert data.orders.equals(art.orders)
    assert data.messages.equals(art.messages)
    assert data.report == art.report


def test_equity_parquet_metadata(tmp_path: Path, result: EngineResult) -> None:
    out = write_run(tmp_path, RID, make_artifacts(result))
    meta = pl.read_parquet_metadata(out / "equity.parquet")
    assert meta["bt.ts"] == "bar open time; values hold at close_ts"


def test_byte_determinism(tmp_path: Path, result: EngineResult) -> None:
    a = write_run(tmp_path, RID, make_artifacts(result))
    rid2 = RID + "-2"
    b = write_run(tmp_path, rid2, make_artifacts(result, rid2))
    for name in DETERMINISTIC_FILES:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    assert (a / "meta.json").read_bytes() != (b / "meta.json").read_bytes()
    assert (a / "meta.json").read_text() == meta_to_json(make_meta())


def test_deterministic_files_list() -> None:
    assert set(RUN_FILES) - set(DETERMINISTIC_FILES) == {"meta.json", "report.txt"}


def test_existing_run_id(tmp_path: Path, result: EngineResult) -> None:
    out = write_run(tmp_path, RID, make_artifacts(result))
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(RunError):
        write_run(tmp_path, RID, replace(make_artifacts(result), report="other"))
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    assert [p.name for p in tmp_path.iterdir()] == [RID]


def test_runs_dir_is_file(tmp_path: Path, result: EngineResult) -> None:
    f = tmp_path / "runs"
    f.write_text("x")
    with pytest.raises(RunError):
        write_run(f, RID, make_artifacts(result))
    assert [p.name for p in tmp_path.iterdir()] == ["runs"]
    assert f.read_text() == "x"


def test_bad_dtype(tmp_path: Path, result: EngineResult) -> None:
    art = make_artifacts(result)
    bad = art.trades.with_columns(pl.col("qty").cast(pl.Float32))
    with pytest.raises(RunError):
        write_run(tmp_path, RID, replace(art, trades=bad))
    assert list(tmp_path.iterdir()) == []


def test_bad_columns(tmp_path: Path, result: EngineResult) -> None:
    art = make_artifacts(result)
    with pytest.raises(RunError):
        write_run(
            tmp_path, RID, replace(art, equity=art.equity.select(reversed(art.equity.columns)))
        )


def test_meta_run_id_mismatch(tmp_path: Path, result: EngineResult) -> None:
    with pytest.raises(RunError):
        write_run(tmp_path, RID + "-2", make_artifacts(result))
    assert list(tmp_path.iterdir()) == []


def test_preexisting_tmp(tmp_path: Path, result: EngineResult) -> None:
    tmp = tmp_path / f".{RID}.tmp"
    tmp.mkdir()
    (tmp / "keep.txt").write_text("k")
    with pytest.raises(RunError):
        write_run(tmp_path, RID, make_artifacts(result))
    assert (tmp / "keep.txt").read_text() == "k"
    assert not (tmp_path / RID).exists()


def test_read_missing_file(tmp_path: Path, result: EngineResult) -> None:
    out = write_run(tmp_path, RID, make_artifacts(result))
    (out / "orders.parquet").unlink()
    with pytest.raises(RunError, match=r"orders\.parquet"):
        read_run(out)


def test_read_corrupt_files(tmp_path: Path, result: EngineResult) -> None:
    out = write_run(tmp_path, RID, make_artifacts(result))
    for name, junk in (
        ("metrics.json", "{nope"),
        ("metrics.json", '{"a": "x"}'),
        ("metrics.json", "[1]"),
        ("config.toml", "not = [valid"),
        ("config.toml", "[run]\nstrategy = 'x'\n"),
        ("equity.parquet", "junk"),
        ("meta.json", "{}"),
    ):
        copy = tmp_path / f"copy-{name}-{len(junk)}"
        shutil.copytree(out, copy)
        (copy / name).write_text(junk)
        with pytest.raises(RunError):
            read_run(copy)


def test_read_schema_mismatch(tmp_path: Path, result: EngineResult) -> None:
    out = write_run(tmp_path, RID, make_artifacts(result))
    pl.DataFrame({"a": [1]}).write_parquet(out / "trades.parquet")
    with pytest.raises(RunError, match=r"trades\.parquet"):
        read_run(out)


def _mk_run(runs: Path, name: str, started: datetime, with_meta: bool = True) -> Path:
    d = runs / name
    d.mkdir(parents=True)
    if with_meta:
        (d / "meta.json").write_text(meta_to_json(make_meta(name, started)))
    return d


def test_resolve_run_ref(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    first = _mk_run(runs, "2024-01-01T0000_s_aaaa1111", datetime(2024, 1, 1, tzinfo=UTC))
    second = _mk_run(runs, "2024-01-02T0000_s_aaaa2222", datetime(2024, 1, 2, tzinfo=UTC))
    _mk_run(runs, ".2024-01-03T0000_s_bbbb3333.tmp", datetime(2024, 1, 3, tzinfo=UTC))
    assert resolve_run_ref(runs, "latest") == second
    assert resolve_run_ref(runs, "aaaa1") == first
    with pytest.raises(RunError, match="ambiguous"):
        resolve_run_ref(runs, "aaaa")
    with pytest.raises(RunError, match="no run"):
        resolve_run_ref(runs, "bbbb")
    with pytest.raises(RunError, match="no run"):
        resolve_run_ref(runs, "aaa")
    assert resolve_run_ref(runs, first.name) == first
    assert resolve_run_ref(runs, str(second.resolve())) == second.resolve()


def test_resolve_latest_uses_started_at(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    first = _mk_run(runs, "2024-01-01T0000_s_aaaa1111", datetime(2024, 1, 9, tzinfo=UTC))
    _mk_run(runs, "2024-01-02T0000_s_aaaa2222", datetime(2024, 1, 2, tzinfo=UTC))
    assert resolve_run_ref(runs, "latest") == first


def test_resolve_latest_skips_unreadable_and_empty(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    with pytest.raises(RunError, match="no run"):
        resolve_run_ref(runs, "latest")
    good = _mk_run(runs, "2024-01-01T0000_s_aaaa1111", datetime(2024, 1, 1, tzinfo=UTC))
    _mk_run(runs, "2024-01-05T0000_s_cccc1111", datetime(2024, 1, 5, tzinfo=UTC), with_meta=False)
    assert resolve_run_ref(runs, "latest") == good
