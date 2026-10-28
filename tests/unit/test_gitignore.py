import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", path],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


@pytest.mark.parametrize("path", ["src/bt/data/__init__.py", "src/bt/runs/__init__.py"])
def test_source_packages_are_not_ignored(path: str) -> None:
    assert not _ignored(path)


@pytest.mark.parametrize("path", ["data/x.parquet", "runs/x/metrics.json"])
def test_data_and_runs_dirs_are_ignored(path: str) -> None:
    assert _ignored(path)
