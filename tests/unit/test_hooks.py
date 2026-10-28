import os
import shutil
import stat
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _hooks_path(repo: Path) -> str:
    out = subprocess.run(
        ["git", "config", "core.hooksPath"], cwd=repo, check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


def test_install_sets_hooks_path(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "scripts" / "hooks").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "install-hooks.sh", tmp_path / "scripts" / "install-hooks.sh")
    shutil.copy(
        ROOT / "scripts" / "hooks" / "pre-commit", tmp_path / "scripts" / "hooks" / "pre-commit"
    )
    for _ in range(2):
        subprocess.run(["bash", "scripts/install-hooks.sh"], cwd=tmp_path, check=True)
        assert _hooks_path(tmp_path) == "scripts/hooks"
    assert os.access(tmp_path / "scripts" / "hooks" / "pre-commit", os.X_OK)


def test_hook_and_check_are_executable() -> None:
    for rel in ("scripts/hooks/pre-commit", "scripts/install-hooks.sh", "scripts/check.sh"):
        assert (ROOT / rel).stat().st_mode & stat.S_IXUSR, rel
    hook = (ROOT / "scripts" / "hooks" / "pre-commit").read_text()
    assert "ruff format --check" in hook
    assert "ruff check" in hook


def test_worktrees_excluded() -> None:
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]
    assert ".worktrees" in cfg["ruff"]["extend-exclude"]
    assert ".worktrees" in cfg["pyright"]["exclude"]
    assert ".worktrees" in cfg["pytest"]["ini_options"]["norecursedirs"]
