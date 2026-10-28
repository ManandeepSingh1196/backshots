import ast
import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

PACKAGES = [
    "core",
    "data",
    "strategy",
    "execution",
    "portfolio",
    "costs",
    "engine",
    "analytics",
    "research",
    "runs",
    "live",
    "cli",
]

# owner prefix -> allowed internal prefixes. Unlisted owners are unrestricted.
ALLOWED: dict[str, tuple[str, ...]] = {
    "bt.core": ("bt.core",),
    "bt.strategy": ("bt.core", "bt.strategy"),
    "bt.execution": ("bt.core", "bt.execution"),
    "bt.portfolio": ("bt.core", "bt.portfolio"),
    "bt.costs": ("bt.core", "bt.costs"),
    "strategies": ("bt.core", "bt.strategy"),
}


def _is_under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def _is_internal(name: str) -> bool:
    return _is_under(name, "bt") or _is_under(name, "strategies")


def _module_name(path: Path, root: Path) -> tuple[str, bool]:
    """Return (dotted module name, is_package) for a file under root (src/ or repo root)."""
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    is_pkg = parts[-1] == "__init__"
    if is_pkg:
        parts = parts[:-1]
    return ".".join(parts), is_pkg


def _imported_modules(tree: ast.AST, module: str, is_pkg: bool) -> list[str]:
    """Absolute module names imported by `tree`, one entry per imported target."""
    package = module if is_pkg else module.rpartition(".")[0]
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                parts = package.split(".") if package else []
                parts = parts[: len(parts) - (node.level - 1)]
                base = ".".join([*parts, *([node.module] if node.module else [])])
            if node.module is None or base == "bt":
                # `from . import x` / `from bt import x`: each name may be a submodule
                found.extend(f"{base}.{alias.name}" if base else alias.name for alias in node.names)
            else:
                found.append(base)
    return found


def _check_tree(src_root: Path, strategies_root: Path) -> list[str]:
    """Return violations as '<file>: <module> not allowed in <owner>'.

    Files under src_root are named relative to it (bt/core/x.py); files under
    strategies_root are named relative to its parent (strategies/s.py).
    """
    violations: list[str] = []
    for scan, base in ((src_root, src_root), (strategies_root, strategies_root.parent)):
        if not scan.exists():
            continue
        for path in sorted(scan.rglob("*.py")):
            module, is_pkg = _module_name(path, base)
            owner = next((o for o in ALLOWED if _is_under(module, o)), None)
            if owner is None:
                continue
            for target in _imported_modules(ast.parse(path.read_text()), module, is_pkg):
                if _is_internal(target) and not any(_is_under(target, a) for a in ALLOWED[owner]):
                    rel = path.relative_to(base).as_posix()
                    violations.append(f"{rel}: {target} not allowed in {owner}")
    return violations


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _run(root: Path) -> list[str]:
    return _check_tree(root / "src", root / "strategies")


def test_repo_obeys_import_rule() -> None:
    assert _check_tree(REPO_ROOT / "src", REPO_ROOT / "strategies") == []


def test_checker_flags_core_importing_engine(tmp_path: Path) -> None:
    _write(tmp_path, "src/bt/core/x.py", "from bt.engine import backtest\n")
    violations = _run(tmp_path)
    assert len(violations) == 1
    assert "bt/core/x.py" in violations[0]
    assert "bt.engine" in violations[0]


def test_checker_resolves_relative_imports(tmp_path: Path) -> None:
    _write(tmp_path, "src/bt/execution/y.py", "from ..portfolio import z\n")
    violations = _run(tmp_path)
    assert len(violations) == 1
    assert "bt.portfolio" in violations[0]

    ok = tmp_path / "ok"
    _write(ok, "src/bt/execution/y.py", "from . import fees\n")
    assert _run(ok) == []


def test_checker_flags_strategy_importing_data(tmp_path: Path) -> None:
    _write(tmp_path, "strategies/s.py", "import bt.data.catalog\n")
    assert len(_run(tmp_path)) == 1


def test_checker_allows_strategies_using_core_and_strategy(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "strategies/s.py",
        "from bt.core.types import Bar\nfrom bt.strategy.base import Strategy\n",
    )
    assert _run(tmp_path) == []


def test_checker_flags_strategies_importing_strategies(tmp_path: Path) -> None:
    _write(tmp_path, "strategies/s.py", "from strategies import other\n")
    assert len(_run(tmp_path)) == 1


def test_type_checking_imports_are_checked(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/bt/strategy/c.py",
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from bt.portfolio import portfolio\n",
    )
    assert len(_run(tmp_path)) == 1


@pytest.mark.parametrize("pkg", PACKAGES)
def test_every_package_imports(pkg: str) -> None:
    module = importlib.import_module(f"bt.{pkg}")
    assert module.__doc__
    assert module.__doc__.strip()
