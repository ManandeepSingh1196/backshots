import sys
import textwrap
from pathlib import Path

import pytest
from pydantic import BaseModel

from bt.core.errors import StrategyError
from bt.core.types import Instrument
from bt.strategy import registry
from bt.strategy.base import Intent, NoAction, Strategy
from bt.strategy.context import Context
from bt.strategy.registry import build_params, get_strategy, list_strategies, register


class Hold(Strategy):
    class Params(BaseModel):
        lookback: int = 20
        vol_target: float = 0.25

    def on_bar(self, ctx: Context) -> Intent:
        return NoAction()


class Hold2(Hold):
    pass


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "_REGISTRY", {})


def test_register_and_lookup() -> None:
    assert register("hold")(Hold) is Hold
    assert get_strategy("hold") is Hold
    assert ("hold", Hold) in _list_no_discover()


def _list_no_discover() -> list[tuple[str, type[Strategy]]]:
    return sorted(registry._REGISTRY.items())  # pyright: ignore[reportPrivateUsage]


def test_list_strategies_includes_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "_discover", lambda: None)
    register("hold")(Hold)
    assert list_strategies() == [("hold", Hold)]


@pytest.mark.parametrize("bad", ["Hold", "1x", "a" * 33, "bt", "", "a-b"])
def test_bad_names(bad: str) -> None:
    with pytest.raises(StrategyError):
        register(bad)(Hold)


@pytest.mark.parametrize("good", ["a" * 32, "bt2"])
def test_good_names(good: str) -> None:
    assert register(good)(Hold) is Hold


def test_required_param_rejected() -> None:
    class Req(Strategy):
        class Params(BaseModel):
            x: int

        def on_bar(self, ctx: Context) -> Intent:
            return NoAction()

    with pytest.raises(StrategyError):
        register("req")(Req)


def test_non_strategy_rejected() -> None:
    class NotOne:
        pass

    with pytest.raises(StrategyError):
        register("nope")(NotOne)  # type: ignore[arg-type]
    with pytest.raises(StrategyError):
        register("base")(Strategy)  # type: ignore[type-abstract]


def test_name_collision_and_reregister() -> None:
    register("hold")(Hold)
    with pytest.raises(StrategyError, match="Hold2"):
        register("hold")(Hold2)
    register("hold")(Hold)
    assert {"hold": Hold} == registry._REGISTRY  # pyright: ignore[reportPrivateUsage]


def _make_pkg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, files: dict[str, str]) -> None:
    pkg = tmp_path / "strategies"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for name, body in files.items():
        (pkg / name).write_text(textwrap.dedent(body))
    monkeypatch.syspath_prepend(str(tmp_path))  # pyright: ignore[reportUnknownMemberType]
    for m in [m for m in sys.modules if m == "strategies" or m.startswith("strategies.")]:
        monkeypatch.delitem(sys.modules, m, raising=False)


ALPHA = """
    from pydantic import BaseModel
    from bt.strategy.base import NoAction, Strategy
    from bt.strategy.registry import register

    @register("alpha")
    class Alpha(Strategy):
        class Params(BaseModel):
            pass

        def on_bar(self, ctx):
            return NoAction()
"""


def test_discovery_skips_private(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_pkg(
        tmp_path,
        monkeypatch,
        {"alpha.py": ALPHA, "_private.py": "raise RuntimeError('private imported')\n"},
    )
    assert get_strategy("alpha").__name__ == "Alpha"
    assert [n for n, _ in list_strategies()] == ["alpha"]
    assert "strategies._private" not in sys.modules


def test_discovery_import_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_pkg(tmp_path, monkeypatch, {"broken.py": "raise RuntimeError('boom')\n"})
    with pytest.raises(StrategyError, match=r"strategies\.broken"):
        list_strategies()


def test_discovery_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    for m in [m for m in sys.modules if m == "strategies" or m.startswith("strategies.")]:
        monkeypatch.delitem(sys.modules, m, raising=False)
    monkeypatch.setitem(sys.modules, "strategies", None)  # makes import raise ImportError
    with pytest.raises(StrategyError, match="strategies"):
        list_strategies()


def test_get_strategy_missing() -> None:
    register("hold")(Hold)
    with pytest.raises(StrategyError, match="hold"):
        get_strategy("missing")


def test_build_params() -> None:
    p = build_params(Hold, {"lookback": 60})
    assert isinstance(p, Hold.Params)
    assert p.lookback == 60  # type: ignore[attr-defined]
    assert p.vol_target == pytest.approx(0.25, abs=1e-12)  # type: ignore[attr-defined]
    assert build_params(Hold, {}) == Hold.Params()
    with pytest.raises(StrategyError, match="lokback"):
        build_params(Hold, {"lokback": 60})
    with pytest.raises(StrategyError):
        build_params(Hold, {"lookback": "abc"})


def test_instrument_import_unused() -> None:
    assert Instrument is not None


def test_discovery_nested_missing_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_pkg(tmp_path, monkeypatch, {})
    (tmp_path / "strategies" / "__init__.py").write_text("import nonexistent_xyz\n")
    with pytest.raises(StrategyError, match="nonexistent_xyz") as ei:
        list_strategies()
    assert "repo root" not in str(ei.value)


def test_discovery_init_runtime_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_pkg(tmp_path, monkeypatch, {})
    (tmp_path / "strategies" / "__init__.py").write_text('raise RuntimeError("bad init")\n')
    with pytest.raises(StrategyError, match="bad init") as ei:
        list_strategies()
    assert isinstance(ei.value.__cause__, RuntimeError)
    assert "repo root" not in str(ei.value)


def test_discovery_missing_package_mentions_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    for m in [m for m in sys.modules if m == "strategies" or m.startswith("strategies.")]:
        monkeypatch.delitem(sys.modules, m, raising=False)
    monkeypatch.setitem(sys.modules, "strategies", None)
    with pytest.raises(StrategyError, match="repo root"):
        list_strategies()


def test_leak_check_helper() -> None:
    from tests.conftest import leaked_strategies

    main_cls = type("Stray", (), {"__module__": "__main__"})
    real_cls = type("BuyHold", (), {"__module__": "strategies.buy_hold"})
    assert leaked_strategies({"stray": main_cls}) == ["stray"]
    assert leaked_strategies({"buy_hold": real_cls}) == []
    assert leaked_strategies({"test_buy_once": main_cls}) == []
