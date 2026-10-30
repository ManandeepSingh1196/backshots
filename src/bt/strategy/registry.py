"""Strategy registry: name -> class, discovery of the top-level `strategies` package."""

import importlib
import pkgutil
import re
from collections.abc import Callable, Mapping
from typing import Final

from pydantic import BaseModel, ValidationError

from bt.core.errors import StrategyError
from bt.strategy.base import Strategy

STRATEGY_NAME_PATTERN: Final = r"^(?!bt$)[a-z][a-z0-9_]{0,31}$"  # "bt" is reserved

# The one sanctioned piece of module-level mutable state; write-once per name.
_REGISTRY: dict[str, type[Strategy]] = {}


def register(name: str) -> Callable[[type[Strategy]], type[Strategy]]:
    """Class decorator registering a Strategy subclass under `name`; returns the class."""

    def deco(cls: type[Strategy]) -> type[Strategy]:
        if not re.fullmatch(STRATEGY_NAME_PATTERN, name):
            raise StrategyError(
                f"invalid strategy name {name!r}; must match {STRATEGY_NAME_PATTERN}"
            )
        if not (isinstance(cls, type) and issubclass(cls, Strategy)) or cls is Strategy:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise StrategyError(f"{cls!r} is not a proper subclass of Strategy")
        params_cls: object = getattr(cls, "Params", None)
        if not (isinstance(params_cls, type) and issubclass(params_cls, BaseModel)):
            raise StrategyError(f"{cls.__qualname__}.Params must be a pydantic BaseModel")
        try:
            params_cls()
        except ValidationError as exc:
            raise StrategyError(
                f"{cls.__qualname__}.Params must construct from defaults alone: {exc}"
            ) from exc
        old = _REGISTRY.get(name)
        if old is not None and (old.__module__, old.__qualname__) != (
            cls.__module__,
            cls.__qualname__,
        ):
            raise StrategyError(
                f"strategy name {name!r} already taken by {old.__module__}.{old.__qualname__}; "
                f"cannot register {cls.__module__}.{cls.__qualname__}"
            )
        _REGISTRY[name] = cls
        return cls

    return deco


def _discover() -> None:
    try:
        pkg = importlib.import_module("strategies")
    except ImportError as exc:
        if exc.name == "strategies":
            raise StrategyError(
                "cannot import the 'strategies' package; run from the repo root"
            ) from exc
        raise StrategyError(f"failed to import the 'strategies' package: {exc!r}") from exc
    except Exception as exc:
        raise StrategyError(f"failed to import the 'strategies' package: {exc!r}") from exc
    names = sorted(m.name for m in pkgutil.iter_modules(pkg.__path__) if not m.name.startswith("_"))
    for mod in names:
        full = f"strategies.{mod}"
        try:
            importlib.import_module(full)
        except Exception as exc:
            raise StrategyError(f"failed to import strategy module {full}: {exc!r}") from exc


def list_strategies() -> list[tuple[str, type[Strategy]]]:
    """Import every module in strategies/ and return the registry sorted by name."""
    _discover()
    return sorted(_REGISTRY.items())


def get_strategy(name: str) -> type[Strategy]:
    """Look up a strategy class by name, running discovery on a miss."""
    found = _REGISTRY.get(name)
    if found is not None:
        return found
    _discover()
    found = _REGISTRY.get(name)
    if found is None:
        raise StrategyError(f"unknown strategy {name!r}; known: {sorted(_REGISTRY)}")
    return found


def build_params(strategy_cls: type[Strategy], values: Mapping[str, object]) -> BaseModel:
    """Validate `values` against strategy_cls.Params; unknown keys or errors -> StrategyError."""
    unknown = sorted(set(values) - set(strategy_cls.Params.model_fields))
    if unknown:
        raise StrategyError(f"unknown params for {strategy_cls.__qualname__}: {unknown}")
    try:
        return strategy_cls.Params.model_validate(dict(values))
    except ValidationError as exc:
        raise StrategyError(f"invalid params for {strategy_cls.__qualname__}: {exc}") from exc
