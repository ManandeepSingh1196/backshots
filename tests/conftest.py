"""Suite-wide checks."""

from collections.abc import Mapping

import pytest

from bt.strategy import registry

# Registered at import by design (tests/support); every other test-only name must not leak.
_ALLOWED = frozenset({"test_buy_once"})


def leaked_strategies(registry_items: Mapping[str, type[object]]) -> list[str]:
    """Names of registered classes defined outside `strategies.` and not in _ALLOWED."""
    return sorted(
        n
        for n, cls in registry_items.items()
        if not cls.__module__.startswith("strategies.") and n not in _ALLOWED
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    leaked = leaked_strategies(registry._REGISTRY)  # pyright: ignore[reportPrivateUsage]
    if leaked:
        print(f"\ntest-only strategies leaked into the global registry: {leaked}")
        session.exitstatus = 1
