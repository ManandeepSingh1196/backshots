"""Suite-wide checks."""

import pytest

from bt.strategy import registry

# Registered at import by design (tests/support); every other test-only name must not leak.
_ALLOWED = frozenset({"test_buy_once"})


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    leaked = sorted(
        n
        for n, cls in registry._REGISTRY.items()  # pyright: ignore[reportPrivateUsage]
        if cls.__module__.startswith("tests.") and n not in _ALLOWED
    )
    if leaked:
        print(f"\ntest-only strategies leaked into the global registry: {leaked}")
        session.exitstatus = 1
