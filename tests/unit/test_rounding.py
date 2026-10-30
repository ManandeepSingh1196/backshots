from decimal import InvalidOperation

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from bt.core.errors import InvariantError
from bt.core.rounding import floor_to_step


@pytest.mark.parametrize(
    ("v", "s", "want"),
    [
        (1.2345, 0.001, 1.234),
        (0.3, 0.1, 0.3),
        (0.0, 0.1, 0.0),
        (9.99999, 1.0, 9.0),
        (3e-05, 1e-05, 3e-05),
    ],
)
def test_exact_floor(v: float, s: float, want: float) -> None:
    assert floor_to_step(v, s) == want


def test_numpy_scalars_return_python_float() -> None:
    r = floor_to_step(np.float64(1.2345), 0.001)
    assert r == 1.234
    assert type(r) is float
    r32 = floor_to_step(float(np.float32(0.5)), 0.25)
    assert r32 == 0.5
    r32b = floor_to_step(np.float32(0.5), 0.25)  # pyright: ignore[reportArgumentType]
    assert r32b == 0.5
    assert type(r32b) is float


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), True, np.True_, None])
def test_bad_value(bad: object) -> None:
    with pytest.raises(InvariantError):
        floor_to_step(bad, 0.1)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan"), True])
def test_bad_step(bad: float) -> None:
    with pytest.raises(InvariantError):
        floor_to_step(1.0, bad)


def test_huge_value_chained() -> None:
    with pytest.raises(InvariantError) as ei:
        floor_to_step(1e300, 1e-8)
    assert isinstance(ei.value.__cause__, InvalidOperation)


@given(
    v=st.floats(min_value=0, max_value=1e9, allow_nan=False),
    s=st.sampled_from([1e-5, 0.001, 0.1, 1.0]),
)
def test_floor_property(v: float, s: float) -> None:
    r = floor_to_step(v, s)
    assert 0 <= v - r <= s * (1 + 1e-9)
    assert floor_to_step(r, s) == r
