"""TSMOM: hand-computed weights, vol annualisation, Params validation."""

import tomllib
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from bt.core.errors import LookAheadError, StrategyError
from bt.core.types import Bar, Instrument, InstrumentKind
from bt.runs.config import RunConfig, config_to_toml, load_config, resolve_config
from bt.strategy.base import TargetWeights
from bt.strategy.context import StepContext
from bt.strategy.registry import build_params, get_strategy
from strategies.tsmom import TSMOM, TSMOMParams
from tests.support.memory_feed import MemoryFeed

TOL = 1e-9
T0 = datetime(2024, 1, 1, tzinfo=UTC)
A = [100, 102, 101, 104, 106, 108]
M = [100, 102, 101, 104, 100, 103]
B = [50, 49, 51, 48, 47, 46]
C = [10] * 6
SERIES = {"A": A, "M": M, "B": B, "C": C}


def inst(s: str) -> Instrument:
    return Instrument(s, s, InstrumentKind.SPOT, s, "USDT", 0.001, 0.001, 5.0, 0.01, 0.001, 0.001)


def bars(sym: str, closes: Sequence[float], step: timedelta, start: int = 0) -> list[Bar]:
    return [Bar(sym, T0 + step * (start + i), c, c, c, c, 1.0, 1.0) for i, c in enumerate(closes)]


def weights(
    syms: list[str],
    step: timedelta = timedelta(days=1),
    series: dict[str, Sequence[float]] | None = None,
    **params: object,
) -> dict[str, float]:
    data = series or SERIES
    all_bars = [b for s in syms for b in bars(s, data[s], step)]
    insts = [inst(s) for s in syms]
    feed = MemoryFeed(insts, all_bars)
    base: dict[str, object] = {"lookbacks": [2, 4], "vol_window": 3}
    strat = TSMOM(build_params(TSMOM, {**base, **params}), insts)
    last: dict[str, float] = {}
    for step_ in feed.steps():
        if step_.ts != T0 + step * 5:
            continue
        ctx = StepContext(step_.ts, feed, _pf(), insts, _log())  # type: ignore[arg-type]
        out = strat.on_bar(ctx)
        assert isinstance(out, TargetWeights)
        last = dict(out.weights)
    return last


def _pf():  # noqa: ANN202
    from bt.portfolio.portfolio import Portfolio

    return Portfolio(1000.0, [inst("A")]).view()


def _log():  # noqa: ANN202
    from bt.core.events import EventLog

    return EventLog()


def test_uncapped() -> None:
    w = weights(["A", "M"], vol_target=0.05)
    assert w["A"] == pytest.approx(0.217831630719, abs=TOL)
    assert w["M"] == pytest.approx(0.016511014609, abs=TOL)


def test_cap() -> None:
    w = weights(["A", "M"])
    assert w["A"] == pytest.approx(0.929543278025, abs=TOL)
    assert w["M"] == pytest.approx(0.070456721975, abs=TOL)
    assert w["A"] + w["M"] == pytest.approx(1.0, abs=1e-12)


def test_universe_counts_inactive() -> None:
    w = weights(["A", "B"], vol_target=0.05)
    assert w["B"] == 0.0
    assert w["A"] == pytest.approx(0.217831630719, abs=TOL)


def test_zero_vol() -> None:
    w = weights(["A", "C"], vol_target=0.05)
    assert w["C"] == 0.0
    assert w["A"] == pytest.approx(0.217831630719, abs=TOL)


def test_falling_is_target_weights_of_zeros() -> None:
    w = weights(["B", "C"])
    assert w == {"B": 0.0, "C": 0.0}


def test_4h_spacing() -> None:
    w = weights(["A", "B"], step=timedelta(hours=4), vol_target=0.05)
    assert w["A"] == pytest.approx(0.088929390851, abs=TOL)
    # vol check: A alone with n=1 and target = vol gives weight 1.0 (cap not hit)
    w1 = weights(["A"], step=timedelta(hours=4), vol_target=0.281121907629)
    assert w1["A"] == pytest.approx(1.0, abs=1e-9)


def test_no_look_ahead() -> None:
    syms = ["A", "M"]
    insts = [inst(s) for s in syms]

    def run(tail: float) -> dict[str, float]:
        data = {s: list(SERIES[s]) + [tail] * 3 for s in syms}
        feed = MemoryFeed(insts, [b for s in syms for b in bars(s, data[s], timedelta(days=1))])
        strat = TSMOM(build_params(TSMOM, {"lookbacks": [2, 4], "vol_window": 3}), insts)
        for st in feed.steps():
            if st.ts == T0 + timedelta(days=5):
                out = strat.on_bar(StepContext(st.ts, feed, _pf(), insts, _log()))  # type: ignore[arg-type]
                assert isinstance(out, TargetWeights)
                with pytest.raises(LookAheadError):
                    feed.history("A", T0 + timedelta(days=6), 3)
                return dict(out.weights)
        raise AssertionError

    assert run(1.0) == run(1e6)


def test_types_and_keys() -> None:
    w = weights(["M", "A", "C"])
    assert list(w) == ["A", "C", "M"]
    assert all(type(v) is float for v in w.values())


def test_warmup() -> None:
    insts = [inst("A")]
    assert TSMOM(build_params(TSMOM, {}), insts).warmup_bars() == 121
    p = build_params(TSMOM, {"lookbacks": [2], "vol_window": 30})
    assert TSMOM(p, insts).warmup_bars() == 31


@pytest.mark.parametrize(
    "bad",
    [
        {"lookbacks": []},
        {"lookbacks": [0]},
        {"lookbacks": [5, 5]},
        {"lookbacks": [True]},
        {"vol_window": 1},
        {"vol_window": True},
        {"vol_target": 0},
        {"vol_target": float("nan")},
        {"max_leverage": 0},
        {"max_leverage": 2.5},
        {"nope": 1},
    ],
)
def test_bad_params(bad: dict[str, object]) -> None:
    with pytest.raises(StrategyError):
        build_params(TSMOM, bad)


def test_params_ok_and_frozen() -> None:
    p = build_params(TSMOM, {"lookbacks": [20, 60]})
    assert isinstance(p, TSMOMParams)
    assert p.lookbacks == (20, 60)
    assert build_params(TSMOM, {}).model_dump(mode="json")["lookbacks"] == [20, 60, 120]
    with pytest.raises(ValidationError):
        p.vol_window = 5  # type: ignore[misc]


def test_resolve_idempotent_and_toml() -> None:
    c = load_config(None, ["run.strategy=tsmom", "run.start=2024-01-01", "run.end=2024-01-05"])
    r = resolve_config(c)
    assert r.params["lookbacks"] == [20, 60, 120]
    assert resolve_config(r) == r
    back = RunConfig.model_validate(tomllib.loads(config_to_toml(r)))
    assert back == r


def test_registered() -> None:
    assert get_strategy("tsmom") is TSMOM
