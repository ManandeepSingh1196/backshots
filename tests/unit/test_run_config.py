import math
import pickle
import tomllib
from datetime import date
from pathlib import Path

import pytest
from pydantic import BaseModel

from bt.core.errors import BtError, ConfigError, RunError, StrategyError
from bt.runs.config import (
    IndiaVdaConfig,
    RunConfig,
    config_to_toml,
    load_config,
    parse_override,
    resolve_config,
)
from bt.strategy.base import Intent, NoAction, Strategy
from bt.strategy.context import Context
from bt.strategy.registry import register


@register("cfgtest")
class CfgTest(Strategy):
    class Params(BaseModel):
        a: int = 1
        b: list[int] = [20, 60]

    def on_bar(self, ctx: Context) -> Intent:
        return NoAction()


@register("cfgnull")
class CfgNull(Strategy):
    class Params(BaseModel):
        x: int | None = None

    def on_bar(self, ctx: Context) -> Intent:
        return NoAction()


MINIMAL = """
[run]
strategy = "cfgtest"
start = 2024-01-01
end = 2024-01-05
"""

FULL = """
[run]
strategy = "cfgtest"
start = "2020-01-01"
end = "2023-12-31"
timeframe = "1d"
initial_equity_inr = 100000
inr_per_usdt = 85.0
close_at_end = true
seed = 7

[universe]
symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
kind = "spot"

[params]
a = 2
b = [20, 60, 120]

[costs]
fees = "venue"
slippage = { model = "fixed_bps", bps = 5 }
tax = TAX

[engine]
rebalance_band = 0.02
order_ttl_bars = 3
global_max_leverage = 2.0
max_drawdown_kill = 0.40
""".replace(
    "TAX",
    '{ model = "india_vda", rate = 0.30, cess = 0.04, perp_taxed_as_vda = true, '
    "funding_taxable = true, tds_rate = 0.01 }",
)

BASE = ["run.strategy=cfgtest", "run.start=2024-01-01", "run.end=2024-01-05"]


def _file(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "c.toml"
    p.write_text(text, encoding="utf-8")
    return p


def _load(tmp_path: Path, text: str, overrides: list[str] | None = None) -> RunConfig:
    return load_config(_file(tmp_path, text), overrides or [])


def test_minimal_defaults(tmp_path: Path) -> None:
    c = _load(tmp_path, MINIMAL)
    assert c.run.timeframe == "1d"
    assert c.run.initial_equity_inr == pytest.approx(100000.0, abs=1e-9)
    assert c.run.inr_per_usdt == pytest.approx(85.0, abs=1e-9)
    assert c.run.seed == 7
    assert c.universe.symbols == ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
    assert c.universe.kind == "spot"
    assert c.costs.slippage.bps == pytest.approx(5.0, abs=1e-12)
    assert c.costs.tax.model == "none"
    assert c.engine.max_drawdown_kill == pytest.approx(0.40, abs=1e-12)
    assert c.engine.order_ttl_bars == 3


def test_full_example(tmp_path: Path) -> None:
    c = _load(tmp_path, FULL)
    assert isinstance(c.costs.tax, IndiaVdaConfig)
    assert c.costs.tax.rate == pytest.approx(0.30, abs=1e-12)
    assert c.costs.tax.cess == pytest.approx(0.04, abs=1e-12)
    assert c.costs.tax.tds_rate == pytest.approx(0.01, abs=1e-12)
    assert c.run.start == date(2020, 1, 1)


def test_overrides_apply(tmp_path: Path) -> None:
    c = _load(
        tmp_path,
        MINIMAL,
        [
            "params.a=3",
            "run.end=2024-02-01",
            "costs.slippage.bps=10",
            "run.timeframe=4h",
            "engine.max_drawdown_kill=off",
            'universe.symbols=["ETHUSDT","BTCUSDT"]',
        ],
    )
    assert c.params["a"] == 3
    assert isinstance(c.params["a"], int)
    assert c.run.end == date(2024, 2, 1)
    assert c.costs.slippage.bps == pytest.approx(10.0, abs=1e-12)
    assert c.run.timeframe == "4h"
    assert c.engine.max_drawdown_kill == "off"
    assert c.universe.symbols == ("ETHUSDT", "BTCUSDT")


def test_overrides_in_order_and_no_file() -> None:
    c = load_config(None, [*BASE, "run.seed=1", "run.seed=2"])
    assert c.run.seed == 2


def test_leaf_table_replaced(tmp_path: Path) -> None:
    c = _load(tmp_path, FULL, ["costs.slippage={bps=10}"])
    assert c.costs.slippage.bps == pytest.approx(10.0, abs=1e-12)


def test_override_through_non_table(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        _load(tmp_path, MINIMAL, ["run.seed.x=1"])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a=1", (("a",), 1)),
        ("a.b=0.3", (("a", "b"), 0.3)),
        ("run.strategy=tsmom", (("run", "strategy"), "tsmom")),
        ('x="a b"', (("x",), "a b")),
        ("x=true", (("x",), True)),
        ("x=[1,2]", (("x",), [1, 2])),
        ("k==v", None),
        ("x=1d", (("x",), "1d")),
        ("x=2024-01-01", (("x",), date(2024, 1, 1))),
    ],
)
def test_parse_override(raw: str, expected: object) -> None:
    if raw == "k==v":
        with pytest.raises(ConfigError):
            parse_override(raw)
        with pytest.raises(ConfigError):
            parse_override("=v")
        return
    assert parse_override(raw) == expected


@pytest.mark.parametrize("raw", ["novalue", "a.=1", "a=", "a b=1", "=v", "x=1\ny=2", "x=07:32:00"])
def test_parse_override_rejects(raw: str) -> None:
    with pytest.raises(ConfigError):
        parse_override(raw)


def test_parse_override_rejects_datetime() -> None:
    with pytest.raises(ConfigError):
        parse_override("x=2024-01-01T00:00:00Z")


def _bad(tmp_path: Path, text: str, overrides: list[str] | None = None) -> None:
    with pytest.raises(ConfigError):
        _load(tmp_path, text, overrides)


@pytest.mark.parametrize(
    "ov",
    [
        ["run.start=2024-02-01"],
        ['run.timeframe="1h"'],
        ["run.initial_equity_inr=0"],
        ["run.inr_per_usdt=true"],
        ["run.inr_per_usdt=nan"],
        ["run.inr_per_usdt=inf"],
        ["run.seed=-1"],
        ["run.seed=true"],
        ["run.seed=1.5"],
        ["run.close_at_end=1"],
        ["run.strategy=Bad"],
        ["run.strategy=bt"],
        ["run.foo=1"],
        ["foo.x=1"],
        ["universe.symbols=[]"],
        ['universe.symbols=["BTCUSDT","BTCUSDT"]'],
        ['universe.symbols=["btc"]'],
        ["universe.kind=margin"],
        ["costs.slippage.bps=10000"],
        ["costs.slippage.bps=-1"],
        ['costs.tax={model="us"}'],
        ['costs.tax={model="india_vda",rate=1}'],
        ["engine.max_drawdown_kill=0"],
        ["engine.max_drawdown_kill=1.5"],
        ["engine.max_drawdown_kill=no"],
        ["engine.max_drawdown_kill=true"],
        ["engine.order_ttl_bars=0"],
        ["engine.order_ttl_bars=2.5"],
        ["engine.global_max_leverage=0.5"],
        ["engine.rebalance_band=-0.1"],
        ["params.d=2024-01-01"],
        ["params.x=nan"],
        ["params.x=[1,inf]"],
        ["run.start=07:32:00"],
        ["run.start=5"],
    ],
)
def test_rejections(tmp_path: Path, ov: list[str]) -> None:
    _bad(tmp_path, MINIMAL, ov)


def test_rejections_files(tmp_path: Path) -> None:
    _bad(tmp_path, MINIMAL + "\n[foo]\nx = 1\n")
    _bad(tmp_path, "[run\nstrategy =")
    _bad(tmp_path, "")  # missing required run section
    with pytest.raises(ConfigError, match="config not found"):
        load_config(tmp_path / "nope.toml")


def test_tax_unknown_model_names_allowed(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="india_vda"):
        _load(tmp_path, MINIMAL, ['costs.tax={model="us"}'])


def test_string_date_accepted(tmp_path: Path) -> None:
    c = _load(tmp_path, MINIMAL, ['run.start="2024-01-01"'])
    assert c.run.start == date(2024, 1, 1)
    assert load_config(None, ["run.strategy=cfgtest", "run.start=2024-01-01", "run.end=2024-01-01"])


def test_null_params_rejected_at_resolve() -> None:
    c = load_config(None, ["run.strategy=cfgnull", *BASE[1:]])
    with pytest.raises(ConfigError, match="null"):
        resolve_config(c)


def test_null_in_raw_params() -> None:
    with pytest.raises(ConfigError, match="no null"):
        RunConfig.model_validate(
            {
                "run": {"strategy": "cfgtest", "start": "2024-01-01", "end": "2024-01-01"},
                "params": {"x": [{"y": None}]},
            }
        )


def test_unicode_round_trip(tmp_path: Path) -> None:
    c = _load(tmp_path, MINIMAL, ['params.s="x"'])
    c2 = RunConfig.model_validate({**c.model_dump(), "params": {"s": "😀é\n\x01\x7f\u2028"}})
    back = RunConfig.model_validate(tomllib.loads(config_to_toml(c2)))
    assert back == c2
    assert back.params["s"] == "😀é\n\x01\x7f\u2028"


def test_resolve() -> None:
    c = load_config(None, BASE)
    r = resolve_config(c)
    assert r.params == {"a": 1, "b": [20, 60]}
    assert list(r.params) == ["a", "b"]
    assert resolve_config(load_config(None, [*BASE, "params.a=2"])).params == {
        "a": 2,
        "b": [20, 60],
    }
    with pytest.raises(StrategyError):
        resolve_config(load_config(None, [*BASE, "params.zz=1"]))
    with pytest.raises(StrategyError):
        resolve_config(load_config(None, ["run.strategy=nope", *BASE[1:]]))
    s = resolve_config(load_config(None, [*BASE, 'universe.symbols=["XRPUSDT","BTCUSDT"]']))
    assert s.universe.symbols == ("BTCUSDT", "XRPUSDT")
    assert resolve_config(r) == r


def _round_trip_cases(tmp_path: Path) -> list[RunConfig]:
    return [
        _load(tmp_path, MINIMAL),
        _load(tmp_path, FULL),
        _load(
            tmp_path,
            MINIMAL,
            [
                "engine.max_drawdown_kill=off",
                "costs.slippage.bps=1e-05",
                'params.s="q\\"u\\\\o\\te"',
                'params.n={z=1, "a b"=[1.5, {y=true}]}',
                "params.e=[]",
                "params.t={}",
                'universe.symbols=["ETHUSDT","BTCUSDT"]',
            ],
        ),
    ]


def test_toml_round_trip(tmp_path: Path) -> None:
    for c in _round_trip_cases(tmp_path):
        text = config_to_toml(c)
        assert RunConfig.model_validate(tomllib.loads(text)) == c
        assert config_to_toml(c) == text
        assert text.endswith("\n")
        assert not text.endswith("\n\n")
        assert all(line == line.rstrip() for line in text.split("\n"))
        heads = [ln for ln in text.split("\n") if ln.startswith("[")]
        assert heads == ["[run]", "[universe]", "[params]", "[costs]", "[engine]"]


def test_resolved_round_trip(tmp_path: Path) -> None:
    r = resolve_config(_load(tmp_path, FULL))
    assert RunConfig.model_validate(tomllib.loads(config_to_toml(r))) == r


def test_special_values_round_trip(tmp_path: Path) -> None:
    c = _round_trip_cases(tmp_path)[2]
    back = RunConfig.model_validate(tomllib.loads(config_to_toml(c)))
    assert back.engine.max_drawdown_kill == "off"
    assert back.costs.slippage.bps == pytest.approx(1e-05, rel=1e-12)
    assert back.params["s"] == 'q"u\\o\te'
    assert math.isfinite(back.costs.slippage.bps)
    assert 'n = { "a b" = [1.5, { y = true }], z = 1 }' in config_to_toml(c)


GOLDEN = """[run]
strategy = "cfgtest"
start = 2024-01-01
end = 2024-01-05
timeframe = "1d"
initial_equity_inr = 100000.0
inr_per_usdt = 85.0
close_at_end = true
seed = 7

[universe]
symbols = ["BNBUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
kind = "spot"

[params]
a = 1
b = [20, 60]

[costs]
fees = "venue"
slippage = { model = "fixed_bps", bps = 5.0 }
tax = { model = "none" }

[engine]
rebalance_band = 0.02
order_ttl_bars = 3
global_max_leverage = 2.0
max_drawdown_kill = 0.4
"""


def test_golden_toml(tmp_path: Path) -> None:
    assert config_to_toml(resolve_config(_load(tmp_path, MINIMAL))) == GOLDEN


def test_immutable_and_pickle(tmp_path: Path) -> None:
    c = _load(tmp_path, FULL)
    with pytest.raises(ValueError, match="frozen"):
        c.run.seed = 1  # pyright: ignore[reportAttributeAccessIssue]
    assert pickle.loads(pickle.dumps(c)) == c


def test_errors_are_bt_errors() -> None:
    assert issubclass(ConfigError, BtError)
    assert issubclass(RunError, BtError)


@pytest.mark.parametrize("key", ["run.initial_equity_inr", "costs.slippage.bps"])
def test_huge_int_is_config_error(tmp_path: Path, key: str) -> None:
    _bad(tmp_path, MINIMAL, [f"{key}=" + "9" * 400])


def test_undecodable_file_is_config_error(tmp_path: Path) -> None:
    p = tmp_path / "bad.toml"
    p.write_bytes(b"\xff\xfe")
    with pytest.raises(ConfigError):
        load_config(p)
