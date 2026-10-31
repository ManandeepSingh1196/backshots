"""M5 golden target-weights run on the golden_m2 fixture. Every number comes from an
independent hand-computed reference model (ticket T038), never from the engine's own output."""

import json
from pathlib import Path

import polars as pl
import pytest

import bt.engine.backtest as bt_backtest
import bt.engine.loop as bt_loop
from bt.core.errors import StrategyError
from bt.engine.backtest import run_backtest
from bt.engine.loop import EngineConfig
from bt.runs.config import load_config
from bt.runs.reader import RunData, read_run
from bt.runs.writer import DETERMINISTIC_FILES
from bt.strategy.registry import get_strategy
from tests.support.make_golden_fixture import FIXTURE_DIR
from tests.support.target_weights import TestWeights

A, B = "AAAUSDT", "BBBUSDT"

TEMPLATE = """
[run]
strategy = "test_weights"
start = 2024-01-01
end = 2024-01-05
initial_equity_inr = 85000
inr_per_usdt = 85

[universe]
symbols = {symbols}
kind = "spot"

[params]
targets = {targets}

[costs.slippage]
model = "fixed_bps"
bps = 10
"""

MAIN_TARGETS = (
    "[{AAAUSDT = 0.5, BBBUSDT = 0.25}, {AAAUSDT = 0.25, BBBUSDT = 0.5}, {BBBUSDT = 0.5}, {}]"
)


def go(tmp_path: Path, symbols: str, targets: str, name: str = "runs") -> Path:
    cfg = tmp_path / f"{name}.toml"
    cfg.write_text(TEMPLATE.format(symbols=symbols, targets=targets))
    return run_backtest(load_config(cfg), data_dir=FIXTURE_DIR, runs_dir=tmp_path / name).run_dir


def test_registered() -> None:
    assert get_strategy("test_weights") is TestWeights


@pytest.fixture(scope="module")
def main_run(tmp_path_factory: pytest.TempPathFactory) -> RunData:
    tmp = tmp_path_factory.mktemp("w")
    return read_run(go(tmp, f'["{A}", "{B}"]', MAIN_TARGETS))


def test_fills(main_run: RunData) -> None:
    t = main_run.trades
    got = list(
        zip(
            [x.date().day for x in t["ts"].to_list()],
            t["symbol"].to_list(),
            t["side"].to_list(),
            t["qty"].to_list(),
            t["price"].to_list(),
            strict=True,
        )
    )
    want = [
        (2, A, "buy", 5.0, 102.102),
        (2, B, "buy", 12.5, 20.02),
        (3, A, "sell", 2.59, 107.892),
        (4, A, "sell", 2.41, 110.889),
        (4, B, "buy", 12.2, 22.022),
        (5, B, "sell", 24.7, 22.977),
    ]
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        assert g[:3] == w[:3]
        assert g[3] == pytest.approx(w[3], abs=1e-9)
        assert g[4] == pytest.approx(w[4], abs=1e-9)


def test_equity_and_costs(main_run: RunData) -> None:
    want = [1000, 1040.97924, 1045.24007972, 1071.42825883, 1082.64262693]
    assert main_run.equity["equity_pre_tax"].to_list() == pytest.approx(want, abs=1e-6)
    m = main_run.metrics
    assert m["cost_fees_usdt"] == pytest.approx(2.14364307, abs=1e-6)
    assert m["cost_slippage_usdt"] == pytest.approx(2.14373, abs=1e-6)
    assert m["n_rejected"] == 0
    assert m["n_skipped"] == 0


def test_open_order_netting(main_run: RunData) -> None:
    o = main_run.orders
    bbb = o.filter((pl.col("symbol") == B) & (pl.col("status") == "new"))
    assert bbb["order_id"].to_list() == [
        "bt-rb-BBBUSDT-202401010000",
        "bt-rb-BBBUSDT-202401020000",
        "bt-rb-BBBUSDT-202401040000",
    ]


def test_ids_tags_and_timing(main_run: RunData) -> None:
    o = main_run.orders
    assert all(i.startswith("bt-rb-") for i in o["order_id"].to_list())
    assert set(o["tag"].to_list()) == {"rebalance"}
    decided = {
        r["order_id"]: r["ts"] for r in o.filter(pl.col("status") == "new").iter_rows(named=True)
    }
    for r in main_run.trades.iter_rows(named=True):
        assert r["ts"] > decided[r["order_id"]]


def test_flat_at_end(main_run: RunData) -> None:
    assert not any(i.startswith("bt-close") for i in main_run.orders["order_id"].to_list())
    assert main_run.equity["net_exposure"].to_list()[-1] == pytest.approx(0.0, abs=1e-9)
    assert main_run.equity["gross_exposure"].to_list()[-1] == pytest.approx(0.0, abs=1e-9)


GAP_TARGETS = "[{AAAUSDT = 1.0}, {AAAUSDT = 1.0}, {AAAUSDT = 1.0}, {AAAUSDT = 1.0}]"


def test_gap_handling(tmp_path: Path) -> None:
    r = read_run(go(tmp_path, f'["{A}"]', GAP_TARGETS))
    o = r.orders
    d1 = o.filter(pl.col("order_id") == "bt-rb-AAAUSDT-202401010000")
    assert d1["qty"].to_list()[0] == pytest.approx(9.93, abs=1e-9)
    rej = d1.filter(pl.col("status") == "rejected")
    assert rej["reason"].to_list() == ["insufficient_cash_at_fill"]
    assert rej["ts"].to_list()[0].day == 2
    d2 = o.filter(pl.col("order_id") == "bt-rb-AAAUSDT-202401020000")
    assert d2["qty"].to_list()[0] == pytest.approx(9.194, abs=1e-9)
    f = r.trades.filter(pl.col("order_id") == "bt-rb-AAAUSDT-202401020000")
    assert f["ts"].to_list()[0].day == 3
    assert f["price"].to_list()[0] == pytest.approx(108.108, abs=1e-9)
    want = [1000, 1000, 1016.401103048, 1034.789103048, 1051.081919164]
    assert r.equity["equity_pre_tax"].to_list() == pytest.approx(want, abs=1e-6)


def test_buy_buffer_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[EngineConfig] = []
    real = bt_loop.run_engine

    def spy(*args: object, **kw: object) -> object:
        cfg = args[5]
        assert isinstance(cfg, EngineConfig)
        seen.append(cfg)
        return real(*args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(bt_backtest, "run_engine", spy)
    base = TEMPLATE.format(symbols=f'["{A}"]', targets="[]")
    p = tmp_path / "c.toml"
    for bps, gap, want in ((5, None, 0.0055), (10, 0.01, 0.011)):
        text = base.replace("bps = 10", f"bps = {bps}")
        if gap is not None:
            text += f"\n[engine]\ngap_buffer = {gap}\n"
        p.write_text(text)
        run_backtest(load_config(p), data_dir=FIXTURE_DIR, runs_dir=tmp_path / f"r{bps}")
        assert seen[-1].buy_buffer == pytest.approx(want, abs=1e-12)


def test_determinism(tmp_path: Path) -> None:
    a = go(tmp_path, f'["{A}", "{B}"]', MAIN_TARGETS, "r1")
    b = go(tmp_path, f'["{A}", "{B}"]', MAIN_TARGETS, "r2")
    for name in DETERMINISTIC_FILES:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    assert json.loads((a / "meta.json").read_text())["hash8"]


def test_errors_write_nothing(tmp_path: Path) -> None:
    for targets, who in (("[{ZZZUSDT = 0.1}]", "ZZZUSDT"), ("[{AAAUSDT = -0.1}]", "AAAUSDT")):
        with pytest.raises(StrategyError, match=who):
            go(tmp_path, f'["{A}"]', targets, "bad")
        assert not (tmp_path / "bad").exists() or not any((tmp_path / "bad").iterdir())
