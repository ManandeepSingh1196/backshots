"""run_backtest: gates, warmup, engine-config mapping, wiring, determinism, pickling."""

import math
import pickle
import socket
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

import bt.engine.backtest as bt_backtest
from bt.core.errors import ConfigError, DataError, EngineError, PortfolioError, StrategyError
from bt.core.events import EventLog
from bt.core.types import Instrument
from bt.data.feed import DataFeed
from bt.engine.backtest import RunResult, run_backtest
from bt.engine.loop import EngineConfig, EngineResult
from bt.execution.broker import BacktestBroker
from bt.execution.broker_sim import BrokerSim
from bt.portfolio.portfolio import Portfolio
from bt.runs.config import RunConfig
from bt.runs.writer import DETERMINISTIC_FILES
from bt.strategy.base import Strategy
from bt.strategy.registry import get_strategy
from tests.support.buy_once import BuyOnce
from tests.support.make_golden_fixture import FIXTURE_DIR

TOL = 1e-9


def test_strategy_registered() -> None:
    assert get_strategy("test_buy_once") is BuyOnce


def cfg(
    *,
    symbols: Sequence[str] = ("AAAUSDT",),
    start: str = "2024-01-01",
    strategy: str = "test_buy_once",
    kind: str = "spot",
    params: dict[str, object] | None = None,
    costs: dict[str, object] | None = None,
    engine: dict[str, object] | None = None,
    run: dict[str, object] | None = None,
) -> RunConfig:
    return RunConfig.model_validate(
        {
            "run": {
                "strategy": strategy,
                "start": start,
                "end": "2024-01-05",
                "initial_equity_inr": 85000,
                "inr_per_usdt": 85,
                **(run or {}),
            },
            "universe": {"symbols": list(symbols), "kind": kind},
            "params": {"buys": {"AAAUSDT": 5.0}} if params is None else params,
            "costs": {"slippage": {"model": "fixed_bps", "bps": 10}, **(costs or {})},
            "engine": engine or {},
        }
    )


def visible(runs: Path) -> list[Path]:
    return [p for p in runs.iterdir() if not p.name.startswith(".")] if runs.exists() else []


def test_warmup_feeds_history_before_start(tmp_path: Path) -> None:
    c = cfg(start="2024-01-03", params={"buys": {"AAAUSDT": 5.0}, "warmup": 2})
    res = run_backtest(c, data_dir=FIXTURE_DIR, runs_dir=tmp_path)
    from bt.runs.reader import read_run

    data = read_run(res.run_dir)
    assert data.equity.height == 3
    assert data.equity["ts"].to_list() == [datetime(2024, 1, d, tzinfo=UTC) for d in (3, 4, 5)]
    assert data.equity["equity_after_tax"].to_list() == pytest.approx(
        [1000.0, 1003.889445, 1012.750015], abs=TOL
    )
    assert data.trades["price"][0] == pytest.approx(111.111, abs=TOL)
    assert data.trades["ts"][0] == datetime(2024, 1, 4, tzinfo=UTC)


@pytest.mark.parametrize(
    ("c", "exc", "match"),
    [
        (cfg(kind="perp"), EngineError, "perps"),
        (cfg(costs={"tax": {"model": "india_vda"}}), ConfigError, "india_vda"),
        (cfg(symbols=("AAAUSDT", "CCCUSDT")), DataError, "CCCUSDT"),
        (cfg(strategy="no_such_strategy"), StrategyError, "no_such_strategy"),
        (cfg(params={"nope": 1}), StrategyError, "nope"),
    ],
)
def test_gates_write_nothing(
    tmp_path: Path, c: RunConfig, exc: type[Exception], match: str
) -> None:
    with pytest.raises(exc, match=match):
        run_backtest(c, data_dir=FIXTURE_DIR, runs_dir=tmp_path)
    assert visible(tmp_path) == []


def test_missing_symbols_listed_sorted(tmp_path: Path) -> None:
    c = cfg(symbols=("ZZZUSDT", "CCCUSDT", "AAAUSDT"))
    with pytest.raises(DataError, match=r"\['CCCUSDT', 'ZZZUSDT'\]"):
        run_backtest(c, data_dir=FIXTURE_DIR, runs_dir=tmp_path)


class _Spy:
    def __init__(self) -> None:
        self.engine_cfg: EngineConfig | None = None
        self.strategy: Strategy | None = None
        self.feed: DataFeed | None = None
        self.broker: BacktestBroker | None = None
        self.pf: Portfolio | None = None
        self.broker_insts: tuple[Instrument, ...] = ()
        self.ttl: int | None = None


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> _Spy:
    s = _Spy()
    real_engine = bt_backtest.run_engine
    real_broker = bt_backtest.BrokerSim

    def fake_engine(
        strategy: Strategy,
        feed: DataFeed,
        broker: BacktestBroker,
        portfolio: Portfolio,
        log: EventLog,
        config: EngineConfig,
    ) -> EngineResult:
        s.strategy, s.feed, s.broker, s.pf, s.engine_cfg = strategy, feed, broker, portfolio, config
        return real_engine(strategy, feed, broker, portfolio, log, config)

    def fake_broker(
        instruments: Sequence[Instrument], *args: object, ttl_bars: int, **kwargs: object
    ) -> BrokerSim:
        s.broker_insts = tuple(instruments)
        s.ttl = ttl_bars
        return real_broker(instruments, *args, ttl_bars=ttl_bars, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(bt_backtest, "run_engine", fake_engine)
    monkeypatch.setattr(bt_backtest, "BrokerSim", fake_broker)
    return s


@pytest.mark.parametrize(
    ("engine", "run", "kill", "close"),
    [
        ({"max_drawdown_kill": "off"}, {}, None, True),
        ({"max_drawdown_kill": 0.25}, {}, 0.25, True),
        ({}, {"close_at_end": False}, 0.40, False),
    ],
)
def test_engine_config_mapping(
    tmp_path: Path,
    spy: _Spy,
    engine: dict[str, object],
    run: dict[str, object],
    kill: float | None,
    close: bool,
) -> None:
    run_backtest(cfg(engine=engine, run=run), data_dir=FIXTURE_DIR, runs_dir=tmp_path)
    assert spy.engine_cfg is not None
    assert spy.engine_cfg.max_drawdown_kill == kill
    assert spy.engine_cfg.close_at_end is close


def test_engine_config_other_fields(tmp_path: Path, spy: _Spy) -> None:
    c = cfg(engine={"rebalance_band": 0.05, "global_max_leverage": 3.0})
    run_backtest(c, data_dir=FIXTURE_DIR, runs_dir=tmp_path)
    assert spy.engine_cfg is not None
    assert spy.engine_cfg.rebalance_band == pytest.approx(0.05, abs=TOL)
    assert spy.engine_cfg.global_max_leverage == pytest.approx(3.0, abs=TOL)


def test_ttl_reaches_broker(tmp_path: Path, spy: _Spy) -> None:
    run_backtest(cfg(engine={"order_ttl_bars": 5}), data_dir=FIXTURE_DIR, runs_dir=tmp_path)
    assert spy.ttl == 5


def test_one_instrument_tuple(tmp_path: Path, spy: _Spy) -> None:
    c = cfg(symbols=("BBBUSDT", "AAAUSDT"))
    run_backtest(c, data_dir=FIXTURE_DIR, runs_dir=tmp_path)
    assert spy.strategy is not None
    assert spy.feed is not None
    assert spy.pf is not None
    assert spy.strategy.instruments == tuple(spy.feed.instruments())
    assert spy.broker_insts == spy.strategy.instruments
    assert [i.symbol for i in spy.strategy.instruments] == ["AAAUSDT", "BBBUSDT"]
    for inst in spy.broker_insts:
        assert spy.pf.position(inst.symbol).qty == pytest.approx(0.0, abs=TOL)
    with pytest.raises(PortfolioError):
        spy.pf.position("ZZZUSDT")


def test_same_minute_runs_are_distinct_and_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bt_backtest, "_utcnow", lambda: datetime(2024, 3, 1, 12, 30, 5, tzinfo=UTC))
    a = run_backtest(cfg(), data_dir=FIXTURE_DIR, runs_dir=tmp_path)
    b = run_backtest(cfg(), data_dir=FIXTURE_DIR, runs_dir=tmp_path)
    assert a.run_dir != b.run_dir
    assert b.run_dir.name == a.run_dir.name + "-2"
    assert a.run_dir.name.startswith("2024-03-01T1230_test_buy_once_")
    for name in DETERMINISTIC_FILES:
        assert (a.run_dir / name).read_bytes() == (b.run_dir / name).read_bytes(), name


def test_result_pickles(tmp_path: Path) -> None:
    res = run_backtest(cfg(), data_dir=FIXTURE_DIR, runs_dir=tmp_path)
    back: RunResult = pickle.loads(pickle.dumps(res))
    assert back.run_dir == res.run_dir
    assert back.liquidated == res.liquidated
    assert set(back.metrics) == set(res.metrics)
    for k, v in res.metrics.items():
        w = back.metrics[k]
        assert (math.isnan(v) and math.isnan(w)) or v == w, k


def test_no_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", boom)
    run_backtest(cfg(), data_dir=FIXTURE_DIR, runs_dir=tmp_path)


# ---- T033: BTC buy-and-hold benchmark ----

BTC_OPEN = [100.0, 100.2, 108.0, 111.0, 113.0]
BTC_CLOSE = [100.0, 108.0, 110.0, 112.0, 114.0]


def _btc_catalog(
    root: Path, *, close: Sequence[float] = BTC_CLOSE, days: Sequence[int] = (1, 2, 3, 4, 5)
) -> None:
    import polars as pl

    from bt.core.types import InstrumentKind
    from bt.data.catalog import Catalog
    from bt.data.validate import KLINE_SCHEMA
    from tests.support.make_golden_fixture import AAA, AAA_BARS, INGESTED_AT, day, frame

    def btc_inst() -> Instrument:
        return Instrument(
            symbol="BTCUSDT",
            venue_symbol="BTCUSDT",
            kind=InstrumentKind.SPOT,
            base="BTC",
            quote="USDT",
            min_qty=0.001,
            qty_step=0.001,
            min_notional=5.0,
            price_tick=0.01,
            maker_fee=0.001,
            taker_fee=0.001,
        )

    cat = Catalog(root)
    cat.write_klines("spot", "1d", "AAAUSDT", frame(AAA_BARS), ("t",), {}, INGESTED_AT)
    if days:
        n = len(days)
        btc = pl.DataFrame(
            {
                "ts": [day(d) for d in days],
                "open": [BTC_OPEN[d - 1] for d in days],
                "high": [max(BTC_OPEN[d - 1], close[d - 1]) + 1 for d in days],
                "low": [min(BTC_OPEN[d - 1], close[d - 1]) - 1 for d in days],
                "close": [close[d - 1] for d in days],
                "volume": [10.0] * n,
                "quote_volume": [10.0 * close[d - 1] for d in days],
                "trades": [1] * n,
            },
            schema=dict(KLINE_SCHEMA),
        )
        cat.write_klines("spot", "1d", "BTCUSDT", btc, ("t",), {}, INGESTED_AT)
    cat.write_instruments([AAA, btc_inst()], INGESTED_AT)


def test_benchmark_metrics_and_report(tmp_path: Path) -> None:
    from bt.runs.reader import read_run

    _btc_catalog(tmp_path / "d")
    res = run_backtest(cfg(), data_dir=tmp_path / "d", runs_dir=tmp_path / "r")
    assert res.metrics["benchmark_total_return"] == pytest.approx(0.132913841172, abs=TOL)
    data = read_run(res.run_dir)
    assert data.metrics["benchmark_total_return"] == pytest.approx(0.132913841172, abs=TOL)
    row = next(ln for ln in data.report.splitlines() if ln.startswith("Total return"))
    assert "13.29%" in row


def test_benchmark_hash_covers_btc(tmp_path: Path) -> None:
    from bt.runs.reader import read_run

    _btc_catalog(tmp_path / "a")
    _btc_catalog(tmp_path / "b", close=[100.0, 108.0, 110.0, 112.0, 115.0])
    ha = read_run(run_backtest(cfg(), data_dir=tmp_path / "a", runs_dir=tmp_path / "r").run_dir)
    hb = read_run(run_backtest(cfg(), data_dir=tmp_path / "b", runs_dir=tmp_path / "r").run_dir)
    assert ha.meta.hash8 != hb.meta.hash8


def test_benchmark_in_universe_fingerprinted_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _btc_catalog(tmp_path / "d")
    seen: list[tuple[str, ...]] = []
    real = bt_backtest.data_fingerprint

    def spy_fp(catalog: object, insts: Sequence[Instrument], *a: object) -> object:
        seen.append(tuple(i.symbol for i in insts))
        return real(catalog, insts, *a)  # type: ignore[arg-type]

    monkeypatch.setattr(bt_backtest, "data_fingerprint", spy_fp)
    run_backtest(cfg(), data_dir=tmp_path / "d", runs_dir=tmp_path / "r")
    run_backtest(
        cfg(symbols=("AAAUSDT", "BTCUSDT"), params={"buys": {"AAAUSDT": 5.0}}),
        data_dir=tmp_path / "d",
        runs_dir=tmp_path / "r",
    )
    assert seen == [("AAAUSDT", "BTCUSDT"), ("AAAUSDT", "BTCUSDT")]


def test_benchmark_no_rows_in_window_is_nan(tmp_path: Path) -> None:
    _btc_catalog(tmp_path / "d", days=(1, 2))  # BTC ends before the window starts
    res = run_backtest(cfg(start="2024-01-03"), data_dir=tmp_path / "d", runs_dir=tmp_path / "r")
    assert all(math.isnan(v) for k, v in res.metrics.items() if k.startswith("benchmark_"))


def test_benchmark_missing_klines_is_nan(tmp_path: Path) -> None:
    _btc_catalog(tmp_path / "d", days=())
    res = run_backtest(cfg(), data_dir=tmp_path / "d", runs_dir=tmp_path / "r")
    assert all(math.isnan(v) for k, v in res.metrics.items() if k.startswith("benchmark_"))


def test_benchmark_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bt_backtest, "_utcnow", lambda: datetime(2024, 3, 1, 12, 30, 5, tzinfo=UTC))
    _btc_catalog(tmp_path / "d")
    a = run_backtest(cfg(), data_dir=tmp_path / "d", runs_dir=tmp_path / "r")
    b = run_backtest(cfg(), data_dir=tmp_path / "d", runs_dir=tmp_path / "r")
    assert (a.run_dir / "metrics.json").read_bytes() == (b.run_dir / "metrics.json").read_bytes()
