"""Look-ahead canary: strategies cannot reach future data and earn nothing by trying."""

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from bt.core.errors import LookAheadError
from bt.core.events import EventLog
from bt.core.types import Bar
from bt.data.catalog import Catalog
from bt.data.feed import DataFeed, ParquetFeed
from bt.engine.loop import EngineConfig, EngineResult, run_engine
from bt.execution.broker_sim import BrokerSim
from bt.execution.fees import VenueFees
from bt.execution.slippage import FixedBps
from bt.portfolio.portfolio import Portfolio
from bt.strategy.base import Strategy
from tests.support.canary import (
    LeakyFeed,
    OracleCanary,
    ProbeStrategy,
    SneakyStrategy,
    TrendCanary,
)
from tests.support.make_golden_fixture import AAA as FIXTURE_AAA
from tests.support.make_golden_fixture import day, frame
from tests.support.memory_feed import MemoryFeed

SYM = "AAAUSDT"
INST = replace(FIXTURE_AAA, maker_fee=0.0, taker_fee=0.0)
# O -> C for d1..d6; H = max(O, C), L = min(O, C).
OC = [
    (100.0, 110.0),
    (110.0, 99.0),
    (99.0, 108.9),
    (108.9, 98.01),
    (98.01, 107.811),
    (107.811, 97.0299),
]
Table = list[tuple[int, float, float, float, float]]
TABLE: Table = [(i + 1, o, max(o, c), min(o, c), c) for i, (o, c) in enumerate(OC)]
CUTOFFS = [1, 2, 3, 4, 5]


def to_bars(table: Table) -> list[Bar]:
    return [Bar(SYM, day(d), o, h, lo, c, 10.0, 10.0 * c) for d, o, h, lo, c in table]


def execute(strategy: Strategy, feed: DataFeed) -> EngineResult:
    log = EventLog()
    pf = Portfolio(1000.0, [INST])
    broker = BrokerSim([INST], VenueFees(), FixedBps(0), pf.view(), log)
    return run_engine(strategy, feed, broker, pf, log, EngineConfig())


def parquet_feed(tmp_path: Path, table: Table) -> ParquetFeed:
    cat = Catalog(tmp_path)
    cat.write_klines(
        "spot", "1d", SYM, frame(tuple(table)), (), {}, datetime(2024, 2, 1, tzinfo=day(1).tzinfo)
    )
    return ParquetFeed(cat, [INST], "1d", day(1), day(7))


def mem(table: Table = TABLE, feed_cls: type[MemoryFeed] = MemoryFeed) -> MemoryFeed:
    return feed_cls([INST], to_bars(table))


def trend(table: Table = TABLE) -> tuple[TrendCanary, EngineResult]:
    strat = TrendCanary(Strategy.Params(), [INST])
    return strat, execute(strat, mem(table))


HONEST = 1000 - 55 - 54.45 - 53.9055
ORACLE = 1000 + 49.5 + 49.005


def test_honest_trend_canary_earns_nothing() -> None:
    _, res = trend()
    assert res.equity[-1].equity == pytest.approx(HONEST, abs=1e-6)
    assert pytest.approx(836.6445, abs=1e-6) == HONEST
    assert [f.price for f in res.fills] == pytest.approx(
        [110.0, 99.0, 108.9, 98.01, 107.811, 97.0299], abs=1e-9
    )


def test_honest_trend_canary_over_parquet_feed(tmp_path: Path) -> None:
    strat = TrendCanary(Strategy.Params(), [INST])
    res = execute(strat, parquet_feed(tmp_path, TABLE))
    assert res.equity[-1].equity == pytest.approx(836.6445, abs=1e-6)


def test_oracle_discriminates() -> None:
    strat = OracleCanary(Strategy.Params(), [INST], to_bars(TABLE))
    res = execute(strat, mem())
    assert res.equity[-1].equity == pytest.approx(1098.505, abs=1e-6)
    assert pytest.approx(1098.505, abs=1e-6) == ORACLE


def test_leaky_feed_is_caught_by_step_context() -> None:
    strat = TrendCanary(Strategy.Params(), [INST])
    with pytest.raises(LookAheadError, match="after now"):
        execute(strat, mem(feed_cls=LeakyFeed))


def test_sneaky_strategy_memory_feed() -> None:
    feed = mem()
    with pytest.raises(LookAheadError):
        execute(SneakyStrategy(Strategy.Params(), [INST], feed), feed)


def test_sneaky_strategy_parquet_feed(tmp_path: Path) -> None:
    feed = parquet_feed(tmp_path, TABLE)
    with pytest.raises(LookAheadError):
        execute(SneakyStrategy(Strategy.Params(), [INST], feed), feed)


def perturbed(k: int) -> Table:
    rng = np.random.default_rng(k)
    out = list(TABLE[:k])
    for d in range(k + 1, 7):
        o, c = (float(x) for x in rng.uniform(50.0, 150.0, 2))
        h = max(o, c) + float(rng.uniform(0.0, 10.0))
        lo = min(o, c) - float(rng.uniform(0.0, 10.0))
        out.append((d, o, h, lo, c))
    return out


@pytest.mark.parametrize("k", CUTOFFS)
def test_future_perturbation_invariance(k: int) -> None:
    cut = day(k)
    s1, r1 = trend(TABLE)
    s2, r2 = trend(perturbed(k))
    assert r1.equity[-1] != r2.equity[-1]  # the futures really differ
    assert [o for o in r1.orders if o.ts <= cut] == [o for o in r2.orders if o.ts <= cut]
    assert [e for e in r1.equity if e.ts <= cut] == [e for e in r2.equity if e.ts <= cut]
    snaps1 = [s for s in s1.snapshots if s[0] <= cut]
    snaps2 = [s for s in s2.snapshots if s[0] <= cut]
    assert len(snaps1) == k
    assert snaps1 == snaps2


def test_probe_contract_holds_on_every_step() -> None:
    probe = ProbeStrategy(Strategy.Params(), [INST])
    execute(probe, mem())
    assert len(probe.checks) == 6 * 10
    failed = [name for name, ok in probe.checks if not ok]
    assert failed == []
    assert {name for name, _ in probe.checks} >= {
        "last_ts_is_now",
        "length_is_steps_so_far",
        "no_funding",
        "negative_n_rejected",
        "ts_readonly",
        "close_readonly",
    }
