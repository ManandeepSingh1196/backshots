"""run_backtest: RunConfig -> components -> M2 engine -> run directory (DESIGN sections 3, 6, 9)."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path

from bt.analytics.metrics import compute_metrics
from bt.analytics.report import render_report
from bt.core.errors import ConfigError, DataError, EngineError
from bt.core.events import EventLog
from bt.core.types import Instrument
from bt.data.catalog import Catalog
from bt.data.feed import ParquetFeed
from bt.engine.loop import EngineConfig, run_engine
from bt.execution.broker_sim import BrokerSim
from bt.execution.fees import VenueFees
from bt.execution.slippage import FixedBps
from bt.portfolio.portfolio import Portfolio
from bt.runs.config import RunConfig, resolve_config
from bt.runs.fingerprint import canonical_json, data_fingerprint, window_coverage
from bt.runs.frames import equity_frame, messages_frame, orders_frame, trades_frame
from bt.runs.meta import RunMeta, git_info, package_versions
from bt.runs.writer import RunArtifacts, next_run_id, run_hash, write_run
from bt.strategy.registry import build_params, get_strategy


@dataclass(frozen=True, slots=True)
class RunResult:
    """Outcome of one run. Invariants: metrics is a plain dict (picklable);
    run_dir is complete."""

    run_dir: Path
    metrics: Mapping[str, float]
    liquidated: bool


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _select_instruments(catalog: Catalog, cfg: RunConfig) -> tuple[Instrument, ...]:
    all_insts, _ = catalog.read_instruments()
    wanted = set(cfg.universe.symbols)
    selected = [
        i for i in all_insts if i.kind.value == cfg.universe.kind and i.venue_symbol in wanted
    ]
    missing = sorted(wanted - {i.venue_symbol for i in selected})
    if missing:
        raise DataError(f"symbols not in the instrument snapshot: {missing}")
    return tuple(sorted(selected, key=lambda i: i.symbol))


def run_backtest(
    config: RunConfig, *, data_dir: Path = Path("data"), runs_dir: Path = Path("runs")
) -> RunResult:
    """Run one backtest and write its run directory; BtErrors propagate, nothing is written then."""
    cfg = resolve_config(config)
    if cfg.universe.kind == "perp":
        raise EngineError("perps arrive in M6")
    if cfg.costs.tax.model != "none":
        raise ConfigError('india_vda tax arrives in M8; use tax = { model = "none" }')

    catalog = Catalog(data_dir)
    insts = _select_instruments(catalog, cfg)
    market, tf = cfg.universe.kind, cfg.run.timeframe

    strategy_cls = get_strategy(cfg.run.strategy)
    strategy = strategy_cls(build_params(strategy_cls, cfg.params), insts)
    warmup = strategy.warmup_bars()

    start = datetime.combine(cfg.run.start, time(0), UTC)
    end = datetime.combine(cfg.run.end + timedelta(days=1), time(0), UTC)  # end is inclusive
    feed = ParquetFeed(catalog, insts, tf, start, end, warmup_bars=warmup)

    fp = data_fingerprint(catalog, insts, market, tf)
    hash8 = run_hash(cfg, fp)[:8]
    config_sha256 = hashlib.sha256(canonical_json(cfg.model_dump(mode="json")).encode()).hexdigest()

    initial = cfg.run.initial_equity_inr / cfg.run.inr_per_usdt
    log = EventLog()
    pf = Portfolio(initial, insts)
    broker = BrokerSim(
        insts,
        VenueFees(),
        FixedBps(cfg.costs.slippage.bps),
        pf.view(),
        log,
        ttl_bars=cfg.engine.order_ttl_bars,
    )
    kill = cfg.engine.max_drawdown_kill
    ecfg = EngineConfig(
        cfg.run.close_at_end,
        None if kill == "off" else kill,
        cfg.engine.rebalance_band,
        cfg.engine.global_max_leverage,
    )

    started_at = _utcnow()
    run_id = next_run_id(runs_dir, started_at, cfg.run.strategy, hash8)
    result = run_engine(strategy, feed, broker, pf, log, ecfg)

    metrics = compute_metrics(result, tf, initial, cfg.run.inr_per_usdt)
    cov = window_coverage(catalog, insts, market, tf, start, end)
    metrics["missing_bars"] = float(sum(c.missing for c in cov))

    finished_at = _utcnow()
    # The repo root is the current directory; git state is provenance only, not part of the hash.
    git_commit, git_dirty = git_info(Path.cwd())
    meta = RunMeta(
        run_id=run_id,
        strategy=cfg.run.strategy,
        hash8=hash8,
        config_sha256=config_sha256,
        data_sha256=fp.sha256,
        git_commit=git_commit,
        git_dirty=git_dirty,
        versions=package_versions(),
        started_at=started_at,
        finished_at=finished_at,
        coverage=cov,
        killed_at=result.killed_at,
        liquidated=result.liquidated,
    )
    equity = equity_frame(result, tf)
    report = render_report(cfg, metrics, equity, meta)
    artifacts = RunArtifacts(
        config=cfg,
        meta=meta,
        metrics=metrics,
        equity=equity,
        trades=trades_frame(result),
        orders=orders_frame(result),
        messages=messages_frame(result),
        report=report,
    )
    run_dir = write_run(runs_dir, run_id, artifacts)
    return RunResult(run_dir, dict(metrics), result.liquidated)
