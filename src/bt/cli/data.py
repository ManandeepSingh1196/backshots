"""`bt data`: fetch, check and instruments commands. Side effects live in the private factories."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, cast, get_args

import httpx
import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from bt.core.errors import DataError
from bt.data.catalog import Catalog
from bt.data.fetch_ccxt import CcxtExchange, make_exchange
from bt.data.ingest import (
    DEFAULT_START,
    DEFAULT_SYMBOLS,
    IngestResult,
    check_catalog,
    ingest_funding,
    ingest_instruments,
    ingest_klines,
)
from bt.data.validate import Market, Timeframe, ValidationReport, check_symbol

data_app = typer.Typer(no_args_is_help=True)

_EXIT_FAILED = 1
_EXIT_USAGE = 2
_MAX_DETAIL = 50
_WIDE = 140
_DEFAULT_SYMBOLS_ARG = ",".join(DEFAULT_SYMBOLS)

_SymbolsOpt = Annotated[
    str, typer.Option("--symbols", help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT.")
]
_MarketsOpt = Annotated[str, typer.Option("--markets", help="Comma-separated: spot,perp.")]
_DataDirOpt = Annotated[Path, typer.Option("--data-dir", help="Data root directory.")]


def _make_client() -> httpx.Client:
    return httpx.Client(timeout=30.0, follow_redirects=True)


def _make_exchange() -> CcxtExchange:
    return make_exchange()


def _now() -> datetime:
    return datetime.now(UTC)


def _console() -> Console:
    """Plain rich console; wide when not a terminal so dataset names are never wrapped."""
    console = Console(highlight=False)
    if not console.is_terminal:
        console = Console(highlight=False, width=_WIDE)
    return console


@dataclass(frozen=True, slots=True)
class _Row:
    """One dataset's outcome for display: its report and any missing archive URLs."""

    report: ValidationReport
    missing: tuple[str, ...] = ()


def _usage_error(console: Console, message: str) -> typer.Exit:
    console.print(f"[red]error:[/red] {escape(message)}")
    return typer.Exit(_EXIT_USAGE)


def _split(raw: str) -> list[str]:
    return list(dict.fromkeys(p.strip() for p in raw.split(",") if p.strip()))


def _parse_symbols(console: Console, raw: str) -> list[str]:
    symbols = list(dict.fromkeys(s.upper() for s in _split(raw)))
    if not symbols:
        raise _usage_error(console, "no symbols given")
    try:
        for symbol in symbols:
            check_symbol(symbol)
    except DataError as e:
        raise _usage_error(console, str(e)) from e
    return symbols


def _parse_choices(console: Console, raw: str, allowed: tuple[str, ...], what: str) -> list[str]:
    values = _split(raw)
    if not values:
        raise _usage_error(console, f"no {what}s given")
    for v in values:
        if v not in allowed:
            raise _usage_error(console, f"invalid {what} {v!r}: expected one of {allowed}")
    return values


def _parse_markets(console: Console, raw: str) -> list[Market]:
    return cast(list[Market], _parse_choices(console, raw, get_args(Market), "market"))


def _parse_timeframes(console: Console, raw: str) -> list[Timeframe]:
    return cast(list[Timeframe], _parse_choices(console, raw, get_args(Timeframe), "timeframe"))


def _parse_date(console: Console, raw: str, what: str) -> date:
    try:
        return date.fromisoformat(raw)
    except ValueError as e:
        raise _usage_error(console, f"invalid {what} date {raw!r}: expected YYYY-MM-DD") from e


def _ts(value: datetime | None) -> str:
    return "-" if value is None else f"{value:%Y-%m-%d %H:%M}"


def _status(report: ValidationReport) -> str:
    return "[green]ok[/green]" if report.ok else "[red]FAILED[/red]"


def _progress_line(console: Console, row: _Row) -> None:
    r = row.report
    console.print(
        f"{escape(r.dataset)}  rows={r.rows}  first={_ts(r.first_ts)}  last={_ts(r.last_ts)}  "
        f"gaps={len(r.gaps)}  flags={len(r.flags)}  {_status(r)}"
    )


def _summary_table(rows: Sequence[_Row]) -> Table:
    table = Table(title="Datasets")
    table.add_column("dataset", no_wrap=True)
    for name in ("rows", "first", "last", "gaps", "flags"):
        table.add_column(name, justify="right", no_wrap=True)
    table.add_column("status")
    for row in rows:
        r = row.report
        table.add_row(
            escape(r.dataset),
            str(r.rows),
            _ts(r.first_ts),
            _ts(r.last_ts),
            str(len(r.gaps)),
            str(len(r.flags)),
            _status(r),
        )
    return table


def _print_details(console: Console, rows: Sequence[_Row]) -> None:
    def more(shown: int, total: int) -> None:
        if total > shown:
            console.print(f"  ... and {total - shown} more")

    for row in rows:
        r = row.report
        if not (r.errors or r.gaps or r.flags or row.missing):
            continue
        console.print(f"[bold]{escape(r.dataset)}[/bold]")
        for error in r.errors:
            console.print(f"  [red]error: {escape(error)}[/red]")
        for gap in r.gaps[:_MAX_DETAIL]:
            console.print(
                f"  gap: after {_ts(gap.after_ts)} -> before {_ts(gap.before_ts)}, "
                f"missing {gap.missing}"
            )
        more(_MAX_DETAIL, len(r.gaps))
        for flag in r.flags[:_MAX_DETAIL]:
            console.print(f"  flag: {_ts(flag.ts)} {escape(flag.reason)}")
        more(_MAX_DETAIL, len(r.flags))
        for url in row.missing[:_MAX_DETAIL]:
            console.print(f"  [yellow]missing archive: {escape(url)}[/yellow]")
        more(_MAX_DETAIL, len(row.missing))


def _report(console: Console, rows: Sequence[_Row]) -> None:
    console.print(_summary_table(rows))
    _print_details(console, rows)


def _failed_row(dataset: str, error: DataError) -> _Row:
    return _Row(
        ValidationReport(dataset, 0, None, None, (f"{type(error).__name__}: {error}",), (), ())
    )


def _attempt(console: Console, dataset: str, run: Callable[[], IngestResult]) -> _Row:
    """Run one ingest; a DataError becomes a FAILED row so the next dataset still runs."""
    try:
        result = run()
        row = _Row(result.report, result.missing_archives)
    except DataError as e:
        row = _failed_row(dataset, e)
    _progress_line(console, row)
    return row


@data_app.command()
def fetch(
    symbols: _SymbolsOpt = _DEFAULT_SYMBOLS_ARG,
    markets: _MarketsOpt = "spot,perp",
    timeframes: Annotated[
        str, typer.Option("--timeframes", help="Comma-separated: 1d,4h.")
    ] = "1d,4h",
    start: Annotated[
        str, typer.Option("--start", help="First day, YYYY-MM-DD.")
    ] = DEFAULT_START.isoformat(),
    end: Annotated[
        str | None, typer.Option("--end", help="Last day, YYYY-MM-DD (default today).")
    ] = None,
    no_funding: Annotated[bool, typer.Option("--no-funding", help="Skip funding rates.")] = False,
    data_dir: _DataDirOpt = Path("data"),
) -> None:
    """Download and validate candles and funding for the chosen symbols and timeframes."""
    console = _console()
    syms = _parse_symbols(console, symbols)
    mkts = _parse_markets(console, markets)
    tfs = _parse_timeframes(console, timeframes)
    start_d = _parse_date(console, start, "start")
    end_d = None if end is None else _parse_date(console, end, "end")
    now = _now()
    if start_d > (now.date() if end_d is None else end_d):
        raise _usage_error(console, f"start {start_d} is after end {end_d or now.date()}")

    catalog = Catalog(data_dir)
    rows: list[_Row] = []
    with _make_client() as client:
        exchange = _make_exchange()
        for market in mkts:
            for tf in tfs:
                for symbol in syms:
                    rows.append(
                        _attempt(
                            console,
                            f"{market}/klines/{tf}/{symbol}",
                            lambda m=market, t=tf, s=symbol: ingest_klines(
                                catalog, client, exchange, m, t, s, start_d, end_d, now
                            ),
                        )
                    )
        if "perp" in mkts and not no_funding:
            for symbol in syms:
                rows.append(
                    _attempt(
                        console,
                        f"perp/funding/{symbol}",
                        lambda s=symbol: ingest_funding(
                            catalog, client, exchange, s, start_d, end_d, now
                        ),
                    )
                )
    _report(console, rows)
    console.print(
        "[dim]Universe is a fixed list of today's majors, so results carry survivorship "
        "bias (DESIGN.md section 6).[/dim]"
    )
    if not all(r.report.ok for r in rows):
        raise typer.Exit(_EXIT_FAILED)


@data_app.command()
def check(data_dir: _DataDirOpt = Path("data")) -> None:
    """Re-validate stored data against the manifest and list gaps."""
    console = _console()
    try:
        reports = check_catalog(Catalog(data_dir))
    except DataError as e:
        console.print(f"[red]error: {escape(str(e))}[/red]")
        raise typer.Exit(_EXIT_FAILED) from e
    if not reports:
        console.print("no datasets")
        return
    rows = [_Row(r) for r in reports]
    _report(console, rows)
    if not all(r.ok for r in reports):
        raise typer.Exit(_EXIT_FAILED)


@data_app.command()
def instruments(
    symbols: _SymbolsOpt = _DEFAULT_SYMBOLS_ARG,
    markets: _MarketsOpt = "spot,perp",
    data_dir: _DataDirOpt = Path("data"),
) -> None:
    """Refresh exchange rules (lot sizes, fees, leverage) into the catalog."""
    console = _console()
    syms = _parse_symbols(console, symbols)
    mkts = _parse_markets(console, markets)
    try:
        found = ingest_instruments(Catalog(data_dir), _make_exchange(), syms, mkts, _now())
    except DataError as e:
        console.print(f"[red]error: {escape(str(e))}[/red]")
        raise typer.Exit(_EXIT_FAILED) from e
    table = Table(title="Instruments")
    table.add_column("symbol", no_wrap=True)
    table.add_column("kind")
    for name in (
        "min_qty",
        "qty_step",
        "min_notional",
        "price_tick",
        "maker",
        "taker",
        "max_lev",
        "mmr",
    ):
        table.add_column(name, justify="right", no_wrap=True)
    for i in found:
        table.add_row(
            escape(i.symbol),
            i.kind.value,
            f"{i.min_qty:g}",
            f"{i.qty_step:g}",
            f"{i.min_notional:g}",
            f"{i.price_tick:g}",
            f"{i.maker_fee:g}",
            f"{i.taker_fee:g}",
            f"{i.max_leverage:g}",
            f"{i.maint_margin_rate:g}",
        )
    console.print(table)
