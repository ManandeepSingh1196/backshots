"""Run configuration: validated TOML schema, dotted overrides, resolution, TOML writer."""

import math
import re
import tomllib
from collections.abc import Sequence
from datetime import date, datetime, time
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from bt.core.errors import ConfigError, DataError
from bt.data.validate import TIMEFRAME_DELTA, Market, Timeframe, check_symbol
from bt.strategy.registry import STRATEGY_NAME_PATTERN, build_params, get_strategy

ParamValue = JsonValue

_KEY_PART = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BARE_WORD = re.compile(r"^[A-Za-z0-9_.:/+-]+$")
_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _number(v: object) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise ValueError(f"expected a number, got {v!r}")
    f = float(v)
    if not math.isfinite(f):
        raise ValueError("number must be finite")
    return f


def _integer(v: object) -> int:
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise ValueError(f"expected an integer, got {v!r}")
    if isinstance(v, float) and not (math.isfinite(v) and v.is_integer()):
        raise ValueError(f"expected an integer, got {v!r}")
    return int(v)


def _day(v: object) -> date:
    if isinstance(v, datetime) or not isinstance(v, date | str):
        raise ValueError(f"expected a date, got {v!r}")
    return v if isinstance(v, date) else date.fromisoformat(v)


_Num = Annotated[float, BeforeValidator(_number)]
_Int = Annotated[int, BeforeValidator(_integer)]
_Day = Annotated[date, BeforeValidator(_day)]
_Bool = Annotated[bool, Field(strict=True)]

_MODEL = ConfigDict(frozen=True, extra="forbid")


class RunSection(BaseModel):
    """Run identity and window. Invariants: frozen; start <= end (inclusive); finite positive amounts."""

    model_config = _MODEL

    strategy: str
    start: _Day
    end: _Day
    timeframe: Timeframe = "1d"
    initial_equity_inr: Annotated[_Num, Field(gt=0)] = 100_000.0
    inr_per_usdt: Annotated[_Num, Field(gt=0)] = 85.0
    close_at_end: _Bool = True
    seed: Annotated[_Int, Field(ge=0)] = 7

    @field_validator("strategy")
    @classmethod
    def _check_strategy(cls, v: str) -> str:
        if not re.fullmatch(STRATEGY_NAME_PATTERN, v):
            raise ValueError(f"strategy must match {STRATEGY_NAME_PATTERN}")
        return v

    @field_validator("timeframe")
    @classmethod
    def _check_timeframe(cls, v: Timeframe) -> Timeframe:
        if v not in TIMEFRAME_DELTA:
            raise ValueError(f"timeframe must be one of {sorted(TIMEFRAME_DELTA)}")
        return v

    @model_validator(mode="after")
    def _check_window(self) -> "RunSection":
        if self.start > self.end:
            raise ValueError(f"start {self.start} is after end {self.end}")
        return self


class UniverseSection(BaseModel):
    """Venue symbols and market. Invariants: frozen; symbols non-empty, unique, valid."""

    model_config = _MODEL

    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
    kind: Market = "spot"

    @field_validator("symbols")
    @classmethod
    def _check_symbols(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("symbols must be non-empty")
        if len(set(v)) != len(v):
            raise ValueError("symbols must be unique")
        for s in v:
            try:
                check_symbol(s)
            except DataError as exc:
                raise ConfigError(f"bad symbol {s!r}: {exc}") from exc
        return v


class FixedBpsSlippage(BaseModel):
    """Fixed-bps slippage. Invariants: frozen; 0 <= bps < 10_000."""

    model_config = _MODEL

    model: Literal["fixed_bps"] = "fixed_bps"
    bps: Annotated[_Num, Field(ge=0, lt=10_000)] = 5.0


class NoTaxConfig(BaseModel):
    """No tax model. Invariants: frozen."""

    model_config = _MODEL

    model: Literal["none"] = "none"


class IndiaVdaConfig(BaseModel):
    """India VDA tax. Invariants: frozen; rate, cess, tds_rate in [0, 1)."""

    model_config = _MODEL

    model: Literal["india_vda"] = "india_vda"
    rate: Annotated[_Num, Field(ge=0, lt=1)] = 0.30
    cess: Annotated[_Num, Field(ge=0, lt=1)] = 0.04
    perp_taxed_as_vda: _Bool = True
    funding_taxable: _Bool = True
    tds_rate: Annotated[_Num, Field(ge=0, lt=1)] = 0.01


class CostsSection(BaseModel):
    """Cost models. Invariants: frozen; tax is discriminated on `model`."""

    model_config = _MODEL

    fees: Literal["venue"] = "venue"
    slippage: FixedBpsSlippage = FixedBpsSlippage()
    tax: Annotated[NoTaxConfig | IndiaVdaConfig, Field(discriminator="model")] = NoTaxConfig()


class EngineSection(BaseModel):
    """Engine knobs. Invariants: frozen; max_drawdown_kill is in (0, 1] or the string "off"."""

    model_config = _MODEL

    rebalance_band: Annotated[_Num, Field(ge=0)] = 0.02
    order_ttl_bars: Annotated[_Int, Field(ge=1)] = 3
    global_max_leverage: Annotated[_Num, Field(ge=1)] = 2.0
    max_drawdown_kill: float | Literal["off"] = 0.40

    @field_validator("max_drawdown_kill", mode="before")
    @classmethod
    def _check_kill(cls, v: object) -> float | str:
        if isinstance(v, str):
            if v != "off":
                raise ValueError('max_drawdown_kill must be a number in (0, 1] or "off"')
            return v
        f = _number(v)
        if not 0.0 < f <= 1.0:
            raise ValueError("max_drawdown_kill must be in (0, 1]")
        return f


def _check_json(v: object, where: str = "params") -> None:
    """Raise ConfigError unless `v` is JSON-like with no null, no date and only finite floats."""
    if v is None:
        raise ConfigError(f"TOML has no null ({where})")
    if isinstance(v, float) and not math.isfinite(v):
        raise ConfigError(f"params must be finite ({where})")
    if isinstance(v, date | time):
        raise ConfigError(f"params may not contain dates or times ({where})")
    if isinstance(v, list | tuple):
        for i, x in enumerate(cast("Sequence[object]", v)):
            _check_json(x, f"{where}[{i}]")
    elif isinstance(v, dict):
        for k, x in cast("dict[str, object]", v).items():
            _check_json(x, f"{where}.{k}")


class RunConfig(BaseModel):
    """A complete, validated run description.

    Invariants: frozen, extra keys forbidden, picklable; every section is validated. Caveat:
    `params` is a plain dict, so it can still be mutated in place; callers must not mutate it.
    """

    model_config = _MODEL

    run: RunSection
    universe: UniverseSection = UniverseSection()
    params: dict[str, ParamValue] = {}
    costs: CostsSection = CostsSection()
    engine: EngineSection = EngineSection()

    @field_validator("params", mode="before")
    @classmethod
    def _check_params(cls, v: object) -> object:
        _check_json(v)
        return v


def _no_times(v: object) -> None:
    if isinstance(v, datetime | time):
        raise ConfigError(f"only dates are allowed, not times or datetimes: {v!r}")
    if isinstance(v, list):
        for x in cast("list[object]", v):
            _no_times(x)
    elif isinstance(v, dict):
        for x in cast("dict[str, object]", v).values():
            _no_times(x)


def parse_override(raw: str) -> tuple[tuple[str, ...], ParamValue | date]:
    """Parse `a.b=value` into (key path, value); value is TOML, or a bare word as a string."""
    key, sep, value = raw.partition("=")
    if not sep:
        raise ConfigError(f"override {raw!r} has no '='")
    parts = tuple(key.split("."))
    if not all(_KEY_PART.fullmatch(p) for p in parts):
        raise ConfigError(f"bad override key {key!r}")
    value = value.strip()
    if not value:
        raise ConfigError(f"override {raw!r} has an empty value")
    parsed: object
    try:
        doc = tomllib.loads("v = " + value)
        if set(doc) != {"v"}:
            raise ConfigError(f"bad override value {value!r}")
        parsed = doc["v"]
    except tomllib.TOMLDecodeError as exc:
        if not _BARE_WORD.fullmatch(value):
            raise ConfigError(f"bad override value {value!r}: {exc}") from exc
        parsed = value
    _no_times(parsed)
    return parts, parsed  # pyright: ignore[reportReturnType]


def _apply(raw: dict[str, object], path: tuple[str, ...], value: object) -> None:
    node = raw
    for i, part in enumerate(path[:-1]):
        child: object = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ConfigError(f"cannot set {'.'.join(path)}: {'.'.join(path[: i + 1])} is not a table")
        node = cast("dict[str, object]", child)
    node[path[-1]] = value


def load_config(path: Path | None = None, overrides: Sequence[str] = ()) -> RunConfig:
    """Load a TOML file (or start empty), apply `a.b=v` overrides in order, validate once."""
    raw: dict[str, object] = {}
    if path is not None:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ConfigError(f"config not found: {path}") from exc
        except OSError as exc:
            raise ConfigError(f"cannot read config {path}: {exc}") from exc
        try:
            raw = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"bad TOML in {path}: {exc}") from exc
    for o in overrides:
        keys, value = parse_override(o)
        _apply(raw, keys, value)
    try:
        return RunConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid run config: {exc}") from exc


_PARAMS_ADAPTER: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])


def resolve_config(config: RunConfig) -> RunConfig:
    """Canonical form: params filled from the strategy's defaults, symbols sorted."""
    cls = get_strategy(config.run.strategy)
    dumped = build_params(cls, config.params).model_dump(mode="json")
    _check_json(dumped)
    try:
        params = _PARAMS_ADAPTER.validate_python(dumped)
    except ValidationError as exc:
        raise ConfigError(f"resolved params are not JSON values: {exc}") from exc
    universe = config.universe.model_copy(update={"symbols": tuple(sorted(config.universe.symbols))})
    return config.model_copy(update={"params": params, "universe": universe})


_ESCAPES = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r"}


def _str(s: str) -> str:
    out: list[str] = []
    for ch in s:
        o = ord(ch)
        if ch in _ESCAPES:
            out.append(_ESCAPES[ch])
        elif 0x20 <= o < 0x7F:
            out.append(ch)
        elif o <= 0xFFFF:
            out.append(f"\\u{o:04X}")
        else:
            out.append(f"\\U{o:08X}")
    return '"' + "".join(out) + '"'


def _key(k: str) -> str:
    return k if _BARE_KEY.fullmatch(k) else _str(k)


def _fmt(v: object, sort_keys: bool = False) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return _str(v)
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, list | tuple):
        return "[" + ", ".join(_fmt(x, sort_keys) for x in cast("Sequence[object]", v)) + "]"
    if isinstance(v, dict):
        d = cast("dict[str, object]", v)
        items = sorted(d.items()) if sort_keys else list(d.items())
        if not items:
            return "{}"
        return "{ " + ", ".join(f"{_key(k)} = {_fmt(x, sort_keys)}" for k, x in items) + " }"
    raise ConfigError(f"cannot write {type(v).__name__} to TOML")


def config_to_toml(config: RunConfig) -> str:
    """Deterministic TOML for `config`; every key written, round-trips through RunConfig."""
    data = config.model_dump(mode="python")
    sections: list[str] = []
    for name in ("run", "universe", "params", "costs", "engine"):
        body: dict[str, object] = data[name]
        sort = name == "params"
        lines = [f"[{name}]"] + [f"{_key(k)} = {_fmt(v, sort)}" for k, v in body.items()]
        sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"
