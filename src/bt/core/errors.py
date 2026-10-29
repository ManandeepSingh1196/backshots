"""Domain exceptions. Every error raised on purpose by backshots derives from BtError."""


class BtError(Exception):
    """Base of every domain error."""


class TimezoneError(BtError, ValueError):
    """A datetime was naive or not in UTC."""


class InvariantError(BtError, ValueError):
    """A core type was constructed in an invalid state."""


class DataError(BtError):
    """Base of data-layer failures."""


class DataValidationError(DataError):
    """A dataset failed validation; the message lists the errors."""


class ChecksumMismatchError(DataError):
    """A download does not match Binance's published SHA-256."""


class ArchiveError(DataError):
    """Malformed archive, inconsistent listing, or HTTP failure after retries."""


class LookAheadError(BtError):
    """A read reached data at or after a step not yet delivered."""


class StrategyError(BtError):
    """Registry misuse, bad params, or a strategy that raised or returned an invalid intent."""


class ExecutionError(BtError):
    """Bad arguments to fees, slippage, rules or broker, or broker misuse."""


class PortfolioError(BtError):
    """A fill or mark that would break portfolio invariants; state is left unchanged."""


class EngineError(BtError):
    """Inconsistent wiring: instrument sets differ, feed reused, or unsupported kind."""


class ConfigError(BtError):
    """Bad run config: TOML syntax, schema, override syntax, or unsupported option."""


class RunError(BtError):
    """A run directory cannot be written, read or found, or has the wrong schema."""
