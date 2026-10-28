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
