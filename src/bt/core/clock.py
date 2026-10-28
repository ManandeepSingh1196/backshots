"""UTC guard used wherever a timestamp enters the system."""

from datetime import datetime, timedelta

from bt.core.errors import TimezoneError


def ensure_utc(ts: datetime, name: str = "ts") -> datetime:
    """Return `ts` unchanged if it is timezone-aware with a zero UTC offset.

    Raises TimezoneError (message contains `name`) for naive datetimes and for
    aware datetimes with a non-zero offset.
    """
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise TimezoneError(f"{name} is naive (no tzinfo); expected timezone-aware UTC: {ts!r}")
    if ts.utcoffset() != timedelta(0):
        raise TimezoneError(
            f"{name} has non-zero UTC offset {ts.utcoffset()}; expected UTC: {ts!r}"
        )
    return ts
