"""UTC guard used wherever a timestamp enters the system."""

from datetime import UTC, datetime, timedelta

from bt.core.errors import TimezoneError


def ensure_utc(ts: datetime, name: str = "ts") -> datetime:
    """Validate that `ts` is aware with a zero UTC offset and return it normalised to UTC.

    The result always has tzinfo `datetime.UTC`, so a zero-offset foreign zone
    (e.g. Europe/London in winter) cannot leak wall-clock arithmetic downstream.
    Raises TimezoneError (message contains `name`) for non-datetimes, naive
    datetimes, and aware datetimes whose offset at that instant is non-zero.
    """
    if not isinstance(ts, datetime):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TimezoneError(f"{name} must be a datetime, got {type(ts).__name__}")
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise TimezoneError(f"{name} is naive (no tzinfo); expected timezone-aware UTC: {ts!r}")
    if ts.utcoffset() != timedelta(0):
        raise TimezoneError(
            f"{name} has non-zero UTC offset {ts.utcoffset()}; expected UTC: {ts!r}"
        )
    return ts.astimezone(UTC)
