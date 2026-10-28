from datetime import UTC, date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from bt.core.clock import ensure_utc
from bt.core.errors import TimezoneError

NAIVE = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)


def test_returns_same_object_for_utc() -> None:
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    assert ensure_utc(ts) is ts


def test_accepts_timezone_utc_and_zoneinfo() -> None:
    a = datetime(2024, 1, 1, tzinfo=UTC)
    b = datetime(2024, 1, 1, tzinfo=ZoneInfo("UTC"))
    assert ensure_utc(a) is a
    assert ensure_utc(b) == b
    assert ensure_utc(b).tzinfo is UTC


def test_naive_raises_with_name() -> None:
    with pytest.raises(TimezoneError, match="naive") as ei:
        ensure_utc(NAIVE, "my_field")
    assert "my_field" in str(ei.value)


def test_default_name_is_ts() -> None:
    with pytest.raises(TimezoneError, match="ts"):
        ensure_utc(NAIVE)


def test_non_utc_offset_raises() -> None:
    ist = timezone(timedelta(hours=5, minutes=30))
    with pytest.raises(TimezoneError, match="offset") as ei:
        ensure_utc(datetime(2024, 1, 1, 5, 30, tzinfo=ist), "when")
    assert "when" in str(ei.value)


LONDON = ZoneInfo("Europe/London")


def test_london_winter_accepted_and_normalised_to_utc() -> None:
    ts = datetime(2024, 1, 1, tzinfo=LONDON)
    out = ensure_utc(ts)
    assert out == ts
    assert out.tzinfo is UTC


def test_zoneinfo_utc_normalised_to_utc() -> None:
    out = ensure_utc(datetime(2024, 1, 1, tzinfo=ZoneInfo("UTC")))
    assert out.tzinfo is UTC


def test_london_summer_rejected() -> None:
    with pytest.raises(TimezoneError, match="offset"):
        ensure_utc(datetime(2024, 7, 1, tzinfo=LONDON))


def test_london_dst_boundary() -> None:
    # clocks go forward at 01:00 UTC on 2024-03-31
    before = ensure_utc(datetime(2024, 3, 31, 0, 30, tzinfo=LONDON))
    assert before.tzinfo is UTC
    with pytest.raises(TimezoneError, match="offset"):
        ensure_utc(datetime(2024, 3, 31, 2, 30, tzinfo=LONDON))


def test_non_datetime_rejected() -> None:
    with pytest.raises(TimezoneError):
        ensure_utc(date(2024, 1, 1))  # type: ignore[arg-type]
