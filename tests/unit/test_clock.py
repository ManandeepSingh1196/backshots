from datetime import UTC, datetime, timedelta, timezone
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
    assert ensure_utc(b) is b


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


def test_zero_offset_non_utc_zone_is_accepted() -> None:
    # offset == 0 at that instant (e.g. Europe/London in winter): spec is utcoffset()==0
    ts = datetime(2024, 1, 1, tzinfo=ZoneInfo("Europe/London"))
    assert ensure_utc(ts) is ts
