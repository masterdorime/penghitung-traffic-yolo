from datetime import UTC, date, datetime, timedelta, timezone, tzinfo
from itertools import pairwise
from typing import Final
from zoneinfo import ZoneInfo

import pytest

from traffic_counter.models import PeriodName
from traffic_counter.timeutils import (
    JAKARTA,
    PERIOD_MINUTE_BOUNDS,
    REPORT_MINUTES_PER_DAY,
    WallClock,
    period_bounds,
    period_for,
    report_date,
    report_minutes_through_current,
    to_jakarta,
)

MAKASSAR: Final[ZoneInfo] = ZoneInfo("Asia/Makassar")
SEVEN_HOURS: Final[timezone] = timezone(timedelta(hours=7))
DAY: Final[date] = date(2026, 9, 25)

PERIOD_CASES: Final[tuple[tuple[tuple[int, int, int], PeriodName | None], ...]] = (
    ((0, 0, 0), None),
    ((5, 0, 0), None),
    ((5, 59, 59), None),
    ((6, 0, 0), PeriodName.PAGI),
    ((6, 0, 1), PeriodName.PAGI),
    ((9, 59, 58), PeriodName.PAGI),
    ((9, 59, 59), PeriodName.PAGI),
    ((10, 0, 0), PeriodName.SIANG),
    ((12, 30, 0), PeriodName.SIANG),
    ((14, 59, 59), PeriodName.SIANG),
    ((15, 0, 0), PeriodName.SORE),
    ((18, 59, 58), PeriodName.SORE),
    ((18, 59, 59), PeriodName.SORE),
    ((19, 0, 0), None),
    ((19, 0, 1), None),
    ((23, 59, 59), None),
)

REPORT_MINUTE_CASES: Final[tuple[tuple[tuple[int, int, int], int], ...]] = (
    ((0, 0, 0), 0),
    ((5, 0, 0), 0),
    ((5, 59, 59), 0),
    ((6, 0, 0), 1),
    ((6, 0, 1), 1),
    ((6, 0, 59), 1),
    ((6, 1, 0), 2),
    ((6, 30, 45), 31),
    ((7, 0, 0), 61),
    ((9, 59, 59), 240),
    ((10, 0, 0), 241),
    ((14, 59, 0), 540),
    ((15, 0, 0), 541),
    ((18, 58, 59), 779),
    ((18, 59, 0), 780),
    ((18, 59, 1), 780),
    ((18, 59, 59), 780),
    ((19, 0, 0), 780),
    ((19, 0, 1), 780),
    ((23, 59, 59), 780),
)


class NoOffsetTimezone(tzinfo):
    def utcoffset(self, moment: datetime | None) -> timedelta | None:
        return None

    def dst(self, moment: datetime | None) -> timedelta | None:
        return None

    def tzname(self, moment: datetime | None) -> str | None:
        return None


def at(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, 9, 25, hour, minute, second, tzinfo=JAKARTA)


class SampledClock:
    def __init__(self, moment: datetime, monotonic_value: float) -> None:
        self.moment = moment
        self.monotonic_value = monotonic_value
        self.wall_calls = 0

    def wall(self) -> datetime:
        self.wall_calls += 1
        return self.moment

    def monotonic(self) -> float:
        return self.monotonic_value


def test_jakarta_is_a_zoneinfo_not_a_fixed_offset() -> None:
    assert str(JAKARTA) == "Asia/Jakarta"
    assert isinstance(JAKARTA, ZoneInfo)
    assert JAKARTA.utcoffset(at(6, 0)) == timedelta(hours=7)


def test_report_minutes_per_day_constant() -> None:
    assert REPORT_MINUTES_PER_DAY == 780


def test_period_minute_bounds_table() -> None:
    assert dict(PERIOD_MINUTE_BOUNDS) == {
        PeriodName.PAGI: (6 * 60, 10 * 60),
        PeriodName.SIANG: (10 * 60, 15 * 60),
        PeriodName.SORE: (15 * 60, 19 * 60),
    }


@pytest.mark.parametrize(("clock_parts", "expected"), PERIOD_CASES)
def test_period_boundaries_are_half_open(
    clock_parts: tuple[int, int, int], expected: PeriodName | None
) -> None:
    assert period_for(at(*clock_parts)) is expected


def test_period_for_returns_the_enum_member() -> None:
    assert period_for(at(7, 0)) is PeriodName.PAGI
    assert period_for(at(11, 0)) is PeriodName.SIANG
    assert period_for(at(16, 0)) is PeriodName.SORE


def test_period_for_converts_other_aware_timezones() -> None:
    assert period_for(datetime(2026, 9, 25, 3, 0, tzinfo=UTC)) is PeriodName.SIANG
    assert period_for(datetime(2026, 9, 25, 11, 0, tzinfo=MAKASSAR)) is PeriodName.SIANG
    assert period_for(datetime(2026, 9, 25, 17, 0, tzinfo=SEVEN_HOURS)) is PeriodName.SORE
    assert period_for(datetime(2026, 9, 25, 8, 0, tzinfo=MAKASSAR)) is PeriodName.PAGI
    assert period_for(datetime(2026, 9, 25, 12, 0, tzinfo=UTC)) is None


def test_period_for_uses_the_jakarta_calendar_day() -> None:
    assert period_for(datetime(2026, 9, 24, 23, 0, tzinfo=UTC)) is PeriodName.PAGI
    assert period_for(datetime(2026, 9, 25, 23, 0, tzinfo=UTC)) is PeriodName.PAGI
    assert period_for(datetime(2026, 9, 25, 17, 0, tzinfo=UTC)) is None
    assert period_for(datetime(2026, 9, 25, 11, 59, tzinfo=UTC)) is PeriodName.SORE
    assert period_for(datetime(2026, 9, 25, 12, 0, tzinfo=UTC)) is None


def test_period_for_rejects_naive_datetimes() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        period_for(datetime(2026, 9, 25, 6, 0))


def test_period_for_rejects_tzinfo_without_utc_offset() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        period_for(datetime(2026, 9, 25, 6, 0, tzinfo=NoOffsetTimezone()))


def test_to_jakarta_normalizes_and_preserves_instant() -> None:
    moment = datetime(2026, 9, 25, 12, 0, 30, 500000, tzinfo=MAKASSAR)

    converted = to_jakarta(moment)

    assert converted.tzinfo is JAKARTA
    assert converted == moment
    assert converted.hour == 11


def test_to_jakarta_rejects_naive_datetimes() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        to_jakarta(datetime(2026, 9, 25, 6, 0))


def test_to_jakarta_leaves_jakarta_input_untouched() -> None:
    moment = at(6, 0)

    assert to_jakarta(moment) is moment


def test_period_bounds_are_half_open_day_intervals() -> None:
    assert period_bounds(PeriodName.PAGI, DAY) == (at(6, 0), at(10, 0))
    assert period_bounds(PeriodName.SIANG, DAY) == (at(10, 0), at(15, 0))
    assert period_bounds(PeriodName.SORE, DAY) == (at(15, 0), at(19, 0))


def test_period_bounds_respects_the_requested_day() -> None:
    later = date(2026, 9, 26)

    assert period_bounds(PeriodName.PAGI, later) == (
        datetime(2026, 9, 26, 6, 0, tzinfo=JAKARTA),
        datetime(2026, 9, 26, 10, 0, tzinfo=JAKARTA),
    )


@pytest.mark.parametrize(("clock_parts", "expected_count"), REPORT_MINUTE_CASES)
def test_report_minute_counts(clock_parts: tuple[int, int, int], expected_count: int) -> None:
    assert len(report_minutes_through_current(at(*clock_parts))) == expected_count


def test_report_minutes_start_at_six_and_step_by_one_minute() -> None:
    minutes = report_minutes_through_current(at(6, 3, 27))

    assert minutes[0] == at(6, 0)
    assert minutes[-1] == at(6, 3)
    assert len(minutes) == 4
    assert all(moment.tzinfo is JAKARTA for moment in minutes)
    assert all(moment.second == 0 and moment.microsecond == 0 for moment in minutes)
    assert all(later - earlier == timedelta(minutes=1) for earlier, later in pairwise(minutes))


def test_report_minutes_are_empty_before_six() -> None:
    assert report_minutes_through_current(at(5, 59, 59)) == ()
    assert report_minutes_through_current(at(0, 0, 0)) == ()


def test_report_minutes_cap_at_eighteen_fifty_nine() -> None:
    minutes = report_minutes_through_current(at(19, 0, 0))

    assert len(minutes) == REPORT_MINUTES_PER_DAY
    assert minutes[-1] == at(18, 59)
    assert report_minutes_through_current(at(23, 59, 59)) == minutes


def test_report_minutes_use_the_jakarta_day_not_the_input_day() -> None:
    assert report_minutes_through_current(datetime(2026, 9, 25, 16, 59, tzinfo=UTC)) == (
        report_minutes_through_current(at(23, 59))
    )
    assert report_minutes_through_current(datetime(2026, 9, 25, 17, 0, tzinfo=UTC)) == ()


def test_report_minutes_reject_naive_datetimes() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        report_minutes_through_current(datetime(2026, 9, 25, 6, 0))


def test_report_date_uses_the_jakarta_calendar_date() -> None:
    assert report_date(datetime(2026, 9, 25, 17, 0, tzinfo=UTC)) == date(2026, 9, 26)
    assert report_date(at(6, 0)) == DAY
    assert report_date(at(23, 59, 59)) == DAY
    assert report_date(datetime(2026, 9, 25, 16, 59, tzinfo=UTC)) == DAY


def test_report_date_without_arguments_returns_the_current_jakarta_day() -> None:
    assert report_date() == datetime.now(JAKARTA).date()


def test_report_date_rejects_naive_datetimes() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        report_date(datetime(2026, 9, 25, 6, 0))


def test_wall_clock_default_constructor_returns_aware_jakarta_now() -> None:
    moment = WallClock().now()

    assert moment.tzinfo is JAKARTA
    assert moment.utcoffset() == timedelta(hours=7)
    assert (datetime.now(JAKARTA) - moment).total_seconds() < 5.0


def test_wall_clock_advances_with_monotonic_time() -> None:
    sampler = SampledClock(at(6, 0, 0), 1_000.0)
    clock = WallClock(wall=sampler.wall, monotonic=sampler.monotonic)

    assert clock.now() == at(6, 0, 0)

    sampler.monotonic_value = 1_000.5
    assert clock.now() == at(6, 0, 0) + timedelta(milliseconds=500)

    sampler.monotonic_value = 1_000.75
    assert clock.now() == at(6, 0, 0) + timedelta(milliseconds=750)


def test_wall_clock_refreshes_offset_at_most_once_per_second() -> None:
    sampler = SampledClock(at(6, 0, 0), 1_000.0)
    clock = WallClock(wall=sampler.wall, monotonic=sampler.monotonic, refresh_seconds=1.0)

    assert clock.now() == at(6, 0, 0)
    assert sampler.wall_calls == 1

    sampler.moment = at(7, 0, 0)
    sampler.monotonic_value = 1_000.2
    assert clock.now() == at(6, 0, 0) + timedelta(milliseconds=200)
    assert sampler.wall_calls == 1

    sampler.monotonic_value = 1_000.9
    assert clock.now() == at(6, 0, 0) + timedelta(milliseconds=900)
    assert sampler.wall_calls == 1

    sampler.monotonic_value = 1_001.0
    assert clock.now() == at(7, 0, 0)
    assert sampler.wall_calls == 2


def test_wall_clock_honours_a_custom_refresh_interval() -> None:
    sampler = SampledClock(at(6, 0, 0), 1_000.0)
    clock = WallClock(wall=sampler.wall, monotonic=sampler.monotonic, refresh_seconds=0.25)

    assert clock.now() == at(6, 0, 0)

    sampler.moment = at(7, 0, 0)
    sampler.monotonic_value = 1_000.2
    assert clock.now() == at(6, 0, 0) + timedelta(milliseconds=200)
    assert sampler.wall_calls == 1

    sampler.monotonic_value = 1_000.3
    assert clock.now() == at(7, 0, 0)
    assert sampler.wall_calls == 2


def test_wall_clock_rejects_nonpositive_refresh_interval() -> None:
    with pytest.raises(ValueError, match="refresh"):
        WallClock(refresh_seconds=0.0)
    with pytest.raises(ValueError, match="refresh"):
        WallClock(refresh_seconds=-1.0)


def test_wall_clock_rejects_a_naive_wall_sampler() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        WallClock(wall=lambda: datetime(2026, 9, 25, 6, 0), monotonic=lambda: 1_000.0)
