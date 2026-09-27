from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from datetime import time as clock_time
from types import MappingProxyType
from typing import Final
from zoneinfo import ZoneInfo

from traffic_counter.models import PeriodName

JAKARTA: Final[ZoneInfo] = ZoneInfo("Asia/Jakarta")

_PAGI_START_MINUTE: Final[int] = 6 * 60
_PAGI_END_MINUTE: Final[int] = 10 * 60
_SIANG_START_MINUTE: Final[int] = 10 * 60
_SIANG_END_MINUTE: Final[int] = 15 * 60
_SORE_START_MINUTE: Final[int] = 15 * 60
_SORE_END_MINUTE: Final[int] = 19 * 60

PERIOD_MINUTE_BOUNDS: Final[Mapping[PeriodName, tuple[int, int]]] = MappingProxyType(
    {
        PeriodName.PAGI: (_PAGI_START_MINUTE, _PAGI_END_MINUTE),
        PeriodName.SIANG: (_SIANG_START_MINUTE, _SIANG_END_MINUTE),
        PeriodName.SORE: (_SORE_START_MINUTE, _SORE_END_MINUTE),
    }
)

REPORT_START_MINUTE: Final[int] = _PAGI_START_MINUTE
REPORT_END_MINUTE: Final[int] = _SORE_END_MINUTE
REPORT_MINUTES_PER_DAY: Final[int] = REPORT_END_MINUTE - REPORT_START_MINUTE

WallSampler = Callable[[], datetime]
MonotonicSampler = Callable[[], float]

_NAIVE_MESSAGE: Final[str] = (
    "Timestamps must be timezone-aware datetimes with a defined UTC offset."
)
_DEFAULT_REFRESH_SECONDS: Final[float] = 1.0


def _jakarta_wall_now() -> datetime:
    return datetime.now(JAKARTA)


def to_jakarta(moment: datetime) -> datetime:
    if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
        raise ValueError(_NAIVE_MESSAGE)
    if moment.tzinfo is JAKARTA:
        return moment
    return moment.astimezone(JAKARTA)


def _minute_of_day(moment: datetime) -> int:
    return moment.hour * 60 + moment.minute


def _midnight_of(moment: datetime) -> datetime:
    return datetime.combine(moment.date(), clock_time.min, tzinfo=JAKARTA)


def _monotonic_as_jakarta(monotonic_value: float) -> datetime:
    return datetime.fromtimestamp(monotonic_value, tz=UTC).astimezone(JAKARTA)


class WallClock:
    __slots__ = ("_anchor_wall", "_mono", "_offset", "_refresh_seconds", "_sampled_mono", "_wall")

    def __init__(
        self,
        wall: WallSampler | None = None,
        monotonic: MonotonicSampler | None = None,
        refresh_seconds: float = _DEFAULT_REFRESH_SECONDS,
    ) -> None:
        if refresh_seconds <= 0:
            raise ValueError("Wall clock refresh interval must be a positive number of seconds.")
        self._wall: WallSampler = _jakarta_wall_now if wall is None else wall
        self._mono: MonotonicSampler = time.monotonic if monotonic is None else monotonic
        self._refresh_seconds = refresh_seconds
        self._offset, self._anchor_wall, self._sampled_mono = self._sample()

    def _sample(self) -> tuple[timedelta, datetime, float]:
        anchor = to_jakarta(self._wall())
        monotonic = self._mono()
        return anchor - _monotonic_as_jakarta(monotonic), anchor, monotonic

    def now(self) -> datetime:
        monotonic = self._mono()
        if monotonic - self._sampled_mono >= self._refresh_seconds:
            self._offset, self._anchor_wall, self._sampled_mono = self._sample()
            return self._anchor_wall
        return self._offset + _monotonic_as_jakarta(monotonic)


def report_date(now: datetime | None = None) -> date:
    if now is None:
        return datetime.now(JAKARTA).date()
    return to_jakarta(now).date()


def period_for(moment: datetime) -> PeriodName | None:
    local = to_jakarta(moment)
    minute = _minute_of_day(local)
    for period, (start_minute, end_minute) in PERIOD_MINUTE_BOUNDS.items():
        if start_minute <= minute < end_minute:
            return period
    return None


def period_bounds(period: PeriodName, day: date) -> tuple[datetime, datetime]:
    start_minute, end_minute = PERIOD_MINUTE_BOUNDS[period]
    midnight = datetime.combine(day, clock_time.min, tzinfo=JAKARTA)
    return (
        midnight + timedelta(minutes=start_minute),
        midnight + timedelta(minutes=end_minute),
    )


def report_minutes_through_current(now: datetime) -> tuple[datetime, ...]:
    local = to_jakarta(now)
    midnight = _midnight_of(local)
    first = midnight + timedelta(minutes=REPORT_START_MINUTE)
    current_minute = local.replace(second=0, microsecond=0)
    if current_minute < first:
        return ()
    last = min(current_minute, midnight + timedelta(minutes=REPORT_END_MINUTE - 1))
    elapsed = int((last - first).total_seconds() // 60)
    return tuple(first + timedelta(minutes=offset) for offset in range(elapsed + 1))
