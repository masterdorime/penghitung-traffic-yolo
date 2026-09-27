from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from datetime import time as clock_time
from typing import Final

from traffic_counter.models import CountEvent, HealthBucket, PeriodName, PeriodSummary
from traffic_counter.timeutils import (
    JAKARTA,
    period_bounds,
    period_for,
    to_jakarta,
)

type HealthBucketSource = Mapping[datetime, HealthBucket] | Iterable[HealthBucket]

SECONDS_PER_MINUTE: Final[int] = 60
DEFAULT_ROLLING_WINDOW: Final[timedelta] = timedelta(minutes=5)
DEFAULT_ROLLING_MIN_HEALTHY_SECONDS: Final[int] = 60
_RESOLUTION: Final[timedelta] = timedelta(microseconds=1)
_NEGATIVE_MESSAGE: Final[str] = "Minimum healthy seconds must not be negative."


def _bucket_tuple(source: HealthBucketSource) -> tuple[HealthBucket, ...]:
    if isinstance(source, Mapping):
        return tuple(source.values())
    return tuple(source)


def average_rate(total: int, observed_seconds: int) -> float | None:
    if observed_seconds <= 0:
        return None
    return total / observed_seconds * SECONDS_PER_MINUTE


def _second_of_day(moment: datetime) -> int:
    return moment.hour * 3600 + moment.minute * 60 + moment.second


def _bucket_start(moment: datetime, bucket_seconds: int) -> datetime:
    midnight = datetime.combine(moment.date(), clock_time.min, tzinfo=JAKARTA)
    offset = _second_of_day(moment) // bucket_seconds * bucket_seconds
    return midnight + timedelta(seconds=offset)


class HealthAccumulator:
    __slots__ = ("_seconds",)

    def __init__(self) -> None:
        self._seconds: set[datetime] = set()

    def __len__(self) -> int:
        return len(self._seconds)

    def mark_healthy(self, observed_at: datetime) -> None:
        self._seconds.add(to_jakarta(observed_at).replace(microsecond=0))

    def drain(self, bucket_seconds: int) -> tuple[HealthBucket, ...]:
        if bucket_seconds <= 0:
            raise ValueError("Health bucket seconds must be a positive whole number of seconds.")
        grouped: dict[datetime, list[datetime]] = {}
        for second in self._seconds:
            grouped.setdefault(_bucket_start(second, bucket_seconds), []).append(second)
        self._seconds.clear()
        buckets: list[HealthBucket] = []
        for bucket_start in sorted(grouped):
            period = period_for(bucket_start)
            if period is None:
                continue
            buckets.append(HealthBucket(bucket_start, period, tuple(sorted(grouped[bucket_start]))))
        return tuple(buckets)


def _rolling_window(end: datetime, period: PeriodName | None) -> tuple[datetime, datetime]:
    if period is None:
        return end - DEFAULT_ROLLING_WINDOW, end
    period_start, period_end = period_bounds(period, end.date())
    if end < period_end:
        return max(end - DEFAULT_ROLLING_WINDOW, period_start), end
    return period_end - DEFAULT_ROLLING_WINDOW, period_end - _RESOLUTION


def rolling_rate(
    events: Iterable[CountEvent],
    health_buckets: HealthBucketSource,
    end: datetime,
    minimum_healthy_seconds: int = DEFAULT_ROLLING_MIN_HEALTHY_SECONDS,
    period: PeriodName | None = None,
) -> float | None:
    if minimum_healthy_seconds < 0:
        raise ValueError(_NEGATIVE_MESSAGE)
    window_end = to_jakarta(end)
    window_start, window_end = _rolling_window(window_end, period)
    counted = 0
    for event in events:
        crossed_at = to_jakarta(event.crossed_at)
        if window_start < crossed_at <= window_end:
            counted += 1
    healthy: set[datetime] = set()
    for bucket in _bucket_tuple(health_buckets):
        for second in bucket.seconds:
            moment = to_jakarta(second)
            if window_start < moment <= window_end:
                healthy.add(moment)
    if len(healthy) < minimum_healthy_seconds:
        return None
    return average_rate(counted, len(healthy))


def build_period_summaries(
    events: Iterable[CountEvent],
    health_buckets: HealthBucketSource,
    now: datetime,
    minimum_healthy_seconds: int = DEFAULT_ROLLING_MIN_HEALTHY_SECONDS,
) -> tuple[PeriodSummary, ...]:
    moment = to_jakarta(now)
    buckets = _bucket_tuple(health_buckets)
    events_by_period: dict[PeriodName, list[CountEvent]] = {period: [] for period in PeriodName}
    for event in events:
        period = period_for(event.crossed_at)
        if period is not None:
            events_by_period[period].append(event)
    seconds_by_period: dict[PeriodName, set[datetime]] = {period: set() for period in PeriodName}
    for bucket in buckets:
        for second in bucket.seconds:
            healthy_second = to_jakarta(second)
            period = period_for(healthy_second)
            if period is not None:
                seconds_by_period[period].add(healthy_second)
    summaries: list[PeriodSummary] = []
    for period in PeriodName:
        start, _ = period_bounds(period, moment.date())
        if moment < start:
            summaries.append(
                PeriodSummary(
                    period=period,
                    started=False,
                    total=0,
                    observed_seconds=0,
                    average_rate=None,
                    rolling_rate=None,
                )
            )
            continue
        period_events = events_by_period[period]
        observed_seconds = len(seconds_by_period[period])
        summaries.append(
            PeriodSummary(
                period=period,
                started=True,
                total=len(period_events),
                observed_seconds=observed_seconds,
                average_rate=average_rate(len(period_events), observed_seconds),
                rolling_rate=rolling_rate(
                    period_events, buckets, moment, minimum_healthy_seconds, period
                ),
            )
        )
    return tuple(summaries)
