from datetime import UTC, datetime, timedelta
from typing import Final

import pytest

from traffic_counter.density import (
    DEFAULT_ROLLING_MIN_HEALTHY_SECONDS,
    DEFAULT_ROLLING_WINDOW,
    HealthAccumulator,
    average_rate,
    build_period_summaries,
    rolling_rate,
)
from traffic_counter.models import CountEvent, HealthBucket, PeriodName
from traffic_counter.timeutils import JAKARTA

AVERAGE_RATE_CASES: Final[tuple[tuple[int, int, float | None], ...]] = (
    (60, 120, 30.0),
    (120, 600, 12.0),
    (1, 1, 60.0),
    (0, 60, 0.0),
    (0, 1, 0.0),
    (60, 0, None),
    (0, 0, None),
    (60, -1, None),
    (0, -30, None),
)


def at(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, 9, 25, hour, minute, second, tzinfo=JAKARTA)


def span(start: datetime, count: int) -> tuple[datetime, ...]:
    return tuple(start + timedelta(seconds=offset) for offset in range(count))


def make_event(
    moment: datetime,
    *,
    track_id: int = 1,
    tracking_session_id: int = 1,
    period: PeriodName = PeriodName.PAGI,
    direction: str = "up",
    class_name: str = "car",
) -> CountEvent:
    return CountEvent(
        tracking_session_id=tracking_session_id,
        track_id=track_id,
        crossed_at=moment,
        direction=direction,
        period=period,
        class_name=class_name,
    )


def make_bucket(
    bucket_start: datetime,
    period: PeriodName,
    seconds: tuple[datetime, ...],
) -> HealthBucket:
    return HealthBucket(bucket_start=bucket_start, period=period, seconds=seconds)


def marked(start: datetime, count: int, bucket_seconds: int = 10) -> HealthAccumulator:
    accumulator = HealthAccumulator()
    for second in span(start, count):
        accumulator.mark_healthy(second)
    return accumulator


def test_rolling_window_constant() -> None:
    assert DEFAULT_ROLLING_WINDOW.total_seconds() == 300.0
    assert DEFAULT_ROLLING_MIN_HEALTHY_SECONDS == 60


@pytest.mark.parametrize(("total", "observed", "expected"), AVERAGE_RATE_CASES)
def test_average_rate(total: int, observed: int, expected: float | None) -> None:
    assert average_rate(total, observed) == expected


def test_average_rate_returns_none_for_nonpositive_seconds() -> None:
    assert average_rate(60, 0) is None
    assert average_rate(60, -5) is None


def test_disconnect_seconds_do_not_dilute_average() -> None:
    assert average_rate(60, 120) == 30.0


def test_zero_events_with_healthy_time_is_zero_not_null() -> None:
    assert average_rate(0, 60) == 0.0


def test_mark_healthy_floors_to_the_containing_whole_second() -> None:
    accumulator = HealthAccumulator()
    accumulator.mark_healthy(datetime(2026, 9, 25, 6, 0, 0, 750000, tzinfo=JAKARTA))

    buckets = accumulator.drain(10)

    assert len(buckets) == 1
    assert buckets[0].seconds == (at(6, 0, 0),)


def test_overlapping_marks_do_not_double_count() -> None:
    accumulator = HealthAccumulator()
    accumulator.mark_healthy(datetime(2026, 9, 25, 6, 0, 0, 100000, tzinfo=JAKARTA))
    accumulator.mark_healthy(datetime(2026, 9, 25, 6, 0, 0, 900000, tzinfo=JAKARTA))
    accumulator.mark_healthy(at(6, 0, 0))

    assert len(accumulator) == 1
    assert accumulator.drain(10)[0].seconds == (at(6, 0, 0),)


def test_mark_healthy_converts_other_aware_timezones() -> None:
    accumulator = HealthAccumulator()
    accumulator.mark_healthy(datetime(2026, 9, 25, 3, 0, 1, 400000, tzinfo=UTC))

    buckets = accumulator.drain(10)

    assert buckets[0].bucket_start == at(10, 0)
    assert buckets[0].seconds == (at(10, 0, 1),)


def test_mark_healthy_rejects_naive_datetimes() -> None:
    accumulator = HealthAccumulator()
    with pytest.raises(ValueError, match="timezone-aware"):
        accumulator.mark_healthy(datetime(2026, 9, 25, 6, 0, 0))


def test_drain_aligns_buckets_to_ten_second_boundaries() -> None:
    buckets = marked(at(6, 0, 0), 12).drain(10)

    assert [bucket.bucket_start for bucket in buckets] == [at(6, 0, 0), at(6, 0, 10)]
    assert buckets[0].seconds == span(at(6, 0, 0), 10)
    assert buckets[1].seconds == span(at(6, 0, 10), 2)
    assert all(bucket.period is PeriodName.PAGI for bucket in buckets)


def test_drain_aligns_buckets_to_a_non_dividing_size() -> None:
    buckets = marked(at(6, 0, 2), 8).drain(7)

    assert [bucket.bucket_start for bucket in buckets] == [at(6, 0, 2), at(6, 0, 9)]
    assert buckets[0].seconds == span(at(6, 0, 2), 7)
    assert buckets[1].seconds == (at(6, 0, 9),)


def test_drain_supports_a_whole_hour_bucket() -> None:
    buckets = marked(at(6, 0, 0), 90).drain(60)

    assert [bucket.bucket_start for bucket in buckets] == [at(6, 0), at(6, 1)]
    assert buckets[0].seconds == span(at(6, 0, 0), 60)
    assert buckets[1].seconds == span(at(6, 1, 0), 30)


def test_drain_labels_each_bucket_with_its_period() -> None:
    accumulator = HealthAccumulator()
    accumulator.mark_healthy(at(9, 59, 55))
    accumulator.mark_healthy(at(10, 0, 0))
    accumulator.mark_healthy(at(14, 59, 59))
    accumulator.mark_healthy(at(15, 0, 0))

    buckets = accumulator.drain(10)

    assert [(bucket.bucket_start, bucket.period) for bucket in buckets] == [
        (at(9, 59, 50), PeriodName.PAGI),
        (at(10, 0, 0), PeriodName.SIANG),
        (at(14, 59, 50), PeriodName.SIANG),
        (at(15, 0, 0), PeriodName.SORE),
    ]


def test_drain_discards_seconds_outside_the_reporting_schedule() -> None:
    accumulator = HealthAccumulator()
    accumulator.mark_healthy(at(5, 59, 0))
    accumulator.mark_healthy(at(19, 0, 0))
    accumulator.mark_healthy(at(23, 30, 0))

    assert accumulator.drain(10) == ()


def test_drain_clears_consumed_seconds() -> None:
    accumulator = marked(at(6, 0, 0), 5)

    first = accumulator.drain(10)
    second = accumulator.drain(10)

    assert len(first) == 1
    assert second == ()
    assert len(accumulator) == 0


def test_drain_rejects_nonpositive_bucket_sizes() -> None:
    accumulator = marked(at(6, 0, 0), 1)
    with pytest.raises(ValueError, match="positive"):
        accumulator.drain(0)
    with pytest.raises(ValueError, match="positive"):
        accumulator.drain(-10)


def test_drain_keeps_seconds_the_second_drain_can_still_group() -> None:
    accumulator = marked(at(6, 0, 0), 12)

    first = accumulator.drain(10)
    accumulator.mark_healthy(at(6, 0, 15))
    second = accumulator.drain(10)

    assert len(first[0].seconds) == 10
    assert second[0].seconds == (at(6, 0, 15),)


def test_rolling_rate_requires_minimum_healthy_time() -> None:
    end = at(6, 5)
    events = (make_event(end - timedelta(seconds=10)),)

    assert rolling_rate(events, {}, end, minimum_healthy_seconds=60) is None


def test_rolling_rate_counts_healthy_seconds_exactly() -> None:
    end = at(6, 3)
    healthy = span(at(6, 0, 1), 120)
    events = tuple(make_event(at(6, 1, offset), track_id=offset) for offset in range(6))
    buckets = (make_bucket(at(6, 0), PeriodName.PAGI, healthy),)

    assert rolling_rate(events, buckets, end, 120) == pytest.approx(3.0)


def test_rolling_rate_uses_a_half_open_window() -> None:
    end = at(6, 5)
    events = (
        make_event(at(5, 59, 59), track_id=1),
        make_event(at(6, 0, 0), track_id=2),
        make_event(at(6, 2, 30), track_id=3),
        make_event(at(6, 5, 0), track_id=4),
        make_event(at(6, 5, 1), track_id=5),
    )
    buckets = (make_bucket(at(5, 59, 50), PeriodName.PAGI, span(at(5, 59, 50), 371)),)

    assert rolling_rate(events, buckets, end, 1) == pytest.approx(120.0 / 300)


def test_rolling_rate_unions_duplicate_healthy_seconds() -> None:
    end = at(6, 2)
    duplicated = span(at(6, 0), 60)
    buckets = (
        make_bucket(at(6, 0), PeriodName.PAGI, duplicated),
        make_bucket(at(6, 0), PeriodName.PAGI, duplicated),
        make_bucket(at(6, 1), PeriodName.PAGI, span(at(6, 1), 60)),
    )

    assert rolling_rate((), buckets, end, 120) == pytest.approx(0.0)
    assert rolling_rate((), buckets, end, 121) is None
    assert rolling_rate((), buckets, end, 180) is None


def test_rolling_rate_unions_duplicate_seconds_inside_one_bucket() -> None:
    end = at(6, 1)
    buckets = (make_bucket(at(6, 0), PeriodName.PAGI, (at(6, 0, 0), at(6, 0, 0))),)

    assert rolling_rate((), buckets, end, 2) is None
    assert rolling_rate((), buckets, end, 1) == pytest.approx(0.0)


def test_rolling_rate_is_null_without_healthy_time() -> None:
    assert rolling_rate((), {}, at(6, 5), 0) is None


def test_rolling_rate_accepts_an_empty_mapping_and_a_tuple() -> None:
    bucket = make_bucket(at(6, 0), PeriodName.PAGI, span(at(6, 0), 60))

    assert rolling_rate((), {}, at(6, 1), 1) is None
    assert rolling_rate((), {}, at(6, 1), 0) is None
    assert rolling_rate((), (bucket,), at(6, 1), 60) == pytest.approx(0.0)
    assert rolling_rate((), {at(6, 0): bucket}, at(6, 1), 60) == pytest.approx(0.0)


def test_rolling_rate_converts_non_jakarta_timestamps() -> None:
    end = datetime(2026, 9, 24, 23, 3, tzinfo=UTC)
    buckets = (make_bucket(at(6, 0), PeriodName.PAGI, span(at(6, 0), 120)),)
    events = (make_event(at(6, 1, 30)),)

    assert end.astimezone(JAKARTA) == at(6, 3)
    assert rolling_rate(events, buckets, end, 120) == pytest.approx(0.5)


def test_rolling_rate_rejects_naive_timestamps() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        rolling_rate((), {}, datetime(2026, 9, 25, 6, 5))
    with pytest.raises(ValueError, match="timezone-aware"):
        rolling_rate((make_event(datetime(2026, 9, 25, 6, 1)),), {}, at(6, 5))


def test_rolling_rate_rejects_negative_minimum_healthy_seconds() -> None:
    with pytest.raises(ValueError, match="negative"):
        rolling_rate((), {}, at(6, 5), -1)


def test_rolling_rate_clips_the_window_to_a_supplied_period() -> None:
    end = at(10, 4)
    buckets = (
        make_bucket(at(9, 57), PeriodName.PAGI, span(at(9, 57), 180)),
        make_bucket(at(10, 0), PeriodName.SIANG, span(at(10, 0, 1), 240)),
    )
    events = (
        make_event(at(9, 58, 0), track_id=1),
        make_event(at(10, 2, 0), track_id=2),
    )

    assert rolling_rate(events, buckets, end, 1, period=PeriodName.PAGI) == pytest.approx(1 / 3)
    assert rolling_rate(events, buckets, end, 1) == pytest.approx(60 / 299)


def test_rolling_rate_for_an_unstarted_period_is_null() -> None:
    assert rolling_rate((), {}, at(6, 5), 1, period=PeriodName.SIANG) is None


def test_rolling_rate_at_ten_oh_four_clips_to_each_period() -> None:
    now = at(10, 4)
    buckets = (
        make_bucket(at(9, 57), PeriodName.PAGI, span(at(9, 57), 180)),
        make_bucket(at(10, 0), PeriodName.SIANG, span(at(10, 0, 1), 240)),
    )
    events = (
        make_event(at(9, 58, 0), track_id=1),
        make_event(at(10, 2, 0), track_id=2),
    )

    summaries = {s.period: s for s in build_period_summaries(events, buckets, now, 1)}

    assert summaries[PeriodName.PAGI].rolling_rate == pytest.approx(1 / 3)
    assert summaries[PeriodName.SIANG].rolling_rate == pytest.approx(0.25)
    assert summaries[PeriodName.SORE].rolling_rate is None


def test_rolling_rate_at_fifteen_oh_four_clips_to_each_period() -> None:
    now = at(15, 4)
    buckets = (
        make_bucket(at(14, 57), PeriodName.SIANG, span(at(14, 57), 180)),
        make_bucket(at(15, 0), PeriodName.SORE, span(at(15, 0, 1), 240)),
    )
    events = (
        make_event(at(14, 58, 0), track_id=1),
        make_event(at(15, 2, 0), track_id=2),
    )

    summaries = {s.period: s for s in build_period_summaries(events, buckets, now, 1)}

    assert summaries[PeriodName.SIANG].rolling_rate == pytest.approx(1 / 3)
    assert summaries[PeriodName.SORE].rolling_rate == pytest.approx(0.25)
    assert summaries[PeriodName.PAGI].total == 0


def test_completed_period_rolling_rate_uses_its_final_five_minutes() -> None:
    now = at(12, 0)
    buckets = (
        make_bucket(at(9, 57), PeriodName.PAGI, span(at(9, 57), 180)),
        make_bucket(at(11, 59), PeriodName.SIANG, span(at(11, 59), 60)),
    )
    events = (make_event(at(9, 58, 0), track_id=1),)

    summaries = {s.period: s for s in build_period_summaries(events, buckets, now, 1)}

    assert summaries[PeriodName.PAGI].rolling_rate == pytest.approx(1 / 3)
    assert summaries[PeriodName.SIANG].rolling_rate == pytest.approx(0.0)
    assert summaries[PeriodName.SIANG].observed_seconds == 60


def test_clipped_window_excludes_the_instant_a_period_ends() -> None:
    now = at(10, 4)
    boundary_second = at(10, 0)
    buckets = (
        make_bucket(at(9, 57), PeriodName.PAGI, span(at(9, 57), 180)),
        make_bucket(
            at(10, 0),
            PeriodName.SIANG,
            (boundary_second, at(10, 0, 1)),
        ),
    )
    events = (
        make_event(at(9, 58, 0), track_id=1),
        make_event(at(10, 0, 1), track_id=2),
    )

    summaries = {s.period: s for s in build_period_summaries(events, buckets, now, 1)}

    assert summaries[PeriodName.PAGI].rolling_rate == pytest.approx(1 / 3)
    assert summaries[PeriodName.PAGI].observed_seconds == 180
    assert summaries[PeriodName.SIANG].rolling_rate == pytest.approx(60.0)
    assert summaries[PeriodName.SIANG].observed_seconds == 2


def test_summaries_always_contain_all_three_periods() -> None:
    summaries = build_period_summaries((), {}, at(6, 5))

    assert tuple(summary.period for summary in summaries) == tuple(PeriodName)


def test_unstarted_periods_have_zero_totals_and_null_rates() -> None:
    summaries = build_period_summaries((), {}, at(6, 5))

    assert summaries[0].period is PeriodName.PAGI
    assert summaries[0].started is True
    assert summaries[0].total == 0
    assert summaries[0].observed_seconds == 0
    assert summaries[0].average_rate is None
    assert summaries[0].rolling_rate is None
    for summary in summaries[1:]:
        assert summary.started is False
        assert summary.total == 0
        assert summary.observed_seconds == 0
        assert summary.average_rate is None
        assert summary.rolling_rate is None


def test_all_periods_are_unstarted_before_the_schedule_opens() -> None:
    for summary in build_period_summaries((), {}, at(5, 30)):
        assert summary.started is False
        assert summary.total == 0
        assert summary.observed_seconds == 0
        assert summary.average_rate is None
        assert summary.rolling_rate is None


def test_started_flips_exactly_at_the_period_start() -> None:
    before = {summary.period: summary for summary in build_period_summaries((), {}, at(9, 59, 59))}
    at_start = {summary.period: summary for summary in build_period_summaries((), {}, at(10, 0))}

    assert before[PeriodName.PAGI].started is True
    assert before[PeriodName.SIANG].started is False
    assert at_start[PeriodName.SIANG].started is True
    assert at_start[PeriodName.SORE].started is False


def test_summaries_total_the_events_per_period() -> None:
    now = at(15, 0)
    events = (
        make_event(at(6, 1), track_id=1),
        make_event(at(9, 59, 59), track_id=2),
        make_event(at(10, 0), track_id=3),
        make_event(at(14, 30), track_id=4),
        make_event(at(15, 0), track_id=5),
        make_event(at(18, 59, 59), track_id=6),
    )

    summaries = {summary.period: summary for summary in build_period_summaries(events, {}, now)}

    assert summaries[PeriodName.PAGI].total == 2
    assert summaries[PeriodName.SIANG].total == 2
    assert summaries[PeriodName.SORE].total == 2
    assert all(summary.started for summary in summaries.values())
    assert all(summary.average_rate is None for summary in summaries.values())


def test_period_membership_follows_the_crossing_timestamp_not_the_stored_label() -> None:
    events = (make_event(at(12, 0), period=PeriodName.PAGI, track_id=1),)
    summaries = {
        summary.period: summary for summary in build_period_summaries(events, {}, at(12, 1))
    }

    assert summaries[PeriodName.SIANG].total == 1
    assert summaries[PeriodName.PAGI].total == 0


def test_events_outside_the_schedule_are_ignored() -> None:
    events = (make_event(at(19, 0, 0), period=PeriodName.SORE), make_event(at(5, 0)))

    summaries = build_period_summaries(events, {}, at(19, 1))

    assert all(summary.total == 0 for summary in summaries)


def test_observed_seconds_come_from_unioned_healthy_seconds() -> None:
    now = at(6, 10)
    duplicated = span(at(6, 0, 0), 60)
    buckets = (
        make_bucket(at(6, 0), PeriodName.PAGI, duplicated),
        make_bucket(at(6, 0), PeriodName.PAGI, duplicated),
        make_bucket(at(6, 1), PeriodName.PAGI, span(at(6, 1), 60)),
    )
    events = (make_event(at(6, 1, 0)), make_event(at(6, 2, 0)), make_event(at(6, 3, 0)))

    summary = build_period_summaries(events, buckets, now)[0]

    assert summary.total == 3
    assert summary.observed_seconds == 120
    assert summary.average_rate == pytest.approx(1.5)


def test_observed_seconds_ignore_seconds_outside_the_schedule() -> None:
    buckets = (
        make_bucket(at(5, 59, 50), PeriodName.PAGI, span(at(5, 59, 50), 10)),
        make_bucket(at(6, 0), PeriodName.PAGI, span(at(6, 0), 60)),
        make_bucket(at(19, 0), PeriodName.SORE, span(at(19, 0), 10)),
    )

    summary = build_period_summaries((), buckets, at(6, 2))[0]

    assert summary.observed_seconds == 60


def test_average_rate_is_null_without_healthy_time() -> None:
    events = (make_event(at(6, 0, 30)), make_event(at(6, 0, 45)))

    summary = build_period_summaries(events, {}, at(6, 5))[0]

    assert summary.total == 2
    assert summary.observed_seconds == 0
    assert summary.average_rate is None
    assert summary.rolling_rate is None


def test_rolling_rate_uses_the_current_time_and_minimum_health() -> None:
    now = at(6, 5)
    events = tuple(make_event(at(6, 4, 30), track_id=offset) for offset in range(3))
    buckets = (make_bucket(at(6, 0), PeriodName.PAGI, span(at(6, 4), 60)),)

    allowed = build_period_summaries(events, buckets, now, 60)[0]
    blocked = build_period_summaries(events, buckets, now, 61)[0]

    assert allowed.rolling_rate == pytest.approx(3.0)
    assert blocked.rolling_rate is None


def test_unstarted_period_forces_zero_totals_and_null_rates() -> None:
    now = at(6, 5)
    siang_buckets = (make_bucket(at(10, 0), PeriodName.SIANG, span(at(10, 0), 300)),)
    events = (make_event(at(6, 1, 0), track_id=1),)
    summaries = {
        summary.period: summary for summary in build_period_summaries(events, siang_buckets, now)
    }
    siang = summaries[PeriodName.SIANG]

    assert siang.started is False
    assert siang.total == 0
    assert siang.observed_seconds == 0
    assert siang.average_rate is None
    assert siang.rolling_rate is None
    assert summaries[PeriodName.PAGI].total == 1


def test_summaries_reject_naive_now() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        build_period_summaries((), {}, datetime(2026, 9, 25, 6, 5))
