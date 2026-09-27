from datetime import UTC, datetime, timedelta
from typing import Final
from zoneinfo import ZoneInfo

import pytest

from traffic_counter.config import LineConfig
from traffic_counter.counter import MAX_INTERPOLATION_GAP, LineCounter, bottom_center
from traffic_counter.geometry import signed_distance
from traffic_counter.models import (
    DIRECTION_DOWN,
    DIRECTION_UP,
    VEHICLE_CLASSES,
    CountEvent,
    PeriodName,
    TrackedVehicle,
)

JAKARTA: Final[ZoneInfo] = ZoneInfo("Asia/Jakarta")
WALL_CLOCK_RE_ANCHOR_INTERVAL: Final[timedelta] = timedelta(seconds=1)
TWO_SECONDS: Final[timedelta] = timedelta(seconds=2)

PLAN_LINE: Final[LineConfig] = LineConfig(0.1, 0.8, 0.9, 0.8, 0.005)
UNIT_LINE: Final[LineConfig] = LineConfig(0.0, 0.5, 1.0, 0.5, 0.005)
WIDE_BAND_LINE: Final[LineConfig] = LineConfig(0.0, 0.5, 1.0, 0.5, 0.125)


def at(hour: int, minute: int, second: int = 0, microsecond: int = 0) -> datetime:
    return datetime(2026, 9, 25, hour, minute, second, microsecond, tzinfo=JAKARTA)


def vehicle_at(
    bottom_y: float,
    track_id: int = 7,
    class_name: str = "car",
) -> TrackedVehicle:
    return TrackedVehicle(
        track_id=track_id,
        class_name=class_name,
        confidence=0.9,
        box_xyxy=(0.45, bottom_y - 0.10, 0.55, bottom_y),
    )


def test_bottom_center_uses_the_horizontal_middle_and_the_bottom_edge() -> None:
    assert bottom_center((0.10, 0.20, 0.30, 0.40)) == (0.20, 0.40)
    assert bottom_center((0.45, 0.60, 0.55, 0.90)) == (0.50, 0.90)
    assert bottom_center((0.0, 0.0, 1.0, 1.0)) == (0.5, 1.0)


def test_bottom_center_rejects_a_box_that_is_not_four_coordinates() -> None:
    with pytest.raises(ValueError, match="four coordinates"):
        bottom_center((0.1, 0.2, 0.3))


def test_line_counter_rejects_a_zero_length_line_at_construction() -> None:
    with pytest.raises(ValueError, match="non-zero length"):
        LineCounter(LineConfig(0.4, 0.4, 0.4, 0.4, 0.005))


def test_line_counter_rejects_non_finite_line_endpoints_at_construction() -> None:
    with pytest.raises(ValueError, match="finite"):
        LineCounter(LineConfig(0.1, float("nan"), 0.9, 0.8, 0.005))


def test_line_counter_rejects_a_negative_hysteresis_band() -> None:
    with pytest.raises(ValueError, match="hysteresis"):
        LineCounter(LineConfig(0.1, 0.8, 0.9, 0.8, -0.005))


def test_first_side_is_baseline_and_jitter_does_not_count() -> None:
    base = at(6, 0)
    counter = LineCounter(PLAN_LINE)

    assert counter.observe(1, vehicle_at(0.70), base) is None
    assert counter.observe(1, vehicle_at(0.804), base + timedelta(seconds=1)) is None
    assert counter.observe(1, vehicle_at(0.70), base + timedelta(seconds=2)) is None
    event = counter.observe(1, vehicle_at(0.90), base + timedelta(seconds=3))
    assert event is not None
    assert event.track_id == 7
    assert event.direction == "up"
    assert counter.observe(1, vehicle_at(0.70), base + timedelta(seconds=4)) is None


def test_repeated_observations_on_one_side_never_count() -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)

    for offset in range(5):
        assert counter.observe(1, vehicle_at(0.4375), base + timedelta(seconds=offset)) is None


def test_negative_to_positive_is_reported_as_up() -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.4375), base) is None
    event = counter.observe(1, vehicle_at(0.5625), base + timedelta(seconds=1))

    assert event is not None
    assert event.direction == DIRECTION_UP
    assert event.direction == "up"


def test_positive_to_negative_is_reported_as_down() -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.5625), base) is None
    event = counter.observe(1, vehicle_at(0.4375), base + timedelta(seconds=1))

    assert event is not None
    assert event.direction == DIRECTION_DOWN
    assert event.direction == "down"


def test_event_carries_the_session_track_class_and_period() -> None:
    base = at(7, 30)
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(42, vehicle_at(0.4375, track_id=9, class_name="bus"), base) is None
    event = counter.observe(
        42, vehicle_at(0.5625, track_id=9, class_name="bus"), base + timedelta(seconds=2)
    )

    assert event == CountEvent(
        tracking_session_id=42,
        track_id=9,
        crossed_at=base + timedelta(seconds=1),
        direction=DIRECTION_UP,
        period=PeriodName.PAGI,
        class_name="bus",
    )


def test_crossing_time_is_interpolated_between_the_two_stable_observations() -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.4375), base) is None
    assert counter.observe(1, vehicle_at(0.504), base + timedelta(milliseconds=400)) is None
    event = counter.observe(1, vehicle_at(0.5625), base + timedelta(seconds=1))

    assert event is not None
    assert event.crossed_at == base + timedelta(milliseconds=500)


def test_the_latest_stable_observation_of_the_previous_side_is_used() -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.0), base) is None
    assert counter.observe(1, vehicle_at(0.25), base + timedelta(milliseconds=250)) is None
    event = counter.observe(1, vehicle_at(0.75), base + timedelta(milliseconds=1500))

    assert event is not None
    assert event.crossed_at == base + timedelta(milliseconds=875)
    assert event.crossed_at != base + timedelta(seconds=1)


def test_the_interpolation_gap_threshold_exceeds_the_wall_clock_re_anchor_interval() -> None:
    assert MAX_INTERPOLATION_GAP == TWO_SECONDS
    assert MAX_INTERPOLATION_GAP == WALL_CLOCK_RE_ANCHOR_INTERVAL * 2
    assert MAX_INTERPOLATION_GAP > WALL_CLOCK_RE_ANCHOR_INTERVAL


def test_a_gap_below_two_seconds_still_interpolates() -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.4375), base) is None
    event = counter.observe(
        1,
        vehicle_at(0.5625),
        base + MAX_INTERPOLATION_GAP - timedelta(microseconds=1),
    )

    assert event is not None
    assert event.crossed_at == base + MAX_INTERPOLATION_GAP / 2


def test_a_gap_of_exactly_two_seconds_still_interpolates() -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.4375), base) is None
    event = counter.observe(1, vehicle_at(0.5625), base + MAX_INTERPOLATION_GAP)

    assert event is not None
    assert event.crossed_at == base + MAX_INTERPOLATION_GAP / 2


def test_a_gap_above_two_seconds_re_baselines_instead_of_fabricating() -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.4375), base) is None
    beyond = base + MAX_INTERPOLATION_GAP + timedelta(microseconds=1)
    assert counter.observe(1, vehicle_at(0.5625), beyond) is None


@pytest.mark.parametrize(
    ("fresh_bottom", "final_bottom", "direction"),
    (
        (0.5625, 0.4375, DIRECTION_DOWN),
        (0.4375, 0.5625, DIRECTION_UP),
    ),
)
def test_a_re_baseline_preserves_the_counting_opportunity(
    fresh_bottom: float, final_bottom: float, direction: str
) -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.4375), base) is None
    assert counter.observe(1, vehicle_at(fresh_bottom), base + timedelta(seconds=3)) is None
    assert counter.observe(1, vehicle_at(fresh_bottom), base + timedelta(seconds=4)) is None
    event = counter.observe(1, vehicle_at(final_bottom), base + timedelta(seconds=5))

    assert event is not None
    assert event.direction == direction
    assert event.crossed_at == base + timedelta(milliseconds=4500)


def test_a_long_stay_inside_the_band_re_baselines_instead_of_counting() -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.4375), base) is None
    assert counter.observe(1, vehicle_at(0.5), base + timedelta(seconds=1)) is None
    assert counter.observe(1, vehicle_at(0.5625), base + timedelta(seconds=3)) is None
    event = counter.observe(1, vehicle_at(0.4375), base + timedelta(seconds=4))

    assert event is not None
    assert event.direction == DIRECTION_DOWN
    assert event.crossed_at == base + timedelta(milliseconds=3500)


def test_a_long_gap_on_one_side_still_refreshes_the_anchor() -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.4375), base) is None
    assert counter.observe(1, vehicle_at(0.4375), base + timedelta(seconds=3)) is None
    event = counter.observe(1, vehicle_at(0.5625), base + timedelta(seconds=4))

    assert event is not None
    assert event.crossed_at == base + timedelta(milliseconds=3500)


def test_a_track_that_stays_inside_the_band_never_counts() -> None:
    base = at(6, 0)
    counter = LineCounter(WIDE_BAND_LINE)

    assert counter.observe(1, vehicle_at(0.5), base) is None
    assert counter.observe(1, vehicle_at(0.625), base + timedelta(seconds=1)) is None
    assert counter.observe(1, vehicle_at(0.375), base + timedelta(seconds=2)) is None
    assert counter.observe(1, vehicle_at(0.5), base + timedelta(seconds=3)) is None


def test_the_band_edges_are_inside_and_never_move_the_latch() -> None:
    base = at(6, 0)
    counter = LineCounter(WIDE_BAND_LINE)

    assert counter.observe(1, vehicle_at(0.25), base) is None
    assert counter.observe(1, vehicle_at(0.625), base + timedelta(seconds=1)) is None
    event = counter.observe(1, vehicle_at(0.75), base + timedelta(seconds=2))

    assert event is not None
    assert event.direction == DIRECTION_UP


def test_a_baseline_inside_the_band_waits_for_the_first_stable_side() -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.5), base) is None
    assert counter.observe(1, vehicle_at(0.496), base + timedelta(seconds=1)) is None
    assert counter.observe(1, vehicle_at(0.5625), base + timedelta(seconds=2)) is None
    event = counter.observe(1, vehicle_at(0.4375), base + timedelta(seconds=3))

    assert event is not None
    assert event.direction == DIRECTION_DOWN
    assert event.crossed_at == base + timedelta(milliseconds=2500)


def test_a_reversal_after_a_count_never_produces_a_second_event() -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.4375), base) is None
    assert counter.observe(1, vehicle_at(0.5625), base + timedelta(seconds=1)) is not None
    assert counter.observe(1, vehicle_at(0.4375), base + timedelta(seconds=2)) is None
    assert counter.observe(1, vehicle_at(0.5625), base + timedelta(seconds=3)) is None
    assert counter.observe(1, vehicle_at(0.4375), base + timedelta(seconds=4)) is None


def test_repeated_reversals_after_a_count_return_none_every_time() -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)
    counter.observe(1, vehicle_at(0.4375), base)
    counter.observe(1, vehicle_at(0.5625), base + timedelta(seconds=1))

    for offset in range(2, 12):
        bottom = 0.4375 if offset % 2 == 0 else 0.5625
        assert counter.observe(1, vehicle_at(bottom), base + timedelta(seconds=offset)) is None


def test_a_counted_track_still_filters_by_class() -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)
    counter.observe(1, vehicle_at(0.4375), base)
    counter.observe(1, vehicle_at(0.5625), base + timedelta(seconds=1))

    assert counter.observe(1, vehicle_at(0.25, class_name="person"), base) is None


@pytest.mark.parametrize(
    "class_name", ("person", "bicycle", "airplane", "train", "boat", "Car", "")
)
def test_non_vehicle_classes_are_never_counted(class_name: str) -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.4375, class_name=class_name), base) is None
    assert (
        counter.observe(1, vehicle_at(0.5625, class_name=class_name), base + timedelta(seconds=1))
        is None
    )


@pytest.mark.parametrize("class_name", VEHICLE_CLASSES)
def test_every_approved_vehicle_class_contributes(class_name: str) -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.4375, class_name=class_name), base) is None
    event = counter.observe(
        1, vehicle_at(0.5625, class_name=class_name), base + timedelta(seconds=1)
    )

    assert event is not None
    assert event.class_name == class_name


def test_ignored_classes_leave_no_latch_behind() -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.4375, class_name="person"), base) is None
    assert (
        counter.observe(1, vehicle_at(0.5625, class_name="person"), base + timedelta(seconds=1))
        is None
    )
    assert counter.observe(1, vehicle_at(0.5625), base + timedelta(seconds=2)) is None


def test_observed_at_is_normalized_to_jakarta_and_keeps_the_instant() -> None:
    counter = LineCounter(UNIT_LINE)
    utc_base = at(6, 0).astimezone(UTC)

    assert counter.observe(1, vehicle_at(0.4375), utc_base) is None
    event = counter.observe(1, vehicle_at(0.5625), utc_base + timedelta(seconds=1))

    assert event is not None
    assert event.crossed_at == at(6, 0) + timedelta(milliseconds=500)
    assert event.crossed_at.tzinfo is JAKARTA
    assert event.period is PeriodName.PAGI


def test_a_naive_observed_at_raises() -> None:
    counter = LineCounter(UNIT_LINE)

    with pytest.raises(ValueError, match="timezone-aware"):
        counter.observe(1, vehicle_at(0.4375), datetime(2026, 9, 25, 6, 0))


def test_a_non_finite_box_raises() -> None:
    counter = LineCounter(UNIT_LINE)
    vehicle = TrackedVehicle(7, "car", 0.9, (0.45, 0.6, 0.55, float("nan")))

    with pytest.raises(ValueError, match="finite"):
        counter.observe(1, vehicle, at(6, 0))


def test_an_inverted_box_still_uses_its_bottom_edge() -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)
    vehicle = TrackedVehicle(7, "car", 0.9, (0.55, 0.5625, 0.45, 0.5625))

    assert counter.observe(1, vehicle, base) is None
    assert counter.observe(1, vehicle_at(0.4375), base + timedelta(seconds=1)) is not None


def test_the_bottom_center_is_used_and_not_the_box_center() -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)
    tall_above = TrackedVehicle(7, "car", 0.9, (0.45, 0.30, 0.55, 0.60))
    tall_below = TrackedVehicle(7, "car", 0.9, (0.45, 0.30, 0.55, 0.40))

    assert signed_distance((0.5, (0.30 + 0.60) / 2), UNIT_LINE) < 0.0
    assert counter.observe(1, tall_above, base) is None
    event = counter.observe(1, tall_below, base + timedelta(seconds=1))

    assert event is not None
    assert event.direction == DIRECTION_DOWN


def test_the_horizontal_center_of_the_box_is_irrelevant() -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)
    left = TrackedVehicle(7, "car", 0.9, (0.05, 0.60, 0.15, 0.60))

    assert counter.observe(1, left, base) is None
    right = TrackedVehicle(7, "car", 0.9, (0.85, 0.30, 0.95, 0.30))
    event = counter.observe(1, right, base + timedelta(seconds=1))

    assert event is not None
    assert event.direction == DIRECTION_DOWN


def test_two_tracks_on_the_same_line_are_counted_independently() -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.4375, track_id=1), base) is None
    assert counter.observe(1, vehicle_at(0.4375, track_id=2), base) is None
    first = counter.observe(1, vehicle_at(0.5625, track_id=1), base + timedelta(seconds=1))
    second = counter.observe(1, vehicle_at(0.5625, track_id=2), base + timedelta(seconds=1))

    assert first is not None
    assert second is not None
    assert first.track_id == 1
    assert second.track_id == 2


def test_two_counters_sharing_a_line_keep_separate_state() -> None:
    base = at(6, 0)
    first = LineCounter(UNIT_LINE)
    second = LineCounter(UNIT_LINE)

    assert first.observe(1, vehicle_at(0.4375), base) is None
    assert first.observe(1, vehicle_at(0.5625), base + timedelta(seconds=1)) is not None
    assert second.observe(1, vehicle_at(0.4375), base) is None
    assert second.observe(1, vehicle_at(0.5625), base + timedelta(seconds=1)) is not None


def test_the_same_track_id_in_a_new_session_is_only_a_baseline() -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.4375), base) is None
    assert counter.observe(2, vehicle_at(0.4375), base + timedelta(seconds=1)) is None


def test_a_new_session_can_count_the_same_numeric_track_id_again() -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.4375), base) is None
    first = counter.observe(1, vehicle_at(0.5625), base + timedelta(seconds=1))
    assert counter.observe(2, vehicle_at(0.5625), base + timedelta(seconds=2)) is None
    second = counter.observe(2, vehicle_at(0.4375), base + timedelta(seconds=3))

    assert first is not None
    assert second is not None
    assert first.tracking_session_id == 1
    assert second.tracking_session_id == 2
    assert first.track_id == second.track_id
    assert first.direction == DIRECTION_UP
    assert second.direction == DIRECTION_DOWN


def test_a_later_session_number_does_not_disturb_an_earlier_one() -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.4375), base) is None
    assert counter.observe(2, vehicle_at(0.4375), base + timedelta(seconds=1)) is None
    first = counter.observe(1, vehicle_at(0.5625), base + timedelta(seconds=2))
    second = counter.observe(2, vehicle_at(0.5625), base + timedelta(seconds=2))

    assert first is not None
    assert second is not None


def test_ending_a_tracking_session_resets_its_latch() -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.4375), base) is None
    assert counter.observe(1, vehicle_at(0.5625), base + timedelta(seconds=1)) is not None
    counter.end_tracking_session(1)
    assert counter.observe(1, vehicle_at(0.4375), base + timedelta(seconds=2)) is None
    assert counter.observe(1, vehicle_at(0.5625), base + timedelta(seconds=3)) is not None


def test_ending_a_tracking_session_allows_a_fresh_count() -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)
    counter.observe(1, vehicle_at(0.4375), base)
    counter.observe(1, vehicle_at(0.5625), base + timedelta(seconds=1))
    counter.end_tracking_session(1)
    counter.observe(1, vehicle_at(0.4375), base + timedelta(seconds=2))
    event = counter.observe(1, vehicle_at(0.5625), base + timedelta(seconds=3))

    assert event is not None
    assert event.tracking_session_id == 1


def test_ending_a_tracking_session_leaves_other_sessions_alone() -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.4375), base) is None
    assert counter.observe(2, vehicle_at(0.4375), base) is None
    counter.end_tracking_session(1)
    assert counter.observe(1, vehicle_at(0.5625), base + timedelta(seconds=1)) is None
    assert counter.observe(2, vehicle_at(0.5625), base + timedelta(seconds=1)) is not None


def test_ending_an_unknown_tracking_session_is_a_no_op() -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)

    counter.end_tracking_session(99)
    counter.end_tracking_session(99)
    assert counter.observe(1, vehicle_at(0.4375), base) is None
    assert counter.observe(1, vehicle_at(0.5625), base + timedelta(seconds=1)) is not None


def test_ending_every_tracking_session_keeps_the_counter_usable() -> None:
    base = at(6, 0)
    counter = LineCounter(UNIT_LINE)
    counter.observe(1, vehicle_at(0.4375), base)
    counter.observe(2, vehicle_at(0.4375), base)

    counter.end_tracking_session(1)
    counter.end_tracking_session(2)

    assert counter.observe(3, vehicle_at(0.4375), base + timedelta(seconds=1)) is None
    assert counter.observe(3, vehicle_at(0.5625), base + timedelta(seconds=2)) is not None


def test_a_crossing_at_six_in_the_morning_is_admitted_as_pagi() -> None:
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.4375), at(5, 59, 59, 750000)) is None
    event = counter.observe(1, vehicle_at(0.5625), at(6, 0, 0, 250000))

    assert event is not None
    assert event.period is PeriodName.PAGI
    assert event.crossed_at == at(6, 0, 0, 0)


def test_a_crossing_at_exactly_ten_is_admitted_as_siang() -> None:
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.4375), at(9, 59, 59, 750000)) is None
    event = counter.observe(1, vehicle_at(0.5625), at(10, 0, 0, 250000))

    assert event is not None
    assert event.period is PeriodName.SIANG
    assert event.crossed_at == at(10, 0, 0, 0)


def test_a_crossing_at_exactly_fifteen_is_admitted_as_sore() -> None:
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.4375), at(14, 59, 59, 750000)) is None
    event = counter.observe(1, vehicle_at(0.5625), at(15, 0, 0, 250000))

    assert event is not None
    assert event.period is PeriodName.SORE


def test_the_period_follows_the_interpolated_instant_not_the_observations() -> None:
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.375), at(9, 59, 59, 500000)) is None
    event = counter.observe(1, vehicle_at(0.875), at(10, 0, 0, 500000))

    assert event is not None
    assert event.crossed_at == at(9, 59, 59, 750000)
    assert event.period is PeriodName.PAGI


def test_a_slow_second_side_approach_after_the_boundary_is_still_siang() -> None:
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.125), at(9, 59, 59, 500000)) is None
    event = counter.observe(1, vehicle_at(0.625), at(10, 0, 0, 500000))

    assert event is not None
    assert event.crossed_at == at(10, 0, 0, 250000)
    assert event.period is PeriodName.SIANG


def test_a_crossing_at_exactly_nineteen_is_not_admitted() -> None:
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.4375), at(18, 59, 59, 750000)) is None
    assert counter.observe(1, vehicle_at(0.5625), at(19, 0, 0, 250000)) is None


@pytest.mark.parametrize(
    ("first_time", "second_time"),
    (
        (at(5, 59, 58, 500000), at(5, 59, 59, 500000)),
        (at(19, 0, 0, 500000), at(19, 0, 1, 500000)),
        (at(20, 30), at(20, 30, 1)),
        (at(23, 59, 58, 500000), at(23, 59, 59, 500000)),
    ),
)
def test_crossings_outside_the_reporting_schedule_are_suppressed(
    first_time: datetime, second_time: datetime
) -> None:
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.4375), first_time) is None
    assert counter.observe(1, vehicle_at(0.5625), second_time) is None


def test_a_drain_window_crossing_before_nineteen_is_still_admitted() -> None:
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.49), at(18, 59, 59, 900000)) is None
    event = counter.observe(1, vehicle_at(0.89), at(19, 0, 0, 100000))

    assert event is not None
    assert event.crossed_at == pytest.approx(at(18, 59, 59, 905000), abs=timedelta(microseconds=1))
    assert event.crossed_at < at(19, 0, 0)
    assert event.period is PeriodName.SORE


def test_a_suppressed_crossing_is_not_reported_twice() -> None:
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.4375), at(19, 0, 0, 500000)) is None
    assert counter.observe(1, vehicle_at(0.5625), at(19, 0, 1, 500000)) is None
    assert counter.observe(1, vehicle_at(0.4375), at(19, 0, 2, 500000)) is None
    assert counter.observe(1, vehicle_at(0.5625), at(19, 0, 3, 500000)) is None


def test_a_pre_six_baseline_suppresses_every_later_crossing() -> None:
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.4375), at(5, 59, 59)) is None
    assert counter.observe(1, vehicle_at(0.5625), at(5, 59, 59, 100000)) is None
    assert counter.observe(1, vehicle_at(0.4375), at(6, 0, 0, 500000)) is None
    assert counter.observe(1, vehicle_at(0.5625), at(6, 0, 1, 500000)) is None


def test_the_pre_six_reversal_would_otherwise_be_an_in_schedule_crossing() -> None:
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.4375), at(6, 0, 0, 500000)) is None
    event = counter.observe(1, vehicle_at(0.5625), at(6, 0, 1, 500000))

    assert event is not None
    assert event.crossed_at == at(6, 0, 1, 0)
    assert event.period is PeriodName.PAGI


def test_a_track_first_seen_after_nineteen_never_counts() -> None:
    counter = LineCounter(UNIT_LINE)

    assert counter.observe(1, vehicle_at(0.4375), at(19, 0, 30)) is None
    assert counter.observe(1, vehicle_at(0.5625), at(19, 0, 31)) is None


def test_the_line_hysteresis_value_is_honoured() -> None:
    base = at(6, 0)
    counter = LineCounter(WIDE_BAND_LINE)

    assert counter.observe(1, vehicle_at(0.25), base) is None
    assert counter.observe(1, vehicle_at(0.60), base + timedelta(seconds=1)) is None
    event = counter.observe(1, vehicle_at(0.875), base + timedelta(seconds=2))

    assert event is not None
    assert event.direction == DIRECTION_UP


def test_a_tight_band_counts_where_a_wide_band_does_not() -> None:
    base = at(6, 0)
    tight = LineCounter(LineConfig(0.0, 0.5, 1.0, 0.5, 0.001))
    wide = LineCounter(LineConfig(0.0, 0.5, 1.0, 0.5, 0.05))

    assert tight.observe(1, vehicle_at(0.495), base) is None
    assert tight.observe(1, vehicle_at(0.505), base + timedelta(seconds=1)) is not None
    assert wide.observe(1, vehicle_at(0.495), base) is None
    assert wide.observe(1, vehicle_at(0.505), base + timedelta(seconds=1)) is None
