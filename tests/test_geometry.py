from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Final
from zoneinfo import ZoneInfo

import pytest

from traffic_counter.config import LineConfig
from traffic_counter.geometry import (
    CrossingSide,
    classify_side,
    interpolate_crossing,
    signed_distance,
)

JAKARTA: Final[ZoneInfo] = ZoneInfo("Asia/Jakarta")

HORIZONTAL: Final[LineConfig] = LineConfig(0.1, 0.8, 0.9, 0.8, 0.005)
HORIZONTAL_UNIT: Final[LineConfig] = LineConfig(0.0, 0.5, 1.0, 0.5, 0.005)
HORIZONTAL_REVERSED: Final[LineConfig] = LineConfig(0.9, 0.8, 0.1, 0.8, 0.005)
VERTICAL: Final[LineConfig] = LineConfig(0.5, 0.1, 0.5, 0.9, 0.005)
DIAGONAL: Final[LineConfig] = LineConfig(0.0, 0.0, 0.5, 0.5, 0.005)


def at(hour: int, minute: int, second: int = 0, microsecond: int = 0) -> datetime:
    return datetime(2026, 9, 25, hour, minute, second, microsecond, tzinfo=JAKARTA)


def test_crossing_side_is_a_string_enum_with_the_three_required_members() -> None:
    assert issubclass(CrossingSide, StrEnum)
    assert [member.name for member in CrossingSide] == ["NEGATIVE", "POSITIVE", "INSIDE"]
    assert CrossingSide.NEGATIVE.value == "negative"
    assert CrossingSide.POSITIVE.value == "positive"
    assert CrossingSide.INSIDE.value == "inside"
    assert CrossingSide("negative") is CrossingSide.NEGATIVE


def test_signed_distance_measures_vertical_offset_from_a_horizontal_line() -> None:
    assert signed_distance((0.5, 0.7), HORIZONTAL) == pytest.approx(-0.1)
    assert signed_distance((0.5, 0.9), HORIZONTAL) == pytest.approx(0.1)
    assert signed_distance((0.5, 0.8), HORIZONTAL) == pytest.approx(0.0)
    assert signed_distance((0.5, 0.8), HORIZONTAL) == 0.0


def test_signed_distance_is_independent_of_the_queried_x_position() -> None:
    assert signed_distance((0.05, 0.9), HORIZONTAL) == signed_distance((0.95, 0.9), HORIZONTAL)


def test_signed_distance_follows_the_direction_of_the_configured_endpoints() -> None:
    assert signed_distance((0.5, 0.9), HORIZONTAL_REVERSED) == -signed_distance(
        (0.5, 0.9), HORIZONTAL
    )
    assert signed_distance((0.5, 0.7), HORIZONTAL_REVERSED) > 0.0


def test_signed_distance_measures_horizontal_offset_from_a_vertical_line() -> None:
    assert signed_distance((0.6, 0.5), VERTICAL) == pytest.approx(-0.1)
    assert signed_distance((0.4, 0.5), VERTICAL) == pytest.approx(0.1)
    assert signed_distance((0.5, 0.5), VERTICAL) == 0.0


def test_signed_distance_normalizes_by_the_line_length() -> None:
    assert signed_distance((0.5, 0.0), DIAGONAL) == pytest.approx(-0.5 / 2**0.5)
    assert signed_distance((0.5, 0.0), DIAGONAL) == pytest.approx(-0.3535533905932738)
    assert signed_distance((0.0, 0.5), DIAGONAL) == pytest.approx(0.3535533905932738)
    assert signed_distance((0.25, 0.25), DIAGONAL) == 0.0


def test_signed_distance_is_a_normalized_quantity_not_a_raw_cross_product() -> None:
    raw_cross_product = 0.5 * 0.0 - 0.5 * 0.5

    assert raw_cross_product == -0.25
    assert signed_distance((0.5, 0.0), DIAGONAL) != raw_cross_product


def test_signed_distance_survives_a_resolution_change() -> None:
    pixel_line = LineConfig(128.0, 576.0, 1152.0, 576.0, 0.005)
    normalized = signed_distance((0.5, 0.9), HORIZONTAL)
    in_pixels = signed_distance((640.0, 648.0), pixel_line)

    assert normalized == pytest.approx(0.1)
    assert in_pixels == pytest.approx(72.0)
    assert in_pixels == pytest.approx(720.0 * normalized)
    assert signed_distance((640.0, 504.0), pixel_line) == pytest.approx(-72.0)


def test_signed_distance_rejects_a_zero_length_line() -> None:
    with pytest.raises(ValueError, match="non-zero length"):
        signed_distance((0.5, 0.5), LineConfig(0.4, 0.4, 0.4, 0.4, 0.005))


def test_signed_distance_rejects_non_finite_line_endpoints() -> None:
    with pytest.raises(ValueError, match="finite"):
        signed_distance((0.5, 0.5), LineConfig(float("nan"), 0.8, 0.9, 0.8, 0.005))
    with pytest.raises(ValueError, match="finite"):
        signed_distance((0.5, 0.5), LineConfig(0.1, 0.8, float("inf"), 0.8, 0.005))


def test_signed_distance_rejects_a_non_finite_point() -> None:
    with pytest.raises(ValueError, match="finite"):
        signed_distance((0.5, float("nan")), HORIZONTAL)
    with pytest.raises(ValueError, match="finite"):
        signed_distance((float("-inf"), 0.5), HORIZONTAL)


@pytest.mark.parametrize(
    ("distance", "hysteresis", "expected"),
    (
        (0.0, 0.005, CrossingSide.INSIDE),
        (0.004, 0.005, CrossingSide.INSIDE),
        (0.005, 0.005, CrossingSide.INSIDE),
        (-0.005, 0.005, CrossingSide.INSIDE),
        (0.0051, 0.005, CrossingSide.POSITIVE),
        (-0.0051, 0.005, CrossingSide.NEGATIVE),
        (0.2, 0.005, CrossingSide.POSITIVE),
        (-0.2, 0.005, CrossingSide.NEGATIVE),
    ),
)
def test_classify_side_treats_the_hysteresis_band_as_inclusive(
    distance: float, hysteresis: float, expected: CrossingSide
) -> None:
    assert classify_side(distance, hysteresis) is expected


def test_classify_side_uses_the_latch_only_outside_the_band() -> None:
    assert classify_side(0.9, 0.005) is CrossingSide.POSITIVE
    assert classify_side(0.006, 0.005) is CrossingSide.POSITIVE
    assert classify_side(0.005, 0.005) is CrossingSide.INSIDE


def test_classify_side_rejects_a_negative_hysteresis_band() -> None:
    with pytest.raises(ValueError, match="hysteresis"):
        classify_side(0.01, -0.005)


def test_interpolate_crossing_matches_the_plan_example() -> None:
    t0 = at(6, 0)
    t1 = t0 + timedelta(milliseconds=200)

    crossing = interpolate_crossing((t0, -0.2), (t1, 0.2))

    assert crossing == t0 + timedelta(milliseconds=100)


def test_interpolate_crossing_weights_by_the_absolute_distances() -> None:
    t0 = at(6, 0)
    t1 = t0 + timedelta(milliseconds=300)

    crossing = interpolate_crossing((t0, -0.2), (t1, 0.1))

    assert (crossing - t0).total_seconds() == pytest.approx(0.2)


def test_interpolate_crossing_is_symmetric_under_argument_order() -> None:
    t0 = at(6, 0)
    t1 = t0 + timedelta(milliseconds=300)

    assert interpolate_crossing((t0, -0.2), (t1, 0.1)) == interpolate_crossing(
        (t1, 0.1), (t0, -0.2)
    )


def test_interpolate_crossing_handles_a_slow_approach_on_the_second_side() -> None:
    t0 = at(6, 0)
    t1 = t0 + timedelta(seconds=4)

    crossing = interpolate_crossing((t0, -0.5), (t1, 0.5))

    assert crossing == t0 + timedelta(seconds=2)


def test_interpolate_crossing_is_exact_at_binary_fractions() -> None:
    t0 = at(9, 59, 59, 500000)
    t1 = at(10, 0, 0, 500000)

    assert interpolate_crossing((t0, -0.125), (t1, 0.375)) == at(9, 59, 59, 750000)
    assert interpolate_crossing((t0, -0.375), (t1, 0.125)) == at(10, 0, 0, 250000)


def test_interpolate_crossing_preserves_the_instant_across_timezones() -> None:
    t0 = at(6, 0)
    t1 = t0 + timedelta(milliseconds=200)
    utc_pair = (
        (t0.astimezone(UTC), -0.2),
        (t1.astimezone(UTC), 0.2),
    )

    crossing = interpolate_crossing(*utc_pair)

    assert crossing == t0 + timedelta(milliseconds=100)


@pytest.mark.parametrize(
    ("previous", "current"),
    (
        ((at(6, 0), 0.2), (at(6, 0, 1), 0.1)),
        ((at(6, 0), -0.2), (at(6, 0, 1), -0.1)),
        ((at(6, 0), 0.0), (at(6, 0, 1), 0.1)),
        ((at(6, 0), 0.1), (at(6, 0, 1), 0.0)),
        ((at(6, 0), -0.0), (at(6, 0, 1), -0.1)),
    ),
)
def test_interpolate_crossing_requires_opposite_nonzero_distances(
    previous: tuple[datetime, float], current: tuple[datetime, float]
) -> None:
    with pytest.raises(ValueError, match="opposite"):
        interpolate_crossing(previous, current)


def test_interpolate_crossing_rejects_earlier_orderings_gracefully() -> None:
    t0 = at(6, 0)
    t1 = t0 + timedelta(milliseconds=200)

    assert interpolate_crossing((t1, 0.2), (t0, -0.2)) == t0 + timedelta(milliseconds=100)


def test_geometry_rejects_a_zero_length_line_only_once_per_call() -> None:
    degenerate = LineConfig(0.4, 0.4, 0.4, 0.4, 0.005)

    for _ in range(2):
        with pytest.raises(ValueError, match="non-zero length"):
            signed_distance((0.5, 0.5), degenerate)
