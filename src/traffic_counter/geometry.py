from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final

from traffic_counter.config import LineConfig

type Point = tuple[float, float]
type TimedDistance = tuple[datetime, float]


class CrossingSide(StrEnum):
    NEGATIVE = "negative"
    POSITIVE = "positive"
    INSIDE = "inside"


_ZERO_LENGTH_MESSAGE: Final[str] = "Line endpoints must define a line with a non-zero length."
_NON_FINITE_LINE_MESSAGE: Final[str] = "Line endpoints must all be finite numbers."
_NON_FINITE_POINT_MESSAGE: Final[str] = "Crossing points must hold two finite coordinates."
_SAME_SIDE_MESSAGE: Final[str] = (
    "Crossing interpolation requires opposite non-zero signed distances."
)
_NEGATIVE_HYSTERESIS_MESSAGE: Final[str] = "Line hysteresis must not be a negative number."


@dataclass(frozen=True, slots=True)
class LineGeometry:
    origin_x: float
    origin_y: float
    delta_x: float
    delta_y: float
    length: float

    @classmethod
    def from_line(cls, line: LineConfig) -> LineGeometry:
        if not all(math.isfinite(value) for value in (line.x1, line.y1, line.x2, line.y2)):
            raise ValueError(_NON_FINITE_LINE_MESSAGE)
        delta_x = line.x2 - line.x1
        delta_y = line.y2 - line.y1
        length = math.hypot(delta_x, delta_y)
        if length == 0.0:
            raise ValueError(_ZERO_LENGTH_MESSAGE)
        return cls(line.x1, line.y1, delta_x, delta_y, length)

    def distance(self, point: Point) -> float:
        if not (math.isfinite(point[0]) and math.isfinite(point[1])):
            raise ValueError(_NON_FINITE_POINT_MESSAGE)
        cross = self.delta_x * (point[1] - self.origin_y) - self.delta_y * (
            point[0] - self.origin_x
        )
        return cross / self.length


def signed_distance(point: Point, line: LineConfig) -> float:
    return LineGeometry.from_line(line).distance(point)


def validate_hysteresis(hysteresis: float) -> None:
    if hysteresis < 0.0:
        raise ValueError(_NEGATIVE_HYSTERESIS_MESSAGE)


def classify_side(distance: float, hysteresis: float) -> CrossingSide:
    validate_hysteresis(hysteresis)
    if not math.isfinite(distance):
        raise ValueError(_NON_FINITE_POINT_MESSAGE)
    if abs(distance) <= hysteresis:
        return CrossingSide.INSIDE
    return CrossingSide.POSITIVE if distance > 0.0 else CrossingSide.NEGATIVE


def interpolate_crossing(previous: TimedDistance, current: TimedDistance) -> datetime:
    previous_time, previous_distance = previous
    current_time, current_distance = current
    if not math.isfinite(previous_distance) or not math.isfinite(current_distance):
        raise ValueError(_NON_FINITE_POINT_MESSAGE)
    if previous_distance == 0.0 or current_distance == 0.0:
        raise ValueError(_SAME_SIDE_MESSAGE)
    if (previous_distance > 0.0) == (current_distance > 0.0):
        raise ValueError(_SAME_SIDE_MESSAGE)
    ratio = abs(previous_distance) / (abs(previous_distance) + abs(current_distance))
    return previous_time + (current_time - previous_time) * ratio
