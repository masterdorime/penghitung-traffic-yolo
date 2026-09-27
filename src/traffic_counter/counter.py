from __future__ import annotations

from datetime import datetime, timedelta
from typing import Final

from traffic_counter.config import LineConfig
from traffic_counter.geometry import (
    CrossingSide,
    LineGeometry,
    Point,
    TimedDistance,
    classify_side,
    interpolate_crossing,
    validate_hysteresis,
)
from traffic_counter.models import (
    DIRECTION_DOWN,
    DIRECTION_UP,
    VEHICLE_CLASSES,
    CountEvent,
    TrackedVehicle,
)
from traffic_counter.timeutils import period_for, to_jakarta

_BOX_LENGTH: Final[int] = 4
MAX_INTERPOLATION_GAP: Final[timedelta] = timedelta(seconds=2)
MAX_TRACKS_PER_SESSION: Final[int] = 5000
STABLE_EVICT_AFTER: Final[timedelta] = timedelta(seconds=60)


class _TrackState:
    __slots__ = ("counted", "side", "stable_distance", "stable_time")

    def __init__(self, side: CrossingSide, stable_distance: float, stable_time: datetime) -> None:
        self.side = side
        self.stable_distance = stable_distance
        self.stable_time = stable_time
        self.counted = False


def bottom_center(box_xyxy: tuple[float, float, float, float]) -> Point:
    if len(box_xyxy) != _BOX_LENGTH:
        raise ValueError("Tracked vehicle boxes must hold exactly four coordinates.")
    left, _, right, bottom = box_xyxy
    return ((left + right) / 2.0, bottom)


class LineCounter:
    __slots__ = ("_geometry", "_hysteresis", "_sessions")

    def __init__(self, line: LineConfig) -> None:
        validate_hysteresis(line.hysteresis)
        self._geometry = LineGeometry.from_line(line)
        self._hysteresis = line.hysteresis
        self._sessions: dict[int, dict[int, _TrackState]] = {}

    def observe(
        self,
        tracking_session_id: int,
        vehicle: TrackedVehicle,
        observed_at: datetime,
    ) -> CountEvent | None:
        if vehicle.class_name not in VEHICLE_CLASSES:
            return None
        moment = to_jakarta(observed_at)
        tracks = self._sessions.setdefault(tracking_session_id, {})
        state = tracks.get(vehicle.track_id)
        if state is not None and state.counted:
            return None
        distance = self._geometry.distance(bottom_center(vehicle.box_xyxy))
        side = classify_side(distance, self._hysteresis)
        if side is CrossingSide.INSIDE:
            return None
        if state is None:
            tracks[vehicle.track_id] = _TrackState(side, distance, moment)
            self._enforce_bound(tracks, moment, vehicle.track_id)
            return None
        if side is state.side:
            state.stable_distance = distance
            state.stable_time = moment
            return None
        previous: TimedDistance = (state.stable_time, state.stable_distance)
        gap = moment - state.stable_time
        was_negative = state.side is CrossingSide.NEGATIVE
        state.side = side
        state.stable_distance = distance
        state.stable_time = moment
        if gap > MAX_INTERPOLATION_GAP:
            return None
        crossed_at = interpolate_crossing(previous, (moment, distance))
        direction = DIRECTION_UP if was_negative else DIRECTION_DOWN
        state.counted = True
        period = period_for(crossed_at)
        if period is None:
            return None
        return CountEvent(
            tracking_session_id=tracking_session_id,
            track_id=vehicle.track_id,
            crossed_at=crossed_at,
            direction=direction,
            period=period,
            class_name=vehicle.class_name,
        )

    def end_tracking_session(self, tracking_session_id: int) -> None:
        self._sessions.pop(tracking_session_id, None)

    def _enforce_bound(
        self,
        tracks: dict[int, _TrackState],
        now: datetime,
        current_track_id: int,
    ) -> None:
        if len(tracks) <= MAX_TRACKS_PER_SESSION:
            return
        cutoff = now - STABLE_EVICT_AFTER
        candidates = [
            track_id
            for track_id, item in tracks.items()
            if track_id != current_track_id and item.stable_time <= cutoff
        ]
        candidates.sort(key=lambda track_id: tracks[track_id].stable_time)
        excess = len(tracks) - MAX_TRACKS_PER_SESSION
        for track_id in candidates[:excess]:
            del tracks[track_id]
