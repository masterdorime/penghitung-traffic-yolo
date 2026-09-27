from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import NDArray

VEHICLE_CLASSES: Final[tuple[str, ...]] = ("car", "motorcycle", "bus", "truck")
DIRECTION_UP: Final[str] = "up"
DIRECTION_DOWN: Final[str] = "down"


class PeriodName(StrEnum):
    PAGI = "pagi"
    SIANG = "siang"
    SORE = "sore"


class AppState(StrEnum):
    STARTING = "starting"
    WAITING_FOR_WINDOW = "waiting_for_window"
    CONNECTING = "connecting"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class StreamState(StrEnum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class Frame:
    image: NDArray[np.uint8]
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class TrackedVehicle:
    track_id: int
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class CountEvent:
    tracking_session_id: int
    track_id: int
    crossed_at: datetime
    direction: str
    period: PeriodName
    class_name: str


@dataclass(frozen=True, slots=True)
class HealthBucket:
    bucket_start: datetime
    period: PeriodName
    seconds: tuple[datetime, ...]


@dataclass(frozen=True, slots=True)
class PeriodSummary:
    period: PeriodName
    started: bool
    total: int
    observed_seconds: int
    average_rate: float | None
    rolling_rate: float | None


@dataclass(frozen=True, slots=True)
class MinutePoint:
    minute_start: datetime
    period: PeriodName
    count: int
    observed_seconds: int


@dataclass(frozen=True, slots=True)
class AppSnapshot:
    generated_at: datetime
    state: AppState
    stream_state: StreamState
    actual_fps: float
    target_fps: int
    replaced_frames: int
    periods: Mapping[PeriodName, PeriodSummary]
    minutes: tuple[MinutePoint, ...]
    csv_export_ok: bool
    dropped_events: int
    incomplete: bool
    last_error: str | None
    tracking_session_changes: int

    @classmethod
    def initial(cls, now: datetime) -> AppSnapshot:
        return cls(
            generated_at=now,
            state=AppState.STOPPED,
            stream_state=StreamState.DISCONNECTED,
            actual_fps=0.0,
            target_fps=0,
            replaced_frames=0,
            periods=MappingProxyType(
                {
                    period: PeriodSummary(
                        period=period,
                        started=False,
                        total=0,
                        observed_seconds=0,
                        average_rate=None,
                        rolling_rate=None,
                    )
                    for period in PeriodName
                }
            ),
            minutes=(),
            csv_export_ok=False,
            dropped_events=0,
            incomplete=False,
            last_error=None,
            tracking_session_changes=0,
        )


@dataclass(frozen=True, slots=True)
class FrameSlotStats:
    received: int
    replaced: int


@dataclass(frozen=True, slots=True)
class StreamDiagnostics:
    state: StreamState
    attempts: int
    replaced_frames: int
    last_error: str | None
    stderr_tail: tuple[str, ...]
