from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from types import MappingProxyType

from traffic_counter.models import AppSnapshot, Frame, FrameSlotStats
from traffic_counter.timeutils import WallClock

type Clock = Callable[[], datetime]


def _sealed(snapshot: AppSnapshot) -> AppSnapshot:
    return replace(
        snapshot,
        periods=MappingProxyType(dict(snapshot.periods)),
        minutes=tuple(snapshot.minutes),
    )


class SnapshotStore:
    __slots__ = ("_lock", "_snapshot")

    def __init__(self, clock: Clock | None = None) -> None:
        wall: Clock = WallClock().now if clock is None else clock
        self._lock = threading.Lock()
        self._snapshot = _sealed(AppSnapshot.initial(wall()))

    def get(self) -> AppSnapshot:
        with self._lock:
            return self._snapshot

    def publish(self, snapshot: AppSnapshot) -> None:
        sealed = _sealed(snapshot)
        with self._lock:
            self._snapshot = sealed


class LatestFrameSlot:
    __slots__ = ("_lock", "_pending", "_received", "_replaced")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: Frame | None = None
        self._received = 0
        self._replaced = 0

    def put(self, frame: Frame) -> None:
        with self._lock:
            if self._pending is not None:
                self._replaced += 1
            self._pending = frame
            self._received += 1

    def take(self) -> Frame | None:
        with self._lock:
            frame = self._pending
            self._pending = None
            return frame

    def stats(self) -> FrameSlotStats:
        with self._lock:
            return FrameSlotStats(received=self._received, replaced=self._replaced)
