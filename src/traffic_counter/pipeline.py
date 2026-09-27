from __future__ import annotations

import sqlite3
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from queue import Queue
from types import MappingProxyType
from typing import Any, Final

from traffic_counter.config import AppConfig
from traffic_counter.density import HealthAccumulator, build_period_summaries
from traffic_counter.models import (
    AppSnapshot,
    AppState,
    CountEvent,
    Frame,
    HealthBucket,
    StreamState,
    TrackedVehicle,
)
from traffic_counter.timeutils import JAKARTA, report_date

type StorageCommand = (
    InsertEvent
    | WriteHealthBuckets
    | RefreshRollups
    | ExportCsv
    | Purge
    | Stop
    | StartAppSession
    | StartTrackingSession
    | StopTrackingSession
    | StopAppSession
)

SNAPSHOT_MIN_INTERVAL: Final[timedelta] = timedelta(seconds=1)
FPS_WINDOW: Final[int] = 30
ERROR_PREVIEW: Final[int] = 200
DRAIN_CUTOFF_HOUR: Final[int] = 19
SLOT_POLL_SECONDS: Final[float] = 0.05
ACK_TIMEOUT_SECONDS: Final[float] = 5.0


@dataclass(frozen=True, slots=True)
class InsertEvent:
    event: CountEvent


@dataclass(frozen=True, slots=True)
class WriteHealthBuckets:
    buckets: tuple[HealthBucket, ...]


@dataclass(frozen=True, slots=True)
class RefreshRollups:
    report_day: date
    now: datetime


@dataclass(frozen=True, slots=True)
class ExportCsv:
    report_day: date
    now: datetime


@dataclass(frozen=True, slots=True)
class Purge:
    report_day: date


@dataclass(frozen=True, slots=True)
class StartAppSession:
    started_at: datetime


@dataclass(frozen=True, slots=True)
class StartTrackingSession:
    app_session_id: int
    started_at: datetime


@dataclass(frozen=True, slots=True)
class StopTrackingSession:
    tracking_session_id: int
    stopped_at: datetime


@dataclass(frozen=True, slots=True)
class StopAppSession:
    app_session_id: int
    stopped_at: datetime
    state: str


@dataclass(frozen=True, slots=True)
class Stop:
    pass


@dataclass(frozen=True, slots=True)
class StorageResult:
    ok: bool
    duplicate: bool
    error: str | None
    value: int | None = None


class StorageWorker:
    __slots__ = ("_closed", "_exporter", "_queue", "_store", "_thread")

    def __init__(self, store: Any, exporter: Any) -> None:
        self._store = store
        self._exporter = exporter
        self._queue: Queue[
            tuple[StorageCommand, list[StorageResult], threading.Event]
        ] = Queue()
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="storage-writer")
        self._thread.start()

    def submit(self, command: StorageCommand) -> StorageResult:
        if self._closed:
            return StorageResult(ok=False, duplicate=False, error="worker closed", value=None)
        if not self._thread.is_alive():
            return StorageResult(ok=False, duplicate=False, error="worker stopped", value=None)
        box: list[StorageResult] = []
        done = threading.Event()
        with suppress(Exception):
            self._queue.put((command, box, done))
        signaled = done.wait(timeout=ACK_TIMEOUT_SECONDS)
        if not signaled or not box:
            return StorageResult(
                ok=False, duplicate=False, error="storage acknowledgement timed out", value=None
            )
        return box[0]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._thread.is_alive():
            box: list[StorageResult] = []
            done = threading.Event()
            with suppress(Exception):
                self._queue.put((Stop(), box, done))
                done.wait(timeout=5.0)
        with suppress(Exception):
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while True:
            command, box, done = self._queue.get()
            try:
                box.append(self._execute(command))
            except Exception as exc:
                box.append(StorageResult(ok=False, duplicate=False, error=str(exc)[:200]))
            finally:
                done.set()
                self._queue.task_done()
            if isinstance(command, Stop):
                return

    def _execute(self, command: StorageCommand) -> StorageResult:
        if isinstance(command, InsertEvent):
            try:
                inserted = self._store.insert_event(command.event)
                if inserted:
                    return StorageResult(ok=True, duplicate=False, error=None, value=None)
                return StorageResult(ok=True, duplicate=True, error=None, value=None)
            except sqlite3.Error as exc:
                text = str(exc)[:ERROR_PREVIEW] or "sqlite error"
                return StorageResult(ok=False, duplicate=False, error=text, value=None)
        if isinstance(command, WriteHealthBuckets):
            try:
                self._store.write_health_buckets(command.buckets)
                return StorageResult(ok=True, duplicate=False, error=None, value=None)
            except sqlite3.Error as exc:
                text = str(exc)[:ERROR_PREVIEW] or "sqlite error"
                return StorageResult(ok=False, duplicate=False, error=text, value=None)
        if isinstance(command, RefreshRollups):
            try:
                self._store.refresh_minute_rollups(command.report_day, command.now)
                return StorageResult(ok=True, duplicate=False, error=None, value=None)
            except sqlite3.Error as exc:
                text = str(exc)[:ERROR_PREVIEW] or "sqlite error"
                return StorageResult(ok=False, duplicate=False, error=text, value=None)
        if isinstance(command, ExportCsv):
            try:
                exported = self._exporter.try_export(command.report_day, command.now)
                if exported:
                    return StorageResult(ok=True, duplicate=False, error=None, value=None)
                return StorageResult(ok=False, duplicate=False, error="csv export failed")
            except OSError as exc:
                text = str(exc)[:ERROR_PREVIEW] or "csv export failed"
                return StorageResult(ok=False, duplicate=False, error=text, value=None)
        if isinstance(command, Purge):
            try:
                self._store.purge_before(command.report_day)
            except sqlite3.Error as exc:
                text = str(exc)[:ERROR_PREVIEW] or "sqlite error"
                return StorageResult(ok=False, duplicate=False, error=text, value=None)
            try:
                self._exporter.purge_before(command.report_day)
            except OSError as exc:
                text = str(exc)[:ERROR_PREVIEW] or "purge failed"
                return StorageResult(ok=False, duplicate=False, error=text, value=None)
            return StorageResult(ok=True, duplicate=False, error=None, value=None)
        if isinstance(command, StartAppSession):
            try:
                sid = self._store.start_app_session(command.started_at)
                return StorageResult(ok=True, duplicate=False, error=None, value=int(sid))
            except sqlite3.Error as exc:
                text = str(exc)[:ERROR_PREVIEW] or "sqlite error"
                return StorageResult(ok=False, duplicate=False, error=text, value=None)
        if isinstance(command, StartTrackingSession):
            try:
                sid = self._store.start_tracking_session(
                    command.app_session_id, command.started_at
                )
                return StorageResult(ok=True, duplicate=False, error=None, value=int(sid))
            except sqlite3.Error as exc:
                text = str(exc)[:ERROR_PREVIEW] or "sqlite error"
                return StorageResult(ok=False, duplicate=False, error=text, value=None)
        if isinstance(command, StopTrackingSession):
            try:
                self._store.stop_tracking_session(
                    command.tracking_session_id, command.stopped_at
                )
                return StorageResult(ok=True, duplicate=False, error=None, value=None)
            except sqlite3.Error as exc:
                text = str(exc)[:ERROR_PREVIEW] or "sqlite error"
                return StorageResult(ok=False, duplicate=False, error=text, value=None)
        if isinstance(command, StopAppSession):
            try:
                self._store.stop_app_session(
                    command.app_session_id, command.stopped_at, command.state
                )
                return StorageResult(ok=True, duplicate=False, error=None, value=None)
            except sqlite3.Error as exc:
                text = str(exc)[:ERROR_PREVIEW] or "sqlite error"
                return StorageResult(ok=False, duplicate=False, error=text, value=None)
        return StorageResult(ok=True, duplicate=False, error=None, value=None)


class TrafficPipeline:
    __slots__ = (
        "_accepting",
        "_acked_events",
        "_all_buckets",
        "_app_session_id",
        "_clock",
        "_config",
        "_counter",
        "_csv_export_ok",
        "_detector",
        "_drain_cutoff",
        "_draining",
        "_dropped_events",
        "_factory",
        "_frame_count",
        "_frame_times",
        "_health",
        "_incomplete",
        "_last_csv_minute",
        "_last_error",
        "_last_snapshot_at",
        "_lock",
        "_needs_new_session",
        "_overlaid",
        "_owns_worker",
        "_report_day",
        "_running",
        "_sleeper",
        "_slot",
        "_snapshots",
        "_state",
        "_stop_requested",
        "_store",
        "_stream_state",
        "_tracking_session_changes",
        "_tracking_session_id",
        "_worker",
    )

    def __init__(
        self,
        *,
        config: AppConfig,
        store: Any,
        exporter: Any,
        snapshots: Any,
        slot: Any,
        counter: Any,
        detector_factory: Any,
        clock: Callable[[], datetime],
        sleeper: Callable[[float], None] | None = None,
        overlay: Any | None = None,
        worker: StorageWorker | None = None,
    ) -> None:
        self._config = config
        self._store = store
        self._snapshots = snapshots
        self._slot = slot
        self._counter = counter
        self._factory = detector_factory
        self._clock = clock
        self._sleeper: Callable[[float], None] = time.sleep if sleeper is None else sleeper
        self._overlaid = overlay
        if worker is None:
            self._worker = StorageWorker(store, exporter)
            self._owns_worker = True
        else:
            self._worker = worker
            self._owns_worker = False
        self._health = HealthAccumulator()
        self._detector: Any | None = None
        self._app_session_id: int | None = None
        self._tracking_session_id: int | None = None
        self._tracking_session_changes = 0
        self._dropped_events = 0
        self._csv_export_ok = False
        self._incomplete = False
        self._last_error: str | None = None
        self._state = AppState.STOPPED
        self._stream_state = StreamState.DISCONNECTED
        self._accepting = False
        self._running = False
        self._stop_requested = False
        self._draining = False
        self._drain_cutoff: datetime | None = None
        self._report_day = report_date(self._clock())
        self._acked_events: list[CountEvent] = []
        self._all_buckets: list[HealthBucket] = []
        self._frame_times: deque[datetime] = deque(maxlen=FPS_WINDOW)
        self._frame_count = 0
        self._last_snapshot_at: datetime | None = None
        self._last_csv_minute: datetime | None = None
        self._needs_new_session = False
        self._lock = threading.RLock()

    @property
    def tracking_session_id(self) -> int | None:
        with self._lock:
            return self._tracking_session_id

    @property
    def report_day(self) -> date:
        with self._lock:
            return self._report_day

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
        now = self._clock()
        day = report_date(now)
        app_result = self._worker.submit(StartAppSession(now))
        if not app_result.ok or app_result.value is None:
            with self._lock:
                self._report_day = day
                self._state = AppState.ERROR
                self._stream_state = StreamState.DISCONNECTED
                self._accepting = False
                self._running = False
                self._incomplete = True
                self._last_error = app_result.error or "session failed"
            self._publish_snapshot(force=True)
            return
        track_result = self._worker.submit(
            StartTrackingSession(app_result.value, now)
        )
        if not track_result.ok or track_result.value is None:
            with self._lock:
                self._report_day = day
                self._app_session_id = app_result.value
                self._state = AppState.ERROR
                self._stream_state = StreamState.DISCONNECTED
                self._accepting = False
                self._running = False
                self._incomplete = True
                self._last_error = track_result.error or "session failed"
            self._publish_snapshot(force=True)
            return
        try:
            fresh = self._factory.create(self._config.model)
        except Exception as exc:
            with self._lock:
                self._report_day = day
                self._app_session_id = app_result.value
                self._state = AppState.ERROR
                self._accepting = False
                self._running = False
                self._incomplete = True
                self._last_error = str(exc)[:ERROR_PREVIEW] or "detector failed"
            self._publish_snapshot(force=True)
            return
        try:
            recovered_events = self._store.events(day)
            recovered_list = list(recovered_events)
        except Exception:
            recovered_list = []
        try:
            recovered_buckets = self._store.health_buckets()
            recovered_bucket_list = list(recovered_buckets)
        except Exception:
            recovered_bucket_list = []
        with self._lock:
            self._report_day = day
            self._app_session_id = app_result.value
            self._tracking_session_id = track_result.value
            self._detector = fresh
            self._tracking_session_changes = 0
            self._dropped_events = 0
            self._csv_export_ok = False
            self._incomplete = False
            self._last_error = None
            self._state = AppState.RUNNING
            self._stream_state = StreamState.CONNECTED
            self._accepting = True
            self._running = True
            self._stop_requested = False
            self._draining = False
            self._drain_cutoff = None
            self._needs_new_session = False
            self._acked_events = recovered_list
            self._all_buckets = recovered_bucket_list
            self._frame_times.clear()
            self._frame_count = 0
            self._last_snapshot_at = None
            self._last_csv_minute = None
        self._publish_snapshot(force=True)

    def process_one(self, frame: Frame) -> None:
        with self._lock:
            if not self._running or self._stop_requested:
                return
            if self._state.value == AppState.ERROR.value:
                return
            if not self._accepting and not self._draining:
                return
            needs = (
                self._needs_new_session
                or self._detector is None
                or self._tracking_session_id is None
            )
        if needs:
            created = self._create_session()
            if not created:
                return
        with self._lock:
            if not self._running or self._stop_requested:
                return
            if self._state.value == AppState.ERROR.value:
                return
            raw_detector = self._detector
            raw_session = self._tracking_session_id
            draining = self._draining
            cutoff = self._drain_cutoff
            stream = self._stream_state
            if raw_detector is None or raw_session is None:
                return
            session_id: int = raw_session
        vehicles: list[TrackedVehicle] = raw_detector.detect(frame)
        with self._lock:
            if not self._running or self._stop_requested:
                return
            if self._state.value == AppState.ERROR.value:
                return
            if session_id != self._tracking_session_id:
                return
            if raw_detector is not self._detector:
                return
            pending_events: list[CountEvent] = []
            for detected in vehicles:
                event = self._counter.observe(session_id, detected, frame.observed_at)
                if event is None:
                    continue
                if draining and cutoff is not None and event.crossed_at >= cutoff:
                    continue
                pending_events.append(event)
        results: list[StorageResult] = []
        for event in pending_events:
            results.append(self._worker.submit(InsertEvent(event)))
        with self._lock:
            if not self._running:
                return
            for event, result in zip(pending_events, results, strict=True):
                if result.ok and not result.duplicate:
                    self._acked_events.append(event)
                elif result.ok and result.duplicate:
                    continue
                else:
                    self._dropped_events += 1
                    self._incomplete = True
                    self._state = AppState.ERROR
                    self._accepting = False
                    self._last_error = result.error
                    break
            failed = self._state.value == AppState.ERROR.value
            if not failed and not draining and stream is StreamState.CONNECTED:
                with suppress(Exception):
                    self._health.mark_healthy(frame.observed_at)
            self._frame_times.append(self._clock())
            self._frame_count += 1
            publish_now = failed
        if publish_now:
            self._publish_snapshot(force=True)
            return
        self._maybe_publish(frame, vehicles)

    def run_until(self, end_at: datetime) -> None:
        while True:
            with self._lock:
                if self._stop_requested or not self._running:
                    return
                if self._clock() >= end_at:
                    return
            frame = self._slot.take()
            if frame is not None:
                self.process_one(frame)
            else:
                self._sleeper(SLOT_POLL_SECONDS)

    def request_stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            if self._state.value in (AppState.STOPPED.value, AppState.STOPPING.value):
                return
            self._stop_requested = True
            self._state = AppState.STOPPING
            track_id = self._tracking_session_id
            app_id = self._app_session_id
        self._flush_health()
        self._flush_exports(force=True)
        if track_id is not None:
            with self._lock:
                current = self._tracking_session_id
                needs_end = current == track_id
            if needs_end:
                with self._lock, suppress(Exception):
                    self._counter.end_tracking_session(track_id)
            self._submit_stop(StopTrackingSession(track_id, self._clock()))
        if app_id is not None:
            self._submit_stop(StopAppSession(app_id, self._clock(), str(AppState.STOPPED)))
        with self._lock:
            self._detector = None
            self._tracking_session_id = None
            self._state = AppState.STOPPED
            self._stream_state = StreamState.STOPPED
            self._accepting = False
            self._running = False
            self._draining = False
            self._drain_cutoff = None
            self._needs_new_session = False
        self._publish_snapshot(force=True)
        if self._owns_worker:
            with suppress(Exception):
                self._worker.submit(Stop())
            self._worker.close()

    def drain_for(self, seconds: float) -> None:
        with self._lock:
            if not self._running:
                return
            if self._state.value in (AppState.STOPPED.value, AppState.STOPPING.value):
                return
            day = self._report_day
            self._drain_cutoff = datetime(
                day.year, day.month, day.day, DRAIN_CUTOFF_HOUR, 0, tzinfo=JAKARTA
            )
            self._draining = True
            drain_end = self._clock() + timedelta(seconds=seconds)
            track_id = self._tracking_session_id
            app_id = self._app_session_id
        while True:
            with self._lock:
                if self._stop_requested and not self._draining:
                    break
                if self._clock() >= drain_end:
                    break
                draining_active = self._draining
            if not draining_active:
                break
            frame = self._slot.take()
            if frame is not None:
                self.process_one(frame)
            else:
                self._sleeper(SLOT_POLL_SECONDS)
        self._flush_health()
        self._flush_exports(force=True)
        with self._lock:
            final_track = self._tracking_session_id
            if final_track is None:
                final_track = track_id
        if final_track is not None:
            with self._lock, suppress(Exception):
                self._counter.end_tracking_session(final_track)
            self._submit_stop(StopTrackingSession(final_track, self._clock()))
        if app_id is not None:
            self._submit_stop(StopAppSession(app_id, self._clock(), str(AppState.STOPPED)))
        with self._lock:
            self._detector = None
            self._tracking_session_id = None
            self._draining = False
            self._drain_cutoff = None
            self._state = AppState.STOPPED
            self._stream_state = StreamState.STOPPED
            self._accepting = False
            self._running = False
            self._stop_requested = True
            self._needs_new_session = False
        self._publish_snapshot(force=True)
        if self._owns_worker:
            with suppress(Exception):
                self._worker.submit(Stop())
            self._worker.close()

    def _submit_stop(self, command: StorageCommand) -> None:
        try:
            result = self._worker.submit(command)
        except Exception as exc:
            with self._lock:
                if self._last_error is None:
                    self._last_error = str(exc)[:ERROR_PREVIEW] or "stop failed"
            return
        if result.ok:
            return
        try:
            retry = self._worker.submit(command)
        except Exception as exc:
            with self._lock:
                if self._last_error is None:
                    self._last_error = str(exc)[:ERROR_PREVIEW] or "stop failed"
            return
        if retry.ok:
            return
        with self._lock:
            if self._last_error is None:
                self._last_error = retry.error or "stop failed"

    def accepting_counts(self) -> bool:
        with self._lock:
            if not self._running or self._stop_requested:
                return False
            if self._state.value == AppState.ERROR.value:
                return False
            if self._state.value in (AppState.STOPPED.value, AppState.STOPPING.value):
                return False
            return self._accepting

    def on_disconnect(self, error: str) -> None:
        with self._lock:
            if not self._running:
                return
            if self._stop_requested:
                return
            if self._state.value in (AppState.STOPPED.value, AppState.STOPPING.value):
                return
            text = error[:ERROR_PREVIEW] if len(error) > ERROR_PREVIEW else error
            self._last_error = text or "stream disconnected"
            old_id = self._tracking_session_id
            if old_id is not None:
                with suppress(Exception):
                    self._counter.end_tracking_session(old_id)
            self._detector = None
            self._tracking_session_id = None
            self._needs_new_session = True
            if self._state.value != AppState.ERROR.value:
                self._state = AppState.DEGRADED
            self._stream_state = StreamState.DISCONNECTED
        self._publish_snapshot(force=True)
        if old_id is not None:
            with suppress(Exception):
                self._worker.submit(StopTrackingSession(old_id, self._clock()))

    def reset_tracking_session(self) -> None:
        with self._lock:
            if not self._running:
                return
            if self._stop_requested:
                return
            if self._state.value in (AppState.STOPPED.value, AppState.STOPPING.value):
                return
            old_id = self._tracking_session_id
            if old_id is None:
                return
            with suppress(Exception):
                self._counter.end_tracking_session(old_id)
            self._detector = None
            self._tracking_session_id = None
            self._needs_new_session = True
        if old_id is not None:
            with suppress(Exception):
                self._worker.submit(StopTrackingSession(old_id, self._clock()))

    def _create_session(self) -> bool:
        with self._lock:
            if not self._running or self._stop_requested:
                return False
            if self._state.value in (AppState.STOPPED.value, AppState.STOPPING.value):
                return False
            app_id = self._app_session_id
        try:
            fresh = self._factory.create(self._config.model)
        except Exception as exc:
            with self._lock:
                self._state = AppState.ERROR
                self._accepting = False
                self._incomplete = True
                self._last_error = str(exc)[:ERROR_PREVIEW] or "detector failed"
            self._publish_snapshot(force=True)
            return False
        if app_id is None:
            app_result = self._worker.submit(StartAppSession(self._clock()))
            if not app_result.ok or app_result.value is None:
                with self._lock:
                    self._state = AppState.ERROR
                    self._accepting = False
                    self._incomplete = True
                    self._last_error = app_result.error or "session failed"
                self._publish_snapshot(force=True)
                return False
            app_id = app_result.value
            with self._lock:
                self._app_session_id = app_id
        track_result = self._worker.submit(StartTrackingSession(app_id, self._clock()))
        if not track_result.ok or track_result.value is None:
            with self._lock:
                self._state = AppState.ERROR
                self._accepting = False
                self._incomplete = True
                self._last_error = track_result.error or "session failed"
            self._publish_snapshot(force=True)
            return False
        new_id = track_result.value
        with self._lock:
            stopped = self._state.value in (AppState.STOPPED.value, AppState.STOPPING.value)
            if not self._running or self._stop_requested or stopped:
                created_id = new_id
            else:
                self._detector = fresh
                self._tracking_session_id = new_id
                self._tracking_session_changes += 1
                self._needs_new_session = False
                self._stream_state = StreamState.CONNECTED
                if self._state.value == AppState.DEGRADED.value:
                    self._state = AppState.RUNNING
                return True
        with suppress(Exception):
            self._worker.submit(StopTrackingSession(created_id, self._clock()))
        return False

    def _compute_fps(self) -> float:
        with self._lock:
            times = tuple(self._frame_times)
        count = len(times)
        if count < 2:
            return 0.0
        span = (times[count - 1] - times[0]).total_seconds()
        if span <= 0.0:
            return 0.0
        return (count - 1) / span

    def _build_snapshot(self, now: datetime) -> AppSnapshot:
        with self._lock:
            acked = list(self._acked_events)
            buckets = list(self._all_buckets)
            state = self._state
            stream = self._stream_state
            csv_ok = self._csv_export_ok
            dropped = self._dropped_events
            incomplete = self._incomplete
            last_error = self._last_error
            changes = self._tracking_session_changes
            target = self._config.timing.target_fps
            day = self._report_day
        try:
            minutes = tuple(self._store.timeseries(day))
        except Exception:
            minutes = ()
        try:
            stats = self._slot.stats()
            replaced = int(stats.replaced)
        except Exception:
            replaced = 0
        actual = self._compute_fps()
        summaries = build_period_summaries(
            acked, buckets, now, self._config.timing.rolling_min_healthy_seconds
        )
        mapping = {summary.period: summary for summary in summaries}
        return AppSnapshot(
            generated_at=now,
            state=state,
            stream_state=stream,
            actual_fps=actual,
            target_fps=target,
            replaced_frames=replaced,
            periods=MappingProxyType(dict(mapping)),
            minutes=minutes,
            csv_export_ok=csv_ok,
            dropped_events=dropped,
            incomplete=incomplete,
            last_error=last_error,
            tracking_session_changes=changes,
        )

    def _publish_snapshot(self, force: bool = False) -> None:
        now = self._clock()
        with self._lock:
            last = self._last_snapshot_at
            if not force and last is not None and now - last < SNAPSHOT_MIN_INTERVAL:
                return
            self._last_snapshot_at = now
        snapshot = self._build_snapshot(now)
        with suppress(Exception):
            self._snapshots.publish(snapshot)

    def _maybe_publish(self, frame: Frame, vehicles: list[TrackedVehicle]) -> None:
        now = self._clock()
        with self._lock:
            last = self._last_snapshot_at
            if last is not None and now - last < SNAPSHOT_MIN_INTERVAL:
                return
            self._last_snapshot_at = now
        self._flush_health()
        self._flush_exports(force=False)
        snapshot = self._build_snapshot(now)
        with suppress(Exception):
            self._snapshots.publish(snapshot)
        overlay = self._overlaid
        if overlay is not None:
            with suppress(Exception):
                overlay(frame, list(vehicles), snapshot)

    def _flush_health(self) -> None:
        with self._lock:
            try:
                pending = self._health.drain(self._config.timing.health_bucket_seconds)
            except Exception:
                return
            if not pending:
                return
        result = self._worker.submit(WriteHealthBuckets(pending))
        if result.ok:
            with self._lock:
                self._all_buckets.extend(pending)
        else:
            with self._lock:
                if self._last_error is None:
                    self._last_error = result.error

    def _flush_exports(self, force: bool) -> None:
        now = self._clock()
        minute = now.replace(second=0, microsecond=0)
        with self._lock:
            last = self._last_csv_minute
            if not force and last is not None and minute == last:
                return
            self._last_csv_minute = minute
            day = self._report_day
        rollup = self._worker.submit(RefreshRollups(day, now))
        if not rollup.ok:
            with self._lock:
                if self._last_error is None:
                    self._last_error = rollup.error
        export = self._worker.submit(ExportCsv(day, now))
        with self._lock:
            if export.ok:
                self._csv_export_ok = True
            else:
                self._csv_export_ok = False
                if self._last_error is None:
                    self._last_error = export.error
