from __future__ import annotations

import sqlite3
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import pytest

import traffic_counter.pipeline as pipeline_module
from traffic_counter.config import (
    AppConfig,
    LineConfig,
    ModelConfig,
    OutputConfig,
    StreamConfig,
    TimingConfig,
)
from traffic_counter.counter import LineCounter
from traffic_counter.models import (
    AppSnapshot,
    AppState,
    CountEvent,
    Frame,
    HealthBucket,
    MinutePoint,
    PeriodName,
    TrackedVehicle,
)
from traffic_counter.pipeline import (
    ExportCsv,
    InsertEvent,
    Purge,
    RefreshRollups,
    StartAppSession,
    StartTrackingSession,
    Stop,
    StopAppSession,
    StopTrackingSession,
    StorageResult,
    StorageWorker,
    TrafficPipeline,
    WriteHealthBuckets,
)
from traffic_counter.timeutils import JAKARTA

DAY: Final[date] = date(2026, 9, 25)
START: Final[datetime] = datetime(2026, 9, 25, 7, 0, tzinfo=JAKARTA)


def at(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, 9, 25, hour, minute, second, tzinfo=JAKARTA)


def make_config() -> AppConfig:
    return AppConfig(
        stream=StreamConfig(url="https://example.test/index.m3u8", width=12, height=8),
        model=ModelConfig(
            name="yolo26n.pt",
            device="cpu",
            confidence=0.25,
            iou=0.70,
            max_detections=100,
            image_size=640,
            tracker="bytetrack.yaml",
            persist=True,
        ),
        line=LineConfig(x1=0.10, y1=0.80, x2=0.90, y2=0.80, hysteresis=0.005),
        timing=TimingConfig(
            target_fps=10,
            reconnect_initial_seconds=1.0,
            reconnect_max_seconds=30.0,
            drain_seconds=5.0,
            health_bucket_seconds=10,
            rolling_min_healthy_seconds=60,
        ),
        output=OutputConfig(
            port=5000,
            database=Path("data/traffic.db"),
            export_directory=Path("data/exports"),
            calibration_directory=Path("data/calibration"),
            log_file=Path("data/traffic-counter.log"),
        ),
    )


def blank_frame(marker: int = 0, moment: datetime = START) -> Frame:
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    image[0, 0, 0] = marker
    return Frame(image=image, observed_at=moment)


def vehicle(track_id: int = 7, bottom_y: float = 0.90) -> TrackedVehicle:
    return TrackedVehicle(
        track_id=track_id,
        class_name="car",
        confidence=0.9,
        box_xyxy=(0.45, bottom_y - 0.10, 0.55, bottom_y),
    )


def crossing_event(session_id: int, track_id: int, moment: datetime) -> CountEvent:
    return CountEvent(
        tracking_session_id=session_id,
        track_id=track_id,
        crossed_at=moment,
        direction="up",
        period=PeriodName.PAGI,
        class_name="car",
    )


class FakeClock:
    def __init__(self, start: datetime = START) -> None:
        self.current = start

    def __call__(self) -> datetime:
        return self.current

    def set(self, moment: datetime) -> None:
        self.current = moment

    def advance(self, seconds: float) -> None:
        self.current = self.current + timedelta(seconds=seconds)


class FakeSleeper:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self.clock.advance(seconds)


class FakeStore:
    def __init__(self) -> None:
        self.seen: set[tuple[int, int]] = set()
        self.inserted: list[CountEvent] = []
        self.insert_calls = 0
        self.raise_on_insert: sqlite3.Error | None = None
        self.health_calls: list[tuple[HealthBucket, ...]] = []
        self.raise_on_health: sqlite3.Error | None = None
        self.rollup_calls: list[tuple[date, datetime]] = []
        self.raise_on_rollup: sqlite3.Error | None = None
        self.app_sessions = 0
        self.tracking_sessions = 0
        self.stopped_tracking: list[int] = []
        self.stopped_apps: list[int] = []
        self.purge_calls: list[date] = []
        self.raise_on_start_app: sqlite3.Error | None = None
        self.raise_on_start_tracking: sqlite3.Error | None = None
        self.raise_on_purge: sqlite3.Error | None = None
        self.order: list[str] = []
        self.session_threads: list[int] = []
        self.delay_insert = 0.0

    def start_app_session(self, started_at: datetime) -> int:
        self.order.append("start_app")
        self.session_threads.append(threading.get_ident())
        if self.raise_on_start_app is not None:
            raise self.raise_on_start_app
        self.app_sessions += 1
        return self.app_sessions

    def start_tracking_session(self, app_session_id: int, started_at: datetime) -> int:
        self.order.append("start_tracking")
        self.session_threads.append(threading.get_ident())
        if self.raise_on_start_tracking is not None:
            raise self.raise_on_start_tracking
        self.tracking_sessions += 1
        return self.tracking_sessions

    def stop_tracking_session(self, tracking_session_id: int, stopped_at: datetime) -> None:
        self.order.append("stop_tracking")
        self.session_threads.append(threading.get_ident())
        self.stopped_tracking.append(tracking_session_id)

    def stop_app_session(
        self, app_session_id: int, stopped_at: datetime, state: AppState | str
    ) -> None:
        self.order.append("stop_app")
        self.session_threads.append(threading.get_ident())
        self.stopped_apps.append(app_session_id)
        return None

    def insert_event(self, event: CountEvent) -> bool:
        self.insert_calls += 1
        self.order.append("insert")
        if self.delay_insert > 0.0:
            time.sleep(self.delay_insert)
        if self.raise_on_insert is not None:
            raise self.raise_on_insert
        key = (event.tracking_session_id, event.track_id)
        if key in self.seen:
            return False
        self.seen.add(key)
        self.inserted.append(event)
        return True

    def write_health_buckets(self, buckets: Iterable[HealthBucket]) -> None:
        items = tuple(buckets)
        self.order.append("health")
        self.health_calls.append(items)
        if self.raise_on_health is not None:
            raise self.raise_on_health

    def refresh_minute_rollups(self, report_day: date, now: datetime) -> None:
        self.order.append("rollup")
        self.rollup_calls.append((report_day, now))
        if self.raise_on_rollup is not None:
            raise self.raise_on_rollup

    def timeseries(self, report_day: date) -> tuple[MinutePoint, ...]:
        return ()

    def events(self, day: date, period: PeriodName | None = None) -> tuple[CountEvent, ...]:
        return tuple(self.inserted)

    def health_buckets(self, period: PeriodName | None = None) -> tuple[HealthBucket, ...]:
        merged: list[HealthBucket] = []
        for items in self.health_calls:
            merged.extend(items)
        return tuple(merged)

    def purge_before(self, report_day: date) -> None:
        self.order.append("purge")
        self.purge_calls.append(report_day)
        if self.raise_on_purge is not None:
            raise self.raise_on_purge


class FakeExporter:
    def __init__(self) -> None:
        self.calls: list[tuple[date, datetime]] = []
        self.ok = True
        self.raise_error: OSError | None = None
        self.purge_calls: list[date] = []

    def try_export(self, report_day: date, now: datetime) -> bool:
        self.calls.append((report_day, now))
        if self.raise_error is not None:
            raise self.raise_error
        return self.ok

    def export(self, report_day: date, now: datetime) -> Path:
        self.calls.append((report_day, now))
        if self.raise_error is not None:
            raise self.raise_error
        if not self.ok:
            raise OSError("export failed")
        return Path("data/exports") / "traffic_2026-09-25.csv"

    def purge_before(self, report_day: date) -> None:
        self.purge_calls.append(report_day)
        if self.raise_error is not None:
            raise self.raise_error


class FakeDetector:
    def __init__(self, handler: Callable[[Frame], list[TrackedVehicle]]) -> None:
        self.handler = handler
        self.calls: list[Frame] = []

    def detect(self, frame: Frame) -> list[TrackedVehicle]:
        self.calls.append(frame)
        return self.handler(frame)


class FakeFactory:
    def __init__(self, handler: Callable[[Frame], list[TrackedVehicle]]) -> None:
        self.handler = handler
        self.created: list[FakeDetector] = []

    def create(self, config: ModelConfig) -> FakeDetector:
        detector = FakeDetector(self.handler)
        self.created.append(detector)
        return detector


class FakeSlot:
    def __init__(self) -> None:
        self.pending: Frame | None = None
        self.received = 0
        self.replaced = 0

    def put(self, frame: Frame) -> None:
        if self.pending is not None:
            self.replaced += 1
        self.pending = frame
        self.received += 1

    def take(self) -> Frame | None:
        frame = self.pending
        self.pending = None
        return frame

    def stats(self) -> object:
        from traffic_counter.models import FrameSlotStats

        return FrameSlotStats(received=self.received, replaced=self.replaced)


class FakeSnapshots:
    def __init__(self, now: datetime = START) -> None:
        self.current: AppSnapshot = AppSnapshot.initial(now)
        self.publishes = 0

    def get(self) -> AppSnapshot:
        return self.current

    def publish(self, snapshot: AppSnapshot) -> None:
        self.current = snapshot
        self.publishes += 1


class FakeCounter:
    def __init__(self) -> None:
        self.observe_calls: list[tuple[int, TrackedVehicle, datetime]] = []
        self.end_calls: list[int] = []
        self.queue: deque[CountEvent | None] = deque()

    def observe(
        self,
        tracking_session_id: int,
        detected: TrackedVehicle,
        observed_at: datetime,
    ) -> CountEvent | None:
        self.observe_calls.append((tracking_session_id, detected, observed_at))
        if self.queue:
            return self.queue.popleft()
        return None

    def end_tracking_session(self, tracking_session_id: int) -> None:
        self.end_calls.append(tracking_session_id)


class SpyCounter:
    def __init__(self, line: LineConfig) -> None:
        self.inner = LineCounter(line)
        self.observe_calls: list[tuple[int, int, datetime]] = []
        self.end_calls: list[int] = []

    def observe(
        self,
        tracking_session_id: int,
        detected: TrackedVehicle,
        observed_at: datetime,
    ) -> CountEvent | None:
        self.observe_calls.append((tracking_session_id, detected.track_id, observed_at))
        return self.inner.observe(tracking_session_id, detected, observed_at)

    def end_tracking_session(self, tracking_session_id: int) -> None:
        self.end_calls.append(tracking_session_id)
        self.inner.end_tracking_session(tracking_session_id)


class FakeOverlay:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, frame: Frame, vehicles: list[TrackedVehicle], snapshot: AppSnapshot) -> None:
        self.calls += 1


def build_pipeline(
    handler: Callable[[Frame], list[TrackedVehicle]],
    counter: object,
    clock: FakeClock,
    store: FakeStore | None = None,
    exporter: FakeExporter | None = None,
    snapshots: FakeSnapshots | None = None,
    slot: FakeSlot | None = None,
    sleeper: FakeSleeper | None = None,
    overlay: FakeOverlay | None = None,
) -> tuple[TrafficPipeline, FakeStore, FakeExporter, FakeSnapshots, FakeSlot, FakeFactory]:
    owned_store = store if store is not None else FakeStore()
    owned_exporter = exporter if exporter is not None else FakeExporter()
    owned_snapshots = snapshots if snapshots is not None else FakeSnapshots(clock.current)
    owned_slot = slot if slot is not None else FakeSlot()
    factory = FakeFactory(handler)
    pipeline = TrafficPipeline(
        config=make_config(),
        store=owned_store,
        exporter=owned_exporter,
        snapshots=owned_snapshots,
        slot=owned_slot,
        counter=counter,
        detector_factory=factory,
        clock=clock,
        sleeper=sleeper,
        overlay=overlay,
    )
    return pipeline, owned_store, owned_exporter, owned_snapshots, owned_slot, factory


def test_storage_worker_executes_fifo_and_signals_completion() -> None:
    store = FakeStore()
    exporter = FakeExporter()
    worker = StorageWorker(store, exporter)
    try:
        first = crossing_event(1, 1, at(7, 0))
        second = crossing_event(1, 2, at(7, 1))
        assert worker.submit(InsertEvent(first)).ok is True
        assert worker.submit(InsertEvent(second)).ok is True
        assert [item.track_id for item in store.inserted] == [1, 2]
        buckets: tuple[HealthBucket, ...] = ()
        assert worker.submit(WriteHealthBuckets(buckets)).ok is True
        assert worker.submit(RefreshRollups(DAY, at(7, 2))).ok is True
        assert worker.submit(ExportCsv(DAY, at(7, 3))).ok is True
        assert worker.submit(Purge(DAY)).ok is True
        assert exporter.calls
        assert store.rollup_calls
    finally:
        worker.submit(Stop())
        worker.close()


def test_storage_worker_insert_catches_sqlite_error() -> None:
    store = FakeStore()
    store.raise_on_insert = sqlite3.IntegrityError("foreign key")
    exporter = FakeExporter()
    worker = StorageWorker(store, exporter)
    try:
        result = worker.submit(InsertEvent(crossing_event(9, 9, at(7, 0))))
        assert result.ok is False
        assert result.duplicate is False
        assert result.error is not None
    finally:
        worker.submit(Stop())
        worker.close()


def test_storage_worker_export_catches_oserror() -> None:
    store = FakeStore()
    exporter = FakeExporter()
    exporter.raise_error = OSError("disk locked")
    worker = StorageWorker(store, exporter)
    try:
        result = worker.submit(ExportCsv(DAY, at(7, 0)))
        assert result.ok is False
        assert result.error is not None
    finally:
        worker.submit(Stop())
        worker.close()


def test_snapshot_total_published_only_after_storage_ack() -> None:
    clock = FakeClock(at(7, 0))
    counter = FakeCounter()
    counter.queue.append(crossing_event(1, 7, at(7, 0)))
    snapshots = FakeSnapshots(at(7, 0))
    pipeline, store, _, owned, _, _ = build_pipeline(
        lambda frame: [vehicle(7)], counter, clock, snapshots=snapshots
    )
    pipeline.start()
    assert owned.get().periods[PeriodName.PAGI].total == 0
    clock.set(at(7, 0, 1))
    pipeline.process_one(blank_frame(7, at(7, 0, 1)))
    assert store.inserted
    assert owned.get().periods[PeriodName.PAGI].total == 1
    pipeline.request_stop()


def test_duplicate_event_does_not_increment_dropped_nor_error() -> None:
    clock = FakeClock(at(7, 0))
    counter = FakeCounter()
    counter.queue.append(crossing_event(1, 7, at(7, 0)))
    counter.queue.append(crossing_event(1, 7, at(7, 0, 1)))
    pipeline, store, _, snapshots, _, _ = build_pipeline(lambda frame: [vehicle(7)], counter, clock)
    pipeline.start()
    pipeline.process_one(blank_frame(1, at(7, 0, 1)))
    clock.advance(2.0)
    pipeline.process_one(blank_frame(1, at(7, 0, 2)))
    assert snapshots.get().dropped_events == 0
    assert snapshots.get().incomplete is False
    assert snapshots.get().state is not AppState.ERROR
    assert pipeline.accepting_counts() is True
    assert len(store.inserted) == 1
    pipeline.request_stop()


def test_database_failure_marks_incomplete_and_stops_accepting() -> None:
    clock = FakeClock(at(7, 0))
    counter = FakeCounter()
    counter.queue.append(crossing_event(1, 7, at(7, 0)))
    store = FakeStore()
    store.raise_on_insert = sqlite3.OperationalError("disk failure")
    pipeline, _, _, snapshots, _, _ = build_pipeline(
        lambda frame: [vehicle(7)], counter, clock, store=store
    )
    pipeline.start()
    pipeline.process_one(blank_frame(1, at(7, 0, 1)))
    assert snapshots.get().state is AppState.ERROR
    assert snapshots.get().dropped_events == 1
    assert snapshots.get().incomplete is True
    assert pipeline.accepting_counts() is False
    pipeline.request_stop()


def test_csv_failure_keeps_counting_and_running() -> None:
    clock = FakeClock(at(7, 0))
    counter = FakeCounter()
    counter.queue.append(crossing_event(1, 7, at(7, 0)))
    counter.queue.append(crossing_event(1, 8, at(7, 1)))
    exporter = FakeExporter()
    exporter.ok = False
    pipeline, store, _, snapshots, _, _ = build_pipeline(
        lambda frame: [vehicle(7)], counter, clock, exporter=exporter
    )
    pipeline.start()
    pipeline.process_one(blank_frame(1, at(7, 0, 1)))
    clock.advance(70.0)
    pipeline.process_one(blank_frame(1, at(7, 1, 5)))
    assert snapshots.get().csv_export_ok is False
    assert snapshots.get().state is AppState.RUNNING
    assert pipeline.accepting_counts() is True
    assert len(store.inserted) == 2
    pipeline.request_stop()


def test_disconnect_resets_session_before_replacement_observation() -> None:
    clock = FakeClock(at(7, 0))
    line = LineConfig(x1=0.0, y1=0.5, x2=1.0, y2=0.5, hysteresis=0.005)
    spy = SpyCounter(line)

    def handler(frame: Frame) -> list[TrackedVehicle]:
        marker = int(frame.image[0, 0, 0])
        if marker == 1:
            return [vehicle(7, bottom_y=0.4375)]
        return [vehicle(7, bottom_y=0.5625)]

    pipeline, store, _, snapshots, _, _ = build_pipeline(handler, spy, clock)
    pipeline.start()
    first_id = pipeline.tracking_session_id
    pipeline.process_one(blank_frame(1, at(7, 0, 1)))
    assert spy.end_calls == []
    pipeline.on_disconnect("link down")
    assert first_id is not None
    assert spy.end_calls == [first_id]
    prior_ends = list(spy.end_calls)
    pipeline.process_one(blank_frame(2, at(7, 0, 2)))
    assert snapshots.get().periods[PeriodName.PAGI].total == 0
    assert store.inserted == []
    assert len(spy.observe_calls) == 2
    assert spy.observe_calls[0][0] != spy.observe_calls[1][0]
    assert spy.end_calls == prior_ends
    pipeline.request_stop()


def test_tracker_reset_establishes_fresh_baseline() -> None:
    clock = FakeClock(at(7, 0))
    line = LineConfig(x1=0.0, y1=0.5, x2=1.0, y2=0.5, hysteresis=0.005)
    spy = SpyCounter(line)

    def handler(frame: Frame) -> list[TrackedVehicle]:
        marker = int(frame.image[0, 0, 0])
        if marker == 1:
            return [vehicle(7, bottom_y=0.4375)]
        return [vehicle(7, bottom_y=0.5625)]

    pipeline, store, _, snapshots, _, _ = build_pipeline(handler, spy, clock)
    pipeline.start()
    first_id = pipeline.tracking_session_id
    pipeline.process_one(blank_frame(1, at(7, 0, 1)))
    pipeline.reset_tracking_session()
    assert first_id is not None
    assert spy.end_calls == [first_id]
    pipeline.process_one(blank_frame(2, at(7, 0, 2)))
    assert snapshots.get().periods[PeriodName.PAGI].total == 0
    assert store.inserted == []
    assert spy.observe_calls[0][0] != spy.observe_calls[1][0]
    pipeline.request_stop()


def test_snapshot_publication_capped_at_1hz_with_independent_fps() -> None:
    clock = FakeClock(at(7, 0))
    counter = FakeCounter()
    snapshots = FakeSnapshots(at(7, 0))
    slot = FakeSlot()
    pipeline, _, _, owned, _, _ = build_pipeline(
        lambda frame: [], counter, clock, snapshots=snapshots, slot=slot
    )
    pipeline.start()
    base = owned.publishes
    moment = at(7, 0)
    for index in range(10):
        moment = moment + timedelta(milliseconds=100)
        clock.set(moment)
        pipeline.process_one(blank_frame(index, moment))
    assert owned.publishes - base <= 2
    assert owned.get().actual_fps > 5.0
    assert owned.get().replaced_frames == slot.replaced
    pipeline.request_stop()


def test_out_of_schedule_suppressed() -> None:
    clock = FakeClock(at(5, 0))
    line = LineConfig(x1=0.0, y1=0.5, x2=1.0, y2=0.5, hysteresis=0.005)
    counter = LineCounter(line)
    pipeline, store, _, snapshots, _, _ = build_pipeline(
        lambda frame: [vehicle(7, bottom_y=0.5625)], counter, clock
    )
    pipeline.start()
    pipeline.process_one(blank_frame(1, at(5, 0, 1)))
    clock.set(at(5, 0, 2))
    pipeline.process_one(blank_frame(1, at(5, 0, 2)))
    assert snapshots.get().periods[PeriodName.PAGI].total == 0
    assert store.inserted == []
    pipeline.request_stop()


def test_drain_admits_only_pre_1900_and_ends_session() -> None:
    clock = FakeClock(at(18, 59, 50))
    counter = FakeCounter()
    slot = FakeSlot()
    snapshots = FakeSnapshots(at(18, 59, 50))
    sleeper = FakeSleeper(clock)
    pipeline, store, exporter, owned, _, _ = build_pipeline(
        lambda frame: [vehicle(7)],
        counter,
        clock,
        snapshots=snapshots,
        slot=slot,
        sleeper=sleeper,
    )
    pipeline.start()
    counter.queue.clear()
    counter.queue.append(crossing_event(1, 7, at(18, 59, 59)))
    counter.queue.append(crossing_event(1, 8, at(19, 0, 0)))
    slot.put(blank_frame(1, at(18, 59, 59)))
    slot.put(blank_frame(2, at(19, 0, 1)))
    clock.set(at(19, 0, 0))
    pipeline.drain_for(2.0)
    assert len(store.inserted) == 1
    assert store.inserted[0].track_id == 7
    assert counter.end_calls or store.stopped_tracking
    assert exporter.calls
    assert owned.get().state is AppState.STOPPED
    assert pipeline.accepting_counts() is False


def test_graceful_stop_stops_accepting_and_flushes() -> None:
    clock = FakeClock(at(7, 0))
    counter = FakeCounter()
    exporter = FakeExporter()
    store = FakeStore()
    pipeline, _, _, snapshots, _, _ = build_pipeline(
        lambda frame: [], counter, clock, store=store, exporter=exporter
    )
    pipeline.start()
    assert pipeline.accepting_counts() is True
    pipeline.process_one(blank_frame(0, at(7, 0, 1)))
    pipeline.request_stop()
    assert pipeline.accepting_counts() is False
    assert snapshots.get().state is AppState.STOPPED
    assert exporter.calls
    assert store.health_calls or store.rollup_calls


def test_run_until_processes_frames_until_end() -> None:
    clock = FakeClock(at(7, 0))
    counter = FakeCounter()
    slot = FakeSlot()
    slot.put(blank_frame(3, at(7, 0, 1)))
    sleeper = FakeSleeper(clock)
    pipeline, _, _, snapshots, _, _ = build_pipeline(
        lambda frame: [], counter, clock, slot=slot, sleeper=sleeper
    )
    pipeline.start()
    pipeline.run_until(at(7, 0, 2))
    assert snapshots.get().state is AppState.RUNNING
    pipeline.request_stop()


def test_submit_after_stop_is_rejected() -> None:
    store = FakeStore()
    exporter = FakeExporter()
    worker = StorageWorker(store, exporter)
    first = worker.submit(StartAppSession(at(7, 0)))
    assert first.ok is True
    assert first.value == 1
    worker.submit(Stop())
    worker.close()
    rejected = worker.submit(InsertEvent(crossing_event(1, 1, at(7, 0))))
    assert rejected.ok is False
    assert rejected.error is not None


def test_submit_ack_timeout_returns_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_module, "ACK_TIMEOUT_SECONDS", 0.05)
    store = FakeStore()
    store.delay_insert = 0.5
    exporter = FakeExporter()
    worker = StorageWorker(store, exporter)
    try:
        result = worker.submit(InsertEvent(crossing_event(1, 1, at(7, 0))))
        assert result.ok is False
        assert result.error is not None
    finally:
        store.delay_insert = 0.0
        worker.close()


def test_session_event_rollup_fifo_order() -> None:
    store = FakeStore()
    exporter = FakeExporter()
    worker = StorageWorker(store, exporter)
    try:
        app = worker.submit(StartAppSession(at(7, 0)))
        assert app.value == 1
        track = worker.submit(StartTrackingSession(1, at(7, 0)))
        assert track.value == 1
        assert worker.submit(InsertEvent(crossing_event(1, 1, at(7, 0)))).ok is True
        assert worker.submit(WriteHealthBuckets(())).ok is True
        assert worker.submit(RefreshRollups(DAY, at(7, 1))).ok is True
        assert worker.submit(ExportCsv(DAY, at(7, 1))).ok is True
        assert worker.submit(StopTrackingSession(1, at(7, 2))).ok is True
        assert worker.submit(StopAppSession(1, at(7, 2), str(AppState.STOPPED))).ok is True
        assert store.order == [
            "start_app",
            "start_tracking",
            "insert",
            "health",
            "rollup",
            "stop_tracking",
            "stop_app",
        ]
        assert exporter.calls
    finally:
        worker.submit(Stop())
        worker.close()


def test_pipeline_session_writes_use_worker_thread() -> None:
    clock = FakeClock(at(7, 0))
    counter = FakeCounter()
    store = FakeStore()
    main_id = threading.get_ident()
    pipeline, _, _, _, _, _ = build_pipeline(lambda frame: [], counter, clock, store=store)
    pipeline.start()
    assert store.session_threads
    assert all(tid != main_id for tid in store.session_threads)
    pipeline.request_stop()
    assert store.stopped_tracking
    assert store.stopped_apps


def test_concurrent_process_disconnect_stop() -> None:
    clock = FakeClock(at(7, 0))
    counter = FakeCounter()
    pipeline, _, _, snapshots, _, _ = build_pipeline(lambda frame: [vehicle(7)], counter, clock)
    pipeline.start()
    errors: list[BaseException] = []

    def run_process() -> None:
        try:
            for idx in range(20):
                pipeline.process_one(blank_frame(idx % 256, at(7, 0, 1)))
        except Exception as exc:
            errors.append(exc)

    def run_disconnect() -> None:
        try:
            for _ in range(5):
                pipeline.on_disconnect("flap")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run_process) for _ in range(3)]
    threads.append(threading.Thread(target=run_disconnect))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    pipeline.request_stop()
    assert snapshots.get().state is AppState.STOPPED


def test_disconnect_not_starved_by_storage_latency() -> None:
    clock = FakeClock(at(7, 0))
    counter = FakeCounter()
    counter.queue.append(crossing_event(1, 7, at(7, 0)))
    store = FakeStore()
    store.delay_insert = 0.8
    pipeline, _, _, snapshots, _, _ = build_pipeline(
        lambda frame: [vehicle(7)], counter, clock, store=store
    )
    pipeline.start()
    clock.set(at(7, 0, 1))

    def run_frame() -> None:
        pipeline.process_one(blank_frame(1, at(7, 0, 1)))

    thread = threading.Thread(target=run_frame)
    thread.start()
    time.sleep(0.15)
    probe_start = time.monotonic()
    assert pipeline.accepting_counts() is True
    assert time.monotonic() - probe_start < 0.5
    done: list[bool] = []

    def run_disconnect() -> None:
        pipeline.on_disconnect("stall")
        done.append(True)

    dthread = threading.Thread(target=run_disconnect)
    dthread.start()
    deadline = time.monotonic() + 2.0
    seen_degraded = False
    while time.monotonic() < deadline:
        if snapshots.get().state is AppState.DEGRADED:
            seen_degraded = True
            break
        time.sleep(0.02)
    assert seen_degraded is True
    thread.join(timeout=10.0)
    dthread.join(timeout=10.0)
    assert snapshots.get().state is not AppState.STOPPED
    pipeline.request_stop()


def test_start_tracking_failure_forces_error_snapshot() -> None:
    clock = FakeClock(at(7, 0))
    counter = FakeCounter()
    store = FakeStore()
    store.raise_on_start_tracking = sqlite3.OperationalError("no session")
    pipeline, _, _, snapshots, _, _ = build_pipeline(lambda frame: [], counter, clock, store=store)
    pipeline.start()
    assert snapshots.get().state is AppState.ERROR
    assert snapshots.get().incomplete is True
    assert snapshots.get().last_error is not None
    assert pipeline.accepting_counts() is False
    pipeline.request_stop()


def test_factory_exception_forces_error_snapshot() -> None:
    clock = FakeClock(at(7, 0))
    counter = FakeCounter()

    class BadFactory:
        def create(self, config: ModelConfig) -> object:
            raise RuntimeError("weights missing")

    pipeline = TrafficPipeline(
        config=make_config(),
        store=FakeStore(),
        exporter=FakeExporter(),
        snapshots=FakeSnapshots(at(7, 0)),
        slot=FakeSlot(),
        counter=counter,
        detector_factory=BadFactory(),
        clock=clock,
    )
    snapshots = pipeline._snapshots
    pipeline.start()
    assert snapshots.get().state is AppState.ERROR
    assert snapshots.get().incomplete is True
    assert pipeline.accepting_counts() is False
    pipeline.request_stop()


def test_overlay_invoked_with_copy() -> None:
    clock = FakeClock(at(7, 0))
    counter = FakeCounter()
    overlay = FakeOverlay()
    seen_lists: list[list[TrackedVehicle]] = []

    class CopyCheck(FakeOverlay):
        def __call__(
            self, frame: Frame, vehicles: list[TrackedVehicle], snapshot: AppSnapshot
        ) -> None:
            seen_lists.append(vehicles)
            vehicles.append(vehicle(99))

    pipeline, _, _, _, _, _ = build_pipeline(
        lambda frame: [vehicle(7)], counter, clock, overlay=CopyCheck()
    )
    assert overlay.calls == 0
    pipeline.start()
    clock.set(at(7, 0, 1))
    pipeline.process_one(blank_frame(1, at(7, 0, 1)))
    clock.set(at(7, 0, 2))
    pipeline.process_one(blank_frame(1, at(7, 0, 2)))
    assert seen_lists
    assert all(len(items) == 2 for items in seen_lists)
    pipeline.request_stop()


def test_health_and_rollup_failures_stay_non_fatal() -> None:
    clock = FakeClock(at(7, 0))
    counter = FakeCounter()
    counter.queue.append(crossing_event(1, 7, at(7, 0)))
    store = FakeStore()
    store.raise_on_health = sqlite3.OperationalError("health down")
    store.raise_on_rollup = sqlite3.OperationalError("rollup down")
    pipeline, _, _, snapshots, _, _ = build_pipeline(
        lambda frame: [vehicle(7)], counter, clock, store=store
    )
    pipeline.start()
    clock.set(at(7, 0, 1))
    pipeline.process_one(blank_frame(1, at(7, 0, 1)))
    clock.set(at(7, 0, 2))
    pipeline.process_one(blank_frame(1, at(7, 0, 2)))
    assert snapshots.get().state is AppState.RUNNING
    assert pipeline.accepting_counts() is True
    assert snapshots.get().dropped_events == 0
    assert snapshots.get().incomplete is False
    pipeline.request_stop()


def test_worker_multithread_fifo() -> None:
    store = FakeStore()
    exporter = FakeExporter()
    worker = StorageWorker(store, exporter)
    try:
        results: list[StorageResult] = []
        lock = threading.Lock()

        def submit_many(base: int) -> None:
            local: list[StorageResult] = []
            for idx in range(10):
                event = crossing_event(1, base * 100 + idx, at(7, 0))
                local.append(worker.submit(InsertEvent(event)))
            with lock:
                results.extend(local)

        threads = [threading.Thread(target=submit_many, args=(n,)) for n in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10.0)
        assert len(results) == 40
        assert all(item.ok for item in results)
        assert len(store.inserted) == 40
    finally:
        worker.submit(Stop())
        worker.close()


def test_purge_error_behavior() -> None:
    store = FakeStore()
    store.raise_on_purge = sqlite3.OperationalError("purge down")
    exporter = FakeExporter()
    worker = StorageWorker(store, exporter)
    try:
        failed = worker.submit(Purge(DAY))
        assert failed.ok is False
        assert failed.error is not None
        store2 = FakeStore()
        exporter2 = FakeExporter()
        exporter2.raise_error = OSError("exporter purge down")
        worker2 = StorageWorker(store2, exporter2)
        try:
            failed2 = worker2.submit(Purge(DAY))
            assert failed2.ok is False
        finally:
            worker2.submit(Stop())
            worker2.close()
    finally:
        worker.submit(Stop())
        worker.close()


def test_real_counter_drain_interpolation() -> None:
    clock = FakeClock(at(18, 59, 50))
    line = LineConfig(x1=0.0, y1=0.5, x2=1.0, y2=0.5, hysteresis=0.005)
    counter = LineCounter(line)

    class FifoSlot(FakeSlot):
        def __init__(self) -> None:
            super().__init__()
            self.items: deque[Frame] = deque()

        def put(self, frame: Frame) -> None:
            self.items.append(frame)
            self.received += 1

        def take(self) -> Frame | None:
            if not self.items:
                return None
            return self.items.popleft()

    slot = FifoSlot()
    snapshots = FakeSnapshots(at(18, 59, 50))
    sleeper = FakeSleeper(clock)

    def handler(frame: Frame) -> list[TrackedVehicle]:
        marker = int(frame.image[0, 0, 0])
        if marker == 1:
            return [vehicle(7, bottom_y=0.40)]
        if marker == 2:
            return [vehicle(7, bottom_y=0.60)]
        if marker == 3:
            return [vehicle(8, bottom_y=0.40)]
        return [vehicle(8, bottom_y=0.60)]

    pipeline, store, exporter, owned, _, _ = build_pipeline(
        handler, counter, clock, snapshots=snapshots, slot=slot, sleeper=sleeper
    )
    pipeline.start()
    early_first = datetime(2026, 9, 25, 18, 59, 58, tzinfo=JAKARTA)
    early_second = datetime(2026, 9, 25, 18, 59, 59, tzinfo=JAKARTA)
    late_first = datetime(2026, 9, 25, 18, 59, 59, tzinfo=JAKARTA)
    late_second = datetime(2026, 9, 25, 19, 0, 1, tzinfo=JAKARTA)
    clock.set(early_second)
    pipeline.process_one(blank_frame(1, early_first))
    pipeline.process_one(blank_frame(2, early_second))
    assert len(store.inserted) == 1
    assert store.inserted[0].track_id == 7
    slot.put(blank_frame(3, late_first))
    slot.put(blank_frame(4, late_second))
    clock.set(at(19, 0, 0))
    pipeline.drain_for(2.0)
    assert [event.track_id for event in store.inserted] == [7]
    assert owned.get().state is AppState.STOPPED
    assert exporter.calls


def test_report_day_and_stop_bounds() -> None:
    clock = FakeClock(at(7, 0))
    counter = FakeCounter()
    pipeline, _store, _, snapshots, _, _ = build_pipeline(lambda frame: [], counter, clock)
    pipeline.start()
    assert pipeline.report_day == DAY
    pipeline.request_stop()
    assert snapshots.get().state is AppState.STOPPED
    pipeline.on_disconnect("late flap")
    assert snapshots.get().state is AppState.STOPPED
    pipeline.reset_tracking_session()
    assert snapshots.get().state is AppState.STOPPED
    assert pipeline.accepting_counts() is False
