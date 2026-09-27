from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

JAKARTA = ZoneInfo("Asia/Jakarta")
DAY = date(2026, 9, 25)
PRIOR = date(2026, 9, 24)


def at(hour: int, minute: int = 0, second: int = 0, day: date = DAY) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=JAKARTA)


def make_config(tmp_path: Path | None = None):
    from traffic_counter.config import (
        AppConfig,
        LineConfig,
        ModelConfig,
        OutputConfig,
        StreamConfig,
        TimingConfig,
    )

    base = Path("data/traffic.db") if tmp_path is None else tmp_path / "traffic.db"
    exports = Path("data/exports") if tmp_path is None else tmp_path / "exports"
    calib = Path("data/calibration") if tmp_path is None else tmp_path / "calibration"
    log = Path("data/traffic-counter.log") if tmp_path is None else tmp_path / "t.log"
    return AppConfig(
        stream=StreamConfig(url="https://example.test/index.m3u8", width=12, height=8),
        model=ModelConfig(name="yolo26n.pt"),
        line=LineConfig(x1=0.10, y1=0.80, x2=0.90, y2=0.80, hysteresis=0.005),
        timing=TimingConfig(drain_seconds=5.0),
        output=OutputConfig(
            port=5000,
            database=base,
            export_directory=exports,
            calibration_directory=calib,
            log_file=log,
        ),
    )


class FakeClock:
    def __init__(self, start: datetime):
        self.current = start

    def __call__(self) -> datetime:
        return self.current

    def set(self, moment: datetime) -> None:
        self.current = moment


class FakePipeline:
    def __init__(self) -> None:
        self.start_calls = 0
        self.run_calls: list[datetime] = []
        self.fail_on_start = None

    def start(self) -> None:
        self.start_calls += 1
        if self.fail_on_start is not None:
            raise self.fail_on_start

    def run_until(self, end_at: datetime) -> None:
        self.run_calls.append(end_at)

    def drain_for(self, seconds: float) -> None:
        return None

    def request_stop(self) -> None:
        return None


class FakeSnapshots:
    def __init__(self, now: datetime):
        from traffic_counter.models import AppSnapshot

        self.current = AppSnapshot.initial(now)

    def get(self):
        return self.current

    def publish(self, snapshot) -> None:
        self.current = snapshot


class OrderStore:
    def __init__(self, order: list[str], fail_purge: bool = False):
        self.order = order
        self.fail_purge = fail_purge
        self.purge_calls = 0

    def purge_before(self, day: date) -> None:
        self.purge_calls += 1
        self.order.append("store")
        if self.fail_purge:
            raise OSError("purge locked")


class OrderExporter:
    def __init__(
        self,
        order: list[str],
        ok: bool = True,
        fail_purge: bool = False,
        raise_export: bool = False,
    ):
        self.order = order
        self.ok = ok
        self.fail_purge = fail_purge
        self.raise_export = raise_export
        self.purge_calls = 0
        self.export_calls = 0

    def purge_before(self, day: date) -> None:
        self.purge_calls += 1
        self.order.append("exporter")
        if self.fail_purge:
            raise OSError("exporter purge locked")

    def try_export(self, day: date, now: datetime) -> bool:
        self.export_calls += 1
        self.order.append("export")
        if self.raise_export:
            raise OSError("export locked")
        return self.ok


def test_c1_serve_dashboard_delegates_to_web_module(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    import traffic_counter.app as app_module

    called: list[tuple[object, object, object]] = []
    fake = types.ModuleType("traffic_counter.web")

    def fake_serve(a: object, b: object, c: object) -> None:
        called.append((a, b, c))

    fake.serve_dashboard = fake_serve  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "traffic_counter.web", fake)
    sentinel_store = object()
    sentinel_provider = object()
    cfg = make_config()
    app_module.serve_dashboard(sentinel_store, sentinel_provider, cfg)
    assert len(called) == 1
    assert called[0][0] is sentinel_store
    assert called[0][1] is sentinel_provider
    assert called[0][2] is cfg


def test_c1_run_normal_wiring_passes_real_dashboard() -> None:
    import traffic_counter.app as app_module
    import traffic_counter.cli as cli_module

    assert cli_module.serve_dashboard is app_module.serve_dashboard


def test_c3_purge_failure_is_fatal_and_blocks_retry() -> None:
    from traffic_counter.app import DailyController
    from traffic_counter.models import AppState

    clock = FakeClock(at(7, 0))
    pipeline = FakePipeline()
    snapshots = FakeSnapshots(clock())
    order: list[str] = []
    controller = DailyController(
        pipeline,
        config=make_config(),
        store=OrderStore(order, fail_purge=True),
        exporter=OrderExporter(order, ok=True),
        snapshots=snapshots,
        clock=clock,
        sleep=lambda s: None,
    )
    with pytest.raises(OSError):
        controller.recover()
    assert snapshots.get().state is AppState.ERROR
    clock2 = FakeClock(at(7, 0))
    snapshots2 = FakeSnapshots(clock2())
    store = OrderStore([], fail_purge=True)
    exporter = OrderExporter([], ok=True)
    ctl = DailyController(
        pipeline,
        config=make_config(),
        store=store,
        exporter=exporter,
        snapshots=snapshots2,
        clock=clock2,
        sleep=lambda s: None,
    )
    ctl.run_for_day()
    assert snapshots2.get().state is AppState.ERROR
    first_calls = store.purge_calls
    ctl.run_for_day()
    assert store.purge_calls == first_calls


def test_c3_csv_failure_does_not_raise_and_surfaces_flag() -> None:
    from traffic_counter.app import DailyController
    from traffic_counter.models import AppState

    clock = FakeClock(at(7, 0))
    pipeline = FakePipeline()
    snapshots = FakeSnapshots(clock())
    controller = DailyController(
        pipeline,
        config=make_config(),
        store=OrderStore([]),
        exporter=OrderExporter([], ok=False),
        snapshots=snapshots,
        clock=clock,
        sleep=lambda s: None,
    )
    controller.recover()
    assert controller._csv_export_ok is False  # type: ignore[attr-defined]
    assert snapshots.get().csv_export_ok is False
    assert snapshots.get().state is not AppState.ERROR


def test_c3_csv_exception_does_not_raise_and_surfaces_flag() -> None:
    from traffic_counter.app import DailyController
    from traffic_counter.models import AppState

    clock = FakeClock(at(7, 0))
    pipeline = FakePipeline()
    snapshots = FakeSnapshots(clock())
    controller = DailyController(
        pipeline,
        config=make_config(),
        store=OrderStore([]),
        exporter=OrderExporter([], ok=True, raise_export=True),
        snapshots=snapshots,
        clock=clock,
        sleep=lambda s: None,
    )
    controller.recover()
    assert snapshots.get().csv_export_ok is False
    assert snapshots.get().state is not AppState.ERROR


def test_c4_midnight_purge_removes_yesterday_keeps_today(tmp_path: Path) -> None:
    from traffic_counter.app import DailyController
    from traffic_counter.models import CountEvent, PeriodName
    from traffic_counter.storage import CsvExporter, TrafficStore

    clock = FakeClock(at(7, 0, day=PRIOR))
    store = TrafficStore(tmp_path / "traffic.db")
    store.initialize()
    exporter = CsvExporter(store, tmp_path / "exports")
    snapshots = FakeSnapshots(clock())
    pipeline = FakePipeline()
    cfg = make_config(tmp_path)
    ctl = DailyController(
        pipeline,
        config=cfg,
        store=store,
        exporter=exporter,
        snapshots=snapshots,
        clock=clock,
        sleep=lambda s: None,
    )
    app_id = store.start_app_session(at(6, 0, day=PRIOR))
    track_id = store.start_tracking_session(app_id, at(6, 0, day=PRIOR))
    yesterday = CountEvent(
        tracking_session_id=track_id,
        track_id=1,
        crossed_at=at(7, 0, day=PRIOR),
        direction="up",
        period=PeriodName.PAGI,
        class_name="car",
    )
    assert store.insert_event(yesterday) is True
    exporter.try_export(PRIOR, at(7, 0, day=PRIOR))
    assert (tmp_path / "exports" / "traffic_2026-09-24.csv").is_file()
    app_id2 = store.start_app_session(at(6, 0))
    track_id2 = store.start_tracking_session(app_id2, at(6, 0))
    today = CountEvent(
        tracking_session_id=track_id2,
        track_id=2,
        crossed_at=at(7, 0),
        direction="up",
        period=PeriodName.PAGI,
        class_name="car",
    )
    assert store.insert_event(today) is True
    clock.set(at(7, 0))
    ctl.run_for_day()
    assert store.summary_counts(DAY, PeriodName.PAGI) == 1
    assert store.summary_counts(PRIOR, PeriodName.PAGI) == 0
    assert len(store.events(DAY)) == 1
    assert not (tmp_path / "exports" / "traffic_2026-09-24.csv").is_file()
    store.close()


def test_c4_purge_runs_once_per_day() -> None:
    from traffic_counter.app import DailyController

    clock = FakeClock(at(7, 0))
    pipeline = FakePipeline()
    snapshots = FakeSnapshots(clock())
    order: list[str] = []
    store = OrderStore(order)
    exporter = OrderExporter(order, ok=True)
    ctl = DailyController(
        pipeline,
        config=make_config(),
        store=store,
        exporter=exporter,
        snapshots=snapshots,
        clock=clock,
        sleep=lambda s: None,
    )
    ctl.run_for_day()
    first_store = store.purge_calls
    first_exporter = exporter.purge_calls
    assert first_store == 1
    assert first_exporter == 1
    clock.set(at(8, 0))
    ctl.run_for_day()
    assert store.purge_calls == first_store
    assert exporter.purge_calls == first_exporter


def test_c4_purge_covers_pre_window_midnight(tmp_path: Path) -> None:
    from traffic_counter.app import DailyController
    from traffic_counter.models import AppState, CountEvent, PeriodName
    from traffic_counter.storage import CsvExporter, TrafficStore

    clock = FakeClock(at(2, 0))
    store = TrafficStore(tmp_path / "traffic.db")
    store.initialize()
    exporter = CsvExporter(store, tmp_path / "exports")
    snapshots = FakeSnapshots(clock())
    pipeline = FakePipeline()
    cfg = make_config(tmp_path)
    ctl = DailyController(
        pipeline,
        config=cfg,
        store=store,
        exporter=exporter,
        snapshots=snapshots,
        clock=clock,
        sleep=lambda s: None,
    )
    app_id = store.start_app_session(at(6, 0, day=PRIOR))
    track_id = store.start_tracking_session(app_id, at(6, 0, day=PRIOR))
    yesterday = CountEvent(
        tracking_session_id=track_id,
        track_id=5,
        crossed_at=at(7, 0, day=PRIOR),
        direction="up",
        period=PeriodName.PAGI,
        class_name="car",
    )
    assert store.insert_event(yesterday) is True
    exporter.try_export(PRIOR, at(7, 0, day=PRIOR))
    clock.set(at(2, 0))
    ctl.run_for_day()
    assert snapshots.get().state is AppState.WAITING_FOR_WINDOW
    assert store.summary_counts(PRIOR, PeriodName.PAGI) == 0
    assert not (tmp_path / "exports" / "traffic_2026-09-24.csv").is_file()
    store.close()


def test_i5_drain_seconds_accepts_60_and_rejects_61(tmp_path: Path) -> None:
    from traffic_counter.config import load_config

    base = (
        '[stream]\nurl = "https://example.test/index.m3u8"\n\n[model]\nname = "yolo26n.pt"\n\n'
        "[line]\nx1 = 0.10\ny1 = 0.80\nx2 = 0.90\ny2 = 0.80\n\n[timing]\n"
    )
    out = "[output]\n"
    p60 = tmp_path / "c60.toml"
    p60.write_text(base + "drain_seconds = 60.0\n" + out, encoding="utf-8")
    cfg = load_config(p60)
    assert cfg.timing.drain_seconds == 60.0
    p61 = tmp_path / "c61.toml"
    p61.write_text(base + "drain_seconds = 61.0\n" + out, encoding="utf-8")
    from traffic_counter.config import ConfigError
    with pytest.raises(ConfigError):
        load_config(p61)


def test_c2_dashboard_js_references_incomplete_fields() -> None:
    text = (
        Path(__file__).resolve().parents[1] / "src" / "traffic_counter" / "static" / "dashboard.js"
    ).read_text(encoding="utf-8")
    assert "incomplete" in text
    assert "dropped_events" in text
    assert "csv_export_ok" in text
    assert "last_error" in text
    assert "renderWarning" in text


def test_c2_dashboard_js_midnight_banner() -> None:
    text = (
        Path(__file__).resolve().parents[1] / "src" / "traffic_counter" / "static" / "dashboard.js"
    ).read_text(encoding="utf-8")
    assert "midnight" in text.lower() or "tengah malam" in text.lower()
    assert "19" in text
    assert "generated_at" in text


def test_i6_web_redacts_relative_traversals() -> None:
    from traffic_counter.web import _sanitize_error

    assert "<path>" in (_sanitize_error("failed at ../../etc/passwd boom") or "")
    assert ".." not in (_sanitize_error("failed at ../../etc/passwd boom") or "").replace(
        "<path>", ""
    )
    assert "<path>" in (_sanitize_error("failed at ..\\windows\\secret boom") or "")
    kept = _sanitize_error("GET /api/status failed with timeout") or ""
    assert "/api/status" in kept


def test_i6_stream_redacts_relative_traversals() -> None:
    from traffic_counter.stream import sanitize_diagnostic_line

    assert "<path>" in sanitize_diagnostic_line("failed at ../../etc/passwd boom")
    assert "<path>" in sanitize_diagnostic_line("failed at ..\\windows\\secret boom")
    assert "/api/status" in sanitize_diagnostic_line("GET /api/status failed")


def test_i1_day_scoping_excludes_yesterday(tmp_path: Path) -> None:
    from traffic_counter.models import CountEvent, PeriodName
    from traffic_counter.storage import TrafficStore

    store = TrafficStore(tmp_path / "traffic.db")
    store.initialize()
    try:
        app_id = store.start_app_session(at(6, 0, day=PRIOR))
        track_id = store.start_tracking_session(app_id, at(6, 0, day=PRIOR))
        assert (
            store.insert_event(
                CountEvent(
                    tracking_session_id=track_id,
                    track_id=1,
                    crossed_at=at(7, 0, day=PRIOR),
                    direction="up",
                    period=PeriodName.PAGI,
                    class_name="car",
                )
            )
            is True
        )
        app_id2 = store.start_app_session(at(6, 0))
        track_id2 = store.start_tracking_session(app_id2, at(6, 0))
        assert (
            store.insert_event(
                CountEvent(
                    tracking_session_id=track_id2,
                    track_id=2,
                    crossed_at=at(7, 0),
                    direction="up",
                    period=PeriodName.PAGI,
                    class_name="car",
                )
            )
            is True
        )
        assert store.summary_counts(DAY, PeriodName.PAGI) == 1
        assert store.summary_counts(PRIOR, PeriodName.PAGI) == 1
        today_events = store.events(DAY)
        assert len(today_events) == 1
        assert today_events[0].track_id == 2
        assert len(store.events(PRIOR)) == 1
    finally:
        store.close()


def test_i1_timeseries_day_bounded(tmp_path: Path) -> None:
    from traffic_counter.models import CountEvent, HealthBucket, PeriodName
    from traffic_counter.storage import TrafficStore

    store = TrafficStore(tmp_path / "traffic.db")
    store.initialize()
    try:
        app_id = store.start_app_session(at(6, 0, day=PRIOR))
        track_id = store.start_tracking_session(app_id, at(6, 0, day=PRIOR))
        store.insert_event(
            CountEvent(
                tracking_session_id=track_id,
                track_id=1,
                crossed_at=at(6, 0, 30, day=PRIOR),
                direction="up",
                period=PeriodName.PAGI,
                class_name="car",
            )
        )
        store.write_health_buckets(
            (
                HealthBucket(
                    at(6, 0, day=PRIOR),
                    PeriodName.PAGI,
                    tuple(at(6, 0, s, day=PRIOR) for s in range(45)),
                ),
            )
        )
        store.refresh_minute_rollups(PRIOR, at(6, 2, day=PRIOR))
        assert store.timeseries(DAY) == ()
        assert len(store.timeseries(PRIOR)) == 3
    finally:
        store.close()


def test_i2_pre_six_suppression_deliberately_consumes_budget() -> None:
    from traffic_counter.config import LineConfig
    from traffic_counter.counter import LineCounter
    from traffic_counter.models import TrackedVehicle

    line = LineConfig(0.0, 0.5, 1.0, 0.5, 0.005)

    def vehicle(bottom: float) -> TrackedVehicle:
        return TrackedVehicle(
            track_id=7,
            class_name="car",
            confidence=0.9,
            box_xyxy=(0.45, bottom - 0.10, 0.55, bottom),
        )

    base = at(5, 59, 59)
    counter2 = LineCounter(line)
    assert counter2.observe(1, vehicle(0.4375), base) is None
    assert counter2.observe(1, vehicle(0.5625), base + timedelta(milliseconds=100)) is None
    assert counter2.observe(1, vehicle(0.4375), at(6, 0, 0)) is None
    assert counter2.observe(1, vehicle(0.5625), at(6, 0, 1)) is None


def test_geometry_rejects_non_finite_distance() -> None:
    from traffic_counter.geometry import classify_side, interpolate_crossing

    with pytest.raises(ValueError):
        classify_side(float("nan"), 0.005)
    with pytest.raises(ValueError):
        classify_side(float("inf"), 0.005)
    with pytest.raises(ValueError):
        interpolate_crossing((at(6, 0), float("nan")), (at(6, 0, 1), 0.1))
    with pytest.raises(ValueError):
        interpolate_crossing((at(6, 0), -0.1), (at(6, 0, 1), float("inf")))


def test_counter_evicts_stable_old_tracks_over_5000() -> None:
    from traffic_counter.config import LineConfig
    from traffic_counter.counter import LineCounter
    from traffic_counter.models import TrackedVehicle

    line = LineConfig(0.0, 0.5, 1.0, 0.5, 0.005)
    counter = LineCounter(line)
    base = at(6, 0)

    def vehicle(tid: int) -> TrackedVehicle:
        return TrackedVehicle(
            track_id=tid, class_name="car", confidence=0.9, box_xyxy=(0.45, 0.3375, 0.55, 0.4375)
        )

    for tid in range(1, 5001):
        assert counter.observe(1, vehicle(tid), base) is None
    assert len(counter._sessions[1]) == 5000  # type: ignore[attr-defined]
    late = base + timedelta(seconds=61)
    assert counter.observe(1, vehicle(9999), late) is None
    assert len(counter._sessions[1]) == 5000  # type: ignore[attr-defined]
    assert 9999 in counter._sessions[1]  # type: ignore[attr-defined]


def test_counter_never_evicts_recently_updated_tracks() -> None:
    from traffic_counter.config import LineConfig
    from traffic_counter.counter import LineCounter
    from traffic_counter.models import TrackedVehicle

    line = LineConfig(0.0, 0.5, 1.0, 0.5, 0.005)
    counter = LineCounter(line)
    base = at(6, 0)

    def vehicle(tid: int) -> TrackedVehicle:
        return TrackedVehicle(
            track_id=tid, class_name="car", confidence=0.9, box_xyxy=(0.45, 0.3375, 0.55, 0.4375)
        )

    for tid in range(1, 5001):
        counter.observe(1, vehicle(tid), base)
    late = base + timedelta(seconds=10)
    assert counter.observe(1, vehicle(9999), late) is None
    assert len(counter._sessions[1]) == 5001  # type: ignore[attr-defined]
    assert 1 in counter._sessions[1]  # type: ignore[attr-defined]


def test_detector_rejects_zero_and_negative_ids() -> None:
    from traffic_counter.detector import adapt_tracked_boxes
    from traffic_counter.models import VEHICLE_CLASSES

    names = {0: "person", 1: "car"}

    class Boxes:
        def __init__(self, ids, cls, conf, xyxy):
            self.id = ids
            self.cls = cls
            self.conf = conf
            self.xyxy = xyxy

    class Result:
        def __init__(self, boxes, names):
            self.boxes = boxes
            self.names = names

    result = Result(
        Boxes([0, -1, 3], [1, 1, 1], [0.9, 0.9, 0.9], [[10.0, 20.0, 30.0, 40.0]] * 3), names
    )
    vehicles = adapt_tracked_boxes(result, set(VEHICLE_CLASSES), 100.0, 100.0)
    assert [v.track_id for v in vehicles] == [3]


def test_detector_rejects_out_of_range_confidence() -> None:
    from traffic_counter.detector import adapt_tracked_boxes
    from traffic_counter.models import VEHICLE_CLASSES

    names = {1: "car"}

    class Boxes:
        def __init__(self, ids, cls, conf, xyxy):
            self.id = ids
            self.cls = cls
            self.conf = conf
            self.xyxy = xyxy

    class Result:
        def __init__(self, boxes, names):
            self.boxes = boxes
            self.names = names

    result = Result(
        Boxes([1, 2, 3], [1, 1, 1], [2.0, -0.5, 0.5], [[10.0, 20.0, 30.0, 40.0]] * 3), names
    )
    vehicles = adapt_tracked_boxes(result, set(VEHICLE_CLASSES), 100.0, 100.0)
    assert [v.track_id for v in vehicles] == [3]


def test_detector_clamps_edge_boxes_and_drops_collapsed() -> None:
    from traffic_counter.detector import adapt_tracked_boxes
    from traffic_counter.models import VEHICLE_CLASSES

    names = {1: "car"}

    class Boxes:
        def __init__(self, ids, cls, conf, xyxy):
            self.id = ids
            self.cls = cls
            self.conf = conf
            self.xyxy = xyxy

    class Result:
        def __init__(self, boxes, names):
            self.boxes = boxes
            self.names = names

    edge = (-5.0, 20.0, 102.0, 40.0)
    collapsed = (150.0, 50.0, 160.0, 60.0)
    result = Result(Boxes([1, 2], [1, 1], [0.9, 0.9], [list(edge), list(collapsed)]), names)
    vehicles = adapt_tracked_boxes(result, set(VEHICLE_CLASSES), 100.0, 100.0)
    assert len(vehicles) == 1
    assert vehicles[0].track_id == 1
    assert vehicles[0].box_xyxy == (0.0, 0.2, 1.0, 0.4)


def test_pipeline_stop_failure_retries_once_and_records() -> None:
    from traffic_counter.config import (
        AppConfig,
        LineConfig,
        ModelConfig,
        OutputConfig,
        StreamConfig,
        TimingConfig,
    )
    from traffic_counter.pipeline import TrafficPipeline
    from traffic_counter.state import SnapshotStore

    config = AppConfig(
        stream=StreamConfig(url="https://example.test/index.m3u8", width=12, height=8),
        model=ModelConfig(name="yolo26n.pt"),
        line=LineConfig(x1=0.10, y1=0.80, x2=0.90, y2=0.80, hysteresis=0.005),
        timing=TimingConfig(),
        output=OutputConfig(
            port=5000,
            database=Path("data/traffic.db"),
            export_directory=Path("data/exports"),
            calibration_directory=Path("data/calibration"),
            log_file=Path("data/t.log"),
        ),
    )

    class FailStore:
        def __init__(self) -> None:
            self.stop_calls = 0

        def start_app_session(self, started_at: datetime) -> int:
            return 1

        def start_tracking_session(self, app_session_id: int, started_at: datetime) -> int:
            return 1

        def stop_tracking_session(self, tracking_session_id: int, stopped_at: datetime) -> None:
            self.stop_calls += 1
            raise sqlite3.OperationalError("stop locked")

        def stop_app_session(self, app_session_id: int, stopped_at: datetime, state) -> None:
            return None

        def insert_event(self, event) -> bool:
            return True

        def write_health_buckets(self, buckets) -> None:
            return None

        def refresh_minute_rollups(self, day: date, now: datetime) -> None:
            return None

        def timeseries(self, day: date):
            return ()

        def events(self, day: date, period=None):
            return ()

        def health_buckets(self, period=None):
            return ()

    class FakeExporter:
        def try_export(self, day: date, now: datetime) -> bool:
            return True

        def purge_before(self, day: date) -> None:
            return None

    class FakeSlot:
        def take(self):
            return None

        def stats(self):
            from traffic_counter.models import FrameSlotStats

            return FrameSlotStats(received=0, replaced=0)

    class FakeCounter:
        def end_tracking_session(self, sid: int) -> None:
            return None

    class FakeFactory:
        def create(self, cfg) -> object:
            class Det:
                def detect(self, frame) -> list:
                    return []

            return Det()

    clock = FakeClock(at(7, 0))
    snapshots = SnapshotStore(clock=clock)
    store = FailStore()
    pipeline = TrafficPipeline(
        config=config,
        store=store,
        exporter=FakeExporter(),
        snapshots=snapshots,
        slot=FakeSlot(),
        counter=FakeCounter(),
        detector_factory=FakeFactory(),
        clock=clock,
    )
    pipeline.start()
    pipeline.request_stop()
    assert store.stop_calls == 2
    assert snapshots.get().last_error is not None


def test_stream_release_closes_stdout() -> None:
    from io import BytesIO

    from traffic_counter.config import StreamConfig, TimingConfig

    class FakePipe(BytesIO):
        def __init__(self, data: bytes = b""):
            super().__init__(data)
            self.is_closed = False

        def close(self) -> None:
            self.is_closed = True

    class FakeStdin(BytesIO):
        def __init__(self) -> None:
            super().__init__()
            self.is_closed = False

        def close(self) -> None:
            self.is_closed = True

    class FakeProc:
        def __init__(self) -> None:
            self.stdin: object = FakeStdin()
            self.stdout: object = FakePipe(b"")
            self.terminate_calls = 0

        def poll(self):
            return None

        def terminate(self) -> None:
            self.terminate_calls += 1

        def kill(self) -> None:
            return None

        def wait(self, timeout=None) -> int:
            return 0

    from traffic_counter.stream import HlsFrameSource

    cfg = StreamConfig(url="https://example.test/index.m3u8", width=12, height=8)
    timing = TimingConfig()
    src = HlsFrameSource(
        cfg,
        timing,
        executable="ffmpeg",
        spawn=lambda cmd: (_ for _ in ()).throw(AssertionError("no spawn")),
    )
    proc = FakeProc()
    src._process = proc  # type: ignore[attr-defined]
    src._diagnostics = BytesIO(b"")  # type: ignore[attr-defined]
    src._release()
    assert proc.stdout.is_closed is True  # type: ignore[attr-defined]
    assert proc.stdin.is_closed is True  # type: ignore[attr-defined]


def test_stream_stop_during_spawn_terminates_child() -> None:
    from io import BytesIO

    from traffic_counter.config import StreamConfig, TimingConfig
    from traffic_counter.stream import HlsFrameSource, SpawnedProcess

    cfg = StreamConfig(url="https://example.test/index.m3u8", width=12, height=8)
    timing = TimingConfig()
    holder: dict[str, object] = {}

    class FakePipe(BytesIO):
        def __init__(self, data: bytes = b""):
            super().__init__(data)
            self.is_closed = False

        def close(self) -> None:
            self.is_closed = True

    class FakeProc:
        def __init__(self) -> None:
            self.stdin = FakePipe(b"")
            self.stdout = FakePipe(b"")
            self.terminate_calls = 0

        def poll(self):
            return None

        def terminate(self) -> None:
            self.terminate_calls += 1

        def kill(self) -> None:
            return None

        def wait(self, timeout=None) -> int:
            return 0

    src_holder: dict[str, HlsFrameSource] = {}

    def spawn(cmd):  # type: ignore[no-untyped-def]
        proc = FakeProc()
        holder["proc"] = proc
        src_holder["src"]._stopped = True  # type: ignore[attr-defined]
        return SpawnedProcess(process=proc, diagnostics=BytesIO(b""))

    src = HlsFrameSource(cfg, timing, executable="ffmpeg", spawn=spawn)
    src_holder["src"] = src
    handle = src._open()
    assert handle is None
    proc = holder["proc"]
    assert proc.terminate_calls >= 1  # type: ignore[attr-defined]


def test_overlay_picks_last_started_period_sore() -> None:
    from traffic_counter.models import PeriodName, PeriodSummary
    from traffic_counter.overlay import _totals

    summaries = {
        PeriodName.PAGI: PeriodSummary(PeriodName.PAGI, True, 10, 100, 6.0, None),
        PeriodName.SIANG: PeriodSummary(PeriodName.SIANG, True, 20, 100, 12.0, None),
        PeriodName.SORE: PeriodSummary(PeriodName.SORE, True, 5, 50, 6.0, None),
    }
    current, total = _totals(summaries)
    assert current == "sore"
    assert total == 35


def test_overlay_box_scales_unconditionally_to_edge() -> None:
    from traffic_counter.overlay import _box_pixels

    x1, y1, x2, y2 = _box_pixels((0.10, 0.20, 1.015, 0.40), 320, 240)
    assert x2 == 319
    assert x1 == int(0.10 * 320)
    assert y1 == int(0.20 * 240)
    assert y2 == int(0.40 * 240)
    assert x1 != 0
    assert x2 != 1


def test_cli_smoke_includes_api_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:  # type: ignore[no-untyped-def]
    import numpy as np

    import traffic_counter.cli as cli_module

    text = (
        "[stream]\nurl = \"https://example.test/index.m3u8\"\n"
        "width = 12\nheight = 8\n\n[model]\nname = \"yolo26n.pt\"\n"
        "[line]\nx1 = 0.10\ny1 = 0.80\nx2 = 0.90\ny2 = 0.80\n"
        "\n[timing]\n\n[output]\n"
    )
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(text, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    from traffic_counter.models import Frame

    moment = datetime(2026, 9, 25, 12, 0, tzinfo=JAKARTA)
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    frame = Frame(image=image, observed_at=moment)

    class FakeSource:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

        def frames(self):  # type: ignore[no-untyped-def]
            def gen():  # type: ignore[no-untyped-def]
                yield frame

            return gen()

        def stop(self) -> None:
            return None

    class FakeDetector:
        def detect(self, frame: Frame) -> list:
            return []

    class FakeFactory:
        def create(self, config) -> FakeDetector:  # type: ignore[no-untyped-def]
            return FakeDetector()

    monkeypatch.setattr(cli_module, "HlsFrameSource", FakeSource)
    monkeypatch.setattr(cli_module, "DetectorFactory", FakeFactory)
    code = cli_module.main(["--config", str(cfg_path), "--smoke-seconds", "1"])
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out.strip().splitlines()[-1])
    assert "api_ok" in payload
    assert payload["api_ok"] is True


def test_docs_record_gpu_operational_profile() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = (
        root / "docs" / "superpowers" / "specs" / "2026-09-25-traffic-counter-design.md"
    ).read_text(encoding="utf-8")
    plan = (
        root / "docs" / "superpowers" / "plans" / "2026-09-25-one-day-traffic-counter.md"
    ).read_text(encoding="utf-8")
    for token in ("yolo26s.pt", "cuda:0", "0.15", "960", "0.08", "0.62", "calibrated=true"):
        assert token in spec
    for token in ("yolo26s.pt", "cuda:0", "0.15", "960"):
        assert token in plan
    for token in ("yolo26n.pt", "cpu", "640"):
        assert token in spec
