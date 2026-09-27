from __future__ import annotations

import inspect
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from traffic_counter.config import (
    AppConfig,
    LineConfig,
    ModelConfig,
    OutputConfig,
    StreamConfig,
    TimingConfig,
)
from traffic_counter.models import AppSnapshot, AppState

JAKARTA = ZoneInfo("Asia/Jakarta")
DAY = date(2026, 9, 25)


def at(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, 9, 25, hour, minute, second, tzinfo=JAKARTA)


def make_config(tmp_path: Path | None = None) -> AppConfig:
    base = Path("data/traffic.db") if tmp_path is None else tmp_path / "traffic.db"
    exports = Path("data/exports") if tmp_path is None else tmp_path / "exports"
    calib = Path("data/calibration") if tmp_path is None else tmp_path / "calibration"
    log = Path("data/traffic-counter.log") if tmp_path is None else tmp_path / "t.log"
    stream = StreamConfig(url="https://example.test/index.m3u8", width=12, height=8)
    model = ModelConfig(
        name="yolo26n.pt",
        device="cpu",
        confidence=0.25,
        iou=0.70,
        max_detections=100,
        image_size=640,
        tracker="bytetrack.yaml",
        persist=True,
    )
    line = LineConfig(x1=0.10, y1=0.80, x2=0.90, y2=0.80, hysteresis=0.005)
    timing = TimingConfig(
        target_fps=10,
        reconnect_initial_seconds=1.0,
        reconnect_max_seconds=30.0,
        drain_seconds=5.0,
        health_bucket_seconds=10,
        rolling_min_healthy_seconds=60,
    )
    output = OutputConfig(
        port=5000,
        database=base,
        export_directory=exports,
        calibration_directory=calib,
        log_file=log,
    )
    return AppConfig(stream=stream, model=model, line=line, timing=timing, output=output)


class FakeClock:
    def __init__(self, start: datetime) -> None:
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


class FakePipeline:
    def __init__(self) -> None:
        self.started_at: datetime | None = None
        self.start_calls = 0
        self.run_until_calls: list[datetime] = []
        self.drain_seconds: float | None = None
        self.stopped = False
        self.stop_calls = 0
        self.fail_on_start: BaseException | None = None

    def start(self) -> None:
        self.start_calls += 1
        if self.fail_on_start is not None:
            raise self.fail_on_start
        self.started_at = datetime(2026, 9, 25, 6, 0, tzinfo=JAKARTA)

    def run_until(self, end_at: datetime) -> None:
        self.run_until_calls.append(end_at)

    def drain_for(self, seconds: float) -> None:
        self.drain_seconds = seconds
        self.stopped = True

    def request_stop(self) -> None:
        self.stop_calls += 1
        self.stopped = True

    def accepting_counts(self) -> bool:
        return not self.stopped and self.started_at is not None


class FakeSnapshots:
    def __init__(self, now: datetime) -> None:
        self.current: AppSnapshot = AppSnapshot.initial(now)
        self.publishes = 0

    def get(self) -> AppSnapshot:
        return self.current

    def publish(self, snapshot: AppSnapshot) -> None:
        self.current = snapshot
        self.publishes += 1


class OrderStore:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    def purge_before(self, day: date) -> None:
        self.order.append("store")

    def initialize(self) -> None:
        return None


class OrderExporter:
    def __init__(self, order: list[str], ok: bool = True) -> None:
        self.order = order
        self.ok = ok

    def purge_before(self, day: date) -> None:
        self.order.append("exporter")

    def try_export(self, day: date, now: datetime) -> bool:
        self.order.append("export")
        return self.ok


class FailingExporter:
    def purge_before(self, day: date) -> None:
        return None

    def try_export(self, day: date, now: datetime) -> bool:
        raise OSError("export locked")


def make_controller(
    clock: FakeClock,
    pipeline: FakePipeline,
    snapshots: FakeSnapshots,
    order: list[str] | None = None,
    config: AppConfig | None = None,
    sleeper: FakeSleeper | None = None,
    exporter_ok: bool = True,
) -> Any:
    from traffic_counter.app import DailyController

    use_order: list[str] = [] if order is None else order
    store = OrderStore(use_order)
    exporter = OrderExporter(use_order, ok=exporter_ok)
    used_config = make_config() if config is None else config
    used_sleep = FakeSleeper(clock) if sleeper is None else sleeper
    return DailyController(
        pipeline,
        config=used_config,
        store=store,
        exporter=exporter,
        snapshots=snapshots,
        clock=clock,
        sleep=used_sleep,
    )


def test_waiting_before_window_publishes_waiting() -> None:
    clock = FakeClock(at(5, 59, 59))
    pipeline = FakePipeline()
    snapshots = FakeSnapshots(clock())
    controller = make_controller(clock, pipeline, snapshots)
    controller.run_for_day()
    assert pipeline.started_at is None
    assert pipeline.start_calls == 0
    assert snapshots.get().state is AppState.WAITING_FOR_WINDOW


def test_starts_at_six_and_drains_at_nineteen() -> None:
    clock = FakeClock(at(5, 59, 59))
    pipeline = FakePipeline()
    snapshots = FakeSnapshots(clock())
    controller = make_controller(clock, pipeline, snapshots)
    controller.run_for_day()
    assert pipeline.started_at is None
    clock.set(at(6, 0, 0))
    controller.run_for_day()
    assert pipeline.start_calls == 1
    assert pipeline.drain_seconds is None
    clock.set(at(19, 0, 0))
    controller.run_for_day()
    assert pipeline.drain_seconds == 5.0
    assert pipeline.stopped is True
    assert snapshots.get().state is AppState.STOPPED


def test_waiting_after_nineteen_publishes_stopped() -> None:
    clock = FakeClock(at(19, 5, 0))
    pipeline = FakePipeline()
    snapshots = FakeSnapshots(clock())
    controller = make_controller(clock, pipeline, snapshots)
    controller.run_for_day()
    assert pipeline.start_calls == 0
    assert snapshots.get().state is AppState.STOPPED


def test_recovery_purge_order_store_then_exporter_then_regenerate() -> None:
    clock = FakeClock(at(7, 0, 0))
    pipeline = FakePipeline()
    snapshots = FakeSnapshots(clock())
    order: list[str] = []
    controller = make_controller(clock, pipeline, snapshots, order=order)
    controller.recover()
    assert order == ["store", "exporter", "export"]


def test_recover_preserves_current_day_events(tmp_path: Path) -> None:
    from traffic_counter.app import DailyController
    from traffic_counter.models import CountEvent, PeriodName
    from traffic_counter.storage import CsvExporter, TrafficStore

    clock = FakeClock(at(7, 0, 0))
    store = TrafficStore(tmp_path / "traffic.db")
    store.initialize()
    exporter = CsvExporter(store, tmp_path / "exports")
    snapshots = FakeSnapshots(clock())
    pipeline = FakePipeline()
    config = make_config(tmp_path)

    def idle(seconds: float) -> None:
        return None

    controller = DailyController(
        pipeline,
        config=config,
        store=store,
        exporter=exporter,
        snapshots=snapshots,
        clock=clock,
        sleep=idle,
    )
    app_id = store.start_app_session(at(6, 0, 0))
    track_id = store.start_tracking_session(app_id, at(6, 0, 0))
    event = CountEvent(
        tracking_session_id=track_id,
        track_id=9,
        crossed_at=at(7, 0, 0),
        direction="up",
        period=PeriodName.PAGI,
        class_name="car",
    )
    assert store.insert_event(event) is True
    controller.recover()
    assert store.summary_counts(DAY, PeriodName.PAGI) == 1
    assert (tmp_path / "exports" / "traffic_2026-09-25.csv").is_file()
    store.close()


def test_recover_false_export_surfaces_error() -> None:
    clock = FakeClock(at(7, 0, 0))
    pipeline = FakePipeline()
    snapshots = FakeSnapshots(clock())
    controller = make_controller(clock, pipeline, snapshots, exporter_ok=False)
    controller.recover()
    assert snapshots.get().csv_export_ok is False
    assert snapshots.get().state is not AppState.ERROR


def test_recover_raising_export_surfaces_error() -> None:
    from traffic_counter.app import DailyController

    clock = FakeClock(at(7, 0, 0))
    pipeline = FakePipeline()
    snapshots = FakeSnapshots(clock())
    controller = DailyController(
        pipeline,
        config=make_config(),
        store=OrderStore([]),
        exporter=FailingExporter(),
        snapshots=snapshots,
        clock=clock,
        sleep=FakeSleeper(clock),
    )
    controller.recover()
    assert snapshots.get().csv_export_ok is False
    assert snapshots.get().state is not AppState.ERROR


def test_ctrl_c_requests_graceful_stop() -> None:
    clock = FakeClock(at(7, 0, 0))
    pipeline = FakePipeline()
    snapshots = FakeSnapshots(clock())

    class InterruptSleep:
        def __call__(self, seconds: float) -> None:
            raise KeyboardInterrupt

    from traffic_counter.app import DailyController

    controller = DailyController(
        pipeline,
        config=make_config(),
        store=OrderStore([]),
        exporter=OrderExporter([]),
        snapshots=snapshots,
        clock=clock,
        sleep=InterruptSleep(),
    )
    clock.set(at(6, 0, 0))
    pipeline.started_at = at(6, 0, 0)
    try:
        controller.run_forever()
    except KeyboardInterrupt as err:
        raise AssertionError("run_forever must swallow interrupt") from err
    assert pipeline.stop_calls >= 1 or pipeline.stopped is True


def test_sleep_failure_publishes_terminal_state() -> None:
    from traffic_counter.app import DailyController

    clock = FakeClock(at(5, 0, 0))

    def bad_sleep(seconds: float) -> None:
        raise RuntimeError("clock broken")

    pipeline = FakePipeline()
    snapshots = FakeSnapshots(clock())
    controller = DailyController(
        pipeline,
        config=make_config(),
        store=OrderStore([]),
        exporter=OrderExporter([]),
        snapshots=snapshots,
        clock=clock,
        sleep=bad_sleep,
    )
    controller.run_forever()
    assert snapshots.get().state is AppState.STOPPED


def test_database_error_prevents_automatic_restart() -> None:
    clock = FakeClock(at(6, 0, 0))
    pipeline = FakePipeline()
    pipeline.fail_on_start = OSError("disk down")
    snapshots = FakeSnapshots(clock())
    controller = make_controller(clock, pipeline, snapshots)
    controller.run_for_day()
    assert snapshots.get().state is AppState.ERROR
    assert snapshots.get().incomplete is True
    first_calls = pipeline.start_calls
    first_runs = len(pipeline.run_until_calls)
    controller.run_for_day()
    assert pipeline.start_calls == first_calls
    assert len(pipeline.run_until_calls) == first_runs
    assert snapshots.get().state is AppState.ERROR


def test_failed_day_clears_on_rollover() -> None:
    clock = FakeClock(at(6, 0, 0))
    pipeline = FakePipeline()
    pipeline.fail_on_start = OSError("disk down")
    snapshots = FakeSnapshots(clock())
    controller = make_controller(clock, pipeline, snapshots)
    controller.run_for_day()
    assert snapshots.get().state is AppState.ERROR
    clock.set(datetime(2026, 9, 26, 6, 0, 0, tzinfo=JAKARTA))
    pipeline.fail_on_start = None
    controller.run_for_day()
    assert pipeline.start_calls == 2
    assert snapshots.get().state is not AppState.ERROR


def test_dashboard_param_defaults_to_seam() -> None:
    from traffic_counter.app import DailyController
    from traffic_counter.app import serve_dashboard as seam

    params = inspect.signature(DailyController.__init__).parameters
    assert "serve_dashboard" not in params
    assert "serve_dashboard_fn" in params
    assert params["serve_dashboard_fn"].default is seam


def test_dashboard_legacy_alias_still_accepted() -> None:
    from traffic_counter.app import DailyController

    clock = FakeClock(at(7, 0, 0))
    pipeline = FakePipeline()
    snapshots = FakeSnapshots(clock())
    seen: list[str] = []

    def legacy(store: object, provider: object, config: object) -> None:
        seen.append("called")

    controller = DailyController(
        pipeline,
        config=make_config(),
        store=OrderStore([]),
        exporter=OrderExporter([]),
        snapshots=snapshots,
        clock=clock,
        sleep=FakeSleeper(clock),
        serve_dashboard=legacy,
    )
    assert seen == []
    assert controller is not None


def test_provider_uses_config_directory(tmp_path: Path) -> None:
    from traffic_counter.app import DailyController

    clock = FakeClock(at(6, 0, 0))
    pipeline = FakePipeline()
    snapshots = FakeSnapshots(clock())
    config = make_config(tmp_path)
    captured: list[Path] = []

    def fake_dashboard(store: object, provider: Callable[[], Path], cfg: object) -> None:
        captured.append(provider())

    controller = DailyController(
        pipeline,
        config=config,
        store=OrderStore([]),
        exporter=OrderExporter([]),
        snapshots=snapshots,
        clock=clock,
        sleep=FakeSleeper(clock),
        serve_dashboard_fn=fake_dashboard,
    )
    controller.run_for_day()
    assert len(captured) == 1
    assert captured[0].parent == Path(config.output.export_directory)
    assert captured[0].name == "traffic_2026-09-25.csv"


def test_app_has_no_private_exporter_access() -> None:
    from traffic_counter import app as app_module

    text = Path(app_module.__file__ or "").read_text(encoding="utf-8")
    assert "._directory" not in text


def test_serve_dashboard_seam_exists_without_web_import() -> None:
    from traffic_counter import app as app_module

    assert hasattr(app_module, "serve_dashboard")
    text = Path(app_module.__file__ or "").read_text(encoding="utf-8")
    assert "from traffic_counter.web import" not in text
    assert "from .web import" not in text
    assert "import traffic_counter.web" not in text
