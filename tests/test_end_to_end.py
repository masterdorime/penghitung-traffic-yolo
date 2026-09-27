from __future__ import annotations

import csv
from contextlib import suppress
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from traffic_counter.config import (
    AppConfig,
    LineConfig,
    ModelConfig,
    OutputConfig,
    StreamConfig,
    TimingConfig,
)
from traffic_counter.counter import LineCounter
from traffic_counter.models import Frame, PeriodName, TrackedVehicle
from traffic_counter.pipeline import TrafficPipeline
from traffic_counter.state import LatestFrameSlot, SnapshotStore
from traffic_counter.storage import CsvExporter, TrafficStore
from traffic_counter.timeutils import JAKARTA
from traffic_counter.web import create_app

DAY = date(2026, 9, 25)


def at(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, 9, 25, hour, minute, second, tzinfo=JAKARTA)


class FakeClock:
    def __init__(self, start: datetime) -> None:
        self.current = start

    def __call__(self) -> datetime:
        return self.current

    def set(self, moment: datetime) -> None:
        self.current = moment

    def advance(self, seconds: float) -> None:
        self.current = self.current + timedelta(seconds=seconds)


class ScriptedDetector:
    def detect(self, frame: Frame) -> list[TrackedVehicle]:
        marker = int(frame.image[0, 0, 0])
        bottom_y = 0.40 if marker == 1 else 0.60
        return [
            TrackedVehicle(
                track_id=7,
                class_name="car",
                confidence=0.9,
                box_xyxy=(0.45, bottom_y - 0.10, 0.55, bottom_y),
            )
        ]


class ScriptedFactory:
    def __init__(self) -> None:
        self.created: list[ScriptedDetector] = []

    def create(self, config: object) -> ScriptedDetector:
        detector = ScriptedDetector()
        self.created.append(detector)
        return detector


def make_config(tmp_path: Path) -> AppConfig:
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
        line=LineConfig(x1=0.0, y1=0.5, x2=1.0, y2=0.5, hysteresis=0.005),
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
            database=tmp_path / "traffic.db",
            export_directory=tmp_path / "exports",
            calibration_directory=tmp_path / "calibration",
            log_file=tmp_path / "traffic-counter.log",
        ),
    )


def make_frame(marker: int, moment: datetime) -> Frame:
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    image[0, 0] = marker
    return Frame(image=image, observed_at=moment)


def test_end_to_end_single_crossing_with_reconnect(tmp_path: Path) -> None:
    clock = FakeClock(at(7, 0))
    config = make_config(tmp_path)
    store = TrafficStore(tmp_path / "traffic.db", clock=clock)
    store.initialize()
    exporter = CsvExporter(store, tmp_path / "exports")
    snapshots = SnapshotStore(clock=clock)
    slot = LatestFrameSlot()
    counter = LineCounter(config.line)
    factory = ScriptedFactory()
    pipeline = TrafficPipeline(
        config=config,
        store=store,
        exporter=exporter,
        snapshots=snapshots,
        slot=slot,
        counter=counter,
        detector_factory=factory,
        clock=clock,
    )
    try:
        pipeline.start()
        first_session = pipeline.tracking_session_id
        assert first_session is not None
        clock.set(at(7, 0, 1))
        pipeline.process_one(make_frame(1, at(7, 0, 1)))
        clock.set(at(7, 0, 2))
        pipeline.process_one(make_frame(2, at(7, 0, 2)))
        assert len(store.events(DAY)) == 1
        pipeline.on_disconnect("fake-hls reconnect")
        clock.set(at(7, 0, 3))
        pipeline.process_one(make_frame(2, at(7, 0, 3)))
        clock.set(at(7, 0, 4))
        pipeline.process_one(make_frame(2, at(7, 0, 4)))
        second_session = pipeline.tracking_session_id
        assert second_session is not None
        assert second_session != first_session
        assert len(store.events(DAY)) == 1
        clock.set(at(7, 0, 5))
        pipeline.request_stop()
        snapshot = snapshots.get()
        assert snapshot.tracking_session_changes == 1
        assert snapshot.dropped_events == 0
        assert snapshot.periods[PeriodName.PAGI].total == 1
        database_total = store.summary_counts(DAY, PeriodName.PAGI)
        assert database_total == 1
        assert len(store.events(DAY)) == 1
        export_dir = tmp_path / "exports"
        csv_files = sorted(export_dir.glob("traffic_*.csv"))
        assert len(csv_files) == 1
        csv_path = export_dir / "traffic_2026-09-25.csv"
        assert csv_path.is_file()
        with csv_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        assert rows
        csv_total = sum(int(row["count"]) for row in rows)
        assert csv_total == database_total
        pagi_rows = [row for row in rows if row["period"] == "pagi"]
        assert pagi_rows
        assert int(pagi_rows[-1]["period_total"]) == database_total
        app = create_app(snapshots, lambda: csv_path, 5000)
        client = app.test_client()
        response = client.get("/api/summary", headers={"Host": "127.0.0.1:5000"})
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["periods"]["pagi"]["total"] == 1
        status = client.get("/api/status", headers={"Host": "127.0.0.1:5000"}).get_json()
        assert status["dropped_events"] == 0
        assert status["tracking_session_changes"] == 1
    finally:
        with suppress(Exception):
            pipeline.request_stop()
        with suppress(Exception):
            store.close()
