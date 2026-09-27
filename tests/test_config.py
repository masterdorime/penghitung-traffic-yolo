import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

import pytest

from traffic_counter.config import (
    AppConfig,
    ConfigError,
    LineConfig,
    ModelConfig,
    OutputConfig,
    StreamConfig,
    TimingConfig,
    load_config,
)
from traffic_counter.models import (
    AppSnapshot,
    AppState,
    CountEvent,
    Frame,
    FrameSlotStats,
    HealthBucket,
    MinutePoint,
    PeriodName,
    PeriodSummary,
    StreamDiagnostics,
    StreamState,
    TrackedVehicle,
)

JAKARTA: Final[ZoneInfo] = ZoneInfo("Asia/Jakarta")

VALID_CONFIG_TEXT: Final[str] = """
[stream]
url = "https://example.test/index.m3u8"
width = 1280
height = 720

[model]
name = "yolo26n.pt"
device = "cpu"
confidence = 0.25
iou = 0.70
max_detections = 100
image_size = 640
tracker = "bytetrack.yaml"
persist = true

[line]
x1 = 0.10
y1 = 0.80
x2 = 0.90
y2 = 0.80
hysteresis = 0.005
calibrated = false

[timing]
target_fps = 10
reconnect_initial_seconds = 1.0
reconnect_max_seconds = 30.0
drain_seconds = 5.0
health_bucket_seconds = 10
rolling_min_healthy_seconds = 60

[output]
port = 5000
database = "data/traffic.db"
export_directory = "data/exports"
calibration_directory = "data/calibration"
log_file = "data/traffic-counter.log"
"""

REJECTED_OVERRIDES: Final[tuple[tuple[str, str], ...]] = (
    ('url = "https://example.test/index.m3u8"', 'url = "http://example.test/index.m3u8"'),
    ('url = "https://example.test/index.m3u8"', 'url = "ftp://example.test/index.m3u8"'),
    ("width = 1280", "width = 0"),
    ("height = 720", "height = -720"),
    ('name = "yolo26n.pt"', 'name = "   "'),
    ('device = "cpu"', 'device = "tpu"'),
    ("confidence = 0.25", "confidence = 1.5"),
    ("confidence = 0.25", "confidence = 0.0"),
    ("iou = 0.70", "iou = 1.5"),
    ("iou = 0.70", "iou = -0.10"),
    ("max_detections = 100", "max_detections = 0"),
    ("image_size = 640", "image_size = 0"),
    ('tracker = "bytetrack.yaml"', 'tracker = ""'),
    ("persist = true", "persist = 7"),
    ("x1 = 0.10", "x1 = 1.20"),
    ("y1 = 0.80", "y1 = -0.01"),
    ("x2 = 0.90", "x2 = 0.90\nneeds_calibration = true"),
    ("y2 = 0.80", "y2 = 2.0"),
    ("x1 = 0.10", "x1 = nan"),
    ("y1 = 0.80", "y1 = inf"),
    ("x2 = 0.90", "x2 = -inf"),
    ("x1 = 0.10\ny1 = 0.80\nx2 = 0.90\ny2 = 0.80", "x1 = 0.5\ny1 = 0.5\nx2 = 0.5\ny2 = 0.5"),
    ("hysteresis = 0.005", "hysteresis = -0.01"),
    ("hysteresis = 0.005", "hysteresis = 0.5"),
    ("hysteresis = 0.005", 'hysteresis = "0.005"'),
    ("calibrated = false", 'calibrated = "yes"'),
    ("calibrated = false", "calibrated = 1"),
    ("target_fps = 10", "target_fps = 0"),
    ("reconnect_initial_seconds = 1.0", "reconnect_initial_seconds = 0.0"),
    ("reconnect_max_seconds = 30.0", "reconnect_max_seconds = 0.5"),
    ("drain_seconds = 5.0", "drain_seconds = 0.0"),
    ("drain_seconds = 5.0", 'drain_seconds = "5"'),
    ("health_bucket_seconds = 10", "health_bucket_seconds = 0"),
    ("rolling_min_healthy_seconds = 60", "rolling_min_healthy_seconds = 0"),
    ("port = 5000", "port = 0"),
    ("port = 5000", "port = 70000"),
    ('database = "data/traffic.db"', 'database = "../secrets/traffic.db"'),
    ('export_directory = "data/exports"', 'export_directory = "./exports/../.."'),
    ('calibration_directory = "data/calibration"', 'calibration_directory = ""'),
    ('log_file = "data/traffic-counter.log"', 'log_file = "logs/../../traffic.log"'),
    ('database = "data/traffic.db"', "database = 5"),
    ('log_file = "data/traffic-counter.log"', "log_file = 5"),
    ("width = 1280", 'width = "1280"'),
    ("target_fps = 10", "target_fps = 10.5"),
)

REJECTED_SECTIONS: Final[tuple[str, ...]] = (
    "[stream]",
    "[model]",
    "[line]",
    "[timing]",
    "[output]",
)

UNKNOWN_SECTION: Final[str] = "\n[overlay]\nwindow = true\n"

DUPLICATE_SECTION: Final[str] = "\n[stream]\nframe_rate = 25\n"


def config_variants() -> Iterator[str]:
    for section in REJECTED_SECTIONS:
        yield VALID_CONFIG_TEXT.replace(f"{section}\n", "")
    yield VALID_CONFIG_TEXT + UNKNOWN_SECTION
    yield VALID_CONFIG_TEXT + DUPLICATE_SECTION
    yield VALID_CONFIG_TEXT.replace("height = 720\n", "height = 720\nframe_rate = 25\n", 1)
    for old, new in REJECTED_OVERRIDES:
        yield VALID_CONFIG_TEXT.replace(old, new, 1)


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_valid_defaults_load_with_approved_values(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, VALID_CONFIG_TEXT))

    assert config == AppConfig(
        stream=StreamConfig(url="https://example.test/index.m3u8", width=1280, height=720),
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
        line=LineConfig(x1=0.10, y1=0.80, x2=0.90, y2=0.80, hysteresis=0.005, calibrated=False),
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


def test_shipped_config_file_is_valid() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config.toml")

    assert config.stream.url.startswith("https://")
    assert config.stream.url.endswith(".m3u8")
    assert config.model.name in ("yolo26n.pt", "yolo26s.pt", "yolo26m.pt")
    assert config.model.device in ("cpu", "cuda:0")
    assert config.timing.target_fps == 10
    assert isinstance(config.line.calibrated, bool)


def test_omitted_optional_values_use_approved_defaults(tmp_path: Path) -> None:
    minimal = (
        "[stream]\n"
        'url = "https://example.test/index.m3u8"\n'
        "\n[model]\n"
        'name = "yolo26n.pt"\n'
        "\n[line]\n"
        "x1 = 0.10\ny1 = 0.80\nx2 = 0.90\ny2 = 0.80\n"
        "\n[timing]\n"
        "\n[output]\n"
    )

    config = load_config(write_config(tmp_path, minimal))

    assert config.stream.width == 1280
    assert config.model.confidence == 0.25
    assert config.line.hysteresis == 0.005
    assert config.line.calibrated is False
    assert config.timing.reconnect_max_seconds == 30.0
    assert config.output.port == 5000
    assert config.output.database == Path("data/traffic.db")


def test_rejects_out_of_bounds_line(tmp_path: Path) -> None:
    path = write_config(tmp_path, VALID_CONFIG_TEXT.replace("x1 = 0.10", "x1 = 1.20", 1))

    with pytest.raises(ConfigError, match="normalized frame bounds"):
        load_config(path)


def test_accepts_inclusive_normalized_line_bounds(tmp_path: Path) -> None:
    extreme = (
        VALID_CONFIG_TEXT.replace("x1 = 0.10", "x1 = 0.0", 1)
        .replace("y1 = 0.80", "y1 = 0.0", 1)
        .replace("x2 = 0.90", "x2 = 1.0", 1)
        .replace("y2 = 0.80", "y2 = 1.0", 1)
    )

    config = load_config(write_config(tmp_path, extreme))

    assert (config.line.x1, config.line.y1, config.line.x2, config.line.y2) == (
        0.0,
        0.0,
        1.0,
        1.0,
    )


def test_out_of_bounds_line_error_names_the_field(tmp_path: Path) -> None:
    path = write_config(tmp_path, VALID_CONFIG_TEXT.replace("x2 = 0.90", "x2 = 1.01", 1))

    with pytest.raises(ConfigError, match=r"field x2 must stay within normalized frame bounds"):
        load_config(path)


def test_calibration_flag_round_trips_as_a_boolean(tmp_path: Path) -> None:
    path = write_config(
        tmp_path, VALID_CONFIG_TEXT.replace("calibrated = false", "calibrated = true", 1)
    )

    config = load_config(path)

    assert config.line.calibrated is True


def test_calibration_flag_defaults_to_false() -> None:
    assert LineConfig(x1=0.1, y1=0.8, x2=0.9, y2=0.8).calibrated is False


@pytest.mark.parametrize("text", list(config_variants()))
def test_rejects_invalid_configuration(tmp_path: Path, text: str) -> None:
    path = write_config(tmp_path, text)

    with pytest.raises(ConfigError) as excinfo:
        load_config(path)

    message = str(excinfo.value).replace(str(path), "<cfg>").replace(str(tmp_path), "<dir>")
    assert message.endswith(".")
    assert re.search(r"[.!?]\s+\S", message) is None


def test_rejects_missing_and_malformed_toml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, "[stream\nurl = 1\n"))


def test_rejects_root_table_keys(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, "version = 1\n" + VALID_CONFIG_TEXT))


def test_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.toml")


def test_rejects_configuration_directory(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path)


def test_configuration_records_are_frozen_and_slotted(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, VALID_CONFIG_TEXT))
    records = (config, config.stream, config.model, config.line, config.timing, config.output)

    for record in records:
        assert not hasattr(record, "__dict__")
        with pytest.raises(AttributeError):
            record.nonexistent = 1  # type: ignore[attr-defined]

    with pytest.raises(AttributeError):
        config.output.port = 9000  # type: ignore[misc]


def test_period_enum_values() -> None:
    assert [period.value for period in PeriodName] == ["pagi", "siang", "sore"]


def test_app_state_members() -> None:
    assert [state.value for state in AppState] == [
        "starting",
        "waiting_for_window",
        "connecting",
        "running",
        "degraded",
        "stopping",
        "stopped",
        "error",
    ]


def test_stream_state_members() -> None:
    assert [state.value for state in StreamState] == [
        "disconnected",
        "connecting",
        "connected",
        "stopped",
    ]


def test_domain_records_are_immutable() -> None:
    moment = datetime(2026, 9, 25, 6, 0, tzinfo=JAKARTA)
    later = datetime(2026, 9, 25, 6, 1, tzinfo=JAKARTA)
    records = (
        TrackedVehicle(7, "car", 0.9, (0.1, 0.2, 0.3, 0.4)),
        CountEvent(1, 7, moment, "up", PeriodName.PAGI, "car"),
        HealthBucket(moment, PeriodName.PAGI, (moment, later)),
        PeriodSummary(PeriodName.PAGI, True, 2, 120, 1.0, 2.0),
        MinutePoint(moment, PeriodName.PAGI, 1, 60),
        AppSnapshot.initial(moment),
        FrameSlotStats(2, 1),
        StreamDiagnostics(StreamState.CONNECTED, 2, 1, "boom", ("line",)),
    )

    for record in records:
        assert not hasattr(record, "__dict__")
        with pytest.raises(AttributeError):
            record.nonexistent = 1  # type: ignore[attr-defined]

    vehicle = TrackedVehicle(7, "car", 0.9, (0.1, 0.2, 0.3, 0.4))
    with pytest.raises(AttributeError):
        vehicle.track_id = 8  # type: ignore[misc]

    bucket = HealthBucket(moment, PeriodName.PAGI, (moment, later))
    with pytest.raises(AttributeError):
        bucket.seconds = ()  # type: ignore[misc]


def test_frame_record_is_immutable() -> None:
    np = pytest.importorskip("numpy")
    moment = datetime(2026, 9, 25, 6, 0, tzinfo=JAKARTA)

    frame = Frame(image=np.zeros((4, 4, 3), dtype=np.uint8), observed_at=moment)

    assert not hasattr(frame, "__dict__")
    with pytest.raises(AttributeError):
        frame.observed_at = moment  # type: ignore[misc]


def test_app_snapshot_initial_is_stopped_and_immutable() -> None:
    now = datetime(2026, 9, 25, 5, 59, tzinfo=JAKARTA)

    snapshot = AppSnapshot.initial(now)

    assert snapshot.generated_at == now
    assert snapshot.state is AppState.STOPPED
    assert snapshot.stream_state is StreamState.DISCONNECTED
    assert snapshot.actual_fps == 0.0
    assert snapshot.target_fps == 0
    assert snapshot.replaced_frames == 0
    assert snapshot.minutes == ()
    assert snapshot.csv_export_ok is False
    assert snapshot.dropped_events == 0
    assert snapshot.incomplete is False
    assert snapshot.last_error is None
    assert snapshot.tracking_session_changes == 0
    assert set(snapshot.periods) == set(PeriodName)
    assert all(not summary.started for summary in snapshot.periods.values())
    assert all(summary.total == 0 for summary in snapshot.periods.values())
    assert all(summary.observed_seconds == 0 for summary in snapshot.periods.values())
    assert all(summary.average_rate is None for summary in snapshot.periods.values())
    assert all(summary.rolling_rate is None for summary in snapshot.periods.values())
    with pytest.raises(TypeError):
        snapshot.periods["pagi"] = PeriodSummary(  # type: ignore[index]
            PeriodName.PAGI, True, 0, 0, None, None
        )
