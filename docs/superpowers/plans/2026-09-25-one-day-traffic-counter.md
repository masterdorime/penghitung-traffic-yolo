# One-Day CCTV Traffic Counter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Windows application that counts vehicles crossing a calibrated CCTV line, reports Pagi/Siang/Sore traffic rates, and exports a one-day CSV with a live overlay and dashboard.

**Architecture:** A modular Python monolith uses a bounded ffmpeg-to-OpenCV frame path, Ultralytics YOLO tracking, pure crossing/density logic, one SQLite writer, immutable UI snapshots, and a Flask API served by Waitress only on localhost. Every component is testable through injected clocks, frame sources, detectors, and storage interfaces.

**Tech Stack:** Python 3.14, Ultralytics YOLO26/ByteTrack, PyTorch, OpenCV, imageio-ffmpeg, Flask 3.1, Waitress, SQLite, TOML, HTML/CSS/JavaScript, pytest, Ruff, mypy. Deployed model profile: `yolo26s.pt` on `cuda:0` (`imgsz=960`, confidence `0.15`); CPU baseline `yolo26n.pt` (`imgsz=640`, confidence `0.25`) remains the fallback.

**Spec:** `docs/superpowers/specs/2026-09-25-traffic-counter-design.md`

## Global Constraints

- Target Windows and Python 3.14.
- Use fixed `Asia/Jakarta` periods: Pagi `[06:00,10:00)`, Siang `[10:00,15:00)`, Sore `[15:00,19:00)`.
- Count only car, motorcycle, bus, and truck; combine both directions in user-facing totals.
- Use `enum.StrEnum` for `PeriodName`, `AppState`, and `StreamState`; the project Ruff gate treats `str, Enum` as an error.
- Ship `tzdata==2026.4` because stock Windows Python may not provide the IANA database required by `ZoneInfo("Asia/Jakarta")`.
- Default to `yolo26n.pt`, CPU, `imgsz=640`, and 10 processed frames per second without silently lowering the configured rate.
- GPU operational profile (deployed): `yolo26s.pt`, `cuda:0`, confidence `0.15`, `imgsz=960`, SS Kopo line `(0.08,0.62)-(0.92,0.62)`, `calibrated=true`; CPU values remain the fallback profile.
- Store normalized crossing timestamps and linear crossing-time interpolation; inference and database timestamps are diagnostic only.
- Persist SQLite as the source of truth; derive the current-day CSV atomically.
- Retain only the current Jakarta calendar date in SQLite and downloadable CSV.
- Save no video; calibration may save one user-triggered JPEG still.
- Bind the web server unconditionally to `127.0.0.1`; trust only localhost host headers; expose read-only GET routes.
- Use no external font, chart, icon, or JavaScript CDN.
- This directory is not a Git repository. Do not initialize Git or create commits unless the user explicitly requests it.

## Review Focus

1. A crossing interpolated exactly at 10:00, 15:00, or 19:00 must enter Siang, Sore, or no period respectively.
2. ByteTrack ID reuse after reconnect must not count a vehicle merely because the tracker reset.
3. Stream disconnects inside a period must not reduce healthy seconds or inflate the average rate.
4. A locked or failed CSV replacement must not pause counting or corrupt SQLite totals.
5. Midnight rollover must delete prior-day database rows and CSV files while preserving the new day's report.

## File Map

- `pyproject.toml` — package metadata, exact runtime/test dependencies, Ruff and mypy settings.
- `config.toml` — user-editable HLS, model, line, schedule-output, timing, path, and local-port settings.
- `src/traffic_counter/__init__.py` — package version.
- `src/traffic_counter/models.py` — immutable domain records and enums shared across modules.
- `src/traffic_counter/config.py` — TOML parsing and validation.
- `src/traffic_counter/timeutils.py` — Jakarta date, period, report-minute, and clock helpers.
- `src/traffic_counter/density.py` — health accumulation, average rates, rolling rates, and summaries.
- `src/traffic_counter/geometry.py` — normalized line math, hysteresis, and crossing interpolation.
- `src/traffic_counter/counter.py` — per-track baselines, deduplication, and crossing events.
- `src/traffic_counter/state.py` — immutable snapshot publication and latest-frame replacement.
- `src/traffic_counter/storage.py` — SQLite schema, transactional writes, retention, rollups, and CSV export.
- `src/traffic_counter/stream.py` — managed ffmpeg HLS process, raw frames, backoff, and diagnostics.
- `src/traffic_counter/detector.py` — fresh Ultralytics model per tracking session and result adaptation.
- `src/traffic_counter/pipeline.py` — stream/inference/storage workers, persistence acknowledgements, and snapshots.
- `src/traffic_counter/overlay.py` — OpenCV annotation and optional live window.
- `src/traffic_counter/app.py` — daily scheduling, drain window, lifecycle transitions, and startup recovery.
- `src/traffic_counter/web.py` — Flask app factory and local JSON/CSV routes.
- `src/traffic_counter/cli.py` — run, smoke, calibration, and no-overlay entry points.
- `src/traffic_counter/templates/dashboard.html` — semantic dashboard structure.
- `src/traffic_counter/static/dashboard.css` — approved traffic-operations visual system.
- `src/traffic_counter/static/dashboard.js` — polling, semantic rendering, inline SVG chart, and export state.
- `tests/` — unit, integration, fake-stream, and application lifecycle tests.
- `setup.ps1` — create `.venv` and install the package.
- `run.ps1` — activate `.venv` and launch the CLI.

---

### Task 1: Package, Configuration, and Domain Contracts

**Files:**
- Create: `pyproject.toml`
- Create: `config.toml`
- Create: `src/traffic_counter/__init__.py`
- Create: `src/traffic_counter/models.py`
- Create: `src/traffic_counter/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `PeriodName`, `AppState`, `StreamState`, `Frame`, `TrackedVehicle`, `CountEvent`, `HealthBucket`, `PeriodSummary`, `MinutePoint`, `AppSnapshot`, `FrameSlotStats`, and `StreamDiagnostics`.
- Produces: frozen configuration records `StreamConfig`, `ModelConfig`, `LineConfig`, `TimingConfig`, `OutputConfig`, and `AppConfig`.
- Produces: `load_config(path: Path = Path("config.toml")) -> AppConfig`.
- Consumes: Python 3.14 standard-library `dataclasses`, `datetime`, `enum`, `pathlib`, and `tomllib`.

- [ ] **Step 1: Create dependency metadata and the virtual environment**

Use these exact runtime pins, which have Windows/Python 3.14-compatible wheels:

```toml
[build-system]
requires = ["setuptools>=80"]
build-backend = "setuptools.build_meta"

[project]
name = "jasamarga-traffic-counter"
version = "0.1.0"
requires-python = ">=3.14,<3.15"
dependencies = [
  "flask==3.1.3",
  "imageio-ffmpeg==0.6.0",
  "lap==0.5.13",
  "opencv-python==5.0.0.93",
  "torch==2.11.0",
  "tzdata==2026.4",
  "ultralytics==8.4.163",
  "waitress==3.0.2"
]

[project.optional-dependencies]
dev = [
  "mypy==2.3.1",
  "pytest==9.1.1",
  "ruff==0.16.9"
]

[project.scripts]
traffic-counter = "traffic_counter.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
traffic_counter = ["templates/*.html", "static/*.css", "static/*.js"]

[tool.ruff]
line-length = 100
target-version = "py314"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "RUF"]

[tool.mypy]
python_version = "3.14"
strict = true
packages = ["traffic_counter"]
ignore_missing_imports = true
```

Run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Expected: exit code 0 and editable-install output for `jasamarga-traffic-counter==0.1.0`.

- [ ] **Step 2: Write configuration tests before configuration code**

Create `tests/test_config.py` with tests that write temporary TOML files and assert:

```python
from pathlib import Path

import pytest

from traffic_counter.config import ConfigError, load_config
from traffic_counter.models import PeriodName


def valid_config_text() -> str:
    return """
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


def test_load_config_uses_approved_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(valid_config_text(), encoding="utf-8")

    config = load_config(path)

    assert config.model.name == "yolo26n.pt"
    assert config.line.hysteresis == 0.005
    assert config.timing.target_fps == 10
    assert config.output.port == 5000
    assert PeriodName.SIANG.value == "siang"


def test_load_config_rejects_out_of_bounds_line(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        valid_config_text().replace("x1 = 0.10", "x1 = 1.20"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="normalized frame bounds"):
        load_config(path)
```

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_config.py -v`

Expected: FAIL because `traffic_counter.config` does not exist.

- [ ] **Step 3: Implement immutable domain records and strict configuration validation**

Define `PeriodName` as `str, Enum` with `PAGI`, `SIANG`, and `SORE`; define `AppState` with the eight approved states. Define frozen, slotted dataclasses with these exact fields:

```python
@dataclass(frozen=True, slots=True)
class StreamConfig:
    url: str
    width: int = 1280
    height: int = 720


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str
    device: str = "cpu"
    confidence: float = 0.25
    iou: float = 0.70
    max_detections: int = 100
    image_size: int = 640
    tracker: str = "bytetrack.yaml"
    persist: bool = True


@dataclass(frozen=True, slots=True)
class LineConfig:
    x1: float
    y1: float
    x2: float
    y2: float
    hysteresis: float = 0.005
    calibrated: bool = False


@dataclass(frozen=True, slots=True)
class TimingConfig:
    target_fps: int = 10
    reconnect_initial_seconds: float = 1.0
    reconnect_max_seconds: float = 30.0
    drain_seconds: float = 5.0
    health_bucket_seconds: int = 10
    rolling_min_healthy_seconds: int = 60


@dataclass(frozen=True, slots=True)
class OutputConfig:
    port: int = 5000
    database: Path = Path("data/traffic.db")
    export_directory: Path = Path("data/exports")
    calibration_directory: Path = Path("data/calibration")
    log_file: Path = Path("data/traffic-counter.log")
```

Define the shared runtime records with these exact fields so later tasks can type-check against one contract:

```python
class StreamState(str, Enum):
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
```

`AppSnapshot.initial(now)` returns a stopped, disconnected, zero-valued snapshot with a `MappingProxyType` period mapping. Import `NDArray` from `numpy.typing`; NumPy arrives through the pinned OpenCV/PyTorch dependency graph.

Define these operational records in the same module:

```python
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
```

`AppConfig` contains `stream`, `model`, `line`, `timing`, and `output`. `load_config` must reject missing sections, unsupported extra sections, invalid numeric ranges, non-HTTPS stream URLs, nonpositive dimensions/FPS, unsupported devices, out-of-bounds line endpoints, and relative paths containing parent traversal. Validate `[line].calibrated` as a boolean and keep the shipped default `false`. Raise `ConfigError` with one actionable sentence.

- [ ] **Step 4: Add the real default configuration**

`config.toml` must use the approved stream URL, `yolo26n.pt`, CPU, 10 FPS, 640 inference size, `0.005` hysteresis, port 5000, and the specified `data/` paths. Start with a horizontal line at `(0.10, 0.80)` to `(0.90, 0.80)` and mark it as requiring calibration through the overlay rather than embedding explanatory comments in the file.

- [ ] **Step 5: Verify the package foundation**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_config.py -v
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m mypy src
```

Expected: all commands exit 0.

---

### Task 2: Jakarta Schedule, Health Buckets, and Density Math

**Files:**
- Create: `src/traffic_counter/timeutils.py`
- Create: `src/traffic_counter/density.py`
- Create: `tests/test_timeutils.py`
- Create: `tests/test_density.py`

**Interfaces:**
- Consumes: `PeriodName` and `CountEvent` from `models.py`.
- Produces: `JAKARTA = ZoneInfo("Asia/Jakarta")`, `WallClock.now()`, `report_date()`, `period_for(ts)`, and `report_minutes_through_current(now)`.
- Produces: `average_rate(total, observed_seconds)`, `rolling_rate(events, health_buckets, end, minimum_healthy_seconds)` with the rolling window clipped to the period interval, `HealthAccumulator`, and `build_period_summaries(events, health_buckets, now)`.

- [ ] **Step 1: Write boundary tests for exact period assignment**

```python
from datetime import datetime
from zoneinfo import ZoneInfo

from traffic_counter.models import PeriodName
from traffic_counter.timeutils import period_for


def test_period_boundaries_are_half_open() -> None:
    jakarta = ZoneInfo("Asia/Jakarta")
    assert period_for(datetime(2026, 9, 25, 10, 0, tzinfo=jakarta)) is PeriodName.SIANG
    assert period_for(datetime(2026, 9, 25, 15, 0, tzinfo=jakarta)) is PeriodName.SORE
    assert period_for(datetime(2026, 9, 25, 19, 0, tzinfo=jakarta)) is None
    assert period_for(datetime(2026, 9, 25, 5, 59, 59, tzinfo=jakarta)) is None
```

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_timeutils.py -v`

Expected: FAIL because `timeutils.py` does not exist.

- [ ] **Step 2: Write density tests for disconnect and null-rate behavior**

Tests must assert:

```python
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from traffic_counter.density import average_rate, rolling_rate
from traffic_counter.models import CountEvent, PeriodName


def test_disconnect_seconds_do_not_dilute_average() -> None:
    assert average_rate(60, 120) == 30.0
    assert average_rate(60, 0) is None


def test_rolling_rate_requires_minimum_healthy_time() -> None:
    end = datetime(2026, 9, 25, 6, 5, tzinfo=ZoneInfo("Asia/Jakarta"))
    events = [
        CountEvent(
            tracking_session_id=1,
            track_id=2,
            crossed_at=end - timedelta(seconds=10),
            direction="up",
            period=PeriodName.PAGI,
            class_name="car",
        )
    ]
    assert rolling_rate(events, {}, end, minimum_healthy_seconds=60) is None
```

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_density.py -v`

Expected: FAIL because `density.py` does not exist.

- [ ] **Step 3: Implement fixed Jakarta time helpers**

Use `ZoneInfo("Asia/Jakarta")`, never a fixed UTC offset. `WallClock` samples `datetime.now(JAKARTA)` and `time.monotonic()` together, caches their offset for one second, and exposes the current aware Jakarta datetime derived from the monotonic value. `period_for` converts any aware datetime to Jakarta time and returns the half-open period. `report_minutes_through_current` returns every minute bucket from 06:00 through the current minute, capped at 06:00–18:59; it returns an empty sequence before 06:00 and 780 values at or after 19:00.

- [ ] **Step 4: Implement pure density calculations and health set union**

`HealthAccumulator.mark_healthy(observed_at)` floors to the containing whole second and stores that second in a set. `drain(bucket_seconds)` groups the set into `HealthBucket` records whose `seconds` tuple preserves every exact second, then clears emitted buckets. `average_rate` returns `None` when seconds are not positive. `rolling_rate` uses `(end - 5 minutes, end]` clipped to the requested period's `[start, end)`, unions the exact seconds from all supplied buckets, and returns `None` below the configured minimum. `build_period_summaries` returns all three `PeriodSummary` records, including periods that have not started.

- [ ] **Step 5: Verify all schedule and density tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_timeutils.py tests/test_density.py -v
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m mypy src
```

Expected: all commands exit 0.

---

### Task 3: Line Geometry and Per-Track Crossing Counter

**Files:**
- Create: `src/traffic_counter/geometry.py`
- Create: `src/traffic_counter/counter.py`
- Create: `tests/test_geometry.py`
- Create: `tests/test_counter.py`

**Interfaces:**
- Consumes: `LineConfig`, `TrackedVehicle`, `PeriodName`, and `period_for`.
- Produces: `CrossingSide`, `signed_distance(point, line) -> float`, `interpolate_crossing(previous, current) -> datetime`, `LineCounter(line: LineConfig)`, and `LineCounter.observe(tracking_session_id, vehicle, observed_at) -> CountEvent | None`.
- Produces: `LineCounter.end_tracking_session(tracking_session_id)`.

- [ ] **Step 1: Write line-geometry tests**

```python
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from traffic_counter.config import LineConfig
from traffic_counter.geometry import interpolate_crossing, signed_distance


def test_signed_distance_and_interpolation() -> None:
    line = LineConfig(0.1, 0.8, 0.9, 0.8, 0.005)
    assert signed_distance((0.5, 0.7), line) < 0
    assert signed_distance((0.5, 0.9), line) > 0
    t0 = datetime(2026, 9, 25, 6, 0, tzinfo=ZoneInfo("Asia/Jakarta"))
    t1 = t0 + timedelta(milliseconds=200)
    crossing = interpolate_crossing((t0, -0.2), (t1, 0.2))
    assert crossing == t0 + timedelta(milliseconds=100)
```

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_geometry.py -v`

Expected: FAIL because `geometry.py` does not exist.

- [ ] **Step 2: Write counter tests for baseline, jitter, deduplication, and reset**

Tests must prove:

```python
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from traffic_counter.config import LineConfig
from traffic_counter.counter import LineCounter
from traffic_counter.models import TrackedVehicle


def vehicle_at(bottom_y: float) -> TrackedVehicle:
    return TrackedVehicle(
        track_id=7,
        class_name="car",
        confidence=0.9,
        box_xyxy=(0.45, bottom_y - 0.10, 0.55, bottom_y),
    )


def test_first_side_is_baseline_and_jitter_does_not_count() -> None:
    base = datetime(2026, 9, 25, 6, 0, tzinfo=ZoneInfo("Asia/Jakarta"))
    counter = LineCounter(LineConfig(0.1, 0.8, 0.9, 0.8, 0.005))
    assert counter.observe(1, vehicle_at(0.70), base) is None
    assert counter.observe(
        1,
        vehicle_at(0.804),
        base + timedelta(seconds=1),
    ) is None
    assert counter.observe(
        1,
        vehicle_at(0.70),
        base + timedelta(seconds=2),
    ) is None
    event = counter.observe(
        1,
        vehicle_at(0.90),
        base + timedelta(seconds=3),
    )
    assert event is not None
    assert event.track_id == 7
    assert event.direction == "up"
    assert counter.observe(
        1,
        vehicle_at(0.70),
        base + timedelta(seconds=4),
    ) is None
```

Add a separate test that constructs tracking session `2` with the same numeric track ID, observes one vehicle, and asserts no event because the new session has only a baseline.

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_counter.py -v`

Expected: FAIL because `counter.py` does not exist.

- [ ] **Step 3: Implement normalized line math and interpolation**

Define `CrossingSide` as an enum with `NEGATIVE`, `POSITIVE`, and `INSIDE`. Compute normalized line length and reject zero-length lines. `signed_distance` returns the normalized cross-product. `interpolate_crossing` applies:

```python
ratio = abs(previous_distance) / (abs(previous_distance) + abs(current_distance))
return previous_time + (current_time - previous_time) * ratio
```

Require opposite nonzero signs; otherwise raise `ValueError`.

- [ ] **Step 4: Implement hysteretic per-track state**

`LineCounter` stores a side latch, last stable signed distance/time, and counted flag for each `(tracking_session_id, track_id)`. It accepts only the four fixed class names, uses bbox bottom-center coordinates, never counts while inside the hysteresis band, and never emits a second event for the same pair. If the gap between stable observations exceeds two seconds, it re-baselines on the newer observation instead of interpolating. It calls `period_for(crossing_time)` and returns `None` when the interpolated event is outside the reporting schedule. Reset all state for a tracking-session ID before processing its first new frame.

- [ ] **Step 5: Verify crossing behavior**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_geometry.py tests/test_counter.py -v
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m mypy src
```

Expected: all commands exit 0.

---

### Task 4: SQLite Storage, Minute Rollups, Retention, and Atomic CSV

**Files:**
- Create: `src/traffic_counter/storage.py`
- Create: `tests/test_storage.py`
- Create: `tests/test_csv_export.py`

**Interfaces:**
- Consumes: domain records from `models.py`, `period_for`, and `report_minutes_through_current`.
- Produces: `TrafficStore` with `initialize`, `start_app_session`, `start_tracking_session`, `stop_tracking_session`, `stop_app_session`, `insert_event`, `write_health_buckets`, `refresh_minute_rollups`, `summary_counts`, `timeseries`, `purge_before`, and `close`.
- Produces: `CsvExporter.export(date, now) -> Path`, `try_export(date, now) -> bool`, and `purge_before(date)`.

- [ ] **Step 1: Write storage tests for uniqueness and retention**

```python
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from traffic_counter.models import CountEvent, PeriodName
from traffic_counter.storage import TrafficStore


def test_duplicate_tracking_pair_is_rejected(tmp_path: Path) -> None:
    store = TrafficStore(tmp_path / "traffic.db")
    store.initialize()
    started_at = datetime(2026, 9, 25, 6, 0, tzinfo=ZoneInfo("Asia/Jakarta"))
    app_session_id = store.start_app_session(started_at)
    tracking_session_id = store.start_tracking_session(app_session_id, started_at)
    event = CountEvent(
        tracking_session_id=tracking_session_id,
        track_id=9,
        crossed_at=datetime(2026, 9, 25, 6, 1, tzinfo=ZoneInfo("Asia/Jakarta")),
        direction="up",
        period=PeriodName.PAGI,
        class_name="car",
    )
    assert store.insert_event(event) is True
    assert store.insert_event(event) is False
    assert store.summary_counts(PeriodName.PAGI) == 1
    store.close()
```

Add tests that insert prior/current events and health buckets, call `TrafficStore.purge_before(current_date)`, create prior/current export filenames, call `CsvExporter.purge_before(current_date)`, and assert only current-date database rows and files remain.

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_storage.py -v`

Expected: FAIL because `storage.py` does not exist.

- [ ] **Step 2: Write CSV tests for zero-minute rows and atomic replacement**

```python
import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from traffic_counter.storage import CsvExporter, TrafficStore


def test_csv_contains_zero_minutes_and_cumulative_period_total(tmp_path: Path) -> None:
    store = TrafficStore(tmp_path / "traffic.db")
    store.initialize()
    exporter = CsvExporter(store, tmp_path / "exports")
    report_day = datetime(2026, 9, 25, tzinfo=ZoneInfo("Asia/Jakarta")).date()
    current_time = datetime(2026, 9, 25, 6, 2, tzinfo=ZoneInfo("Asia/Jakarta"))
    path = exporter.export(report_day, now=current_time)
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    assert [row["minute_start"] for row in rows] == [
        "2026-09-25T06:00:00+07:00",
        "2026-09-25T06:01:00+07:00",
        "2026-09-25T06:02:00+07:00",
    ]
    assert rows[0]["count"] == "0"
    assert rows[0]["period_total"] == "0"
    store.close()
```

Add a monkeypatched `os.replace` failure test that proves `try_export` returns `False`, leaves the previous valid CSV untouched, and succeeds on a later call.

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_csv_export.py -v`

Expected: FAIL because the storage classes do not exist.

- [ ] **Step 3: Implement the SQLite schema and transactions**

Create these tables with ISO-8601 Jakarta timestamps stored as text:

```sql
CREATE TABLE IF NOT EXISTS app_sessions (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    stopped_at TEXT,
    state TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tracking_sessions (
    id INTEGER PRIMARY KEY,
    app_session_id INTEGER NOT NULL REFERENCES app_sessions(id),
    started_at TEXT NOT NULL,
    stopped_at TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    tracking_session_id INTEGER NOT NULL REFERENCES tracking_sessions(id),
    track_id INTEGER NOT NULL,
    crossed_at TEXT NOT NULL,
    direction TEXT NOT NULL,
    period TEXT NOT NULL,
    class_name TEXT NOT NULL,
    inference_completed_at TEXT,
    inserted_at TEXT NOT NULL,
    UNIQUE(tracking_session_id, track_id)
);
CREATE TABLE IF NOT EXISTS health_buckets (
    bucket_start TEXT PRIMARY KEY,
    period TEXT NOT NULL,
    healthy_seconds INTEGER NOT NULL CHECK(healthy_seconds BETWEEN 0 AND 600),
    seconds_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS minute_buckets (
    minute_start TEXT PRIMARY KEY,
    period TEXT NOT NULL,
    event_count INTEGER NOT NULL,
    observed_seconds INTEGER NOT NULL CHECK(observed_seconds BETWEEN 0 AND 60)
);
CREATE INDEX IF NOT EXISTS events_crossed_at_idx ON events(crossed_at);
CREATE INDEX IF NOT EXISTS minute_buckets_period_idx ON minute_buckets(period, minute_start);
```

Use `sqlite3`, foreign keys on, WAL mode, explicit `with connection:` transactions, and one connection owned by the storage worker.

- [ ] **Step 4: Implement rollups, retention, and CSV generation**

`write_health_buckets` stores `seconds_json` as sorted ISO-8601 second values so partial rolling windows remain exact. `refresh_minute_rollups` materializes every minute returned by `report_minutes_through_current`, including zero rows, and keeps a completed minute writable for a two-minute settle margin so the two-second crossing interpolation gap cannot produce a late event after the minute is first exported. It uses the latest existing minute as its cursor, reads recent health buckets only, and keeps the full event query to catch late events. `TrafficStore.purge_before` deletes dependent rows before their parent sessions. `CsvExporter.purge_before` deletes only `traffic_*.csv` filenames whose date differs from the supplied current Jakarta date. `CsvExporter` writes a sibling `.tmp` file, flushs and closes it, then calls `os.replace`; it calculates `minute_rate`, cumulative `period_total`, cumulative observed seconds, and `period_rate` without writing rolling-rate data to CSV.

- [ ] **Step 5: Verify persistence invariants**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_storage.py tests/test_csv_export.py -v
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m mypy src
```

Expected: all commands exit 0.

---

### Task 5: Immutable State and Latest-Frame Backpressure

**Files:**
- Create: `src/traffic_counter/state.py`
- Create: `tests/__init__.py`
- Create: `tests/helpers.py`
- Create: `tests/test_state.py`

**Interfaces:**
- Produces: `SnapshotStore.get() -> AppSnapshot` and `SnapshotStore.publish(snapshot: AppSnapshot) -> None` under a short lock.
- Produces: `LatestFrameSlot.put(frame: Frame) -> None`, `take() -> Frame | None`, and `stats() -> FrameSlotStats` with replaced-frame and received-frame counters.
- Consumes: immutable records from `models.py`.

- [ ] **Step 1: Write state replacement tests**

```python
from traffic_counter.state import LatestFrameSlot
from tests.helpers import make_frame


def test_latest_frame_slot_replaces_stale_frame() -> None:
    slot = LatestFrameSlot()
    first = make_frame(1)
    second = make_frame(2)
    slot.put(first)
    slot.put(second)
    assert slot.take() == second
    assert slot.stats().received == 2
    assert slot.stats().replaced == 1
```

Add a snapshot immutability test that publishes one snapshot, replaces it atomically, and confirms the first object never changes.

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_state.py -v`

Expected: FAIL because `state.py` does not exist.

- [ ] **Step 2: Implement the snapshot store and latest-frame slot**

Use `threading.Lock` only around pointer replacement/copy-out. Never expose mutable dictionaries: `AppSnapshot.periods`, `minutes`, diagnostics, and status values are immutable mappings or tuples. `LatestFrameSlot.take` clears the pending slot so the next producer write remains the newest frame. In `tests/helpers.py`, implement `make_frame(marker)` as a zero-filled `uint8` NumPy array with `marker` written into pixel zero plus a fixed aware Jakarta `observed_at`, so equality checks are deterministic.

- [ ] **Step 3: Verify concurrency primitives**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_state.py -v
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m mypy src
```

Expected: all commands exit 0.

---

### Task 6: Resilient HLS Frame Source

**Files:**
- Create: `src/traffic_counter/stream.py`
- Create: `tests/test_stream.py`

**Interfaces:**
- Consumes: `StreamConfig`, `TimingConfig`, `Frame`, and `LatestFrameSlot`.
- Consumes as a hard contract: the `on_disconnect(error)` callback runs on the producer thread after the source records the failure and before it waits to retry. Task 8 uses it to stop the detector and call `LineCounter.end_tracking_session(old_session_id)` before any replacement frame is observed.
- Produces: `build_ffmpeg_command(executable, url, width, height, target_fps=10) -> list[str]`, `HlsFrameSource.frames() -> Iterator[Frame]`, `stop()`, `diagnostics() -> StreamDiagnostics`, and an `on_disconnect` callback invoked after the source degrades and before its retry wait so Task 8 can reset the detector/counter session before replacement frames are observed.
- Uses: `imageio_ffmpeg.get_ffmpeg_exe()` and a managed `subprocess.Popen` raw-video pipe.

- [ ] **Step 1: Write ffmpeg command and raw-frame tests**

```python
from traffic_counter.stream import build_ffmpeg_command


def test_ffmpeg_command_uses_fixed_dimensions_and_raw_bgr() -> None:
    command = build_ffmpeg_command(
        executable="ffmpeg.exe",
        url="https://example.test/index.m3u8",
        width=640,
        height=360,
    )
    assert command[:2] == ["ffmpeg.exe", "-hide_banner"]
    assert "-reconnect_streamed" in command
    assert "fps=10,scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2:black" in command
    assert command[-5:] == ["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
```

Add a fake-process test that returns one exact `width * height * 3` byte payload and then raises `StreamDisconnected`; assert one frame is yielded with an aware Jakarta timestamp and the next call reconnects.

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_stream.py -v`

Expected: FAIL because `stream.py` does not exist.

- [ ] **Step 2: Implement bounded ffmpeg process management**

Build this argument order: global flags, a browser `-user_agent` (the origin rejects ffmpeg's default agent on segment requests), HLS reconnect flags, `-i URL`, `-an`, fixed-size `fps/scale/pad` filter using configured target FPS, raw BGR output, and `pipe:1`. Read exact frame-byte chunks, convert with NumPy, wrap in `Frame`, and replace any partial trailing buffer as a disconnect. Capture only the final 20 stderr lines for sanitized diagnostics.

- [ ] **Step 3: Implement capped jittered reconnect**

Retry delays start at one second, double to 30 seconds maximum, and use deterministic injected jitter in tests. On reconnect, publish `connecting` before process start and `degraded` after failure. `stop()` closes stdin, terminates the process, waits five seconds, then kills it if needed. Do not catch `KeyboardInterrupt` inside the generator.

- [ ] **Step 4: Verify stream behavior without the live network**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_stream.py -v
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m mypy src
```

Expected: all commands exit 0.

---

### Task 7: Ultralytics Tracking Adapter

**Files:**
- Create: `src/traffic_counter/detector.py`
- Create: `tests/test_detector.py`

**Interfaces:**
- Consumes: `ModelConfig`, `Frame`, and `TrackedVehicle`.
- Produces: `YoloDetector.detect(frame) -> list[TrackedVehicle]` and `DetectorFactory.create() -> YoloDetector`.
- Produces: `adapt_tracked_boxes(result, allowed_names, width, height) -> list[TrackedVehicle]` as a pure test seam.

- [ ] **Step 1: Write model-result adaptation tests**

```python
from traffic_counter.detector import adapt_tracked_boxes


def test_adapter_keeps_only_named_vehicle_classes() -> None:
    result = fake_result(
        rows=[
            (1, "car", 0.91, (128, 144, 384, 288)),
            (2, "person", 0.99, (256, 144, 512, 360)),
            (3, "truck", 0.84, (512, 144, 896, 432)),
        ]
    )
    vehicles = adapt_tracked_boxes(result, {"car", "motorcycle", "bus", "truck"}, 1280, 720)
    assert [vehicle.class_name for vehicle in vehicles] == ["car", "truck"]
    assert [vehicle.track_id for vehicle in vehicles] == [1, 3]
    assert vehicles[0].box_xyxy == (0.1, 0.2, 0.3, 0.4)
```

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_detector.py -v`

Expected: FAIL because `detector.py` does not exist.

- [ ] **Step 2: Implement fresh-session model creation**

`DetectorFactory.create` constructs `YOLO(config.model.name)`, passes `device=config.model.device` to tracking, derives class IDs from `model.names` for the fixed four names, and returns a detector with `persist=True`. A new detector instance is created for every tracking session so ByteTrack state cannot leak across reconnects. Direct `YoloDetector` construction is for dependency-injected tests only; production code must use `DetectorFactory.create`. The factory raises if any of the four required names is absent.

- [ ] **Step 3: Implement per-frame tracking**

Call:

```python
result = self._model.track(
    source=frame.image,
    persist=self._config.persist,
    tracker=self._config.tracker,
    conf=self._config.confidence,
    iou=self._config.iou,
    classes=self._class_ids,
    max_det=self._config.max_detections,
    imgsz=self._config.image_size,
    device=self._config.device,
    verbose=False,
)
results = list(result or [])
if not results:
    return []
first = results[0]
```

Adapt `boxes.id`, `boxes.cls`, `boxes.conf`, and `boxes.xyxy`; discard any row with no track ID or a non-allowed name. The module must not own OpenCV display or SQLite.

- [ ] **Step 4: Verify detector isolation**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_detector.py -v
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m mypy src
```

Expected: all commands exit 0.

---

### Task 8: Processing Pipeline and Storage Worker

**Files:**
- Create: `src/traffic_counter/pipeline.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `HlsFrameSource`, `DetectorFactory`, `LineCounter`, `TrafficStore`, `SnapshotStore`, optional `OverlayRenderer`, and injected clock/sleep functions.
- Consumes as a hard contract: `LineCounter.end_tracking_session(tracking_session_id)` must be invoked on every reconnect, every tracker reset, and at the end of the 19:00 drain, with the outgoing session id, before any observation of the replacement session. `LineCounter` retains ~220 bytes of state per track id for the lifetime of a session and evicts nothing on its own, so this call is the sole bound on both stale-latch leakage and memory growth.
- Produces: `TrafficPipeline.start()`, `process_one(frame: Frame)`, `run_until(end_at)`, `request_stop()`, `drain_for(seconds)`, and `accepting_counts() -> bool`.
- Produces: `StorageWorker.submit(command) -> StorageResult`, where `StorageResult` carries success and a sanitized error.

- [ ] **Step 1: Write ordering and failure tests with fakes**

```python
def test_snapshot_total_is_published_only_after_storage_ack() -> None:
    pipeline, storage, snapshots = build_pipeline_with_fake_storage()
    pipeline.process_one(frame())
    assert snapshots.get().periods[PeriodName.PAGI].total == 1
    assert storage.inserted_events == 1


def test_database_failure_marks_data_incomplete_and_stops() -> None:
    pipeline, storage, snapshots = build_pipeline_with_failing_storage()
    pipeline.process_one(frame())
    assert snapshots.get().state is AppState.ERROR
    assert snapshots.get().dropped_events == 1
    assert not pipeline.accepting_counts()
```

Add a test that a CSV export failure sets `csv_export_ok=False` while the next accepted count still persists and the lifecycle remains `running`.

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -v`

Expected: FAIL because `pipeline.py` does not exist.

- [ ] **Step 2: Implement typed storage commands and one writer thread**

Define frozen command records for `InsertEvent`, `WriteHealthBuckets`, `RefreshRollups`, `ExportCsv`, `Purge`, and `Stop`. Each submission carries a `threading.Event`, result slot, and error slot. The writer loop executes commands in FIFO order and always signals completion in `finally`. A CSV command catches `OSError` and returns failure; an event command catches `sqlite3.Error` and returns failure.

- [ ] **Step 3: Implement processing order and health accounting**

For each frame: detect, call `LineCounter.end_tracking_session(previous_tracking_session_id)` exactly once for the outgoing session whenever the detector was recreated (reconnect or tracker reset) before observing any vehicle of the new session, observe each tracked vehicle, submit crossing events and wait for acknowledgement, mark the frame's wall-clock second healthy after inference completes, publish a new immutable snapshot no more than once per second, and render the optional overlay. Never publish an event total before its storage acknowledgement. A failed event command increments `dropped_events`, marks the snapshot incomplete, and transitions to `error`. Keep frame FPS measurement independent from snapshot publication rate. A test must assert that `end_tracking_session` is called with the *outgoing* session id, that no observation of the new session is processed before that call, and that a session reset lets the same numeric `track_id` establish a fresh baseline instead of inheriting the old latch.

- [ ] **Step 4: Implement reconnect and drain behavior**

When `StreamDisconnected` occurs, stop the active detector, call `LineCounter.end_tracking_session(old_session_id)` for the outgoing session before any observation of the replacement session — this is the only thing that discards per-track latches, per-track counted flags, and the retained per-track state (~220 bytes per track, ~10.5 MiB at 50 000 track ids), so skipping it both mis-attributes crossings and leaks memory across a day of reconnects — begin a new tracking session on the next connection, and keep existing SQLite events. The same call is required when the tracker is reset without a disconnect, and also at the 19:00 drain's end so the day's state is released. From 19:00, close schedule admission only after continuing to feed observations through the drain so a crossing interpolated before 19:00 can be admitted; reject event timestamps at or after 19:00, flush health/rollups/CSV, and then stop the stream. Drain time never contributes healthy seconds.

- [ ] **Step 5: Verify pipeline behavior**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -v
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m mypy src
```

Expected: all commands exit 0.

---

### Task 9: Daily Lifecycle, Overlay, CLI, and Calibration

**Files:**
- Create: `src/traffic_counter/overlay.py`
- Create: `src/traffic_counter/app.py`
- Create: `src/traffic_counter/cli.py`
- Create: `tests/test_app.py`
- Create: `tests/test_overlay.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Produces: `DailyController.run_forever()`, `DailyController.run_for_day()`, and `DailyController.recover()` with injected `clock` and `sleep` dependencies.
- Produces: `OverlayRenderer.render(frame, vehicles, line, summaries, state) -> ndarray` and `show_once(window_name)`.
- Produces: CLI flags `--config`, `--no-overlay`, `--calibrate`, and `--smoke-seconds`.

- [ ] **Step 1: Write deterministic lifecycle tests**

```python
def test_controller_starts_at_six_and_drains_at_nineteen() -> None:
    clock = FakeClock("2026-09-25T05:59:59+07:00")
    pipeline = FakePipeline()
    controller = DailyController(pipeline, clock=clock)
    controller.run_for_day()
    assert pipeline.started_at is None
    clock.set("2026-09-25T06:00:00+07:00")
    controller.run_for_day()
    assert pipeline.started_at == clock.now()
    clock.set("2026-09-25T19:00:00+07:00")
    controller.run_for_day()
    assert pipeline.drain_seconds == 5
    assert pipeline.stopped is True
```

Add tests for waiting after 19:00, next-day recovery purge, Ctrl+C request, and a database error preventing automatic restart.

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_app.py -v`

Expected: FAIL because the lifecycle modules do not exist.

- [ ] **Step 2: Implement overlay rendering without display side effects**

`render` copies the frame, draws the normalized line, vehicle boxes and IDs, lifecycle badge, current period, and combined total. It must use fixed text positions derived from frame dimensions and return a new BGR image. `show_once` is the only function that calls `cv2.imshow` or `cv2.waitKey`, making headless tests possible.

- [ ] **Step 3: Implement daily scheduling and recovery**

Before 06:00 publish `waiting_for_window`. At 06:00 initialize current-day storage, publish `connecting`, and run the pipeline until 19:00 plus drain. At 19:05 publish `stopped` and sleep in bounded intervals. `recover` purges noncurrent dates, regenerates the current CSV when present, and returns a snapshot without deleting current events. `KeyboardInterrupt` requests a graceful stop and final export.

- [ ] **Step 4: Implement CLI and calibration still capture**

Normal mode wires configuration, logging, storage, state, detector, stream, pipeline, overlay, and Waitress. `--no-overlay` suppresses OpenCV display. `--calibrate` starts the stream outside reporting schedule, saves one timestamped JPEG under the configured calibration directory, sets `[line].calibrated = true` in the configuration through a small atomic TOML update, and exits. `--smoke-seconds N` bypasses schedule for at most N seconds, prints JSON diagnostics, stops cleanly, and never records a crossing as production output.

- [ ] **Step 5: Verify lifecycle and CLI behavior**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_app.py tests/test_overlay.py tests/test_cli.py -v
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m mypy src
```

Expected: all commands exit 0.

---

### Task 10: Local Flask API and CSV Download

**Files:**
- Create: `src/traffic_counter/web.py`
- Create: `tests/test_web.py`

**Interfaces:**
- Consumes: `SnapshotStore` and a callable returning the current CSV `Path`.
- Produces: `create_app(snapshot_store, csv_path_provider, port) -> Flask`.
- Routes: `/`, `/api/status`, `/api/summary`, `/api/timeseries`, `/api/export.csv`.

- [ ] **Step 1: Write API contract and host-security tests**

```python
from traffic_counter.web import create_app


def test_rejects_untrusted_host(test_client) -> None:
    response = test_client.get("/api/status", headers={"Host": "evil.test"})
    assert response.status_code == 400


def test_summary_comes_from_immutable_snapshot(test_client, snapshot_store) -> None:
    response = test_client.get("/api/summary")
    assert response.status_code == 200
    assert response.get_json()["pagi"]["total"] == 2


def test_export_returns_404_before_first_generation(test_client) -> None:
    response = test_client.get("/api/export.csv")
    assert response.status_code == 404
    assert response.get_json()["error"] == "csv_not_generated"
```

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_web.py -v`

Expected: FAIL because `web.py` does not exist.

- [ ] **Step 2: Implement the Flask app factory and local trust boundary**

Set `app.config["TRUSTED_HOSTS"] = ["127.0.0.1", "localhost", "::1"]`; Werkzeug ignores the port for this check. Register only GET routes, return `Cache-Control: no-store` on JSON, and never access SQLite. Render `dashboard.html` for `/`. Serialize dataclass snapshots with explicit dictionaries so internal fields and paths are not leaked.

- [ ] **Step 3: Implement summary, time-series, status, and export responses**

`/api/summary` always returns all three period records. `/api/timeseries` returns `date`, `timezone`, and ascending minute objects. `/api/status` includes lifecycle, source state, actual/target FPS, replaced frames, snapshot age, `csv_export_ok`, `dropped_events`, `incomplete`, `tracking_session_changes`, and sanitized error. `/api/export.csv` resolves the provider's directory and filename, then calls `send_from_directory(str(directory), filename, as_attachment=True)`; it returns a 404 JSON error when the file is absent.

- [ ] **Step 4: Verify API contracts and security**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_web.py -v
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m mypy src
```

Expected: all commands exit 0.

---

### Task 11: Traffic-Operations Dashboard Frontend

**Files:**
- Create: `src/traffic_counter/templates/dashboard.html`
- Create: `src/traffic_counter/static/dashboard.css`
- Create: `src/traffic_counter/static/dashboard.js`
- Create: `tests/test_dashboard_assets.py`

**Interfaces:**
- Consumes: the Task 10 JSON routes.
- Produces DOM IDs `#app-state`, `#stream-state`, `#actual-fps`, `#snapshot-age`, `#period-cards`, `#traffic-chart`, `#timeseries-table`, `#export-link`, `#data-warning`, and `#last-updated`.
- Produces: no globals except namespaced `window.trafficDashboard`.

- [ ] **Step 1: Write static semantic-contract tests**

```python
from pathlib import Path


def test_dashboard_has_accessible_fallbacks_and_no_cdn() -> None:
    html = Path("src/traffic_counter/templates/dashboard.html").read_text(encoding="utf-8")
    css = Path("src/traffic_counter/static/dashboard.css").read_text(encoding="utf-8")
    script = Path("src/traffic_counter/static/dashboard.js").read_text(encoding="utf-8")
    assert 'aria-live="polite"' in html
    assert "<table" in html
    assert "https://" not in html
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert "prefers-color-scheme" in css
    assert "fetch('/api/summary')" in script
```

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_dashboard_assets.py -v`

Expected: FAIL because frontend files do not exist.

- [ ] **Step 2: Build the v0 semantic layout**

Use this order: skip link; header with lifecycle/stream badges and local timestamp; warning region; three period cards; SVG chart with an accessible heading; diagnostics strip; semantic minute table inside a scroll container; CSV export link. Status meaning must use text and shape in addition to color.

- [ ] **Step 3: Implement the approved visual system**

Define CSS custom properties for navy surfaces, text, amber primary, green healthy, red error, and neutral grid lines. Use Bahnschrift headings, Segoe UI body, and Consolas metrics. Use an 8px/4px spacing rhythm, 6px radius, hairline borders, one subtle shadow level, visible `:focus-visible` rings, responsive grids at 900px and 640px, dark/light schemes, and reduced-motion overrides. Do not add gradients, glass effects, emoji, external assets, or icon placeholders.

- [ ] **Step 4: Implement polling, chart, states, and export behavior**

`window.trafficDashboard.init()` polls status, summary, and timeseries once per second with non-overlapping `setTimeout` scheduling. It renders null rates as an em dash, formats integers with `Intl.NumberFormat("id-ID")`, and creates the chart as inline SVG paths with a table fallback. On fetch failure, retain the last good snapshot, show `data-warning`, and mark the state stale; do not replace totals with zeros.

- [ ] **Step 5: Render the v0 and pause for visual approval**

Run the app with fake snapshot data and render the dashboard in a browser. Show the v0 to the user with the three period cards, zero/loading/error states, responsive layout, and chart fallback. Wait for explicit approval before continuing to Task 12; do not silently proceed past this checkpoint.

- [ ] **Step 6: Verify frontend contracts**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_dashboard_assets.py tests/test_web.py -v
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m mypy src
```

Expected: all commands exit 0.

---

### Task 12: End-to-End Fake Run, Live Smoke, and Windows Launchers

**Files:**
- Create: `tests/test_end_to_end.py`
- Create: `setup.ps1`
- Create: `run.ps1`

**Interfaces:**
- Consumes: all application components through public constructors.
- Produces: one deterministic fake-HLS end-to-end test and `python -m traffic_counter.cli --smoke-seconds 30 --no-overlay` verification.
- Produces: `setup.ps1` and `run.ps1` for Windows.

- [ ] **Step 1: Write the fake-HLS end-to-end test**

The test must generate raw BGR frames in memory, feed a scripted YOLO adapter, cross one vehicle, inject a reconnect between two frames, generate one current-day CSV, query Flask through the test client, and assert:

```python
assert response.get_json()["pagi"]["total"] == 1
assert csv_rows[-1]["period_total"] == "1"
assert snapshots.get().dropped_events == 0
assert snapshots.get().tracking_session_changes == 1
```

The reconnect's first new track must establish a baseline and cannot create a second event.

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_end_to_end.py -v`

Expected: FAIL until all components are wired.

- [ ] **Step 2: Add Windows launch scripts**

`setup.ps1` must fail clearly if Python 3.14 is unavailable, create `.venv`, and run `python -m pip install -e ".[dev]"`. `run.ps1` must fail if `.venv` is absent, then execute:

```powershell
.\.venv\Scripts\python.exe -m traffic_counter.cli --config config.toml @args
```

Use `Set-StrictMode -Version Latest` and `$ErrorActionPreference = "Stop"` in both scripts.

- [ ] **Step 3: Run the complete local verification suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m mypy src
```

Expected: all tests pass and both static checks exit 0.

- [ ] **Step 4: Run the real HLS and model smoke test**

Run outside production counting:

```powershell
.\.venv\Scripts\python.exe -m traffic_counter.cli --config config.toml --smoke-seconds 30 --no-overlay
```

Expected: the command downloads or loads `yolo26n.pt`, connects through bundled ffmpeg, decodes 720p frames, processes frames, opens the local API, reports measured FPS and dropped-frame count, and exits 0 without database event writes.

- [ ] **Step 5: Verify the one-day operational sequence**

1. Run `.\run.ps1 --no-overlay` before 06:00 and confirm the dashboard shows `waiting_for_window`.
2. At 06:00 confirm lifecycle changes through `connecting` to `running` and actual FPS is visible.
3. Trigger one controlled network interruption and confirm `degraded` then automatic recovery.
4. At 19:00 confirm five-second drain, final CSV generation, and `stopped` state.
5. Download `traffic_YYYY-MM-DD.csv` before midnight and compare dashboard, SQLite, and CSV totals.
6. On the next date, confirm old rows/files are removed only after the new Jakarta date begins.

- [ ] **Step 6: Run the approved accuracy calibration**

1. Calibrate the line on a current still frame.
2. Capture a representative 10-minute sample.
3. Manually count the same crossings.
4. If the sample contains fewer than 30 crossings, extend it up to 30 minutes.
5. Compute `abs(app_total - manual_total) / manual_total`.
6. Adjust the line or thresholds and repeat until the result is no greater than `0.10` for a sample with at least 30 manual crossings.

- [ ] **Step 7: Re-run verification after operational fixes**

Run the complete pytest, Ruff, and mypy commands again. Expected: every command exits 0 before declaring the build complete.
