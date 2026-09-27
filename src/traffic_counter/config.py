from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Final
from urllib.parse import urlsplit

SUPPORTED_DEVICES: Final[tuple[str, ...]] = ("cpu", "cuda:0")

_STREAM_KEYS: Final[frozenset[str]] = frozenset({"url", "width", "height"})
_MODEL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "name",
        "device",
        "confidence",
        "iou",
        "max_detections",
        "image_size",
        "tracker",
        "persist",
    }
)
_LINE_KEYS: Final[frozenset[str]] = frozenset({"x1", "y1", "x2", "y2", "hysteresis", "calibrated"})
_TIMING_KEYS: Final[frozenset[str]] = frozenset(
    {
        "target_fps",
        "reconnect_initial_seconds",
        "reconnect_max_seconds",
        "drain_seconds",
        "health_bucket_seconds",
        "rolling_min_healthy_seconds",
    }
)
_OUTPUT_KEYS: Final[frozenset[str]] = frozenset(
    {"port", "database", "export_directory", "calibration_directory", "log_file"}
)
_SECTIONS: Final[tuple[str, ...]] = ("stream", "model", "line", "timing", "output")


class ConfigError(ValueError):
    pass


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


@dataclass(frozen=True, slots=True)
class AppConfig:
    stream: StreamConfig
    model: ModelConfig
    line: LineConfig
    timing: TimingConfig
    output: OutputConfig


def load_config(path: Path = Path("config.toml")) -> AppConfig:
    source = Path(path)
    tables = _validate_sections(source, _read_document(source))
    return AppConfig(
        stream=_stream_config(tables["stream"]),
        model=_model_config(tables["model"]),
        line=_line_config(tables["line"]),
        timing=_timing_config(tables["timing"]),
        output=_output_config(tables["output"]),
    )


def _read_document(source: Path) -> Mapping[str, object]:
    if source.is_dir():
        raise ConfigError(f"Configuration path {source} is a directory instead of a file.")
    if not source.is_file():
        raise ConfigError(f"Configuration file {source} was not found.")
    try:
        payload = source.read_bytes()
    except OSError as error:
        raise ConfigError(f"Configuration file {source} could not be read.") from error
    try:
        return tomllib.loads(payload.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ConfigError(f"Configuration file {source} must be UTF-8 encoded text.") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"Configuration file {source} is not valid TOML syntax.") from error


def _validate_sections(
    source: Path, document: Mapping[str, object]
) -> Mapping[str, Mapping[str, object]]:
    tables: dict[str, Mapping[str, object]] = {}
    for name, value in document.items():
        if not isinstance(value, dict):
            raise ConfigError(
                f"Configuration file {source} must contain only approved section tables."
            )
        tables[name] = value
    unsupported = sorted(set(tables) - set(_SECTIONS))
    if unsupported:
        raise ConfigError(
            f"Configuration file {source} has an unsupported section named {unsupported[0]}."
        )
    missing = [name for name in _SECTIONS if name not in tables]
    if missing:
        raise ConfigError(
            f"Configuration file {source} is missing the required section {missing[0]}."
        )
    return tables


def _reject_unknown(section: str, values: Mapping[str, object], allowed: frozenset[str]) -> None:
    unsupported = sorted(set(values) - allowed)
    if unsupported:
        raise ConfigError(f"Section [{section}] has an unsupported field named {unsupported[0]}.")


def _text(section: str, key: str, values: Mapping[str, object], default: str) -> str:
    if key not in values:
        return default
    raw = values[key]
    if not isinstance(raw, str):
        raise ConfigError(f"Section [{section}] field {key} must be text.")
    value = raw.strip()
    if not value:
        raise ConfigError(f"Section [{section}] field {key} must not be blank.")
    return value


def _flag(section: str, key: str, values: Mapping[str, object], default: bool) -> bool:
    if key not in values:
        return default
    raw = values[key]
    if not isinstance(raw, bool):
        raise ConfigError(f"Section [{section}] field {key} must be true or false.")
    return raw


def _number(section: str, key: str, values: Mapping[str, object]) -> float:
    if key not in values:
        raise ConfigError(f"Section [{section}] field {key} is required.")
    raw = values[key]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ConfigError(f"Section [{section}] field {key} must be a number.")
    value = float(raw)
    if not math.isfinite(value):
        raise ConfigError(f"Section [{section}] field {key} must be a finite number.")
    return value


def _whole_number(
    section: str,
    key: str,
    values: Mapping[str, object],
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    if key not in values:
        return default
    raw = values[key]
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ConfigError(f"Section [{section}] field {key} must be a whole number.")
    if raw < minimum or raw > maximum:
        raise ConfigError(
            f"Section [{section}] field {key} must be a whole number "
            f"between {minimum} and {maximum}."
        )
    return raw


def _unit_fraction(
    section: str,
    key: str,
    values: Mapping[str, object],
    default: float,
    *,
    lower_inclusive: bool,
    maximum: float,
    upper_text: str,
) -> float:
    if key not in values:
        return default
    value = _number(section, key, values)
    inside_lower = value >= 0.0 if lower_inclusive else value > 0.0
    if inside_lower and value < maximum:
        return value
    lower_text = "at or above 0" if lower_inclusive else "above 0"
    raise ConfigError(f"Section [{section}] field {key} must be {lower_text} and {upper_text}.")


def _positive_seconds(
    section: str,
    key: str,
    values: Mapping[str, object],
    default: float,
) -> float:
    if key not in values:
        return default
    value = _number(section, key, values)
    if value <= 0.0:
        raise ConfigError(f"Section [{section}] field {key} must be a positive number of seconds.")
    return value


def _normalized_coordinate(section: str, key: str, values: Mapping[str, object]) -> float:
    value = _number(section, key, values)
    if 0.0 <= value <= 1.0:
        return value
    raise ConfigError(
        f"Section [{section}] field {key} must stay within normalized frame bounds from 0 to 1."
    )


def _relative_path(section: str, key: str, values: Mapping[str, object], default: Path) -> Path:
    if key not in values:
        return default
    raw = values[key]
    if not isinstance(raw, str):
        raise ConfigError(f"Section [{section}] field {key} must be a text path.")
    text = raw.strip()
    if not text:
        raise ConfigError(f"Section [{section}] field {key} must not be a blank path.")
    windows = PureWindowsPath(text)
    if windows.drive or windows.root or Path(text).is_absolute():
        raise ConfigError(
            f"Section [{section}] field {key} must be a relative path inside the project."
        )
    if any(part == ".." for part in windows.parts):
        raise ConfigError(
            f"Section [{section}] field {key} must not climb out of the project directory."
        )
    return Path(text)


def _stream_config(values: Mapping[str, object]) -> StreamConfig:
    _reject_unknown("stream", values, _STREAM_KEYS)
    url = _text("stream", "url", values, "")
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise ConfigError("Section [stream] field url must be an https address.")
    if not parts.netloc:
        raise ConfigError("Section [stream] field url must name a host.")
    return StreamConfig(
        url=url,
        width=_whole_number("stream", "width", values, 1280, 1, 8192),
        height=_whole_number("stream", "height", values, 720, 1, 8192),
    )


def _model_config(values: Mapping[str, object]) -> ModelConfig:
    _reject_unknown("model", values, _MODEL_KEYS)
    device = _text("model", "device", values, "cpu").lower()
    if device not in SUPPORTED_DEVICES:
        raise ConfigError(
            f"Section [model] field device must be one of {', '.join(SUPPORTED_DEVICES)}."
        )
    return ModelConfig(
        name=_text("model", "name", values, "yolo26n.pt"),
        device=device,
        confidence=_unit_fraction(
            "model",
            "confidence",
            values,
            0.25,
            lower_inclusive=False,
            maximum=1.0,
            upper_text="below 1",
        ),
        iou=_unit_fraction(
            "model", "iou", values, 0.70, lower_inclusive=True, maximum=1.0, upper_text="below 1"
        ),
        max_detections=_whole_number("model", "max_detections", values, 100, 1, 10000),
        image_size=_whole_number("model", "image_size", values, 640, 32, 4096),
        tracker=_text("model", "tracker", values, "bytetrack.yaml"),
        persist=_flag("model", "persist", values, True),
    )


def _line_config(values: Mapping[str, object]) -> LineConfig:
    _reject_unknown("line", values, _LINE_KEYS)
    x1 = _normalized_coordinate("line", "x1", values)
    y1 = _normalized_coordinate("line", "y1", values)
    x2 = _normalized_coordinate("line", "x2", values)
    y2 = _normalized_coordinate("line", "y2", values)
    hysteresis = _unit_fraction(
        "line",
        "hysteresis",
        values,
        0.005,
        lower_inclusive=False,
        maximum=0.5,
        upper_text="below half the normalized frame",
    )
    if math.isclose(x1, x2) and math.isclose(y1, y2):
        raise ConfigError("Section [line] endpoints must define a line with a non-zero length.")
    return LineConfig(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        hysteresis=hysteresis,
        calibrated=_flag("line", "calibrated", values, False),
    )


def _timing_config(values: Mapping[str, object]) -> TimingConfig:
    _reject_unknown("timing", values, _TIMING_KEYS)
    initial = _positive_seconds("timing", "reconnect_initial_seconds", values, 1.0)
    maximum = _positive_seconds("timing", "reconnect_max_seconds", values, 30.0)
    if maximum < initial:
        raise ConfigError(
            "Section [timing] field reconnect_max_seconds must be at least "
            "reconnect_initial_seconds."
        )
    drain = _positive_seconds("timing", "drain_seconds", values, 5.0)
    if drain > 60.0:
        raise ConfigError(
            "Section [timing] field drain_seconds must not exceed 60 seconds."
        )
    return TimingConfig(
        target_fps=_whole_number("timing", "target_fps", values, 10, 1, 240),
        reconnect_initial_seconds=initial,
        reconnect_max_seconds=maximum,
        drain_seconds=drain,
        health_bucket_seconds=_whole_number("timing", "health_bucket_seconds", values, 10, 1, 600),
        rolling_min_healthy_seconds=_whole_number(
            "timing", "rolling_min_healthy_seconds", values, 60, 1, 3600
        ),
    )


def _output_config(values: Mapping[str, object]) -> OutputConfig:
    _reject_unknown("output", values, _OUTPUT_KEYS)
    return OutputConfig(
        port=_whole_number("output", "port", values, 5000, 1024, 65535),
        database=_relative_path("output", "database", values, Path("data/traffic.db")),
        export_directory=_relative_path("output", "export_directory", values, Path("data/exports")),
        calibration_directory=_relative_path(
            "output", "calibration_directory", values, Path("data/calibration")
        ),
        log_file=_relative_path("output", "log_file", values, Path("data/traffic-counter.log")),
    )
