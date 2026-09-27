from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from typing import Any

from ultralytics.models.yolo.model import YOLO

from traffic_counter.config import ModelConfig
from traffic_counter.models import VEHICLE_CLASSES, Frame, TrackedVehicle


def _column_length(value: Any) -> int | None:
    try:
        return len(value)
    except TypeError:
        return None


def _valid_dimension(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _valid_track_id(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 1.0 or not number.is_integer():
        return None
    return int(number)


def _valid_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if number < 0.0 or number > 1.0:
        return None
    return number


def _valid_box(value: Any) -> tuple[float, float, float, float] | None:
    try:
        count = len(value)
    except TypeError:
        return None
    if count != 4:
        return None
    try:
        box = (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if not all(math.isfinite(edge) for edge in box):
        return None
    left = min(max(box[0], 0.0), 1.0)
    top = min(max(box[1], 0.0), 1.0)
    right = min(max(box[2], 0.0), 1.0)
    bottom = min(max(box[3], 0.0), 1.0)
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def adapt_tracked_boxes(
    result: Any, allowed_names: Collection[str], width: Any, height: Any
) -> list[TrackedVehicle]:
    allowed = frozenset(allowed_names)
    frame_width = _valid_dimension(width)
    if frame_width is None:
        return []
    frame_height = _valid_dimension(height)
    if frame_height is None:
        return []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    ids = getattr(boxes, "id", None)
    classes = getattr(boxes, "cls", None)
    confidences = getattr(boxes, "conf", None)
    coordinates = getattr(boxes, "xyxy", None)
    if ids is None or classes is None or confidences is None or coordinates is None:
        return []
    id_count = _column_length(ids)
    cls_count = _column_length(classes)
    conf_count = _column_length(confidences)
    coord_count = _column_length(coordinates)
    if id_count is None or cls_count is None or conf_count is None:
        return []
    if coord_count is None:
        return []
    if id_count != cls_count or id_count != conf_count or id_count != coord_count:
        return []
    names = getattr(result, "names", {})
    vehicles: list[TrackedVehicle] = []
    for index in range(id_count):
        try:
            raw_id = ids[index]
            raw_cls = classes[index]
            raw_conf = confidences[index]
            raw_box = coordinates[index]
        except (IndexError, KeyError, TypeError):
            continue
        track_id = _valid_track_id(raw_id)
        if track_id is None:
            continue
        try:
            class_id = int(float(raw_cls))
        except (TypeError, ValueError, OverflowError):
            continue
        name = names.get(class_id) if isinstance(names, Mapping) else None
        if name not in allowed:
            continue
        confidence = _valid_confidence(raw_conf)
        if confidence is None:
            continue
        try:
            pixel_count = len(raw_box)
        except TypeError:
            continue
        if pixel_count != 4:
            continue
        try:
            normalized = (
                float(raw_box[0]) / frame_width,
                float(raw_box[1]) / frame_height,
                float(raw_box[2]) / frame_width,
                float(raw_box[3]) / frame_height,
            )
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        box = _valid_box(normalized)
        if box is None:
            continue
        vehicles.append(
            TrackedVehicle(
                track_id=track_id,
                class_name=str(name),
                confidence=confidence,
                box_xyxy=box,
            )
        )
    return vehicles


def _vehicle_class_ids(names: Mapping[Any, Any]) -> list[int]:
    inverted: dict[str, int] = {}
    for class_id, name in names.items():
        if isinstance(name, str) and name not in inverted:
            inverted[name] = int(class_id)
    ordered: list[int] = []
    for wanted in VEHICLE_CLASSES:
        if wanted in inverted:
            ordered.append(inverted[wanted])
    return ordered


def _require_vehicle_class_ids(names: Mapping[Any, Any]) -> list[int]:
    present = {name for name in names.values() if isinstance(name, str)}
    missing = [wanted for wanted in VEHICLE_CLASSES if wanted not in present]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Model names are missing required vehicle classes: {joined}.")
    return _vehicle_class_ids(names)


class YoloDetector:
    __slots__ = ("_allowed_names", "_class_ids", "_config", "_model")

    def __init__(self, config: ModelConfig, model: Any) -> None:
        self._config = config
        self._model = model
        self._allowed_names = frozenset(VEHICLE_CLASSES)
        names = getattr(model, "names", {})
        if isinstance(names, Mapping):
            self._class_ids = _vehicle_class_ids(names)
        else:
            self._class_ids = []

    def detect(self, frame: Frame) -> list[TrackedVehicle]:
        shape = frame.image.shape
        if len(shape) < 2 or shape[0] <= 0 or shape[1] <= 0:
            return []
        results = self._model.track(
            source=frame.image,
            device=self._config.device,
            persist=self._config.persist,
            tracker=self._config.tracker,
            conf=self._config.confidence,
            iou=self._config.iou,
            classes=self._class_ids,
            max_det=self._config.max_detections,
            imgsz=self._config.image_size,
            verbose=False,
        )
        items: list[Any] = list(results or [])
        if not items:
            return []
        return adapt_tracked_boxes(
            items[0], self._allowed_names, shape[1], shape[0]
        )


class DetectorFactory:
    __slots__ = ()

    @staticmethod
    def create(config: ModelConfig) -> YoloDetector:
        model = YOLO(config.name)
        names = getattr(model, "names", {})
        if not isinstance(names, Mapping):
            raise ValueError("Model names must include car, motorcycle, bus, and truck.")
        _require_vehicle_class_ids(names)
        return YoloDetector(config, model)
