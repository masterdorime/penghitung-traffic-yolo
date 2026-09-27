from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from numpy.typing import NDArray

import traffic_counter.detector as detector_module
from traffic_counter.config import ModelConfig
from traffic_counter.detector import DetectorFactory, YoloDetector, adapt_tracked_boxes
from traffic_counter.models import VEHICLE_CLASSES, Frame

JAKARTA: Final[ZoneInfo] = ZoneInfo("Asia/Jakarta")
OBSERVED_AT: Final[datetime] = datetime(2026, 9, 25, 6, 0, tzinfo=JAKARTA)
ALLOWED: Final[frozenset[str]] = frozenset(VEHICLE_CLASSES)
DEFAULT_NAMES: Final[dict[int, str]] = {
    0: "person",
    1: "car",
    2: "bus",
    3: "truck",
    4: "motorcycle",
    5: "bicycle",
}
CUSTOM_NAMES: Final[dict[int, str]] = {
    0: "person",
    3: "motorcycle",
    7: "car",
    9: "truck",
    12: "bus",
}
BOX_ONE: Final[tuple[float, float, float, float]] = (10.0, 20.0, 30.0, 40.0)
BOX_TWO: Final[tuple[float, float, float, float]] = (40.0, 20.0, 70.0, 60.0)
BOX_THREE: Final[tuple[float, float, float, float]] = (0.0, 0.0, 5.0, 5.0)
ADAPT_WIDTH: Final[float] = 100.0
ADAPT_HEIGHT: Final[float] = 100.0
NORM_ONE: Final[tuple[float, float, float, float]] = (0.1, 0.2, 0.3, 0.4)
NORM_TWO: Final[tuple[float, float, float, float]] = (0.4, 0.2, 0.7, 0.6)
NORM_THREE: Final[tuple[float, float, float, float]] = (0.0, 0.0, 0.05, 0.05)
PIXEL_A: Final[tuple[float, float, float, float]] = (6.0, 4.0, 9.0, 6.0)
PIXEL_B: Final[tuple[float, float, float, float]] = (3.0, 2.0, 6.0, 4.0)
NORM_A: Final[tuple[float, float, float, float]] = (0.5, 0.5, 0.75, 0.75)
NORM_B: Final[tuple[float, float, float, float]] = (0.25, 0.25, 0.5, 0.5)
FRAME_WIDTH: Final[int] = 1280
FRAME_HEIGHT: Final[int] = 720
KOPO_BOX: Final[tuple[float, float, float, float]] = (640.0, 360.0, 960.0, 540.0)
KOPO_NORM: Final[tuple[float, float, float, float]] = (0.5, 0.5, 0.75, 0.75)


class FakeBoxes:
    def __init__(self, ids: Any, cls: Any, conf: Any, xyxy: Any) -> None:
        self.id = ids
        self.cls = cls
        self.conf = conf
        self.xyxy = xyxy


class FakeResult:
    def __init__(self, boxes: Any, names: dict[int, str]) -> None:
        self.boxes = boxes
        self.names = names


class FakeModel:
    def __init__(self, names: dict[int, str], results: Any = None) -> None:
        self.names = names
        self.calls: list[dict[str, Any]] = []
        self._results: Any = [] if results is None else results

    def track(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return self._results


def make_boxes(
    ids: Any,
    cls: Any,
    conf: Any,
    xyxy: Any,
) -> FakeBoxes:
    return FakeBoxes(ids, cls, conf, xyxy)


def make_result(
    rows: list[tuple[Any, Any, Any, Any]],
    names: dict[int, str],
) -> FakeResult:
    ids: list[Any] = [row[0] for row in rows]
    cls: list[Any] = [row[1] for row in rows]
    conf: list[Any] = [row[2] for row in rows]
    xyxy: list[Any] = [list(row[3]) for row in rows]
    return FakeResult(make_boxes(ids, cls, conf, xyxy), names)


def make_frame(marker: int = 7) -> Frame:
    image: NDArray[np.uint8] = np.zeros((8, 12, 3), dtype=np.uint8)
    image[0, 0, 0] = np.uint8(marker)
    return Frame(image=image, observed_at=OBSERVED_AT)


def make_sized_frame(height: int, width: int) -> Frame:
    image: NDArray[np.uint8] = np.zeros((height, width, 3), dtype=np.uint8)
    return Frame(image=image, observed_at=OBSERVED_AT)


def make_config(name: str = "yolo26n.pt") -> ModelConfig:
    return ModelConfig(name=name)


def test_adapter_keeps_only_named_vehicle_classes() -> None:
    result = make_result(
        [
            (1, 1, 0.91, BOX_ONE),
            (2, 0, 0.99, BOX_TWO),
            (3, 3, 0.84, BOX_THREE),
        ],
        dict(DEFAULT_NAMES),
    )
    vehicles = adapt_tracked_boxes(result, set(ALLOWED), ADAPT_WIDTH, ADAPT_HEIGHT)
    assert [vehicle.class_name for vehicle in vehicles] == ["car", "truck"]
    assert [vehicle.track_id for vehicle in vehicles] == [1, 3]
    assert vehicles[0].confidence == pytest.approx(0.91)
    assert vehicles[0].box_xyxy == pytest.approx(NORM_ONE)
    assert vehicles[1].confidence == pytest.approx(0.84)
    assert vehicles[1].box_xyxy == pytest.approx(NORM_THREE)


def test_adapter_normalizes_pixel_boxes_to_frame_coordinates() -> None:
    result = make_result([(1, 1, 0.9, KOPO_BOX)], dict(DEFAULT_NAMES))
    vehicles = adapt_tracked_boxes(result, set(ALLOWED), 1280.0, 720.0)
    assert len(vehicles) == 1
    assert vehicles[0].box_xyxy == pytest.approx(KOPO_NORM)


def test_adapter_maps_a_full_frame_box_to_unit_corners() -> None:
    full = (0.0, 0.0, 1280.0, 720.0)
    result = make_result([(2, 2, 0.8, full)], dict(DEFAULT_NAMES))
    vehicles = adapt_tracked_boxes(result, set(ALLOWED), 1280.0, 720.0)
    assert len(vehicles) == 1
    assert vehicles[0].box_xyxy == pytest.approx((0.0, 0.0, 1.0, 1.0))


@pytest.mark.parametrize(
    ("width", "height"),
    (
        (0.0, 720.0),
        (1280.0, 0.0),
        (-1280.0, 720.0),
        (1280.0, -720.0),
        (float("nan"), 720.0),
        (1280.0, float("inf")),
    ),
)
def test_adapter_rejects_non_positive_or_non_finite_dimensions(width: float, height: float) -> None:
    result = make_result([(1, 1, 0.9, KOPO_BOX)], dict(DEFAULT_NAMES))
    assert adapt_tracked_boxes(result, set(ALLOWED), width, height) == []


def test_adapter_rejects_rows_with_missing_track_ids() -> None:
    result = make_result(
        [
            (None, 1, 0.9, BOX_ONE),
            (5, 1, 0.8, BOX_TWO),
            (None, 3, 0.7, BOX_THREE),
        ],
        dict(DEFAULT_NAMES),
    )
    vehicles = adapt_tracked_boxes(result, set(ALLOWED), ADAPT_WIDTH, ADAPT_HEIGHT)
    assert [vehicle.track_id for vehicle in vehicles] == [5]
    assert [vehicle.class_name for vehicle in vehicles] == ["car"]


def test_adapter_rejects_rows_with_nan_track_ids() -> None:
    result = make_result(
        [
            (float("nan"), 1, 0.9, BOX_ONE),
            (9, 4, 0.75, BOX_TWO),
        ],
        dict(DEFAULT_NAMES),
    )
    vehicles = adapt_tracked_boxes(result, set(ALLOWED), ADAPT_WIDTH, ADAPT_HEIGHT)
    assert [vehicle.track_id for vehicle in vehicles] == [9]
    assert [vehicle.class_name for vehicle in vehicles] == ["motorcycle"]


def test_adapter_rejects_non_integral_and_negative_track_ids() -> None:
    result = make_result(
        [
            (1.9, 1, 0.9, BOX_ONE),
            (-1, 1, 0.9, BOX_TWO),
            (3, 1, 0.9, BOX_THREE),
        ],
        dict(DEFAULT_NAMES),
    )
    vehicles = adapt_tracked_boxes(result, set(ALLOWED), ADAPT_WIDTH, ADAPT_HEIGHT)
    assert [vehicle.track_id for vehicle in vehicles] == [3]
    assert [vehicle.class_name for vehicle in vehicles] == ["car"]


def test_adapter_rejects_missing_and_non_finite_confidence() -> None:
    result = make_result(
        [
            (1, 1, float("nan"), BOX_ONE),
            (2, 1, float("inf"), BOX_TWO),
            (3, 1, None, BOX_THREE),
            (4, 1, 0.85, BOX_ONE),
        ],
        dict(DEFAULT_NAMES),
    )
    vehicles = adapt_tracked_boxes(result, set(ALLOWED), ADAPT_WIDTH, ADAPT_HEIGHT)
    assert [vehicle.track_id for vehicle in vehicles] == [4]


def test_adapter_rejects_missing_and_non_finite_bbox_coordinates() -> None:
    bad_nan = (float("nan"), 20.0, 30.0, 40.0)
    bad_none: Any = (None, 20.0, 30.0, 40.0)
    result = make_result(
        [
            (1, 1, 0.9, bad_nan),
            (2, 1, 0.9, bad_none),
            (3, 1, 0.9, BOX_TWO),
        ],
        dict(DEFAULT_NAMES),
    )
    vehicles = adapt_tracked_boxes(result, set(ALLOWED), ADAPT_WIDTH, ADAPT_HEIGHT)
    assert [vehicle.track_id for vehicle in vehicles] == [3]
    assert vehicles[0].box_xyxy == pytest.approx(NORM_TWO)


def test_adapter_returns_empty_when_boxes_is_none() -> None:
    result = FakeResult(None, dict(DEFAULT_NAMES))
    assert adapt_tracked_boxes(result, set(ALLOWED), ADAPT_WIDTH, ADAPT_HEIGHT) == []


def test_adapter_returns_empty_when_track_ids_are_none() -> None:
    boxes = make_boxes(None, [1], [0.9], [list(BOX_ONE)])
    result = FakeResult(boxes, dict(DEFAULT_NAMES))
    assert adapt_tracked_boxes(result, set(ALLOWED), ADAPT_WIDTH, ADAPT_HEIGHT) == []


def test_adapter_returns_empty_when_column_lengths_disagree() -> None:
    boxes = make_boxes([1, 2], [1], [0.9, 0.8], [list(BOX_ONE), list(BOX_TWO)])
    result = FakeResult(boxes, dict(DEFAULT_NAMES))
    assert adapt_tracked_boxes(result, set(ALLOWED), ADAPT_WIDTH, ADAPT_HEIGHT) == []


def test_adapter_accepts_list_and_dict_track_ids() -> None:
    from_lists = make_result([(1, 1, 0.9, BOX_ONE)], dict(DEFAULT_NAMES))
    listed = adapt_tracked_boxes(from_lists, set(ALLOWED), ADAPT_WIDTH, ADAPT_HEIGHT)
    assert [vehicle.track_id for vehicle in listed] == [1]
    dict_boxes = make_boxes({0: 2}, [1], [0.8], [list(BOX_TWO)])
    from_dicts = FakeResult(dict_boxes, dict(DEFAULT_NAMES))
    mapped = adapt_tracked_boxes(from_dicts, set(ALLOWED), ADAPT_WIDTH, ADAPT_HEIGHT)
    assert [vehicle.track_id for vehicle in mapped] == [2]


def test_adapter_resolves_class_names_through_result_names() -> None:
    result = make_result(
        [
            (11, 7, 0.9, BOX_ONE),
            (12, 3, 0.8, BOX_TWO),
            (13, 12, 0.7, BOX_THREE),
            (14, 9, 0.6, BOX_ONE),
            (15, 0, 0.99, BOX_TWO),
        ],
        dict(CUSTOM_NAMES),
    )
    vehicles = adapt_tracked_boxes(result, set(ALLOWED), ADAPT_WIDTH, ADAPT_HEIGHT)
    assert [vehicle.track_id for vehicle in vehicles] == [11, 12, 13, 14]
    assert [vehicle.class_name for vehicle in vehicles] == [
        "car",
        "motorcycle",
        "bus",
        "truck",
    ]


@pytest.mark.parametrize(
    ("class_id", "class_name"),
    (
        (0, "person"),
        (5, "bicycle"),
    ),
)
def test_adapter_never_leaks_unapproved_names(class_id: int, class_name: str) -> None:
    result = make_result(
        [
            (1, class_id, 0.99, BOX_ONE),
            (2, 1, 0.5, BOX_TWO),
        ],
        dict(DEFAULT_NAMES),
    )
    vehicles = adapt_tracked_boxes(result, set(ALLOWED), ADAPT_WIDTH, ADAPT_HEIGHT)
    assert [vehicle.class_name for vehicle in vehicles] == ["car"]
    assert class_name not in [vehicle.class_name for vehicle in vehicles]


def test_adapter_is_case_sensitive_for_vehicle_names() -> None:
    names: dict[int, str] = {0: "Car", 1: "car"}
    result = make_result(
        [
            (1, 0, 0.99, BOX_ONE),
            (2, 1, 0.9, BOX_TWO),
        ],
        names,
    )
    vehicles = adapt_tracked_boxes(result, set(ALLOWED), ADAPT_WIDTH, ADAPT_HEIGHT)
    assert [vehicle.track_id for vehicle in vehicles] == [2]


def test_factory_loads_the_default_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_yolo(name: str) -> FakeModel:
        captured["name"] = name
        return FakeModel(dict(DEFAULT_NAMES))

    monkeypatch.setattr(detector_module, "YOLO", fake_yolo)
    detector = DetectorFactory.create(make_config("yolo26n.pt"))
    assert isinstance(detector, YoloDetector)
    assert captured["name"] == "yolo26n.pt"


def test_factory_propagates_a_custom_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_yolo(name: str) -> FakeModel:
        captured["name"] = name
        return FakeModel(dict(DEFAULT_NAMES))

    monkeypatch.setattr(detector_module, "YOLO", fake_yolo)
    DetectorFactory.create(make_config("custom-weights.pt"))
    assert captured["name"] == "custom-weights.pt"


def test_factory_rejects_incomplete_model_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incomplete: dict[int, str] = {0: "person", 1: "car", 3: "truck", 4: "motorcycle"}

    def fake_yolo(name: str) -> FakeModel:
        return FakeModel(incomplete)

    monkeypatch.setattr(detector_module, "YOLO", fake_yolo)
    with pytest.raises(ValueError, match="bus"):
        DetectorFactory.create(make_config())


def test_factory_propagates_device_to_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = dict(DEFAULT_NAMES)
    seen: list[FakeModel] = []

    def fake_yolo(name: str) -> FakeModel:
        model = FakeModel(names, [make_result([(1, 1, 0.9, PIXEL_A)], names)])
        seen.append(model)
        return model

    monkeypatch.setattr(detector_module, "YOLO", fake_yolo)
    config = ModelConfig(name="yolo26n.pt", device="cuda:0")
    detector = DetectorFactory.create(config)
    detector.detect(make_frame())
    assert seen[0].calls[0]["device"] == "cuda:0"


def test_direct_construction_accepts_partial_names_for_injected_doubles() -> None:
    partial: dict[int, str] = {0: "car", 1: "person"}
    model = FakeModel(partial, [make_result([(1, 0, 0.9, PIXEL_A)], partial)])
    detector = YoloDetector(make_config(), model)
    vehicles = detector.detect(make_frame())
    assert [vehicle.track_id for vehicle in vehicles] == [1]
    assert vehicles[0].box_xyxy == pytest.approx(NORM_A)
    assert model.calls[0]["classes"] == [0]


def test_detector_derives_filtered_classes_from_model_names() -> None:
    model = FakeModel(
        dict(CUSTOM_NAMES),
        [make_result([(1, 7, 0.9, PIXEL_A)], dict(CUSTOM_NAMES))],
    )
    config = ModelConfig(
        name="yolo26n.pt",
        confidence=0.33,
        iou=0.55,
        max_detections=7,
        image_size=320,
        tracker="botsort.yaml",
        persist=False,
    )
    detector = YoloDetector(config, model)
    detector.detect(make_frame())
    assert model.calls[0]["classes"] == [7, 3, 12, 9]


def test_detector_calls_track_once_with_exact_arguments() -> None:
    names = dict(DEFAULT_NAMES)
    result = make_result([(4, 1, 0.91, PIXEL_A)], names)
    model = FakeModel(names, [result])
    config = ModelConfig(
        name="yolo26n.pt",
        device="cuda:0",
        confidence=0.33,
        iou=0.55,
        max_detections=7,
        image_size=320,
        tracker="botsort.yaml",
        persist=False,
    )
    detector = YoloDetector(config, model)
    frame = make_frame()
    vehicles = detector.detect(frame)
    assert len(model.calls) == 1
    call = model.calls[0]
    assert set(call.keys()) == {
        "source",
        "device",
        "persist",
        "tracker",
        "conf",
        "iou",
        "classes",
        "max_det",
        "imgsz",
        "verbose",
    }
    assert call["source"] is frame.image
    assert call["device"] == "cuda:0"
    assert call["persist"] is False
    assert call["tracker"] == "botsort.yaml"
    assert call["conf"] == pytest.approx(0.33)
    assert call["iou"] == pytest.approx(0.55)
    assert call["classes"] == [1, 4, 2, 3]
    assert call["max_det"] == 7
    assert call["imgsz"] == 320
    assert call["verbose"] is False
    assert [(vehicle.track_id, vehicle.class_name) for vehicle in vehicles] == [(4, "car")]
    assert vehicles[0].box_xyxy == pytest.approx(NORM_A)


def test_detector_uses_approved_defaults_in_track_call() -> None:
    names = dict(DEFAULT_NAMES)
    model = FakeModel(names, [FakeResult(None, names)])
    detector = YoloDetector(make_config("yolo26n.pt"), model)
    detector.detect(make_frame())
    call = model.calls[0]
    assert call["device"] == "cpu"
    assert call["persist"] is True
    assert call["tracker"] == "bytetrack.yaml"
    assert call["conf"] == pytest.approx(0.25)
    assert call["iou"] == pytest.approx(0.70)
    assert call["max_det"] == 100
    assert call["imgsz"] == 640
    assert call["verbose"] is False


def test_detect_normalizes_pixel_boxes_with_frame_dimensions() -> None:
    names = dict(DEFAULT_NAMES)
    model = FakeModel(names, [make_result([(7, 1, 0.9, KOPO_BOX)], names)])
    detector = YoloDetector(make_config(), model)
    vehicles = detector.detect(make_sized_frame(FRAME_HEIGHT, FRAME_WIDTH))
    assert [vehicle.track_id for vehicle in vehicles] == [7]
    assert vehicles[0].box_xyxy == pytest.approx(KOPO_NORM)


def test_detect_rejects_a_frame_with_fewer_than_two_dimensions() -> None:
    names = dict(DEFAULT_NAMES)
    model = FakeModel(names, [make_result([(1, 1, 0.9, PIXEL_A)], names)])
    detector = YoloDetector(make_config(), model)
    image: NDArray[np.uint8] = np.zeros((12,), dtype=np.uint8)
    frame = Frame(image=image, observed_at=OBSERVED_AT)
    assert detector.detect(frame) == []
    assert model.calls == []


def test_detect_rejects_a_frame_with_non_positive_dimensions() -> None:
    names = dict(DEFAULT_NAMES)
    model = FakeModel(names, [make_result([(1, 1, 0.9, PIXEL_A)], names)])
    detector = YoloDetector(make_config(), model)
    image: NDArray[np.uint8] = np.zeros((0, 12, 3), dtype=np.uint8)
    frame = Frame(image=image, observed_at=OBSERVED_AT)
    assert detector.detect(frame) == []
    assert model.calls == []


def test_detect_returns_empty_when_tracker_reports_no_results() -> None:
    model = FakeModel(dict(DEFAULT_NAMES), [])
    detector = YoloDetector(make_config(), model)
    assert detector.detect(make_frame()) == []
    assert len(model.calls) == 1


def test_detect_returns_empty_when_tracker_output_is_none() -> None:
    model = FakeModel(dict(DEFAULT_NAMES))
    model._results = None
    detector = YoloDetector(make_config(), model)
    assert detector.detect(make_frame()) == []
    assert len(model.calls) == 1


def test_detector_calls_track_once_per_frame() -> None:
    names = dict(DEFAULT_NAMES)
    first = make_result([(1, 1, 0.9, PIXEL_B)], names)
    second = make_result([(2, 2, 0.8, PIXEL_A)], names)
    model = FakeModel(names, [first])
    detector = YoloDetector(make_config(), model)
    detector.detect(make_frame(1))
    model._results[0] = second
    out = detector.detect(make_frame(2))
    assert len(model.calls) == 2
    assert model.calls[0]["source"] is not model.calls[1]["source"]
    assert [(vehicle.track_id, vehicle.class_name) for vehicle in out] == [(2, "bus")]
    assert out[0].box_xyxy == pytest.approx(NORM_A)


def test_fresh_detectors_do_not_share_tracker_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[FakeModel] = []

    def fake_yolo(name: str) -> FakeModel:
        model = FakeModel(dict(DEFAULT_NAMES), [FakeResult(None, dict(DEFAULT_NAMES))])
        created.append(model)
        return model

    monkeypatch.setattr(detector_module, "YOLO", fake_yolo)
    config = make_config("yolo26n.pt")
    first = DetectorFactory.create(config)
    second = DetectorFactory.create(config)
    assert first is not second
    assert len(created) == 2
    assert created[0] is not created[1]
    first.detect(make_frame(1))
    assert len(created[0].calls) == 1
    assert len(created[1].calls) == 0
    second.detect(make_frame(2))
    assert len(created[0].calls) == 1
    assert len(created[1].calls) == 1


def test_detect_does_not_mutate_the_input_image() -> None:
    names = dict(DEFAULT_NAMES)
    result = make_result([(1, 1, 0.9, PIXEL_A), (2, 0, 0.99, PIXEL_B)], names)
    model = FakeModel(names, [result])
    detector = YoloDetector(make_config(), model)
    frame = make_frame(9)
    before = frame.image.tobytes()
    snapshot: NDArray[np.uint8] = frame.image.copy()
    detector.detect(frame)
    assert frame.image.tobytes() == before
    assert np.array_equal(frame.image, snapshot)


def test_detector_module_does_not_touch_storage_or_windows() -> None:
    text = Path(detector_module.__file__ or "").read_text(encoding="utf-8").lower()
    assert "sqlite" not in text
    assert "cv2" not in text
    assert "imshow" not in text
    assert "namedwindow" not in text
    assert "waitkey" not in text
