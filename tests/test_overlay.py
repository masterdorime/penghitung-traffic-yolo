from __future__ import annotations

import inspect
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from numpy.typing import NDArray

from traffic_counter.config import LineConfig
from traffic_counter.models import AppState, Frame, PeriodName, PeriodSummary, TrackedVehicle

JAKARTA = ZoneInfo("Asia/Jakarta")


def make_line() -> LineConfig:
    return LineConfig(x1=0.10, y1=0.80, x2=0.90, y2=0.80, hysteresis=0.005)


def make_frame(width: int = 320, height: int = 240) -> Frame:
    image: NDArray[np.uint8] = np.zeros((height, width, 3), dtype=np.uint8)
    moment = datetime(2026, 9, 25, 7, 0, tzinfo=JAKARTA)
    return Frame(image=image, observed_at=moment)


def make_vehicle() -> TrackedVehicle:
    return TrackedVehicle(
        track_id=7,
        class_name="car",
        confidence=0.9,
        box_xyxy=(0.45, 0.70, 0.55, 0.90),
    )


def make_summaries() -> Mapping[PeriodName, PeriodSummary]:
    return {
        PeriodName.PAGI: PeriodSummary(PeriodName.PAGI, True, 3, 120, 1.5, 1.0),
        PeriodName.SIANG: PeriodSummary(PeriodName.SIANG, False, 0, 0, None, None),
        PeriodName.SORE: PeriodSummary(PeriodName.SORE, False, 0, 0, None, None),
    }


def test_render_returns_new_copy_without_mutating_input() -> None:
    from traffic_counter.overlay import OverlayRenderer

    renderer = OverlayRenderer()
    frame = make_frame()
    before: NDArray[np.uint8] = frame.image.copy()
    out = renderer.render(
        frame,
        [make_vehicle()],
        make_line(),
        make_summaries(),
        AppState.RUNNING,
        calibrated=True,
    )
    assert isinstance(out, np.ndarray)
    assert out.shape == frame.image.shape
    assert out.dtype == np.uint8
    assert out is not frame.image
    assert np.array_equal(frame.image, before)


def test_render_draws_line_boxes_and_status_text() -> None:
    from traffic_counter.overlay import OverlayRenderer

    renderer = OverlayRenderer()
    frame = make_frame()
    out = renderer.render(
        frame,
        [make_vehicle()],
        make_line(),
        make_summaries(),
        AppState.RUNNING,
        calibrated=True,
    )
    assert np.any(out != 0)


def test_render_empty_vehicles_still_draws_line_and_state() -> None:
    from traffic_counter.overlay import OverlayRenderer

    renderer = OverlayRenderer()
    frame = make_frame()
    out = renderer.render(
        frame,
        [],
        make_line(),
        make_summaries(),
        AppState.WAITING_FOR_WINDOW,
        calibrated=True,
    )
    assert np.any(out != 0)


def test_render_calibration_warning_only_when_uncalibrated() -> None:
    from traffic_counter.overlay import OverlayRenderer

    renderer = OverlayRenderer()
    frame = make_frame()
    warned = renderer.render(
        frame,
        [],
        make_line(),
        make_summaries(),
        AppState.RUNNING,
        calibrated=False,
    )
    clean = renderer.render(
        frame,
        [],
        make_line(),
        make_summaries(),
        AppState.RUNNING,
        calibrated=True,
    )
    assert np.any(warned != 0)
    assert not np.array_equal(warned, clean)


def test_render_total_changes_with_summaries() -> None:
    from traffic_counter.overlay import OverlayRenderer

    renderer = OverlayRenderer()
    frame = make_frame()
    base = make_summaries()
    first = renderer.render(frame, [], make_line(), base, AppState.RUNNING, calibrated=True)
    bigger: Mapping[PeriodName, PeriodSummary] = {
        PeriodName.PAGI: PeriodSummary(PeriodName.PAGI, True, 99, 120, 49.5, 10.0),
        PeriodName.SIANG: PeriodSummary(PeriodName.SIANG, False, 0, 0, None, None),
        PeriodName.SORE: PeriodSummary(PeriodName.SORE, False, 0, 0, None, None),
    }
    second = renderer.render(frame, [], make_line(), bigger, AppState.RUNNING, calibrated=True)
    assert not np.array_equal(first, second)


def test_show_once_matches_brief_signature() -> None:
    from traffic_counter.overlay import OverlayRenderer

    params = list(inspect.signature(OverlayRenderer.show_once).parameters.values())
    assert [p.name for p in params] == ["self", "window_name"]
    assert params[1].default == "traffic-counter"


def test_show_once_is_only_display_call() -> None:
    path = Path(__file__).resolve().parents[1] / "src" / "traffic_counter" / "overlay.py"
    text = path.read_text(encoding="utf-8")
    assert text.count("imshow") == 1
    assert text.count("waitKey") == 1
    assert "show_once" in text
    lines = text.splitlines()
    holder = [i for i, line in enumerate(lines) if "def show_once" in line]
    assert len(holder) == 1
    start = holder[0]
    imshow_line = next(i for i, line in enumerate(lines) if "imshow" in line)
    waitkey_line = next(i for i, line in enumerate(lines) if "waitKey" in line)
    assert imshow_line > start
    assert waitkey_line > start


def test_show_once_uses_stored_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    from traffic_counter.overlay import OverlayRenderer

    seen: list[tuple[str, object]] = []

    def fake_imshow(name: str, img: object) -> None:
        seen.append((name, img))

    def fake_waitkey(delay: int) -> int:
        return 0

    monkeypatch.setattr("traffic_counter.overlay.cv2.imshow", fake_imshow)
    monkeypatch.setattr("traffic_counter.overlay.cv2.waitKey", fake_waitkey)
    renderer = OverlayRenderer()
    frame = make_frame()
    renderer.render(frame, [], make_line(), make_summaries(), AppState.RUNNING, calibrated=True)
    assert seen == []
    renderer.show_once("traffic-test")
    assert len(seen) == 1
    assert seen[0][0] == "traffic-test"
    assert isinstance(seen[0][1], np.ndarray)


def test_show_once_without_frame_does_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    from traffic_counter.overlay import OverlayRenderer

    calls: list[str] = []

    def fake_imshow(name: str, img: object) -> None:
        calls.append(name)

    def fake_waitkey(delay: int) -> int:
        calls.append("wait")
        return 0

    monkeypatch.setattr("traffic_counter.overlay.cv2.imshow", fake_imshow)
    monkeypatch.setattr("traffic_counter.overlay.cv2.waitKey", fake_waitkey)
    renderer = OverlayRenderer()
    renderer.show_once("empty-window")
    assert calls == []


def test_render_has_no_display_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    from traffic_counter.overlay import OverlayRenderer

    calls: list[str] = []

    def fake_imshow(name: str, img: object) -> None:
        calls.append(name)

    def fake_waitkey(delay: int) -> int:
        calls.append("wait")
        return 0

    monkeypatch.setattr("traffic_counter.overlay.cv2.imshow", fake_imshow)
    monkeypatch.setattr("traffic_counter.overlay.cv2.waitKey", fake_waitkey)
    renderer = OverlayRenderer()
    frame = make_frame()
    renderer.render(frame, [], make_line(), make_summaries(), AppState.RUNNING, calibrated=True)
    assert calls == []
