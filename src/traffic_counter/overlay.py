from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from traffic_counter.config import LineConfig
from traffic_counter.models import (
    AppSnapshot,
    AppState,
    Frame,
    PeriodName,
    PeriodSummary,
    TrackedVehicle,
)

type Summaries = Mapping[PeriodName, PeriodSummary] | Sequence[PeriodSummary]
type MaybeImage = Frame | NDArray[np.uint8]


def _as_image(frame: MaybeImage) -> NDArray[np.uint8]:
    if isinstance(frame, Frame):
        return frame.image
    return frame


def _as_state_text(state: object) -> str:
    if isinstance(state, AppSnapshot):
        return str(state.state.value)
    if isinstance(state, AppState):
        return str(state.value)
    return str(state)


def _totals(summaries: Summaries) -> tuple[str, int]:
    items = list(summaries.values()) if isinstance(summaries, Mapping) else list(summaries)
    order = {PeriodName.PAGI: 0, PeriodName.SIANG: 1, PeriodName.SORE: 2}
    items.sort(key=lambda item: order.get(item.period, 99))
    total = 0
    current = ""
    for item in items:
        total = total + int(item.total)
        if bool(item.started):
            current = str(item.period.value)
    if current == "" and len(items) > 0:
        current = str(items[0].period.value)
    return current, total


def _box_pixels(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    x1 = int(float(left) * float(width))
    y1 = int(float(top) * float(height))
    x2 = int(float(right) * float(width))
    y2 = int(float(bottom) * float(height))
    x1 = min(max(x1, 0), width - 1)
    x2 = min(max(x2, 0), width - 1)
    y1 = min(max(y1, 0), height - 1)
    y2 = min(max(y2, 0), height - 1)
    low_x = x1 if x1 < x2 else x2
    high_x = x2 if x1 < x2 else x1
    low_y = y1 if y1 < y2 else y2
    high_y = y2 if y1 < y2 else y1
    return low_x, low_y, high_x, high_y


def _draw_text(
    canvas: NDArray[np.uint8],
    text: str,
    pos: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
) -> None:
    cv2.putText(
        canvas,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


class OverlayRenderer:
    __slots__ = ("_last",)

    def __init__(self) -> None:
        self._last: NDArray[np.uint8] | None = None

    def render(
        self,
        frame: MaybeImage,
        vehicles: Sequence[TrackedVehicle],
        line: LineConfig,
        summaries: Summaries,
        state: AppState | str | AppSnapshot,
        *,
        calibrated: bool,
    ) -> NDArray[np.uint8]:
        base = _as_image(frame)
        canvas: NDArray[np.uint8] = base.copy()
        height = int(canvas.shape[0])
        width = int(canvas.shape[1])
        if height < 2 or width < 2:
            self._last = canvas
            return canvas
        x1 = int(float(line.x1) * float(width))
        y1 = int(float(line.y1) * float(height))
        x2 = int(float(line.x2) * float(width))
        y2 = int(float(line.y2) * float(height))
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
        for vehicle in list(vehicles):
            bx1, by1, bx2, by2 = _box_pixels(vehicle.box_xyxy, width, height)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            label = str(vehicle.track_id) + " " + str(vehicle.class_name)
            text_y = by1 - 6
            if text_y < 10:
                text_y = by2 + 16
            cv2.putText(
                canvas,
                label,
                (bx1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        state_text = _as_state_text(state)
        period_text, total = _totals(summaries)
        margin = max(8, width // 128)
        first_y = max(20, int(float(height) * 0.06))
        second_y = first_y + max(18, int(float(height) * 0.05))
        font_scale = max(0.5, float(width) / 1280.0)
        _draw_text(canvas, state_text, (margin, first_y), font_scale, (255, 255, 255))
        middle = period_text + " total " + str(total)
        _draw_text(canvas, middle, (margin, second_y), font_scale, (255, 255, 255))
        if not bool(calibrated):
            warn_y = height - max(12, int(float(height) * 0.04))
            _draw_text(canvas, "NOT CALIBRATED", (margin, warn_y), font_scale, (0, 0, 255))
        self._last = canvas
        return canvas

    def show_once(self, window_name: str = "traffic-counter") -> None:
        target: Any = self._last
        if target is None:
            return
        cv2.imshow(window_name, target)
        cv2.waitKey(1)
