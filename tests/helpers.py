from __future__ import annotations

from datetime import datetime
from typing import Final
from zoneinfo import ZoneInfo

import numpy as np
from numpy.typing import NDArray

from traffic_counter.models import Frame

JAKARTA: Final[ZoneInfo] = ZoneInfo("Asia/Jakarta")
FRAME_SHAPE: Final[tuple[int, int, int]] = (8, 12, 3)
FIXED_OBSERVED_AT: Final[datetime] = datetime(2026, 9, 25, 6, 0, tzinfo=JAKARTA)
MAX_MARKER: Final[int] = 255


def make_frame(marker: int) -> Frame:
    if not isinstance(marker, int) or not 0 <= marker <= MAX_MARKER:
        raise ValueError(f"Frame markers must be integers between 0 and {MAX_MARKER}.")
    image: NDArray[np.uint8] = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    image[0, 0] = marker
    return Frame(image=image, observed_at=FIXED_OBSERVED_AT)


def frame_marker(frame: Frame) -> int:
    return int(frame.image[0, 0, 0])
