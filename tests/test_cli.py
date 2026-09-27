from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from traffic_counter.config import load_config

JAKARTA = ZoneInfo("Asia/Jakarta")

VALID_CONFIG_TEXT = """
[stream]
url = "https://example.test/index.m3u8"
width = 12
height = 8

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


def write_config(tmp_path: Path, text: str = VALID_CONFIG_TEXT) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    return path


def make_frame() -> Any:
    from traffic_counter.models import Frame

    moment = datetime(2026, 9, 25, 12, 0, tzinfo=JAKARTA)
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    image[2, 2] = 200
    return Frame(image=image, observed_at=moment)


def test_calibrate_jpeg_and_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import traffic_counter.cli as cli_module

    config_path = write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    frame = make_frame()

    class FakeSource:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

        def frames(self) -> Any:
            def gen() -> Any:
                yield frame

            return gen()

        def stop(self) -> None:
            return None

    monkeypatch.setattr(cli_module, "HlsFrameSource", FakeSource)
    code = cli_module.main(["--config", str(config_path), "--calibrate"])
    assert code == 0
    calib_dir = tmp_path / "data" / "calibration"
    files = list(calib_dir.glob("*.jpg"))
    assert len(files) == 1
    assert files[0].stat().st_size > 0
    text = config_path.read_text(encoding="utf-8")
    assert "calibrated = true" in text
    assert "calibrated = false" not in text
    leftovers = list(tmp_path.glob("**/*.tmp"))
    assert leftovers == []


def test_calibrate_spacing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import traffic_counter.cli as cli_module

    spaced = VALID_CONFIG_TEXT.replace("calibrated = false", "calibrated=false")
    config_path = write_config(tmp_path, spaced)
    monkeypatch.chdir(tmp_path)
    frame = make_frame()

    class FakeSource:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

        def frames(self) -> Any:
            def gen() -> Any:
                yield frame

            return gen()

        def stop(self) -> None:
            return None

    monkeypatch.setattr(cli_module, "HlsFrameSource", FakeSource)
    code = cli_module.main(["--config", str(config_path), "--calibrate"])
    assert code == 0
    text = config_path.read_text(encoding="utf-8")
    assert "calibrated = true" in text
    assert "calibrated=false" not in text


def test_calibrate_missing_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import traffic_counter.cli as cli_module

    missing = VALID_CONFIG_TEXT.replace(
        "hysteresis = 0.005\ncalibrated = false\n",
        "hysteresis = 0.005\n",
    )
    assert "calibrated" not in missing
    config_path = write_config(tmp_path, missing)
    monkeypatch.chdir(tmp_path)
    frame = make_frame()

    class FakeSource:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

        def frames(self) -> Any:
            def gen() -> Any:
                yield frame

            return gen()

        def stop(self) -> None:
            return None

    monkeypatch.setattr(cli_module, "HlsFrameSource", FakeSource)
    code = cli_module.main(["--config", str(config_path), "--calibrate"])
    assert code == 0
    text = config_path.read_text(encoding="utf-8")
    assert "calibrated = true" in text


def test_calibrate_timeout_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import traffic_counter.cli as cli_module

    config_path = write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    stopped: list[bool] = []

    class BlockingSource:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

        def frames(self) -> Any:
            def gen() -> Any:
                time.sleep(5.0)
                yield make_frame()

            return gen()

        def stop(self) -> None:
            stopped.append(True)

    monkeypatch.setattr(cli_module, "HlsFrameSource", BlockingSource)
    code = cli_module._run_calibration(
        load_config(config_path),
        config_path,
        timeout=0.1,
    )
    assert code == 1
    assert stopped == [True]


def test_calibrate_empty_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import traffic_counter.cli as cli_module

    config_path = write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    stopped: list[bool] = []

    class EmptySource:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

        def frames(self) -> Any:
            def gen() -> Any:
                yield from ()

            return gen()

        def stop(self) -> None:
            stopped.append(True)

    monkeypatch.setattr(cli_module, "HlsFrameSource", EmptySource)
    code = cli_module.main(["--config", str(config_path), "--calibrate"])
    assert code == 1
    assert stopped == [True]


def test_calibrate_unique_filenames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import traffic_counter.cli as cli_module

    config_path = write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    frame = make_frame()

    class FakeSource:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

        def frames(self) -> Any:
            def gen() -> Any:
                yield frame

            return gen()

        def stop(self) -> None:
            return None

    monkeypatch.setattr(cli_module, "HlsFrameSource", FakeSource)
    assert cli_module.main(["--config", str(config_path), "--calibrate"]) == 0
    assert cli_module.main(["--config", str(config_path), "--calibrate"]) == 0
    files = sorted((tmp_path / "data" / "calibration").glob("*.jpg"))
    assert len(files) == 2
    assert files[0].name != files[1].name


def test_atomic_replace_cleans_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import traffic_counter.cli as cli_module

    config_path = write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    frame = make_frame()

    class FakeSource:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

        def frames(self) -> Any:
            def gen() -> Any:
                yield frame

            return gen()

        def stop(self) -> None:
            return None

    monkeypatch.setattr(cli_module, "HlsFrameSource", FakeSource)

    def bad_replace(src: object, dst: object) -> None:
        raise OSError("disk locked")

    monkeypatch.setattr(os, "replace", bad_replace)
    code = cli_module.main(["--config", str(config_path), "--calibrate"])
    assert code == 1
    assert list(tmp_path.glob("**/*.tmp")) == []
    assert "calibrated = false" in config_path.read_text(encoding="utf-8")


def test_smoke_isolated_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    import traffic_counter.cli as cli_module
    from traffic_counter.models import Frame

    config_path = write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    prod_db = tmp_path / "data" / "traffic.db"
    if prod_db.is_file():
        prod_db.unlink()
    moment = datetime(2026, 9, 25, 12, 0, tzinfo=JAKARTA)
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    frame = Frame(image=image, observed_at=moment)

    class FakeSource:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

        def frames(self) -> Any:
            def gen() -> Any:
                yield frame

            return gen()

        def stop(self) -> None:
            return None

    class FakeDetector:
        def detect(self, frame: Frame) -> list[Any]:
            return []

    class FakeFactory:
        def create(self, config: Any) -> FakeDetector:
            return FakeDetector()

    monkeypatch.setattr(cli_module, "HlsFrameSource", FakeSource)
    monkeypatch.setattr(cli_module, "DetectorFactory", FakeFactory)
    code = cli_module.main(["--config", str(config_path), "--smoke-seconds", "1"])
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out.strip().splitlines()[-1])
    assert payload["smoke_seconds"] == 1
    assert isinstance(payload["frames"], int)
    assert isinstance(payload["events"], int)
    assert isinstance(payload["diagnostics"], dict)
    assert payload["diagnostics"]["received"] == payload["frames"]
    assert payload["diagnostics"]["event_count"] == payload["events"]
    assert "report_day" in payload["diagnostics"]
    assert not prod_db.is_file()


def test_smoke_error_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    import traffic_counter.cli as cli_module

    config_path = write_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    class BoomSource:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise OSError("no ffmpeg")

    monkeypatch.setattr(cli_module, "HlsFrameSource", BoomSource)
    code = cli_module.main(["--config", str(config_path), "--smoke-seconds", "1"])
    assert code == 1
    out = capsys.readouterr().out
    payload = json.loads(out.strip().splitlines()[-1])
    assert payload["state"] == "error"
    assert payload["smoke_seconds"] == 1
    assert payload["diagnostics"]["received"] == 0
    assert "report_day" in payload["diagnostics"]


def test_smoke_invalid_seconds_rejected(tmp_path: Path) -> None:
    import traffic_counter.cli as cli_module

    config_path = write_config(tmp_path)
    with pytest.raises(SystemExit):
        cli_module.main(["--config", str(config_path), "--smoke-seconds", "0"])


def test_cli_no_overlay_flag_parses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import traffic_counter.cli as cli_module

    config_path = write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    seen: dict[str, bool] = {}

    def fake_normal(config: object, no_overlay: bool) -> int:
        seen["no_overlay"] = no_overlay
        return 0

    monkeypatch.setattr(cli_module, "_run_normal", fake_normal)
    code = cli_module.main(["--config", str(config_path), "--no-overlay"])
    assert code == 0
    assert seen["no_overlay"] is True
