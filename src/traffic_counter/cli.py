from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import sys
import tempfile
import threading
import time
import tomllib
from collections.abc import Iterator, Sequence
from contextlib import suppress
from datetime import timedelta
from pathlib import Path
from typing import Any

import cv2

from traffic_counter.app import DailyController, serve_dashboard
from traffic_counter.config import AppConfig, ConfigError, load_config
from traffic_counter.counter import LineCounter
from traffic_counter.detector import DetectorFactory
from traffic_counter.models import Frame
from traffic_counter.pipeline import TrafficPipeline
from traffic_counter.state import LatestFrameSlot, SnapshotStore
from traffic_counter.storage import CsvExporter, TrafficStore
from traffic_counter.stream import HlsFrameSource
from traffic_counter.timeutils import JAKARTA, WallClock, report_date

SMOKE_PREVIEW = 200
CALIBRATION_TIMEOUT_SECONDS = 30.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="traffic-counter")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--smoke-seconds", type=_positive_seconds, default=None)
    return parser


def _positive_seconds(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("smoke seconds must be positive") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("smoke seconds must be positive")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = Path(str(args.config))
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        parser.error(str(exc))
        return 2
    if bool(args.calibrate):
        return _run_calibration(config, config_path)
    smoke_seconds: int | None = args.smoke_seconds
    if smoke_seconds is not None:
        return _run_smoke(config, int(smoke_seconds))
    return _run_normal(config, bool(args.no_overlay))


def _run_normal(config: AppConfig, no_overlay: bool) -> int:
    _setup_logging(config)
    store = TrafficStore(Path(config.output.database))
    store.initialize()
    exporter = CsvExporter(store, Path(config.output.export_directory))
    snapshots = SnapshotStore()
    slot = LatestFrameSlot()
    counter = LineCounter(config.line)
    factory = DetectorFactory()
    overlay_cb: Any = None
    if not no_overlay:
        from traffic_counter.overlay import OverlayRenderer

        renderer = OverlayRenderer()

        def overlay_cb(frame: Frame, vehicles: Any, snapshot: Any) -> None:
            try:
                renderer.render(
                    frame,
                    vehicles,
                    config.line,
                    snapshot.periods,
                    snapshot.state,
                    calibrated=bool(config.line.calibrated),
                )
            except Exception:
                return
            with suppress(Exception):
                renderer.show_once("traffic-counter")

    pipeline = TrafficPipeline(
        config=config,
        store=store,
        exporter=exporter,
        snapshots=snapshots,
        slot=slot,
        counter=counter,
        detector_factory=factory,
        clock=WallClock().now,
        overlay=overlay_cb,
    )
    try:
        source = HlsFrameSource(
            config.stream,
            config.timing,
            on_disconnect=pipeline.on_disconnect,
            slot_stats=slot.stats,
        )
    except Exception as exc:
        logging.getLogger("traffic_counter").error(str(exc)[:SMOKE_PREVIEW])
        return 1
    stop_flag = threading.Event()

    def feed() -> None:
        try:
            for frame in source.frames():
                if stop_flag.is_set():
                    break
                with suppress(Exception):
                    slot.put(frame)
        except Exception:
            return

    feeder = threading.Thread(target=feed, daemon=True, name="frame-feeder")
    feeder.start()
    controller = DailyController(
        pipeline,
        config=config,
        store=store,
        exporter=exporter,
        snapshots=snapshots,
        clock=WallClock().now,
        sleep=time.sleep,
        serve_dashboard_fn=serve_dashboard,
    )
    try:
        controller.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_flag.set()
        with suppress(Exception):
            source.stop()
        with suppress(Exception):
            pipeline.request_stop()
    return 0


def _acquire_frame(iterator: Iterator[Frame], timeout: float) -> Frame | None:
    box: queue.Queue[tuple[bool, Any, Any]] = queue.Queue(maxsize=1)

    def work() -> None:
        try:
            frame = next(iterator)
            with suppress(Exception):
                box.put((True, frame, None))
        except StopIteration:
            with suppress(Exception):
                box.put((False, None, None))
        except Exception as exc:
            with suppress(Exception):
                box.put((False, None, exc))

    thread = threading.Thread(target=work, daemon=True, name="calibration-acquire")
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return None
    try:
        ok, got, _ = box.get_nowait()
    except Exception:
        return None
    if not ok:
        return None
    if isinstance(got, Frame):
        return got
    return None


def _unique_calibration_path(directory: Path, moment: Any) -> Path:
    try:
        local: Any = moment.astimezone(JAKARTA)
    except Exception:
        local = moment
    stamp: str = str(local.strftime("%Y%m%d_%H%M%S_%f"))
    stem: str = "calibration_" + stamp
    candidate = directory / (stem + ".jpg")
    counter = 1
    while candidate.exists():
        candidate = directory / (stem + "_" + str(counter) + ".jpg")
        counter = counter + 1
        if counter > 1000:
            break
    return candidate


def _run_calibration(
    config: AppConfig,
    config_path: Path,
    timeout: float = CALIBRATION_TIMEOUT_SECONDS,
) -> int:
    directory = Path(config.output.calibration_directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = HlsFrameSource(config.stream, config.timing)
    try:
        try:
            iterator = source.frames()
        except Exception:
            return 1
        frame = _acquire_frame(iterator, timeout)
        if frame is None:
            return 1
        try:
            image = frame.image
        except Exception:
            return 1
        target = _unique_calibration_path(directory, frame.observed_at)
        ok = False
        try:
            ok = bool(cv2.imwrite(str(target), image))
        except Exception:
            return 1
        if not ok:
            with suppress(Exception):
                target.unlink(missing_ok=True)
            return 1
    finally:
        with suppress(Exception):
            source.stop()
    try:
        _atomic_enable_calibrated(config_path)
    except Exception:
        return 1
    return 0


def _run_smoke(config: AppConfig, seconds: int) -> int:
    clock_fn = WallClock().now
    started = clock_fn()
    deadline = started + timedelta(seconds=float(seconds))
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        temp_db = tmp_path / "smoke.db"
        temp_exports = tmp_path / "exports"
        store = TrafficStore(temp_db)
        store.initialize()
        try:
            exporter = CsvExporter(store, temp_exports)
            snapshots = SnapshotStore(clock=clock_fn)
            slot = LatestFrameSlot()
            counter = LineCounter(config.line)
            factory = DetectorFactory()
            pipeline = TrafficPipeline(
                config=config,
                store=store,
                exporter=exporter,
                snapshots=snapshots,
                slot=slot,
                counter=counter,
                detector_factory=factory,
                clock=clock_fn,
            )
            try:
                source = HlsFrameSource(
                    config.stream,
                    config.timing,
                    on_disconnect=pipeline.on_disconnect,
                    slot_stats=slot.stats,
                )
            except Exception as exc:
                print(json.dumps(_smoke_error(seconds, exc, started)))
                return 1
            stop_flag = threading.Event()

            def feed() -> None:
                try:
                    for frame in source.frames():
                        if stop_flag.is_set():
                            break
                        with suppress(Exception):
                            slot.put(frame)
                        if clock_fn() >= deadline:
                            break
                except Exception:
                    return

            feeder = threading.Thread(target=feed, daemon=True, name="smoke-feeder")
            feeder.start()
            try:
                pipeline.start()
            except Exception as exc:
                print(json.dumps(_smoke_error(seconds, exc, started)))
                stop_flag.set()
                with suppress(Exception):
                    source.stop()
                return 1
            with suppress(Exception):
                pipeline.run_until(deadline)
            received = 0
            with suppress(Exception):
                received = int(slot.stats().received)
            events = 0
            with suppress(Exception):
                events = len(store.events(report_date(started)))
            state_text = "running"
            with suppress(Exception):
                state_text = str(snapshots.get().state.value)
            api_ok = False
            with suppress(Exception):
                from traffic_counter.web import create_app
                day = report_date(started)
                csv_path = temp_exports / ("traffic_" + day.isoformat() + ".csv")
                app = create_app(snapshots, lambda: csv_path, int(config.output.port))
                client = app.test_client()
                host = "127.0.0.1:" + str(int(config.output.port))
                resp = client.get("/api/status", headers={"Host": host})
                api_ok = resp.status_code == 200
            with suppress(Exception):
                pipeline.request_stop()
            stop_flag.set()
            with suppress(Exception):
                source.stop()
            payload = {
                "smoke_seconds": int(seconds),
                "state": state_text,
                "frames": int(received),
                "events": int(events),
                "api_ok": bool(api_ok),
                "diagnostics": {
                    "received": int(received),
                    "event_count": int(events),
                    "report_day": report_date(started).isoformat(),
                },
            }
            print(json.dumps(payload))
            return 0
        finally:
            with suppress(Exception):
                store.close()


def _smoke_error(seconds: int, exc: BaseException, started: Any) -> dict[str, Any]:
    try:
        day_text = report_date(started).isoformat()
    except Exception:
        day_text = ""
    return {
        "smoke_seconds": int(seconds),
        "state": "error",
        "frames": 0,
        "events": 0,
        "api_ok": False,
        "error": str(exc)[:SMOKE_PREVIEW],
        "diagnostics": {"received": 0, "event_count": 0, "report_day": day_text},
    }


def _atomic_enable_calibrated(config_path: Path) -> None:
    text = config_path.read_text(encoding="utf-8")
    try:
        document = tomllib.loads(text)
    except Exception as exc:
        raise ConfigError("Configuration file is not valid TOML syntax.") from exc
    table = document.get("line", None) if isinstance(document, dict) else None
    if not isinstance(table, dict):
        raise ConfigError("Configuration file is missing the line section.")
    mark = chr(35)
    ends_newline = text.endswith("\n") or text.endswith("\r\n")
    lines = text.splitlines()
    out: list[str] = []
    in_line = False
    found = False
    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            inner = stripped[1:-1].strip()
            in_line = inner == "line"
            out.append(raw)
            continue
        if in_line and not found:
            code, sep, _rest = raw.partition(mark)
            if "=" in code:
                left, _, _ = code.partition("=")
                if left.strip() == "calibrated":
                    indent_len = len(raw) - len(raw.lstrip())
                    indent = raw[:indent_len]
                    tail = ""
                    if sep:
                        idx = raw.find(mark)
                        trailing = raw[idx:]
                        cleaned = trailing.strip()
                        tail = "  " + cleaned
                    out.append(indent + "calibrated = true" + tail)
                    found = True
                    continue
        out.append(raw)
    if not found:
        header_idx = -1
        for idx, raw in enumerate(out):
            section = raw.strip()
            is_header = section.startswith("[") and section.endswith("]")
            if is_header and section[1:-1].strip() == "line":
                header_idx = idx
                break
        if header_idx < 0:
            raise ConfigError("Configuration file is missing the line section.")
        follower = -1
        for idx in range(header_idx + 1, len(out)):
            section = out[idx].strip()
            if section.startswith("[") and section.endswith("]"):
                follower = idx
                break
        if follower < 0:
            out.append("calibrated = true")
        else:
            out.insert(follower, "calibrated = true")
    updated = "\n".join(out)
    if ends_newline:
        updated = updated + "\n"
    if updated == text:
        return
    sibling = config_path.with_name(config_path.name + ".tmp")
    try:
        sibling.write_text(updated, encoding="utf-8")
        try:
            os.replace(sibling, config_path)
        except Exception:
            with suppress(Exception):
                sibling.unlink(missing_ok=True)
            raise
    except Exception:
        with suppress(Exception):
            sibling.unlink(missing_ok=True)
        raise


def _setup_logging(config: AppConfig) -> None:
    try:
        target = Path(config.output.log_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=str(target), level=logging.INFO, force=True)
    except Exception:
        return


if __name__ == "__main__":
    sys.exit(main())
