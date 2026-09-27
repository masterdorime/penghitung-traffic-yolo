from __future__ import annotations

import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

from traffic_counter.config import AppConfig
from traffic_counter.models import AppSnapshot, AppState, StreamState
from traffic_counter.timeutils import JAKARTA, WallClock, report_date

type Clock = Callable[[], datetime]
type Sleeper = Callable[[float], None]
type ServeDashboard = Callable[[Any, Any, AppConfig], object]

ERROR_PREVIEW = 200
LOOP_SLEEP_SECONDS = 1.0
DEFAULT_DRAIN_SECONDS = 5.0


def serve_dashboard(
    snapshot_store: Any,
    csv_path_provider: Any,
    config: AppConfig,
) -> None:
    import importlib
    mod = importlib.import_module("traffic_counter.web")
    mod.serve_dashboard(snapshot_store, csv_path_provider, config)


class DailyController:
    __slots__ = (
        "_active_day",
        "_clock",
        "_config",
        "_csv_export_ok",
        "_dashboard_started",
        "_exporter",
        "_failed_day",
        "_failed_error",
        "_pipeline",
        "_purged_day",
        "_serve_dashboard",
        "_sleep",
        "_snapshots",
        "_store",
    )

    def __init__(
        self,
        pipeline: Any,
        *,
        config: AppConfig | None = None,
        store: Any | None = None,
        exporter: Any | None = None,
        snapshots: Any | None = None,
        clock: Clock | None = None,
        sleep: Sleeper | None = None,
        serve_dashboard_fn: ServeDashboard | None = serve_dashboard,
        **kwargs: Any,
    ) -> None:
        legacy = kwargs.pop("serve_dashboard", None)
        if kwargs:
            unexpected = sorted(kwargs.keys())[0]
            raise TypeError("unexpected keyword argument " + str(unexpected))
        chosen = legacy if legacy is not None else serve_dashboard_fn
        self._pipeline = pipeline
        self._config = config
        self._store = store
        self._exporter = exporter
        self._snapshots = snapshots
        self._clock: Clock = WallClock().now if clock is None else clock
        self._sleep: Sleeper = time.sleep if sleep is None else sleep
        self._serve_dashboard = chosen
        self._active_day: date | None = None
        self._failed_day: date | None = None
        self._failed_error: str | None = None
        self._purged_day: date | None = None
        self._csv_export_ok = True
        self._dashboard_started = False

    def recover(self) -> None:
        now = self._clock()
        day = report_date(now)
        if self._purged_day != day:
            try:
                if self._store is not None:
                    self._store.purge_before(day)
                if self._exporter is not None:
                    self._exporter.purge_before(day)
            except Exception as exc:
                message = _preview(exc)
                self._publish(AppState.ERROR, StreamState.DISCONNECTED, message)
                raise
            self._purged_day = day
        if self._exporter is not None:
            try:
                ok = self._exporter.try_export(day, now)
            except Exception:
                self._csv_export_ok = False
                self._publish(AppState.CONNECTING, StreamState.CONNECTING, None)
                return
            if not ok:
                self._csv_export_ok = False
                self._publish(AppState.CONNECTING, StreamState.CONNECTING, None)
                return
            self._csv_export_ok = True

    def run_for_day(self) -> None:
        now = self._clock()
        day = report_date(now)
        if self._failed_day is not None:
            if self._failed_day != day:
                self._failed_day = None
                self._failed_error = None
            else:
                self._publish(AppState.ERROR, StreamState.DISCONNECTED, self._failed_error)
                return
        if self._purged_day != day:
            try:
                if self._store is not None:
                    self._store.purge_before(day)
                if self._exporter is not None:
                    self._exporter.purge_before(day)
            except Exception as exc:
                message = _preview(exc)
                self._failed_day = day
                self._failed_error = message
                self._publish(AppState.ERROR, StreamState.DISCONNECTED, message)
                return
            if self._exporter is not None:
                try:
                    ok = self._exporter.try_export(day, now)
                except Exception:
                    self._csv_export_ok = False
                else:
                    self._csv_export_ok = bool(ok)
            self._purged_day = day
        start = datetime(day.year, day.month, day.day, 6, 0, tzinfo=JAKARTA)
        end = datetime(day.year, day.month, day.day, 19, 0, tzinfo=JAKARTA)
        if now < start:
            self._publish(AppState.WAITING_FOR_WINDOW, StreamState.DISCONNECTED, None)
            return
        if now < end:
            if self._active_day != day:
                try:
                    self.recover()
                except Exception as exc:
                    message = _preview(exc)
                    self._failed_day = day
                    self._failed_error = message
                    self._publish(AppState.ERROR, StreamState.DISCONNECTED, message)
                    return
                self._publish(AppState.CONNECTING, StreamState.CONNECTING, None)
                try:
                    self._pipeline.start()
                except Exception as exc:
                    message = _preview(exc)
                    self._failed_day = day
                    self._failed_error = message
                    self._publish(AppState.ERROR, StreamState.DISCONNECTED, message)
                    return
                self._active_day = day
                self._maybe_start_dashboard()
            try:
                self._pipeline.run_until(end)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                self._publish(AppState.ERROR, StreamState.DISCONNECTED, _preview(exc))
                return
            if self._clock() >= end:
                seconds = _drain_seconds(self._config)
                try:
                    self._pipeline.drain_for(seconds)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    self._publish(AppState.ERROR, StreamState.DISCONNECTED, _preview(exc))
                    return
                self._active_day = None
                self._publish(AppState.STOPPED, StreamState.STOPPED, None)
            return
        if self._active_day == day:
            seconds = _drain_seconds(self._config)
            try:
                self._pipeline.drain_for(seconds)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                self._publish(AppState.ERROR, StreamState.DISCONNECTED, _preview(exc))
                return
            self._active_day = None
            self._publish(AppState.STOPPED, StreamState.STOPPED, None)
        else:
            self._publish(AppState.STOPPED, StreamState.STOPPED, None)

    def run_forever(self) -> None:
        while True:
            try:
                self.run_for_day()
            except KeyboardInterrupt:
                with suppress(Exception):
                    self._pipeline.request_stop()
                self._publish(AppState.STOPPED, StreamState.STOPPED, None)
                return
            except Exception as exc:
                self._publish(AppState.ERROR, StreamState.DISCONNECTED, _preview(exc))
            try:
                self._sleep(LOOP_SLEEP_SECONDS)
            except KeyboardInterrupt:
                with suppress(Exception):
                    self._pipeline.request_stop()
                self._publish(AppState.STOPPED, StreamState.STOPPED, None)
                return
            except Exception:
                self._publish(AppState.STOPPED, StreamState.STOPPED, None)
                return

    def _publish(
        self,
        state: AppState,
        stream: StreamState,
        error: str | None,
    ) -> None:
        if self._snapshots is None:
            return
        now = self._clock()
        try:
            current = self._snapshots.get()
        except Exception:
            current = AppSnapshot.initial(now)
        incomplete = True if state is AppState.ERROR else bool(current.incomplete)
        try:
            current_ok = bool(current.csv_export_ok)
        except Exception:
            current_ok = False
        csv_ok = False if not self._csv_export_ok else current_ok
        try:
            snapshot = replace(
                current,
                generated_at=now,
                state=state,
                stream_state=stream,
                last_error=error,
                incomplete=incomplete,
                csv_export_ok=csv_ok,
            )
        except Exception:
            base = AppSnapshot.initial(now)
            snapshot = replace(
                base,
                generated_at=now,
                state=state,
                stream_state=stream,
                last_error=error,
                incomplete=incomplete,
                csv_export_ok=csv_ok,
            )
        with suppress(Exception):
            self._snapshots.publish(snapshot)

    def _maybe_start_dashboard(self) -> None:
        if self._serve_dashboard is None:
            return
        if self._dashboard_started:
            return
        if self._snapshots is None:
            return
        if self._config is None:
            return
        func = self._serve_dashboard
        snapshots = self._snapshots
        config = self._config
        clock = self._clock

        def provider() -> Path:
            day = report_date(clock())
            base = Path(config.output.export_directory)
            return base / ("traffic_" + day.isoformat() + ".csv")

        try:
            thread = threading.Thread(
                target=func,
                args=(snapshots, provider, config),
                daemon=True,
                name="dashboard",
            )
            thread.start()
            self._dashboard_started = True
        except Exception:
            return


def _preview(exc: BaseException) -> str:
    text = str(exc).strip()
    if text == "":
        return "controller failed"
    if len(text) > ERROR_PREVIEW:
        return text[:ERROR_PREVIEW]
    return text


def _drain_seconds(config: AppConfig | None) -> float:
    if config is None:
        return DEFAULT_DRAIN_SECONDS
    try:
        return float(config.timing.drain_seconds)
    except Exception:
        return DEFAULT_DRAIN_SECONDS
