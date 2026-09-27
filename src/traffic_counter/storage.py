from __future__ import annotations

import csv
import json
import os
import sqlite3
import threading
from collections.abc import Callable, Iterable
from contextlib import suppress
from datetime import date, datetime, timedelta
from datetime import time as clock_time
from pathlib import Path
from types import TracebackType
from typing import IO, Final

from traffic_counter.models import (
    AppState,
    CountEvent,
    HealthBucket,
    MinutePoint,
    PeriodName,
)
from traffic_counter.timeutils import (
    JAKARTA,
    WallClock,
    period_for,
    report_minutes_through_current,
    to_jakarta,
)

type Clock = Callable[[], datetime]

CSV_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "period",
    "minute_start",
    "count",
    "observed_seconds",
    "minute_rate",
    "period_total",
    "period_observed_seconds",
    "period_rate",
)
CSV_FILENAME_PREFIX: Final[str] = "traffic_"
CSV_SUFFIX: Final[str] = ".csv"
CSV_TEMPORARY_SUFFIX: Final[str] = ".tmp"
CSV_TERMINATOR: Final[str] = "\r\n"
SECONDS_PER_MINUTE: Final[int] = 60
MAX_HEALTH_BUCKET_SECONDS: Final[int] = 600
ROLLUP_SETTLE_MINUTES: Final[int] = 2

_DAY_MISMATCH: Final[str] = (
    "The report day must be the Asia/Jakarta calendar date of the supplied moment."
)
_UNINITIALIZED: Final[str] = "The traffic store must be initialized before it is used."

_SCHEMA: Final[str] = """
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
"""

_INSERT_APP_SESSION: Final[str] = (
    "INSERT INTO app_sessions(started_at, stopped_at, state) VALUES(?, NULL, ?)"
)
_INSERT_TRACKING_SESSION: Final[str] = (
    "INSERT INTO tracking_sessions(app_session_id, started_at, stopped_at) VALUES(?, ?, NULL)"
)
_STOP_TRACKING_SESSION: Final[str] = "UPDATE tracking_sessions SET stopped_at = ? WHERE id = ?"
_STOP_APP_SESSION: Final[str] = "UPDATE app_sessions SET stopped_at = ?, state = ? WHERE id = ?"
_INSERT_EVENT: Final[str] = (
    "INSERT INTO events(tracking_session_id, track_id, crossed_at, direction, period,"
    " class_name, inference_completed_at, inserted_at) VALUES(?, ?, ?, ?, ?, ?, NULL, ?)"
    " ON CONFLICT(tracking_session_id, track_id) DO NOTHING"
)
_UPSERT_HEALTH_BUCKET: Final[str] = (
    "INSERT INTO health_buckets(bucket_start, period, healthy_seconds, seconds_json)"
    " VALUES(?, ?, ?, ?)"
    " ON CONFLICT(bucket_start) DO UPDATE SET period = excluded.period,"
    " healthy_seconds = excluded.healthy_seconds, seconds_json = excluded.seconds_json"
)
_UPSERT_MINUTE_BUCKET: Final[str] = (
    "INSERT INTO minute_buckets(minute_start, period, event_count, observed_seconds)"
    " VALUES(?, ?, ?, ?)"
    " ON CONFLICT(minute_start) DO UPDATE SET period = excluded.period,"
    " event_count = excluded.event_count, observed_seconds = excluded.observed_seconds"
)
_COUNT_EVENTS_IN_PERIOD: Final[str] = (
    "SELECT COUNT(*) FROM events WHERE period = ? AND crossed_at >= ? AND crossed_at < ?"
)
_SELECT_EVENTS_BASE: Final[str] = (
    "SELECT tracking_session_id, track_id, crossed_at, direction, period, class_name"
    " FROM events WHERE crossed_at >= ? AND crossed_at < ?"
)
_SELECT_HEALTH_BUCKETS: Final[str] = "SELECT bucket_start, period, seconds_json FROM health_buckets"
_SELECT_HEALTH_SECONDS_IN_RANGE: Final[str] = (
    "SELECT seconds_json FROM health_buckets WHERE bucket_start >= ? AND bucket_start < ?"
)
_SELECT_MINUTES: Final[str] = (
    "SELECT minute_start, period, event_count, observed_seconds FROM minute_buckets"
    " WHERE minute_start >= ? AND minute_start < ? ORDER BY minute_start"
)
_SELECT_CROSSINGS_BY_MINUTE: Final[str] = (
    "SELECT substr(crossed_at, 1, 16) || ':00' || substr(crossed_at, -6) AS minute_start,"
    " COUNT(*) FROM events WHERE crossed_at >= ? AND crossed_at < ? GROUP BY minute_start"
)
_SELECT_ROLLUP_CURSOR: Final[str] = (
    "SELECT MAX(minute_start) FROM minute_buckets WHERE minute_start >= ? AND minute_start < ?"
)
_PURGE_EVENTS: Final[str] = "DELETE FROM events WHERE crossed_at < ?"
_PURGE_HEALTH_BUCKETS: Final[str] = "DELETE FROM health_buckets WHERE bucket_start < ?"
_PURGE_MINUTE_BUCKETS: Final[str] = "DELETE FROM minute_buckets WHERE minute_start < ?"
_PURGE_TRACKING_SESSIONS: Final[str] = (
    "DELETE FROM tracking_sessions WHERE started_at < ?"
    " AND id NOT IN (SELECT DISTINCT tracking_session_id FROM events)"
)
_PURGE_APP_SESSIONS: Final[str] = (
    "DELETE FROM app_sessions WHERE started_at < ?"
    " AND id NOT IN (SELECT app_session_id FROM tracking_sessions)"
)


def _midnight(day: date) -> datetime:
    return datetime.combine(day, clock_time.min, tzinfo=JAKARTA)


def _day_start_text(day: date) -> str:
    return _midnight(day).isoformat()


def _next_day_start_text(day: date) -> str:
    return (_midnight(day) + timedelta(days=1)).isoformat()


def _minute_floor(moment: datetime) -> datetime:
    return to_jakarta(moment).replace(second=0, microsecond=0)


def _parse(text: str) -> datetime:
    return to_jakarta(datetime.fromisoformat(text))


def _whole_seconds(moment: datetime) -> datetime:
    return to_jakarta(moment).replace(microsecond=0)


def _rate(total: int, observed_seconds: int) -> float | str:
    if observed_seconds <= 0:
        return ""
    return total / observed_seconds * SECONDS_PER_MINUTE


def _discard(path: Path) -> None:
    with suppress(OSError):
        path.unlink(missing_ok=True)


def _inserted_id(cursor: sqlite3.Cursor) -> int:
    identifier = cursor.lastrowid
    if identifier is None:
        raise RuntimeError("The database did not report an inserted row identifier.")
    return int(identifier)


def _export_filename(day: date) -> str:
    return f"{CSV_FILENAME_PREFIX}{day.isoformat()}{CSV_SUFFIX}"


def _filename_day(name: str) -> date | None:
    if not name.startswith(CSV_FILENAME_PREFIX) or not name.endswith(CSV_SUFFIX):
        return None
    stem = name[len(CSV_FILENAME_PREFIX) : -len(CSV_SUFFIX)]
    try:
        return date.fromisoformat(stem)
    except ValueError:
        return None


class TrafficStore:
    __slots__ = ("_clock", "_connection", "_lock", "_path")

    def __init__(self, path: Path, clock: Clock | None = None) -> None:
        self._path = Path(path)
        self._clock: Clock = WallClock().now if clock is None else clock
        self._connection: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    def initialize(self) -> None:
        with self._lock:
            if self._connection is not None:
                return
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self._path, check_same_thread=False)
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(_SCHEMA)
            self._connection = connection

    def close(self) -> None:
        with self._lock:
            connection = self._connection
            if connection is None:
                return
            self._connection = None
            connection.close()

    def __enter__(self) -> TrafficStore:
        self.initialize()
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _connection_or_raise(self) -> sqlite3.Connection:
        connection = self._connection
        if connection is None:
            raise RuntimeError(_UNINITIALIZED)
        return connection

    def start_app_session(self, started_at: datetime) -> int:
        moment = to_jakarta(started_at)
        connection = self._connection_or_raise()
        with self._lock, connection:
            cursor = connection.execute(
                _INSERT_APP_SESSION, (moment.isoformat(), str(AppState.STARTING))
            )
        return _inserted_id(cursor)

    def start_tracking_session(self, app_session_id: int, started_at: datetime) -> int:
        moment = to_jakarta(started_at)
        connection = self._connection_or_raise()
        with self._lock, connection:
            cursor = connection.execute(
                _INSERT_TRACKING_SESSION, (app_session_id, moment.isoformat())
            )
        return _inserted_id(cursor)

    def stop_tracking_session(self, tracking_session_id: int, stopped_at: datetime) -> None:
        moment = to_jakarta(stopped_at)
        connection = self._connection_or_raise()
        with self._lock, connection:
            connection.execute(_STOP_TRACKING_SESSION, (moment.isoformat(), tracking_session_id))

    def stop_app_session(
        self, app_session_id: int, stopped_at: datetime, state: AppState | str
    ) -> None:
        moment = to_jakarta(stopped_at)
        connection = self._connection_or_raise()
        with self._lock, connection:
            connection.execute(_STOP_APP_SESSION, (moment.isoformat(), str(state), app_session_id))

    def insert_event(self, event: CountEvent) -> bool:
        connection = self._connection_or_raise()
        crossed_at = to_jakarta(event.crossed_at)
        inserted_at = to_jakarta(self._clock())
        with self._lock, connection:
            cursor = connection.execute(
                _INSERT_EVENT,
                (
                    event.tracking_session_id,
                    event.track_id,
                    crossed_at.isoformat(),
                    event.direction,
                    str(event.period),
                    event.class_name,
                    inserted_at.isoformat(),
                ),
            )
        return cursor.rowcount == 1

    def write_health_buckets(self, buckets: Iterable[HealthBucket]) -> None:
        connection = self._connection_or_raise()
        rows = [
            (
                to_jakarta(record.bucket_start).isoformat(),
                str(record.period),
                len(seconds),
                json.dumps([moment.isoformat() for moment in seconds]),
            )
            for record in buckets
            for seconds in (_unioned_seconds(record),)
        ]
        if not rows:
            return
        with self._lock, connection:
            connection.executemany(_UPSERT_HEALTH_BUCKET, rows)

    def refresh_minute_rollups(self, report_day: date, now: datetime) -> None:
        connection = self._connection_or_raise()
        minutes = _elapsed_report_minutes(report_day, now)
        if not minutes:
            return
        with self._lock, connection:
            refresh_from = self._rollup_refresh_from(connection, report_day, minutes)
            pending = tuple(minute for minute in minutes if minute >= refresh_from)
            if not pending:
                return
            counts = _minute_event_counts(connection, minutes)
            observed = _minute_observed_seconds(connection, pending, refresh_from)
            rows = [
                (
                    minute.isoformat(),
                    str(period),
                    counts.get(minute, 0),
                    observed.get(minute, 0),
                )
                for minute, period in _labelled_minutes(pending)
            ]
            with connection:
                connection.executemany(_UPSERT_MINUTE_BUCKET, rows)

    def _rollup_refresh_from(
        self,
        connection: sqlite3.Connection,
        report_day: date,
        minutes: tuple[datetime, ...],
    ) -> datetime:
        row = connection.execute(
            _SELECT_ROLLUP_CURSOR, (_day_start_text(report_day), _next_day_start_text(report_day))
        ).fetchone()
        cursor = minutes[0] if row is None or row[0] is None else _parse(str(row[0]))
        settled = cursor - timedelta(minutes=ROLLUP_SETTLE_MINUTES)
        return min(max(settled, minutes[0]), minutes[-1])

    def summary_counts(self, day: date, period: PeriodName) -> int:
        connection = self._connection_or_raise()
        with self._lock:
            row = connection.execute(
                _COUNT_EVENTS_IN_PERIOD,
                (str(period), _day_start_text(day), _next_day_start_text(day)),
            ).fetchone()
        return 0 if row is None else int(row[0])

    def timeseries(self, report_day: date) -> tuple[MinutePoint, ...]:
        connection = self._connection_or_raise()
        with self._lock:
            rows = connection.execute(
                _SELECT_MINUTES, (_day_start_text(report_day), _next_day_start_text(report_day))
            ).fetchall()
        return tuple(
            MinutePoint(
                minute_start=_parse(str(minute_start)),
                period=PeriodName(str(period)),
                count=int(event_count),
                observed_seconds=int(observed_seconds),
            )
            for minute_start, period, event_count, observed_seconds in rows
        )

    def health_buckets(self, period: PeriodName | None = None) -> tuple[HealthBucket, ...]:
        connection = self._connection_or_raise()
        sql = _SELECT_HEALTH_BUCKETS
        params: tuple[str, ...] = ()
        if period is not None:
            sql = f"{sql} WHERE period = ?"
            params = (str(period),)
        sql = f"{sql} ORDER BY bucket_start"
        with self._lock:
            rows = connection.execute(sql, params).fetchall()
        return tuple(
            HealthBucket(
                bucket_start=_parse(str(bucket_start)),
                period=PeriodName(str(period)),
                seconds=tuple(_parse(text) for text in _decoded(str(seconds_json))),
            )
            for bucket_start, period, seconds_json in rows
        )

    def events(
        self, day: date, period: PeriodName | None = None
    ) -> tuple[CountEvent, ...]:
        connection = self._connection_or_raise()
        sql = _SELECT_EVENTS_BASE
        params: tuple[str, ...] = (_day_start_text(day), _next_day_start_text(day))
        if period is not None:
            sql = f"{sql} AND period = ?"
            params = (_day_start_text(day), _next_day_start_text(day), str(period))
        sql = f"{sql} ORDER BY crossed_at, id"
        with self._lock:
            rows = connection.execute(sql, params).fetchall()
        return tuple(
            CountEvent(
                tracking_session_id=int(tracking_session_id),
                track_id=int(track_id),
                crossed_at=_parse(str(crossed_at)),
                direction=str(direction),
                period=PeriodName(str(period)),
                class_name=str(class_name),
            )
            for tracking_session_id, track_id, crossed_at, direction, period, class_name in rows
        )

    def purge_before(self, report_day: date) -> None:
        connection = self._connection_or_raise()
        start = _day_start_text(report_day)
        with self._lock, connection:
            connection.execute(_PURGE_EVENTS, (start,))
            connection.execute(_PURGE_HEALTH_BUCKETS, (start,))
            connection.execute(_PURGE_MINUTE_BUCKETS, (start,))
            connection.execute(_PURGE_TRACKING_SESSIONS, (start,))
            connection.execute(_PURGE_APP_SESSIONS, (start,))


def _unioned_seconds(record: HealthBucket) -> tuple[datetime, ...]:
    return tuple(sorted({_whole_seconds(second) for second in record.seconds}))


def _decoded(payload: str) -> tuple[str, ...]:
    values: list[str] = json.loads(payload)
    return tuple(values)


def _elapsed_report_minutes(report_day: date, now: datetime) -> tuple[datetime, ...]:
    return tuple(
        minute for minute in report_minutes_through_current(now) if minute.date() == report_day
    )


def _labelled_minutes(minutes: tuple[datetime, ...]) -> tuple[tuple[datetime, PeriodName], ...]:
    labelled = ((minute, period_for(minute)) for minute in minutes)
    return tuple((minute, period) for minute, period in labelled if period is not None)


def _minute_event_counts(
    connection: sqlite3.Connection, minutes: tuple[datetime, ...]
) -> dict[datetime, int]:
    start_text = minutes[0].isoformat()
    end_text = (minutes[-1] + timedelta(minutes=1)).isoformat()
    counts: dict[datetime, int] = {}
    for minute_start, total in connection.execute(
        _SELECT_CROSSINGS_BY_MINUTE, (start_text, end_text)
    ).fetchall():
        minute = _parse(str(minute_start))
        counts[minute] = counts.get(minute, 0) + int(total)
    return counts


def _minute_observed_seconds(
    connection: sqlite3.Connection, minutes: tuple[datetime, ...], refresh_from: datetime
) -> dict[datetime, int]:
    earliest_bucket = (refresh_from - timedelta(seconds=MAX_HEALTH_BUCKET_SECONDS)).isoformat()
    latest_bucket = (minutes[-1] + timedelta(minutes=1)).isoformat()
    seconds: set[datetime] = set()
    for (payload,) in connection.execute(
        _SELECT_HEALTH_SECONDS_IN_RANGE, (earliest_bucket, latest_bucket)
    ).fetchall():
        seconds.update(datetime.fromisoformat(text) for text in _decoded(str(payload)))
    counts: dict[datetime, int] = {}
    for second in seconds:
        minute = _minute_floor(second)
        counts[minute] = counts.get(minute, 0) + 1
    return counts


class CsvExporter:
    __slots__ = ("_directory", "_store")

    def __init__(self, store: TrafficStore, export_directory: Path) -> None:
        self._store = store
        self._directory = Path(export_directory)

    def export(self, report_day: date, now: datetime) -> Path:
        moment = to_jakarta(now)
        if moment.date() != report_day:
            raise ValueError(_DAY_MISMATCH)
        path = self._directory / _export_filename(report_day)
        temporary = path.with_name(path.name + CSV_TEMPORARY_SUFFIX)
        self._directory.mkdir(parents=True, exist_ok=True)
        try:
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                self._write(handle, report_day, moment)
                handle.flush()
            os.replace(temporary, path)
        except BaseException:
            _discard(temporary)
            raise
        return path

    def try_export(self, report_day: date, now: datetime) -> bool:
        try:
            self.export(report_day, now)
        except OSError:
            return False
        return True

    def purge_before(self, report_day: date) -> None:
        if not self._directory.is_dir():
            return
        pattern = f"{CSV_FILENAME_PREFIX}*{CSV_SUFFIX}"
        for path in sorted(self._directory.glob(pattern)):
            if _filename_day(path.name) in (None, report_day):
                continue
            _discard(path)

    def _write(self, handle: IO[str], report_day: date, moment: datetime) -> None:
        self._store.refresh_minute_rollups(report_day, moment)
        writer = csv.writer(handle, lineterminator=CSV_TERMINATOR)
        writer.writerow(CSV_COLUMNS)
        stored = {point.minute_start: point for point in self._store.timeseries(report_day)}
        period_total: dict[PeriodName, int] = {}
        period_seconds: dict[PeriodName, int] = {}
        for minute in _elapsed_report_minutes(report_day, moment):
            point = stored.get(minute)
            period = period_for(minute)
            if period is None:
                continue
            count = 0 if point is None else point.count
            observed_seconds = 0 if point is None else point.observed_seconds
            total = period_total.get(period, 0) + count
            cumulative = period_seconds.get(period, 0) + observed_seconds
            period_total[period] = total
            period_seconds[period] = cumulative
            writer.writerow(
                [
                    report_day.isoformat(),
                    str(period),
                    minute.isoformat(),
                    count,
                    observed_seconds,
                    _rate(count, observed_seconds),
                    total,
                    cumulative,
                    _rate(total, cumulative),
                ]
            )
