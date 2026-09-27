import re
from collections.abc import Callable
from contextlib import closing
from datetime import date, datetime, timedelta
from datetime import time as clock_time
from pathlib import Path
from sqlite3 import IntegrityError, connect
from threading import Thread
from typing import Final

import pytest

from traffic_counter.models import AppState, CountEvent, HealthBucket, PeriodName
from traffic_counter.storage import (
    MAX_HEALTH_BUCKET_SECONDS,
    ROLLUP_SETTLE_MINUTES,
    TrafficStore,
)
from traffic_counter.timeutils import JAKARTA, period_for

DAY: Final[date] = date(2026, 9, 25)
PRIOR_DAY: Final[date] = date(2026, 9, 24)
JOURNAL_MODE: Final[str] = "wal"
TABLES: Final[tuple[str, ...]] = (
    "app_sessions",
    "tracking_sessions",
    "events",
    "health_buckets",
    "minute_buckets",
)
INDEXES: Final[tuple[str, ...]] = ("events_crossed_at_idx", "minute_buckets_period_idx")
COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "app_session_id",
    "started_at",
    "stopped_at",
    "state",
    "tracking_session_id",
    "track_id",
    "crossed_at",
    "direction",
    "period",
    "class_name",
    "inference_completed_at",
    "inserted_at",
    "bucket_start",
    "healthy_seconds",
    "seconds_json",
    "minute_start",
    "event_count",
    "observed_seconds",
)


def at(
    hour: int,
    minute: int = 0,
    second: int = 0,
    day: date = DAY,
    microsecond: int = 0,
) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, second, microsecond, tzinfo=JAKARTA)


def span(start: datetime, count: int) -> tuple[datetime, ...]:
    return tuple(start + timedelta(seconds=offset) for offset in range(count))


def make_store(path: Path, *, clock: Callable[[], datetime] | None = None) -> TrafficStore:
    store = TrafficStore(path, clock=clock) if clock is not None else TrafficStore(path)
    store.initialize()
    return store


def event(
    tracking_session_id: int,
    track_id: int,
    crossed_at: datetime,
    *,
    direction: str = "up",
    class_name: str = "car",
    period: PeriodName | None = None,
) -> CountEvent:
    resolved = period_for(crossed_at) if period is None else period
    assert resolved is not None
    return CountEvent(
        tracking_session_id=tracking_session_id,
        track_id=track_id,
        crossed_at=crossed_at,
        direction=direction,
        period=resolved,
        class_name=class_name,
    )


def bucket(
    start: datetime,
    seconds: tuple[datetime, ...],
    *,
    period: PeriodName | None = None,
) -> HealthBucket:
    resolved = period_for(start) if period is None else period
    assert resolved is not None
    return HealthBucket(bucket_start=start, period=resolved, seconds=seconds)


def rows(path: Path, sql: str, params: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    with closing(connect(path)) as connection:
        return connection.execute(sql, params).fetchall()


def names(path: Path, kind: str) -> set[str]:
    with closing(connect(path)) as connection:
        return {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = ?", (kind,))
        }


def opened_session(store: TrafficStore, started_at: datetime | None = None) -> tuple[int, int]:
    moment = at(6, 0) if started_at is None else started_at
    app_session_id = store.start_app_session(moment)
    return app_session_id, store.start_tracking_session(app_session_id, moment)


def test_initialize_creates_every_table(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    make_store(path).close()

    assert names(path, "table") >= set(TABLES)


def test_initialize_creates_every_planned_column(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    make_store(path).close()

    declared: set[str] = set()
    for table in TABLES:
        declared.update(str(row[1]) for row in rows(path, f"PRAGMA table_info({table})"))

    assert declared == set(COLUMNS)


def test_initialize_uses_write_ahead_logging(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    make_store(path).close()

    with closing(connect(path)) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == JOURNAL_MODE


def test_initialize_creates_the_required_indexes(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    make_store(path).close()

    assert names(path, "index") >= set(INDEXES)


def test_event_uniqueness_is_declared_in_the_schema(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    make_store(path).close()

    constraints = rows(path, "SELECT sql FROM sqlite_master WHERE name = 'events'")

    assert len(constraints) == 1
    assert "UNIQUE" in str(constraints[0][0])
    assert "tracking_session_id" in str(constraints[0][0])
    assert "track_id" in str(constraints[0][0])


def test_minute_buckets_constrain_observed_seconds_to_sixty(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    make_store(path).close()

    schema = str(rows(path, "SELECT sql FROM sqlite_master WHERE name = 'minute_buckets'")[0][0])

    assert "observed_seconds BETWEEN 0 AND 60" in schema


def test_initialize_creates_the_database_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deeper" / "traffic.db"

    make_store(path).close()

    assert path.is_file()


def test_initialize_is_idempotent_and_preserves_data(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    app_session_id, tracking_session_id = opened_session(store)
    assert store.insert_event(event(tracking_session_id, 1, at(6, 1)))

    store.initialize()

    assert store.summary_counts(DAY, PeriodName.PAGI) == 1
    assert app_session_id == 1
    store.close()


def test_use_before_initialize_is_rejected(tmp_path: Path) -> None:
    store = TrafficStore(tmp_path / "traffic.db")

    with pytest.raises(RuntimeError, match="initialized"):
        store.summary_counts(DAY, PeriodName.PAGI)
    with pytest.raises(RuntimeError, match="initialized"):
        store.start_app_session(at(6, 0))
    with pytest.raises(RuntimeError, match="initialized"):
        store.insert_event(event(1, 1, at(6, 1)))


def test_session_identifiers_increase(tmp_path: Path) -> None:
    store = make_store(tmp_path / "traffic.db")

    first_app = store.start_app_session(at(6, 0))
    second_app = store.start_app_session(at(7, 0))
    first_tracking = store.start_tracking_session(first_app, at(6, 0))
    second_tracking = store.start_tracking_session(first_app, at(6, 1))
    third_tracking = store.start_tracking_session(second_app, at(7, 0))

    assert first_app == 1
    assert second_app == 2
    assert (first_tracking, second_tracking, third_tracking) == (1, 2, 3)
    store.close()


def test_start_app_session_records_the_start_time_and_starting_state(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)

    app_session_id = store.start_app_session(at(6, 0, 7))

    stored = rows(
        path,
        "SELECT started_at, stopped_at, state FROM app_sessions WHERE id = ?",
        (app_session_id,),
    )
    assert stored == [("2026-09-25T06:00:07+07:00", None, str(AppState.STARTING))]
    store.close()


def test_start_app_session_normalizes_a_non_jakarta_timestamp(tmp_path: Path) -> None:
    from datetime import UTC

    path = tmp_path / "traffic.db"
    store = make_store(path)

    app_session_id = store.start_app_session(datetime(2026, 9, 24, 23, 0, 5, tzinfo=UTC))

    stored = rows(path, "SELECT started_at FROM app_sessions WHERE id = ?", (app_session_id,))
    assert stored == [("2026-09-25T06:00:05+07:00",)]
    store.close()


def test_start_app_session_rejects_a_naive_timestamp(tmp_path: Path) -> None:
    store = make_store(tmp_path / "traffic.db")

    with pytest.raises(ValueError, match="timezone-aware"):
        store.start_app_session(datetime(2026, 9, 25, 6, 0))
    store.close()


def test_start_tracking_session_records_the_start_time_and_parent(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    app_session_id = store.start_app_session(at(6, 0))

    tracking_session_id = store.start_tracking_session(app_session_id, at(6, 0, 3))

    stored = rows(
        path,
        "SELECT app_session_id, started_at, stopped_at FROM tracking_sessions WHERE id = ?",
        (tracking_session_id,),
    )
    assert stored == [(app_session_id, "2026-09-25T06:00:03+07:00", None)]
    store.close()


def test_start_tracking_session_requires_a_real_app_session(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)

    with pytest.raises(IntegrityError):
        store.start_tracking_session(999, at(6, 0))

    assert rows(path, "SELECT COUNT(*) FROM tracking_sessions") == [(0,)]
    store.close()


def test_foreign_keys_reject_deleting_a_referenced_app_session(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    app_session_id, _ = opened_session(store)

    with closing(connect(path)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(IntegrityError):
            connection.execute("DELETE FROM app_sessions WHERE id = ?", (app_session_id,))

    assert rows(path, "SELECT COUNT(*) FROM app_sessions") == [(1,)]
    store.close()


def test_foreign_keys_reject_deleting_a_referenced_tracking_session(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    _, tracking_session_id = opened_session(store)
    assert store.insert_event(event(tracking_session_id, 1, at(6, 1)))

    with closing(connect(path)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(IntegrityError):
            connection.execute("DELETE FROM tracking_sessions WHERE id = ?", (tracking_session_id,))

    assert store.summary_counts(DAY, PeriodName.PAGI) == 1
    store.close()


def test_events_require_a_real_tracking_session(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)

    with pytest.raises(IntegrityError):
        store.insert_event(event(999, 1, at(6, 1)))

    assert rows(path, "SELECT COUNT(*) FROM events") == [(0,)]
    store.close()


def test_stop_tracking_session_records_the_stop_time(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    _, tracking_session_id = opened_session(store)

    store.stop_tracking_session(tracking_session_id, at(6, 45, 12))

    stored = rows(
        path, "SELECT stopped_at FROM tracking_sessions WHERE id = ?", (tracking_session_id,)
    )
    assert stored == [("2026-09-25T06:45:12+07:00",)]
    store.close()


def test_stop_tracking_session_is_ignored_for_an_unknown_session(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    opened_session(store)

    store.stop_tracking_session(999, at(6, 45))

    assert rows(path, "SELECT COUNT(*) FROM tracking_sessions WHERE stopped_at IS NOT NULL") == [
        (0,)
    ]
    store.close()


def test_stop_app_session_records_the_stop_time_and_state(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    app_session_id, _ = opened_session(store)

    store.stop_app_session(app_session_id, at(19, 5), AppState.STOPPED)

    stored = rows(
        path, "SELECT stopped_at, state FROM app_sessions WHERE id = ?", (app_session_id,)
    )
    assert stored == [("2026-09-25T19:05:00+07:00", str(AppState.STOPPED))]
    store.close()


def test_stop_app_session_accepts_a_plain_state_string(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    app_session_id, _ = opened_session(store)

    store.stop_app_session(app_session_id, at(19, 5), "error")

    assert rows(path, "SELECT state FROM app_sessions WHERE id = ?", (app_session_id,)) == [
        ("error",)
    ]
    store.close()


def test_stop_app_session_rejects_a_naive_timestamp(tmp_path: Path) -> None:
    store = make_store(tmp_path / "traffic.db")
    app_session_id, _ = opened_session(store)

    with pytest.raises(ValueError, match="timezone-aware"):
        store.stop_app_session(app_session_id, datetime(2026, 9, 25, 19, 5), AppState.STOPPED)
    store.close()


def test_duplicate_tracking_pair_is_rejected(tmp_path: Path) -> None:
    store = make_store(tmp_path / "traffic.db")
    _, tracking_session_id = opened_session(store)
    duplicate = event(tracking_session_id, 9, at(6, 1))

    assert store.insert_event(duplicate) is True
    assert store.insert_event(duplicate) is False
    assert store.summary_counts(DAY, PeriodName.PAGI) == 1
    store.close()


def test_a_rejected_duplicate_does_not_disturb_the_stored_event(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    _, tracking_session_id = opened_session(store)
    assert store.insert_event(event(tracking_session_id, 9, at(6, 1), direction="up"))

    assert store.insert_event(event(tracking_session_id, 9, at(6, 40), direction="down")) is False

    stored = rows(
        path,
        "SELECT crossed_at, direction FROM events WHERE tracking_session_id = ?",
        (tracking_session_id,),
    )
    assert stored == [("2026-09-25T06:01:00+07:00", "up")]
    store.close()


def test_a_rejected_duplicate_leaves_the_store_usable(tmp_path: Path) -> None:
    store = make_store(tmp_path / "traffic.db")
    _, tracking_session_id = opened_session(store)
    assert store.insert_event(event(tracking_session_id, 1, at(6, 1)))

    assert store.insert_event(event(tracking_session_id, 1, at(6, 2))) is False

    assert store.insert_event(event(tracking_session_id, 2, at(6, 3))) is True
    assert store.summary_counts(DAY, PeriodName.PAGI) == 2
    store.close()


def test_the_same_track_id_in_another_tracking_session_is_a_new_event(tmp_path: Path) -> None:
    store = make_store(tmp_path / "traffic.db")
    app_session_id = store.start_app_session(at(6, 0))
    first = store.start_tracking_session(app_session_id, at(6, 0))
    second = store.start_tracking_session(app_session_id, at(6, 30))

    assert store.insert_event(event(first, 4, at(6, 1))) is True
    assert store.insert_event(event(second, 4, at(6, 31))) is True
    assert store.insert_event(event(second, 4, at(6, 32))) is False
    assert store.summary_counts(DAY, PeriodName.PAGI) == 2
    store.close()


def test_insert_event_stores_every_field(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path, clock=lambda: at(6, 1, 9) + timedelta(days=1))
    _, tracking_session_id = opened_session(store)

    assert store.insert_event(
        event(tracking_session_id, 12, at(7, 2, 3), direction="down", class_name="truck")
    )

    stored = rows(
        path,
        "SELECT track_id, crossed_at, direction, period, class_name,"
        " inference_completed_at, inserted_at FROM events WHERE tracking_session_id = ?",
        (tracking_session_id,),
    )
    assert stored == [
        (
            12,
            "2026-09-25T07:02:03+07:00",
            "down",
            "pagi",
            "truck",
            None,
            "2026-09-26T06:01:09+07:00",
        )
    ]
    store.close()


def test_insert_event_rejects_a_naive_timestamp(tmp_path: Path) -> None:
    store = make_store(tmp_path / "traffic.db")
    _, tracking_session_id = opened_session(store)

    with pytest.raises(ValueError, match="timezone-aware"):
        store.insert_event(event(tracking_session_id, 1, datetime(2026, 9, 25, 6, 1)))
    store.close()


def test_events_reader_round_trips_the_stored_record(tmp_path: Path) -> None:
    store = make_store(tmp_path / "traffic.db")
    app_session_id = store.start_app_session(at(6, 0))
    first = store.start_tracking_session(app_session_id, at(6, 0))
    second = store.start_tracking_session(app_session_id, at(6, 30))
    later = event(second, 8, at(9, 59, 59), direction="down", class_name="bus")
    earlier = event(first, 1, at(6, 1, 5), direction="up", class_name="car")
    assert store.insert_event(later)
    assert store.insert_event(earlier)

    assert store.events(DAY) == (earlier, later)
    assert store.events(DAY, PeriodName.PAGI) == (earlier, later)
    assert store.events(DAY, PeriodName.SIANG) == ()
    store.close()


def test_health_bucket_json_round_trip_preserves_exact_seconds(tmp_path: Path) -> None:
    store = make_store(tmp_path / "traffic.db")
    seconds = (at(6, 0, 30), at(6, 0, 0), at(6, 0, 20), at(6, 0, 10))

    store.write_health_buckets((bucket(at(6, 0), seconds),))

    restored = store.health_buckets()
    assert restored == (bucket(at(6, 0), (at(6, 0, 0), at(6, 0, 10), at(6, 0, 20), at(6, 0, 30))),)
    assert restored[0].seconds == tuple(sorted(seconds))
    store.close()


def test_health_bucket_seconds_are_stored_sorted_and_counted(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    store.write_health_buckets((bucket(at(6, 0, 20), (at(6, 0, 30), at(6, 0, 10))),))

    stored = rows(path, "SELECT period, healthy_seconds, seconds_json FROM health_buckets")
    assert stored == [
        (
            "pagi",
            2,
            '["2026-09-25T06:00:10+07:00", "2026-09-25T06:00:30+07:00"]',
        )
    ]
    store.close()


def test_health_bucket_seconds_are_floored_to_whole_seconds(tmp_path: Path) -> None:
    store = make_store(tmp_path / "traffic.db")

    store.write_health_buckets(
        (bucket(at(6, 0), (at(6, 0, microsecond=750000), at(6, 0, 1, microsecond=1))),)
    )

    assert store.health_buckets()[0].seconds == (at(6, 0, 0), at(6, 0, 1))
    store.close()


def test_health_bucket_duplicates_within_one_bucket_are_unioned(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)

    store.write_health_buckets((bucket(at(6, 0), (at(6, 0, 0), at(6, 0, 0), at(6, 0, 1))),))

    assert rows(path, "SELECT healthy_seconds, seconds_json FROM health_buckets") == [
        (2, '["2026-09-25T06:00:00+07:00", "2026-09-25T06:00:01+07:00"]')
    ]
    store.close()


def test_writing_the_same_bucket_twice_replaces_the_earlier_seconds(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)

    store.write_health_buckets((bucket(at(6, 0), span(at(6, 0), 2)),))
    store.write_health_buckets((bucket(at(6, 0), span(at(6, 0), 3)),))

    assert rows(path, "SELECT COUNT(*) FROM health_buckets") == [(1,)]
    assert store.health_buckets()[0].seconds == span(at(6, 0), 3)
    store.close()


def test_health_buckets_accepts_an_empty_sequence(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)

    store.write_health_buckets(())

    assert rows(path, "SELECT COUNT(*) FROM health_buckets") == [(0,)]
    store.close()


def test_health_buckets_writer_rejects_more_seconds_than_a_bucket_allows(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    oversized = bucket(at(6, 0), span(at(6, 0), 601))

    with pytest.raises(IntegrityError):
        store.write_health_buckets((oversized,))

    assert rows(path, "SELECT COUNT(*) FROM health_buckets") == [(0,)]
    store.close()


def test_health_buckets_writer_rejects_a_naive_timestamp(tmp_path: Path) -> None:
    store = make_store(tmp_path / "traffic.db")

    with pytest.raises(ValueError, match="timezone-aware"):
        store.write_health_buckets(
            (HealthBucket(datetime(2026, 9, 25, 6, 0), PeriodName.PAGI, ()),)
        )
    store.close()


def test_health_buckets_reader_can_filter_by_period(tmp_path: Path) -> None:
    store = make_store(tmp_path / "traffic.db")
    store.write_health_buckets(
        (
            bucket(at(6, 0), span(at(6, 0), 1)),
            bucket(at(10, 0), span(at(10, 0), 1)),
            bucket(at(15, 0), span(at(15, 0), 1)),
        )
    )

    assert len(store.health_buckets()) == 3
    assert store.health_buckets(PeriodName.PAGI) == (bucket(at(6, 0), span(at(6, 0), 1)),)
    assert store.health_buckets(PeriodName.SIANG) == (bucket(at(10, 0), span(at(10, 0), 1)),)
    assert store.health_buckets(PeriodName.SORE) == (bucket(at(15, 0), span(at(15, 0), 1)),)
    store.close()


def test_refresh_minute_rollups_materializes_zero_minutes(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)

    store.refresh_minute_rollups(DAY, at(6, 2, 30))

    assert rows(
        path, "SELECT minute_start, period, event_count, observed_seconds FROM minute_buckets"
    ) == [
        ("2026-09-25T06:00:00+07:00", "pagi", 0, 0),
        ("2026-09-25T06:01:00+07:00", "pagi", 0, 0),
        ("2026-09-25T06:02:00+07:00", "pagi", 0, 0),
    ]
    store.close()


def test_refresh_minute_rollups_counts_events_and_healthy_seconds_per_minute(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    _, tracking_session_id = opened_session(store)
    assert store.insert_event(event(tracking_session_id, 1, at(6, 0, 1)))
    assert store.insert_event(event(tracking_session_id, 2, at(6, 0, 59)))
    assert store.insert_event(event(tracking_session_id, 3, at(6, 1, 0)))
    store.write_health_buckets(
        (bucket(at(6, 0), span(at(6, 0), 60)), bucket(at(6, 1), span(at(6, 1), 30)))
    )

    store.refresh_minute_rollups(DAY, at(6, 2))

    assert rows(path, "SELECT minute_start, event_count, observed_seconds FROM minute_buckets") == [
        ("2026-09-25T06:00:00+07:00", 2, 60),
        ("2026-09-25T06:01:00+07:00", 1, 30),
        ("2026-09-25T06:02:00+07:00", 0, 0),
    ]
    store.close()


def test_refresh_minute_rollups_unions_healthy_seconds_across_buckets(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    store.write_health_buckets(
        (
            bucket(at(6, 0), span(at(6, 0), 60)),
            bucket(at(6, 0, 30), span(at(6, 0, 30), 60)),
            bucket(at(6, 1), span(at(6, 1), 10)),
        )
    )

    store.refresh_minute_rollups(DAY, at(6, 2))

    assert rows(path, "SELECT minute_start, observed_seconds FROM minute_buckets") == [
        ("2026-09-25T06:00:00+07:00", 60),
        ("2026-09-25T06:01:00+07:00", 30),
        ("2026-09-25T06:02:00+07:00", 0),
    ]
    store.close()


def clear_trace(store: TrafficStore) -> None:
    store._connection.set_trace_callback(None)


def test_refresh_minute_rollups_does_not_rewrite_minutes_before_the_refresh_window(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    store.refresh_minute_rollups(DAY, at(6, 3))
    planted = "2026-09-25T06:00:00+07:00"
    with closing(connect(path)) as connection:
        connection.execute(
            "UPDATE minute_buckets SET event_count = 41, observed_seconds = 17"
            " WHERE minute_start = ?",
            (planted,),
        )
        connection.commit()
    captured: list[str] = []
    store._connection.set_trace_callback(captured.append)

    store.refresh_minute_rollups(DAY, at(6, 4))
    clear_trace(store)

    written = [statement for statement in captured if "INSERT INTO minute_buckets" in statement]
    assert written
    assert all("06:00:00" not in statement for statement in written)
    assert rows(
        path,
        "SELECT event_count, observed_seconds FROM minute_buckets WHERE minute_start = ?",
        (planted,),
    ) == [(41, 17)]
    assert rows(path, "SELECT COUNT(*) FROM minute_buckets") == [(5,)]
    store.close()


def test_refresh_minute_rollups_advances_its_cursor_so_the_rewrite_window_stays_small(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path / "traffic.db")

    def measure(now: datetime) -> int:
        captured: list[str] = []
        store._connection.set_trace_callback(captured.append)
        store.refresh_minute_rollups(DAY, now)
        clear_trace(store)
        return len([s for s in captured if "INSERT INTO minute_buckets" in s])

    store.refresh_minute_rollups(DAY, at(6, 30))
    morning = [measure(at(6, minute)) for minute in (31, 32, 33)]
    store.refresh_minute_rollups(DAY, at(18, 50))
    evening = [measure(at(18, minute)) for minute in (51, 52, 53)]

    assert morning == [4, 4, 4]
    assert evening == [4, 4, 4]
    assert len(store.timeseries(DAY)) == 774
    store.close()


def test_refresh_minute_rollups_rewrites_only_the_current_minute_when_the_clock_moves_back(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path / "traffic.db")
    store.refresh_minute_rollups(DAY, at(6, 30))
    store.refresh_minute_rollups(DAY, at(18, 50))
    captured: list[str] = []
    store._connection.set_trace_callback(captured.append)

    store.refresh_minute_rollups(DAY, at(6, 31))
    clear_trace(store)
    written = len([s for s in captured if "INSERT INTO minute_buckets" in s])

    assert written == 1
    assert len(store.timeseries(DAY)) == 771
    store.close()


def test_refresh_minute_rollups_materializes_every_minute_a_clock_jump_skips(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    store.refresh_minute_rollups(DAY, at(6, 0))
    captured: list[str] = []
    store._connection.set_trace_callback(captured.append)

    store.refresh_minute_rollups(DAY, at(12, 0))
    clear_trace(store)

    written = len([s for s in captured if "INSERT INTO minute_buckets" in s])
    assert written == 361
    assert rows(path, "SELECT COUNT(*) FROM minute_buckets") == [(361,)]
    store.close()


def test_refresh_minute_rollups_stays_exact_for_a_health_bucket_spanning_the_cursor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    spanning = at(6, 5)
    long_bucket = bucket(
        spanning, tuple(spanning + timedelta(seconds=offset) for offset in range(600))
    )
    store.write_health_buckets((long_bucket,))
    store.refresh_minute_rollups(DAY, at(6, 10))
    assert rows(
        path,
        "SELECT observed_seconds FROM minute_buckets WHERE minute_start = ?",
        ("2026-09-25T06:08:00+07:00",),
    ) == [(60,)]

    store.refresh_minute_rollups(DAY, at(6, 11))

    assert rows(path, "SELECT minute_start, observed_seconds FROM minute_buckets") == [
        ("2026-09-25T06:00:00+07:00", 0),
        ("2026-09-25T06:01:00+07:00", 0),
        ("2026-09-25T06:02:00+07:00", 0),
        ("2026-09-25T06:03:00+07:00", 0),
        ("2026-09-25T06:04:00+07:00", 0),
        ("2026-09-25T06:05:00+07:00", 60),
        ("2026-09-25T06:06:00+07:00", 60),
        ("2026-09-25T06:07:00+07:00", 60),
        ("2026-09-25T06:08:00+07:00", 60),
        ("2026-09-25T06:09:00+07:00", 60),
        ("2026-09-25T06:10:00+07:00", 60),
        ("2026-09-25T06:11:00+07:00", 60),
    ]
    store.close()


def test_refresh_minute_rollups_reads_a_bounded_range_of_health_buckets(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    buckets = []
    for index in range(7800):
        start = at(6, 0) + timedelta(seconds=index * 6)
        buckets.append(
            bucket(start, (start, start + timedelta(seconds=1), start + timedelta(seconds=2)))
        )
    store.write_health_buckets(buckets)
    store.refresh_minute_rollups(DAY, at(19, 0))
    captured: list[str] = []
    store._connection.set_trace_callback(captured.append)

    store.refresh_minute_rollups(DAY, at(19, 1))
    clear_trace(store)
    statements = [s for s in captured if "FROM health_buckets" in s]

    assert len(statements) == 1
    earliest = re.search(r"bucket_start >= '([^']+)'", statements[0])
    latest = re.search(r"bucket_start < '([^']+)'", statements[0])
    assert earliest is not None
    assert latest is not None
    window_start = datetime.fromisoformat(earliest.group(1))
    window_end = datetime.fromisoformat(latest.group(1))
    assert window_start == at(18, 47)
    assert window_end == at(19, 0)
    assert window_end - window_start <= timedelta(
        seconds=MAX_HEALTH_BUCKET_SECONDS + ROLLUP_SETTLE_MINUTES * 60 + 60
    )
    assert rows(path, "SELECT COUNT(*) FROM minute_buckets") == [(780,)]
    store.close()


def test_refresh_minute_rollups_counts_a_crossing_that_carries_microseconds(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    _, tracking_session_id = opened_session(store)
    assert store.insert_event(event(tracking_session_id, 1, at(6, 0, 30, microsecond=905000)))
    assert store.insert_event(event(tracking_session_id, 2, at(6, 0, 30, microsecond=250000)))
    assert store.insert_event(event(tracking_session_id, 3, at(6, 0, 30, microsecond=1)))
    assert rows(path, "SELECT length(crossed_at) FROM events") == [(32,), (32,), (32,)]

    store.refresh_minute_rollups(DAY, at(6, 1))

    assert rows(path, "SELECT minute_start, event_count FROM minute_buckets") == [
        ("2026-09-25T06:00:00+07:00", 3),
        ("2026-09-25T06:01:00+07:00", 0),
    ]
    assert store.summary_counts(DAY, PeriodName.PAGI) == 3
    assert store.timeseries(DAY)[0].count == 3
    store.close()


def test_refresh_minute_rollups_separates_microsecond_crossings_across_a_minute_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    _, tracking_session_id = opened_session(store)
    moments = (
        at(6, 0, 0, microsecond=1),
        at(6, 0, 59, microsecond=905000),
        at(6, 1, 0, microsecond=1),
        at(6, 1, 0),
    )
    for index, moment in enumerate(moments, start=1):
        assert store.insert_event(event(tracking_session_id, index, moment))

    store.refresh_minute_rollups(DAY, at(6, 2))

    assert rows(path, "SELECT minute_start, event_count FROM minute_buckets") == [
        ("2026-09-25T06:00:00+07:00", 2),
        ("2026-09-25T06:01:00+07:00", 2),
        ("2026-09-25T06:02:00+07:00", 0),
    ]
    assert store.summary_counts(DAY, PeriodName.PAGI) == 4
    store.close()


def test_a_microsecond_crossing_survives_a_reopen_and_stays_aggregated(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    _, tracking_session_id = opened_session(store)
    assert store.insert_event(event(tracking_session_id, 1, at(6, 0, 30, microsecond=905000)))
    store.refresh_minute_rollups(DAY, at(6, 1))
    store.close()

    reopened = make_store(path)

    assert reopened.summary_counts(DAY, PeriodName.PAGI) == 1
    assert reopened.timeseries(DAY)[0].count == 1
    assert reopened.timeseries(DAY)[0].minute_start == at(6, 0)
    reopened.close()


def test_refresh_minute_rollups_still_counts_a_late_crossing_two_ticks_after_its_minute(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    _, tracking_session_id = opened_session(store)
    store.refresh_minute_rollups(DAY, at(6, 1))
    assert store.insert_event(event(tracking_session_id, 1, at(6, 0, 30, microsecond=905000)))
    store.refresh_minute_rollups(DAY, at(6, 2))
    assert rows(
        path,
        "SELECT event_count FROM minute_buckets WHERE minute_start = ?",
        (at(6, 0).isoformat(),),
    ) == [(1,)]

    store.refresh_minute_rollups(DAY, at(6, 3))

    assert rows(
        path,
        "SELECT event_count FROM minute_buckets WHERE minute_start = ?",
        (at(6, 0).isoformat(),),
    ) == [(1,)]
    assert store.summary_counts(DAY, PeriodName.PAGI) == 1
    store.close()


def test_a_late_crossing_is_picked_up_on_the_first_refresh_after_it_arrives(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    _, tracking_session_id = opened_session(store)
    store.refresh_minute_rollups(DAY, at(6, 10))
    assert store.insert_event(event(tracking_session_id, 1, at(6, 9, 30, microsecond=905000)))

    store.refresh_minute_rollups(DAY, at(6, 11))

    assert rows(
        path,
        "SELECT event_count FROM minute_buckets WHERE minute_start = ?",
        (at(6, 9).isoformat(),),
    ) == [(1,)]
    store.close()


def test_a_minute_stays_writable_until_the_tick_three_after_its_own(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    store.refresh_minute_rollups(DAY, at(6, 1))
    planted = at(6, 0).isoformat()

    def replant_and_refresh(now: datetime) -> int:
        with closing(connect(path)) as connection:
            connection.execute(
                "UPDATE minute_buckets SET event_count = 99 WHERE minute_start = ?", (planted,)
            )
            connection.commit()
        store.refresh_minute_rollups(DAY, now)
        return int(
            store._connection.execute(
                "SELECT event_count FROM minute_buckets WHERE minute_start = ?", (planted,)
            ).fetchone()[0]
        )

    assert replant_and_refresh(at(6, 2)) == 0
    assert replant_and_refresh(at(6, 3)) == 0
    assert replant_and_refresh(at(6, 4)) == 99
    assert replant_and_refresh(at(6, 5)) == 99
    store.close()


def test_refresh_minute_rollups_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    _, tracking_session_id = opened_session(store)
    assert store.insert_event(event(tracking_session_id, 1, at(6, 0, 1)))
    store.write_health_buckets((bucket(at(6, 0), span(at(6, 0), 60)),))
    store.refresh_minute_rollups(DAY, at(6, 1))

    store.refresh_minute_rollups(DAY, at(6, 1))
    store.refresh_minute_rollups(DAY, at(6, 1))

    assert rows(path, "SELECT minute_start, event_count, observed_seconds FROM minute_buckets") == [
        ("2026-09-25T06:00:00+07:00", 1, 60),
        ("2026-09-25T06:01:00+07:00", 0, 0),
    ]
    store.close()


def test_refresh_minute_rollups_picks_up_a_later_event(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    _, tracking_session_id = opened_session(store)
    store.refresh_minute_rollups(DAY, at(6, 1))
    assert store.insert_event(event(tracking_session_id, 1, at(6, 0, 30)))

    store.refresh_minute_rollups(DAY, at(6, 1))

    assert rows(
        path,
        "SELECT event_count FROM minute_buckets WHERE minute_start = ?",
        ("2026-09-25T06:00:00+07:00",),
    ) == [(1,)]
    store.close()


def test_refresh_minute_rollups_never_writes_a_future_minute(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    _, tracking_session_id = opened_session(store)
    assert store.insert_event(event(tracking_session_id, 1, at(6, 5, 10)))
    store.write_health_buckets((bucket(at(6, 5), span(at(6, 5), 60)),))

    store.refresh_minute_rollups(DAY, at(6, 2))

    assert rows(path, "SELECT minute_start FROM minute_buckets") == [
        ("2026-09-25T06:00:00+07:00",),
        ("2026-09-25T06:01:00+07:00",),
        ("2026-09-25T06:02:00+07:00",),
    ]
    store.close()


def test_refresh_minute_rollups_ignores_an_event_outside_the_reporting_schedule(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    app_session_id = store.start_app_session(at(5, 0))
    tracking_session_id = store.start_tracking_session(app_session_id, at(5, 0))
    assert store.insert_event(
        CountEvent(tracking_session_id, 1, at(5, 30), "up", PeriodName.PAGI, "car")
    )

    store.refresh_minute_rollups(DAY, at(6, 1))

    assert rows(path, "SELECT event_count FROM minute_buckets") == [(0,), (0,)]
    store.close()


def test_refresh_minute_rollups_ignores_a_report_day_that_is_not_the_current_day(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)

    store.refresh_minute_rollups(PRIOR_DAY, at(6, 1))

    assert rows(path, "SELECT COUNT(*) FROM minute_buckets") == [(0,)]
    store.close()


def test_refresh_minute_rollups_labels_every_minute_with_its_period(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)

    store.refresh_minute_rollups(DAY, at(15, 1))

    labelled = {
        str(row[0]): str(row[1])
        for row in rows(path, "SELECT minute_start, period FROM minute_buckets")
    }
    assert labelled["2026-09-25T09:59:00+07:00"] == "pagi"
    assert labelled["2026-09-25T10:00:00+07:00"] == "siang"
    assert labelled["2026-09-25T14:59:00+07:00"] == "siang"
    assert labelled["2026-09-25T15:00:00+07:00"] == "sore"
    store.close()


def test_refresh_minute_rollups_covers_every_elapsed_report_minute(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)

    store.refresh_minute_rollups(DAY, at(12, 0))

    assert rows(path, "SELECT COUNT(*) FROM minute_buckets") == [(361,)]
    store.close()


def test_refresh_minute_rollups_stops_at_the_last_reporting_minute(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)

    store.refresh_minute_rollups(DAY, at(23, 59, 59))

    stored = rows(path, "SELECT MAX(minute_start), MIN(minute_start) FROM minute_buckets")
    assert stored == [("2026-09-25T18:59:00+07:00", "2026-09-25T06:00:00+07:00")]
    assert rows(path, "SELECT COUNT(*) FROM minute_buckets") == [(780,)]
    store.close()


def test_refresh_minute_rollups_writes_nothing_before_the_schedule_opens(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)

    store.refresh_minute_rollups(DAY, at(5, 59, 59))

    assert rows(path, "SELECT COUNT(*) FROM minute_buckets") == [(0,)]
    store.close()


def test_refresh_minute_rollups_writes_one_row_at_the_opening_minute(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)

    store.refresh_minute_rollups(DAY, at(6, 0))

    assert rows(path, "SELECT minute_start FROM minute_buckets") == [("2026-09-25T06:00:00+07:00",)]
    store.close()


def test_summary_counts_counts_events_per_period(tmp_path: Path) -> None:
    store = make_store(tmp_path / "traffic.db")
    app_session_id = store.start_app_session(at(6, 0))
    tracking_session_id = store.start_tracking_session(app_session_id, at(6, 0))
    for index, moment in enumerate(
        (at(6, 1), at(9, 59), at(10, 0), at(14, 30), at(15, 0)), start=1
    ):
        assert store.insert_event(event(tracking_session_id, index, moment))

    assert store.summary_counts(DAY, PeriodName.PAGI) == 2
    assert store.summary_counts(DAY, PeriodName.SIANG) == 2
    assert store.summary_counts(DAY, PeriodName.SORE) == 1
    store.close()


def test_summary_counts_is_zero_for_a_period_without_events(tmp_path: Path) -> None:
    store = make_store(tmp_path / "traffic.db")

    assert store.summary_counts(DAY, PeriodName.PAGI) == 0
    assert store.summary_counts(DAY, PeriodName.SIANG) == 0
    assert store.summary_counts(DAY, PeriodName.SORE) == 0
    store.close()


def test_summary_counts_matches_the_dashboards_period_membership_rule(tmp_path: Path) -> None:
    from traffic_counter.density import build_period_summaries

    store = make_store(tmp_path / "traffic.db")
    app_session_id = store.start_app_session(at(6, 0))
    tracking_session_id = store.start_tracking_session(app_session_id, at(6, 0))
    stored: list[CountEvent] = []
    for index, moment in enumerate((at(6, 1), at(10, 0), at(15, 0), at(18, 59, 59)), start=1):
        record = event(tracking_session_id, index, moment)
        stored.append(record)
        assert store.insert_event(record)
    store.write_health_buckets((bucket(at(6, 0), span(at(6, 0), 60)),))
    summaries = build_period_summaries(stored, store.health_buckets(), at(18, 0))

    for summary in summaries:
        assert store.summary_counts(DAY, summary.period) == summary.total
    store.close()


def test_every_stored_event_period_matches_the_authoritative_rule(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    _, tracking_session_id = opened_session(store)
    for index, moment in enumerate(
        (at(6, 0), at(9, 59, 59), at(10, 0), at(14, 59, 59), at(15, 0), at(18, 59, 59)), start=1
    ):
        assert store.insert_event(event(tracking_session_id, index, moment))

    stored = rows(path, "SELECT crossed_at, period FROM events")

    for crossed_at, period in stored:
        assert str(period_for(datetime.fromisoformat(str(crossed_at)))) == str(period)
    store.close()


def test_summary_counts_follows_the_stored_label_not_the_timestamp(tmp_path: Path) -> None:
    from traffic_counter.density import build_period_summaries

    store = make_store(tmp_path / "traffic.db")
    _, tracking_session_id = opened_session(store)
    mislabelled = event(tracking_session_id, 1, at(12, 0), period=PeriodName.PAGI)
    assert store.insert_event(mislabelled)

    summaries = build_period_summaries(store.events(DAY), store.health_buckets(), at(12, 1))

    assert store.summary_counts(DAY, PeriodName.PAGI) == 1
    assert store.summary_counts(DAY, PeriodName.SIANG) == 0
    assert summaries[0].total == 0
    assert summaries[1].total == 1
    store.close()


def test_timeseries_returns_materialized_minutes_in_ascending_order(tmp_path: Path) -> None:
    store = make_store(tmp_path / "traffic.db")
    _, tracking_session_id = opened_session(store)
    assert store.insert_event(event(tracking_session_id, 1, at(6, 0, 30)))
    store.write_health_buckets((bucket(at(6, 0), span(at(6, 0), 45)),))
    store.refresh_minute_rollups(DAY, at(6, 2))

    points = store.timeseries(DAY)

    assert [point.minute_start for point in points] == [at(6, 0), at(6, 1), at(6, 2)]
    assert points[0].count == 1
    assert points[0].observed_seconds == 45
    assert points[0].period is PeriodName.PAGI
    assert points[1].count == 0
    assert points[1].observed_seconds == 0
    store.close()


def test_timeseries_is_empty_before_any_rollup(tmp_path: Path) -> None:
    store = make_store(tmp_path / "traffic.db")
    _, tracking_session_id = opened_session(store)
    assert store.insert_event(event(tracking_session_id, 1, at(6, 0)))

    assert store.timeseries(DAY) == ()
    store.close()


def test_timeseries_excludes_rows_from_another_day(tmp_path: Path) -> None:
    store = make_store(tmp_path / "traffic.db")
    store.refresh_minute_rollups(DAY, at(6, 1))

    assert store.timeseries(PRIOR_DAY) == ()
    assert store.timeseries(DAY + timedelta(days=1)) == ()
    assert len(store.timeseries(DAY)) == 2
    store.close()


def test_purge_before_keeps_the_current_day_and_deletes_older_rows(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    prior_app = store.start_app_session(at(6, 0, day=PRIOR_DAY))
    prior_tracking = store.start_tracking_session(prior_app, at(6, 0, day=PRIOR_DAY))
    current_app = store.start_app_session(at(6, 0))
    current_tracking = store.start_tracking_session(current_app, at(6, 0))
    assert store.insert_event(event(prior_tracking, 1, at(6, 1, day=PRIOR_DAY)))
    assert store.insert_event(event(current_tracking, 1, at(6, 1)))
    store.write_health_buckets(
        (
            bucket(at(6, 0, day=PRIOR_DAY), span(at(6, 0, day=PRIOR_DAY), 10)),
            bucket(at(6, 0), span(at(6, 0), 10)),
        )
    )
    store.refresh_minute_rollups(PRIOR_DAY, at(6, 1, day=PRIOR_DAY))
    store.refresh_minute_rollups(DAY, at(6, 1))

    store.purge_before(DAY)

    assert rows(path, "SELECT crossed_at FROM events") == [("2026-09-25T06:01:00+07:00",)]
    assert rows(path, "SELECT bucket_start FROM health_buckets") == [("2026-09-25T06:00:00+07:00",)]
    assert rows(path, "SELECT minute_start FROM minute_buckets") == [
        ("2026-09-25T06:00:00+07:00",),
        ("2026-09-25T06:01:00+07:00",),
    ]
    assert rows(path, "SELECT id FROM app_sessions") == [(current_app,)]
    assert rows(path, "SELECT id FROM tracking_sessions") == [(current_tracking,)]
    assert store.summary_counts(DAY, PeriodName.PAGI) == 1
    assert prior_app != current_app
    assert prior_tracking != current_tracking
    store.close()


def test_purge_before_deletes_prior_parent_sessions_that_hold_nothing(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    prior_app = store.start_app_session(at(6, 0, day=PRIOR_DAY))
    prior_tracking = store.start_tracking_session(prior_app, at(6, 0, day=PRIOR_DAY))
    store.stop_tracking_session(prior_tracking, at(18, 0, day=PRIOR_DAY))
    store.stop_app_session(prior_app, at(19, 5, day=PRIOR_DAY), AppState.STOPPED)
    current_app = store.start_app_session(at(6, 0))
    current_tracking = store.start_tracking_session(current_app, at(6, 0))

    store.purge_before(DAY)

    assert rows(path, "SELECT id FROM app_sessions") == [(current_app,)]
    assert rows(path, "SELECT id FROM tracking_sessions") == [(current_tracking,)]
    store.close()


def test_purge_before_keeps_an_older_session_that_still_holds_a_current_event(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    overnight_app = store.start_app_session(at(23, 0, day=PRIOR_DAY))
    overnight_tracking = store.start_tracking_session(overnight_app, at(23, 0, day=PRIOR_DAY))
    assert store.insert_event(event(overnight_tracking, 1, at(6, 0, 30)))

    store.purge_before(DAY)

    assert rows(path, "SELECT id FROM app_sessions") == [(overnight_app,)]
    assert rows(path, "SELECT id FROM tracking_sessions") == [(overnight_tracking,)]
    assert store.summary_counts(DAY, PeriodName.PAGI) == 1
    store.close()


def test_purge_before_keeps_everything_when_nothing_is_older(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    _, tracking_session_id = opened_session(store)
    assert store.insert_event(event(tracking_session_id, 1, at(6, 1)))
    store.write_health_buckets((bucket(at(6, 0), span(at(6, 0), 10)),))
    store.refresh_minute_rollups(DAY, at(6, 1))
    before = rows(path, "SELECT * FROM events")

    store.purge_before(DAY)

    assert rows(path, "SELECT * FROM events") == before
    assert rows(path, "SELECT COUNT(*) FROM minute_buckets") == [(2,)]
    assert rows(path, "SELECT COUNT(*) FROM health_buckets") == [(1,)]
    store.close()


def test_purge_before_deletes_nothing_from_a_later_day(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    _, tracking_session_id = opened_session(store)
    assert store.insert_event(event(tracking_session_id, 1, at(6, 1)))
    store.refresh_minute_rollups(DAY, at(6, 1))

    store.purge_before(PRIOR_DAY)

    assert rows(path, "SELECT COUNT(*) FROM events") == [(1,)]
    assert rows(path, "SELECT COUNT(*) FROM minute_buckets") == [(2,)]
    store.close()


def test_purge_before_keeps_rows_stamped_at_the_report_day_midnight(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    midnight = datetime.combine(DAY, clock_time.min, tzinfo=JAKARTA)
    app_session_id = store.start_app_session(midnight)
    tracking_session_id = store.start_tracking_session(app_session_id, midnight)
    assert store.insert_event(
        CountEvent(tracking_session_id, 1, midnight, "up", PeriodName.PAGI, "car")
    )
    store.write_health_buckets((HealthBucket(midnight, PeriodName.PAGI, (midnight,)),))
    store.refresh_minute_rollups(DAY, at(6, 0))

    store.purge_before(DAY)

    assert rows(path, "SELECT id FROM app_sessions") == [(app_session_id,)]
    assert rows(path, "SELECT id FROM tracking_sessions") == [(tracking_session_id,)]
    assert rows(path, "SELECT COUNT(*) FROM events") == [(1,)]
    assert rows(path, "SELECT COUNT(*) FROM health_buckets") == [(1,)]
    store.close()


def test_purge_before_keeps_an_eventless_session_started_at_the_report_day_midnight(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    midnight = datetime.combine(DAY, clock_time.min, tzinfo=JAKARTA)
    app_session_id = store.start_app_session(midnight)
    tracking_session_id = store.start_tracking_session(app_session_id, midnight)

    store.purge_before(DAY)

    assert rows(path, "SELECT id FROM app_sessions") == [(app_session_id,)]
    assert rows(path, "SELECT id FROM tracking_sessions") == [(tracking_session_id,)]
    store.close()


def test_purge_before_keeps_a_childless_app_session_started_at_the_report_day_midnight(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    midnight = datetime.combine(DAY, clock_time.min, tzinfo=JAKARTA)
    app_session_id = store.start_app_session(midnight)
    prior_app_session_id = store.start_app_session(at(6, 0, day=PRIOR_DAY))

    store.purge_before(DAY)

    assert rows(path, "SELECT id FROM app_sessions") == [(app_session_id,)]
    assert prior_app_session_id != app_session_id
    store.close()


def test_purge_before_keeps_a_minute_bucket_stamped_at_the_report_day_midnight(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    midnight_text = datetime.combine(DAY, clock_time.min, tzinfo=JAKARTA).isoformat()
    with closing(connect(path)) as connection:
        connection.execute(
            "INSERT INTO minute_buckets(minute_start, period, event_count, observed_seconds)"
            " VALUES(?, 'pagi', 0, 0)",
            (midnight_text,),
        )
        connection.commit()

    store.purge_before(DAY)

    assert rows(path, "SELECT minute_start FROM minute_buckets") == [(midnight_text,)]
    store.close()


def test_purge_before_deletes_a_session_started_one_second_before_midnight(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    midnight = datetime.combine(DAY, clock_time.min, tzinfo=JAKARTA)
    app_session_id = store.start_app_session(midnight - timedelta(seconds=1))

    store.purge_before(DAY)

    assert rows(path, "SELECT COUNT(*) FROM app_sessions") == [(0,)]
    assert app_session_id == 1
    store.close()


def test_timeseries_sorts_minutes_stored_out_of_order(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    with closing(connect(path)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executemany(
            "INSERT INTO minute_buckets(minute_start, period, event_count, observed_seconds)"
            " VALUES(?, ?, ?, ?)",
            [
                ("2026-09-25T06:03:00+07:00", "pagi", 3, 60),
                ("2026-09-25T06:01:00+07:00", "pagi", 1, 60),
                ("2026-09-25T06:02:00+07:00", "pagi", 2, 60),
            ],
        )
        connection.commit()

    points = store.timeseries(DAY)

    assert [point.minute_start for point in points] == [at(6, 1), at(6, 2), at(6, 3)]
    assert [point.count for point in points] == [1, 2, 3]
    store.close()


def test_purge_before_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    prior_app = store.start_app_session(at(6, 0, day=PRIOR_DAY))
    store.start_tracking_session(prior_app, at(6, 0, day=PRIOR_DAY))

    store.purge_before(DAY)
    store.purge_before(DAY)
    store.purge_before(DAY)

    assert rows(path, "SELECT COUNT(*) FROM app_sessions") == [(0,)]
    assert rows(path, "SELECT COUNT(*) FROM tracking_sessions") == [(0,)]
    store.close()


def test_purge_before_is_allowed_on_an_empty_store(tmp_path: Path) -> None:
    store = make_store(tmp_path / "traffic.db")

    store.purge_before(DAY)

    assert store.summary_counts(DAY, PeriodName.PAGI) == 0
    store.close()


def test_stored_timestamps_sort_lexically_because_jakarta_has_no_daylight_saving(
    tmp_path: Path,
) -> None:
    from datetime import UTC

    moments = (
        datetime(2026, 9, 25, 6, 0, tzinfo=JAKARTA),
        datetime(2026, 9, 24, 23, 0, tzinfo=UTC),
        datetime(2026, 9, 25, 19, 0, tzinfo=JAKARTA),
        datetime(2026, 9, 25, 0, 0, tzinfo=JAKARTA),
    )
    instants = sorted(moments)
    assert instants == sorted(moments, key=lambda moment: moment.astimezone(JAKARTA))
    assert len({moment.astimezone(JAKARTA).isoformat()[-6:] for moment in moments}) == 1


def test_events_and_minutes_survive_a_reopen(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    _, tracking_session_id = opened_session(store)
    assert store.insert_event(event(tracking_session_id, 1, at(6, 0, 30)))
    store.write_health_buckets((bucket(at(6, 0), span(at(6, 0), 60)),))
    store.refresh_minute_rollups(DAY, at(6, 1))
    store.close()

    reopened = make_store(path)
    assert reopened.summary_counts(DAY, PeriodName.PAGI) == 1
    assert reopened.health_buckets()[0].seconds == span(at(6, 0), 60)
    assert len(reopened.timeseries(DAY)) == 2
    assert reopened.insert_event(event(tracking_session_id, 1, at(6, 2))) is False
    assert reopened.insert_event(event(tracking_session_id, 2, at(6, 2))) is True
    reopened.close()


def test_a_store_created_on_one_thread_is_usable_from_another(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"
    store = make_store(path)
    app_session_id, tracking_session_id = opened_session(store)
    failures: list[str] = []

    def worker() -> None:
        try:
            if store.insert_event(event(tracking_session_id, 1, at(6, 1))) is not True:
                failures.append("insert_event returned False on the worker thread")
            store.write_health_buckets((bucket(at(6, 0), span(at(6, 0), 10)),))
            store.refresh_minute_rollups(DAY, at(6, 2))
            if store.summary_counts(DAY, PeriodName.PAGI) != 1:
                failures.append("summary_counts disagreed on the worker thread")
            if len(store.timeseries(DAY)) != 3:
                failures.append("timeseries disagreed on the worker thread")
        except Exception as error:
            failures.append(f"{type(error).__name__}: {error}")

    thread = Thread(target=worker)
    thread.start()
    thread.join()

    assert failures == []
    assert app_session_id == 1
    assert store.summary_counts(DAY, PeriodName.PAGI) == 1
    store.close()


def test_close_is_idempotent(tmp_path: Path) -> None:
    store = make_store(tmp_path / "traffic.db")

    store.close()
    store.close()
    store.close()


def test_use_after_close_is_rejected(tmp_path: Path) -> None:
    store = make_store(tmp_path / "traffic.db")
    app_session_id, _ = opened_session(store)
    store.close()

    with pytest.raises(RuntimeError, match="initialized"):
        store.summary_counts(DAY, PeriodName.PAGI)
    with pytest.raises(RuntimeError, match="initialized"):
        store.start_app_session(at(6, 1))
    with pytest.raises(RuntimeError, match="initialized"):
        store.stop_app_session(app_session_id, at(6, 1), AppState.STOPPED)
    with pytest.raises(RuntimeError, match="initialized"):
        store.insert_event(event(1, 1, at(6, 1)))
    with pytest.raises(RuntimeError, match="initialized"):
        store.write_health_buckets((bucket(at(6, 0), span(at(6, 0), 1)),))
    with pytest.raises(RuntimeError, match="initialized"):
        store.refresh_minute_rollups(DAY, at(6, 1))
    with pytest.raises(RuntimeError, match="initialized"):
        store.timeseries(DAY)
    with pytest.raises(RuntimeError, match="initialized"):
        store.health_buckets()
    with pytest.raises(RuntimeError, match="initialized"):
        store.events(DAY)
    with pytest.raises(RuntimeError, match="initialized"):
        store.purge_before(DAY)


def test_the_context_manager_initializes_and_closes(tmp_path: Path) -> None:
    path = tmp_path / "traffic.db"

    with TrafficStore(path) as managed:
        managed.initialize()
        assert managed.summary_counts(DAY, PeriodName.PAGI) == 0

    with pytest.raises(RuntimeError, match="initialized"):
        managed.summary_counts(DAY, PeriodName.PAGI)
