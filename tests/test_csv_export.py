import csv
import os
import re
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from sqlite3 import connect
from typing import Final

import pytest

from traffic_counter.models import CountEvent, HealthBucket, PeriodName
from traffic_counter.storage import CsvExporter, TrafficStore
from traffic_counter.timeutils import JAKARTA, period_for

DAY: Final[date] = date(2026, 9, 25)
PRIOR_DAY: Final[date] = date(2026, 9, 24)
LATER_DAY: Final[date] = date(2026, 9, 26)
COLUMNS: Final[tuple[str, ...]] = (
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
MINUTE_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+07:00$")


def at(
    hour: int, minute: int = 0, second: int = 0, day: date = DAY, microsecond: int = 0
) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, second, microsecond, tzinfo=JAKARTA)


def span(start: datetime, count: int) -> tuple[datetime, ...]:
    return tuple(start + timedelta(seconds=offset) for offset in range(count))


def event(tracking_session_id: int, track_id: int, moment: datetime) -> CountEvent:
    resolved = period_for(moment)
    assert resolved is not None
    return CountEvent(
        tracking_session_id=tracking_session_id,
        track_id=track_id,
        crossed_at=moment,
        direction="up",
        period=resolved,
        class_name="car",
    )


def bucket(start: datetime, seconds: tuple[datetime, ...]) -> HealthBucket:
    resolved = period_for(start)
    assert resolved is not None
    return HealthBucket(bucket_start=start, period=resolved, seconds=seconds)


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [dict(zip(header, values, strict=True)) for values in reader]
    return header, rows


@contextmanager
def prepared(tmp_path: Path) -> Iterator[tuple[TrafficStore, CsvExporter]]:
    store = TrafficStore(tmp_path / "traffic.db")
    store.initialize()
    exporter = CsvExporter(store, tmp_path / "exports")
    try:
        yield store, exporter
    finally:
        store.close()


def seed(store: TrafficStore, records: tuple[tuple[int, datetime], ...]) -> None:
    app_session_id = store.start_app_session(at(6, 0))
    tracking_session_id = store.start_tracking_session(app_session_id, at(6, 0))
    for track_id, moment in records:
        assert store.insert_event(event(tracking_session_id, track_id, moment))


def test_csv_contains_zero_minutes_and_cumulative_period_total(tmp_path: Path) -> None:
    with prepared(tmp_path) as (store, exporter):
        path = exporter.export(DAY, now=at(6, 2))

        header, rows = read(path)
        totals = store.summary_counts(DAY, PeriodName.PAGI)

    assert header == list(COLUMNS)
    assert [row["minute_start"] for row in rows] == [
        "2026-09-25T06:00:00+07:00",
        "2026-09-25T06:01:00+07:00",
        "2026-09-25T06:02:00+07:00",
    ]
    assert rows[0]["count"] == "0"
    assert rows[0]["period_total"] == "0"
    assert totals == 0


def test_export_writes_the_expected_filename_and_creates_the_directory(tmp_path: Path) -> None:
    with prepared(tmp_path) as (_, exporter):
        path = exporter.export(DAY, now=at(6, 0))

    assert path == tmp_path / "exports" / "traffic_2026-09-25.csv"
    assert path.is_file()
    assert path.parent.is_dir()


def test_export_returns_the_sibling_path_of_the_report_day(tmp_path: Path) -> None:
    with prepared(tmp_path) as (_, exporter):
        first = exporter.export(DAY, now=at(6, 0))
        second = exporter.export(PRIOR_DAY, now=at(6, 0, day=PRIOR_DAY))

    assert first.name == "traffic_2026-09-25.csv"
    assert second.name == "traffic_2026-09-24.csv"


def test_csv_date_column_repeats_the_report_day(tmp_path: Path) -> None:
    with prepared(tmp_path) as (_, exporter):
        path = exporter.export(DAY, now=at(6, 2))

        _, rows = read(path)

    assert {row["date"] for row in rows} == {"2026-09-25"}


def test_csv_column_order_is_exact(tmp_path: Path) -> None:
    with prepared(tmp_path) as (_, exporter):
        path = exporter.export(DAY, now=at(6, 0))

        text = path.read_text(encoding="utf-8")

    assert text.splitlines()[0] == ",".join(COLUMNS)


def test_csv_omits_the_rolling_rate_column(tmp_path: Path) -> None:
    with prepared(tmp_path) as (_, exporter):
        path = exporter.export(DAY, now=at(6, 1))

        header, _ = read(path)

    assert not any("rolling" in column for column in header)
    assert len(header) == 9


def test_csv_uses_crlf_record_separators(tmp_path: Path) -> None:
    with prepared(tmp_path) as (_, exporter):
        path = exporter.export(DAY, now=at(6, 1))

        raw = path.read_bytes()

    assert raw.count(b"\r\n") == 3
    assert b"\n" not in raw.replace(b"\r\n", b"")


def test_every_minute_start_carries_the_seven_hundred_offset(tmp_path: Path) -> None:
    with prepared(tmp_path) as (store, exporter):
        seed(store, ((1, at(6, 0, 1)),))
        path = exporter.export(DAY, now=at(6, 3))

        _, rows = read(path)

    assert rows
    assert all(MINUTE_PATTERN.match(row["minute_start"]) for row in rows)
    assert all(row["minute_start"].endswith("+07:00") for row in rows)


def test_csv_writes_no_future_minutes(tmp_path: Path) -> None:
    with prepared(tmp_path) as (store, exporter):
        seed(store, ((1, at(6, 30, 1)),))
        path = exporter.export(DAY, now=at(6, 2))

        _, rows = read(path)
        stored_total = store.summary_counts(DAY, PeriodName.PAGI)

    assert [row["minute_start"] for row in rows] == [
        "2026-09-25T06:00:00+07:00",
        "2026-09-25T06:01:00+07:00",
        "2026-09-25T06:02:00+07:00",
    ]
    assert stored_total == 1


def test_csv_covers_every_elapsed_report_minute(tmp_path: Path) -> None:
    with prepared(tmp_path) as (_, exporter):
        path = exporter.export(DAY, now=at(12, 0))

        _, rows = read(path)

    assert len(rows) == 361
    assert rows[0]["minute_start"] == "2026-09-25T06:00:00+07:00"
    assert rows[-1]["minute_start"] == "2026-09-25T12:00:00+07:00"


def test_csv_caps_at_the_last_reporting_minute(tmp_path: Path) -> None:
    with prepared(tmp_path) as (_, exporter):
        path = exporter.export(DAY, now=at(23, 59, 59))

        _, rows = read(path)

    assert len(rows) == 780
    assert rows[-1]["minute_start"] == "2026-09-25T18:59:00+07:00"


def test_csv_before_the_schedule_opens_is_a_header_only_file(tmp_path: Path) -> None:
    with prepared(tmp_path) as (_, exporter):
        path = exporter.export(DAY, now=at(5, 59, 59))

        header, rows = read(path)

    assert header == list(COLUMNS)
    assert rows == []


def test_csv_at_the_opening_minute_has_exactly_one_row(tmp_path: Path) -> None:
    with prepared(tmp_path) as (_, exporter):
        path = exporter.export(DAY, now=at(6, 0))

        _, rows = read(path)

    assert len(rows) == 1
    assert rows[0]["minute_start"] == "2026-09-25T06:00:00+07:00"


def test_csv_null_rates_are_empty_fields(tmp_path: Path) -> None:
    with prepared(tmp_path) as (store, exporter):
        seed(store, ((1, at(6, 0, 1)),))
        path = exporter.export(DAY, now=at(6, 1))

        _, rows = read(path)

    assert rows[0]["count"] == "1"
    assert rows[0]["observed_seconds"] == "0"
    assert rows[0]["minute_rate"] == ""
    assert rows[0]["period_total"] == "1"
    assert rows[0]["period_observed_seconds"] == "0"
    assert rows[0]["period_rate"] == ""


def test_csv_minute_rate_is_count_over_observed_seconds_times_sixty(tmp_path: Path) -> None:
    with prepared(tmp_path) as (store, exporter):
        seed(store, ((1, at(6, 0, 1)), (2, at(6, 0, 40))))
        store.write_health_buckets((bucket(at(6, 0), span(at(6, 0), 60)),))
        path = exporter.export(DAY, now=at(6, 1))

        _, rows = read(path)

    assert rows[0]["observed_seconds"] == "60"
    assert rows[0]["minute_rate"] == "2.0"
    assert rows[0]["period_rate"] == "2.0"
    assert rows[1]["observed_seconds"] == "0"
    assert rows[1]["minute_rate"] == ""


def test_csv_zero_events_with_healthy_time_is_zero_not_empty(tmp_path: Path) -> None:
    with prepared(tmp_path) as (store, exporter):
        store.write_health_buckets((bucket(at(6, 0), span(at(6, 0), 60)),))
        path = exporter.export(DAY, now=at(6, 1))

        _, rows = read(path)

    assert rows[0]["count"] == "0"
    assert rows[0]["minute_rate"] == "0.0"
    assert rows[0]["period_rate"] == "0.0"


def test_csv_period_total_is_cumulative_within_the_period(tmp_path: Path) -> None:
    with prepared(tmp_path) as (store, exporter):
        seed(store, ((1, at(6, 0, 1)), (2, at(6, 1, 1)), (3, at(6, 2, 1))))
        path = exporter.export(DAY, now=at(6, 4))

        _, rows = read(path)

    assert [row["count"] for row in rows] == ["1", "1", "1", "0", "0"]
    assert [row["period_total"] for row in rows] == ["1", "2", "3", "3", "3"]


def test_csv_period_total_resets_at_the_siang_boundary(tmp_path: Path) -> None:
    with prepared(tmp_path) as (store, exporter):
        seed(store, ((1, at(9, 59, 59)), (2, at(10, 0, 1)), (3, at(10, 1, 1))))
        path = exporter.export(DAY, now=at(10, 2))

        _, rows = read(path)

    boundary = [row for row in rows if row["minute_start"] == "2026-09-25T10:00:00+07:00"]
    last_pagi = [row for row in rows if row["period"] == "pagi"][-1]
    assert boundary[0]["period"] == "siang"
    assert boundary[0]["period_total"] == "1"
    assert rows[-1]["period_total"] == "2"
    assert last_pagi["minute_start"] == "2026-09-25T09:59:00+07:00"
    assert last_pagi["period_total"] == "1"


def test_csv_labels_every_minute_with_its_period(tmp_path: Path) -> None:
    with prepared(tmp_path) as (_, exporter):
        path = exporter.export(DAY, now=at(15, 1))

        _, rows = read(path)

    labelled = {row["minute_start"]: row["period"] for row in rows}
    assert labelled["2026-09-25T09:59:00+07:00"] == "pagi"
    assert labelled["2026-09-25T10:00:00+07:00"] == "siang"
    assert labelled["2026-09-25T14:59:00+07:00"] == "siang"
    assert labelled["2026-09-25T15:00:00+07:00"] == "sore"


def test_csv_period_observed_seconds_is_cumulative(tmp_path: Path) -> None:
    with prepared(tmp_path) as (store, exporter):
        store.write_health_buckets(
            (
                bucket(at(6, 0), span(at(6, 0), 60)),
                bucket(at(6, 1), span(at(6, 1), 30)),
            )
        )
        path = exporter.export(DAY, now=at(6, 3))

        _, rows = read(path)

    assert [row["observed_seconds"] for row in rows] == ["60", "30", "0", "0"]
    assert [row["period_observed_seconds"] for row in rows] == ["60", "90", "90", "90"]


def test_csv_period_rate_is_the_cumulative_ratio(tmp_path: Path) -> None:
    with prepared(tmp_path) as (store, exporter):
        seed(store, ((1, at(6, 0, 1)), (2, at(6, 1, 1)), (3, at(6, 1, 2))))
        store.write_health_buckets(
            (
                bucket(at(6, 0), span(at(6, 0), 60)),
                bucket(at(6, 1), span(at(6, 1), 60)),
            )
        )
        path = exporter.export(DAY, now=at(6, 2))

        _, rows = read(path)

    assert rows[0]["period_total"] == "1"
    assert rows[0]["period_observed_seconds"] == "60"
    assert rows[0]["period_rate"] == "1.0"
    assert rows[1]["period_total"] == "3"
    assert rows[1]["period_observed_seconds"] == "120"
    assert rows[1]["period_rate"] == "1.5"


def test_csv_counts_sum_to_the_database_event_total(tmp_path: Path) -> None:
    with prepared(tmp_path) as (store, exporter):
        seed(
            store,
            (
                (1, at(6, 0, 1)),
                (2, at(6, 0, 59)),
                (3, at(6, 1, 0)),
                (4, at(9, 59, 59)),
                (5, at(10, 0, 0)),
                (6, at(15, 0, 0)),
                (7, at(18, 59, 59)),
            ),
        )
        path = exporter.export(DAY, now=at(19, 30))

        _, rows = read(path)
        database_total = sum(store.summary_counts(DAY, period) for period in PeriodName)
        minute_total = sum(int(row["count"]) for row in rows)
        final_totals = {
            period: [int(row["period_total"]) for row in rows if row["period"] == period][-1]
            for period in (PeriodName.PAGI, PeriodName.SIANG, PeriodName.SORE)
        }

    assert minute_total == database_total == 7
    assert final_totals == {PeriodName.PAGI: 4, PeriodName.SIANG: 1, PeriodName.SORE: 2}


def test_csv_observed_seconds_match_the_stored_rollups(tmp_path: Path) -> None:
    with prepared(tmp_path) as (store, exporter):
        seed(store, ((1, at(6, 0, 1)),))
        store.write_health_buckets((bucket(at(6, 0), span(at(6, 0), 45)),))
        store.refresh_minute_rollups(DAY, at(6, 2))
        rollup_seconds = sum(point.observed_seconds for point in store.timeseries(DAY))

        path = exporter.export(DAY, now=at(6, 2))
        _, rows = read(path)
        csv_seconds = sum(int(row["observed_seconds"]) for row in rows)

    assert rollup_seconds == 45
    assert csv_seconds == 45
    assert rows[0]["observed_seconds"] == "45"


def test_csv_matches_the_database_with_and_without_a_rollup(tmp_path: Path) -> None:
    with prepared(tmp_path) as (store, exporter):
        seed(store, ((1, at(6, 0, 1)), (2, at(6, 1, 30))))
        store.write_health_buckets((bucket(at(6, 0), span(at(6, 0), 30)),))

        before = read(exporter.export(DAY, now=at(6, 2)))[1]
        store.refresh_minute_rollups(DAY, at(6, 2))
        after = read(exporter.export(DAY, now=at(6, 2)))[1]

    assert before == after


def test_csv_keeps_stored_observed_seconds_for_minutes_before_the_refresh_cursor(
    tmp_path: Path,
) -> None:
    with prepared(tmp_path) as (store, exporter):
        seed(store, ((1, at(6, 0, 1)),))
        store.write_health_buckets((bucket(at(6, 0), span(at(6, 0), 30)),))
        store.refresh_minute_rollups(DAY, at(6, 2))
        first = read(exporter.export(DAY, now=at(6, 2)))[1]
        later = read(exporter.export(DAY, now=at(6, 4)))[1]

    assert first[0]["observed_seconds"] == "30"
    assert [row["observed_seconds"] for row in later] == ["30", "0", "0", "0", "0"]
    assert later[0]["period_total"] == "1"


def test_csv_keeps_a_stored_minute_value_that_a_later_refresh_would_overwrite(
    tmp_path: Path,
) -> None:
    with prepared(tmp_path) as (store, exporter):
        store.refresh_minute_rollups(DAY, at(6, 3))
        stale_minute = "2026-09-25T06:00:00+07:00"
        with closing(connect(tmp_path / "traffic.db")) as connection:
            connection.execute(
                "UPDATE minute_buckets SET event_count = 9, observed_seconds = 11"
                " WHERE minute_start = ?",
                (stale_minute,),
            )
            connection.commit()

        path = exporter.export(DAY, now=at(6, 5))
        _, rows = read(path)
        stored = {
            point.minute_start: (point.count, point.observed_seconds)
            for point in store.timeseries(DAY)
        }

    assert stored[at(6, 0)] == (9, 11)
    assert [row for row in rows if row["minute_start"] == stale_minute] == [
        {
            "date": "2026-09-25",
            "period": "pagi",
            "minute_start": stale_minute,
            "count": "9",
            "observed_seconds": "11",
            "minute_rate": str(9 / 11 * 60),
            "period_total": "9",
            "period_observed_seconds": "11",
            "period_rate": str(9 / 11 * 60),
        }
    ]


def test_csv_count_includes_a_crossing_that_carries_microseconds(tmp_path: Path) -> None:
    with prepared(tmp_path) as (store, exporter):
        seed(store, ((1, at(6, 0, 30, microsecond=905000)),))
        store.write_health_buckets((bucket(at(6, 0), span(at(6, 0), 60)),))

        path = exporter.export(DAY, now=at(6, 1))
        _, rows = read(path)
        database_total = store.summary_counts(DAY, PeriodName.PAGI)
        csv_total = sum(int(row["count"]) for row in rows)

    assert rows[0]["minute_start"] == "2026-09-25T06:00:00+07:00"
    assert rows[0]["count"] == "1"
    assert rows[0]["minute_rate"] == "1.0"
    assert rows[0]["period_total"] == "1"
    assert database_total == 1
    assert csv_total == database_total


def test_csv_separates_microsecond_crossings_across_a_minute_boundary(tmp_path: Path) -> None:
    with prepared(tmp_path) as (store, exporter):
        seed(
            store,
            (
                (1, at(6, 0, 59, microsecond=905000)),
                (2, at(6, 1, 0, microsecond=1)),
            ),
        )

        path = exporter.export(DAY, now=at(6, 2))
        _, rows = read(path)
        database_total = store.summary_counts(DAY, PeriodName.PAGI)
        csv_total = sum(int(row["count"]) for row in rows)

    assert [row["count"] for row in rows] == ["1", "1", "0"]
    assert [row["period_total"] for row in rows] == ["1", "2", "2"]
    assert csv_total == database_total == 2


def test_csv_export_replaces_the_previous_file_in_place(tmp_path: Path) -> None:
    with prepared(tmp_path) as (store, exporter):
        first = exporter.export(DAY, now=at(6, 0))
        first_text = first.read_text(encoding="utf-8")
        seed(store, ((1, at(6, 0, 30)),))

        second = exporter.export(DAY, now=at(6, 1))
        second_text = second.read_text(encoding="utf-8")

    assert first == second
    assert second_text != first_text
    assert len(first_text.splitlines()) == 2
    assert len(second_text.splitlines()) == 3
    assert second_text.splitlines()[2].startswith(
        "2026-09-25,pagi,2026-09-25T06:01:00+07:00,0,0,,1,0,"
    )


def test_csv_export_leaves_no_temporary_file_behind(tmp_path: Path) -> None:
    with prepared(tmp_path) as (_, exporter):
        path = exporter.export(DAY, now=at(6, 1))

        listing = sorted(item.name for item in path.parent.iterdir())

    assert listing == ["traffic_2026-09-25.csv"]


def test_csv_export_rewrites_a_stale_temporary_file(tmp_path: Path) -> None:
    with prepared(tmp_path) as (_, exporter):
        exporter.export(DAY, now=at(6, 0))
        stale = exporter.export(DAY, now=at(6, 1)).with_name("traffic_2026-09-25.csv.tmp")
        stale.write_text("garbage", encoding="utf-8")

        path = exporter.export(DAY, now=at(6, 1))
        header, rows = read(path)
        listing = sorted(item.name for item in path.parent.iterdir())

    assert header == list(COLUMNS)
    assert len(rows) == 2
    assert listing == ["traffic_2026-09-25.csv"]


def test_try_export_returns_true_and_writes_the_file(tmp_path: Path) -> None:
    with prepared(tmp_path) as (store, exporter):
        seed(store, ((1, at(6, 0, 1)),))

        assert exporter.try_export(DAY, at(6, 1)) is True

        path = tmp_path / "exports" / "traffic_2026-09-25.csv"
        _, rows = read(path)

    assert len(rows) == 2
    assert rows[0]["count"] == "1"


def test_try_export_returns_false_and_preserves_the_prior_file_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with prepared(tmp_path) as (store, exporter):
        seed(store, ((1, at(6, 0, 1)),))
        path = exporter.export(DAY, now=at(6, 1))
        original = path.read_text(encoding="utf-8")
        assert store.insert_event(event(1, 2, at(6, 1, 1))) is True

        def failing_replace(source: object, destination: object) -> None:
            raise OSError("simulated transient replace failure")

        monkeypatch.setattr(os, "replace", failing_replace)

        assert exporter.try_export(DAY, at(6, 2)) is False
        assert path.read_text(encoding="utf-8") == original
        assert path.is_file()
        listing = sorted(item.name for item in path.parent.iterdir())
        database_total = store.summary_counts(DAY, PeriodName.PAGI)

    assert listing == ["traffic_2026-09-25.csv"]
    assert database_total == 2


def test_try_export_succeeds_on_a_later_call_after_a_transient_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with prepared(tmp_path) as (store, exporter):
        seed(store, ((1, at(6, 0, 1)),))
        path = exporter.export(DAY, now=at(6, 1))
        original = path.read_text(encoding="utf-8")
        calls: list[tuple[object, object]] = []
        real_replace = os.replace

        def flaky_replace(source: object, destination: object) -> None:
            calls.append((source, destination))
            if len(calls) == 1:
                raise OSError("simulated transient replace failure")
            real_replace(source, destination)

        monkeypatch.setattr(os, "replace", flaky_replace)

        assert exporter.try_export(DAY, at(6, 2)) is False
        assert path.read_text(encoding="utf-8") == original
        assert exporter.try_export(DAY, at(6, 3)) is True
        assert path.read_text(encoding="utf-8") != original
        assert len(calls) == 2
        assert [os.fspath(item) for item in calls[0]] == [
            os.fspath(path) + ".tmp",
            os.fspath(path),
        ]
        assert len(read(path)[1]) == 4


def test_try_export_returns_false_when_the_directory_cannot_be_created(tmp_path: Path) -> None:
    store = TrafficStore(tmp_path / "traffic.db")
    store.initialize()
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    exporter = CsvExporter(store, blocker)

    assert exporter.try_export(DAY, at(6, 1)) is False
    store.close()


def test_try_export_propagates_a_non_os_error(tmp_path: Path) -> None:
    with prepared(tmp_path) as (_, exporter), pytest.raises(ValueError, match="report day"):
        exporter.try_export(DAY, at(6, 1, day=LATER_DAY))


def test_export_rejects_a_report_day_that_is_not_the_current_date(tmp_path: Path) -> None:
    with prepared(tmp_path) as (_, exporter):
        with pytest.raises(ValueError, match="report day"):
            exporter.export(DAY, now=at(6, 1, day=LATER_DAY))
        with pytest.raises(ValueError, match="report day"):
            exporter.export(DAY, now=at(6, 1, day=PRIOR_DAY))

        assert not (tmp_path / "exports").exists()


def test_export_normalizes_a_non_jakarta_now(tmp_path: Path) -> None:
    from datetime import UTC

    with prepared(tmp_path) as (_, exporter):
        path = exporter.export(DAY, now=datetime(2026, 9, 24, 23, 2, tzinfo=UTC))

        _, rows = read(path)

    assert [row["minute_start"] for row in rows] == [
        "2026-09-25T06:00:00+07:00",
        "2026-09-25T06:01:00+07:00",
        "2026-09-25T06:02:00+07:00",
    ]


def test_export_rejects_a_naive_now(tmp_path: Path) -> None:
    with prepared(tmp_path) as (_, exporter), pytest.raises(ValueError, match="timezone-aware"):
        exporter.export(DAY, now=datetime(2026, 9, 25, 6, 1))


def test_purge_before_deletes_only_files_from_other_dates(tmp_path: Path) -> None:
    with prepared(tmp_path) as (_, exporter):
        for day in (DAY, PRIOR_DAY, LATER_DAY):
            exporter.export(day, now=at(6, 0, day=day))

        exporter.purge_before(DAY)

        remaining = sorted(item.name for item in (tmp_path / "exports").iterdir())

    assert remaining == ["traffic_2026-09-25.csv"]


def test_purge_before_keeps_the_current_file_untouched(tmp_path: Path) -> None:
    with prepared(tmp_path) as (_, exporter):
        path = exporter.export(DAY, now=at(6, 1))
        original = path.read_text(encoding="utf-8")

        exporter.purge_before(DAY)

    assert path.is_file()
    assert path.read_text(encoding="utf-8") == original


def test_purge_before_ignores_files_that_are_not_traffic_exports(tmp_path: Path) -> None:
    directory = tmp_path / "exports"
    directory.mkdir()
    keeper = directory / "notes.csv"
    keeper.write_text("keep me", encoding="utf-8")
    calibration = directory / "line-2026-09-25.jpg"
    calibration.write_text("keep me", encoding="utf-8")
    with prepared(tmp_path) as (_, exporter):
        exporter.export(PRIOR_DAY, now=at(6, 0, day=PRIOR_DAY))

        exporter.purge_before(DAY)

        remaining = sorted(item.name for item in directory.iterdir())

    assert remaining == ["line-2026-09-25.jpg", "notes.csv"]


def test_purge_before_ignores_a_traffic_file_without_a_parsable_date(tmp_path: Path) -> None:
    directory = tmp_path / "exports"
    directory.mkdir()
    odd = directory / "traffic_archive.csv"
    odd.write_text("keep me", encoding="utf-8")
    wrong_shape = directory / "traffic_2026-9-5.csv"
    wrong_shape.write_text("keep me", encoding="utf-8")
    with prepared(tmp_path) as (_, exporter):
        exporter.export(PRIOR_DAY, now=at(6, 0, day=PRIOR_DAY))

        exporter.purge_before(DAY)

        remaining = sorted(item.name for item in directory.iterdir())

    assert remaining == ["traffic_2026-9-5.csv", "traffic_archive.csv"]


def test_purge_before_is_idempotent(tmp_path: Path) -> None:
    with prepared(tmp_path) as (_, exporter):
        exporter.export(PRIOR_DAY, now=at(6, 0, day=PRIOR_DAY))

        exporter.purge_before(DAY)
        exporter.purge_before(DAY)
        exporter.purge_before(DAY)

        remaining = sorted(item.name for item in (tmp_path / "exports").iterdir())

    assert remaining == []


def test_purge_before_on_a_missing_directory_is_a_no_op(tmp_path: Path) -> None:
    store = TrafficStore(tmp_path / "traffic.db")
    store.initialize()
    exporter = CsvExporter(store, tmp_path / "never-created")

    exporter.purge_before(DAY)
    exporter.purge_before(DAY)

    assert not (tmp_path / "never-created").exists()
    store.close()


def test_purge_before_can_run_before_the_first_export(tmp_path: Path) -> None:
    with prepared(tmp_path) as (_, exporter):
        exporter.purge_before(DAY)

        assert not (tmp_path / "exports").exists()

        path = exporter.export(DAY, now=at(6, 0))
        exporter.purge_before(DAY)

        assert sorted(item.name for item in path.parent.iterdir()) == ["traffic_2026-09-25.csv"]


def test_csv_survives_an_export_after_a_purge(tmp_path: Path) -> None:
    with prepared(tmp_path) as (store, exporter):
        exporter.purge_before(DAY)
        seed(store, ((1, at(6, 0, 1)),))

        path = exporter.export(DAY, now=at(6, 1))
        _, rows = read(path)

    assert len(rows) == 2
    assert rows[0]["count"] == "1"


def test_exporter_and_store_together_cover_the_acceptance_criterion(tmp_path: Path) -> None:
    with prepared(tmp_path) as (store, exporter):
        seed(store, tuple((index, at(6, index, 1)) for index in range(5)))
        store.write_health_buckets((bucket(at(6, 0), span(at(6, 0), 60)),))
        store.refresh_minute_rollups(DAY, at(6, 4))

        path = exporter.export(DAY, now=at(6, 4))
        _, rows = read(path)

        dashboard_total = store.summary_counts(DAY, PeriodName.PAGI)
        csv_total = sum(int(row["count"]) for row in rows)
        final_period_total = int(rows[-1]["period_total"])

    assert dashboard_total == 5
    assert csv_total == dashboard_total
    assert final_period_total == dashboard_total
    assert sum(1 for item in path.parent.iterdir()) == 1


def test_purge_before_csv_and_database_run_together(tmp_path: Path) -> None:
    with prepared(tmp_path) as (store, exporter):
        prior_app = store.start_app_session(at(6, 0, day=PRIOR_DAY))
        prior_tracking = store.start_tracking_session(prior_app, at(6, 0, day=PRIOR_DAY))
        assert store.insert_event(event(prior_tracking, 1, at(6, 5, day=PRIOR_DAY)))
        store.write_health_buckets(
            (bucket(at(6, 0, day=PRIOR_DAY), span(at(6, 0, day=PRIOR_DAY), 30)),)
        )
        store.refresh_minute_rollups(PRIOR_DAY, at(6, 5, day=PRIOR_DAY))
        seed(store, ((1, at(6, 0, 1)),))
        store.write_health_buckets((bucket(at(6, 0), span(at(6, 0), 60)),))
        store.refresh_minute_rollups(DAY, at(6, 2))
        exporter.export(PRIOR_DAY, now=at(6, 5, day=PRIOR_DAY))
        exporter.export(DAY, now=at(6, 2))

        store.purge_before(DAY)
        exporter.purge_before(DAY)

        remaining_files = sorted(item.name for item in (tmp_path / "exports").iterdir())
        _, rows = read(tmp_path / "exports" / "traffic_2026-09-25.csv")
        database_total = store.summary_counts(DAY, PeriodName.PAGI)

    assert remaining_files == ["traffic_2026-09-25.csv"]
    assert database_total == 1
    assert sum(int(row["count"]) for row in rows) == 1


def test_export_uses_a_sibling_temporary_file_of_the_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[tuple[str, str]] = []
    real_replace = os.replace

    def recording_replace(source: object, destination: object) -> None:
        observed.append((os.fspath(source), os.fspath(destination)))
        real_replace(source, destination)

    with prepared(tmp_path) as (_, exporter):
        monkeypatch.setattr(os, "replace", recording_replace)

        path = exporter.export(DAY, now=at(6, 1))

    assert observed == [(os.fspath(path) + ".tmp", os.fspath(path))]
    assert os.fspath(path).startswith(os.fspath(path.parent))


def test_export_is_atomic_while_another_reader_holds_the_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with prepared(tmp_path) as (store, exporter):
        path = exporter.export(DAY, now=at(6, 0))
        original = path.read_text(encoding="utf-8")
        seed(store, ((1, at(6, 0, 30)),))
        seen: dict[str, str] = {}
        real_replace = os.replace

        def replace_then_read(source: object, destination: object) -> None:
            seen["before"] = path.read_text(encoding="utf-8")
            real_replace(source, destination)
            seen["after"] = path.read_text(encoding="utf-8")

        monkeypatch.setattr(os, "replace", replace_then_read)

        exporter.export(DAY, now=at(6, 1))

    assert seen["before"] == original
    assert seen["after"] != original
    assert len(seen["after"].splitlines()) == 3


def test_a_failed_export_does_not_leave_a_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real_replace = os.replace

    def once_failing(source: object, destination: object) -> None:
        calls.append(1)
        if len(calls) == 1:
            raise OSError("simulated transient replace failure")
        real_replace(source, destination)

    with prepared(tmp_path) as (_, exporter):
        monkeypatch.setattr(os, "replace", once_failing)

        assert exporter.try_export(DAY, at(6, 1)) is False
        assert exporter.try_export(DAY, at(6, 1)) is True

        path = tmp_path / "exports" / "traffic_2026-09-25.csv"
        _, rows = read(path)
        listing = sorted(item.name for item in (tmp_path / "exports").iterdir())

    assert len(rows) == 2
    assert listing == ["traffic_2026-09-25.csv"]
    assert len(calls) == 2


def test_export_after_a_failed_first_attempt_is_complete(tmp_path: Path) -> None:
    with prepared(tmp_path) as (_, exporter):
        path = exporter.export(DAY, now=at(6, 1))

        _, rows = read(path)

    assert len(rows) == 2
    assert rows[0]["minute_start"] == "2026-09-25T06:00:00+07:00"
    assert rows[1]["minute_start"] == "2026-09-25T06:01:00+07:00"


def test_csv_export_of_a_store_with_no_sessions(tmp_path: Path) -> None:
    with prepared(tmp_path) as (_, exporter):
        path = exporter.export(DAY, now=at(6, 5))

        _, rows = read(path)

    assert len(rows) == 6
    assert all(row["count"] == "0" for row in rows)
    assert all(row["minute_rate"] == "" for row in rows)


def test_csv_export_uses_the_store_rollup_count_for_a_minute_with_no_events(
    tmp_path: Path,
) -> None:
    with prepared(tmp_path) as (store, exporter):
        seed(store, ((1, at(6, 0, 1)),))
        store.write_health_buckets((bucket(at(6, 0), span(at(6, 0), 60)),))
        store.refresh_minute_rollups(DAY, at(6, 0, 5))
        path = exporter.export(DAY, now=at(6, 0, 5))

        _, rows = read(path)

    assert len(rows) == 1
    assert rows[0]["count"] == "1"
    assert rows[0]["observed_seconds"] == "60"
    assert rows[0]["minute_rate"] == "1.0"
    assert rows[0]["period_rate"] == "1.0"


def test_export_path_helper_is_stable_for_a_repeated_day(tmp_path: Path) -> None:
    with prepared(tmp_path) as (_, exporter):
        first = exporter.try_export(DAY, at(6, 1))
        second = exporter.try_export(DAY, at(6, 2))

        path = tmp_path / "exports" / "traffic_2026-09-25.csv"
        _, rows = read(path)

    assert first is True
    assert second is True
    assert len(rows) == 3


def test_csv_exporter_accepts_a_string_directory(tmp_path: Path) -> None:
    store = TrafficStore(tmp_path / "traffic.db")
    store.initialize()
    exporter = CsvExporter(store, Path(str(tmp_path / "exports")))

    path = exporter.export(DAY, now=at(6, 0))

    assert path.parent == tmp_path / "exports"
    assert path.is_file()
    store.close()


def test_csv_period_label_comes_from_the_schedule_not_the_stored_column(
    tmp_path: Path,
) -> None:
    with prepared(tmp_path) as (store, exporter):
        seed(store, ((1, at(6, 0, 1)),))
        store.refresh_minute_rollups(DAY, at(6, 3))
        stored_minute = "2026-09-25T06:00:00+07:00"
        with closing(connect(tmp_path / "traffic.db")) as connection:
            connection.execute(
                "UPDATE minute_buckets SET period = 'sore' WHERE minute_start = ?",
                (stored_minute,),
            )
            connection.commit()

        path = exporter.export(DAY, now=at(6, 5))
        _, rows = read(path)
        stored = store.timeseries(DAY)
        stored_label = next(point.period for point in stored if point.minute_start == at(6, 0))

    assert stored_label is PeriodName.SORE
    assert [row["period"] for row in rows if row["minute_start"] == stored_minute] == ["pagi"]
    assert len(rows) == 6


def test_csv_period_labels_match_the_schedule_at_every_boundary(tmp_path: Path) -> None:
    with prepared(tmp_path) as (_, exporter):
        path = exporter.export(DAY, now=at(15, 1))

        _, rows = read(path)

    labelled = {row["minute_start"]: row["period"] for row in rows}
    assert labelled["2026-09-25T09:59:00+07:00"] == "pagi"
    assert labelled["2026-09-25T10:00:00+07:00"] == "siang"
    assert labelled["2026-09-25T14:59:00+07:00"] == "siang"
    assert labelled["2026-09-25T15:00:00+07:00"] == "sore"
