from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from traffic_counter.models import AppSnapshot, AppState, MinutePoint, PeriodName, StreamState
from traffic_counter.state import SnapshotStore
from traffic_counter.timeutils import JAKARTA, report_date

DAY_YEAR: Final[int] = 2026
DAY_MONTH: Final[int] = 9
DAY_NUMBER: Final[int] = 25
TEST_PORT: Final[int] = 5000
OTHER_PORT: Final[int] = 5077


def at(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(DAY_YEAR, DAY_MONTH, DAY_NUMBER, hour, minute, second, tzinfo=JAKARTA)


def summary_for(period: PeriodName, total: int, observed: int) -> Any:
    from traffic_counter.models import PeriodSummary

    return PeriodSummary(
        period=period,
        started=True,
        total=total,
        observed_seconds=observed,
        average_rate=None if observed <= 0 else total / observed * 60.0,
        rolling_rate=None,
    )


def blank_summary(period: PeriodName) -> Any:
    from traffic_counter.models import PeriodSummary

    return PeriodSummary(
        period=period,
        started=False,
        total=0,
        observed_seconds=0,
        average_rate=None,
        rolling_rate=None,
    )


def point(hour: int, minute: int, count: int, observed: int) -> MinutePoint:
    from traffic_counter.timeutils import period_for

    moment = at(hour, minute)
    resolved = period_for(moment)
    assert resolved is not None
    return MinutePoint(
        minute_start=moment,
        period=resolved,
        count=count,
        observed_seconds=observed,
    )


def store_with_snapshot(moment: datetime, snapshot: AppSnapshot) -> SnapshotStore:
    store = SnapshotStore(clock=lambda: moment)
    store.publish(snapshot)
    return store


def published_snapshot(moment: datetime) -> AppSnapshot:
    base = AppSnapshot.initial(moment)
    periods = MappingProxyType(
        {
            PeriodName.PAGI: summary_for(PeriodName.PAGI, 5, 120),
            PeriodName.SIANG: blank_summary(PeriodName.SIANG),
            PeriodName.SORE: blank_summary(PeriodName.SORE),
        }
    )
    minutes = (point(6, 0, 2, 60), point(6, 1, 3, 60))
    return replace(
        base,
        generated_at=moment,
        state=AppState.RUNNING,
        stream_state=StreamState.CONNECTED,
        actual_fps=9.5,
        target_fps=10,
        replaced_frames=3,
        periods=periods,
        minutes=minutes,
        csv_export_ok=True,
        dropped_events=0,
        incomplete=False,
        last_error=None,
        tracking_session_changes=1,
    )


def make_app(
    store: SnapshotStore,
    provider: Callable[[], Path],
    port: int = TEST_PORT,
    clock: Callable[[], datetime] | None = None,
    calibrated: bool | None = None,
) -> Any:
    from traffic_counter.web import create_app

    if clock is None and calibrated is None:
        return create_app(store, provider, port)
    if clock is not None and calibrated is not None:
        return create_app(store, provider, port, clock=clock, calibrated=calibrated)
    if clock is not None:
        return create_app(store, provider, port, clock=clock)
    return create_app(store, provider, port, calibrated=calibrated)


def missing_provider() -> Path:
    return Path("data") / "exports" / "traffic_2026-09-25.csv"


def test_trusted_hosts_are_accepted() -> None:
    moment = at(7, 0)
    store = store_with_snapshot(moment, published_snapshot(moment))
    app = make_app(store, missing_provider, TEST_PORT, clock=lambda: moment)
    client = app.test_client()
    for host in ("127.0.0.1:5000", "localhost:5000", "127.0.0.1", "localhost"):
        response = client.get("/api/status", headers={"Host": host})
        assert response.status_code == 200


def test_untrusted_host_is_rejected() -> None:
    moment = at(7, 0)
    store = store_with_snapshot(moment, published_snapshot(moment))
    app = make_app(store, missing_provider, TEST_PORT, clock=lambda: moment)
    client = app.test_client()
    response = client.get("/api/status", headers={"Host": "evil.example:5000"})
    assert response.status_code == 403
    payload = response.get_json()
    assert isinstance(payload, dict)


def test_untrusted_host_rejected_on_every_route() -> None:
    moment = at(7, 0)
    store = store_with_snapshot(moment, published_snapshot(moment))
    app = make_app(store, missing_provider, TEST_PORT, clock=lambda: moment)
    client = app.test_client()
    for route in ("/", "/api/status", "/api/summary", "/api/timeseries", "/api/export.csv"):
        response = client.get(route, headers={"Host": "attacker.test:5000"})
        assert response.status_code == 403


def test_summary_returns_all_periods_with_required_fields() -> None:
    moment = at(7, 0)
    store = store_with_snapshot(moment, published_snapshot(moment))
    app = make_app(store, missing_provider, TEST_PORT, clock=lambda: moment)
    client = app.test_client()
    response = client.get("/api/summary", headers={"Host": "127.0.0.1:5000"})
    assert response.status_code == 200
    payload = response.get_json()
    assert isinstance(payload, dict)
    periods = payload.get("periods", payload)
    assert isinstance(periods, dict)
    for name in ("pagi", "siang", "sore"):
        assert name in periods
        entry = periods[name]
        assert entry["started"] in (True, False)
        assert isinstance(entry["total"], int)
        assert isinstance(entry["observed_seconds"], int)
        assert "average_rate" in entry
        assert "rolling_rate" in entry
    assert periods["pagi"]["total"] == 5
    assert periods["pagi"]["started"] is True
    assert periods["siang"]["started"] is False
    assert periods["siang"]["total"] == 0
    assert periods["siang"]["average_rate"] is None


def test_timeseries_returns_date_timezone_and_ascending_minutes() -> None:
    moment = at(7, 0)
    store = store_with_snapshot(moment, published_snapshot(moment))
    app = make_app(store, missing_provider, TEST_PORT, clock=lambda: moment)
    client = app.test_client()
    response = client.get("/api/timeseries", headers={"Host": "127.0.0.1:5000"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["date"] == "2026-09-25"
    assert payload["timezone"] == "Asia/Jakarta"
    minutes = payload["minutes"]
    assert isinstance(minutes, list)
    assert len(minutes) == 2
    starts = [entry["minute_start"] for entry in minutes]
    assert starts == sorted(starts)
    assert starts[0] == "2026-09-25T06:00:00+07:00"
    for entry in minutes:
        assert isinstance(entry["count"], int)
        assert isinstance(entry["observed_seconds"], int)
        assert "minute_start" in entry
    assert minutes[0]["count"] == 2
    assert minutes[1]["count"] == 3


def test_export_before_generation_returns_json_404() -> None:
    moment = at(7, 0)
    store = store_with_snapshot(moment, published_snapshot(moment))
    app = make_app(store, missing_provider, TEST_PORT, clock=lambda: moment)
    client = app.test_client()
    response = client.get("/api/export.csv", headers={"Host": "127.0.0.1:5000"})
    assert response.status_code == 404
    assert response.content_type.startswith("application/json")
    payload = response.get_json()
    assert payload["error"] == "csv_not_generated"


def test_export_after_generation_streams_attachment(tmp_path: Path) -> None:
    moment = at(7, 0)
    store = store_with_snapshot(moment, published_snapshot(moment))
    target = tmp_path / "traffic_2026-09-25.csv"
    target.write_text("date,period\n2026-09-25,pagi\n", encoding="utf-8")
    app = make_app(store, lambda: target, TEST_PORT, clock=lambda: moment)
    client = app.test_client()
    response = client.get("/api/export.csv", headers={"Host": "127.0.0.1:5000"})
    assert response.status_code == 200
    disposition = response.headers.get("Content-Disposition", "")
    assert "attachment" in disposition
    assert "traffic_2026-09-25.csv" in disposition
    assert "2026-09-25" in response.get_data(as_text=True)


def test_export_missing_file_after_provider_error_returns_404() -> None:
    moment = at(7, 0)
    store = store_with_snapshot(moment, published_snapshot(moment))

    def failing() -> Path:
        raise OSError("disk down")

    app = make_app(store, failing, TEST_PORT, clock=lambda: moment)
    client = app.test_client()
    response = client.get("/api/export.csv", headers={"Host": "127.0.0.1:5000"})
    assert response.status_code == 404
    assert response.get_json()["error"] == "csv_not_generated"


def test_snapshot_serialization_does_not_mutate_periods_or_minutes() -> None:
    moment = at(7, 0)
    snapshot = published_snapshot(moment)
    store = store_with_snapshot(moment, snapshot)
    before_periods = dict(store.get().periods)
    before_minutes = store.get().minutes
    app = make_app(store, missing_provider, TEST_PORT, clock=lambda: moment)
    client = app.test_client()
    for route in ("/api/status", "/api/summary", "/api/timeseries"):
        assert client.get(route, headers={"Host": "127.0.0.1:5000"}).status_code == 200
    after = store.get()
    assert after.minutes == before_minutes
    assert dict(after.periods) == before_periods
    assert isinstance(after.periods, MappingProxyType)
    assert isinstance(after.minutes, tuple)
    assert after.periods[PeriodName.PAGI].total == 5


def test_web_never_touches_sqlite() -> None:
    from traffic_counter import web as web_module

    text = Path(str(web_module.__file__)).read_text(encoding="utf-8")
    lowered = text.lower()
    assert "sqlite" not in lowered
    assert "trafficstore" not in lowered
    assert "deepcopy" not in lowered
    assert "pickle" not in lowered


def test_pre_publication_reports_not_published_with_age() -> None:
    start = at(6, 0)
    store = SnapshotStore(clock=lambda: start)
    initial_at = store.get().generated_at
    assert initial_at == start
    current = at(6, 0, 30)
    app = make_app(store, missing_provider, TEST_PORT, clock=lambda: current)
    client = app.test_client()
    response = client.get("/api/status", headers={"Host": "127.0.0.1:5000"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["published"] is False
    assert payload["live"] is False
    assert payload["snapshot_age_seconds"] == 30.0
    assert payload["generated_at"] == "2026-09-25T06:00:00+07:00"


def test_published_snapshot_reports_live_with_small_age() -> None:
    start = at(6, 0)
    store = SnapshotStore(clock=lambda: start)
    app = make_app(store, missing_provider, TEST_PORT, clock=lambda: start)
    published = published_snapshot(start + timedelta(seconds=5))
    store.publish(published)
    client = app.test_client()
    response = client.get("/api/status", headers={"Host": "127.0.0.1:5000"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["published"] is True
    assert payload["live"] is True
    assert payload["snapshot_age_seconds"] >= 0.0


def test_status_shape_covers_lifecycle_and_health() -> None:
    moment = at(7, 0)
    store = store_with_snapshot(moment, published_snapshot(moment))
    app = make_app(store, missing_provider, TEST_PORT, clock=lambda: moment, calibrated=True)
    client = app.test_client()
    response = client.get("/api/status", headers={"Host": "127.0.0.1:5000"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["state"] == "running"
    assert payload["stream_state"] == "connected"
    assert payload["target_fps"] == 10
    assert payload["actual_fps"] == 9.5
    assert payload["replaced_frames"] == 3
    assert isinstance(payload["snapshot_age_seconds"], float)
    assert payload["csv_export_ok"] is True
    assert payload["dropped_events"] == 0
    assert payload["incomplete"] is False
    assert payload["tracking_session_changes"] == 1
    assert payload["calibrated"] is True
    assert payload["last_error"] is None
    assert payload["generated_at"] == "2026-09-25T07:00:00+07:00"


def test_status_sanitizes_error_without_paths() -> None:
    moment = at(7, 0)
    base = published_snapshot(moment)
    nasty = "disk down at C:/Users/TRISTAN/data/traffic.db with token abc " + "x" * 300
    poisoned = replace(base, last_error=nasty, state=AppState.ERROR, incomplete=True)
    store = store_with_snapshot(moment, poisoned)
    app = make_app(store, missing_provider, TEST_PORT, clock=lambda: moment)
    client = app.test_client()
    response = client.get("/api/status", headers={"Host": "127.0.0.1:5000"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["state"] == "error"
    assert payload["incomplete"] is True
    cleaned = payload["last_error"]
    assert isinstance(cleaned, str)
    assert len(cleaned) <= 200
    assert "C:/Users" not in cleaned
    assert "traffic.db" not in cleaned


def test_status_omits_calibrated_when_unavailable() -> None:
    moment = at(7, 0)
    store = store_with_snapshot(moment, published_snapshot(moment))
    app = make_app(store, missing_provider, TEST_PORT, clock=lambda: moment)
    client = app.test_client()
    payload = client.get("/api/status", headers={"Host": "127.0.0.1:5000"}).get_json()
    assert "calibrated" not in payload


def test_routes_are_get_only() -> None:
    moment = at(7, 0)
    store = store_with_snapshot(moment, published_snapshot(moment))
    app = make_app(store, missing_provider, TEST_PORT, clock=lambda: moment)
    client = app.test_client()
    for route in ("/api/status", "/api/summary", "/api/timeseries", "/api/export.csv"):
        for method in ("post", "put", "delete", "patch"):
            caller = getattr(client, method)
            response = caller(route, headers={"Host": "127.0.0.1:5000"})
            assert response.status_code == 405


def test_responses_carry_no_cors_headers() -> None:
    moment = at(7, 0)
    store = store_with_snapshot(moment, published_snapshot(moment))
    app = make_app(store, missing_provider, TEST_PORT, clock=lambda: moment)
    client = app.test_client()
    for route in ("/", "/api/status", "/api/summary", "/api/timeseries"):
        response = client.get(route, headers={"Host": "127.0.0.1:5000"})
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" not in response.headers


def test_dashboard_page_renders_without_external_calls() -> None:
    moment = at(7, 0)
    store = store_with_snapshot(moment, published_snapshot(moment))
    app = make_app(store, missing_provider, TEST_PORT, clock=lambda: moment)
    client = app.test_client()
    response = client.get("/", headers={"Host": "127.0.0.1:5000"})
    assert response.status_code == 200
    assert "text/html" in response.content_type
    text = response.get_data(as_text=True)
    assert "Traffic Counter" in text
    assert "/api/export.csv" in text
    assert "/api/status" in text
    lowered = text.lower()
    assert "http" not in lowered


def test_report_day_matches_snapshot_day() -> None:
    moment = at(7, 0)
    store = store_with_snapshot(moment, published_snapshot(moment))
    app = make_app(store, missing_provider, TEST_PORT, clock=lambda: moment)
    client = app.test_client()
    summary_day = report_date(moment).isoformat()
    timeseries = client.get("/api/timeseries", headers={"Host": "127.0.0.1:5000"}).get_json()
    assert timeseries["date"] == summary_day


def test_trusted_hosts_constant_and_port_binding() -> None:
    from traffic_counter import web as web_module

    assert tuple(web_module.TRUSTED_HOSTS) == ("127.0.0.1", "localhost", "::1")
    text = Path(str(web_module.__file__)).read_text(encoding="utf-8")
    assert "127.0.0.1" in text
    assert "waitress" in text.lower()


def test_serve_dashboard_signature_matches_controller_seam() -> None:
    import inspect

    from traffic_counter.web import serve_dashboard

    params = list(inspect.signature(serve_dashboard).parameters.keys())
    assert params == ["snapshot_store", "csv_path_provider", "config"]


def test_create_app_signature_matches_brief() -> None:
    import inspect

    from traffic_counter.web import create_app

    params = list(inspect.signature(create_app).parameters.keys())
    assert params[:3] == ["snapshot_store", "csv_path_provider", "port"]


def status_with_error(moment: datetime, message: str) -> Any:
    base = published_snapshot(moment)
    return replace(base, last_error=message, state=AppState.ERROR)


def test_sanitizer_redacts_spaced_windows_path() -> None:
    moment = at(7, 0)
    nasty = "failed at C:/Users/TRISTAN/My Data/traffic.db while running"
    store = store_with_snapshot(moment, status_with_error(moment, nasty))
    app = make_app(store, missing_provider, TEST_PORT, clock=lambda: moment)
    client = app.test_client()
    payload = client.get("/api/status", headers={"Host": "127.0.0.1:5000"}).get_json()
    cleaned = payload["last_error"]
    assert "C:/" not in cleaned
    assert "My Data" not in cleaned
    assert "traffic.db" not in cleaned
    assert "<path>" in cleaned


def test_sanitizer_redacts_bare_filename() -> None:
    moment = at(7, 0)
    nasty = "cannot open traffic.db for writing"
    store = store_with_snapshot(moment, status_with_error(moment, nasty))
    app = make_app(store, missing_provider, TEST_PORT, clock=lambda: moment)
    client = app.test_client()
    payload = client.get("/api/status", headers={"Host": "127.0.0.1:5000"}).get_json()
    cleaned = payload["last_error"]
    assert "traffic.db" not in cleaned
    assert "<path>" in cleaned


def test_sanitizer_preserves_api_route_text() -> None:
    moment = at(7, 0)
    nasty = "GET /api/status failed with timeout"
    store = store_with_snapshot(moment, status_with_error(moment, nasty))
    app = make_app(store, missing_provider, TEST_PORT, clock=lambda: moment)
    client = app.test_client()
    payload = client.get("/api/status", headers={"Host": "127.0.0.1:5000"}).get_json()
    assert "/api/status" in payload["last_error"]


def test_sanitizer_truncates_after_redaction() -> None:
    moment = at(7, 0)
    nasty = "error at C:/Users/My Data/traffic.db " + "y" * 500
    store = store_with_snapshot(moment, status_with_error(moment, nasty))
    app = make_app(store, missing_provider, TEST_PORT, clock=lambda: moment)
    client = app.test_client()
    payload = client.get("/api/status", headers={"Host": "127.0.0.1:5000"}).get_json()
    cleaned = payload["last_error"]
    assert len(cleaned) <= 200
    assert "C:/" not in cleaned
    assert "traffic.db" not in cleaned
    assert "<path>" in cleaned


def test_first_publish_with_identical_timestamp_reports_published() -> None:
    start = at(6, 0)
    store = SnapshotStore(clock=lambda: start)
    app = make_app(store, missing_provider, TEST_PORT, clock=lambda: start)
    store.publish(published_snapshot(start))
    client = app.test_client()
    payload = client.get("/api/status", headers={"Host": "127.0.0.1:5000"}).get_json()
    assert payload["published"] is True
    assert payload["live"] is True


def test_stale_published_snapshot_reports_not_live() -> None:
    start = at(6, 0)
    late = at(7, 0)
    store = SnapshotStore(clock=lambda: start)
    app = make_app(store, missing_provider, TEST_PORT, clock=lambda: late)
    store.publish(published_snapshot(start))
    client = app.test_client()
    payload = client.get("/api/status", headers={"Host": "127.0.0.1:5000"}).get_json()
    assert payload["published"] is True
    assert payload["live"] is False
    assert payload["snapshot_age_seconds"] == 3600.0


def test_unstarted_store_reports_not_published() -> None:
    start = at(6, 0)
    store = SnapshotStore(clock=lambda: start)
    app = make_app(store, missing_provider, TEST_PORT, clock=lambda: start)
    client = app.test_client()
    payload = client.get("/api/status", headers={"Host": "127.0.0.1:5000"}).get_json()
    assert payload["published"] is False
    assert payload["live"] is False


def test_dashboard_svg_contains_bars_and_table_rows() -> None:
    moment = at(7, 0)
    store = store_with_snapshot(moment, published_snapshot(moment))
    app = make_app(store, missing_provider, TEST_PORT, clock=lambda: moment)
    client = app.test_client()
    text = client.get("/", headers={"Host": "127.0.0.1:5000"}).get_data(as_text=True)
    assert "<svg" in text
    assert "<rect" in text
    assert "<polyline" in text
    assert "2026-09-25T06:00:00+07:00" in text
    assert "2026-09-25T06:01:00+07:00" in text
    assert text.count("<rect") == 2


def test_dashboard_svg_empty_minutes_still_renders() -> None:
    start = at(6, 0)
    store = SnapshotStore(clock=lambda: start)
    app = make_app(store, missing_provider, TEST_PORT, clock=lambda: start)
    client = app.test_client()
    text = client.get("/", headers={"Host": "127.0.0.1:5000"}).get_data(as_text=True)
    assert "<svg" in text
    assert "<line" in text
    assert "no data" in text


def test_host_matrix_accepted_forms() -> None:
    moment = at(7, 0)
    accepted = (
        "127.0.0.1",
        "localhost",
        "[::1]",
        "127.0.0.1:5000",
        "localhost:5000",
        "[::1]:5000",
        "LOCALHOST:5000",
    )
    for host in accepted:
        store = store_with_snapshot(moment, published_snapshot(moment))
        app = make_app(store, missing_provider, TEST_PORT, clock=lambda: moment)
        client = app.test_client()
        response = client.get("/api/status", headers={"Host": host})
        assert response.status_code == 200


def test_host_matrix_rejected_forms_return_json_403() -> None:
    moment = at(7, 0)
    rejected = (
        "127.0.0.1:9999",
        "localhost:9999",
        "[::1]:9999",
        "192.168.1.10:5000",
        "0.0.0.0:5000",
        "example.com:5000",
        "evil-localhost:5000",
        "localhost.evil.com:5000",
        "127.0.0.1.evil:5000",
        "a127.0.0.1:5000",
        "127.0.0.1a:5000",
        "my-localhost:5000",
    )
    for host in rejected:
        store = store_with_snapshot(moment, published_snapshot(moment))
        app = make_app(store, missing_provider, TEST_PORT, clock=lambda: moment)
        client = app.test_client()
        for route in ("/", "/api/status", "/api/summary"):
            response = client.get(route, headers={"Host": host})
            assert response.status_code == 403
            assert response.content_type.startswith("application/json")
            assert response.get_json()["error"] == "untrusted_host"


def test_summary_aliases_are_independent_copies() -> None:
    from traffic_counter.web import _summary_payload

    moment = at(7, 0)
    payload = _summary_payload(published_snapshot(moment))
    payload["pagi"]["total"] = 999
    assert payload["periods"]["pagi"]["total"] == 5
    payload["periods"]["siang"]["total"] = 888
    assert payload["siang"]["total"] == 0


class FailingStore:
    def get(self) -> AppSnapshot:
        raise RuntimeError("store down")


def test_snapshot_failure_returns_json_500() -> None:
    from traffic_counter.web import create_app

    store = FailingStore()
    app = create_app(store, missing_provider, TEST_PORT)
    client = app.test_client()
    for route in ("/", "/api/status", "/api/summary", "/api/timeseries"):
        response = client.get(route, headers={"Host": "127.0.0.1:5000"})
        assert response.status_code == 500
        assert response.content_type.startswith("application/json")
        payload = response.get_json()
        assert "error" in payload
        assert "traceback" not in response.get_data(as_text=True).lower()
