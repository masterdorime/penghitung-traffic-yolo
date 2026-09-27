from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final
from zoneinfo import ZoneInfo

from traffic_counter.models import AppSnapshot, AppState, MinutePoint, PeriodName, StreamState
from traffic_counter.state import SnapshotStore

JAKARTA: Final[ZoneInfo] = ZoneInfo("Asia/Jakarta")
TEST_PORT: Final[int] = 5000
ROOT: Final[Path] = Path(__file__).resolve().parent.parent
PKG: Final[Path] = ROOT / "src" / "traffic_counter"
TEMPLATE_PATH: Final[Path] = PKG / "templates" / "dashboard.html"
CSS_PATH: Final[Path] = PKG / "static" / "dashboard.css"
JS_PATH: Final[Path] = PKG / "static" / "dashboard.js"
WEB_PATH: Final[Path] = PKG / "web.py"

REQUIRED_IDS: Final[tuple[str, ...]] = (
    "app-state",
    "stream-state",
    "actual-fps",
    "snapshot-age",
    "period-cards",
    "traffic-chart",
    "timeseries-table",
    "export-link",
    "data-warning",
    "last-updated",
)


def at(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, 9, 25, hour, minute, second, tzinfo=JAKARTA)


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


def filled_summary(period: PeriodName, total: int, observed: int) -> Any:
    from traffic_counter.models import PeriodSummary

    return PeriodSummary(
        period=period,
        started=True,
        total=total,
        observed_seconds=observed,
        average_rate=None if observed <= 0 else total / observed * 60.0,
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


def published_snapshot(moment: datetime) -> AppSnapshot:
    base = AppSnapshot.initial(moment)
    periods = MappingProxyType(
        {
            PeriodName.PAGI: filled_summary(PeriodName.PAGI, 5, 120),
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


def store_with(moment: datetime, snapshot: AppSnapshot) -> SnapshotStore:
    store = SnapshotStore(clock=lambda: moment)
    store.publish(snapshot)
    return store


def missing_provider() -> Path:
    return Path("data") / "exports" / "traffic_2026-09-25.csv"


def make_app(
    store: SnapshotStore,
    provider: Callable[[], Path],
    clock: Callable[[], datetime] | None = None,
) -> Any:
    from traffic_counter.web import create_app

    if clock is None:
        return create_app(store, provider, TEST_PORT)
    return create_app(store, provider, TEST_PORT, clock=clock)


def read_template() -> str:
    return TEMPLATE_PATH.read_text(encoding="utf-8")


def read_css() -> str:
    return CSS_PATH.read_text(encoding="utf-8")


def read_js() -> str:
    return JS_PATH.read_text(encoding="utf-8")


def read_web() -> str:
    return WEB_PATH.read_text(encoding="utf-8")


def test_template_file_exists() -> None:
    assert TEMPLATE_PATH.is_file()


def test_static_files_exist() -> None:
    assert CSS_PATH.is_file()
    assert JS_PATH.is_file()


def test_required_ids_present_in_template() -> None:
    html = read_template()
    for name in REQUIRED_IDS:
        assert str(name) in html
        assert str("id=" + chr(34) + name + chr(34)) in html


def test_semantic_structure() -> None:
    html = read_template()
    lowered = html.lower()
    assert "<header" in lowered
    assert "<main" in lowered
    assert "<section" in lowered
    assert "<table" in lowered
    assert "<thead" in lowered
    assert "<tbody" in lowered
    assert "<footer" in lowered or "<section" in lowered
    assert str("lang=" + chr(34)) in lowered
    assert "<title" in lowered
    assert "traffic counter" in lowered
    assert "viewport" in lowered
    assert "export-link" in html
    assert "/api/export.csv" in html


def test_accessibility_fallbacks() -> None:
    html = read_template()
    css = read_css()
    lowered_html = html.lower()
    lowered_css = css.lower()
    assert "skip" in lowered_html
    assert str("id=" + chr(34) + "main" + chr(34)) in html
    assert "aria-live" in html
    assert "role=" in html
    assert "aria-label" in html or "aria-labelledby" in html
    assert "<th" in lowered_html
    assert "caption" in lowered_html or "<th" in lowered_html
    assert "focus-visible" in lowered_css
    assert "prefers-reduced-motion" in lowered_css
    assert "prefers-color-scheme" in lowered_css


def test_status_not_color_only() -> None:
    html = read_template()
    css = read_css()
    js = read_js()
    combined = (html + css + js).lower()
    assert "data-state" in combined
    has_shape = False
    for token in ("circle", "square", "triangle", "shape", "icon", "symbol"):
        if token in combined:
            has_shape = True
    has_glyph = False
    for glyph in ("●", "▲", "■", "!", "OK", "LIVE", "STALE"):
        if glyph in (html + css + js):
            has_glyph = True
    assert has_shape or has_glyph


def test_visual_system_tokens() -> None:
    css = read_css()
    lowered = css.lower()
    assert "bahnschrift" in lowered
    assert "segoe ui" in lowered
    assert "consolas" in lowered
    assert "6px" in css
    assert "120ms" in css
    assert "box-shadow" in lowered
    assert "transition" in lowered
    assert "900px" in css
    assert "640px" in css
    assert "10,18,32" in css or "10, 18, 32" in css or "0a1220" in lowered
    assert "255,180,60" in css or "255, 180, 60" in css or "ffb43c" in lowered
    assert "46,204,113" in css or "46, 204, 113" in css or "2ecc71" in lowered
    assert "231,76,60" in css or "231, 76, 60" in css or "e74c3c" in lowered


def test_no_external_references() -> None:
    html = read_template()
    css = read_css()
    js = read_js()
    for text in (html, css, js):
        assert "https://" not in text
        assert "cdn" not in text.lower()
    assert "@import" not in css.lower() or "https://" not in css
    assert "http-equiv" not in html.lower()


def test_no_comment_markers_in_assets() -> None:
    html = read_template()
    css = read_css()
    js = read_js()
    assert "<!--" not in html
    assert "/*" not in css
    assert "*/" not in css
    assert "/*" not in js
    assert "*/" not in js
    for line in js.splitlines():
        assert not line.strip().startswith("//")


def test_js_polling_contract() -> None:
    js = read_js()
    assert "/api/status" in js
    assert "/api/summary" in js
    assert "/api/timeseries" in js
    assert "setTimeout" in js
    assert "1000" in js
    assert "setInterval" not in js
    assert "fetch" in js
    assert "window.trafficDashboard" in js
    assert "init" in js


def test_js_error_and_stale_contract() -> None:
    js = read_js()
    lowered = js.lower()
    assert "stale" in lowered
    assert "lastGood" in js or "last-good" in js or "last_good" in js
    assert "try" in js and "catch" in js
    assert chr(8212) in js


def test_js_number_and_chart_contract() -> None:
    js = read_js()
    assert "Intl.NumberFormat" in js
    assert "id-ID" in js
    assert "<svg" in js
    assert "rect" in js.lower()
    assert "tbody" in js.lower() or "timeseries-table" in js
    assert js.count("window.") == 1


def test_web_uses_template_rendering() -> None:
    text = read_web()
    assert "render_template" in text
    assert "dashboard.html" in text
    assert "templates" in text.lower() or "render_template" in text
    assert "sqlite" not in text.lower()
    assert "deepcopy" not in text.lower()
    assert "pickle" not in text.lower()


def test_flask_index_serves_template() -> None:
    moment = at(7, 0)
    store = store_with(moment, published_snapshot(moment))
    app = make_app(store, missing_provider, clock=lambda: moment)
    client = app.test_client()
    response = client.get("/", headers={"Host": "127.0.0.1:5000"})
    assert response.status_code == 200
    assert "text/html" in response.content_type
    text = response.get_data(as_text=True)
    for name in REQUIRED_IDS:
        assert str("id=" + chr(34) + name + chr(34)) in text
    assert "/static/dashboard.css" in text or "dashboard.css" in text
    assert "/static/dashboard.js" in text or "dashboard.js" in text
    assert "/api/export.csv" in text
    assert "https://" not in text
    assert "cdn" not in text.lower()
    assert "http" not in text.lower()


def test_flask_static_assets_served() -> None:
    moment = at(7, 0)
    store = store_with(moment, published_snapshot(moment))
    app = make_app(store, missing_provider, clock=lambda: moment)
    client = app.test_client()
    css = client.get("/static/dashboard.css", headers={"Host": "127.0.0.1:5000"})
    assert css.status_code == 200
    assert "text/css" in css.content_type
    js = client.get("/static/dashboard.js", headers={"Host": "127.0.0.1:5000"})
    assert js.status_code == 200
    assert "javascript" in js.content_type


def tag_for(html: str, marker: str) -> str:
    start = html.index(marker)
    open_start = html.rindex("<", 0, start)
    close = html.index(">", start)
    return html[open_start : close + 1]


def test_status_shape_mechanism_effective() -> None:
    html = read_template()
    css = read_css()
    js = read_js()
    assert "content:" not in css
    assert 'class="shape"' in html
    assert 'class="label"' in html
    assert ">?</span>" in html
    assert "●" in js
    seg = js.split("function shapeForState")[1].split("function paintState")[0]
    assert 'return "●";' in seg
    assert 'return "▲";' in seg
    assert 'return "■";' in seg
    assert 'return "?";' in seg
    assert ".shape" in js
    assert ".label" in js


def test_poll_guard_schedules_single_chain() -> None:
    js = read_js()
    assert js.count("setTimeout") == 1
    guard = js.split("if (pending)")[1].split("}")[0]
    assert "return;" in guard
    assert "setTimeout" not in guard
    assert "pollOnce();" in js


def test_live_regions_scoped() -> None:
    html = read_template()
    quote = chr(34)
    app_tag = tag_for(html, "id=" + quote + "app-state" + quote)
    stream_tag = tag_for(html, "id=" + quote + "stream-state" + quote)
    warn_tag = tag_for(html, "id=" + quote + "data-warning" + quote)
    cards_tag = tag_for(html, "id=" + quote + "period-cards" + quote)
    assert "aria-live" in app_tag
    assert "aria-live" in stream_tag
    assert "aria-live" in warn_tag
    assert "aria-live" not in cards_tag


def test_no_dead_listeners() -> None:
    js = read_js()
    assert "DOMContentLoaded" not in js
    assert "addEventListener" not in js
    assert "removeEventListener" not in js
    assert "pollOnce();" in js


def test_literal_dash_without_units() -> None:
    js = read_js()
    assert 'formatInt(status.snapshot_age_seconds) + " dtk"' not in js
    assert 'formatRate(entry.average_rate) + " per mnt"' not in js
    assert '"dtk"' in js
    assert '"per mnt"' in js
    assert chr(8212) in js
    assert "formatUnit" in js


def test_escape_both_quotes() -> None:
    js = read_js()
    assert "&quot;" in js
    assert ("&" + chr(35) + "39;") in js


def test_no_dead_css() -> None:
    css = read_css()
    assert "symbol-legend" not in css
    assert "icon-shape" not in css


def test_light_scheme_contrast() -> None:
    css = read_css()
    light = css.split("prefers-color-scheme: light")[1].split("@media")[0]
    assert "146, 64, 14" in light
    assert "21, 128, 61" in light
    assert "185, 28, 28" in light
    assert "warning" in light
    assert "a:hover" in light


def test_no_inline_comment_markers() -> None:
    html = read_template()
    js = read_js()
    assert "//" not in js
    assert "-->" not in html


def test_last_good_restoration() -> None:
    js = read_js()
    assert "lastGood.status" in js
    assert "lastGood.summary" in js
    assert "lastGood.series" in js
    assert "markStale" in js
