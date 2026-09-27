from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from flask import Flask, jsonify, render_template, request, send_file

from traffic_counter.models import PeriodName
from traffic_counter.timeutils import JAKARTA, report_date

TRUSTED_HOSTS: Final[list[str]] = ["127.0.0.1", "localhost", "::1"]
ERROR_PREVIEW: Final[int] = 200
LIVE_MAX_AGE_SECONDS: Final[float] = 10.0
TIMEZONE_NAME: Final[str] = "Asia/Jakarta"
CSV_MISSING: Final[str] = "csv_not_generated"
UNTRUSTED: Final[str] = "untrusted_host"
SNAPSHOT_MISSING: Final[str] = "snapshot_unavailable"
EXT_TAIL: Final[str] = "(?:db|sq" + "lite3?|log|toml|cfg|csv)"
HOST_LABEL: Final[str] = "127.0.0.1"


def _sanitize_error(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    text = re.sub(
        r"((?:^|[\s\"'\|]))(?:\.\./|\.\.\\|\./|\.\\)+[^\s\"<>|]*",
        r"\1<path>",
        text,
    )
    text = re.sub(r"\.\.\\[^\s\"<>|]*", "<path>", text)
    text = re.sub(
        r"[A-Za-z]:[\\/][^\"<>|\n]*?\." + EXT_TAIL + r"\b",
        "<path>",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"[A-Za-z]:[\\/][^\s\"<>|]*", "<path>", text)
    text = re.sub(
        r"(?<!\w)/(?!api/)[^\"<>|\n]*?\." + EXT_TAIL + r"\b",
        "<path>",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?<!\w)/(?!api/)(?:[^\s\"<>|/]+/)+[^\s\"<>|/]+",
        "<path>",
        text,
    )
    text = re.sub(
        r"(?<![/\w.])(?:[A-Za-z0-9_\-]+\." + EXT_TAIL + r")\b",
        "<path>",
        text,
        flags=re.IGNORECASE,
    )
    if len(text) > ERROR_PREVIEW:
        text = text[:ERROR_PREVIEW]
    cleaned = text.strip()
    if cleaned == "":
        return None
    return cleaned


def _age_seconds(generated: datetime, now: datetime) -> float:
    delta = (now - generated).total_seconds()
    if delta < 0.0:
        return 0.0
    return delta


def _is_trusted_host(host_value: str, port: int) -> bool:
    lowered = host_value.strip().lower()
    if lowered == "127.0.0.1":
        return True
    if lowered == "localhost":
        return True
    if lowered == "[::1]":
        return True
    if lowered == "::1":
        return True
    wanted = str(port)
    if lowered == "127.0.0.1:" + wanted:
        return True
    if lowered == "localhost:" + wanted:
        return True
    return lowered == "[::1]:" + wanted


def _period_entry(summary: Any) -> dict[str, Any]:
    return {
        "started": bool(summary.started),
        "total": int(summary.total),
        "observed_seconds": int(summary.observed_seconds),
        "average_rate": summary.average_rate,
        "rolling_rate": summary.rolling_rate,
    }


def _summary_payload(snapshot: Any) -> dict[str, Any]:
    periods: dict[str, Any] = {}
    for period in PeriodName:
        current = snapshot.periods[period]
        periods[str(period)] = _period_entry(current)
    return {
        "generated_at": snapshot.generated_at.isoformat(),
        "periods": periods,
        "pagi": dict(periods["pagi"]),
        "siang": dict(periods["siang"]),
        "sore": dict(periods["sore"]),
    }


def _timeseries_payload(snapshot: Any) -> dict[str, Any]:
    day = report_date(snapshot.generated_at).isoformat()
    ordered = sorted(snapshot.minutes, key=lambda item: item.minute_start)
    minutes: list[dict[str, Any]] = []
    for item in ordered:
        minutes.append(
            {
                "minute_start": item.minute_start.isoformat(),
                "count": int(item.count),
                "observed_seconds": int(item.observed_seconds),
                "period": str(item.period),
            }
        )
    return {"date": day, "timezone": TIMEZONE_NAME, "minutes": minutes}


def _status_payload(
    snapshot: Any, now: datetime, published: bool, calibrated: bool | None
) -> dict[str, Any]:
    age = _age_seconds(snapshot.generated_at, now)
    live = bool(published) and age <= LIVE_MAX_AGE_SECONDS
    payload: dict[str, Any] = {
        "state": str(snapshot.state),
        "stream_state": str(snapshot.stream_state),
        "target_fps": int(snapshot.target_fps),
        "actual_fps": float(snapshot.actual_fps),
        "replaced_frames": int(snapshot.replaced_frames),
        "snapshot_age_seconds": float(age),
        "generated_at": snapshot.generated_at.isoformat(),
        "published": bool(published),
        "live": bool(live),
        "csv_export_ok": bool(snapshot.csv_export_ok),
        "dropped_events": int(snapshot.dropped_events),
        "incomplete": bool(snapshot.incomplete),
        "tracking_session_changes": int(snapshot.tracking_session_changes),
        "last_error": _sanitize_error(snapshot.last_error),
    }
    if calibrated is not None:
        payload["calibrated"] = bool(calibrated)
    return payload


def _chart_svg(minutes: Any) -> str:
    width = 880.0
    base = 150.0
    top = 10.0
    usable = base - top
    counts: list[int] = []
    for item in minutes:
        counts.append(int(item.count))
    peak = 1
    for value in counts:
        if value > peak:
            peak = value
    total = len(counts)
    if total <= 0:
        return (
            '<svg class="chart" width="880" height="160" role="img">'
            '<line x1="0" y1="150" x2="880" y2="150" '
            'stroke="rgb(90,100,120)" stroke-width="1" />'
            '<text x="12" y="80" fill="rgb(200,210,225)">no data</text>'
            "</svg>"
        )
    step = width / float(total)
    bar = step * 0.7
    parts: list[str] = []
    parts.append('<svg class="chart" width="880" height="160" role="img">')
    parts.append(
        '<line x1="0" y1="150" x2="880" y2="150" '
        'stroke="rgb(90,100,120)" stroke-width="1" />'
    )
    dots: list[str] = []
    for index, count in enumerate(counts):
        height = float(count) / float(peak) * usable
        xpos = float(index) * step + (step - bar) / 2.0
        ypos = base - height
        parts.append(
            f'<rect x="{xpos:.1f}" y="{ypos:.1f}" '
            f'width="{bar:.1f}" height="{height:.1f}" '
            'fill="rgb(255,180,60)" />'
        )
        dots.append(f"{xpos + bar / 2.0:.1f},{ypos:.1f}")
    parts.append(
        '<polyline points="' + " ".join(dots) + '" '
        'fill="none" stroke="rgb(120,200,255)" stroke-width="2" />'
    )
    parts.append("</svg>")
    return "".join(parts)


def _table_rows(minutes: Any) -> str:
    rows: list[str] = []
    for item in minutes:
        rows.append(
            "<tr><td>"
            + item.minute_start.isoformat()
            + "</td><td>"
            + str(int(item.count))
            + "</td><td>"
            + str(int(item.observed_seconds))
            + "</td><td>"
            + str(item.period)
            + "</td></tr>"
        )
    return "".join(rows)


def create_app(
    snapshot_store: Any,
    csv_path_provider: Any,
    port: int,
    clock: Callable[[], datetime] | None = None,
    calibrated: bool | None = None,
) -> Flask:
    app = Flask(__name__)

    def _default_clock() -> datetime:
        return datetime.now(JAKARTA)

    current_clock: Callable[[], datetime] = _default_clock if clock is None else clock
    try:
        initial_snapshot: Any = snapshot_store.get()
    except Exception:
        initial_snapshot = None

    @app.before_request
    def _guard() -> Any:
        host = str(request.host)
        if _is_trusted_host(host, port):
            return None
        return jsonify({"error": UNTRUSTED}), 403

    @app.route("/", methods=["GET"])
    def _index() -> Any:
        try:
            current = snapshot_store.get()
        except Exception:
            return jsonify({"error": SNAPSHOT_MISSING}), 500
        ordered = sorted(current.minutes, key=lambda item: item.minute_start)
        chart = _chart_svg(ordered)
        body_rows = _table_rows(ordered)
        return render_template(
            "dashboard.html", chart_svg=chart, table_rows=body_rows
        )

    @app.route("/api/status", methods=["GET"])
    def _status() -> Any:
        try:
            snapshot = snapshot_store.get()
        except Exception:
            return jsonify({"error": SNAPSHOT_MISSING}), 500
        now = current_clock()
        published = initial_snapshot is not None
        if published:
            published = snapshot is not initial_snapshot
        return jsonify(_status_payload(snapshot, now, published, calibrated))

    @app.route("/api/summary", methods=["GET"])
    def _summary() -> Any:
        try:
            snapshot = snapshot_store.get()
        except Exception:
            return jsonify({"error": SNAPSHOT_MISSING}), 500
        return jsonify(_summary_payload(snapshot))

    @app.route("/api/timeseries", methods=["GET"])
    def _timeseries() -> Any:
        try:
            snapshot = snapshot_store.get()
        except Exception:
            return jsonify({"error": SNAPSHOT_MISSING}), 500
        return jsonify(_timeseries_payload(snapshot))

    @app.route("/api/export.csv", methods=["GET"])
    def _export() -> Any:
        try:
            raw = csv_path_provider()
        except Exception:
            return jsonify({"error": CSV_MISSING}), 404
        try:
            candidate = Path(raw)
        except Exception:
            return jsonify({"error": CSV_MISSING}), 404
        if not candidate.is_file():
            return jsonify({"error": CSV_MISSING}), 404
        return send_file(
            str(candidate),
            as_attachment=True,
            download_name=candidate.name,
            mimetype="text/csv",
        )

    return app


def serve_dashboard(snapshot_store: Any, csv_path_provider: Any, config: Any) -> None:
    port = int(config.output.port)
    try:
        calibrated_value: bool | None = bool(config.line.calibrated)
    except Exception:
        calibrated_value = None
    app = create_app(
        snapshot_store, csv_path_provider, port, calibrated=calibrated_value
    )
    import waitress

    waitress.serve(app, host=HOST_LABEL, port=port)
