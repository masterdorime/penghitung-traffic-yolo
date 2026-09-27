# One-Day CCTV Traffic Counter Design

**Date:** 2026-09-25  
**Status:** Approved  
**Stream:** `https://jmlive.jasamarga.com/hls/6/8ef68a94-4b79-4316-af23-7a6e6b7b88f5/index.m3u8` (Padalarang–Cileunyi KM 135+600 | SS KOPO, rotated ID discovered 2026-09-27; previous IDs: `.../1c65bfac-.../index.m3u8`, clearer feed `.../0aba6701-.../index.m3u8`, KM 138+500 | SS M TOHA `.../870e1e2b-.../index.m3u8`). NOTE: Jasamarga rotates feed IDs, so the URL must be re-resolved from the Bina Marga CCTV page when a playlist goes permanently empty.

## Purpose

Build a local Windows application that reads the Jasamarga CCTV HLS livestream, detects vehicles, counts each tracked vehicle once when it crosses a configured road line, and reports traffic density for the Pagi, Siang, and Sore periods. The application provides a live annotated video window, a local web dashboard, and a CSV export. It retains traffic data for the current Jakarta calendar day only and does not record video.

## Goals

- Run on the user's local Windows laptop or PC with Python 3.14.
- Operate reliably for one daily reporting cycle.
- Process only the requested reporting windows.
- Count cars, motorcycles, buses, and trucks once per valid track.
- Show live status, period totals, and traffic-rate information locally.
- Persist the current day's results in SQLite and export a matching CSV.
- Recover automatically from temporary HLS or network failures.
- Remain simple to launch, calibrate, test, and operate without a cloud service.

## Non-goals

- Saving or archiving CCTV video.
- Counting traffic outside 06:00–19:00.
- Vehicle-type breakdowns in the CSV, dashboard, or API.
- Multi-camera support, remote access, user authentication, or cloud deployment.
- Retaining traffic history beyond the current Jakarta calendar day.
- Replacing manual calibration or professional traffic-engineering measurement.

## Reporting Schedule

All schedule calculations use the `Asia/Jakarta` timezone.

| Period | Interval | Rule |
|---|---:|---|
| Pagi | 06:00–10:00 | `[06:00, 10:00)` |
| Siang | 10:00–15:00 | `[10:00, 15:00)` |
| Sore | 15:00–19:00 | `[15:00, 19:00)` |

A crossing at exactly 10:00 belongs to Siang. A crossing at exactly 15:00 belongs to Sore. A crossing at exactly 19:00 is outside the reporting schedule and is not counted. The application may remain open outside the reporting window, but it waits rather than running YOLO. At 19:00 it closes schedule admission and continues evaluating tracked vehicles for a configurable five-second drain window; only crossings whose interpolated timestamp is earlier than 19:00 are admitted.

## Architecture

Use a modular monolith: one local Python process with isolated modules and controlled background threads.

1. `config.py` loads and validates editable TOML configuration.
2. `stream.py` starts the bundled ffmpeg binary, reads HLS frames, and reconnects after failures.
3. `detector.py` loads Ultralytics YOLO and runs tracking.
4. `counter.py` converts detections into unique line-crossing events and healthy observation intervals.
5. `density.py` assigns events to periods and calculates totals and rates.
6. `storage.py` writes SQLite records, purges old data, and atomically regenerates the current-day CSV.
7. `overlay.py` renders the optional live OpenCV window.
8. `web.py` serves the local Flask dashboard and read-only JSON endpoints.
9. `main.py` owns startup, scheduling, lifecycle state, and graceful shutdown.

A bounded frame handoff prevents memory growth. When inference is slower than decoding, the stream producer replaces stale frames rather than queueing them. SQLite has one writer thread; inference submits persistence commands through a bounded queue. The Flask request threads never access SQLite or mutable tracking collections. Instead, the control layer publishes immutable, timestamped application snapshots under a short-lived lock, and all read-only UI and overlay paths consume those snapshots.

## Processing Flow

1. Validate configuration, output paths, model availability, and the virtual line before opening the stream.
2. Before 06:00, enter `waiting_for_window` and sleep without running inference.
3. At 06:00, start ffmpeg and the processing pipeline.
4. Decode frames at a target of 10 processed frames per second and retain only the newest unprocessed frame.
5. Run YOLO vehicle detection with ByteTrack-compatible tracking.
6. Filter detections to the fixed set of car, motorcycle, bus, and truck classes.
7. Use each tracked bounding box's bottom-center point for line-crossing evaluation.
8. Count a track only after it establishes one stable side and later establishes the opposite side.
9. Store a deduplicated crossing event transactionally before exposing it to the UI.
10. Update the live overlay and dashboard from immutable application snapshots.
11. At 19:00, close schedule admission, drain tracked vehicles for five seconds while admitting only pre-19:00 crossings, regenerate the final CSV, close the stream cleanly, and wait for the next day.
12. On restart, recover any current-day SQLite records and regenerate the matching CSV.

## Detection and Counting Rules

- The default model is Ultralytics `yolo26n.pt`; the model name remains configurable.
- The fixed contributing classes are car, motorcycle, bus, and truck. No other model class contributes to totals.
- The virtual line is stored as normalized coordinates so it remains valid if the source resolution changes.
- A track's side is the sign of its bottom-center point's signed distance from the directed line. A configurable normalized hysteresis band, default `0.005`, prevents line jitter; a latched side changes only after the point exits the band in the opposite direction.
- A track's first stable observation establishes its baseline side and is not itself a crossing. At least two stable observations on opposite sides are required.
- Every accepted event uses the interpolated crossing instant between the two stable observations that changed sides. When signed distances are `d0` and `d1` with opposite signs, the crossing time is `t0 + (t1 - t0) * abs(d0) / (abs(d0) + abs(d1))`. If the stable observations are more than two seconds apart, the older observation is discarded, the newer side becomes a fresh baseline, and no crossing is emitted. Frame observation times use a monotonic clock converted with a wall-clock offset sampled once per second; inference completion and database insertion times are diagnostic only.
- Crossing direction is recorded internally for diagnostics, but dashboard and CSV totals combine both directions.
- Every `(tracking_session_id, track_id)` pair can produce at most one count, regardless of later reversals or repeated crossings.
- A new `tracking_session_id` is created when the stream reconnects or the tracker resets. Tracks in the new session establish fresh baseline sides; a vehicle is not counted merely because tracking was reset.
- Vehicles that appear without crossing, remain inside the hysteresis band, or cross outside the reporting window are not counted.
- Named inference settings include confidence, IoU, maximum detections, inference size, tracker configuration, and `persist`; they are configurable without editing source code.

## Density Metrics

For each period, report:

- **Total:** number of accepted crossing events.
- **Observed seconds:** whole one-second buckets in which at least one frame completed inference while the stream was connected.
- **Average rate:** `total / observed seconds * 60`.
- **Rolling five-minute rate:** `60 * events in the window / healthy seconds in the window` for the five wall-clock minutes ending at the requested time, clipped to the period's half-open interval. For a completed period, use its final five minutes; for an unstarted period, return `null`.

Healthy seconds are set-unioned by whole wall-clock second and persisted in configurable buckets, default ten seconds, so overlapping processing intervals cannot be double-counted. Dropped or superseded frames do not by themselves make a second unhealthy, but a second with no completed inference is unhealthy. Disconnected and drain intervals are not healthy. Division-by-zero results return `null`.

The rolling rate requires at least `rolling_min_healthy_seconds`, default 60, and uses the current time for the live dashboard. Minute-aligned calculations use the end of the measured minute so results are deterministic.

The dashboard shows combined directional totals and does not display vehicle classes.

## Persistence

SQLite is the source of truth. The database is stored at `data/traffic.db`, and the current-day CSV is generated at `data/exports/traffic_YYYY-MM-DD.csv`, where the date is in `Asia/Jakarta`.

Logical records include:

- Application session start, stop, and lifecycle state.
- Tracking session identifier used for event deduplication.
- Unique count event with interpolated crossing timestamp, internal direction, and period.
- Aggregated healthy-second buckets used to calculate rates.
- Mandatory minute buckets for dashboard charting and CSV export.
- A session-scoped `dropped_events` counter for crossings that could not be persisted.

The event uniqueness constraint is enforced by SQLite, not only by application memory. Database writes use explicit transactions. A failed event write discards that uncommitted event, increments the session-scoped `dropped_events` counter, records the failure in the application log when possible, and transitions the application to `error`; the incomplete total is never presented as complete.

The current-day CSV is regenerated atomically once per minute, at 19:00, and during clean shutdown. It is derived from SQLite rather than maintained as an independent event log, preventing divergence after a crash. A minute rollup remains writable for two minutes after its end so an event interpolated from frames up to two seconds apart cannot arrive after the minute is first exported.

The CSV is minute-grained. It contains one row for every elapsed reporting minute from 06:00 through the current minute, including zero-event minutes, and no future rows. `minute_start` is ISO 8601 with the `+07:00` offset. Columns are:

- `date`
- `period`
- `minute_start`
- `count`: events whose crossing timestamps fall in the minute.
- `observed_seconds`: healthy seconds in the minute, from 0 to 60.
- `minute_rate`: `count / observed_seconds * 60`, or `null` when no healthy time exists.
- `period_total`: cumulative events in the period through the end of the minute.
- `period_observed_seconds`: cumulative healthy seconds in the period.
- `period_rate`: `period_total / period_observed_seconds * 60`, or `null` when no healthy time exists.

## One-Day Retention

The SQLite database and downloadable CSV retain only the current `Asia/Jakarta` calendar date. On startup, on the first request after a date rollover, and immediately after crossing local midnight, it:

1. Deletes older sessions, events, health buckets, and minute buckets.
2. Deletes CSV files whose filename date is not the current Jakarta date.
3. Regenerates the current CSV if needed.
4. Records cleanup results without exposing traffic data from prior days.

The final report remains available from 19:00 until midnight. The dashboard displays a warning after 19:00 that the current report will be removed at midnight; users needing a durable copy must download the CSV before then.

## Local API and Dashboard

The Flask server binds unconditionally to `127.0.0.1`; only the port is configurable. It rejects requests whose `Host` header is not `127.0.0.1:<port>`, `localhost:<port>`, or `[::1]:<port>`, sends no CORS headers, and exposes read-only `GET` routes. Local-only binding is not an authentication boundary: processes or browser contexts running on the same machine may read the current day's data.

| Route | Purpose |
|---|---|
| `/` | Dashboard page |
| `/api/status` | Lifecycle, stream, inference, FPS, snapshot age, export health, dropped events, and sanitized last error |
| `/api/summary` | Current Pagi, Siang, and Sore totals and rates |
| `/api/timeseries` | Elapsed minute-level data for the current day |
| `/api/export.csv` | Download the current-day generated CSV |

A period that has not begun returns `started: false`, `total: 0`, `observed_seconds: 0`, and null rates. `/api/timeseries` returns an ascending array with `minute_start`, `count`, and `observed_seconds` for every elapsed minute in `[06:00, 19:00)`, including zero minutes. `/api/export.csv` returns `404` with a JSON error before the first successful generation; otherwise it downloads `traffic_YYYY-MM-DD.csv`.

The dashboard shows connection status, current processing rate, three period cards, a time-series chart, last update time, persistence warnings, and CSV download. It polls once per second; no public WebSocket or server-sent-event infrastructure is required.

## Configuration

`config.toml` contains:

- Stream URL.
- Model path/name, defaulting to `yolo26n.pt`.
- Device selection, defaulting to CPU.
- Inference image size, defaulting to `640`.
- Target processed frames per second, defaulting to `10`.
- Named confidence, IoU, maximum-detection, tracker, and persistence settings.
- Normalized line endpoints and hysteresis, defaulting to `0.005`, plus a `calibrated` boolean defaulting to `false`.
- Local port, defaulting to `5000`; host is not configurable.
- Database, CSV, calibration-image, and log paths.
- Reconnect backoff bounds, defaulting from one to 30 seconds.
- Health-bucket size, defaulting to ten seconds.
- Minimum healthy seconds for the rolling rate, defaulting to 60.

Operational profiles:

- CPU baseline (fallback): `yolo26n.pt`, `cpu`, confidence `0.25`, image size `640`, line `(0.10,0.80)-(0.90,0.80)`, `calibrated=false`.
- GPU operational profile (deployed): `yolo26s.pt`, `cuda:0`, confidence `0.15`, image size `960`, SS Kopo line `(0.08,0.62)-(0.92,0.62)`, `calibrated=true`.

The `Asia/Jakarta` timezone, the three reporting periods, and the four contributing vehicle classes are fixed application constants rather than configuration. Startup fails with a clear message when required values are missing, malformed, outside validated ranges, or when line endpoints lie outside normalized frame bounds. The line's `calibrated` flag defaults to `false`; the overlay shows an explicit calibration warning until the user runs the calibration flow. The application never silently lowers the configured target frame rate; it reports actual FPS and dropped-frame count.

## Dependencies

- `ultralytics` for YOLO inference and tracking.
- `opencv-python` for decoding-independent frame presentation and overlay rendering.
- `imageio-ffmpeg` for a project-local ffmpeg binary; no manual system installation is required.
- `flask` for the local dashboard and API.
- Python's standard library for scheduling, SQLite, TOML parsing, subprocess management, and CSV generation.

The implementation plan must verify current Python 3.14 wheel compatibility before finalizing dependency pins.

## Reliability and Error Handling

- Validate and warm the model before processing begins.
- Start ffmpeg as a managed subprocess and capture a bounded amount of diagnostic output.
- On stream failure, stop dependent processing and reconnect after capped exponential backoff.
- Use an initial retry delay of one second and a maximum delay of 30 seconds, with jitter.
- While disconnected, show `degraded` status and the last sanitized error on the dashboard; tracking, persistence, and reporting remain available.
- Drop stale frames instead of allowing an unbounded queue, while using the completed-inference rule for healthy seconds.
- Persist each accepted count before updating dashboard totals.
- On database event-write failure, discard the uncommitted event, increment session-scoped `dropped_events`, transition to `error`, and require a user restart.
- On CSV regeneration failure, keep counting because SQLite remains authoritative; expose `csv_export_ok: false` and retry at the next minute boundary without changing the lifecycle state.
- Handle Ctrl+C and process termination with a graceful stop sequence.
- Do not expose raw subprocess output or local filesystem paths unnecessarily through the API.

## Lifecycle States

The application exposes one of these states:

- `starting`
- `waiting_for_window`
- `connecting`
- `running`
- `degraded`
- `stopping`
- `stopped`
- `error`

State transitions are centralized in `main.py` and reflected consistently in the overlay and dashboard. `degraded` means that the stream is disconnected or inference is below its target; it does not represent a CSV export warning. CSV health is exposed separately in `/api/status`, while a database event-write failure uses `error`.

## Dashboard Visual System

The local dashboard uses a restrained, authoritative traffic-operations style. It is laptop-first and responsive, with a dark navy foundation, amber primary signal, and green/red status accents. Bahnschrift is used for headings, Segoe UI for body copy, and Consolas for numeric metrics. Spacing follows a 4px base grid, radii stay at 6px, shadows remain restrained, and transitions use 120ms with reduced-motion support.

The information order is live status, three period cards, minute-level chart, diagnostics, and CSV export. The chart is dependency-free inline SVG with a semantic table fallback. No external font, chart, or icon CDN is required for local operation.

## Testing and Verification

### Unit tests

Verify:

- Period assignment at every exact boundary using the interpolated crossing timestamp.
- Average and deterministic minute-aligned rolling-rate calculations, including empty, partial, and disconnected intervals.
- Whole-second healthy buckets are set-unioned and never double-counted.
- Normalized line validation, signed crossing geometry, hysteresis, and first-observation baseline behavior.
- Duplicate crossing rejection for the same tracking session and track ID, including reversals.
- Direction derivation.
- Zero-minute CSV row inclusion, cumulative period fields, null-rate behavior, and atomic replacement.
- Retention, date-rollover cleanup, and the exact `/api/export.csv` resolution rule.
- Backoff bounds, degraded-state behavior, and lifecycle transitions.
- Host-header rejection and the absence of state-changing routes or CORS headers.

### Integration tests

Verify:

- Fake detections crossing a test line in both directions with jitter around the hysteresis band.
- SQLite uniqueness under repeated event submission.
- Minute aggregation, summary response shapes, and dashboard snapshot reads without web-thread database access.
- CSV regeneration matching the database, including recovery after a transient export failure while counting continues.
- Stream disconnect, retry, and tracker-session recreation using fakes; tracks in a recreated tracker establish a new baseline rather than counting automatically.
- Database event-write failure increments `dropped_events`, exposes an incomplete-data warning, and stops automatic operation.
- Graceful shutdown while a frame is being processed.

### Live smoke test

Run against the actual HLS stream long enough to verify:

- ffmpeg connects and produces frames.
- The model loads on Python 3.14.
- The overlay and dashboard update.
- A user-triggered calibration command can save one still frame under `data/calibration/` for line placement; no video or frame sequence is saved.
- The application remains stable during a controlled disconnect and reconnect.

### Accuracy calibration

1. Place the line where vehicles should be counted.
2. Record a representative 10-minute sample.
3. Manually count the same crossings.
4. Compare the application total with the manual total.
5. If fewer than 30 vehicles are observed, extend the sample until 30 vehicles are observed, up to a practical maximum of 30 minutes.
6. Adjust line placement or thresholds and repeat if needed.

Acceptance requires:

- No duplicate `(tracking_session_id, track_id)` events and no count caused solely by tracker reset.
- `dropped_events` remains zero for a normal acceptance run.
- With no writes in flight, the dashboard period total equals the SQLite event count for that period, and the sum of CSV `count` values equals the same value.
- Automatic recovery after a temporary stream failure.
- For a manual sample containing at least 30 crossings, `abs(app_total - manual_total) / manual_total` is no greater than `0.10`.

## Operational Assumptions

- The stream remains publicly accessible without credentials.
- The camera view is stable enough for a normalized virtual line.
- A vehicle crossing can be tracked long enough to produce a valid before/after line state.
- CPU inference at a reduced processing rate is acceptable; accuracy and stable operation take priority over processing every source frame.
- The local machine remains powered on for the active reporting window.

## Open Questions

None. Product scope, schedule, metrics, output format, retention, architecture, reliability behavior, and verification criteria were reviewed and approved before this specification was written.
