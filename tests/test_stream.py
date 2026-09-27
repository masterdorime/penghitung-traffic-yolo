from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from io import BytesIO
from types import SimpleNamespace
from typing import Final

import numpy as np
import pytest

from traffic_counter import stream as stream_module
from traffic_counter.config import StreamConfig, TimingConfig
from traffic_counter.models import Frame, FrameSlotStats, StreamState
from traffic_counter.state import LatestFrameSlot
from traffic_counter.stream import (
    USER_AGENT,
    HlsFrameSource,
    SpawnedProcess,
    StreamDisconnected,
    build_ffmpeg_command,
)
from traffic_counter.timeutils import JAKARTA

WIDTH: Final[int] = 4
HEIGHT: Final[int] = 2
FRAME_BYTES: Final[int] = WIDTH * HEIGHT * 3
STREAM_URL: Final[str] = "https://example.test/hls/segment-one/index.m3u8"
EXECUTABLE: Final[str] = "ffmpeg-test.exe"
EXPECTED_FILTER: Final[str] = (
    "fps=10,scale=4:2:force_original_aspect_ratio=decrease,pad=4:2:(ow-iw)/2:(oh-ih)/2:black"
)
EXPECTED_COMMAND: Final[tuple[str, ...]] = (
    EXECUTABLE,
    "-hide_banner",
    "-nostdin",
    "-loglevel",
    "warning",
    "-user_agent",
    USER_AGENT,
    "-reconnect",
    "1",
    "-reconnect_at_eof",
    "1",
    "-reconnect_streamed",
    "1",
    "-reconnect_delay_max",
    "5",
    "-i",
    STREAM_URL,
    "-an",
    "-vf",
    EXPECTED_FILTER,
    "-f",
    "rawvideo",
    "-pix_fmt",
    "bgr24",
    "pipe:1",
)
NOMINAL_BACKOFF: Final[tuple[float, ...]] = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0, 30.0, 30.0)
START: Final[datetime] = datetime(2026, 9, 25, 7, 0, tzinfo=JAKARTA)


def payload(marker: int, count: int = FRAME_BYTES) -> bytes:
    return bytes([marker]) * count


def pixels(base: int) -> bytes:
    return bytes([base, base + 1, base + 2]) * (FRAME_BYTES // 3)


def at(second: float) -> datetime:
    return START + timedelta(seconds=second)


class StepClock:
    def __init__(self, start: datetime = START, step: timedelta = timedelta(milliseconds=100)):
        self.current = start
        self.step = step
        self.calls = 0

    def __call__(self) -> datetime:
        value = self.current
        self.current = value + self.step
        self.calls += 1
        return value


class FakePipe(BytesIO):
    def __init__(self, data: bytes, chunk_limit: int | None = None, over_read: int = 0) -> None:
        super().__init__(data)
        self.chunk_limit = chunk_limit
        self.over_read = over_read
        self.requests: list[int] = []
        self.is_closed = False

    def read(self, size: int | None = -1, /) -> bytes:
        self.requests.append(-1 if size is None else size)
        if self.is_closed:
            return b""
        available = self.getbuffer().nbytes
        if available == 0:
            return b""
        wanted = available if size is None or size < 0 else min(size, available)
        limit = wanted if self.chunk_limit is None else min(wanted, self.chunk_limit)
        chunk = super().read(limit)
        if self.over_read:
            chunk += super().read(self.over_read)
        return chunk

    def close(self) -> None:
        self.is_closed = True


class InterruptingPipe(FakePipe):
    def read(self, size: int | None = -1, /) -> bytes:
        raise KeyboardInterrupt


class FakeStdin(BytesIO):
    def __init__(self) -> None:
        super().__init__()
        self.is_closed = False

    def close(self) -> None:
        self.is_closed = True


class FakeCapture(BytesIO):
    def __init__(self, data: bytes = b"") -> None:
        super().__init__(data)
        self.fails = False
        self.bytes_read = 0
        self.read_calls = 0
        self.window_start = 0

    def seek(self, offset: int, whence: int = 0, /) -> int:
        if self.fails:
            raise OSError("simulated diagnostic capture failure")
        return super().seek(offset, whence)

    def tell(self) -> int:
        return super().tell()

    def read(self, size: int | None = -1, /) -> bytes:
        self.read_calls += 1
        self.window_start = super().tell()
        chunk = super().read(size if size is not None else -1)
        self.bytes_read += len(chunk)
        return chunk


class FakeProcess:
    def __init__(
        self,
        stdout: FakePipe | None = None,
        *,
        exit_code: int | None = None,
        ignore_terminate: bool = False,
    ) -> None:
        self.stdin = FakeStdin()
        self.stdout = FakePipe(b"") if stdout is None else stdout
        self.exit_code = exit_code
        self.ignore_terminate = ignore_terminate
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_timeouts: list[float | None] = []
        self.alive = exit_code is None

    @property
    def returncode(self) -> int | None:
        return None if self.alive else self.exit_code

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        if not self.ignore_terminate:
            self.shutdown()

    def kill(self) -> None:
        self.kill_calls += 1
        self.shutdown()

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        if self.alive:
            raise subprocess.TimeoutExpired(cmd=EXECUTABLE, timeout=timeout or 0.0)
        return self.exit_code or 0

    def shutdown(self) -> None:
        self.alive = False
        self.stdout.close()


class FakeEnvironment:
    def __init__(
        self,
        outputs: tuple[bytes, ...] = (),
        *,
        chunk: int | None = None,
        over_read: int = 0,
        exit_codes: tuple[int | None, ...] = (),
        stderr_texts: tuple[str, ...] = (),
        spawn_errors: tuple[OSError, ...] = (),
        captures: tuple[FakeCapture, ...] = (),
        ignore_terminate: bool = False,
        interrupt: bool = False,
        jitter: float = 0.5,
        events: list[str] | None = None,
    ) -> None:
        self.outputs = outputs or (b"",)
        self.chunk = chunk
        self.over_read = over_read
        self.exit_codes = exit_codes
        self.stderr_texts = stderr_texts
        self.spawn_errors = spawn_errors
        self.captures = list(captures)
        self.ignore_terminate = ignore_terminate
        self.interrupt = interrupt
        self.jitter_value = jitter
        self.events = events
        self.commands: list[list[str]] = []
        self.processes: list[FakeProcess] = []
        self.sleeps: list[float] = []
        self.states_at_start: list[StreamState] = []
        self.states_at_sleep: list[StreamState] = []
        self.stop_after_sleeps: int | None = None
        self.source: HlsFrameSource | None = None

    def spawn(self, command: list[str]) -> SpawnedProcess:
        index = len(self.commands)
        self.commands.append(list(command))
        if self.source is not None:
            self.states_at_start.append(self.source.diagnostics().state)
        if index < len(self.spawn_errors):
            raise self.spawn_errors[index]
        capture = self.capture_for(index)
        if index < len(self.stderr_texts):
            capture.write(self.stderr_texts[index].encode("utf-8"))
        process = FakeProcess(
            self.pipe_for(index),
            exit_code=self.exit_code_for(index),
            ignore_terminate=self.ignore_terminate,
        )
        self.processes.append(process)
        return SpawnedProcess(process=process, diagnostics=capture)

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        if self.events is not None:
            self.events.append("sleep")
        if self.source is not None:
            self.states_at_sleep.append(self.source.diagnostics().state)
        limit = self.stop_after_sleeps
        if limit is not None and len(self.sleeps) >= limit and self.source is not None:
            self.source.stop()

    def jitter(self) -> float:
        return self.jitter_value

    def output_for(self, index: int) -> bytes:
        return self.outputs[index] if index < len(self.outputs) else self.outputs[-1]

    def pipe_for(self, index: int) -> FakePipe:
        if self.interrupt:
            return InterruptingPipe(self.output_for(index))
        return FakePipe(self.output_for(index), self.chunk, self.over_read)

    def exit_code_for(self, index: int) -> int | None:
        return self.exit_codes[index] if index < len(self.exit_codes) else None

    def capture_for(self, index: int) -> FakeCapture:
        while len(self.captures) <= index:
            self.captures.append(FakeCapture())
        return self.captures[index]

    def build(
        self,
        *,
        config: StreamConfig | None = None,
        timing: TimingConfig | None = None,
        clock: StepClock | None = None,
        executable: str | None = EXECUTABLE,
        default_sleep: bool = False,
        slot_stats: Callable[[], FrameSlotStats] | None = None,
        on_disconnect: Callable[[str], None] | None = None,
    ) -> HlsFrameSource:
        source = HlsFrameSource(
            config
            if config is not None
            else StreamConfig(url=STREAM_URL, width=WIDTH, height=HEIGHT),
            timing if timing is not None else TimingConfig(),
            clock=clock if clock is not None else StepClock(),
            executable=executable,
            spawn=self.spawn,
            sleep=None if default_sleep else self.sleep,
            jitter=self.jitter,
            slot_stats=slot_stats,
            on_disconnect=on_disconnect,
        )
        self.source = source
        return source

    def collect(self, source: HlsFrameSource, count: int) -> list[Frame]:
        frames: list[Frame] = []
        for frame in source.frames():
            frames.append(frame)
            if count and len(frames) >= count:
                source.stop()
        return frames

    def drain(self, source: HlsFrameSource, stop_after_sleeps: int) -> list[Frame]:
        self.stop_after_sleeps = stop_after_sleeps
        return self.collect(source, 0)


def test_ffmpeg_command_uses_fixed_dimensions_and_raw_bgr() -> None:
    command = build_ffmpeg_command(
        executable=EXECUTABLE,
        url=STREAM_URL,
        width=WIDTH,
        height=HEIGHT,
    )
    assert command[:2] == [EXECUTABLE, "-hide_banner"]
    assert "-reconnect_streamed" in command
    assert EXPECTED_FILTER in command
    assert command[-5:] == ["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]


def test_ffmpeg_command_matches_the_approved_argument_order() -> None:
    command = build_ffmpeg_command(
        executable=EXECUTABLE, url=STREAM_URL, width=WIDTH, height=HEIGHT
    )
    assert tuple(command) == EXPECTED_COMMAND
    assert command.index("-i") + 1 == command.index(STREAM_URL)
    assert command.index("-i") < command.index("-an") < command.index("-vf") < command.index("-f")


def test_ffmpeg_command_places_every_reconnect_flag_before_the_input() -> None:
    command = build_ffmpeg_command(
        executable=EXECUTABLE, url=STREAM_URL, width=WIDTH, height=HEIGHT
    )
    url_index = command.index(STREAM_URL)
    for flag in (
        "-reconnect",
        "-reconnect_at_eof",
        "-reconnect_streamed",
        "-reconnect_delay_max",
    ):
        assert command.index(flag) < url_index


def test_ffmpeg_command_uses_the_configured_frame_rate_and_size() -> None:
    command = build_ffmpeg_command(
        executable=EXECUTABLE, url=STREAM_URL, width=640, height=360, target_fps=5
    )
    assert (
        "fps=5,scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2:black"
    ) in command


def test_ffmpeg_command_keeps_the_url_as_one_argument() -> None:
    url = "https://example.test/hls/a b/index.m3u8?token=secret&x=1"
    command = build_ffmpeg_command(executable=EXECUTABLE, url=url, width=WIDTH, height=HEIGHT)
    assert command[command.index("-i") + 1] == url
    assert len([item for item in command if item == url]) == 1


def test_ffmpeg_command_rejects_non_positive_dimensions() -> None:
    with pytest.raises(ValueError, match="width and height"):
        build_ffmpeg_command(executable=EXECUTABLE, url=STREAM_URL, width=0, height=2)
    with pytest.raises(ValueError, match="width and height"):
        build_ffmpeg_command(executable=EXECUTABLE, url=STREAM_URL, width=4, height=-1)


def test_ffmpeg_command_rejects_blank_identifiers() -> None:
    with pytest.raises(ValueError, match="url"):
        build_ffmpeg_command(executable=EXECUTABLE, url="  ", width=WIDTH, height=HEIGHT)
    with pytest.raises(ValueError, match="executable"):
        build_ffmpeg_command(executable=" ", url=STREAM_URL, width=WIDTH, height=HEIGHT)


def test_ffmpeg_command_rejects_a_non_positive_frame_rate() -> None:
    with pytest.raises(ValueError, match="frame rate"):
        build_ffmpeg_command(
            executable=EXECUTABLE, url=STREAM_URL, width=WIDTH, height=HEIGHT, target_fps=0
        )


def test_injected_executable_is_used_without_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden() -> str:
        raise AssertionError("Resolution must not run when an executable is injected.")

    monkeypatch.setattr(stream_module, "resolve_ffmpeg_executable", forbidden)
    environment = FakeEnvironment((payload(1),))
    source = environment.build()
    environment.collect(source, 1)
    assert environment.commands[0][0] == EXECUTABLE


def test_missing_executable_is_resolved_through_imageio_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved: Final[str] = "C:/tools/ffmpeg-7.1.exe"
    calls: list[str] = []

    def resolver() -> str:
        calls.append("resolved")
        return resolved

    monkeypatch.setattr(stream_module, "resolve_ffmpeg_executable", resolver)
    environment = FakeEnvironment((payload(1),))
    source = environment.build(executable=None)
    environment.collect(source, 1)
    assert calls == ["resolved"]
    assert environment.commands[0][0] == resolved


def test_unresolvable_executable_is_reported_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing() -> str:
        raise RuntimeError("bundled ffmpeg could not be located")

    monkeypatch.setattr(stream_module, "resolve_ffmpeg_executable", failing)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        HlsFrameSource(StreamConfig(url=STREAM_URL, width=WIDTH, height=HEIGHT), TimingConfig())


def fake_imageio_ffmpeg(monkeypatch: pytest.MonkeyPatch, lookup: Callable[[], str]) -> None:
    module = SimpleNamespace(get_ffmpeg_exe=lookup)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", module)


def test_resolve_ffmpeg_executable_delegates_to_imageio_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def lookup() -> str:
        calls.append("get_ffmpeg_exe")
        return "C:/bundled/ffmpeg-7.1.exe"

    fake_imageio_ffmpeg(monkeypatch, lookup)
    assert stream_module.resolve_ffmpeg_executable() == "C:/bundled/ffmpeg-7.1.exe"
    assert calls == ["get_ffmpeg_exe"]


def test_resolve_ffmpeg_executable_reports_a_missing_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        stream_module.resolve_ffmpeg_executable()


def test_resolve_ffmpeg_executable_reports_a_failed_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def lookup() -> str:
        raise OSError("no bundled binary on this machine")

    fake_imageio_ffmpeg(monkeypatch, lookup)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        stream_module.resolve_ffmpeg_executable()


def test_source_rejects_non_positive_frame_dimensions() -> None:
    with pytest.raises(ValueError, match="width and height"):
        HlsFrameSource(
            StreamConfig(url=STREAM_URL, width=0, height=0), TimingConfig(), executable=EXECUTABLE
        )


def test_one_frame_is_read_from_a_fake_process() -> None:
    environment = FakeEnvironment((pixels(10),))
    source = environment.build()
    frames = environment.collect(source, 1)
    assert len(frames) == 1
    assert frames[0].image[0, 0].tolist() == [10, 11, 12]
    assert environment.commands[0] == list(EXPECTED_COMMAND)


def test_frame_image_is_a_contiguous_bgr_uint8_view() -> None:
    environment = FakeEnvironment((pixels(0),))
    source = environment.build()
    frame = environment.collect(source, 1)[0]
    assert frame.image.dtype == np.uint8
    assert frame.image.shape == (HEIGHT, WIDTH, 3)
    assert frame.image.flags["C_CONTIGUOUS"]
    assert not frame.image.flags["WRITEABLE"]


def test_observed_at_comes_from_the_injected_clock_once_per_frame() -> None:
    clock = StepClock(step=timedelta(milliseconds=250))
    environment = FakeEnvironment((payload(1) + payload(2),))
    source = environment.build(clock=clock)
    frames = environment.collect(source, 2)
    assert [frame.observed_at for frame in frames] == [at(0), at(0.25)]
    assert clock.calls == 2


def test_observed_at_is_aware_jakarta() -> None:
    environment = FakeEnvironment((payload(1),))
    source = environment.build(clock=StepClock())
    frame = environment.collect(source, 1)[0]
    assert frame.observed_at.tzinfo is not None
    assert frame.observed_at.utcoffset() == timedelta(hours=7)
    assert frame.observed_at.astimezone(JAKARTA) == frame.observed_at


def test_a_short_read_does_not_splice_two_partial_frames() -> None:
    environment = FakeEnvironment((payload(1) + payload(2),), chunk=7)
    source = environment.build()
    frames = environment.collect(source, 2)
    assert [int(frame.image[0, 0, 0]) for frame in frames] == [1, 2]
    requests = environment.processes[0].stdout.requests
    assert requests[0] == FRAME_BYTES
    assert all(request <= FRAME_BYTES for request in requests)


def test_partial_frame_is_reported_and_the_source_reconnects() -> None:
    environment = FakeEnvironment((payload(1) + b"\x07" * 5,))
    source = environment.build()
    frames = environment.drain(source, 1)
    assert [int(frame.image[0, 0, 0]) for frame in frames] == [1]
    diagnostics = source.diagnostics()
    assert environment.states_at_sleep == [StreamState.DISCONNECTED]
    assert diagnostics.attempts == 1
    assert diagnostics.last_error is not None
    assert f"5 of {FRAME_BYTES} bytes" in diagnostics.last_error
    assert len(environment.commands) == 1
    assert environment.sleeps == [1.0]


def test_clean_end_of_stream_is_reported() -> None:
    environment = FakeEnvironment((b"",))
    source = environment.build()
    assert environment.drain(source, 1) == []
    diagnostics = source.diagnostics()
    assert environment.states_at_sleep == [StreamState.DISCONNECTED]
    assert diagnostics.last_error is not None
    assert "ended" in diagnostics.last_error


def test_process_exit_code_is_reported_after_a_final_frame() -> None:
    environment = FakeEnvironment((payload(1),), exit_codes=(1,))
    source = environment.build()
    frames = environment.drain(source, 1)
    assert [int(frame.image[0, 0, 0]) for frame in frames] == [1]
    diagnostics = source.diagnostics()
    assert environment.states_at_sleep == [StreamState.DISCONNECTED]
    assert diagnostics.last_error is not None
    assert "exited" in diagnostics.last_error
    assert "1" in diagnostics.last_error


def test_a_process_that_exits_before_any_frame_reports_the_exit_code() -> None:
    environment = FakeEnvironment((b"",), exit_codes=(137,))
    source = environment.build()
    assert environment.drain(source, 1) == []
    diagnostics = source.diagnostics()
    assert diagnostics.last_error is not None
    assert "137" in diagnostics.last_error


def test_spawn_failure_is_reported_without_paths() -> None:
    environment = FakeEnvironment(
        (b"",),
        spawn_errors=(OSError("Cannot run program 'C:\\secret\\ffmpeg.exe'"),),
    )
    source = environment.build()
    assert environment.drain(source, 1) == []
    diagnostics = source.diagnostics()
    assert environment.states_at_sleep == [StreamState.DISCONNECTED]
    assert diagnostics.attempts == 1
    assert diagnostics.last_error == stream_module._SPAWN_FAILED
    assert "secret" not in (diagnostics.last_error or "")


def test_a_pipe_that_over_delivers_is_treated_as_a_disconnect() -> None:
    environment = FakeEnvironment((payload(1) + payload(2),), over_read=4)
    source = environment.build()
    assert environment.drain(source, 1) == []
    assert environment.source is not None
    assert environment.source.diagnostics().last_error == stream_module._OVERREAD


def test_stderr_tail_keeps_only_the_last_twenty_lines() -> None:
    lines = "\n".join(f"line {index}" for index in range(40))
    environment = FakeEnvironment((b"",), stderr_texts=(lines,))
    source = environment.build()
    environment.drain(source, 1)
    tail = source.diagnostics().stderr_tail
    assert len(tail) == 20
    assert tail[0] == "line 20"
    assert tail[-1] == "line 39"


def test_stderr_tail_drops_blank_lines_and_progress_carriage_returns() -> None:
    text = "frame= 1\r\n\r\n   \nframe= 2\r\nframe= 3\r\n"
    environment = FakeEnvironment((b"",), stderr_texts=(text,))
    source = environment.build()
    environment.drain(source, 1)
    assert source.diagnostics().stderr_tail == ("frame= 1", "frame= 2", "frame= 3")


def test_stderr_tail_never_exposes_paths_urls_or_secrets() -> None:
    text = (
        "Input #0, hls, from 'https://user:p4ssw0rd@cctv.test/hls/6/abc/index.m3u8?token=abc123':\n"
        "Failed to load 'C:\\Users\\Operator\\secret\\playlist.m3u8'\n"
        "Could not read from /var/lib/ffmpeg/segment-9.ts\n"
        "Authorization: Bearer zzz-secret-value\n"
    )
    environment = FakeEnvironment((b"",), stderr_texts=(text,))
    source = environment.build()
    environment.drain(source, 1)
    tail = source.diagnostics().stderr_tail
    assert len(tail) == 4
    joined = "\n".join(tail)
    for leaked in (
        "p4ssw0rd",
        "abc123",
        "index.m3u8",
        "Operator",
        "secret",
        "zzz-secret-value",
        "/var/lib",
    ):
        assert leaked not in joined
    assert "https://cctv.test/<redacted>" in tail[0]
    assert tail[1] == "Failed to load '<path>'"
    assert tail[2] == "Could not read from <path>"
    assert tail[3] == "Authorization: <redacted>"


def test_stderr_tail_strips_ansi_escapes_and_truncates_long_lines() -> None:
    coloured = "\x1b[0;35m[hls @ 0000] Opening segment\x1b[0m for reading"
    text = f"{coloured}\n" + ("x" * 400) + "\n"
    environment = FakeEnvironment((b"",), stderr_texts=(text,))
    source = environment.build()
    environment.drain(source, 1)
    tail = source.diagnostics().stderr_tail
    assert len(tail) == 2
    assert tail[0] == "[hls @ 0000] Opening segment for reading"
    assert "\x1b" not in tail[0]
    assert len(tail[1]) == 200
    assert tail[1].endswith("...")


def test_stderr_tail_reads_at_most_the_last_sixteen_kibibytes() -> None:
    lines = "\n".join(f"diagnostic line {index:05d}" for index in range(4000))
    assert len(lines.encode("utf-8")) > 20 * 1024
    environment = FakeEnvironment((b"",), stderr_texts=(lines,))
    source = environment.build()
    environment.drain(source, 1)
    capture = environment.captures[0]
    tail = source.diagnostics().stderr_tail
    assert capture.bytes_read <= 16 * 1024
    assert capture.read_calls == 1
    assert len(tail) == 20
    assert tail[-1] == "diagnostic line 03999"
    assert all(entry not in tail for entry in ("line 00000", "line 01999"))


def test_stderr_tail_discards_a_partial_first_line_at_the_window_edge() -> None:
    filler = "\n".join("x" * 60 for _ in range(400))
    text = f"{'diagnostic first line'}\n{filler}\ndiagnostic last line\n"
    environment = FakeEnvironment((b"",), stderr_texts=(text,))
    source = environment.build()
    environment.drain(source, 1)
    capture = environment.captures[0]
    tail = source.diagnostics().stderr_tail
    assert capture.bytes_read == 16 * 1024
    assert tail[-1] == "diagnostic last line"
    assert all(line.strip() == "" or line.startswith(("x", "diagnostic")) for line in tail)
    assert not any(line.startswith("xxdiagnostic") for line in tail)


def test_a_capture_smaller_than_the_window_is_read_whole() -> None:
    environment = FakeEnvironment((b"",), stderr_texts=("first\nsecond\n",))
    source = environment.build()
    environment.drain(source, 1)
    capture = environment.captures[0]
    assert capture.bytes_read == len("first\nsecond\n")
    assert source.diagnostics().stderr_tail == ("first", "second")


def test_a_window_without_a_whole_line_yields_no_tail() -> None:
    environment = FakeEnvironment((b"",), stderr_texts=("y" * (16 * 1024 + 500),))
    source = environment.build()
    environment.drain(source, 1)
    capture = environment.captures[0]
    assert capture.bytes_read == 16 * 1024
    assert source.diagnostics().stderr_tail == ()


def test_a_two_hundred_thousand_character_line_sanitizes_without_pathological_delay() -> None:
    hostile = ("key=" + "a/" * 64 + "https://host/x?token=y ") * 3000
    assert len(hostile) > 200_000
    started = time.monotonic()
    line = stream_module.sanitize_diagnostic_line(hostile)
    elapsed = time.monotonic() - started
    assert elapsed < 10.0
    assert 0 < len(line) <= 200
    for leaked in ("key=a", "token=y", "https://host/x", "hdnts"):
        assert leaked not in line


def test_a_long_single_token_line_sanitizes_without_pathological_delay() -> None:
    hostile = "a" * 250_000
    started = time.monotonic()
    line = stream_module.sanitize_diagnostic_line(hostile)
    assert time.monotonic() - started < 10.0
    assert len(line) == 200
    assert line.endswith("...")


def test_the_sanitizer_only_inspects_the_bounded_input_window() -> None:
    head_secret = "token=" + "h" * 10
    filler = "b" * (stream_module.SANITIZE_INPUT_CAP + 100)
    line = stream_module.sanitize_diagnostic_line(head_secret + filler)
    assert head_secret not in line
    assert len(line) == 200
    assert line.endswith("...")


def test_a_secret_inside_the_retained_window_is_still_redacted() -> None:
    raw = "token=" + "h" * 10 + "b" * (stream_module.SANITIZE_INPUT_CAP - 16)
    assert len(raw) == stream_module.SANITIZE_INPUT_CAP
    line = stream_module.sanitize_diagnostic_line(raw)
    assert "token=<redacted>" in line


def test_sanitizer_binds_input_before_any_regex() -> None:
    assert stream_module.SANITIZE_INPUT_CAP == 4096
    assert stream_module.MAX_SANITIZED_LINE == 200


def test_url_query_without_a_path_is_redacted() -> None:
    line = stream_module.sanitize_diagnostic_line(
        "GET https://cctv.test/index.m3u8?hdnts=exp=1234~acl=/*~hmac=deadbeef HTTP/1.1"
    )
    assert "deadbeef" not in line
    assert "hdnts" not in line
    assert line.startswith("GET https://cctv.test/<redacted>")


def test_url_fragment_without_a_path_is_redacted() -> None:
    line = stream_module.sanitize_diagnostic_line(
        "refused https://cctv.test#access_token=supersecretvalue"
    )
    assert "supersecretvalue" not in line
    assert line == "refused https://cctv.test/<redacted>"


def test_url_userinfo_and_query_are_both_redacted() -> None:
    line = stream_module.sanitize_diagnostic_line("https://user:p4ssw0rd@cctv.test?sig=abcd1234")
    for leaked in ("p4ssw0rd", "user", "abcd1234"):
        assert leaked not in line
    assert line == "https://cctv.test/<redacted>"


def test_url_userinfo_without_a_path_keeps_only_the_host() -> None:
    assert stream_module.sanitize_diagnostic_line("https://user:p4ssw0rd@cctv.test") == (
        "https://cctv.test"
    )
    assert stream_module.sanitize_diagnostic_line("ftp://user:p4ssw0rd@cctv.test") == (
        "ftp://cctv.test"
    )


def test_percent_encoded_userinfo_is_redacted() -> None:
    line = stream_module.sanitize_diagnostic_line("https://user:pa%40ssw0rd@cctv.test/p")
    assert "pa%40ssw0rd" not in line
    assert line == "https://cctv.test/<redacted>"


def test_protocol_relative_userinfo_is_redacted() -> None:
    line = stream_module.sanitize_diagnostic_line("//user:p4ssw0rd@cctv.test/p")
    assert "p4ssw0rd" not in line
    assert line.startswith("//<redacted>@")


def test_userinfo_after_a_digit_prefixed_scheme_is_redacted() -> None:
    line = stream_module.sanitize_diagnostic_line("noise :9b9b://-?~b0%:0b-0@ tail")
    assert "0b-0" not in line
    assert "<redacted>" in line


def test_no_sanitizer_pass_is_quadratic_on_a_dense_window() -> None:
    dense = "a" * stream_module.SANITIZE_INPUT_CAP
    started = time.perf_counter()
    for _ in range(50):
        stream_module.sanitize_diagnostic_line(dense)
    per_call = (time.perf_counter() - started) / 50
    assert per_call < 0.01


def test_sanitizer_cost_is_flat_in_the_input_size() -> None:
    def cost(raw: str) -> float:
        started = time.perf_counter()
        for _ in range(20):
            stream_module.sanitize_diagnostic_line(raw)
        return (time.perf_counter() - started) / 20

    small = cost("a" * stream_module.SANITIZE_INPUT_CAP)
    large = cost("a" * (stream_module.SANITIZE_INPUT_CAP * 64))
    assert large < max(small * 4.0, 0.01)


def test_a_url_port_is_preserved_but_the_path_is_redacted() -> None:
    line = stream_module.sanitize_diagnostic_line("connecting to tcp://127.0.0.1:8080 now")
    assert line == "connecting to tcp://127.0.0.1:8080 now"
    line = stream_module.sanitize_diagnostic_line(
        "connecting to tcp://127.0.0.1:8080/hls/index.m3u8"
    )
    assert line == "connecting to tcp://127.0.0.1:8080/<redacted>"


def test_a_bare_url_without_a_path_or_query_is_preserved() -> None:
    assert stream_module.sanitize_diagnostic_line("resolving https://cctv.test") == (
        "resolving https://cctv.test"
    )


def test_underscored_secret_names_are_redacted() -> None:
    probes = {
        "auth_key=abc123": "auth_key",
        "hdnts=exp=1": "hdnts",
        "jwt=aaa.bbb.ccc": "jwt",
        "sig=deadbeef": "sig",
        "pass=hunter2": "pass",
        "policy=allow-all": "policy",
        "credential=zzz": "credential",
        "keypairid=APKA123": "keypairid",
        "my_api_key=letmein": "api_key",
    }
    for raw, name in probes.items():
        line = stream_module.sanitize_diagnostic_line(f"request failed with {raw} in header")
        assert name in line, raw
        assert "<redacted>" in line, raw
        assert "header" not in line, raw


def test_a_standalone_underscored_secret_name_is_redacted() -> None:
    line = stream_module.sanitize_diagnostic_line("my_api_key=letmein")
    assert line == "my_api_key=<redacted>"


def test_ordinary_words_ending_in_secret_names_are_not_redacted() -> None:
    for raw in ("monkey=banana", "compass=north", "signature_method=ed25519", "keys=5"):
        assert stream_module.sanitize_diagnostic_line(raw) == raw


def test_a_bearer_token_on_its_own_line_is_redacted() -> None:
    line = stream_module.sanitize_diagnostic_line("Bearer abcdef0123456789abcdef")
    assert "abcdef0123456789abcdef" not in line
    assert line == "Bearer <redacted>"


def test_an_aws_style_access_key_id_is_redacted() -> None:
    line = stream_module.sanitize_diagnostic_line("using AKIAIOSFODNN7EXAMPLE for the request")
    assert "AKIAIOSFODNN7EXAMPLE" not in line
    assert line == "using <redacted> for the request"


def test_a_presigned_aws_key_pair_id_and_signature_are_redacted() -> None:
    line = stream_module.sanitize_diagnostic_line(
        "https://d111.cloudfront.net/a.m3u8?Expires=1&Signature=abcdef0123456789&Key-Pair-Id=APKA9"
    )
    assert "abcdef0123456789" not in line
    assert "APKA9" not in line


def test_a_json_web_token_is_redacted_even_without_a_name() -> None:
    token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r"
    line = stream_module.sanitize_diagnostic_line(f"server rejected {token} at handshake")
    assert token not in line
    assert "handshake" in line
    assert "<redacted>" in line


def test_disconnect_callback_runs_before_the_retry_sleeper() -> None:
    events: list[str] = []
    environment = FakeEnvironment((b"",), events=events)
    source = environment.build(on_disconnect=lambda message: events.append("disconnect"))
    environment.drain(source, 1)
    assert events == ["disconnect", "sleep"]


def test_disconnect_callback_runs_before_the_replacement_frame() -> None:
    events: list[str] = []
    environment = FakeEnvironment((payload(1), payload(2)))
    source = environment.build(on_disconnect=lambda message: events.append("disconnect"))
    frames: list[Frame] = []
    for frame in source.frames():
        events.append("frame")
        frames.append(frame)
        if len(frames) == 2:
            source.stop()
    assert [int(entry.image[0, 0, 0]) for entry in frames] == [1, 2]
    assert events == ["frame", "disconnect", "frame"]


def test_disconnect_callback_receives_the_sanitized_cause() -> None:
    causes: list[str] = []
    environment = FakeEnvironment((payload(1) + b"\x07" * 3,))
    source = environment.build(on_disconnect=causes.append)
    environment.drain(source, 1)
    assert len(causes) == 1
    assert stream_module._PARTIAL_FRAME in causes[0]
    assert "3 of" in causes[0]


def test_disconnect_callback_fires_for_a_spawn_failure() -> None:
    causes: list[str] = []
    environment = FakeEnvironment((b"",), spawn_errors=(OSError("boom"),))
    source = environment.build(on_disconnect=causes.append)
    environment.drain(source, 1)
    assert causes == [stream_module._SPAWN_FAILED]


def test_disconnect_callback_does_not_fire_on_a_clean_stop() -> None:
    causes: list[str] = []
    environment = FakeEnvironment((payload(1),))
    source = environment.build(on_disconnect=causes.append)
    environment.collect(source, 1)
    assert causes == []


def test_disconnect_callback_does_not_fire_before_a_frame_is_ever_read() -> None:
    causes: list[str] = []
    environment = FakeEnvironment((payload(1), payload(2), payload(3)))
    source = environment.build(on_disconnect=causes.append)
    environment.collect(source, 1)
    assert causes == []


def test_disconnect_callback_exceptions_do_not_stop_recovery() -> None:
    causes: list[str] = []

    def failing(message: str) -> None:
        causes.append(message)
        raise RuntimeError("listener failed")

    environment = FakeEnvironment((b"", payload(1)), spawn_errors=(OSError("cannot start"),))
    source = environment.build(on_disconnect=failing)
    frames = environment.collect(source, 1)
    assert causes == [stream_module._SPAWN_FAILED]
    assert [int(frame.image[0, 0, 0]) for frame in frames] == [1]
    assert source.diagnostics().attempts == 2
    assert environment.sleeps == [1.0]


def test_disconnect_callback_defaults_to_a_no_op() -> None:
    environment = FakeEnvironment((b"",))
    source = environment.build()
    environment.drain(source, 1)
    assert source.diagnostics().last_error == stream_module._ENDED_BEFORE_FRAME


def test_stderr_capture_failure_is_appended_to_the_known_cause() -> None:
    environment = FakeEnvironment((b"",), captures=(FakeCapture(),))
    environment.captures[0].fails = True
    source = environment.build()
    environment.drain(source, 1)
    diagnostics = source.diagnostics()
    assert diagnostics.last_error is not None
    cause = diagnostics.last_error.index(stream_module._ENDED_BEFORE_FRAME)
    capture = diagnostics.last_error.index(stream_module._STDERR_CAPTURE_FAILED)
    assert cause >= 0
    assert capture > cause
    assert diagnostics.stderr_tail == ()


def test_stderr_capture_failure_alone_is_reported_when_no_cause_is_known() -> None:
    environment = FakeEnvironment((payload(1),), captures=(FakeCapture(),))
    environment.captures[0].fails = True
    source = environment.build()
    environment.collect(source, 1)
    assert source.diagnostics().last_error == stream_module._STDERR_CAPTURE_FAILED


def test_a_successful_capture_keeps_the_disconnect_cause_intact() -> None:
    environment = FakeEnvironment((b"",), stderr_texts=("connection refused",))
    source = environment.build()
    environment.drain(source, 1)
    diagnostics = source.diagnostics()
    assert diagnostics.last_error == stream_module._ENDED_BEFORE_FRAME
    assert diagnostics.stderr_tail == ("connection refused",)


def test_backoff_doubles_from_one_second_to_the_configured_maximum() -> None:
    environment = FakeEnvironment((b"",), jitter=0.5)
    source = environment.build()
    environment.drain(source, 9)
    assert environment.sleeps == list(NOMINAL_BACKOFF)
    assert source.diagnostics().attempts == 9


def test_backoff_jitter_stays_within_the_configured_bounds() -> None:
    for sample in (0.0, 0.25, 1.0, 5.0):
        environment = FakeEnvironment((b"",), jitter=sample)
        source = environment.build()
        environment.drain(source, 7)
        assert environment.sleeps, "a failing source must schedule retries"
        for actual, nominal in zip(environment.sleeps, NOMINAL_BACKOFF[:7], strict=True):
            assert 0.0 < actual <= 30.0
            assert abs(actual - nominal) <= nominal * 0.25


def test_backoff_jitter_spreads_delays_around_the_nominal_value() -> None:
    low = FakeEnvironment((b"",), jitter=0.0)
    low_source = low.build()
    low.drain(low_source, 3)
    high = FakeEnvironment((b"",), jitter=1.0)
    high_source = high.build()
    high.drain(high_source, 3)
    assert low.sleeps == [0.75, 1.5, 3.0]
    assert high.sleeps == [1.25, 2.5, 5.0]


def test_backoff_jitter_never_exceeds_the_configured_maximum() -> None:
    environment = FakeEnvironment((b"",), jitter=1.0)
    source = environment.build()
    environment.drain(source, 7)
    assert max(environment.sleeps) == 30.0
    assert environment.sleeps[-1] == 30.0


def test_backoff_honours_a_reduced_maximum() -> None:
    environment = FakeEnvironment((b"",), jitter=0.5)
    source = environment.build(
        timing=TimingConfig(reconnect_initial_seconds=0.5, reconnect_max_seconds=2.0)
    )
    environment.drain(source, 5)
    assert environment.sleeps == [0.5, 1.0, 2.0, 2.0, 2.0]


def test_backoff_resets_after_a_delivered_frame() -> None:
    environment = FakeEnvironment((payload(1), payload(2)))
    source = environment.build()
    frames = environment.drain(source, 2)
    assert [int(frame.image[0, 0, 0]) for frame in frames] == [1, 2]
    assert environment.sleeps == [1.0, 1.0]
    assert len(environment.commands) == 2


def test_states_report_connecting_connected_and_stopped() -> None:
    environment = FakeEnvironment((payload(1), b""))
    source = environment.build()
    iterator = source.frames()
    next(iterator)
    assert environment.states_at_start[0] is StreamState.CONNECTING
    assert source.diagnostics().state is StreamState.CONNECTED
    source.stop()
    assert source.diagnostics().state is StreamState.STOPPED
    list(iterator)


def test_a_fresh_source_starts_disconnected() -> None:
    environment = FakeEnvironment((b"",))
    source = environment.build()
    diagnostics = source.diagnostics()
    assert diagnostics.state is StreamState.DISCONNECTED
    assert diagnostics.attempts == 0
    assert diagnostics.last_error is None
    assert diagnostics.stderr_tail == ()
    assert diagnostics.replaced_frames == 0


def test_stop_closes_stdin_terminates_and_waits_five_seconds() -> None:
    environment = FakeEnvironment((payload(1),))
    source = environment.build()
    environment.collect(source, 1)
    process = environment.processes[0]
    source.stop()
    assert process.stdin.is_closed
    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert process.wait_timeouts == [5.0]


def test_stop_kills_a_process_that_ignores_termination() -> None:
    environment = FakeEnvironment((payload(1),), ignore_terminate=True)
    source = environment.build()
    environment.collect(source, 1)
    process = environment.processes[0]
    source.stop()
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.wait_timeouts == [5.0, 5.0]


def test_stop_releases_the_process_exactly_once() -> None:
    environment = FakeEnvironment((payload(1),))
    source = environment.build()
    environment.collect(source, 1)
    source.stop()
    source.stop()
    process = environment.processes[0]
    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert process.wait_timeouts == [5.0]


def test_stop_is_safe_before_any_process_starts() -> None:
    environment = FakeEnvironment((payload(1),))
    source = environment.build()
    source.stop()
    source.stop()
    assert source.diagnostics().state is StreamState.STOPPED
    assert list(source.frames()) == []
    assert environment.commands == []


def test_stop_interrupts_a_pending_backoff_wait() -> None:
    environment = FakeEnvironment((b"",))
    source = environment.build(
        timing=TimingConfig(reconnect_initial_seconds=5.0, reconnect_max_seconds=30.0),
        default_sleep=True,
    )
    started = time.monotonic()
    timer = threading.Timer(0.05, source.stop)
    timer.start()
    try:
        assert list(source.frames()) == []
    finally:
        timer.cancel()
    assert time.monotonic() - started < 2.0
    assert source.diagnostics().state is StreamState.STOPPED


def test_frames_are_produced_strictly_sequentially() -> None:
    environment = FakeEnvironment((b"".join(payload(marker) for marker in (1, 2, 3, 4, 5)),))
    source = environment.build(clock=StepClock(step=timedelta(milliseconds=100)))
    slot = LatestFrameSlot()
    frames = environment.collect(source, 5)
    assert len(environment.commands) == 1
    assert [int(frame.image[0, 0, 0]) for frame in frames] == [1, 2, 3, 4, 5]
    assert [frame.observed_at for frame in frames] == sorted(frame.observed_at for frame in frames)
    for frame in frames:
        slot.put(frame)
        taken = slot.take()
        assert taken is frame
    assert slot.stats() == FrameSlotStats(received=5, replaced=0)
    assert slot.take() is None
    assert environment.processes[0].stdout.requests.count(FRAME_BYTES) == 5


def test_a_second_frame_iterator_is_rejected_while_one_runs() -> None:
    environment = FakeEnvironment((payload(1),))
    source = environment.build()
    iterator = source.frames()
    next(iterator)
    second = source.frames()
    with pytest.raises(RuntimeError, match="already running"):
        next(second)
    source.stop()
    list(iterator)


def test_an_unstarted_iterator_never_claims_the_source() -> None:
    environment = FakeEnvironment((payload(1),))
    source = environment.build()
    unused = source.frames()
    assert unused is not None
    assert len(environment.collect(source, 1)) == 1
    unused.close()
    assert environment.processes[0].terminate_calls == 1


def test_an_abandoned_unstarted_iterator_does_not_wedge_the_source() -> None:
    environment = FakeEnvironment((payload(1),))
    source = environment.build()
    for _ in range(3):
        source.frames().close()
    assert len(environment.collect(source, 1)) == 1
    assert source.diagnostics().attempts == 1


def test_keyboard_interrupt_is_not_swallowed_by_the_generator() -> None:
    environment = FakeEnvironment((payload(1),), interrupt=True)
    source = environment.build()
    iterator = source.frames()
    with pytest.raises(KeyboardInterrupt):
        next(iterator)
    process = environment.processes[0]
    assert process.terminate_calls == 1
    assert process.stdin.is_closed
    assert source.diagnostics().state is StreamState.STOPPED


def test_abandoning_a_frame_iterator_releases_the_process_and_stops() -> None:
    environment = FakeEnvironment((b"".join(payload(marker) for marker in (1, 2, 3)),))
    source = environment.build()
    iterator = source.frames()
    next(iterator)
    process = environment.processes[0]
    iterator.close()
    assert process.terminate_calls == 1
    assert process.wait_timeouts == [5.0]
    assert source.diagnostics().state is StreamState.STOPPED
    assert list(source.frames()) == []


def test_a_new_frame_iterator_is_allowed_after_the_previous_one_ends() -> None:
    environment = FakeEnvironment((payload(1), payload(2)))
    source = environment.build()
    assert len(environment.collect(source, 1)) == 1
    source.stop()
    assert list(source.frames()) == []


def test_diagnostics_report_replacements_from_an_injected_slot() -> None:
    slot = LatestFrameSlot()
    environment = FakeEnvironment((payload(1),))
    source = environment.build(slot_stats=slot.stats)
    frames = environment.collect(source, 1)
    assert source.diagnostics().replaced_frames == 0
    slot.put(frames[0])
    slot.put(frames[0])
    assert source.diagnostics().replaced_frames == 1


def test_spawn_ffmpeg_starts_a_process_from_a_list_without_a_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        recorded["command"] = command
        recorded["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr("traffic_counter.stream.subprocess.Popen", fake_popen)
    spawned = stream_module.spawn_ffmpeg([EXECUTABLE, "-i", STREAM_URL])
    assert recorded["command"] == [EXECUTABLE, "-i", STREAM_URL]
    kwargs = recorded["kwargs"]
    assert isinstance(kwargs, dict)
    assert "shell" not in kwargs
    assert kwargs["stdin"] is subprocess.PIPE
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stderr"] is spawned.diagnostics
    assert kwargs["bufsize"] == 0
    if os.name == "nt":
        assert kwargs["creationflags"] == subprocess.CREATE_NO_WINDOW
    assert spawned.diagnostics is not None
    spawned.diagnostics.close()


def test_stream_disconnected_is_a_runtime_error() -> None:
    assert issubclass(StreamDisconnected, RuntimeError)
    with pytest.raises(StreamDisconnected, match="frame"):
        raise StreamDisconnected(stream_module._PARTIAL_FRAME)
