from __future__ import annotations

import contextlib
import os
import random
import re
import subprocess
import tempfile
import threading
from collections.abc import Callable, Generator
from dataclasses import dataclass
from datetime import datetime
from typing import IO, Final, Protocol

import numpy as np
from numpy.typing import NDArray

from traffic_counter.config import StreamConfig, TimingConfig
from traffic_counter.models import Frame, FrameSlotStats, StreamDiagnostics, StreamState
from traffic_counter.timeutils import WallClock

type Clock = Callable[[], datetime]
type Sleeper = Callable[[float], None]
type JitterSource = Callable[[], float]
type FrameStats = Callable[[], FrameSlotStats]
type DisconnectListener = Callable[[str], None]
type Spawner = Callable[[list[str]], "SpawnedProcess"]

CHANNELS: Final[int] = 3
STDERR_TAIL_LINES: Final[int] = 20
STDERR_WINDOW_BYTES: Final[int] = 16 * 1024
STOP_TERMINATE_TIMEOUT_SECONDS: Final[float] = 5.0
BACKOFF_JITTER_FRACTION: Final[float] = 0.25
FFMPEG_RECONNECT_DELAY_MAX_SECONDS: Final[int] = 5
FFMPEG_LOG_LEVEL: Final[str] = "warning"
USER_AGENT: Final[str] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
SANITIZE_INPUT_CAP: Final[int] = 4096
MAX_SANITIZED_LINE: Final[int] = 200
DEFAULT_TARGET_FPS: Final[int] = 10

_ENDED_BEFORE_FRAME: Final[str] = "The ffmpeg stream ended before a complete frame arrived."
_PARTIAL_FRAME: Final[str] = "The ffmpeg stream ended part way through a frame."
_PROCESS_EXITED: Final[str] = "The ffmpeg process exited before the stream ended."
_SPAWN_FAILED: Final[str] = "The ffmpeg process could not be started."
_NO_STDOUT: Final[str] = "The ffmpeg process did not expose a raw video pipe."
_STDERR_CAPTURE_FAILED: Final[str] = "The ffmpeg diagnostic output could not be read."
_OVERREAD: Final[str] = "The ffmpeg video pipe returned more than one frame of data."
_EXECUTABLE_UNAVAILABLE: Final[str] = (
    "A bundled ffmpeg executable could not be located for the stream reader."
)
_ALREADY_STREAMING: Final[str] = "A frame iterator is already running for this stream source."
_DIMENSIONS: Final[str] = "The stream width and height must be positive whole numbers."

_ANSI_ESCAPE: Final[re.Pattern[str]] = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_SECRET_VALUE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?<![A-Za-z0-9])("
    r"hdnts|api_key|apikey|auth_key|keypairid|credentials|credential|authorization"
    r"|sessionid|signature|password|passwd|secret|token|bearer|policy|auth|pass|key|jwt|sig|pwd"
    r")(?![A-Za-z0-9])([ \t]*[=:][ \t]*)[^\n]*"
)
_AWS_KEY_ID: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9])(?:AKIA|ASIA|AIDA|AROA)[A-Z0-9]{16}(?![A-Za-z0-9])"
)
_BEARER_TOKEN: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?<![A-Za-z0-9])(bearer)([ \t]+)[A-Za-z0-9._~+/=\-]{8,}"
)
_JSON_WEB_TOKEN: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}"
)
_URL: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b([a-z][a-z0-9+.\-]*://)([^/\s\"'<>|]+)([^\s\"'<>|]*)"
)
_URL_CREDENTIALS: Final[re.Pattern[str]] = re.compile(r"(?i)//[^\s:@/]+:[^\s/@]+@")
_REL_TRAVERSAL: Final[re.Pattern[str]] = re.compile(
    r"((?:^|[\s\"'\|]))(?:\.\./|\.\.\\|\./|\.\\)+[^\s\"'<>|]*"
)
_REL_BARE: Final[re.Pattern[str]] = re.compile(r"\.\.\\[^\s\"'<>|]*")
_WINDOWS_PATH: Final[re.Pattern[str]] = re.compile(r"(?i)(?<!\w)(?:[a-z]:[\\/]|\\\\)[^\s\"'<>|]*")
_POSIX_PATH: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w./])/(?!api(?:/|$))(?:[^\s\"'<>|/]+/)*[^\s\"'<>|/]+"
)
_INLINE_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"[^\S\n]+")


class StreamDisconnected(RuntimeError):
    pass


class ProcessHandle(Protocol):
    @property
    def stdin(self) -> IO[bytes] | None: ...

    @property
    def stdout(self) -> IO[bytes] | None: ...

    @property
    def returncode(self) -> int | None: ...

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


@dataclass(frozen=True, slots=True)
class SpawnedProcess:
    process: ProcessHandle
    diagnostics: IO[bytes] | None


def build_ffmpeg_command(
    executable: str,
    url: str,
    width: int,
    height: int,
    target_fps: int = DEFAULT_TARGET_FPS,
) -> list[str]:
    if not executable.strip():
        raise ValueError("The ffmpeg executable must not be blank.")
    if not url.strip():
        raise ValueError("The stream url must not be blank.")
    if width < 1 or height < 1:
        raise ValueError(_DIMENSIONS)
    if target_fps < 1:
        raise ValueError("The target frame rate must be a positive whole number.")
    return [
        executable,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        FFMPEG_LOG_LEVEL,
        "-user_agent",
        USER_AGENT,
        "-reconnect",
        "1",
        "-reconnect_at_eof",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_delay_max",
        str(FFMPEG_RECONNECT_DELAY_MAX_SECONDS),
        "-i",
        url,
        "-an",
        "-vf",
        fixed_size_filter(width, height, target_fps),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]


def fixed_size_filter(width: int, height: int, target_fps: int) -> str:
    return (
        f"fps={target_fps},"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"
    )


def resolve_ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError(_EXECUTABLE_UNAVAILABLE) from error
    try:
        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as error:
        raise RuntimeError(_EXECUTABLE_UNAVAILABLE) from error


def spawn_ffmpeg(command: list[str]) -> SpawnedProcess:
    capture = _diagnostic_capture()
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=capture,
            bufsize=0,
            creationflags=_creation_flags(),
        )
    except BaseException:
        _close_stream(capture)
        raise
    return SpawnedProcess(process=process, diagnostics=capture)


def _diagnostic_capture() -> IO[bytes]:
    return tempfile.TemporaryFile(mode="w+b")


def _creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


def sanitize_diagnostic_line(raw: str) -> str:
    text = _ANSI_ESCAPE.sub("", raw[-SANITIZE_INPUT_CAP:]).strip()
    text = _URL.sub(_redact_url, text)
    text = _AWS_KEY_ID.sub("<redacted>", text)
    text = _BEARER_TOKEN.sub(r"\1\2<redacted>", text)
    text = _JSON_WEB_TOKEN.sub("<redacted>", text)
    text = _SECRET_VALUE.sub(r"\1\2<redacted>", text)
    text = _URL_CREDENTIALS.sub("//<redacted>@", text)
    text = _REL_TRAVERSAL.sub(r"\1<path>", text)
    text = _REL_BARE.sub("<path>", text)
    text = _WINDOWS_PATH.sub("<path>", text)
    text = _POSIX_PATH.sub("<path>", text)
    text = _INLINE_WHITESPACE.sub(" ", text).strip()
    if len(text) > MAX_SANITIZED_LINE:
        text = text[: MAX_SANITIZED_LINE - 3] + "..."
    return text


def _redact_url(match: re.Match[str]) -> str:
    scheme, authority, tail = match.group(1), match.group(2), match.group(3)
    host = authority.rpartition("@")[2] or authority
    cut = min(
        (index for index in (host.find("?"), host.find("#")) if index != -1),
        default=-1,
    )
    if cut == -1:
        return f"{scheme}{host}" if not tail else f"{scheme}{host}/<redacted>"
    return f"{scheme}{host[:cut]}/<redacted>"


def _sanitized_diagnostic_lines(text: str) -> tuple[str, ...]:
    cleaned = (sanitize_diagnostic_line(raw) for raw in text.splitlines())
    return tuple(line for line in cleaned if line)[-STDERR_TAIL_LINES:]


def _stderr_tail(stream: IO[bytes]) -> tuple[str, ...]:
    try:
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        start = max(0, end - STDERR_WINDOW_BYTES)
        stream.seek(start)
        window = stream.read(STDERR_WINDOW_BYTES)
    except (OSError, ValueError) as error:
        raise StreamDisconnected(_STDERR_CAPTURE_FAILED) from error
    text = window.decode("utf-8", errors="replace")
    if start == 0:
        return _sanitized_diagnostic_lines(text)
    _, separator, remainder = text.partition("\n")
    if not separator:
        return ()
    return _sanitized_diagnostic_lines(remainder)


def _close_stream(stream: IO[bytes] | None) -> None:
    if stream is None:
        return
    with contextlib.suppress(OSError, ValueError):
        stream.close()


def _close_stdin(handle: ProcessHandle) -> None:
    stream = handle.stdin
    if stream is None:
        return
    with contextlib.suppress(OSError, ValueError):
        stream.close()


def _close_stdout(handle: ProcessHandle) -> None:
    stream = handle.stdout
    if stream is None:
        return
    with contextlib.suppress(OSError, ValueError):
        stream.close()


def _terminate(handle: ProcessHandle) -> None:
    with contextlib.suppress(OSError, ValueError):
        handle.terminate()
    try:
        handle.wait(timeout=STOP_TERMINATE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(OSError, ValueError):
            handle.kill()
        with contextlib.suppress(subprocess.TimeoutExpired, OSError, ValueError):
            handle.wait(timeout=STOP_TERMINATE_TIMEOUT_SECONDS)
    except (OSError, ValueError):
        return


def _jittered_delay(base: float, sample: float) -> float:
    factor = min(max(sample, 0.0), 1.0)
    return base + base * BACKOFF_JITTER_FRACTION * (2.0 * factor - 1.0)


def _no_replacements() -> FrameSlotStats:
    return FrameSlotStats(received=0, replaced=0)


def _ignore_disconnect(message: str) -> None:
    return


def _append_cause(existing: str | None, addition: str) -> str:
    return addition if existing is None else f"{existing} {addition}"


class HlsFrameSource:
    __slots__ = (
        "_attempts",
        "_clock",
        "_config",
        "_delay",
        "_diagnostics",
        "_executable",
        "_frame_bytes",
        "_height",
        "_jitter",
        "_last_error",
        "_lock",
        "_on_disconnect",
        "_process",
        "_sleep",
        "_slot_stats",
        "_spawn",
        "_state",
        "_stderr_lines",
        "_stopped",
        "_streaming",
        "_timing",
        "_wake",
        "_width",
    )

    def __init__(
        self,
        config: StreamConfig,
        timing: TimingConfig,
        clock: Clock | None = None,
        executable: str | None = None,
        *,
        spawn: Spawner | None = None,
        sleep: Sleeper | None = None,
        jitter: JitterSource | None = None,
        slot_stats: FrameStats | None = None,
        on_disconnect: DisconnectListener | None = None,
    ) -> None:
        if config.width < 1 or config.height < 1:
            raise ValueError(_DIMENSIONS)
        self._config = config
        self._timing = timing
        self._clock: Clock = WallClock().now if clock is None else clock
        self._executable = resolve_ffmpeg_executable() if executable is None else executable
        self._spawn: Spawner = spawn_ffmpeg if spawn is None else spawn
        self._wake = threading.Event()
        self._sleep: Sleeper
        if sleep is None:
            self._sleep = self._wait_for_backoff
        else:
            self._sleep = sleep
        self._jitter: JitterSource = random.random if jitter is None else jitter
        self._slot_stats: FrameStats = _no_replacements if slot_stats is None else slot_stats
        self._on_disconnect: DisconnectListener = (
            _ignore_disconnect if on_disconnect is None else on_disconnect
        )
        self._width = config.width
        self._height = config.height
        self._frame_bytes = config.width * config.height * CHANNELS
        self._delay = float(timing.reconnect_initial_seconds)
        self._attempts = 0
        self._state = StreamState.DISCONNECTED
        self._last_error: str | None = None
        self._stderr_lines: tuple[str, ...] = ()
        self._lock = threading.Lock()
        self._process: ProcessHandle | None = None
        self._diagnostics: IO[bytes] | None = None
        self._stopped = False
        self._streaming = False

    def frames(self) -> Generator[Frame]:
        return self._frame_iterator()

    def _wait_for_backoff(self, seconds: float) -> None:
        self._wake.wait(seconds)

    def stop(self) -> None:
        self._stopped = True
        self._wake.set()
        self._release()
        with self._lock:
            self._state = StreamState.STOPPED

    def diagnostics(self) -> StreamDiagnostics:
        stats = self._slot_stats()
        with self._lock:
            return StreamDiagnostics(
                state=self._state,
                attempts=self._attempts,
                replaced_frames=stats.replaced,
                last_error=self._last_error,
                stderr_tail=self._stderr_lines,
            )

    def _frame_iterator(self) -> Generator[Frame]:
        with self._lock:
            if self._streaming:
                raise RuntimeError(_ALREADY_STREAMING)
            self._streaming = True
        try:
            while not self._stopped:
                handle = self._open()
                if handle is not None:
                    delivered = False
                    try:
                        while not self._stopped:
                            image = self._read_frame(handle)
                            if not delivered:
                                delivered = True
                                self._reset_delay()
                            yield Frame(image=image, observed_at=self._clock())
                    except StreamDisconnected as error:
                        if not self._stopped:
                            self._degrade(str(error))
                    finally:
                        self._release()
                if self._stopped:
                    return
                self._notify_disconnect()
                if not self._await_retry():
                    return
        finally:
            self._stopped = True
            self._wake.set()
            with self._lock:
                self._streaming = False
                self._state = StreamState.STOPPED

    def _open(self) -> ProcessHandle | None:
        with self._lock:
            if self._stopped:
                return None
            self._state = StreamState.CONNECTING
            self._attempts += 1
        try:
            spawned = self._spawn(self._command())
        except OSError:
            self._degrade(_SPAWN_FAILED)
            return None
        handle = spawned.process
        if handle.stdout is None:
            self._degrade(_NO_STDOUT)
            _close_stream(spawned.diagnostics)
            _terminate(handle)
            return None
        with self._lock:
            if not self._stopped:
                self._process = handle
                self._diagnostics = spawned.diagnostics
                self._state = StreamState.CONNECTED
                return handle
            self._state = StreamState.STOPPED
        _close_stdin(handle)
        _close_stdout(handle)
        _terminate(handle)
        _close_stream(spawned.diagnostics)
        return None

    def _command(self) -> list[str]:
        return build_ffmpeg_command(
            self._executable,
            self._config.url,
            self._config.width,
            self._config.height,
            self._timing.target_fps,
        )

    def _read_frame(self, handle: ProcessHandle) -> NDArray[np.uint8]:
        stdout = handle.stdout
        if stdout is None:
            raise StreamDisconnected(_NO_STDOUT)
        payload = self._read_exact(handle, stdout)
        return np.frombuffer(payload, dtype=np.uint8).reshape(self._height, self._width, CHANNELS)

    def _read_exact(self, handle: ProcessHandle, stdout: IO[bytes]) -> bytes:
        parts: list[bytes] = []
        remaining = self._frame_bytes
        while remaining > 0:
            chunk = stdout.read(remaining)
            if not chunk:
                raise StreamDisconnected(self._read_failure(handle, self._frame_bytes - remaining))
            if len(chunk) > remaining:
                raise StreamDisconnected(_OVERREAD)
            parts.append(chunk)
            remaining -= len(chunk)
        return parts[0] if len(parts) == 1 else b"".join(parts)

    def _read_failure(self, handle: ProcessHandle, received: int) -> str:
        code = handle.poll()
        if code is not None:
            return f"{_PROCESS_EXITED} Exit code {code}."
        if received > 0:
            return f"{_PARTIAL_FRAME} Received {received} of {self._frame_bytes} bytes."
        return _ENDED_BEFORE_FRAME

    def _release(self) -> None:
        with self._lock:
            handle = self._process
            stream = self._diagnostics
            self._process = None
            self._diagnostics = None
        if handle is not None:
            _close_stdin(handle)
            _close_stdout(handle)
            _terminate(handle)
        if stream is not None:
            self._capture(stream)

    def _capture(self, stream: IO[bytes]) -> None:
        try:
            lines = _stderr_tail(stream)
        except StreamDisconnected as error:
            with self._lock:
                self._last_error = _append_cause(self._last_error, str(error))
        else:
            with self._lock:
                self._stderr_lines = lines
        finally:
            _close_stream(stream)

    def _reset_delay(self) -> None:
        with self._lock:
            self._delay = float(self._timing.reconnect_initial_seconds)

    def _degrade(self, message: str) -> None:
        with self._lock:
            self._last_error = message
            self._state = StreamState.DISCONNECTED

    def _notify_disconnect(self) -> None:
        with self._lock:
            message = self._last_error
        if message is None:
            return
        with contextlib.suppress(Exception):
            self._on_disconnect(message)

    def _await_retry(self) -> bool:
        if self._stopped:
            return False
        with self._lock:
            base = self._delay
            limit = float(self._timing.reconnect_max_seconds)
        self._sleep(min(max(_jittered_delay(base, self._jitter()), 0.0), limit))
        if self._stopped:
            return False
        with self._lock:
            self._delay = min(base * 2.0, limit)
        return True
