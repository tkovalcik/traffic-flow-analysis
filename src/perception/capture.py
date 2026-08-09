"""Timestamped frame capture from a video stream, with reconnect handling.

FrameSource is the single entry point every consumer (recorder, live perception)
uses to read frames. Live HLS streams drop out routinely; FrameSource reconnects
with backoff and keeps the frame index and wall-clock timestamps monotonic
across reconnects.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

# Must be set before cv2 import so its bundled ffmpeg honors I/O timeouts
# (microseconds) instead of hanging forever on a dead stream.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "timeout;10000000")

import cv2  # noqa: E402


@dataclass
class CapturedFrame:
    frame: np.ndarray
    ts_utc: datetime  # wall clock at read — event time for live capture
    index: int  # monotonic across reconnects


class FrameSource:
    """Iterate timestamped frames from a stream URL or video file.

    capture_factory exists for tests: anything returning a cv2.VideoCapture-like
    object (isOpened/read/get/release) can be injected.
    """

    def __init__(
        self,
        url: str,
        reconnect_attempts: int = 5,
        backoff_s: float = 2.0,
        capture_factory: Callable[[str], object] | None = None,
    ):
        self.url = url
        self.reconnect_attempts = reconnect_attempts
        self.backoff_s = backoff_s
        self._factory = capture_factory or (lambda u: cv2.VideoCapture(u, cv2.CAP_FFMPEG))
        self._cap: object | None = None
        self.fps_nominal: float = 0.0
        self.width: int = 0
        self.height: int = 0

    def _open(self) -> bool:
        self.close()
        cap = self._factory(self.url)
        if not cap.isOpened():
            cap.release()
            return False
        self._cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and 0 < fps < 240:
            self.fps_nominal = round(fps, 2)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return True

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def frames(
        self,
        max_frames: int | None = None,
        max_seconds: float | None = None,
    ) -> Iterator[CapturedFrame]:
        """Yield frames until a limit is hit or reconnects are exhausted."""
        started = time.monotonic()
        index = 0
        attempts_left = self.reconnect_attempts
        if self._cap is None and not self._open():
            attempts_left = self._retry_open(attempts_left)
            if attempts_left is None:
                return

        while True:
            if max_frames is not None and index >= max_frames:
                break
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                break
            ok, frame = self._cap.read()
            if not ok:
                # File sources end normally; live streams get reconnected.
                if not self.url.startswith(("http://", "https://", "rtsp://")):
                    break
                attempts_left = self._retry_open(attempts_left)
                if attempts_left is None:
                    break
                continue
            yield CapturedFrame(frame=frame, ts_utc=datetime.now(UTC), index=index)
            index += 1
        self.close()

    def _retry_open(self, attempts_left: int) -> int | None:
        """Reopen with backoff; return remaining attempts or None when exhausted."""
        while attempts_left > 0:
            time.sleep(self.backoff_s)
            attempts_left -= 1
            if self._open():
                return attempts_left
        self.close()
        return None
