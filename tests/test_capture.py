"""Tests for FrameSource: file iteration, limits, and reconnect behavior."""

import numpy as np
import pytest

from src.perception.capture import FrameSource

cv2 = pytest.importorskip("cv2")


@pytest.fixture
def clip_path(tmp_path):
    """A 20-frame synthetic video file."""
    path = str(tmp_path / "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(20):
        frame = np.full((48, 64, 3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_reads_all_frames_from_file(clip_path):
    frames = list(FrameSource(clip_path).frames())
    assert len(frames) == 20
    assert [f.index for f in frames] == list(range(20))
    assert all(f.ts_utc.tzinfo is not None for f in frames)


def test_max_frames_limit(clip_path):
    frames = list(FrameSource(clip_path).frames(max_frames=5))
    assert len(frames) == 5


def test_reports_stream_properties(clip_path):
    src = FrameSource(clip_path)
    frames = src.frames(max_frames=1)
    next(frames)
    assert (src.width, src.height) == (64, 48)
    assert src.fps_nominal == pytest.approx(10.0)


class FlakyCapture:
    """VideoCapture stand-in: yields 3 frames, dies, then works after reopen."""

    opens = 0

    def __init__(self, url):
        FlakyCapture.opens += 1
        self.reads = 0
        self.generation = FlakyCapture.opens

    def isOpened(self):  # noqa: N802 (cv2 API name)
        return True

    def read(self):
        self.reads += 1
        if self.generation == 1 and self.reads > 3:
            return False, None
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def get(self, prop):
        return 0.0

    def release(self):
        pass


def test_reconnects_on_live_stream_dropout():
    FlakyCapture.opens = 0
    src = FrameSource(
        "https://example.invalid/live.m3u8",
        reconnect_attempts=2,
        backoff_s=0.0,
        capture_factory=FlakyCapture,
    )
    frames = list(src.frames(max_frames=6))
    assert len(frames) == 6
    assert FlakyCapture.opens == 2  # initial + one reconnect
    assert [f.index for f in frames] == list(range(6))  # index survives reconnect


class DeadCapture:
    def __init__(self, url):
        pass

    def isOpened(self):  # noqa: N802
        return False

    def release(self):
        pass


def test_gives_up_when_stream_never_opens():
    src = FrameSource(
        "https://example.invalid/dead.m3u8",
        reconnect_attempts=2,
        backoff_s=0.0,
        capture_factory=DeadCapture,
    )
    assert list(src.frames(max_frames=3)) == []
