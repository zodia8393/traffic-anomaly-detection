"""HLS/RTSP 스트림 프레임 샘플러 (ffmpeg subprocess) — run_realtime에서 분리.

cv2.VideoCapture 대비 HLS 호환성이 좋고, 네트워크 오류 시 자동 재연결.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from realtime_constants import SAMPLE_FPS

logger = logging.getLogger(__name__)


class FrameSampler:
    """HLS/RTSP 스트림에서 지정 fps로 프레임을 샘플링한다.

    내부적으로 ffmpeg subprocess를 사용하여 raw 프레임을 읽는다.
    cv2.VideoCapture 대비 HLS 호환성이 좋고, 실패 시 자동 재연결.
    """

    def __init__(self, stream_url: str, sample_fps: int = SAMPLE_FPS,
                 width: int = 640, height: int = 480):
        self.stream_url = stream_url
        self.sample_fps = sample_fps
        self.width = width
        self.height = height
        self._process: subprocess.Popen | None = None
        self._running = False
        self._frame_count = 0

    def start(self) -> bool:
        """스트림 연결 시작. 성공 시 True."""
        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_on_network_error", "1",
            "-reconnect_delay_max", "10",
            "-rw_timeout", "15000000",
            "-fflags", "+discardcorrupt",
            "-i", self.stream_url,
            "-vf", f"fps={self.sample_fps},scale={self.width}:{self.height}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-",
        ]
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self.width * self.height * 3 * 2,
            )
            self._running = True
            self._frame_count = 0
            logger.info("FrameSampler 시작: %s (%dfps, %dx%d)",
                        self.stream_url[:80], self.sample_fps, self.width, self.height)
            return True
        except Exception as e:
            logger.error("FrameSampler 시작 실패: %s", e)
            return False

    def read_frame(self) -> tuple[bool, Any]:
        """1프레임 읽기. (success, numpy_bgr_image)."""
        if not self._running or self._process is None:
            return False, None

        import numpy as np

        nbytes = self.width * self.height * 3
        raw = self._process.stdout.read(nbytes)
        if len(raw) != nbytes:
            return False, None

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            (self.height, self.width, 3)
        )
        self._frame_count += 1
        return True, frame

    def stop(self):
        """스트림 종료."""
        self._running = False
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        logger.info("FrameSampler 종료 (총 %d프레임)", self._frame_count)

    @property
    def is_running(self) -> bool:
        return self._running and self._process is not None
