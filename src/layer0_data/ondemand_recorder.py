"""온디맨드 녹화기 + 수집 레코드 저장 (run_realtime 분해)."""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from realtime_bootstrap import *  # noqa: F401,F403

logger = logging.getLogger(__name__)


class OnDemandRecorder:
    """징조 기반 온디맨드 녹화기.

    트리거 발화 시점에 ffmpeg -c copy로 HLS 스트림 녹화를 시작한다.
    Ring Buffer 불필요: 징조 포착 = 녹화 시작이므로 "이전 영상"이 필요 없다.
    녹화 중 추가 트리거 발화 시 종료 시각을 연장한다 (최대 RECORD_DURATION_MAX_SEC).

    상태 전이:
      idle -> recording -> finalizing -> idle
    """

    def __init__(self, cctv: CCTVInfo):
        self.cctv = cctv
        self._process: subprocess.Popen | None = None
        self._state = "idle"          # idle | recording | finalizing
        self._output_path: Path | None = None
        self._event_id: str | None = None
        self._trigger_type: str | None = None
        self._started_at: float = 0.0
        self._end_at: float = 0.0     # 예정 종료 시각 (time.monotonic 기준)
        self._last_record_end: float = 0.0   # 마지막 녹화 종료 시각 (쿨다운용)
        self._lock = threading.Lock()
        self._logger = logging.getLogger(f"rec.{cctv.cctv_id[:20]}")

    @property
    def is_recording(self) -> bool:
        return self._state == "recording"

    @property
    def is_idle(self) -> bool:
        return self._state == "idle"

    def can_record(self) -> bool:
        """쿨다운 검사. 녹화 가능하면 True."""
        if self._state != "idle":
            return False
        now = time.monotonic()
        if (now - self._last_record_end) < RECORD_COOLDOWN_SEC:
            remaining = RECORD_COOLDOWN_SEC - (now - self._last_record_end)
            self._logger.debug("녹화 쿨다운 중 (%.0f초 남)", remaining)
            return False
        return True

    def start_recording(self, trigger_type: str, event_id: str) -> bool:
        """녹화 시작 (ffmpeg -c copy, 논블로킹).

        Args:
            trigger_type: 트리거 유형 (T1, T3, T4, T5 등).
            event_id: 이벤트 식별자.

        Returns:
            성공 시 True.
        """
        with self._lock:
            if self._state != "idle":
                self._logger.warning("녹화 시작 불가: 현재 상태=%s", self._state)
                return False

            if not self.cctv.stream_url:
                self._logger.error("스트림 URL 없음: %s", self.cctv.name)
                return False

            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = SAVE_DIR / f"{event_id}_{self.cctv.cctv_id}_{ts}.mp4"

            cmd = [
                "ffmpeg",
                "-loglevel", "error",
                "-reconnect", "1",
                "-reconnect_streamed", "1",
                "-reconnect_delay_max", "5",
                "-rw_timeout", "10000000",
                "-i", self.cctv.stream_url,
                "-c", "copy",
                "-movflags", "+faststart",
                "-y",
                str(output),
            ]

            try:
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                )
            except Exception as e:
                self._logger.error("ffmpeg 시작 실패: %s", e)
                return False

            self._state = "recording"
            self._output_path = output
            self._event_id = event_id
            self._trigger_type = trigger_type
            self._started_at = time.monotonic()
            self._end_at = self._started_at + RECORD_DURATION_MIN_SEC

            self._logger.info(
                "녹화 시작: %s [%s] -> %s (최소 %ds)",
                self.cctv.name, trigger_type, output.name, RECORD_DURATION_MIN_SEC,
            )
            return True

    def extend_recording(self, reason: str = "") -> bool:
        """추가 트리거로 녹화 연장 (최대 RECORD_DURATION_MAX_SEC까지).

        Returns:
            연장 성공 시 True.
        """
        with self._lock:
            if self._state != "recording":
                return False

            max_end = self._started_at + RECORD_DURATION_MAX_SEC
            if self._end_at >= max_end:
                self._logger.debug("녹화 연장 불가: 최대 시간 도달")
                return False

            new_end = min(self._end_at + RECORD_EXTEND_SEC, max_end)
            extended_by = new_end - self._end_at
            self._end_at = new_end

            remaining = new_end - time.monotonic()
            self._logger.info(
                "녹화 연장: +%.0fs (잔여 %.0fs) %s",
                extended_by, remaining, reason,
            )
            return True

    def check_and_stop(self) -> tuple[bool, Path | None]:
        """녹화 종료 시각 도달 여부 확인 + 종료.

        메인 루프에서 매 프레임마다 호출한다.
        종료 시각 미도달이면 (False, None) 반환.
        종료 완료 시 (True, output_path) 반환.
        """
        with self._lock:
            if self._state != "recording":
                return False, None

            now = time.monotonic()
            if now < self._end_at:
                return False, None

            # 녹화 종료
            return self._stop_ffmpeg()

    def force_stop(self) -> tuple[bool, Path | None]:
        """강제 종료 (cleanup 용)."""
        with self._lock:
            if self._state == "idle":
                return False, None
            return self._stop_ffmpeg()

    def _stop_ffmpeg(self) -> tuple[bool, Path | None]:
        """ffmpeg 프로세스 종료 (내부용, lock 보유 상태에서 호출).

        Returns:
            (stopped, output_path).
        """
        self._state = "finalizing"
        output = self._output_path

        if self._process:
            # 정상 종료: stdin에 'q' 전송
            try:
                self._process.stdin.write(b"q")
                self._process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
            for fd in (self._process.stdin, self._process.stderr):
                try:
                    fd.close()
                except Exception:
                    pass
            self._process = None

        elapsed = time.monotonic() - self._started_at
        self._last_record_end = time.monotonic()

        if output and output.exists() and output.stat().st_size > 0:
            size_mb = output.stat().st_size / 1e6
            self._logger.info(
                "녹화 종료: %s (%.1f MB, %.0f초)",
                output.name, size_mb, elapsed,
            )
            self._state = "idle"
            return True, output

        self._logger.warning("녹화 파일 없거나 비어있음: %s", output)
        if output and output.exists() and output.stat().st_size == 0:
            output.unlink()
        self._state = "idle"
        return True, None

    @property
    def recording_elapsed(self) -> float:
        """현재 녹화 경과 시간 (초)."""
        if self._state != "recording":
            return 0.0
        return time.monotonic() - self._started_at

    @property
    def recording_remaining(self) -> float:
        """녹화 잔여 시간 (초)."""
        if self._state != "recording":
            return 0.0
        return max(0.0, self._end_at - time.monotonic())




def save_collection_record(record: CollectionRecord):
    """수집 기록을 JSONL로 저장."""
    META_DIR.mkdir(parents=True, exist_ok=True)
    log_file = META_DIR / f"collections_{datetime.now().strftime('%Y%m%d')}.jsonl"
    data = asdict(record)
    data["collected_at"] = datetime.now().isoformat()
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
    logger.info("수집 기록 저장: %s", log_file.name)

    # 개별 JSON도 저장
    meta_file = META_DIR / f"{record.event_id}.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


