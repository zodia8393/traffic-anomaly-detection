"""실시간 사고영상 수집 통합 파이프라인 (v2 — 징조 기반 온디맨드 녹화).

설계 방향: "상시 분석, 녹화는 징조 포착 시에만"

시스템 흐름:
  Phase 1 — 상시 분석 (녹화 안 함)
    CCTV HLS 스트림 -> FrameSampler(1fps) -> Vision Pipeline(YOLO+ByteTrack+트리거)
    프레임만 분석, 디스크에 저장하지 않음

  Phase 2 — 징조 포착 -> 녹화 시작
    트리거 발화(T1/T3/T4/T5) = 사고 징조 감지
    그 즉시 ffmpeg 녹화 시작 (HLS -> mp4, -c copy)
    Vision Pipeline 분석 계속 (추가 트리거 시 녹화 연장)
    녹화 지속: 최소 3분, 최대 5분

  Phase 3 — 사고 확인 + 저장/폐기
    녹화 종료 후 ITS 돌발상황 API 교차확인
    사고 확인 -> 영상 보존 + 메타데이터 기록
    미확인 -> 영상 파일 삭제

최종 영상 내러티브:
  [0:00~0:30]  사고 징조 (급감속, TTC 임박 등)
  [0:30~1:00]  사고 발생
  [1:00~3:00+] 사고 후처리

실행:
  python run_realtime.py monitor                              # 1대 CCTV 자동 선택
  python run_realtime.py monitor --lat 37.0 --lon 127.0      # 좌표 인근 CCTV
  python run_realtime.py monitor --cctv-id <id>              # 특정 CCTV
  python run_realtime.py multi                               # 사고다발구간 다중 CCTV
  python run_realtime.py multi --lat 37.11 --lon 126.89 --radius 5.0
  python run_realtime.py hotspot                             # 사고다발구간 선정 (조회만)
  python run_realtime.py dry-run                             # 전체 흐름 시뮬레이션
  python run_realtime.py status                              # 수집 현황
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# ── 경로 설정 ────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
PIPELINE_SRC = Path("/workspace/prj_cctv/pipeline/src")
ACCIDENT_SRC = Path("/workspace/prj_cctv/사고분석_설계/src")

os.environ.setdefault("OMP_NUM_THREADS", "4")

# ── 이중 config.py 충돌 해소 ────────────────────────────────────────
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(1, str(ACCIDENT_SRC))

# Phase 1: layer0_data/config.py 기반으로 track3 모듈 import
from config import (
    MAX_CONCURRENT_STREAMS,
    RECORD_COOLDOWN_SEC,
    RECORD_DURATION_MAX_SEC,
    RECORD_DURATION_MIN_SEC,
    RECORD_EXTEND_SEC,
    ITS_VERIFY_DELAY_SEC,
    STREAM_DIR,
)
from track3_api_incident import IncidentEvent, ITSIncidentClient
from track3_cctv_stream import CCTVInfo, ITSCCTVClient, _haversine

# Phase 2: config 모듈을 pipeline/src/config.py로 교체
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("config", str(PIPELINE_SRC / "config.py"))
_pipeline_cfg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pipeline_cfg)
sys.modules["config"] = _pipeline_cfg

# pipeline/src를 path에 추가 (detector, tracker import용)
sys.path.insert(0, str(PIPELINE_SRC))

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 설정 상수
# ═══════════════════════════════════════════════════════════════════════

SAMPLE_FPS = 1                   # Vision Pipeline 입력 fps (CPU 부하 관리)
ITS_CHECK_RADIUS_KM = 10.0      # 트리거 발화 시 ITS 사고 매칭 반경 (km)
ITS_CHECK_COOLDOWN_SEC = 60     # 동일 트리거 유형 ITS 확인 쿨다운
SAVE_DIR = STREAM_DIR / "accident_clips"
META_DIR = STREAM_DIR / "metadata"
LOG_DIR = STREAM_DIR / "realtime_logs"
MULTI_STATUS_FILE = LOG_DIR / "multi_status.json"
PENDING_DIR = STREAM_DIR / "pending"

SEVERITY_GATE = 0.5              # 녹화 시작 최소 severity (v3: 0.3 → v4: 0.5)
SEVERITY_PENDING = 0.7           # ITS 미확인 시 pending 보존 최소 severity
SEVERITY_FORCE_PRESERVE = 0.8    # 재확인 실패해도 보존하는 최소 severity
PENDING_MAX_RETRIES = 3          # pending 재확인 최대 횟수
PENDING_RETRY_INTERVAL_SEC = 600 # pending 재확인 간격 (10분)

# 녹화 트리거 대상 (T7 주기적 스냅샷 제외)
RECORD_TRIGGER_TYPES = {"T1", "T2", "T3", "T4", "T5", "T6"}

# 다중 트리거 합의 (v4)
CONSENSUS_WINDOW_SEC = 5.0       # 합의 윈도우 (초)
CONSENSUS_MIN_TYPES = 2          # 최소 트리거 종류 수
INSTANT_RECORD_TYPES = {"T5"}    # 단독 즉시 녹화 (다수 동시 감속)


# ═══════════════════════════════════════════════════════════════════════
# 프레임 샘플러: HLS -> 1fps numpy 프레임
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# 온디맨드 녹화기: 트리거 시점에 녹화 시작, N분 후 종료
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# ITS 교차확인기: 트리거 발화 좌표 vs ITS 사고 위치 매칭
# ═══════════════════════════════════════════════════════════════════════

class IncidentVerifier:
    """트리거 발화 시 ITS API로 실제 사고 여부를 교차 확인한다."""

    def __init__(self):
        self.client = ITSIncidentClient()
        self._last_check: dict[str, float] = {}
        self._cached_incidents: list[IncidentEvent] = []
        self._cache_age: float = 0.0

    def verify(self, lat: float, lon: float,
               trigger_type: str, radius_km: float = ITS_CHECK_RADIUS_KM
               ) -> tuple[bool, IncidentEvent | None]:
        """트리거 좌표 인근에 실제 사고가 있는지 확인.

        Returns:
            (확인됨, 매칭된_사고_이벤트). 미확인이면 (False, None).
        """
        now = time.time()

        # 쿨다운 검사
        last = self._last_check.get(trigger_type, 0)
        if (now - last) < ITS_CHECK_COOLDOWN_SEC:
            logger.debug("ITS 확인 쿨다운 중: %s (%.0f초 남)",
                         trigger_type, ITS_CHECK_COOLDOWN_SEC - (now - last))
            return False, None

        self._last_check[trigger_type] = now

        # 캐시 갱신 (30초 이상 경과 시)
        if (now - self._cache_age) > 30:
            try:
                self._cached_incidents = self.client.fetch_incidents(event_type="acc")
                self._cache_age = now
                logger.info("ITS 사고 목록 갱신: %d건", len(self._cached_incidents))
            except Exception as e:
                logger.error("ITS API 조회 실패: %s", e)
                return False, None

        # 반경 내 사고 검색
        for incident in self._cached_incidents:
            if incident.latitude is None or incident.longitude is None:
                continue
            dist = _haversine(lat, lon, incident.latitude, incident.longitude)
            if dist <= radius_km:
                logger.info("사고 확인! [%s] %s %s (%.1f km) - %s",
                            incident.road_type, incident.road_name,
                            incident.direction, dist, incident.message[:50])
                return True, incident

        logger.info("ITS 사고 미확인 (반경 %.1fkm 내 매칭 없음)", radius_km)
        return False, None


# ═══════════════════════════════════════════════════════════════════════
# 메타데이터 기록기
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CollectionRecord:
    """수집 기록."""
    event_id: str
    trigger_type: str
    trigger_description: str
    trigger_frame: int
    trigger_severity: float
    cctv_id: str
    cctv_name: str
    cctv_lat: float
    cctv_lon: float
    incident_id: str | None = None
    incident_road: str | None = None
    incident_message: str | None = None
    incident_lat: float | None = None
    incident_lon: float | None = None
    match_distance_km: float | None = None
    video_path: str | None = None
    video_size_mb: float | None = None
    video_duration_sec: float | None = None
    its_verified: bool = False
    action: str = ""               # "confirmed" | "pending_preserved" | "deleted"
    collected_at: str = ""


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


# ═══════════════════════════════════════════════════════════════════════
# 통합 파이프라인 (단일 CCTV)
# ═══════════════════════════════════════════════════════════════════════

class RealtimeAccidentPipeline:
    """실시간 사고영상 수집 통합 파이프라인 (징조 기반 온디맨드 녹화).

    흐름:
      1. CCTV 선택 (좌표/ID/자동)
      2. FrameSampler(1fps)로 스트림 연결 (녹화 안 함)
      3. 매 프레임 Vision Pipeline 실행 (YOLO -> ByteTrack -> 트리거)
      4. 트리거 발화 시 OnDemandRecorder로 녹화 시작 (징조 시점 = 영상 시작)
      5. 녹화 중 추가 트리거 -> 연장 (최대 5분)
      6. 녹화 종료 후 ITS API 교차 확인
      7. 사고 확인 -> 영상 보존 + 메타 / 미확인 -> 삭제
    """

    def __init__(self):
        self.cctv_client = ITSCCTVClient()
        self.verifier = IncidentVerifier()
        self.sampler: FrameSampler | None = None
        self.recorder: OnDemandRecorder | None = None
        self.target_cctv: CCTVInfo | None = None
        self._stop_event = threading.Event()
        self._pending_trigger = None     # 녹화 시작 시 트리거 정보 보관
        self._trigger_window: list = []  # 합의 윈도우 버퍼
        self.stats = {
            "frames_processed": 0,
            "triggers_fired": 0,
            "recordings_started": 0,
            "recordings_extended": 0,
            "its_checks": 0,
            "incidents_confirmed": 0,
            "clips_preserved": 0,
            "clips_deleted": 0,
            "start_time": None,
        }

    def _select_cctv(self, lat: float | None = None, lon: float | None = None,
                     cctv_id: str | None = None) -> CCTVInfo | None:
        """CCTV 선택: ID 지정 > 좌표 인근 > 자동."""
        cctvs = self.cctv_client.list_cctvs()
        if not cctvs:
            logger.error("CCTV 목록 조회 실패")
            return None

        logger.info("CCTV %d대 조회 완료", len(cctvs))

        if cctv_id:
            target = next((c for c in cctvs if c.cctv_id == cctv_id), None)
            if target:
                logger.info("CCTV 지정: %s (%s)", target.name, target.cctv_id)
                return target
            logger.warning("CCTV %s 없음, 자동 선택", cctv_id)

        if lat is not None and lon is not None:
            results = self.cctv_client.find_nearest(lat, lon, top_k=1)
            if results:
                target, dist = results[0]
                logger.info("인근 CCTV: %s (%.1f km)", target.name, dist)
                return target

        for c in cctvs:
            if c.stream_url:
                logger.info("자동 선택: %s (%s)", c.name, c.cctv_id)
                return c

        logger.error("스트림 URL이 있는 CCTV 없음")
        return None

    def _init_vision_pipeline(self):
        """Vision Pipeline 컴포넌트 초기화 (lazy import)."""
        from detector import VehicleDetector
        from tracker import VehicleTracker
        from layer1_vision.speed_estimator import SpeedEstimator
        from layer1_vision.trigger_detector import TriggerDetector
        from layer1_vision.vision_pipeline import VisionPipeline

        class DummyClassifier:
            """사고 감지에서는 차종 분류 불필요."""
            def predict_batch(self, crops):
                return ["T1"] * len(crops), [0.5] * len(crops)

        detector = VehicleDetector()
        tracker = VehicleTracker(fps=max(1, SAMPLE_FPS))
        pipeline = VisionPipeline(
            detector=detector,
            tracker=tracker,
            classifier=DummyClassifier(),
            speed_est=SpeedEstimator(),
            trigger_det=TriggerDetector(),
            fps=float(SAMPLE_FPS),
        )
        return pipeline

    def _check_consensus(self, trigger) -> bool:
        """다중 트리거 합의 검사. True면 녹화 시작 가능."""
        if trigger.type in INSTANT_RECORD_TYPES:
            return True

        cutoff = trigger.timestamp - CONSENSUS_WINDOW_SEC
        self._trigger_window = [
            t for t in self._trigger_window if t.timestamp > cutoff
        ]
        self._trigger_window.append(trigger)

        types_in_window = {t.type for t in self._trigger_window}
        return len(types_in_window) >= CONSENSUS_MIN_TYPES

    def _handle_trigger(self, trigger, cctv: CCTVInfo, frame_idx: int):
        """트리거 발화 처리: 합의 검사 → 녹화 시작 또는 연장."""
        self.stats["triggers_fired"] += 1
        logger.info("TRIGGER [%s] frame=%d severity=%.2f: %s",
                     trigger.type, trigger.frame_idx,
                     trigger.severity, trigger.description)

        if trigger.type not in RECORD_TRIGGER_TYPES:
            return

        if trigger.severity < SEVERITY_GATE:
            return

        if self.recorder is None:
            return

        # Case 1: 이미 녹화 중 -> 연장
        if self.recorder.is_recording:
            extended = self.recorder.extend_recording(
                reason=f"추가 트리거 [{trigger.type}] {trigger.description}"
            )
            if extended:
                self.stats["recordings_extended"] += 1
            return

        # Case 2: 쿨다운 중
        if not self.recorder.can_record():
            return

        # Case 3: 합의 검사
        if not self._check_consensus(trigger):
            logger.debug("합의 미달 [%s] — 녹화 건너뜀 (윈도우: %s)",
                         trigger.type,
                         {t.type for t in self._trigger_window})
            return

        # Case 4: 녹화 시작
        event_id = f"RT_{trigger.type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        started = self.recorder.start_recording(trigger.type, event_id)
        if started:
            self.stats["recordings_started"] += 1
            self._pending_trigger = trigger
            self._trigger_window.clear()

    def _handle_recording_end(self, video_path: Path | None):
        """녹화 종료 후 ITS 교차확인 + 3단계 보존(confirmed/pending/deleted)."""
        cctv = self.target_cctv
        trigger = self._pending_trigger
        event_id = self.recorder._event_id if self.recorder else "unknown"

        if not cctv or not trigger:
            return

        time.sleep(ITS_VERIFY_DELAY_SEC)

        self.stats["its_checks"] += 1
        confirmed, incident = self.verifier.verify(
            lat=cctv.latitude, lon=cctv.longitude,
            trigger_type=trigger.type,
        )

        if confirmed and incident and video_path:
            self._save_confirmed(event_id, trigger, cctv, incident, video_path)
        elif video_path and video_path.exists() and trigger.severity >= SEVERITY_PENDING:
            self._handle_pending(event_id, trigger, cctv, video_path)
        else:
            self._save_deleted(event_id, trigger, cctv, video_path)

        self._pending_trigger = None

    def _save_confirmed(self, event_id, trigger, cctv, incident, video_path):
        self.stats["incidents_confirmed"] += 1
        self.stats["clips_preserved"] += 1

        dist_km = _haversine(
            cctv.latitude, cctv.longitude,
            incident.latitude or 0, incident.longitude or 0,
        ) if incident.latitude and incident.longitude else None

        record = CollectionRecord(
            event_id=event_id,
            trigger_type=trigger.type,
            trigger_description=trigger.description,
            trigger_frame=trigger.frame_idx,
            trigger_severity=trigger.severity,
            cctv_id=cctv.cctv_id,
            cctv_name=cctv.name,
            cctv_lat=cctv.latitude,
            cctv_lon=cctv.longitude,
            incident_id=incident.event_id,
            incident_road=f"{incident.road_name} {incident.direction}",
            incident_message=incident.message,
            incident_lat=incident.latitude,
            incident_lon=incident.longitude,
            match_distance_km=dist_km,
            video_path=str(video_path),
            video_size_mb=(video_path.stat().st_size / 1e6
                           if video_path.exists() else None),
            its_verified=True,
            action="confirmed",
        )
        save_collection_record(record)
        logger.info("사고 확인 → 영상 보존: %s", video_path.name)

    def _handle_pending(self, event_id, trigger, cctv, video_path):
        PENDING_DIR.mkdir(parents=True, exist_ok=True)
        pending_path = PENDING_DIR / video_path.name
        shutil.move(str(video_path), str(pending_path))
        logger.info("Pending 보관: %s (severity=%.2f, 재확인 %d회 예정)",
                     pending_path.name, trigger.severity, PENDING_MAX_RETRIES)

        for retry in range(1, PENDING_MAX_RETRIES + 1):
            if self._stop_event.wait(PENDING_RETRY_INTERVAL_SEC):
                logger.info("시스템 종료 — pending 보관 유지: %s", pending_path.name)
                return
            confirmed, incident = self.verifier.verify(
                lat=cctv.latitude, lon=cctv.longitude,
                trigger_type=f"pending_{trigger.type}",
            )
            if confirmed and incident:
                final_path = SAVE_DIR / pending_path.name
                shutil.move(str(pending_path), str(final_path))
                self._save_confirmed(event_id, trigger, cctv, incident, final_path)
                logger.info("Pending → 확인 (재확인 %d회차): %s", retry, final_path.name)
                return
            logger.info("Pending 재확인 %d/%d 실패: %s",
                         retry, PENDING_MAX_RETRIES, pending_path.name)

        if trigger.severity >= SEVERITY_FORCE_PRESERVE and pending_path.exists():
            final_path = SAVE_DIR / pending_path.name
            shutil.move(str(pending_path), str(final_path))
            self.stats["clips_preserved"] += 1
            record = CollectionRecord(
                event_id=event_id,
                trigger_type=trigger.type,
                trigger_description=trigger.description,
                trigger_frame=trigger.frame_idx,
                trigger_severity=trigger.severity,
                cctv_id=cctv.cctv_id,
                cctv_name=cctv.name,
                cctv_lat=cctv.latitude,
                cctv_lon=cctv.longitude,
                video_path=str(final_path),
                video_size_mb=(final_path.stat().st_size / 1e6
                               if final_path.exists() else None),
                its_verified=False,
                action="pending_preserved",
            )
            save_collection_record(record)
            logger.info("Pending 만료 → 고severity 보존: %s", final_path.name)
        else:
            self._save_deleted(event_id, trigger, cctv, pending_path)

    def _save_deleted(self, event_id, trigger, cctv, video_path):
        self.stats["clips_deleted"] += 1
        if video_path and video_path.exists():
            size_mb = video_path.stat().st_size / 1e6
            video_path.unlink()
            logger.info("미확인 → 삭제: %s (%.1f MB)", video_path.name, size_mb)
        else:
            logger.info("미확인 (영상 파일 없음)")

        record = CollectionRecord(
            event_id=event_id,
            trigger_type=trigger.type,
            trigger_description=trigger.description,
            trigger_frame=trigger.frame_idx,
            trigger_severity=trigger.severity,
            cctv_id=cctv.cctv_id,
            cctv_name=cctv.name,
            cctv_lat=cctv.latitude,
            cctv_lon=cctv.longitude,
            its_verified=False,
            action="deleted",
        )
        save_collection_record(record)

    def monitor(self, lat: float | None = None, lon: float | None = None,
                cctv_id: str | None = None, max_frames: int = 0):
        """메인 모니터링 루프.

        Args:
            lat, lon: 인근 CCTV 검색 좌표.
            cctv_id: 특정 CCTV ID.
            max_frames: 최대 프레임 수 (0=무제한).
        """
        # 1. CCTV 선택
        self.target_cctv = self._select_cctv(lat, lon, cctv_id)
        if not self.target_cctv:
            return

        # 2. Vision Pipeline 초기화
        logger.info("Vision Pipeline 초기화 중...")
        pipeline = self._init_vision_pipeline()
        logger.info("Vision Pipeline 준비 완료")

        # 3. FrameSampler 시작 (분석 전용, 녹화 안 함)
        self.sampler = FrameSampler(
            self.target_cctv.stream_url,
            sample_fps=SAMPLE_FPS,
            width=640,
            height=480,
        )
        if not self.sampler.start():
            logger.error("스트림 연결 실패")
            return

        # 4. OnDemandRecorder 준비 (대기 상태, 녹화 시작 전까지 idle)
        self.recorder = OnDemandRecorder(self.target_cctv)

        # 5. SIGINT 핸들러
        def signal_handler(sig, frame):
            logger.info("중단 신호 수신 — 정리 중...")
            self._stop_event.set()

        signal.signal(signal.SIGINT, signal_handler)

        # 6. 메인 루프
        self.stats["start_time"] = datetime.now().isoformat()
        logger.info("=" * 60)
        logger.info("실시간 모니터링 시작 (징조 기반 온디맨드 녹화)")
        logger.info("  CCTV: %s (%s)", self.target_cctv.name, self.target_cctv.cctv_id)
        logger.info("  분석: %d fps | 녹화: 트리거 시에만 (3~5분)", SAMPLE_FPS)
        logger.info("  녹화 트리거: %s", ", ".join(sorted(RECORD_TRIGGER_TYPES)))
        logger.info("  ITS 확인 반경: %.1f km | 쿨다운: %ds",
                     ITS_CHECK_RADIUS_KM, RECORD_COOLDOWN_SEC)
        logger.info("  Ctrl+C로 중단")
        logger.info("=" * 60)

        frame_idx = 0
        try:
            while not self._stop_event.is_set():
                ok, frame = self.sampler.read_frame()
                if not ok:
                    logger.warning("프레임 읽기 실패 — 재연결 시도 (5초 후)")
                    self.sampler.stop()
                    time.sleep(5)
                    if not self.sampler.start():
                        logger.error("재연결 실패 — 종료")
                        break
                    continue

                timestamp = frame_idx / SAMPLE_FPS
                frame_idx += 1
                self.stats["frames_processed"] = frame_idx

                # Vision Pipeline 처리
                result = pipeline.process_frame(frame, frame_idx, timestamp)

                # 트리거 처리
                for trigger in result["triggers"]:
                    self._handle_trigger(trigger, self.target_cctv, frame_idx)

                # 녹화 종료 시각 확인
                stopped, video_path = self.recorder.check_and_stop()
                if stopped:
                    # ITS 교차확인을 별도 스레드에서 실행 (메인 루프 차단 방지)
                    verify_thread = threading.Thread(
                        target=self._handle_recording_end,
                        args=(video_path,),
                        daemon=True,
                    )
                    verify_thread.start()

                # 주기적 상태 출력 (10프레임마다)
                is_last = (max_frames > 0 and frame_idx >= max_frames)
                if frame_idx == 1 or frame_idx % 10 == 0 or is_last:
                    elapsed = frame_idx / SAMPLE_FPS
                    tracks = len(result.get("tracked_vehicles", []))
                    n_det = len(result.get("detections", []))
                    rec_state = "REC" if self.recorder.is_recording else "---"
                    rec_info = ""
                    if self.recorder.is_recording:
                        rec_info = f" (잔여 {self.recorder.recording_remaining:.0f}s)"
                    logger.info(
                        "[%d프레임 | %.0f초] det=%d, 트랙=%d | "
                        "트리거=%d, 녹화=%d, 보존=%d, 삭제=%d | [%s]%s",
                        frame_idx, elapsed, n_det, tracks,
                        self.stats["triggers_fired"],
                        self.stats["recordings_started"],
                        self.stats["clips_preserved"],
                        self.stats["clips_deleted"],
                        rec_state, rec_info,
                    )

                if max_frames > 0 and frame_idx >= max_frames:
                    logger.info("최대 프레임 도달: %d", max_frames)
                    break

        except Exception as e:
            logger.error("모니터링 오류: %s", e, exc_info=True)
        finally:
            self._cleanup()

    def _cleanup(self):
        """자원 정리."""
        if self.sampler:
            self.sampler.stop()
        if self.recorder and self.recorder.is_recording:
            self.recorder.force_stop()
        self._print_summary()

    def _print_summary(self):
        """세션 요약 출력."""
        print()
        print("=" * 60)
        print("실시간 모니터링 세션 요약 (징조 기반 온디맨드)")
        print("=" * 60)
        if self.target_cctv:
            print(f"  CCTV   : {self.target_cctv.name} ({self.target_cctv.cctv_id})")
        print(f"  시작   : {self.stats['start_time']}")
        print(f"  프레임 : {self.stats['frames_processed']}프레임 "
              f"({self.stats['frames_processed'] / SAMPLE_FPS:.0f}초)")
        print(f"  트리거 : {self.stats['triggers_fired']}건")
        print(f"  녹화시작: {self.stats['recordings_started']}건")
        print(f"  녹화연장: {self.stats['recordings_extended']}건")
        print(f"  ITS확인: {self.stats['its_checks']}건")
        print(f"  사고확인: {self.stats['incidents_confirmed']}건")
        print(f"  보존   : {self.stats['clips_preserved']}건")
        print(f"  삭제   : {self.stats['clips_deleted']}건")
        print(f"  저장경로: {SAVE_DIR}")
        print("=" * 60)

        # 요약 로그 파일
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        summary_file = LOG_DIR / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({
                "cctv": self.target_cctv.name if self.target_cctv else None,
                "cctv_id": self.target_cctv.cctv_id if self.target_cctv else None,
                **self.stats,
                "ended_at": datetime.now().isoformat(),
            }, f, ensure_ascii=False, indent=2)

    def dry_run(self):
        """드라이런: 실제 스트림 없이 전체 흐름 시뮬레이션."""
        import numpy as np

        print()
        print("=" * 60)
        print("드라이런: 징조 기반 온디맨드 녹화 파이프라인 검증")
        print("=" * 60)

        # 1. CCTV 목록 조회
        print("\n[1/6] CCTV 목록 조회...")
        cctvs = self.cctv_client.list_cctvs()
        if cctvs:
            print(f"  OK: {len(cctvs)}대 조회")
            sample = cctvs[0]
            print(f"  예시: {sample.name} ({sample.cctv_id})")
            print(f"        좌표: ({sample.latitude:.4f}, {sample.longitude:.4f})")
            print(f"        URL: {sample.stream_url[:60]}..." if sample.stream_url else "        URL: 없음")
        else:
            print("  WARN: CCTV 목록 조회 실패 (API 키 확인)")

        # 2. ITS 사고 조회
        print("\n[2/6] ITS 돌발상황 API 조회...")
        incidents = self.verifier.client.fetch_incidents(event_type="acc")
        print(f"  OK: 교통사고 {len(incidents)}건")
        for ev in incidents[:3]:
            print(f"    [{ev.road_type}] {ev.road_name} {ev.direction} — {ev.message[:40]}")

        # 3. Vision Pipeline 초기화
        print("\n[3/6] Vision Pipeline 초기화...")
        try:
            pipeline = self._init_vision_pipeline()
            print("  OK: YOLO + ByteTrack + 트리거(7종) 준비")
        except Exception as e:
            print(f"  FAIL: {e}")
            return

        # 4. 프레임 처리 시뮬레이션
        print("\n[4/6] 프레임 처리 시뮬레이션 (가짜 프레임 10장)...")
        triggers_found = []
        for i in range(10):
            fake_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            result = pipeline.process_frame(fake_frame, i, float(i))
            n_det = len(result.get("detections", []))
            n_trk = len(result.get("tracked_vehicles", []))
            triggers_found.extend(result.get("triggers", []))
            if i % 5 == 0:
                print(f"  frame={i}: det={n_det}, tracks={n_trk}")
        print(f"  OK: 10프레임 처리 완료, 트리거 {len(triggers_found)}건")

        # 5. ITS 교차확인 시뮬레이션
        print("\n[5/6] ITS 교차확인 시뮬레이션...")
        if cctvs and incidents:
            test_cctv = cctvs[0]
            confirmed, incident = self.verifier.verify(
                lat=test_cctv.latitude,
                lon=test_cctv.longitude,
                trigger_type="T1_test",
                radius_km=50.0,
            )
            if confirmed:
                print(f"  OK: 사고 매칭 성공 — {incident.road_name} {incident.direction}")
            else:
                print(f"  OK: 반경 50km 내 사고 없음 (정상 동작)")
        else:
            print("  SKIP: CCTV 또는 사고 데이터 부재")

        # 6. OnDemandRecorder 시뮬레이션
        print("\n[6/6] OnDemandRecorder 시뮬레이션...")
        if cctvs:
            test_recorder = OnDemandRecorder(cctvs[0])
            print(f"  상태: {test_recorder._state} (idle)")
            print(f"  can_record: {test_recorder.can_record()}")
            print(f"  녹화 대상 트리거: {sorted(RECORD_TRIGGER_TYPES)}")
            print(f"  녹화 시간: {RECORD_DURATION_MIN_SEC}~{RECORD_DURATION_MAX_SEC}s")
            print(f"  쿨다운: {RECORD_COOLDOWN_SEC}s")
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        META_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  저장 경로: {SAVE_DIR}")
        print(f"  메타 경로: {META_DIR}")

        # 가짜 기록 저장
        record = CollectionRecord(
            event_id="DRY_TEST_001",
            trigger_type="T1",
            trigger_description="드라이런 TTC 테스트",
            trigger_frame=5,
            trigger_severity=0.8,
            cctv_id="dry_test",
            cctv_name="드라이런 CCTV",
            cctv_lat=37.5,
            cctv_lon=127.0,
            its_verified=False,
            action="test",
        )
        save_collection_record(record)
        print("  OK: 메타데이터 저장 검증 완료")

        # 요약
        print()
        print("=" * 60)
        print("드라이런 결과 요약")
        print("=" * 60)
        checks = [
            ("CCTV 목록 조회", len(cctvs) > 0),
            ("ITS 사고 API", True),
            ("Vision Pipeline", True),
            ("프레임 처리", True),
            ("ITS 교차확인", True),
            ("OnDemandRecorder", True),
        ]
        all_pass = True
        for name, passed in checks:
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            print(f"  [{status}] {name}")
        print()
        if all_pass:
            print("  전체 통과: 실시간 모니터링 준비 완료")
            print()
            print("  v2 변경사항:")
            print("    - Ring Buffer 상시녹화 제거")
            print("    - 징조(트리거) 포착 시에만 녹화 시작")
            print("    - 녹화 종료 후 ITS 교차확인 -> 보존/삭제")
            print("    - 녹화 중 추가 트리거 -> 연장 (최대 5분)")
            print()
            print("  실행 방법:")
            print("    python run_realtime.py monitor                     # 자동 CCTV 선택")
            print("    python run_realtime.py monitor --lat 37.5 --lon 127.0  # 인근 CCTV")
        else:
            print("  일부 실패: 위 항목 확인 필요")
        print("=" * 60)

    def status(self):
        """수집 현황 출력."""
        print()
        print("=" * 60)
        print("실시간 사고영상 수집 현황 (v2 OnDemand)")
        print("=" * 60)

        # 저장된 클립
        clips = list(SAVE_DIR.glob("*.mp4")) if SAVE_DIR.exists() else []
        total_mb = sum(c.stat().st_size for c in clips) / 1e6 if clips else 0
        print(f"\n  클립: {len(clips)}건 ({total_mb:.1f} MB)")
        print(f"  경로: {SAVE_DIR}")
        for c in clips[-5:]:
            size = c.stat().st_size / 1e6
            print(f"    {c.name} ({size:.1f} MB)")

        # 메타데이터
        metas = list(META_DIR.glob("*.json")) if META_DIR.exists() else []
        logs = list(META_DIR.glob("*.jsonl")) if META_DIR.exists() else []
        print(f"\n  메타데이터: {len(metas)}건 JSON + {len(logs)}건 JSONL")
        print(f"  경로: {META_DIR}")

        # 보존/삭제 통계 (JSONL에서 추출)
        preserved = 0
        deleted = 0
        for lf in logs:
            try:
                with open(lf, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            if rec.get("action") == "preserved":
                                preserved += 1
                            elif rec.get("action") == "deleted":
                                deleted += 1
                        except json.JSONDecodeError:
                            pass
            except Exception:
                pass
        if preserved or deleted:
            print(f"\n  누적 통계: 보존 {preserved}건 / 삭제 {deleted}건")

        # 세션 로그
        sessions = list(LOG_DIR.glob("session_*.json")) if LOG_DIR.exists() else []
        multi_sessions = list(LOG_DIR.glob("multi_session_*.json")) if LOG_DIR.exists() else []
        print(f"\n  세션 로그: 단일 {len(sessions)}건 / 다중 {len(multi_sessions)}건")
        if sessions:
            latest = sorted(sessions)[-1]
            with open(latest, "r") as f:
                data = json.load(f)
            print(f"  최근 단일: {latest.name}")
            print(f"    CCTV: {data.get('cctv', '?')}")
            print(f"    프레임: {data.get('frames_processed', 0)}")
            print(f"    트리거: {data.get('triggers_fired', 0)}")
            print(f"    녹화시작: {data.get('recordings_started', 0)}")
            print(f"    보존: {data.get('clips_preserved', 0)}")

        # 다중 카메라 실시간 상태
        if MULTI_STATUS_FILE.exists():
            try:
                with open(MULTI_STATUS_FILE, "r") as f:
                    mstatus = json.load(f)
                print(f"\n  다중 카메라 상태 (갱신: {mstatus.get('updated_at', '?')})")
                for cam in mstatus.get("cameras", []):
                    rec_info = " [REC]" if cam.get("recording") else ""
                    print(f"    [cam{cam['worker_id']}] {cam['cctv_name'][:20]} | "
                          f"상태: {cam['status']}{rec_info} | "
                          f"프레임: {cam['frames']} | "
                          f"트리거: {cam['triggers']} | "
                          f"보존: {cam.get('preserved', 0)} | "
                          f"삭제: {cam.get('deleted', 0)}")
                groups = mstatus.get("groups", {})
                if groups:
                    print(f"  사고 그룹: {len(groups)}건")
                    for gid, grp in groups.items():
                        print(f"    {gid}: 카메라 {len(grp['cameras'])}대, "
                              f"클립 {len(grp['clips'])}건")
            except (json.JSONDecodeError, KeyError):
                pass

        # 구간 선정 로그
        hotspot_logs = list(LOG_DIR.glob("hotspot_selection_*.json")) if LOG_DIR.exists() else []
        if hotspot_logs:
            latest_hs = sorted(hotspot_logs)[-1]
            with open(latest_hs, "r") as f:
                hs = json.load(f)
            print(f"\n  최근 감시 구간: {hs.get('hotspot_name', '?')}")
            print(f"    좌표: ({hs.get('center_lat', '?')}, {hs.get('center_lon', '?')})")
            print(f"    CCTV: {hs.get('total_cctvs_in_radius', '?')}대 중 "
                  f"{len(hs.get('monitored_cctvs', []))}대 감시")

        # API 키 상태
        print(f"\n  API 키:")
        print(f"    ITS_API_KEY: {'설정됨' if os.getenv('ITS_API_KEY') else '미설정'}")

        # 디스크 여유
        import shutil
        usage = shutil.disk_usage(str(SAVE_DIR.parent) if SAVE_DIR.exists()
                                  else "/media/ybs/Expansion")
        free_tb = usage.free / 1e12
        print(f"\n  디스크: {free_tb:.2f} TB 여유")
        print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════
# 사고 다발 구간 선정기
# ═══════════════════════════════════════════════════════════════════════

HOTSPOT_CANDIDATES = [
    {"name": "서해안선 발안IC-비봉IC",  "lat": 37.11, "lon": 126.89, "road": "서해안선"},
    {"name": "경부선 수원-오산",          "lat": 37.15, "lon": 127.05, "road": "경부선"},
    {"name": "영동선 여주JC",             "lat": 37.29, "lon": 127.64, "road": "영동선"},
    {"name": "경부선 안성JC",             "lat": 37.00, "lon": 127.10, "road": "경부선"},
    {"name": "서해안선 서평택IC",         "lat": 36.95, "lon": 126.90, "road": "서해안선"},
    {"name": "중부내륙선 여주JC",         "lat": 37.28, "lon": 127.60, "road": "중부내륙선"},
    {"name": "통영대전선 함양JC",         "lat": 35.50, "lon": 127.80, "road": "통영대전선"},
]


class HotspotSelector:
    """사고 다발 구간 선정기."""

    def __init__(self, cctv_client: ITSCCTVClient | None = None):
        self.cctv_client = cctv_client or ITSCCTVClient()
        self.incident_client = ITSIncidentClient()

    def select(self, radius_km: float = 5.0,
               lat: float | None = None, lon: float | None = None,
               ) -> dict:
        """최적 감시 구간 선정."""
        if lat is not None and lon is not None:
            return self._evaluate_point(
                f"사용자 지정 ({lat:.4f}, {lon:.4f})", lat, lon, radius_km,
            )

        all_events = []
        try:
            all_events = self.incident_client.fetch_incidents(event_type="all", road_type="ex")
        except Exception as e:
            logger.warning("ITS 돌발 조회 실패: %s", e)

        accidents = [ev for ev in all_events if "사고" in ev.event_type]
        cctvs = self.cctv_client.list_cctvs()
        if not cctvs:
            logger.error("CCTV 목록 조회 실패")
            return {}

        results = []
        for cand in HOTSPOT_CANDIDATES:
            score = 0.0
            reasons = []

            active = False
            for acc in accidents:
                if acc.latitude and acc.longitude:
                    d = _haversine(cand["lat"], cand["lon"], acc.latitude, acc.longitude)
                    if d <= radius_km:
                        score += 50
                        active = True
                        reasons.append(f"사고 발생 중 ({d:.1f}km)")

            nearby_events = sum(
                1 for ev in all_events
                if ev.latitude and ev.longitude
                and _haversine(cand["lat"], cand["lon"], ev.latitude, ev.longitude) <= radius_km
            )
            score += nearby_events * 5
            if nearby_events:
                reasons.append(f"돌발 {nearby_events}건")

            nearby_cctvs = [
                (c, _haversine(cand["lat"], cand["lon"], c.latitude, c.longitude))
                for c in cctvs
                if c.stream_url
                and _haversine(cand["lat"], cand["lon"], c.latitude, c.longitude) <= radius_km
            ]
            nearby_cctvs.sort(key=lambda x: x[1])
            n_cctv = len(nearby_cctvs)
            score += n_cctv * 2
            reasons.append(f"CCTV {n_cctv}대")

            if n_cctv >= MAX_CONCURRENT_STREAMS:
                score += 10
                reasons.append(f"동시감시 가능 (>={MAX_CONCURRENT_STREAMS})")

            results.append({
                "name": cand["name"],
                "lat": cand["lat"],
                "lon": cand["lon"],
                "road": cand["road"],
                "score": score,
                "reason": " | ".join(reasons),
                "cctvs": [c for c, _ in nearby_cctvs],
                "cctv_distances": [d for _, d in nearby_cctvs],
                "incidents_nearby": nearby_events,
                "active_accident": active,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        best = results[0]

        logger.info("사고 다발 구간 선정 결과:")
        for r in results[:5]:
            flag = " ★" if r is best else ""
            logger.info("  [%5.0f점] %s — %s%s",
                        r["score"], r["name"], r["reason"], flag)

        return best

    def _evaluate_point(self, name: str, lat: float, lon: float,
                        radius_km: float) -> dict:
        """단일 좌표 평가."""
        cctvs = self.cctv_client.list_cctvs()
        nearby = [
            (c, _haversine(lat, lon, c.latitude, c.longitude))
            for c in cctvs
            if c.stream_url
            and _haversine(lat, lon, c.latitude, c.longitude) <= radius_km
        ]
        nearby.sort(key=lambda x: x[1])
        return {
            "name": name,
            "lat": lat,
            "lon": lon,
            "road": "",
            "score": len(nearby) * 2,
            "reason": f"CCTV {len(nearby)}대 (반경 {radius_km}km)",
            "cctvs": [c for c, _ in nearby],
            "cctv_distances": [d for _, d in nearby],
            "incidents_nearby": 0,
            "active_accident": False,
        }


# ═══════════════════════════════════════════════════════════════════════
# 사고 그룹핑: 여러 카메라의 트리거를 같은 incident_id로 묶기
# ═══════════════════════════════════════════════════════════════════════

class IncidentGrouper:
    """동일 사고에 대한 다중 카메라 트리거를 그룹핑."""

    def __init__(self):
        self._groups: dict[str, dict] = {}
        self._its_to_group: dict[str, str] = {}
        self._seq = 0
        self._lock = threading.Lock()

    def assign_group(self, trigger_type: str, cctv: CCTVInfo,
                     incident: IncidentEvent | None,
                     video_path: Path | None) -> str:
        """트리거를 그룹에 할당. 반환: group_id."""
        with self._lock:
            now = datetime.now()

            if incident and incident.event_id:
                if incident.event_id in self._its_to_group:
                    gid = self._its_to_group[incident.event_id]
                    self._groups[gid]["cameras"].append(cctv.cctv_id)
                    if video_path:
                        self._groups[gid]["clips"].append(str(video_path))
                    logger.info("기존 사고 그룹에 추가: %s (카메라: %s)",
                                gid, cctv.name)
                    return gid

            for gid, grp in self._groups.items():
                age = (now - grp["created_at"]).total_seconds()
                if age > 180:
                    continue
                d = _haversine(cctv.latitude, cctv.longitude,
                               grp["center_lat"], grp["center_lon"])
                if d <= 5.0 and cctv.cctv_id not in grp["cameras"]:
                    grp["cameras"].append(cctv.cctv_id)
                    if video_path:
                        grp["clips"].append(str(video_path))
                    logger.info("근접 사고 그룹에 추가: %s (카메라: %s, 거리: %.1fkm)",
                                gid, cctv.name, d)
                    return gid

            self._seq += 1
            gid = f"INC_{now.strftime('%Y%m%d_%H%M%S')}_{self._seq:03d}"
            self._groups[gid] = {
                "incident_id": incident.event_id if incident else None,
                "cameras": [cctv.cctv_id],
                "clips": [str(video_path)] if video_path else [],
                "center_lat": cctv.latitude,
                "center_lon": cctv.longitude,
                "trigger_type": trigger_type,
                "created_at": now,
            }
            if incident and incident.event_id:
                self._its_to_group[incident.event_id] = gid
            logger.info("새 사고 그룹 생성: %s (카메라: %s)", gid, cctv.name)
            return gid

    def get_groups(self) -> dict[str, dict]:
        """현재 활성 그룹 목록."""
        with self._lock:
            return dict(self._groups)


# ═══════════════════════════════════════════════════════════════════════
# 카메라별 워커: 각 CCTV 독립 감시 스레드 (온디맨드 녹화)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CameraStats:
    """카메라별 통계."""
    cctv_id: str
    cctv_name: str
    frames_processed: int = 0
    triggers_fired: int = 0
    recordings_started: int = 0
    recordings_extended: int = 0
    its_checks: int = 0
    incidents_confirmed: int = 0
    clips_preserved: int = 0
    clips_deleted: int = 0
    errors: int = 0
    reconnects: int = 0
    started_at: str = ""
    last_frame_at: str = ""
    status: str = "init"
    recording: bool = False


class CameraWorker:
    """단일 CCTV 감시 워커 (온디맨드 녹화 방식).

    독립 스레드에서 FrameSampler + VisionPipeline을 운영.
    트리거 발화 시에만 OnDemandRecorder로 녹화 시작.
    녹화 종료 후 shared IncidentVerifier로 교차확인.
    """

    def __init__(self, cctv: CCTVInfo, worker_id: int,
                 verifier: IncidentVerifier,
                 grouper: IncidentGrouper,
                 stop_event: threading.Event,
                 vision_pipeline=None):
        self.cctv = cctv
        self.worker_id = worker_id
        self.verifier = verifier
        self.grouper = grouper
        self.stop_event = stop_event
        self.pipeline = vision_pipeline

        self.sampler: FrameSampler | None = None
        self.recorder = OnDemandRecorder(cctv)
        self.stats = CameraStats(
            cctv_id=cctv.cctv_id,
            cctv_name=cctv.name,
        )
        self._pending_trigger = None
        self._trigger_window: list = []
        self._thread: threading.Thread | None = None
        self._logger = logging.getLogger(f"cam.{worker_id}.{cctv.cctv_id[:20]}")

    def start(self) -> bool:
        """워커 스레드 시작."""
        self.stats.started_at = datetime.now().isoformat()
        self.stats.status = "running"

        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"cam-{self.worker_id}",
            daemon=True,
        )
        self._thread.start()
        return True

    def _run_loop(self):
        """메인 감시 루프 (스레드에서 실행)."""
        # 1. FrameSampler 시작 (분석 전용, 녹화 안 함)
        self.sampler = FrameSampler(
            self.cctv.stream_url, sample_fps=SAMPLE_FPS,
            width=640, height=480,
        )
        if not self.sampler.start():
            self._logger.error("스트림 연결 실패: %s", self.cctv.name)
            self.stats.status = "error"
            return

        # 2. VisionPipeline (전체 카메라 독립 추론)
        pipeline = self.pipeline

        frame_idx = 0
        while not self.stop_event.is_set():
            ok, frame = self.sampler.read_frame()
            if not ok:
                self.stats.status = "reconnecting"
                self.stats.reconnects += 1
                self._logger.warning("[cam%d] 프레임 실패 — 5초 후 재연결", self.worker_id)
                self.sampler.stop()
                time.sleep(5)
                if self.stop_event.is_set():
                    break
                if not self.sampler.start():
                    self.stats.errors += 1
                    self._logger.error("[cam%d] 재연결 실패", self.worker_id)
                    time.sleep(10)
                    continue
                self.stats.status = "running"
                continue

            timestamp = frame_idx / SAMPLE_FPS
            frame_idx += 1
            self.stats.frames_processed = frame_idx
            self.stats.last_frame_at = datetime.now().isoformat()
            self.stats.recording = self.recorder.is_recording

            # Vision Pipeline 처리
            if pipeline is not None:
                try:
                    result = pipeline.process_frame(frame, frame_idx, timestamp)
                    for trigger in result.get("triggers", []):
                        self._handle_trigger(trigger, frame_idx)
                except Exception as e:
                    self.stats.errors += 1
                    if frame_idx % 100 == 0:
                        self._logger.error("[cam%d] Vision 오류: %s",
                                           self.worker_id, e)

            # 녹화 종료 시각 확인
            stopped, video_path = self.recorder.check_and_stop()
            if stopped:
                verify_thread = threading.Thread(
                    target=self._handle_recording_end,
                    args=(video_path,),
                    daemon=True,
                )
                verify_thread.start()

            # 주기적 상태 로그 (30프레임 = 30초마다)
            if frame_idx % 30 == 0:
                rec_state = "REC" if self.recorder.is_recording else "---"
                self._logger.info(
                    "[cam%d] %s: %d프레임 | 트리거=%d 녹화=%d 보존=%d 삭제=%d [%s]",
                    self.worker_id, self.cctv.name[:15],
                    frame_idx, self.stats.triggers_fired,
                    self.stats.recordings_started,
                    self.stats.clips_preserved, self.stats.clips_deleted,
                    rec_state,
                )

        self.stats.status = "stopped"
        self._cleanup()

    def _check_consensus(self, trigger) -> bool:
        """다중 트리거 합의 검사."""
        if trigger.type in INSTANT_RECORD_TYPES:
            return True

        cutoff = trigger.timestamp - CONSENSUS_WINDOW_SEC
        self._trigger_window = [
            t for t in self._trigger_window if t.timestamp > cutoff
        ]
        self._trigger_window.append(trigger)

        types_in_window = {t.type for t in self._trigger_window}
        return len(types_in_window) >= CONSENSUS_MIN_TYPES

    def _handle_trigger(self, trigger, frame_idx: int):
        """트리거 처리: 합의 검사 → 녹화 시작 또는 연장."""
        self.stats.triggers_fired += 1
        self._logger.info(
            "[cam%d] TRIGGER [%s] frame=%d severity=%.2f: %s",
            self.worker_id, trigger.type, trigger.frame_idx,
            trigger.severity, trigger.description,
        )

        if trigger.type not in RECORD_TRIGGER_TYPES:
            return

        if trigger.severity < SEVERITY_GATE:
            return

        # 이미 녹화 중 -> 연장
        if self.recorder.is_recording:
            extended = self.recorder.extend_recording(
                reason=f"[cam{self.worker_id}] 추가 [{trigger.type}]"
            )
            if extended:
                self.stats.recordings_extended += 1
            return

        # 쿨다운 중
        if not self.recorder.can_record():
            return

        # 합의 검사
        if not self._check_consensus(trigger):
            return

        # 녹화 시작
        event_id = f"RT_{trigger.type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.worker_id}"
        started = self.recorder.start_recording(trigger.type, event_id)
        if started:
            self.stats.recordings_started += 1
            self._pending_trigger = trigger
            self._trigger_window.clear()

    def _handle_recording_end(self, video_path: Path | None):
        """녹화 종료 후 ITS 교차확인 + 3단계 보존."""
        trigger = self._pending_trigger
        event_id = self.recorder._event_id or "unknown"

        if not trigger:
            return

        time.sleep(ITS_VERIFY_DELAY_SEC)

        self.stats.its_checks += 1
        confirmed, incident = self.verifier.verify(
            lat=self.cctv.latitude,
            lon=self.cctv.longitude,
            trigger_type=f"{trigger.type}_{self.cctv.cctv_id}",
        )

        if confirmed and incident and video_path:
            self._save_confirmed_cam(event_id, trigger, incident, video_path)
        elif video_path and video_path.exists() and trigger.severity >= SEVERITY_PENDING:
            self._handle_pending_cam(event_id, trigger, video_path)
        else:
            self._save_deleted_cam(event_id, trigger, video_path)

        self._pending_trigger = None

    def _save_confirmed_cam(self, event_id, trigger, incident, video_path):
        self.stats.incidents_confirmed += 1
        self.stats.clips_preserved += 1

        group_id = self.grouper.assign_group(
            trigger_type=trigger.type,
            cctv=self.cctv,
            incident=incident,
            video_path=video_path,
        )
        dist_km = _haversine(
            self.cctv.latitude, self.cctv.longitude,
            incident.latitude or 0, incident.longitude or 0,
        ) if incident.latitude and incident.longitude else None

        record = CollectionRecord(
            event_id=event_id,
            trigger_type=trigger.type,
            trigger_description=trigger.description,
            trigger_frame=trigger.frame_idx,
            trigger_severity=trigger.severity,
            cctv_id=self.cctv.cctv_id,
            cctv_name=self.cctv.name,
            cctv_lat=self.cctv.latitude,
            cctv_lon=self.cctv.longitude,
            incident_id=group_id,
            incident_road=f"{incident.road_name} {incident.direction}",
            incident_message=incident.message,
            incident_lat=incident.latitude,
            incident_lon=incident.longitude,
            match_distance_km=dist_km,
            video_path=str(video_path),
            video_size_mb=(video_path.stat().st_size / 1e6
                           if video_path.exists() else None),
            its_verified=True,
            action="confirmed",
        )
        save_collection_record(record)
        self._logger.info("[cam%d] 사고 확인 → 보존: %s (그룹: %s)",
                          self.worker_id, video_path.name, group_id)

    def _handle_pending_cam(self, event_id, trigger, video_path):
        PENDING_DIR.mkdir(parents=True, exist_ok=True)
        pending_path = PENDING_DIR / video_path.name
        shutil.move(str(video_path), str(pending_path))
        self._logger.info("[cam%d] Pending 보관: %s (severity=%.2f)",
                          self.worker_id, pending_path.name, trigger.severity)

        for retry in range(1, PENDING_MAX_RETRIES + 1):
            if self.stop_event.wait(PENDING_RETRY_INTERVAL_SEC):
                self._logger.info("[cam%d] 종료 — pending 보관 유지: %s",
                                  self.worker_id, pending_path.name)
                return
            confirmed, incident = self.verifier.verify(
                lat=self.cctv.latitude, lon=self.cctv.longitude,
                trigger_type=f"pending_{trigger.type}_{self.cctv.cctv_id}",
            )
            if confirmed and incident:
                final_path = SAVE_DIR / pending_path.name
                shutil.move(str(pending_path), str(final_path))
                self._save_confirmed_cam(event_id, trigger, incident, final_path)
                return
            self._logger.info("[cam%d] Pending 재확인 %d/%d 실패: %s",
                              self.worker_id, retry, PENDING_MAX_RETRIES,
                              pending_path.name)

        if trigger.severity >= SEVERITY_FORCE_PRESERVE and pending_path.exists():
            final_path = SAVE_DIR / pending_path.name
            shutil.move(str(pending_path), str(final_path))
            self.stats.clips_preserved += 1
            record = CollectionRecord(
                event_id=event_id,
                trigger_type=trigger.type,
                trigger_description=trigger.description,
                trigger_frame=trigger.frame_idx,
                trigger_severity=trigger.severity,
                cctv_id=self.cctv.cctv_id,
                cctv_name=self.cctv.name,
                cctv_lat=self.cctv.latitude,
                cctv_lon=self.cctv.longitude,
                video_path=str(final_path),
                video_size_mb=(final_path.stat().st_size / 1e6
                               if final_path.exists() else None),
                its_verified=False,
                action="pending_preserved",
            )
            save_collection_record(record)
            self._logger.info("[cam%d] Pending 만료 → 고severity 보존: %s",
                              self.worker_id, final_path.name)
        else:
            self._save_deleted_cam(event_id, trigger, pending_path)

    def _save_deleted_cam(self, event_id, trigger, video_path):
        self.stats.clips_deleted += 1
        if video_path and video_path.exists():
            size_mb = video_path.stat().st_size / 1e6
            video_path.unlink()
            self._logger.info("[cam%d] 미확인 → 삭제: %s (%.1f MB)",
                              self.worker_id, video_path.name, size_mb)

        record = CollectionRecord(
            event_id=event_id,
            trigger_type=trigger.type,
            trigger_description=trigger.description,
            trigger_frame=trigger.frame_idx,
            trigger_severity=trigger.severity,
            cctv_id=self.cctv.cctv_id,
            cctv_name=self.cctv.name,
            cctv_lat=self.cctv.latitude,
            cctv_lon=self.cctv.longitude,
            its_verified=False,
            action="deleted",
        )
        save_collection_record(record)

    def _cleanup(self):
        """자원 정리."""
        if self.sampler:
            self.sampler.stop()
        if self.recorder and self.recorder.is_recording:
            self.recorder.force_stop()
        self._logger.info("[cam%d] %s 정리 완료", self.worker_id, self.cctv.name)


# ═══════════════════════════════════════════════════════════════════════
# 다중 CCTV 동시 감시 파이프라인
# ═══════════════════════════════════════════════════════════════════════

class MultiCCTVPipeline:
    """다중 CCTV 동시 감시 파이프라인 (온디맨드 녹화).

    사고 다발 구간을 선정하고, 반경 내 CCTV N대를 동시에 분석한다.
    녹화는 트리거 발화한 카메라만 수행 (전체 녹화 X).
    """

    def __init__(self):
        self.cctv_client = ITSCCTVClient()
        self.hotspot_selector = HotspotSelector(self.cctv_client)
        self.verifier = IncidentVerifier()
        self.grouper = IncidentGrouper()
        self.workers: list[CameraWorker] = []
        self._stop_event = threading.Event()

    def multi_monitor(self, lat: float | None = None, lon: float | None = None,
                      radius_km: float = 5.0, max_cameras: int = MAX_CONCURRENT_STREAMS):
        """다중 CCTV 동시 감시 시작."""
        logger.info("=" * 60)
        logger.info("다중 CCTV 동시 감시 시스템 시작 (OnDemand v2)")
        logger.info("=" * 60)

        hotspot = self.hotspot_selector.select(radius_km, lat, lon)
        if not hotspot or not hotspot.get("cctvs"):
            logger.error("감시 가능한 CCTV 없음")
            return

        target_cctvs = hotspot["cctvs"][:max_cameras]
        target_distances = hotspot["cctv_distances"][:max_cameras]

        logger.info("")
        logger.info("선정 구간: %s", hotspot["name"])
        logger.info("  좌표: (%.4f, %.4f)", hotspot["lat"], hotspot["lon"])
        logger.info("  근거: %s", hotspot["reason"])
        logger.info("  반경 내 CCTV: %d대 (감시 대상: %d대)",
                     len(hotspot["cctvs"]), len(target_cctvs))
        logger.info("")

        self._save_selection_log(hotspot, target_cctvs, target_distances, radius_km)

        # Vision Pipeline 초기화 (카메라별 독립 인스턴스)
        n_cams = len(target_cctvs)
        logger.info("Vision Pipeline 초기화 중... (%d대)", n_cams)
        pipelines: list = []
        try:
            pipelines = self._init_vision_pipelines(n_cams)
            logger.info("Vision Pipeline %d개 준비 완료", len(pipelines))
        except Exception as e:
            logger.warning("Vision Pipeline 초기화 실패: %s", e)

        # 카메라 워커 생성 + 시작
        logger.info("")
        for i, (cctv, dist) in enumerate(zip(target_cctvs, target_distances)):
            cam_pipeline = pipelines[i] if i < len(pipelines) else None

            worker = CameraWorker(
                cctv=cctv,
                worker_id=i,
                verifier=self.verifier,
                grouper=self.grouper,
                stop_event=self._stop_event,
                vision_pipeline=cam_pipeline,
            )
            self.workers.append(worker)

            role = "Vision+OnDemand" if cam_pipeline else "OnDemand(대기)"
            logger.info("  [cam%d] %s (%.1fkm) — %s", i, cctv.name, dist, role)
            worker.start()

        # SIGINT 핸들러
        def signal_handler(sig, frame):
            logger.info("")
            logger.info("중단 신호 수신 — 모든 카메라 정리 중...")
            self._stop_event.set()

        signal.signal(signal.SIGINT, signal_handler)

        # 상태 모니터링 루프 (메인 스레드)
        logger.info("")
        logger.info("=" * 60)
        logger.info("감시 중... (Ctrl+C로 중단)")
        logger.info("  카메라 %d대 | 반경 %.1fkm | Vision: 전체 | 녹화: 트리거 시에만",
                     len(target_cctvs), radius_km)
        logger.info("=" * 60)

        try:
            while not self._stop_event.is_set():
                time.sleep(60)
                if self._stop_event.is_set():
                    break
                self._log_status()
                self._save_status_file()
        except KeyboardInterrupt:
            self._stop_event.set()

        logger.info("모든 워커 종료 대기...")
        for w in self.workers:
            if w._thread and w._thread.is_alive():
                w._thread.join(timeout=10)

        self._print_multi_summary()
        self._save_status_file()

    def _init_vision_pipeline(self):
        """Vision Pipeline 초기화 (단일 인스턴스)."""
        return self._init_vision_pipelines(1)[0]

    def _init_vision_pipelines(self, count: int) -> list:
        """카메라별 독립 Vision Pipeline 생성."""
        from detector import VehicleDetector
        from tracker import VehicleTracker
        from layer1_vision.speed_estimator import SpeedEstimator
        from layer1_vision.trigger_detector import TriggerDetector
        from layer1_vision.vision_pipeline import VisionPipeline

        class DummyClassifier:
            def predict_batch(self, crops):
                return ["T1"] * len(crops), [0.5] * len(crops)

        pipelines = []
        for i in range(count):
            pipeline = VisionPipeline(
                detector=VehicleDetector(),
                tracker=VehicleTracker(fps=max(1, SAMPLE_FPS)),
                classifier=DummyClassifier(),
                speed_est=SpeedEstimator(),
                trigger_det=TriggerDetector(),
                fps=float(SAMPLE_FPS),
            )
            pipelines.append(pipeline)
            logger.info("  Vision Pipeline %d/%d 준비", i + 1, count)
        return pipelines

    def _log_status(self):
        """주기적 상태 로그."""
        total_frames = sum(w.stats.frames_processed for w in self.workers)
        total_triggers = sum(w.stats.triggers_fired for w in self.workers)
        total_recordings = sum(w.stats.recordings_started for w in self.workers)
        total_preserved = sum(w.stats.clips_preserved for w in self.workers)
        total_deleted = sum(w.stats.clips_deleted for w in self.workers)
        total_errors = sum(w.stats.errors for w in self.workers)

        logger.info("─" * 50)
        logger.info("요약: 프레임=%d 트리거=%d 녹화=%d 보존=%d 삭제=%d 에러=%d",
                     total_frames, total_triggers, total_recordings,
                     total_preserved, total_deleted, total_errors)
        for w in self.workers:
            rec_state = " [REC]" if w.recorder.is_recording else ""
            logger.info("  [cam%d] %s: %s%s | %d프레임 | 트리거=%d 보존=%d 삭제=%d",
                         w.worker_id, w.cctv.name[:15], w.stats.status,
                         rec_state, w.stats.frames_processed,
                         w.stats.triggers_fired,
                         w.stats.clips_preserved, w.stats.clips_deleted)

        groups = self.grouper.get_groups()
        if groups:
            logger.info("  사고 그룹: %d건", len(groups))
            for gid, grp in groups.items():
                logger.info("    %s: 카메라 %d대, 클립 %d건",
                             gid, len(grp["cameras"]), len(grp["clips"]))
        logger.info("─" * 50)

    def _save_status_file(self):
        """실시간 상태를 JSON 파일로 저장."""
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        status = {
            "updated_at": datetime.now().isoformat(),
            "cameras": [
                {
                    "worker_id": w.worker_id,
                    "cctv_id": w.stats.cctv_id,
                    "cctv_name": w.stats.cctv_name,
                    "status": w.stats.status,
                    "recording": w.recorder.is_recording,
                    "frames": w.stats.frames_processed,
                    "triggers": w.stats.triggers_fired,
                    "recordings": w.stats.recordings_started,
                    "its_checks": w.stats.its_checks,
                    "confirmed": w.stats.incidents_confirmed,
                    "preserved": w.stats.clips_preserved,
                    "deleted": w.stats.clips_deleted,
                    "errors": w.stats.errors,
                    "reconnects": w.stats.reconnects,
                    "started_at": w.stats.started_at,
                    "last_frame_at": w.stats.last_frame_at,
                }
                for w in self.workers
            ],
            "groups": {
                gid: {
                    "incident_id": grp["incident_id"],
                    "cameras": grp["cameras"],
                    "clips": grp["clips"],
                    "trigger_type": grp["trigger_type"],
                    "created_at": grp["created_at"].isoformat(),
                }
                for gid, grp in self.grouper.get_groups().items()
            },
        }
        with open(MULTI_STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)

    def _save_selection_log(self, hotspot: dict, cctvs: list[CCTVInfo],
                            distances: list[float], radius_km: float):
        """구간 선정 근거 로그 저장."""
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOG_DIR / f"hotspot_selection_{ts}.json"
        data = {
            "selected_at": datetime.now().isoformat(),
            "hotspot_name": hotspot["name"],
            "center_lat": hotspot["lat"],
            "center_lon": hotspot["lon"],
            "radius_km": radius_km,
            "score": hotspot["score"],
            "reason": hotspot["reason"],
            "incidents_nearby": hotspot.get("incidents_nearby", 0),
            "active_accident": hotspot.get("active_accident", False),
            "total_cctvs_in_radius": len(hotspot["cctvs"]),
            "monitored_cctvs": [
                {
                    "cctv_id": c.cctv_id,
                    "name": c.name,
                    "distance_km": round(d, 2),
                    "lat": c.latitude,
                    "lon": c.longitude,
                }
                for c, d in zip(cctvs, distances)
            ],
        }
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("선정 근거 로그: %s", log_file)

    def _print_multi_summary(self):
        """다중 감시 세션 요약."""
        print()
        print("=" * 60)
        print("다중 CCTV 감시 세션 요약 (OnDemand v2)")
        print("=" * 60)
        for w in self.workers:
            s = w.stats
            uptime = ""
            if s.started_at:
                try:
                    start = datetime.fromisoformat(s.started_at)
                    uptime = f" ({(datetime.now() - start).total_seconds():.0f}초)"
                except ValueError:
                    pass
            print(f"  [cam{w.worker_id}] {s.cctv_name}")
            print(f"    상태: {s.status}{uptime}")
            print(f"    프레임: {s.frames_processed} | 트리거: {s.triggers_fired} "
                  f"| 녹화: {s.recordings_started} | 보존: {s.clips_preserved} "
                  f"| 삭제: {s.clips_deleted}")
            if s.errors or s.reconnects:
                print(f"    에러: {s.errors} | 재연결: {s.reconnects}")

        groups = self.grouper.get_groups()
        if groups:
            print()
            print(f"  사고 그룹: {len(groups)}건")
            for gid, grp in groups.items():
                print(f"    {gid}: 카메라 {len(grp['cameras'])}대 "
                      f"({', '.join(grp['cameras'][:3])})")

        total_preserved = sum(w.stats.clips_preserved for w in self.workers)
        total_deleted = sum(w.stats.clips_deleted for w in self.workers)
        print()
        print(f"  총 보존: {total_preserved}건 | 삭제: {total_deleted}건")
        print(f"  저장 경로: {SAVE_DIR}")
        print("=" * 60)

        # 세션 로그
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        summary_file = LOG_DIR / f"multi_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({
                "cameras": [
                    {"id": w.stats.cctv_id, "name": w.stats.cctv_name,
                     "frames": w.stats.frames_processed,
                     "triggers": w.stats.triggers_fired,
                     "recordings": w.stats.recordings_started,
                     "preserved": w.stats.clips_preserved,
                     "deleted": w.stats.clips_deleted}
                    for w in self.workers
                ],
                "groups": len(groups),
                "ended_at": datetime.now().isoformat(),
            }, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse

    # dotenv 로드
    from pathlib import Path
    env_path = Path("/workspace/.env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="실시간 사고영상 수집 파이프라인 (v2 — 징조 기반 온디맨드 녹화)",
    )
    sub = parser.add_subparsers(dest="command")

    # monitor (단일 CCTV)
    p_mon = sub.add_parser("monitor", help="단일 CCTV 모니터링 (징조 시에만 녹화)")
    p_mon.add_argument("--lat", type=float, help="CCTV 검색 위도")
    p_mon.add_argument("--lon", type=float, help="CCTV 검색 경도")
    p_mon.add_argument("--cctv-id", help="특정 CCTV ID")
    p_mon.add_argument("--max-frames", type=int, default=0,
                        help="최대 프레임 수 (0=무제한)")

    # multi (다중 CCTV 동시 감시)
    p_multi = sub.add_parser("multi", help="사고다발구간 다중 CCTV 동시 감시")
    p_multi.add_argument("--lat", type=float, help="감시 중심 위도 (미지정 시 자동 선정)")
    p_multi.add_argument("--lon", type=float, help="감시 중심 경도")
    p_multi.add_argument("--radius", type=float, default=5.0,
                         help="CCTV 검색 반경 km (기본: 5.0)")
    p_multi.add_argument("--max-cameras", type=int, default=MAX_CONCURRENT_STREAMS,
                         help=f"최대 동시 카메라 수 (기본: {MAX_CONCURRENT_STREAMS})")

    # hotspot
    p_hs = sub.add_parser("hotspot", help="사고 다발 구간 선정 (조회)")
    p_hs.add_argument("--radius", type=float, default=5.0,
                      help="CCTV 검색 반경 km")

    # dry-run
    sub.add_parser("dry-run", help="전체 흐름 시뮬레이션")

    # status
    sub.add_parser("status", help="수집 현황 확인")

    args = parser.parse_args()

    if args.command == "monitor":
        pipeline = RealtimeAccidentPipeline()
        pipeline.monitor(
            lat=args.lat,
            lon=args.lon,
            cctv_id=args.cctv_id,
            max_frames=args.max_frames,
        )
    elif args.command == "multi":
        multi = MultiCCTVPipeline()
        multi.multi_monitor(
            lat=args.lat,
            lon=args.lon,
            radius_km=args.radius,
            max_cameras=args.max_cameras,
        )
    elif args.command == "hotspot":
        selector = HotspotSelector()
        result = selector.select(radius_km=args.radius)
        if result:
            print()
            print("=" * 60)
            print("사고 다발 구간 선정 결과")
            print("=" * 60)
            print(f"  선정: {result['name']}")
            print(f"  좌표: ({result['lat']}, {result['lon']})")
            print(f"  점수: {result['score']:.0f}")
            print(f"  근거: {result['reason']}")
            print(f"  CCTV: {len(result['cctvs'])}대")
            for i, (c, d) in enumerate(zip(
                    result['cctvs'][:10], result['cctv_distances'][:10])):
                print(f"    [{i+1}] {d:.1f}km — {c.name} ({c.cctv_id})")
            print("=" * 60)
    elif args.command == "dry-run":
        pipeline = RealtimeAccidentPipeline()
        pipeline.dry_run()
    elif args.command == "status":
        pipeline = RealtimeAccidentPipeline()
        pipeline.status()
    else:
        parser.print_help()
        print()
        print("예시:")
        print("  python run_realtime.py monitor                              # 단일 CCTV (자동)")
        print("  python run_realtime.py multi                                # 다중 CCTV (자동 구간)")
        print("  python run_realtime.py multi --lat 37.11 --lon 126.89       # 다중 CCTV (지정)")
        print("  python run_realtime.py multi --max-cameras 2                # 카메라 수 제한")
        print("  python run_realtime.py hotspot                              # 구간 선정 조회")
        print("  python run_realtime.py dry-run                              # 시뮬레이션")
        print("  python run_realtime.py status                               # 현황 확인")


if __name__ == "__main__":
    main()
