"""전국 363대 CCTV 3-Tier 통합 파이프라인.

3-Tier 아키텍처:
  Tier 1 (313대): 프리필터만 (MOG2/프레임차분). 이상 감지 시 Tier 3 승격.
  Tier 2 (50대):  핫스팟 상시 정밀 (YOLO+ByteTrack+TriggerDetector).
  Tier 3 (동적):  Tier 1 이상 후보 or ITS 사고 편입 -> 정밀 분석 -> 녹화.

실행:
  python run_nationwide.py start                 # 전국 363대 시작
  python run_nationwide.py start --max-cameras 10  # 테스트용 카메라 수 제한
  python run_nationwide.py status                # 수집 현황
"""
from __future__ import annotations

import json
import logging
import os
import re
import queue
import signal
import shutil
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

# ── 경로 설정 ────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
PIPELINE_SRC = Path("/workspace/prj_cctv/pipeline/src")
ACCIDENT_SRC = Path("/workspace/prj_cctv/사고분석_설계/src")

os.environ.setdefault("OMP_NUM_THREADS", "4")

# ── 이중 config.py 충돌 해소 ────────────────────────────────────────
# Phase 1: layer0_data/ 모듈 import (layer0_data/config.py 필요)
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(1, str(ACCIDENT_SRC))

from config import (
    RECORD_COOLDOWN_SEC,
    RECORD_DURATION_MAX_SEC,
    RECORD_DURATION_MIN_SEC,
    RECORD_EXTEND_SEC,
    ITS_VERIFY_DELAY_SEC,
    STREAM_DIR,
)
from track3_api_incident import IncidentEvent, ITSIncidentClient
from track3_cctv_stream import CCTVInfo, ITSCCTVClient, _haversine
from config_new import (
    NATIONWIDE_MAX_CAMERAS,
    TIER2_HOTSPOT_COUNT,
    TIER3_MAX_CONCURRENT,
    STREAM_SAMPLE_FPS,
    PREFILTER_COOLDOWN_SEC,
    PREFILTER_TIER3_HOLD_SEC,
    INCIDENT_POLL_SEC,
    OUTBREAK_ENABLED,
    OUTBREAK_DWELL_SEC,
    OUTBREAK_DWELL_EXTEND_SEC,
    OUTBREAK_CAPTURE_FPS,
    OUTBREAK_CCTV_DELAY,
    OUTBREAK_INCIDENT_DELAY,
    OUTBREAK_POLL_INTERVAL,
    OUTBREAK_MAX_PER_CYCLE,
    OUTBREAK_MAX_PER_INCIDENT,
    OUTBREAK_RECORDING_DIR,
)
from stream_manager import StreamManager
from incident_reactor import IncidentReactor
from layer1_vision.prefilter import PreFilter, PreFilterResult
from run_realtime import (
    CollectionRecord,
    IncidentVerifier,
    OnDemandRecorder,
    save_collection_record,
    SAVE_DIR,
    META_DIR,
    LOG_DIR,
    PENDING_DIR,
    SEVERITY_GATE,
    SEVERITY_PENDING,
    SEVERITY_FORCE_PRESERVE,
    PENDING_MAX_RETRIES,
    PENDING_RETRY_INTERVAL_SEC,
    RECORD_TRIGGER_TYPES,
    CONSENSUS_WINDOW_SEC,
    CONSENSUS_MIN_TYPES,
    INSTANT_RECORD_TYPES,
)

# Phase 2: config 교체 → pipeline/src (detector, tracker lazy import용)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("config", str(PIPELINE_SRC / "config.py"))
_pipeline_cfg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pipeline_cfg)
sys.modules["config"] = _pipeline_cfg
sys.path.insert(0, str(PIPELINE_SRC))

logger = logging.getLogger(__name__)

NATIONWIDE_STATUS_FILE = LOG_DIR / "nationwide_status.json"

# VehicleDetector 싱글턴 (YOLO 모델 로딩 1회)
_shared_detector = None
_detector_lock = threading.Lock()


def _get_shared_detector():
    """VehicleDetector 싱글턴 반환."""
    global _shared_detector
    if _shared_detector is None:
        with _detector_lock:
            if _shared_detector is None:
                from detector import VehicleDetector
                _shared_detector = VehicleDetector()
                logger.info("공유 VehicleDetector 로딩 완료")
    return _shared_detector


def _create_vision_pipeline():
    """VisionPipeline 인스턴스 생성 (Detector 공유, Tracker/Trigger 개별)."""
    from tracker import VehicleTracker
    from layer1_vision.speed_estimator import SpeedEstimator
    from layer1_vision.trigger_detector import TriggerDetector
    from layer1_vision.vision_pipeline import VisionPipeline

    class DummyClassifier:
        """사고 감지에서는 차종 분류 불필요."""
        def predict_batch(self, crops):
            return ["T1"] * len(crops), [0.5] * len(crops)

    return VisionPipeline(
        detector=_get_shared_detector(),
        tracker=VehicleTracker(fps=max(1, STREAM_SAMPLE_FPS)),
        classifier=DummyClassifier(),
        speed_est=SpeedEstimator(),
        trigger_det=TriggerDetector(),
        fps=float(STREAM_SAMPLE_FPS),
    )


# ═══════════════════════════════════════════════════════════════════════
# OutbreakFeeder: 돌발정보 CCTV 체류 캡처 → 파이프라인 브릿지
# ═══════════════════════════════════════════════════════════════════════

try:
    from browser_stream import OutbreakCCTVCapture, OutbreakIncident, CapturedFrame
    _HAS_OUTBREAK = True
except ImportError:
    _HAS_OUTBREAK = False


class OutbreakFeeder:
    """OutbreakCCTVCapture → NationwidePipeline 프레임 브릿지.

    돌발정보 페이지 사고/고장 인근 CCTV를 순찰하며 체류 캡처(30s×1fps).
    캡처된 프레임을 NationwidePipeline Vision 큐에 주입하고,
    트리거 발화 시 체류 연장 + 프레임 시퀀스 저장.
    """

    def __init__(self, pipeline: "NationwidePipeline", stop_event: threading.Event):
        self._pipeline = pipeline
        self._stop_event = stop_event
        self._capture: OutbreakCCTVCapture | None = None
        self._thread: threading.Thread | None = None

        self._extend_events: dict[str, threading.Event] = {}
        self._frame_buffers: dict[str, deque] = {}
        self._recordings: dict[str, dict] = {}
        self._lock = threading.Lock()

        self._output_dir = OUTBREAK_RECORDING_DIR
        self.stats = {
            "patrols": 0,
            "incidents": 0,
            "dwells": 0,
            "frames_injected": 0,
            "triggers_received": 0,
            "recordings_saved": 0,
        }

    def start(self) -> bool:
        if not _HAS_OUTBREAK:
            logger.error("OutbreakFeeder: playwright 미설치, 비활성화")
            return False

        # Playwright sync API는 greenlet 기반 → 브라우저를 사용할 스레드에서 시작해야 함
        # start()는 스레드만 띄우고, 브라우저는 _patrol_loop 내에서 시작
        self._thread = threading.Thread(
            target=self._patrol_loop,
            name="outbreak-feeder",
            daemon=True,
        )
        self._thread.start()
        logger.info("OutbreakFeeder 시작 (dwell=%ds, fps=%.1f)",
                     OUTBREAK_DWELL_SEC, OUTBREAK_CAPTURE_FPS)
        return True

    def stop(self):
        if self._capture:
            self._capture.stop()
            self._capture = None
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("OutbreakFeeder 종료 | %s", self.stats)

    def on_trigger(self, trigger: Any, cctv_id: str):
        """파이프라인 트리거 발화 → 체류 연장 + 프레임 시퀀스 저장 시작."""
        with self._lock:
            self.stats["triggers_received"] += 1

        ev = self._extend_events.get(cctv_id)
        if ev:
            ev.set()
            logger.info("트리거→체류연장: %s [%s]", trigger.type, cctv_id[:30])

        with self._lock:
            if cctv_id not in self._recordings:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                rec_dir = self._output_dir / f"{cctv_id}_{ts}"
                rec_dir.mkdir(parents=True, exist_ok=True)
                (rec_dir / "frames").mkdir(exist_ok=True)
                (rec_dir / "keyframes").mkdir(exist_ok=True)

                pre_frames = list(self._frame_buffers.get(cctv_id, []))
                self._recordings[cctv_id] = {
                    "dir": rec_dir,
                    "frame_count": 0,
                    "triggers": [],
                    "started_at": time.time(),
                }

                from PIL import Image
                for i, (_, frame_bgr) in enumerate(pre_frames):
                    path = rec_dir / "frames" / f"{i:04d}.jpg"
                    Image.fromarray(frame_bgr[:, :, ::-1]).save(
                        str(path), "JPEG", quality=85,
                    )
                self._recordings[cctv_id]["frame_count"] = len(pre_frames)

            rec = self._recordings[cctv_id]
            rec["triggers"].append({
                "type": trigger.type,
                "severity": float(trigger.severity),
                "description": trigger.description,
                "timestamp": time.time(),
                "frame_idx": trigger.frame_idx,
            })

        trigger_log = rec["dir"] / "trigger_log.jsonl"
        with open(trigger_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec["triggers"][-1], ensure_ascii=False) + "\n")

    def get_stats(self) -> dict:
        with self._lock:
            return dict(self.stats)

    # ── 내부 ─────────────────────────────────────────────────────────

    def _patrol_loop(self):
        # Playwright sync API는 greenlet 기반 → 사용 스레드에서 브라우저 시작
        self._capture = OutbreakCCTVCapture(
            target_types=("사고", "고장"),
            cctv_delay=OUTBREAK_CCTV_DELAY,
            incident_delay=OUTBREAK_INCIDENT_DELAY,
            poll_interval=OUTBREAK_POLL_INTERVAL,
            max_per_cycle=OUTBREAK_MAX_PER_CYCLE,
            max_cctvs_per_incident=OUTBREAK_MAX_PER_INCIDENT,
        )
        if not self._capture.start():
            logger.error("OutbreakFeeder: 브라우저 시작 실패 (patrol 스레드)")
            return

        while not self._stop_event.is_set():
            try:
                self._run_one_patrol()
            except Exception as e:
                logger.error("순찰 루프 오류: %s", e, exc_info=True)
                try:
                    if self._capture:
                        self._capture._navigate_to_outbreak()
                except Exception:
                    pass

            if not self._stop_event.is_set():
                self._stop_event.wait(OUTBREAK_POLL_INTERVAL)

    def _run_one_patrol(self):
        if not self._capture or not self._capture._page:
            return

        with self._lock:
            self.stats["patrols"] += 1

        incidents = self._capture.scan_incidents()
        if not incidents:
            logger.info("OutbreakFeeder: 대상 사고 0건")
            return

        with self._lock:
            self.stats["incidents"] += len(incidents)

        dwell_count = 0
        for inc in incidents:
            if self._stop_event.is_set():
                break
            if dwell_count >= OUTBREAK_MAX_PER_CYCLE:
                break

            click_ok = self._capture._click_incident(inc.dom_index)
            if not click_ok:
                continue
            self._capture._page.wait_for_timeout(2000)

            cctvs = self._capture._get_nearby_cctvs()
            if not cctvs:
                continue

            limit = min(len(cctvs), OUTBREAK_MAX_PER_INCIDENT)
            for cctv in cctvs[:limit]:
                if self._stop_event.is_set() or dwell_count >= OUTBREAK_MAX_PER_CYCLE:
                    break

                cctv_id = self._make_cctv_id(inc, cctv)
                extend_event = threading.Event()
                self._extend_events[cctv_id] = extend_event
                self._frame_buffers[cctv_id] = deque(maxlen=30)

                with self._lock:
                    self.stats["dwells"] += 1

                self._capture.capture_dwelling(
                    cctv=cctv,
                    incident=inc,
                    dwell_sec=OUTBREAK_DWELL_SEC,
                    fps=OUTBREAK_CAPTURE_FPS,
                    on_frame=lambda cf, cid=cctv_id: self._on_dwelling_frame(cf, cid),
                    extend_event=extend_event,
                    extend_sec=OUTBREAK_DWELL_EXTEND_SEC,
                )

                self._finalize_recording(cctv_id, inc)
                self._extend_events.pop(cctv_id, None)
                self._frame_buffers.pop(cctv_id, None)
                dwell_count += 1

                if not self._stop_event.is_set():
                    time.sleep(OUTBREAK_CCTV_DELAY)

            if not self._stop_event.is_set():
                time.sleep(OUTBREAK_INCIDENT_DELAY)

        logger.info(
            "OutbreakFeeder 순찰 완료: 사고 %d건, 체류 %d회",
            len(incidents), dwell_count,
        )

    def _on_dwelling_frame(self, captured_frame: Any, cctv_id: str):
        frame_bgr = captured_frame.frame

        buf = self._frame_buffers.get(cctv_id)
        if buf is not None:
            buf.append((time.time(), frame_bgr))

        self._pipeline._on_frame(frame_bgr, cctv_id, tier=3)
        with self._lock:
            self.stats["frames_injected"] += 1

        rec = self._recordings.get(cctv_id)
        if rec:
            from PIL import Image
            idx = rec["frame_count"]
            path = rec["dir"] / "frames" / f"{idx:04d}.jpg"
            Image.fromarray(frame_bgr[:, :, ::-1]).save(
                str(path), "JPEG", quality=85,
            )
            rec["frame_count"] = idx + 1

    def _finalize_recording(self, cctv_id: str, incident: Any):
        rec = self._recordings.pop(cctv_id, None)
        if not rec:
            return
        if rec["frame_count"] == 0:
            shutil.rmtree(rec["dir"], ignore_errors=True)
            return

        metadata = {
            "cctv_id": cctv_id,
            "incident_type": incident.event_type,
            "road_name": incident.road_name,
            "message": incident.message,
            "started_at": datetime.fromtimestamp(rec["started_at"]).isoformat(),
            "ended_at": datetime.now().isoformat(),
            "frame_count": rec["frame_count"],
            "trigger_count": len(rec["triggers"]),
            "triggers": rec["triggers"],
        }
        meta_path = rec["dir"] / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        with self._lock:
            self.stats["recordings_saved"] += 1

        logger.info(
            "녹화 완료: %s — %d프레임, %d트리거",
            rec["dir"].name, rec["frame_count"], len(rec["triggers"]),
        )

    @staticmethod
    def _make_cctv_id(incident: Any, cctv: dict) -> str:
        raw = cctv.get("name", "unknown")
        sanitized = re.sub(r'[\[\]\s/]', '_', raw).strip('_')[:40]
        return f"OB_{sanitized}"


class NationwidePipeline:
    """363대 전국 CCTV 3-Tier 통합 파이프라인.

    CPU 경합 방지: Tier 2/3 프레임을 큐에 넣고, 전용 워커 스레드 N개가
    순차적으로 YOLO 처리. 워커 스레드 수 = max(4, OMP_NUM_THREADS).
    """

    # YOLO 처리 워커 스레드 수 (GIL 고려, OMP_NUM_THREADS와 매칭)
    _VISION_WORKERS = max(4, int(os.environ.get("OMP_NUM_THREADS", "4")))
    _FRAME_QUEUE_SIZE = 200

    def __init__(self, max_cameras: int = 0, outbreak_only: bool = False):
        self._stop_event = threading.Event()
        self._outbreak_only = outbreak_only
        self._cctv_client = ITSCCTVClient()
        self._stream_manager = StreamManager(
            cctv_client=self._cctv_client,
            stop_event=self._stop_event,
        )
        self._incident_reactor = IncidentReactor(
            stream_manager=self._stream_manager,
            cctv_client=self._cctv_client,
            stop_event=self._stop_event,
        )
        self._outbreak_feeder: OutbreakFeeder | None = None

        self._prefilters: dict[str, PreFilter] = {}
        self._vision_pipelines: dict[str, Any] = {}
        self._recorders: dict[str, OnDemandRecorder] = {}
        self._verifier = IncidentVerifier()
        self._trigger_windows: dict[str, list] = {}
        self._pending_triggers: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._max_cameras = max_cameras if max_cameras > 0 else NATIONWIDE_MAX_CAMERAS

        # 프로듀서-컨슈머 큐: Tier 2/3 프레임을 직렬화 처리
        self._frame_queue: queue.Queue = queue.Queue(
            maxsize=self._FRAME_QUEUE_SIZE,
        )
        self._vision_threads: list[threading.Thread] = []

        self.stats = {
            "frames_t1": 0,
            "frames_t2": 0,
            "frames_t3": 0,
            "triggers": 0,
            "recordings": 0,
            "preserved": 0,
            "deleted": 0,
            "anomalies_detected": 0,
            "queue_drops": 0,
            "start_time": None,
        }

    # ── Vision 큐 워커 ────────────────────────────────────────────────

    def _start_vision_workers(self) -> None:
        """Vision 처리 전용 스레드 N개 시작."""
        for i in range(self._VISION_WORKERS):
            t = threading.Thread(
                target=self._vision_worker_loop,
                name=f"vision-worker-{i}",
                daemon=True,
            )
            t.start()
            self._vision_threads.append(t)
        logger.info(
            "Vision 워커 %d개 시작 (큐 크기 %d)",
            self._VISION_WORKERS, self._FRAME_QUEUE_SIZE,
        )

    def _vision_worker_loop(self) -> None:
        """큐에서 프레임을 꺼내 Tier 2/3 처리."""
        while not self._stop_event.is_set():
            try:
                item = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:  # 종료 신호
                break
            frame, cctv_id, tier, timestamp = item
            try:
                self._handle_tier23(frame, cctv_id, tier, timestamp)
            except Exception as e:
                logger.error("Vision 워커 오류 [%s]: %s", cctv_id[:20], e)

    # ── 콜백 ─────────────────────────────────────────────────────────

    def _on_frame(self, frame: Any, cctv_id: str, tier: int) -> None:
        """StreamManager 프레임 도착 콜백 (워커 스레드에서 호출).

        Tier 1: 직접 처리 (경량 PreFilter, GIL 경합 미미).
        Tier 2/3: 큐에 넣어 Vision 워커가 직렬 처리 (CPU 경합 방지).
        """
        timestamp = time.monotonic()

        if tier == 1:
            # Tier 3 -> 1 강등 후 첫 프레임: 파이프라인/녹화기 메모리 회수
            if cctv_id in self._vision_pipelines:
                self._cleanup_pipeline(cctv_id)
            self._handle_tier1(frame, cctv_id, timestamp)
        elif tier in (2, 3):
            try:
                self._frame_queue.put_nowait(
                    (frame, cctv_id, tier, timestamp),
                )
            except queue.Full:
                with self._lock:
                    self.stats["queue_drops"] += 1

    def _handle_tier1(self, frame: Any, cctv_id: str, timestamp: float) -> None:
        """Tier 1: 프리필터 적용, 이상 시 Tier 3 승격."""
        with self._lock:
            self.stats["frames_t1"] += 1

        pf = self._prefilters.get(cctv_id)
        if pf is None:
            pf = PreFilter()
            self._prefilters[cctv_id] = pf

        result = pf.check(frame, timestamp)
        if result.is_anomaly:
            with self._lock:
                self.stats["anomalies_detected"] += 1
            logger.info(
                "이상 감지 [T1->T3]: %s score=%.2f reasons=%s",
                cctv_id[:20], result.score, result.reasons,
            )
            self._stream_manager.promote(cctv_id, to_tier=3)

    def _handle_tier23(
        self, frame: Any, cctv_id: str, tier: int, timestamp: float,
    ) -> None:
        """Tier 2/3: VisionPipeline 정밀 분석 + 트리거 + 녹화."""
        with self._lock:
            if tier == 2:
                self.stats["frames_t2"] += 1
            else:
                self.stats["frames_t3"] += 1

        pipeline = self._get_or_create_pipeline(cctv_id)
        if pipeline is None:
            return

        frame_idx = int(timestamp * STREAM_SAMPLE_FPS)
        try:
            result = pipeline.process_frame(frame, frame_idx, timestamp)
        except Exception as e:
            logger.error("Vision 오류 [%s]: %s", cctv_id[:20], e)
            return

        for trigger in result.get("triggers", []):
            self._handle_trigger(trigger, cctv_id)

        recorder = self._recorders.get(cctv_id)
        if recorder and recorder.is_recording:
            stopped, video_path = recorder.check_and_stop()
            if stopped:
                threading.Thread(
                    target=self._handle_recording_end,
                    args=(cctv_id, video_path),
                    daemon=True,
                ).start()

    # ── VisionPipeline 관리 ──────────────────────────────────────────

    def _get_or_create_pipeline(self, cctv_id: str) -> Any | None:
        """Tier 2/3 VisionPipeline lazy 초기화 (스레드 안전)."""
        pipeline = self._vision_pipelines.get(cctv_id)
        if pipeline is not None:
            return pipeline

        with self._lock:
            pipeline = self._vision_pipelines.get(cctv_id)
            if pipeline is not None:
                return pipeline

            total = len(self._vision_pipelines)
            limit = TIER2_HOTSPOT_COUNT + TIER3_MAX_CONCURRENT
            if total >= limit:
                logger.warning(
                    "VisionPipeline 상한 도달 (%d/%d) — %s 생성 거부",
                    total, limit, cctv_id[:20],
                )
                return None

            try:
                pipeline = _create_vision_pipeline()
            except Exception as e:
                logger.error(
                    "VisionPipeline 생성 실패 [%s]: %s", cctv_id[:20], e,
                )
                return None

            self._vision_pipelines[cctv_id] = pipeline

        logger.info(
            "VisionPipeline 생성: %s (총 %d개)",
            cctv_id[:20], len(self._vision_pipelines),
        )
        return pipeline

    def _cleanup_pipeline(self, cctv_id: str) -> None:
        """Tier 3 -> Tier 1 강등 시 파이프라인 + 녹화기 제거."""
        with self._lock:
            self._vision_pipelines.pop(cctv_id, None)
            self._recorders.pop(cctv_id, None)
            self._trigger_windows.pop(cctv_id, None)
            self._pending_triggers.pop(cctv_id, None)

    # ── 트리거 처리 ──────────────────────────────────────────────────

    def _handle_trigger(self, trigger: Any, cctv_id: str) -> None:
        """severity gate -> 합의 검사 -> 녹화 시작/연장."""
        with self._lock:
            self.stats["triggers"] += 1

        logger.info(
            "TRIGGER [%s] %s severity=%.2f: %s",
            trigger.type, cctv_id[:20], trigger.severity, trigger.description,
        )

        if trigger.type not in RECORD_TRIGGER_TYPES:
            return
        if trigger.severity < SEVERITY_GATE:
            return

        # Outbreak CCTV: 체류 연장 + 프레임 시퀀스 저장 (ffmpeg 녹화 불가)
        if cctv_id.startswith("OB_") and self._outbreak_feeder:
            self._outbreak_feeder.on_trigger(trigger, cctv_id)
            return

        recorder = self._recorders.get(cctv_id)
        if recorder is None:
            worker = self._stream_manager.streams.get(cctv_id)
            if worker is None:
                return
            recorder = OnDemandRecorder(worker.cctv)
            self._recorders[cctv_id] = recorder

        if recorder.is_recording:
            recorder.extend_recording(
                reason=f"추가 트리거 [{trigger.type}] {trigger.description}"
            )
            return

        if not recorder.can_record():
            return

        if not self._check_consensus(trigger, cctv_id):
            return

        event_id = f"NW_{trigger.type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{cctv_id[:10]}"
        started = recorder.start_recording(trigger.type, event_id)
        if started:
            with self._lock:
                self.stats["recordings"] += 1
                self._pending_triggers[cctv_id] = trigger
                self._trigger_windows[cctv_id] = []

    def _check_consensus(self, trigger: Any, cctv_id: str) -> bool:
        """다중 트리거 합의 검사."""
        if trigger.type in INSTANT_RECORD_TYPES:
            return True

        window = self._trigger_windows.get(cctv_id, [])
        cutoff = trigger.timestamp - CONSENSUS_WINDOW_SEC
        window = [t for t in window if t.timestamp > cutoff]
        window.append(trigger)
        self._trigger_windows[cctv_id] = window

        types_in_window = {t.type for t in window}
        return len(types_in_window) >= CONSENSUS_MIN_TYPES

    # ── 녹화 종료 처리 ───────────────────────────────────────────────

    def _handle_recording_end(self, cctv_id: str, video_path: Path | None) -> None:
        """ITS 교차확인 -> 보존/삭제."""
        trigger = self._pending_triggers.get(cctv_id)
        recorder = self._recorders.get(cctv_id)
        worker = self._stream_manager.streams.get(cctv_id)

        if not trigger or not worker:
            return

        cctv = worker.cctv
        event_id = recorder._event_id if recorder else "unknown"

        time.sleep(ITS_VERIFY_DELAY_SEC)

        confirmed, incident = self._verifier.verify(
            lat=cctv.latitude, lon=cctv.longitude,
            trigger_type=f"{trigger.type}_{cctv_id}",
        )

        if confirmed and incident and video_path:
            self._save_confirmed(event_id, trigger, cctv, incident, video_path)
        elif video_path and video_path.exists() and trigger.severity >= SEVERITY_PENDING:
            self._handle_pending(event_id, trigger, cctv, video_path)
        else:
            self._save_deleted(event_id, trigger, cctv, video_path)

        with self._lock:
            self._pending_triggers.pop(cctv_id, None)

    def _save_confirmed(
        self, event_id: str, trigger: Any, cctv: CCTVInfo,
        incident: IncidentEvent, video_path: Path,
    ) -> None:
        """사고 확인 -> 영상 보존."""
        with self._lock:
            self.stats["preserved"] += 1

        dist_km = (
            _haversine(
                cctv.latitude, cctv.longitude,
                incident.latitude or 0, incident.longitude or 0,
            )
            if incident.latitude and incident.longitude
            else None
        )

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
            video_size_mb=(
                video_path.stat().st_size / 1e6 if video_path.exists() else None
            ),
            its_verified=True,
            action="confirmed",
        )
        save_collection_record(record)
        logger.info("사고 확인 -> 보존: %s [%s]", video_path.name, cctv_id_short(cctv.cctv_id))

    def _handle_pending(
        self, event_id: str, trigger: Any, cctv: CCTVInfo, video_path: Path,
    ) -> None:
        """ITS 미확인 + 고severity -> pending 보관 + 재확인."""
        PENDING_DIR.mkdir(parents=True, exist_ok=True)
        pending_path = PENDING_DIR / video_path.name
        shutil.move(str(video_path), str(pending_path))
        logger.info(
            "Pending 보관: %s severity=%.2f [%s]",
            pending_path.name, trigger.severity, cctv_id_short(cctv.cctv_id),
        )

        for retry in range(1, PENDING_MAX_RETRIES + 1):
            if self._stop_event.wait(PENDING_RETRY_INTERVAL_SEC):
                return
            confirmed, incident = self._verifier.verify(
                lat=cctv.latitude, lon=cctv.longitude,
                trigger_type=f"pending_{trigger.type}_{cctv.cctv_id}",
            )
            if confirmed and incident:
                final_path = SAVE_DIR / pending_path.name
                shutil.move(str(pending_path), str(final_path))
                self._save_confirmed(event_id, trigger, cctv, incident, final_path)
                return

        if trigger.severity >= SEVERITY_FORCE_PRESERVE and pending_path.exists():
            final_path = SAVE_DIR / pending_path.name
            shutil.move(str(pending_path), str(final_path))
            with self._lock:
                self.stats["preserved"] += 1
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
                video_size_mb=(
                    final_path.stat().st_size / 1e6 if final_path.exists() else None
                ),
                its_verified=False,
                action="pending_preserved",
            )
            save_collection_record(record)
        else:
            self._save_deleted(event_id, trigger, cctv, pending_path)

    def _save_deleted(
        self, event_id: str, trigger: Any, cctv: CCTVInfo,
        video_path: Path | None,
    ) -> None:
        """미확인 -> 삭제."""
        with self._lock:
            self.stats["deleted"] += 1
        if video_path and video_path.exists():
            video_path.unlink()
            logger.info("미확인 -> 삭제: %s [%s]", video_path.name, cctv_id_short(cctv.cctv_id))

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

    # ── 핫스팟 선정 ──────────────────────────────────────────────────

    def _select_hotspots(self) -> set[str]:
        """Tier 2 핫스팟 CCTV ID 집합 반환.

        max_cameras가 설정되면 전체 비율(hotspot/total)을 유지하여
        T1 최소 1대, T2 최소 1대를 보장한다.
        """
        cctvs = self._cctv_client.list_cctvs()
        if not cctvs:
            return set()

        streamable = [c for c in cctvs if c.stream_url]
        total = len(streamable)
        if total == 0:
            return set()

        # max_cameras 제한 시 비율 유지
        if self._max_cameras > 0 and self._max_cameras < total:
            # 비율 계산: hotspot_count / total_streamable
            ratio = min(TIER2_HOTSPOT_COUNT / total, 1.0)
            n_hotspot = max(1, int(self._max_cameras * ratio))
            # T1도 최소 1대 확보
            n_hotspot = min(n_hotspot, self._max_cameras - 1)
            if n_hotspot < 1:
                n_hotspot = 1
        else:
            n_hotspot = min(TIER2_HOTSPOT_COUNT, total)

        selected = streamable[:n_hotspot]
        ids = {c.cctv_id for c in selected}

        logger.info(
            "핫스팟 선정: %d대 (총 %d대, 스트림 가능 %d대, max_cameras=%d)",
            len(ids), TIER2_HOTSPOT_COUNT, len(streamable),
            self._max_cameras,
        )
        return ids

    # ── 메인 루프 ────────────────────────────────────────────────────

    def start(self) -> None:
        """전국 파이프라인 시작."""
        self.stats["start_time"] = datetime.now().isoformat()

        # SIGINT 핸들러
        def signal_handler(sig, frame):
            logger.info("중단 신호 수신 -- 정리 중...")
            self._stop_event.set()

        signal.signal(signal.SIGINT, signal_handler)

        # Vision 워커 스레드 시작 (Tier 2/3 프레임 큐 처리)
        self._start_vision_workers()

        if not self._outbreak_only:
            # 핫스팟 선정
            hotspots = self._select_hotspots()

            # 스트림 시작
            logger.info("=" * 60)
            logger.info("전국 CCTV 3-Tier 파이프라인 시작")
            logger.info("  Tier 1 (프리필터): 프리필터만")
            logger.info("  Tier 2 (핫스팟):   %d대 상시 정밀", len(hotspots))
            logger.info("  Tier 3 (동적):     최대 %d대 정밀", TIER3_MAX_CONCURRENT)
            logger.info("  Vision 워커:       %d개", self._VISION_WORKERS)
            if self._max_cameras > 0:
                logger.info("  카메라 제한:       %d대", self._max_cameras)
            logger.info("=" * 60)

            started = self._stream_manager.start_all(
                on_frame=self._on_frame,
                hotspot_ids=hotspots,
                max_cameras=self._max_cameras,
            )
            if started == 0:
                logger.warning("StreamManager 스트림 0대 — Outbreak 경로만 가동")

            logger.info("스트림 %d대 시작 완료", started)

            # ITS 사고 반응기
            self._incident_reactor.start()
        else:
            logger.info("=" * 60)
            logger.info("Outbreak 전용 모드 (StreamManager/IncidentReactor 비활성)")
            logger.info("  Vision 워커: %d개", self._VISION_WORKERS)
            logger.info("=" * 60)

        # OutbreakFeeder 시작
        if OUTBREAK_ENABLED and _HAS_OUTBREAK:
            self._outbreak_feeder = OutbreakFeeder(
                pipeline=self,
                stop_event=self._stop_event,
            )
            if not self._outbreak_feeder.start():
                logger.warning("OutbreakFeeder 시작 실패 — 비활성화")
                self._outbreak_feeder = None

        # 초기 메모리 로깅
        self._log_memory("시작")

        # 상태 출력 루프
        try:
            while not self._stop_event.is_set():
                self._stop_event.wait(60)
                if self._stop_event.is_set():
                    break
                self._log_status()
                self._log_memory("주기적")
                self._save_status_file()
        except KeyboardInterrupt:
            self._stop_event.set()

        self.stop()

    def stop(self) -> None:
        """파이프라인 정지."""
        self._stop_event.set()

        if self._outbreak_feeder:
            logger.info("OutbreakFeeder 정지 중...")
            self._outbreak_feeder.stop()
            self._outbreak_feeder = None

        if not self._outbreak_only:
            logger.info("StreamManager 정지 중...")
            self._stream_manager.stop_all()

            logger.info("IncidentReactor 정지 중...")
            self._incident_reactor.stop()

        # Vision 워커 종료
        logger.info("Vision 워커 %d개 종료 중...", len(self._vision_threads))
        for _ in self._vision_threads:
            self._frame_queue.put(None)  # 종료 신호
        for t in self._vision_threads:
            t.join(timeout=5)
        self._vision_threads.clear()

        # 활성 녹화 강제 종료
        for cctv_id, recorder in list(self._recorders.items()):
            if recorder.is_recording:
                recorder.force_stop()
                logger.info("녹화 강제 종료: %s", cctv_id[:20])

        # 최종 메모리 로깅
        self._log_memory("종료")
        self._print_summary()

    # ── 메모리 프로파일링 ──────────────────────────────────────────────

    def _log_memory(self, label: str = "") -> float:
        """현재 프로세스 RSS 메모리를 로깅하고 MB 값을 반환."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            rss_mb = process.memory_info().rss / 1e6
        except ImportError:
            # psutil 없으면 /proc/self/status에서 직접 읽기
            try:
                with open("/proc/self/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            rss_mb = int(line.split()[1]) / 1024  # kB -> MB
                            break
                    else:
                        rss_mb = 0.0
            except Exception:
                rss_mb = 0.0

        n_prefilters = len(self._prefilters)
        n_pipelines = len(self._vision_pipelines)
        q_size = self._frame_queue.qsize()

        logger.info(
            "메모리 [%s]: %.0f MB (prefilters=%d, pipelines=%d, "
            "queue=%d/%d, drops=%d)",
            label, rss_mb, n_prefilters, n_pipelines,
            q_size, self._FRAME_QUEUE_SIZE,
            self.stats.get("queue_drops", 0),
        )
        return rss_mb

    # ── 통계/상태 ────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """전체 통계."""
        with self._lock:
            combined = dict(self.stats)

        combined["stream"] = self._stream_manager.get_stats()
        combined["reactor"] = self._incident_reactor.get_stats()
        combined["pipelines_active"] = len(self._vision_pipelines)
        combined["prefilters_active"] = len(self._prefilters)
        combined["recorders_active"] = sum(
            1 for r in self._recorders.values() if r.is_recording
        )
        combined["vision_queue_size"] = self._frame_queue.qsize()

        if self._outbreak_feeder:
            combined["outbreak"] = self._outbreak_feeder.get_stats()

        # 메모리 측정
        try:
            import psutil
            combined["memory_rss_mb"] = round(
                psutil.Process(os.getpid()).memory_info().rss / 1e6, 1,
            )
        except Exception:
            try:
                with open("/proc/self/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            combined["memory_rss_mb"] = round(
                                int(line.split()[1]) / 1024, 1,
                            )
                            break
            except Exception:
                combined["memory_rss_mb"] = None

        return combined

    def _log_status(self) -> None:
        """주기적 상태 출력."""
        s = self.get_stats()
        sm = s.get("stream", {})

        logger.info("-" * 50)
        logger.info(
            "요약: T1=%d T2=%d T3=%d | 이상=%d 트리거=%d 녹화=%d 보존=%d 삭제=%d",
            s["frames_t1"], s["frames_t2"], s["frames_t3"],
            s["anomalies_detected"], s["triggers"], s["recordings"],
            s["preserved"], s["deleted"],
        )
        logger.info(
            "스트림: total=%d active=%d disabled=%d zero_frame=%d "
            "| T1=%d T2=%d T3=%d",
            sm.get("total", 0), sm.get("active", 0), sm.get("disabled", 0),
            sm.get("zero_frame", 0),
            sm.get("tier_counts", {}).get(1, 0),
            sm.get("tier_counts", {}).get(2, 0),
            sm.get("tier_counts", {}).get(3, 0),
        )
        logger.info(
            "파이프라인: %d개 | 프리필터: %d개 | 녹화중: %d대 | "
            "메모리: %s MB | 큐: %d/%d (drop=%d)",
            s["pipelines_active"], s["prefilters_active"], s["recorders_active"],
            s.get("memory_rss_mb", "?"), s.get("vision_queue_size", 0),
            self._FRAME_QUEUE_SIZE, s.get("queue_drops", 0),
        )
        reactor = s.get("reactor", {})
        if reactor.get("active_reactions", 0) > 0:
            logger.info(
                "ITS 반응: %d건 활성 (CCTV %d대 편입)",
                reactor["active_reactions"], reactor["active_cctvs"],
            )
        ob = s.get("outbreak", {})
        if ob:
            logger.info(
                "Outbreak: 순찰=%d 체류=%d 주입=%d 트리거=%d 녹화=%d",
                ob.get("patrols", 0), ob.get("dwells", 0),
                ob.get("frames_injected", 0), ob.get("triggers_received", 0),
                ob.get("recordings_saved", 0),
            )
        logger.info("-" * 50)

    def _save_status_file(self) -> None:
        """실시간 상태 JSON."""
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        s = self.get_stats()
        s["updated_at"] = datetime.now().isoformat()
        try:
            with open(NATIONWIDE_STATUS_FILE, "w", encoding="utf-8") as f:
                json.dump(s, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("상태 파일 저장 실패: %s", e)

    def _print_summary(self) -> None:
        """세션 요약."""
        s = self.get_stats()
        sm = s.get("stream", {})

        print()
        print("=" * 60)
        print("전국 3-Tier 파이프라인 세션 요약")
        print("=" * 60)
        print(f"  시작     : {s['start_time']}")
        print(f"  스트림   : {sm.get('total', 0)}대 (active={sm.get('active', 0)})")
        print(f"  T1 프레임: {s['frames_t1']}")
        print(f"  T2 프레임: {s['frames_t2']}")
        print(f"  T3 프레임: {s['frames_t3']}")
        print(f"  이상감지 : {s['anomalies_detected']}건")
        print(f"  트리거   : {s['triggers']}건")
        print(f"  녹화시작 : {s['recordings']}건")
        print(f"  보존     : {s['preserved']}건")
        print(f"  삭제     : {s['deleted']}건")
        print(f"  저장경로 : {SAVE_DIR}")
        print("=" * 60)

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        summary_file = LOG_DIR / f"nationwide_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            s["ended_at"] = datetime.now().isoformat()
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(s, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


def cctv_id_short(cctv_id: str) -> str:
    """CCTV ID 축약 표시."""
    return cctv_id[:20] if len(cctv_id) > 20 else cctv_id


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse

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
    # 노이즈 억제: 개별 스트림/프레임/파이프라인 → 에러만 (요약 리포트에서 집계 확인)
    for name in ("run_realtime", "stream", "layer1_vision.vision_pipeline"):
        logging.getLogger(name).setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(
        description="전국 363대 CCTV 3-Tier 통합 파이프라인",
    )
    sub = parser.add_subparsers(dest="command")

    p_start = sub.add_parser("start", help="전국 3-Tier 파이프라인 시작")
    p_start.add_argument(
        "--max-cameras", type=int, default=0,
        help=f"카메라 수 제한 (0=기본값 {NATIONWIDE_MAX_CAMERAS}대)",
    )
    p_start.add_argument(
        "--outbreak-only", action="store_true",
        help="돌발정보 CCTV만 가동 (StreamManager/IncidentReactor 비활성)",
    )

    sub.add_parser("status", help="수집 현황 확인")

    args = parser.parse_args()

    if args.command == "start":
        pipeline = NationwidePipeline(
            max_cameras=args.max_cameras,
            outbreak_only=args.outbreak_only,
        )
        pipeline.start()
    elif args.command == "status":
        from run_realtime import RealtimeAccidentPipeline
        RealtimeAccidentPipeline().status()
    else:
        parser.print_help()
        print()
        print("예시:")
        print("  python run_nationwide.py start                    # 전국 363대 시작")
        print("  python run_nationwide.py start --max-cameras 10   # 테스트 10대")
        print("  python run_nationwide.py start --outbreak-only    # 돌발정보만")
        print("  python run_nationwide.py status                   # 수집 현황")


if __name__ == "__main__":
    main()
