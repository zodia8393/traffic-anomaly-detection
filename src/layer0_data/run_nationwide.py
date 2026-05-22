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
import signal
import shutil
import sys
import threading
import time
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

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("config", str(PIPELINE_SRC / "config.py"))
_pipeline_cfg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pipeline_cfg)
sys.modules["config"] = _pipeline_cfg
sys.path.insert(0, str(PIPELINE_SRC))

from config_new import (
    TIER2_HOTSPOT_COUNT,
    TIER3_MAX_CONCURRENT,
    STREAM_SAMPLE_FPS,
    PREFILTER_COOLDOWN_SEC,
    PREFILTER_TIER3_HOLD_SEC,
    INCIDENT_POLL_SEC,
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


class NationwidePipeline:
    """363대 전국 CCTV 3-Tier 통합 파이프라인."""

    def __init__(self, max_cameras: int = 0):
        self._stop_event = threading.Event()
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

        self._prefilters: dict[str, PreFilter] = {}
        self._vision_pipelines: dict[str, Any] = {}
        self._recorders: dict[str, OnDemandRecorder] = {}
        self._verifier = IncidentVerifier()
        self._trigger_windows: dict[str, list] = {}
        self._pending_triggers: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._max_cameras = max_cameras

        self.stats = {
            "frames_t1": 0,
            "frames_t2": 0,
            "frames_t3": 0,
            "triggers": 0,
            "recordings": 0,
            "preserved": 0,
            "deleted": 0,
            "anomalies_detected": 0,
            "start_time": None,
        }

    # ── 콜백 ─────────────────────────────────────────────────────────

    def _on_frame(self, frame: Any, cctv_id: str, tier: int) -> None:
        """StreamManager 프레임 도착 콜백."""
        timestamp = time.monotonic()

        if tier == 1:
            # Tier 3 -> 1 강등 후 첫 프레임: 파이프라인/녹화기 메모리 회수
            if cctv_id in self._vision_pipelines:
                self._cleanup_pipeline(cctv_id)
            self._handle_tier1(frame, cctv_id, timestamp)
        elif tier in (2, 3):
            self._handle_tier23(frame, cctv_id, tier, timestamp)

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
        """Tier 2/3 VisionPipeline lazy 초기화."""
        pipeline = self._vision_pipelines.get(cctv_id)
        if pipeline is not None:
            return pipeline

        with self._lock:
            # 이중 검사
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
            logger.error("VisionPipeline 생성 실패 [%s]: %s", cctv_id[:20], e)
            return None

        with self._lock:
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
        """Tier 2 핫스팟 CCTV ID 집합 반환."""
        cctvs = self._cctv_client.list_cctvs()
        if not cctvs:
            return set()

        streamable = [c for c in cctvs if c.stream_url]
        selected = streamable[:TIER2_HOTSPOT_COUNT]
        ids = {c.cctv_id for c in selected}

        logger.info(
            "핫스팟 선정: %d/%d대 (스트림 가능 %d대)",
            len(ids), TIER2_HOTSPOT_COUNT, len(streamable),
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

        # 핫스팟 선정
        hotspots = self._select_hotspots()

        # 스트림 시작
        logger.info("=" * 60)
        logger.info("전국 CCTV 3-Tier 파이프라인 시작")
        logger.info("  Tier 1 (프리필터): 프리필터만")
        logger.info("  Tier 2 (핫스팟):   %d대 상시 정밀", len(hotspots))
        logger.info("  Tier 3 (동적):     최대 %d대 정밀", TIER3_MAX_CONCURRENT)
        if self._max_cameras > 0:
            logger.info("  카메라 제한:       %d대", self._max_cameras)
        logger.info("=" * 60)

        started = self._stream_manager.start_all(
            on_frame=self._on_frame,
            hotspot_ids=hotspots,
            max_cameras=self._max_cameras,
        )
        if started == 0:
            logger.error("스트림 시작 실패 -- 종료")
            return

        logger.info("스트림 %d대 시작 완료", started)

        # ITS 사고 반응기
        self._incident_reactor.start()

        # 상태 출력 루프
        try:
            while not self._stop_event.is_set():
                self._stop_event.wait(60)
                if self._stop_event.is_set():
                    break
                self._log_status()
                self._save_status_file()
        except KeyboardInterrupt:
            self._stop_event.set()

        self.stop()

    def stop(self) -> None:
        """파이프라인 정지."""
        self._stop_event.set()

        logger.info("StreamManager 정지 중...")
        self._stream_manager.stop_all()

        logger.info("IncidentReactor 정지 중...")
        self._incident_reactor.stop()

        # 활성 녹화 강제 종료
        for cctv_id, recorder in list(self._recorders.items()):
            if recorder.is_recording:
                recorder.force_stop()
                logger.info("녹화 강제 종료: %s", cctv_id[:20])

        self._print_summary()

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
            "스트림: total=%d active=%d disabled=%d | T1=%d T2=%d T3=%d",
            sm.get("total", 0), sm.get("active", 0), sm.get("disabled", 0),
            sm.get("tier_counts", {}).get(1, 0),
            sm.get("tier_counts", {}).get(2, 0),
            sm.get("tier_counts", {}).get(3, 0),
        )
        logger.info(
            "파이프라인: %d개 | 프리필터: %d개 | 녹화중: %d대",
            s["pipelines_active"], s["prefilters_active"], s["recorders_active"],
        )
        reactor = s.get("reactor", {})
        if reactor.get("active_reactions", 0) > 0:
            logger.info(
                "ITS 반응: %d건 활성 (CCTV %d대 편입)",
                reactor["active_reactions"], reactor["active_cctvs"],
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

    parser = argparse.ArgumentParser(
        description="전국 363대 CCTV 3-Tier 통합 파이프라인",
    )
    sub = parser.add_subparsers(dest="command")

    p_start = sub.add_parser("start", help="전국 3-Tier 파이프라인 시작")
    p_start.add_argument(
        "--max-cameras", type=int, default=0,
        help="테스트용 카메라 수 제한 (0=전체)",
    )

    sub.add_parser("status", help="수집 현황 확인")

    args = parser.parse_args()

    if args.command == "start":
        pipeline = NationwidePipeline(max_cameras=args.max_cameras)
        pipeline.start()
    elif args.command == "status":
        from run_realtime import RealtimeAccidentPipeline
        RealtimeAccidentPipeline().status()
    else:
        parser.print_help()
        print()
        print("예시:")
        print("  python run_nationwide.py start                  # 전국 363대 시작")
        print("  python run_nationwide.py start --max-cameras 10 # 테스트 10대")
        print("  python run_nationwide.py status                 # 수집 현황")


if __name__ == "__main__":
    main()
