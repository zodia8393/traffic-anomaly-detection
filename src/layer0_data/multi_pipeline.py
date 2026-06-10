"""멀티 CCTV 실시간 파이프라인 (run_realtime 분해)."""
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
from ondemand_recorder import OnDemandRecorder, save_collection_record
from camera_worker import CameraWorker, CameraStats
from hotspot import HotspotSelector, IncidentGrouper
from incident_verifier import IncidentVerifier


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

