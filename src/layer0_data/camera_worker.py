"""카메라별 독립 감시 워커 (run_realtime 분해)."""
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
from incident_verifier import IncidentVerifier


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

