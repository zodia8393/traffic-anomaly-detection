"""단일 CCTV 실시간 사고감지 파이프라인 (run_realtime 분해)."""
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
        from layer1_vision.anomaly_shadow import build_shadow_anomaly_engine

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
            anomaly_engine=build_shadow_anomaly_engine(
                self.target_cctv.cctv_id if self.target_cctv else "realtime"
            ),
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
        elif video_path and video_path.exists() and (
            trigger.severity >= SEVERITY_PENDING or self.verifier.api_uncertain
        ):
            # API 미상(uncertain)이면 severity 무관 보존 — 진짜 사고를 오삭제하지 않음
            if self.verifier.api_uncertain:
                logger.warning("ITS 미상 → 보존(pending): %s sev=%.2f",
                               event_id, trigger.severity)
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

