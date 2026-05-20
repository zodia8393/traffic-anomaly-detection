"""Layer 1 통합 실행기.

기존 detector/tracker/classifier를 조합하여 매 프레임을 처리하고,
트리거 발생 시 MLLM 입력 패키지를 생성한다.

매 프레임 흐름:
  detect -> track -> classify(1차) -> speed -> ttc -> trigger check -> anomaly
"""

from __future__ import annotations
import sys as _sys; from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import logging
from pathlib import Path
from typing import Any

import json

import cv2
import numpy as np

from config_new import VEHICLE_13CLASS, L1_KEYFRAMES, L1_PACKAGES
from keyframe_selector import select_keyframes
from speed_estimator import SpeedEstimator
from trigger_detector import TriggerDetector, TriggerEvent
from ttc_calculator import compute_all_ttc

# AnomalyEngine: 선택적 의존성 — 없어도 기존 파이프라인 동작
try:
    from anomaly_engine import AnomalyEngine, FrameResult as AnomalyFrameResult
    _HAS_ANOMALY_ENGINE = True
except ImportError:
    _HAS_ANOMALY_ENGINE = False
    AnomalyEngine = None  # type: ignore[misc,assignment]
    AnomalyFrameResult = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


class VisionPipeline:
    """Layer 1 비전 파이프라인 통합 실행기."""

    def __init__(
        self,
        detector: Any,
        tracker: Any,
        classifier: Any,
        speed_est: SpeedEstimator,
        ttc_calc: Any = None,
        trigger_det: TriggerDetector | None = None,
        keyframe_sel: Any = None,
        fps: float = 1.0,
        anomaly_engine: AnomalyEngine | None = None,
    ) -> None:
        """컴포넌트 주입.

        Args:
            detector: VehicleDetector 인스턴스 (detect_frame 메서드).
            tracker: VehicleTracker 인스턴스 (update 메서드).
            classifier: EnsembleClassifier 인스턴스 (predict_batch 메서드).
            speed_est: SpeedEstimator 인스턴스.
            ttc_calc: 미사용 (모듈 함수 직접 호출). 호환성 유지.
            trigger_det: TriggerDetector 인스턴스.
            keyframe_sel: 미사용 (모듈 함수 직접 호출). 호환성 유지.
            fps: 영상 FPS.
            anomaly_engine: AnomalyEngine 인스턴스 (선택적). None이면 이상징후 탐지 비활성.
        """
        self.detector = detector
        self.tracker = tracker
        self.classifier = classifier
        self.speed_est = speed_est
        self.trigger_det = trigger_det or TriggerDetector()
        self.fps = fps
        self.anomaly_engine = anomaly_engine

        # 프레임별 트랙 기록: {track_id: [{"bbox":..., "timestamp":..., "center":...}]}
        self._track_history: dict[int, list[dict[str, Any]]] = {}
        # 트랙별 속도 이력: {track_id: [speed_kmh, ...]}
        self._speed_history: dict[int, list[float]] = {}
        # 키프레임 선정용 버퍼
        self._frame_buffer: list[dict[str, Any]] = []
        self._frame_buffer_max = 600  # 최대 10분 (1fps 기준)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
    ) -> dict[str, Any]:
        """단일 프레임 처리.

        Args:
            frame: BGR 이미지.
            frame_idx: 프레임 인덱스.
            timestamp: 타임스탬프(초).

        Returns:
            {
                "frame_idx": int,
                "timestamp": float,
                "detections": list,
                "tracked_vehicles": list,
                "classifications": list,
                "triggers": list[TriggerEvent],
                "anomaly": FrameResult | None,
            }
        """
        # 1. 검출
        raw_dets = self.detector.detect_frame(frame)

        # 2. 트래커 입력 형식 변환
        det_dicts = [
            {"bbox": [d[0], d[1], d[2], d[3]], "conf": d[4], "coco_cls": d[5]}
            for d in raw_dets
        ]
        tracked = self.tracker.update(det_dicts, timestamp, frame=frame)

        # 3. 트래킹 결과에서 crop + 분류
        tracked_vehicles: list[dict[str, Any]] = []
        crops: list[np.ndarray] = []
        crop_indices: list[int] = []

        if tracked.tracker_id is not None:
            for i, tid in enumerate(tracked.tracker_id):
                tid = int(tid)
                x1, y1, x2, y2 = tracked.xyxy[i].tolist()
                ix1, iy1 = max(0, int(x1)), max(0, int(y1))
                ix2, iy2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)

                crop = frame[iy1:iy2, ix1:ix2]
                if crop.size > 0:
                    crops.append(crop)
                    crop_indices.append(len(tracked_vehicles))

                vehicle_info = {
                    "track_id": tid,
                    "bbox": [x1, y1, x2, y2],
                    "center": (cx, cy),
                    "area": area,
                    "conf": float(tracked.confidence[i]) if tracked.confidence is not None else 0.0,
                    "cls": None,
                    "cls_conf": 0.0,
                }
                tracked_vehicles.append(vehicle_info)

                # 트랙 히스토리 갱신
                if tid not in self._track_history:
                    self._track_history[tid] = []
                self._track_history[tid].append({
                    "bbox": [x1, y1, x2, y2],
                    "timestamp": timestamp,
                    "center": (cx, cy),
                })
                # 메모리 제한: 트랙당 최대 300프레임
                if len(self._track_history[tid]) > 300:
                    self._track_history[tid] = self._track_history[tid][-300:]

        # 4. 분류
        classifications: list[dict[str, Any]] = []
        if crops:
            preds, confs = self.classifier.predict_batch(crops)
            for idx_in_crops, (pred, conf) in enumerate(zip(preds, confs)):
                vehicle_idx = crop_indices[idx_in_crops]
                tracked_vehicles[vehicle_idx]["cls"] = pred
                tracked_vehicles[vehicle_idx]["cls_conf"] = conf
                cls_name = VEHICLE_13CLASS.get(pred, pred)
                classifications.append({
                    "track_id": tracked_vehicles[vehicle_idx]["track_id"],
                    "class": pred,
                    "class_name": cls_name,
                    "confidence": conf,
                })

        # 5. 속도 추정
        for v in tracked_vehicles:
            tid = v["track_id"]
            history = self._track_history.get(tid, [])
            if len(history) >= 2:
                positions = [(h["center"][0], h["center"][1]) for h in history]
                ts_list = [h["timestamp"] for h in history]
                speeds = self.speed_est.estimate(positions, ts_list)
                if speeds:
                    if tid not in self._speed_history:
                        self._speed_history[tid] = []
                    self._speed_history[tid].append(speeds[-1])
                    # 메모리 제한
                    if len(self._speed_history[tid]) > 300:
                        self._speed_history[tid] = self._speed_history[tid][-300:]

        # 6. TTC 계산
        ttc_list = compute_all_ttc(self._track_history)

        # 7. 트리거 판정
        tracks_info = {
            v["track_id"]: {
                "bbox": v["bbox"],
                "center": v["center"],
                "cls": v["cls"],
            }
            for v in tracked_vehicles
        }
        triggers = self.trigger_det.update(
            frame_idx, timestamp, tracks_info, ttc_list, self._speed_history,
        )

        if triggers:
            for t in triggers:
                logger.info("TRIGGER [%s] frame=%d: %s", t.type, frame_idx, t.description)

        # 8. 이상징후 탐지 (AnomalyEngine, 선택적)
        anomaly_result = None
        if self.anomaly_engine is not None:
            try:
                anomaly_result = self.anomaly_engine.process_frame(
                    timestamp=timestamp,
                    tracked_vehicles=tracked_vehicles,
                    ttc_list=ttc_list,
                    dt=1.0 / self.fps if self.fps > 0 else 1.0,
                )
                if anomaly_result.alert is not None:
                    logger.info(
                        "ANOMALY ALERT frame=%d score=%.3f: %s",
                        frame_idx,
                        anomaly_result.ensemble.final_score,
                        anomaly_result.alert.description
                        if hasattr(anomaly_result.alert, "description")
                        else "alert",
                    )
            except Exception:
                logger.exception("AnomalyEngine 실행 실패 (frame=%d)", frame_idx)

        # 9. 키프레임 버퍼 갱신
        buffer_entry = {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "tracks": [
                {"bbox": v["bbox"], "conf": v["conf"], "track_id": v["track_id"]}
                for v in tracked_vehicles
            ],
        }
        self._frame_buffer.append(buffer_entry)
        if len(self._frame_buffer) > self._frame_buffer_max:
            self._frame_buffer = self._frame_buffer[-self._frame_buffer_max:]

        return {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "detections": raw_dets,
            "tracked_vehicles": tracked_vehicles,
            "classifications": classifications,
            "triggers": triggers,
            "anomaly": anomaly_result,
        }

    def get_mllm_input_package(
        self,
        trigger_event: TriggerEvent,
        video_path: str | Path,
        frame_w: float = 1920.0,
        frame_h: float = 1080.0,
    ) -> dict[str, Any]:
        """MLLM 입력 패키지 생성.

        트리거 이벤트에 대해 키프레임을 선정하고,
        해당 프레임 이미지 + 메타데이터를 패키지로 구성한다.

        Args:
            trigger_event: 발생한 트리거 이벤트.
            video_path: 원본 영상 경로 (키프레임 추출용).
            frame_w: 프레임 너비.
            frame_h: 프레임 높이.

        Returns:
            {
                "trigger": TriggerEvent,
                "keyframe_indices": list[int],
                "keyframes": list[np.ndarray],
                "metadata": {
                    "video_path": str,
                    "fps": float,
                    "involved_tracks": list[dict],
                    "ttc_info": list[dict],
                    "speed_info": dict[int, float],
                },
            }
        """
        # 키프레임 선정
        keyframe_indices = select_keyframes(
            track_buffer=self._frame_buffer,
            trigger_frame=trigger_event.frame_idx,
            fps=self.fps,
            frame_w=frame_w,
            frame_h=frame_h,
        )

        # 키프레임 이미지 추출
        keyframes: list[np.ndarray] = []
        video_path = Path(video_path)
        if video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            try:
                for fidx in keyframe_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                    ret, frame = cap.read()
                    if ret:
                        keyframes.append(frame)
                    else:
                        logger.warning("키프레임 추출 실패: frame=%d", fidx)
            finally:
                cap.release()
        else:
            logger.warning("영상 파일 없음: %s", video_path)

        # 관련 트랙 정보 수집
        involved_info: list[dict[str, Any]] = []
        for tid in trigger_event.involved_tracks:
            track_data = self._track_history.get(tid, [])
            speeds = self._speed_history.get(tid, [])
            involved_info.append({
                "track_id": tid,
                "track_length": len(track_data),
                "last_bbox": track_data[-1]["bbox"] if track_data else None,
                "last_speed_kmh": speeds[-1] if speeds else None,
            })

        # 현재 TTC 정보
        ttc_info = compute_all_ttc(self._track_history)
        # 트리거 관련 트랙의 TTC만 필터
        relevant_ttc = [
            t for t in ttc_info
            if t["track_a"] in trigger_event.involved_tracks
            or t["track_b"] in trigger_event.involved_tracks
        ]

        # 관련 트랙 최신 속도
        speed_info = {
            tid: self._speed_history[tid][-1]
            for tid in trigger_event.involved_tracks
            if tid in self._speed_history and self._speed_history[tid]
        }

        return {
            "trigger": trigger_event,
            "keyframe_indices": keyframe_indices,
            "keyframes": keyframes,
            "metadata": {
                "video_path": str(video_path),
                "fps": self.fps,
                "involved_tracks": involved_info,
                "ttc_info": relevant_ttc,
                "speed_info": speed_info,
            },
        }

    def save_keyframes(
        self,
        keyframes: list[np.ndarray],
        trigger_event: TriggerEvent,
    ) -> list[Path]:
        """키프레임 이미지를 L1_KEYFRAMES에 저장."""
        L1_KEYFRAMES.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        for i, img in enumerate(keyframes):
            fname = f"kf_{trigger_event.type}_{trigger_event.frame_idx}_{i}.jpg"
            fpath = L1_KEYFRAMES / fname
            cv2.imwrite(str(fpath), img)
            saved.append(fpath)
        logger.info("키프레임 %d장 저장 → %s", len(saved), L1_KEYFRAMES)
        return saved

    def save_package(
        self,
        package: dict[str, Any],
    ) -> Path:
        """MLLM 입력 패키지 메타데이터를 L1_PACKAGES에 JSON 저장."""
        L1_PACKAGES.mkdir(parents=True, exist_ok=True)
        trigger = package["trigger"]
        fname = f"pkg_{trigger.type}_{trigger.frame_idx}.json"
        fpath = L1_PACKAGES / fname

        serializable = {
            "trigger_type": trigger.type,
            "trigger_frame": trigger.frame_idx,
            "trigger_timestamp": trigger.timestamp,
            "trigger_description": trigger.description,
            "involved_tracks": trigger.involved_tracks,
            "keyframe_indices": package["keyframe_indices"],
            "metadata": {
                k: v for k, v in package["metadata"].items()
                if k != "keyframes"
            },
        }
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
        logger.info("패키지 저장 → %s", fpath)
        return fpath
