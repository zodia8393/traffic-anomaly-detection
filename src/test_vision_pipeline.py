"""Step 3: Vision 파이프라인 실데이터 테스트.

AI Hub #71566 비정상 클립(안전거리 미확보 차선변경)의 PNG 프레임을
VisionPipeline에 통과시켜 트리거 7종 발화 여부를 검증한다.

분류기 모델은 미설치 상태이므로 DummyClassifier를 사용한다.
(검증 대상: 검출 → 추적 → 속도 → TTC → 트리거, 분류 정확도는 대상 아님)
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

# 경로 설정: pipeline/src + 사고분석/src 모두 필요
PIPELINE_SRC = Path("/workspace/prj_cctv/pipeline/src")
ACCIDENT_SRC = Path("/workspace/prj_cctv/사고분석_설계/src")
sys.path.insert(0, str(PIPELINE_SRC))
sys.path.insert(0, str(ACCIDENT_SRC))

import cv2
import numpy as np

from detector import VehicleDetector
from tracker import VehicleTracker
from layer1_vision.speed_estimator import SpeedEstimator
from layer1_vision.trigger_detector import TriggerDetector
from layer1_vision.ttc_calculator import compute_all_ttc
from layer1_vision.vision_pipeline import VisionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("test_vision")


class DummyClassifier:
    """분류 모델 없이 파이프라인을 테스트하기 위한 더미."""

    def predict_batch(self, crops: list[np.ndarray]) -> tuple[list[str], list[float]]:
        return ["T1"] * len(crops), [0.5] * len(crops)


def load_frames(clip_dir: Path) -> list[tuple[int, np.ndarray]]:
    """PNG 프레임을 번호순 로드. (idx, frame) 쌍 반환."""
    png_files = sorted(clip_dir.glob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"PNG 파일 없음: {clip_dir}")
    frames = []
    for p in png_files:
        img = cv2.imread(str(p))
        if img is not None:
            idx = int(p.stem.split("_")[-1])
            frames.append((idx, img))
    logger.info("프레임 %d장 로드 (from %s)", len(frames), clip_dir.name)
    return frames


def main():
    clip_dir = Path(
        "/DATA/aihub_71566/source/val/비정상"
        "/07.안전거리 미확보 차선변경"
        "/p01_20230107_141213_an7_026_04"
    )

    frames = load_frames(clip_dir)
    h, w = frames[0][1].shape[:2]
    fps = 1.0  # AI Hub PNG는 ~1fps 추정

    # 컴포넌트 초기화
    logger.info("Detector 초기화...")
    detector = VehicleDetector()
    logger.info("Detector: %s", detector.model_path)

    tracker = VehicleTracker(fps=max(1, int(fps)))
    classifier = DummyClassifier()
    speed_est = SpeedEstimator()
    trigger_det = TriggerDetector()

    pipeline = VisionPipeline(
        detector=detector,
        tracker=tracker,
        classifier=classifier,
        speed_est=speed_est,
        trigger_det=trigger_det,
        fps=fps,
    )

    all_triggers = []
    det_counts = []

    for frame_idx, frame in frames:
        timestamp = frame_idx * (1.0 / fps)
        result = pipeline.process_frame(frame, frame_idx, timestamp)

        n_det = len(result["detections"])
        n_tracked = len(result["tracked_vehicles"])
        triggers = result["triggers"]
        det_counts.append(n_det)

        if triggers:
            for t in triggers:
                all_triggers.append(t)
                logger.info(
                    ">>> TRIGGER %s @ frame %d: %s (severity=%.2f)",
                    t.type, t.frame_idx, t.description, t.severity,
                )

        if frame_idx % 10 == 0 or frame_idx == frames[-1][0]:
            logger.info(
                "Frame %03d: det=%d tracked=%d triggers=%d",
                frame_idx, n_det, n_tracked, len(triggers),
            )

    # 결과 요약
    print("\n" + "=" * 60)
    print("Vision Pipeline 테스트 결과")
    print("=" * 60)
    print(f"클립: {clip_dir.name}")
    print(f"프레임: {len(frames)}장 ({w}x{h})")
    print(f"검출 합계: {sum(det_counts)} (평균 {np.mean(det_counts):.1f}/frame)")
    print(f"트리거 발화: {len(all_triggers)}건")
    for t in all_triggers:
        print(f"  [{t.type}] frame={t.frame_idx} severity={t.severity:.2f} — {t.description}")
    if not all_triggers:
        print("  (트리거 없음)")
    print("=" * 60)


if __name__ == "__main__":
    main()
