"""Step 4: E2E 데모 — 트리거 → 키프레임 → MLLM → JSON.

Vision 파이프라인에서 트리거 발화 → 해당 프레임을 MLLM에 전달 →
사고 판단 JSON 응답을 받는 전체 경로를 검증한다.
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import tempfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

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
from layer1_vision.vision_pipeline import VisionPipeline
from layer3_mllm.mllm_client import MLLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("e2e_demo")


class DummyClassifier:
    def predict_batch(self, crops):
        return ["T1"] * len(crops), [0.5] * len(crops)


def load_frames(clip_dir: Path) -> list[tuple[int, np.ndarray]]:
    png_files = sorted(clip_dir.glob("*.png"))
    frames = []
    for p in png_files:
        img = cv2.imread(str(p))
        if img is not None:
            idx = int(p.stem.split("_")[-1])
            frames.append((idx, img))
    return frames


ACCIDENT_PROMPT = """이 CCTV 고속도로 영상을 분석하여 다음 JSON 형식으로 응답하세요.
트리거 정보: {trigger_type} — {trigger_desc}

응답 형식:
{{
  "scene": "장면 설명 (1~2문장)",
  "anomaly": true/false,
  "anomaly_type": "사고유형 (anomaly=true일 때만)",
  "vehicles": [{{"type": "차종", "color": "색상", "behavior": "행동"}}],
  "severity": "low/medium/high",
  "confidence": 0.0~1.0
}}"""


def main():
    clip_dir = Path(
        "/DATA/aihub_71566/source/val/비정상"
        "/07.안전거리 미확보 차선변경"
        "/p01_20230107_141213_an7_026_04"
    )

    frames = load_frames(clip_dir)
    logger.info("프레임 %d장 로드", len(frames))

    # 프레임 인덱스 → 이미지 매핑 (키프레임 추출용)
    frame_map = {idx: img for idx, img in frames}
    h, w = frames[0][1].shape[:2]
    fps = 1.0

    # Vision 파이프라인 초기화
    detector = VehicleDetector()
    tracker = VehicleTracker(fps=max(1, int(fps)))
    pipeline = VisionPipeline(
        detector=detector,
        tracker=tracker,
        classifier=DummyClassifier(),
        speed_est=SpeedEstimator(),
        trigger_det=TriggerDetector(),
        fps=fps,
    )

    # MLLM 클라이언트 초기화 (transformers backend)
    logger.info("MLLM 초기화 (transformers, 최초 로딩 시간 소요)...")
    mllm = MLLMClient(backend="transformers")

    first_trigger = None
    trigger_frame_img = None

    # Phase 1: 전체 프레임 처리하여 첫 트리거 포착
    logger.info("Vision 파이프라인 실행 중...")
    for frame_idx, frame in frames:
        timestamp = frame_idx * (1.0 / fps)
        result = pipeline.process_frame(frame, frame_idx, timestamp)

        if result["triggers"] and first_trigger is None:
            first_trigger = result["triggers"][0]
            trigger_frame_img = frame
            logger.info(
                "첫 트리거 포착: [%s] frame=%d — %s",
                first_trigger.type, first_trigger.frame_idx, first_trigger.description,
            )

    if first_trigger is None:
        logger.warning("트리거 발화 없음 — E2E 데모 중단")
        return

    # Phase 2: 키프레임 선정
    from layer1_vision.keyframe_selector import select_keyframes
    kf_indices = select_keyframes(
        track_buffer=pipeline._frame_buffer,
        trigger_frame=first_trigger.frame_idx,
        fps=fps,
        frame_w=w,
        frame_h=h,
    )
    logger.info("키프레임 선정: %s", kf_indices)

    # 실제 프레임 이미지 수집 (가용한 것만)
    keyframes = []
    for ki in kf_indices:
        if ki in frame_map:
            keyframes.append(frame_map[ki])
    if not keyframes:
        keyframes = [trigger_frame_img]
    logger.info("키프레임 %d장 확보", len(keyframes))

    # Phase 3: MLLM 호출 (첫 번째 키프레임 사용)
    prompt = ACCIDENT_PROMPT.format(
        trigger_type=first_trigger.type,
        trigger_desc=first_trigger.description,
    )

    messages = [{"role": "user", "content": prompt}]
    img_rgb = cv2.cvtColor(keyframes[0], cv2.COLOR_BGR2RGB)

    logger.info("MLLM 요청 전송 (CPU 추론, ~40초 소요)...")
    t0 = time.time()
    response = mllm.chat(messages, images=[img_rgb])
    latency = time.time() - t0

    # Phase 4: 결과 출력
    print("\n" + "=" * 60)
    print("E2E 데모 결과")
    print("=" * 60)
    print(f"클립: {clip_dir.name}")
    print(f"트리거: [{first_trigger.type}] {first_trigger.description}")
    print(f"키프레임: {kf_indices}")
    print(f"MLLM 응답 시간: {latency:.1f}s")
    print(f"\nMLLM 응답:")
    print(json.dumps(response, ensure_ascii=False, indent=2))
    print("=" * 60)

    # 결과 저장
    output_dir = Path("/workspace/prj_cctv/사고분석_설계/src/layer1_vision/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "e2e_demo_result.json"
    result_data = {
        "clip": clip_dir.name,
        "trigger": {
            "type": first_trigger.type,
            "frame_idx": first_trigger.frame_idx,
            "description": first_trigger.description,
            "severity": first_trigger.severity,
        },
        "keyframe_indices": kf_indices,
        "mllm_latency_sec": round(latency, 1),
        "mllm_response": response,
    }
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {result_file}")


if __name__ == "__main__":
    main()
