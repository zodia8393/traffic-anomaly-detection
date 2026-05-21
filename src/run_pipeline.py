"""E2E 통합 러너: 클립 디렉토리 → L1 Vision → L3 MLLM → L2 DuckDB.

Usage:
    python3 run_pipeline.py /DATA/aihub_71566/source/val/비정상/07.../clip_dir/
    python3 run_pipeline.py --clip-dir /path/to/clip --video-id my_clip_001
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import logging
import argparse
from datetime import datetime
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
from layer2_metadata.metadata_writer import MetadataWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("pipeline")


class DummyClassifier:
    def predict_batch(self, crops):
        return ["T1"] * len(crops), [0.5] * len(crops)


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


def load_frames(clip_dir: Path) -> list[tuple[int, np.ndarray]]:
    png_files = sorted(clip_dir.glob("*.png"))
    if not png_files:
        png_files = sorted(clip_dir.glob("*.jpg"))
    frames = []
    for p in png_files:
        img = cv2.imread(str(p))
        if img is not None:
            try:
                idx = int(p.stem.split("_")[-1])
            except ValueError:
                idx = len(frames) + 1
            frames.append((idx, img))
    return frames


def run(clip_dir: Path, video_id: str, max_mllm_calls: int = 2) -> dict:
    """단일 클립 E2E 처리.

    Returns:
        결과 요약 dict.
    """
    frames = load_frames(clip_dir)
    if not frames:
        logger.error("프레임 없음: %s", clip_dir)
        return {"error": "no frames"}

    h, w = frames[0][1].shape[:2]
    fps = 1.0
    logger.info("클립 로드: %s (%d frames, %dx%d)", clip_dir.name, len(frames), w, h)

    # ── L1: Vision Pipeline ──────────────────────────────────────────
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

    all_triggers = []
    all_tracked = {}
    frame_map = {}

    for frame_idx, frame in frames:
        timestamp = frame_idx * (1.0 / fps)
        result = pipeline.process_frame(frame, frame_idx, timestamp)
        frame_map[frame_idx] = frame

        for v in result["tracked_vehicles"]:
            tid = v["track_id"]
            if tid not in all_tracked:
                all_tracked[tid] = {
                    "track_id": tid,
                    "first_frame": frame_idx,
                    "last_frame": frame_idx,
                    "cls": v["cls"],
                    "speeds": [],
                }
            all_tracked[tid]["last_frame"] = frame_idx
            if v["cls"]:
                all_tracked[tid]["cls"] = v["cls"]

        for tid, speeds in pipeline._speed_history.items():
            if tid in all_tracked and speeds:
                all_tracked[tid]["speeds"] = speeds[-10:]

        all_triggers.extend(result["triggers"])

    logger.info("L1 완료: %d tracks, %d triggers", len(all_tracked), len(all_triggers))

    # ── L2: DuckDB 적재 (tracks) ─────────────────────────────────────
    writer = MetadataWriter()

    tracks_data = []
    for tid, info in all_tracked.items():
        avg_speed = float(np.mean(info["speeds"])) if info["speeds"] else None
        tracks_data.append({
            "track_id": tid,
            "ic_name": clip_dir.name,
            "start_time": datetime.now(),
            "end_time": datetime.now(),
            "vehicle_cls_vision": info["cls"],
            "vehicle_cls_mllm": None,
            "vehicle_cls_final": info["cls"],
            "confidence": 0.5,
            "avg_speed": avg_speed,
            "trajectory": None,
        })
    n_tracks = writer.write_tracks(video_id, tracks_data)
    logger.info("L2 tracks 적재: %d건", n_tracks)

    # ── L3: MLLM 호출 (트리거별, max_mllm_calls 제한) ────────────────
    mllm_results = []
    if all_triggers:
        mllm = MLLMClient(backend="transformers")

        for i, trigger in enumerate(all_triggers[:max_mllm_calls]):
            trigger_frame = trigger.frame_idx
            img = frame_map.get(trigger_frame)
            if img is None:
                closest = min(frame_map.keys(), key=lambda k: abs(k - trigger_frame))
                img = frame_map[closest]

            prompt = ACCIDENT_PROMPT.format(
                trigger_type=trigger.type,
                trigger_desc=trigger.description,
            )
            messages = [{"role": "user", "content": prompt}]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            logger.info("MLLM 호출 %d/%d: [%s] frame=%d...",
                        i + 1, min(len(all_triggers), max_mllm_calls),
                        trigger.type, trigger_frame)
            response = mllm.chat(messages, images=[img_rgb])

            response_id = f"resp_{video_id}_{trigger.type}_{trigger_frame}"
            mllm_data = {
                "response_id": response_id,
                "video_id": video_id,
                "trigger_type": trigger.type,
                "trigger_frame": trigger_frame,
                "task": "accident_detection",
                "input_summary": trigger.description,
                "output_json": response.get("content"),
                "latency_sec": response.get("latency_sec"),
                "model_id": "Qwen2.5-VL-3B-Instruct",
                "created_at": datetime.now(),
            }
            writer.write_mllm_response(mllm_data)
            mllm_results.append(mllm_data)
            logger.info("MLLM 응답 적재: %s (%.1fs)", response_id, response.get("latency_sec", 0))

            content = response.get("content", {})
            if isinstance(content, dict) and content.get("anomaly"):
                accident_data = {
                    "event_id": f"evt_{video_id}_{trigger_frame}",
                    "video_id": video_id,
                    "accident_type": content.get("anomaly_type"),
                    "severity": content.get("severity"),
                    "vehicles": content.get("vehicles"),
                    "mllm_response_id": response_id,
                    "mllm_report_json": content,
                    "report_source": "CCTV_AUTO",
                }
                writer.write_accident(accident_data)
                logger.info("사고 이벤트 적재: %s", accident_data["event_id"])

    # ── 결과 요약 ────────────────────────────────────────────────────
    summary = {
        "video_id": video_id,
        "clip": clip_dir.name,
        "frames": len(frames),
        "tracks": n_tracks,
        "triggers": len(all_triggers),
        "trigger_types": [t.type for t in all_triggers],
        "mllm_calls": len(mllm_results),
        "mllm_latency_avg": (
            np.mean([r["latency_sec"] for r in mllm_results if r.get("latency_sec")])
            if mllm_results else None
        ),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="E2E 사고분석 파이프라인")
    parser.add_argument("clip_dir", type=Path, help="PNG 프레임 디렉토리")
    parser.add_argument("--video-id", default=None, help="영상 식별자 (기본: 디렉토리명)")
    parser.add_argument("--max-mllm", type=int, default=2, help="최대 MLLM 호출 수")
    args = parser.parse_args()

    video_id = args.video_id or args.clip_dir.name
    summary = run(args.clip_dir, video_id, args.max_mllm)

    print("\n" + "=" * 60)
    print("E2E 파이프라인 결과")
    print("=" * 60)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print("=" * 60)


if __name__ == "__main__":
    main()
