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


SCENE_PROMPT = """You are a highway CCTV analyst. {trigger_info}

IMPORTANT: Most highway footage shows NORMAL traffic. Lane changes are usually legal.
A lane change is only a violation if it meets one of these criteria:
- The vehicle crosses a SOLID white/yellow line (not dashed)
- The vehicle changes lanes WITHOUT a turn signal blinking
- Multiple vehicles change lanes simultaneously causing danger
- A vehicle drifts across lane markings (straddling)
- A vehicle changes 2+ lanes in one move
- Unsafe merge with very short gap in heavy traffic

Step 1 - Describe what you see: road type, weather, how many lanes, what each vehicle is doing.
Step 2 - Check: are lane lines solid or dashed? Is any vehicle crossing them? Any turn signals visible?
Step 3 - Judge: is this normal driving or a real violation?

Output JSON:
```json
{{"scene": "step1 description", "anomaly": true/false, "anomaly_type": "type or null", "vehicles": [{{"type": "vehicle type", "color": "color", "behavior": "action"}}], "severity": "low/medium/high or null", "confidence": 0.0-1.0}}
```"""

MIN_RESOLUTION = 480


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


def _upscale_if_small(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) < MIN_RESOLUTION:
        scale = MIN_RESOLUTION / max(h, w)
        new_w, new_h = round(w * scale), round(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return img


def _get_multi_frames(frame_map: dict, trigger_frame: int, spread: int = 5) -> list[np.ndarray]:
    """트리거 프레임 전후 spread 간격으로 3장 선택."""
    sorted_idxs = sorted(frame_map.keys())
    center_pos = min(range(len(sorted_idxs)), key=lambda i: abs(sorted_idxs[i] - trigger_frame))

    positions = [
        max(0, center_pos - spread),
        center_pos,
        min(len(sorted_idxs) - 1, center_pos + spread),
    ]
    seen = set()
    unique_positions = []
    for p in positions:
        if p not in seen:
            seen.add(p)
            unique_positions.append(p)

    imgs = []
    for p in unique_positions:
        img = frame_map[sorted_idxs[p]]
        imgs.append(_upscale_if_small(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    return imgs


def _sanitize_content(content):
    """MLLM 응답 content를 DB 저장 가능한 형태로 정제."""
    if isinstance(content, str):
        content = content.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")
        if len(content) > 2000:
            content = content[:2000]
    return content


def run(clip_dir: Path, video_id: str, max_mllm_calls: int = 2,
        force_trigger: bool = False, mllm: MLLMClient | None = None) -> dict:
    """단일 클립 E2E 처리.

    Args:
        mllm: 외부에서 생성한 MLLMClient 인스턴스 (싱글턴 재사용).
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
    n_tracks = writer.write_tracks(video_id, tracks_data) if tracks_data else 0
    logger.info("L2 tracks 적재: %d건", n_tracks)

    # ── L3: MLLM 호출 ──────────────────────────────────────────────────
    mllm_results = []
    triggers_to_process = all_triggers[:max_mllm_calls]

    if not triggers_to_process and force_trigger:
        from layer1_vision.trigger_detector import TriggerEvent
        mid_idx = frames[len(frames) // 2][0]
        triggers_to_process = [TriggerEvent(
            type="T0", frame_idx=mid_idx, timestamp=mid_idx,
            description="강제 분석 (트리거 미발화)", severity=0.0,
        )]
        logger.info("트리거 미발화 → 강제 MLLM 호출 (frame=%d)", mid_idx)

    if triggers_to_process:
        own_mllm = mllm is None
        if own_mllm:
            mllm = MLLMClient(backend="transformers")

        for i, trigger in enumerate(triggers_to_process):
            trigger_frame = trigger.frame_idx

            img = frame_map.get(trigger_frame)
            if img is None:
                closest = min(frame_map.keys(), key=lambda k: abs(k - trigger_frame))
                img = frame_map[closest]
            img_rgb = _upscale_if_small(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            trigger_info = (f"Trigger: {trigger.type} — {trigger.description}"
                           if trigger.type != "T0" else "Routine check (no trigger detected)")
            prompt = SCENE_PROMPT.format(trigger_info=trigger_info)
            messages = [{"role": "user", "content": prompt}]

            logger.info("MLLM 호출 %d/%d: [%s] frame=%d...",
                        i + 1, min(len(all_triggers), max_mllm_calls),
                        trigger.type, trigger_frame)
            try:
                response = mllm.chat(messages, images=[img_rgb], max_tokens=512)
            except Exception as e:
                logger.error("MLLM 호출 실패 [%s] frame=%d: %s", trigger.type, trigger_frame, e)
                continue

            response_id = f"resp_{video_id}_{trigger.type}_{trigger_frame}"
            content = _sanitize_content(response.get("content"))

            mllm_data = {
                "response_id": response_id,
                "video_id": video_id,
                "trigger_type": trigger.type,
                "trigger_frame": trigger_frame,
                "task": "accident_detection",
                "input_summary": trigger.description,
                "output_json": content,
                "latency_sec": response.get("latency_sec"),
                "model_id": "Qwen2.5-VL-3B-Instruct",
                "created_at": datetime.now(),
            }
            try:
                writer.write_mllm_response(mllm_data)
            except Exception as e:
                logger.error("MLLM 응답 DB 적재 실패: %s", e)
            mllm_results.append(mllm_data)
            logger.info("MLLM 응답 적재: %s (%.1fs)", response_id, response.get("latency_sec", 0))

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
                try:
                    writer.write_accident(accident_data)
                    logger.info("사고 이벤트 적재: %s", accident_data["event_id"])
                except Exception as e:
                    logger.error("사고 이벤트 DB 적재 실패: %s", e)

    # ── 결과 요약 ────────────────────────────────────────────────────
    mllm_anomalies = []
    mllm_vehicles = []
    for r in mllm_results:
        content = r.get("output_json", {})
        if isinstance(content, dict):
            mllm_anomalies.append(content.get("anomaly"))
            vs = content.get("vehicles", [])
            has_real = any(
                isinstance(v, dict) and v.get("type", "차종") != "차종"
                for v in (vs if isinstance(vs, list) else [])
            )
            mllm_vehicles.append(has_real)

    summary = {
        "video_id": video_id,
        "clip": clip_dir.name,
        "frames": len(frames),
        "tracks": n_tracks,
        "triggers": len(all_triggers),
        "trigger_types": [t.type for t in all_triggers],
        "mllm_calls": len(mllm_results),
        "mllm_anomaly": mllm_anomalies,
        "mllm_vehicles_real": mllm_vehicles,
        "mllm_latency_avg": (
            float(np.mean([r["latency_sec"] for r in mllm_results if r.get("latency_sec")]))
            if mllm_results else None
        ),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="E2E 사고분석 파이프라인")
    parser.add_argument("clip_dir", type=Path, help="PNG 프레임 디렉토리")
    parser.add_argument("--video-id", default=None, help="영상 식별자 (기본: 디렉토리명)")
    parser.add_argument("--max-mllm", type=int, default=2, help="최대 MLLM 호출 수")
    parser.add_argument("--force-trigger", action="store_true",
                        help="트리거 미발화 시에도 MLLM 강제 호출")
    args = parser.parse_args()

    video_id = args.video_id or args.clip_dir.name
    summary = run(args.clip_dir, video_id, args.max_mllm, args.force_trigger)

    print("\n" + "=" * 60)
    print("E2E 파이프라인 결과")
    print("=" * 60)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print("=" * 60)


if __name__ == "__main__":
    main()
