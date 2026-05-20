"""AI Hub #71566 데이터 로더 — 클립 단위 라벨 시퀀스 로딩."""
from __future__ import annotations

import json
import math
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    """단일 프레임의 차량 어노테이션."""
    frame_idx: int
    vehicles: list[dict]  # [{id, bbox, category, DrivingType, ...}]
    meta: dict


@dataclass
class Clip:
    """하나의 클립 시퀀스."""
    clip_id: str
    label: str       # "정상" or "비정상"
    subtype: str      # "방향지시등 불이행" 등
    frames: list[Frame]

    @property
    def is_normal(self) -> bool:
        return self.label == "정상"


def load_clips(label_root: Path, max_clips: int | None = None) -> list[Clip]:
    """라벨 디렉토리에서 클립 단위로 로딩 (정상/비정상 균등).

    디렉토리 구조: label_root/{정상|비정상}/{subtype}/{clip_id}/*.json
    """
    clips_by_label: dict[str, list[Clip]] = {"정상": [], "비정상": []}
    per_label_max = max_clips // 2 if max_clips else None
    clips = []
    for label_dir in sorted(label_root.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        if label not in clips_by_label:
            continue
        for subtype_dir in sorted(label_dir.iterdir()):
            if not subtype_dir.is_dir():
                continue
            subtype = subtype_dir.name
            for clip_dir in sorted(subtype_dir.iterdir()):
                if not clip_dir.is_dir():
                    continue
                if per_label_max and len(clips_by_label[label]) >= per_label_max:
                    break

                frames = []
                for jf in sorted(clip_dir.glob("*.json")):
                    with open(jf) as f:
                        data = json.load(f)
                    frame_idx = int(jf.stem.rsplit("_", 1)[-1])
                    vehicles = data.get("annotation", [])
                    meta = data.get("meta", {})
                    frames.append(Frame(frame_idx=frame_idx, vehicles=vehicles, meta=meta))

                if frames:
                    frames.sort(key=lambda f: f.frame_idx)
                    clips_by_label[label].append(Clip(
                        clip_id=clip_dir.name,
                        label=label,
                        subtype=subtype,
                        frames=frames,
                    ))
    clips = clips_by_label["정상"] + clips_by_label["비정상"]
    return clips


def extract_clip_features(clip: Clip) -> np.ndarray:
    """클립에서 차량별 시계열 특성벡터 추출 → 클립 레벨 통계 벡터 반환.

    반환: (N_features,) 1D array — 클립 전체의 이상 지표 통계.
    """
    n_feat = len(FEATURE_NAMES_CLIP)
    if len(clip.frames) < 2:
        return np.zeros(len(FEATURE_NAMES_CLIP), dtype=np.float32)

    # 차량별 bbox + 메타 시퀀스 추적
    tracks: dict[int, list[tuple[int, list, dict]]] = defaultdict(list)
    for frame in clip.frames:
        for v in frame.vehicles:
            vid = v.get("id", 0)
            bbox = v.get("bbox", [0, 0, 0, 0])
            tracks[vid].append((frame.frame_idx, bbox, v))

    all_speeds = []
    all_accels = []
    all_heading_changes = []
    all_lateral_speeds = []
    all_aspect_ratios = []
    all_area_changes = []
    min_gaps = []
    lane_changes = 0
    unsafe_distance_count = 0
    total_annotations = 0

    for vid, seq in tracks.items():
        if len(seq) < 2:
            continue

        speeds = []
        headings = []
        centers = []
        prev_lane = None

        for i, (fidx, bbox, vmeta) in enumerate(seq):
            x, y, w, h = bbox
            cx, cy = x + w / 2, y + h / 2
            centers.append((cx, cy))
            total_annotations += 1

            ar = w / h if h > 0 else 1.0
            all_aspect_ratios.append(ar)

            # 차선 변경 감지
            curr_lane = vmeta.get("leftLine", "")
            if prev_lane and curr_lane and curr_lane != prev_lane:
                lane_changes += 1
            prev_lane = curr_lane

            # 안전거리 미확보
            if vmeta.get("safetydistance", 0) == 1:
                unsafe_distance_count += 1

            if i > 0:
                prev_cx, prev_cy = centers[-2]
                dx, dy = cx - prev_cx, cy - prev_cy
                dist = math.hypot(dx, dy)
                speeds.append(dist)
                all_speeds.append(dist)

                heading = math.degrees(math.atan2(dx, -dy)) % 360
                headings.append(heading)

                prev_bbox = seq[i - 1][1]
                prev_area = prev_bbox[2] * prev_bbox[3]
                curr_area = w * h
                if prev_area > 0:
                    all_area_changes.append(abs(curr_area - prev_area) / prev_area)

            if i >= 2:
                accel = speeds[-1] - speeds[-2]
                all_accels.append(accel)

                h_change = abs(headings[-1] - headings[-2])
                if h_change > 180:
                    h_change = 360 - h_change
                all_heading_changes.append(h_change)

                heading_rad = math.radians(headings[-1])
                fwd_x, fwd_y = math.sin(heading_rad), -math.cos(heading_rad)
                dx = centers[-1][0] - centers[-2][0]
                dy = centers[-1][1] - centers[-2][1]
                lon = dx * fwd_x + dy * fwd_y
                lat = math.hypot(dx - lon * fwd_x, dy - lon * fwd_y)
                all_lateral_speeds.append(lat)

        for other_vid, other_seq in tracks.items():
            if other_vid <= vid:
                continue
            for (f1, b1, _), (f2, b2, _) in zip(seq, other_seq):
                if f1 == f2:
                    c1 = (b1[0] + b1[2] / 2, b1[1] + b1[3] / 2)
                    c2 = (b2[0] + b2[2] / 2, b2[1] + b2[3] / 2)
                    min_gaps.append(math.hypot(c1[0] - c2[0], c1[1] - c2[1]))

    def _stats(arr: list) -> list[float]:
        if not arr:
            return [0.0, 0.0, 0.0, 0.0]
        a = np.array(arr)
        return [float(np.mean(a)), float(np.std(a)), float(np.max(a)), float(np.percentile(a, 95))]

    lane_change_rate = lane_changes / max(total_annotations, 1)
    unsafe_rate = unsafe_distance_count / max(total_annotations, 1)

    feat = np.array(
        _stats(all_speeds)            # 0-3: speed
        + _stats(all_accels)          # 4-7: acceleration
        + _stats(all_heading_changes) # 8-11: heading change
        + _stats(all_lateral_speeds)  # 12-15: lateral speed
        + _stats(min_gaps)            # 16-19: min gap
        + _stats(all_area_changes)    # 20-23: area change
        + [lane_change_rate]          # 24: lane change rate
        + [unsafe_rate]               # 25: unsafe distance rate
        + [len(tracks)]               # 26: vehicle count
        , dtype=np.float32,
    )
    return feat


FEATURE_NAMES_CLIP = [
    "speed_mean", "speed_std", "speed_max", "speed_p95",
    "accel_mean", "accel_std", "accel_max", "accel_p95",
    "heading_chg_mean", "heading_chg_std", "heading_chg_max", "heading_chg_p95",
    "lateral_mean", "lateral_std", "lateral_max", "lateral_p95",
    "gap_mean", "gap_std", "gap_max", "gap_p95",
    "area_chg_mean", "area_chg_std", "area_chg_max", "area_chg_p95",
    "lane_change_rate", "unsafe_distance_rate", "vehicle_count",
]
