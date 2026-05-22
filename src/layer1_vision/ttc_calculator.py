"""Time-to-Collision (TTC) 계산.

두 차량 트랙 간 충돌 예상 시간을 bbox 중심 간 거리 변화율로 산출한다.
TTC = distance / closing_speed. closing_speed <= 0이면 접근하지 않으므로 None.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any


def _bbox_center(bbox: list[float] | tuple[float, ...]) -> tuple[float, float]:
    """bbox [x1, y1, x2, y2]의 중심 좌표."""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _center_distance(c1: tuple[float, float], c2: tuple[float, float]) -> float:
    """두 중심 간 유클리드 거리."""
    return math.hypot(c2[0] - c1[0], c2[1] - c1[1])


def compute_ttc(
    track_a: list[dict[str, Any]],
    track_b: list[dict[str, Any]],
) -> float | None:
    """두 트랙 간 TTC 산출.

    각 트랙 원소는 최소 {"bbox": [x1,y1,x2,y2], "timestamp": float}를 포함.
    가장 최근 2개 프레임의 거리 변화율로 closing_speed를 추정한다.

    Args:
        track_a: 차량 A의 시간순 트랙 기록.
        track_b: 차량 B의 시간순 트랙 기록.

    Returns:
        TTC(초). 접근하지 않으면 None.
    """
    if len(track_a) < 2 or len(track_b) < 2:
        return None

    # 시간순 마지막 2개 프레임 사용
    a_prev, a_curr = track_a[-2], track_a[-1]
    b_prev, b_curr = track_b[-2], track_b[-1]

    ca_prev = _bbox_center(a_prev["bbox"])
    ca_curr = _bbox_center(a_curr["bbox"])
    cb_prev = _bbox_center(b_prev["bbox"])
    cb_curr = _bbox_center(b_curr["bbox"])

    dist_prev = _center_distance(ca_prev, cb_prev)
    dist_curr = _center_distance(ca_curr, cb_curr)

    # 시간 간격 (두 트랙의 timestamp 차이 평균)
    dt_a = a_curr["timestamp"] - a_prev["timestamp"]
    dt_b = b_curr["timestamp"] - b_prev["timestamp"]
    dt = (dt_a + dt_b) / 2
    if dt <= 0:
        return None

    # closing_speed = 단위시간당 거리 감소량 (양수 = 접근)
    closing_speed = (dist_prev - dist_curr) / dt

    if closing_speed <= 0:
        # 접근하지 않음 (멀어지거나 평행)
        return None

    if dist_curr <= 0:
        # 이미 겹침
        return 0.0

    return dist_curr / closing_speed


def _is_same_lane(
    hist_a: list[dict[str, Any]],
    hist_b: list[dict[str, Any]],
    overlap_threshold: float = 0.3,
) -> bool:
    """최근 bbox의 X축 겹침으로 같은 차선 여부 추정."""
    ba = hist_a[-1]["bbox"]
    bb = hist_b[-1]["bbox"]

    x_overlap = max(0.0, min(ba[2], bb[2]) - max(ba[0], bb[0]))
    w_a = ba[2] - ba[0]
    w_b = bb[2] - bb[0]
    min_w = min(w_a, w_b)

    if min_w <= 0:
        return False
    return (x_overlap / min_w) > overlap_threshold


def compute_all_ttc(
    tracks: dict[int, list[dict[str, Any]]],
    lane_filter: bool = True,
) -> list[dict[str, Any]]:
    """전체 트랙 페어의 TTC를 계산한다.

    Args:
        tracks: {tracker_id: [{"bbox": [...], "timestamp": float}, ...]}
        lane_filter: True면 같은 차선(bbox X-overlap) 쌍만 계산.

    Returns:
        list of {"track_a": int, "track_b": int, "ttc": float}
        TTC가 None인 페어는 제외.
    """
    results: list[dict[str, Any]] = []
    active_ids = [tid for tid, history in tracks.items() if len(history) >= 2]

    for tid_a, tid_b in combinations(active_ids, 2):
        if lane_filter and not _is_same_lane(tracks[tid_a], tracks[tid_b]):
            continue
        ttc = compute_ttc(tracks[tid_a], tracks[tid_b])
        if ttc is not None:
            results.append({
                "track_a": tid_a,
                "track_b": tid_b,
                "ttc": ttc,
            })

    # TTC 오름차순 정렬 (위험도 높은 순)
    results.sort(key=lambda x: x["ttc"])
    return results
