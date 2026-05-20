"""FeatureStore: 차량별 15차원 특징 벡터 관리.

VisionPipeline의 프레임별 출력을 받아 증분 계산하고,
순환 버퍼로 최근 N프레임을 유지한다.
RuleEngine / ML 모델이 이 FeatureStore를 조회한다.

15차원 특징 벡터:
  0: speed_kmh        현재 속도 (km/h)
  1: acceleration      가속도 (m/s^2)
  2: heading           이동 방향 (degrees, 0=북, 시계방향)
  3: heading_change    방향 변화율 (deg/frame)
  4: heading_deviation 기대 방향 대비 편차 (degrees)
  5: lateral_speed     횡방향 속도 (px/frame)
  6: ttc_min           최소 TTC (초, 없으면 999)
  7: gap_front         전방 최근접 차량 거리 (px)
  8: gap_rear          후방 최근접 차량 거리 (px)
  9: speed_diff_flow   교통류 평균 대비 속도차 (km/h)
  10: stop_duration    연속 정지 시간 (초)
  11: area_change_rate bbox 면적 변화율 (%)
  12: aspect_ratio     bbox 종횡비 (w/h)
  13: track_age        트랙 나이 (프레임 수)
  14: neighbor_count   반경 내 인근 차량 수
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

FEATURE_DIM = 15
BUFFER_SIZE = 300
MOVING_AVG_WINDOW = 5
NEIGHBOR_RADIUS_PX = 200
STOP_SPEED_KMH = 5.0

FEATURE_NAMES = [
    "speed_kmh", "acceleration", "heading", "heading_change",
    "heading_deviation", "lateral_speed", "ttc_min", "gap_front",
    "gap_rear", "speed_diff_flow", "stop_duration", "area_change_rate",
    "aspect_ratio", "track_age", "neighbor_count",
]


@dataclass
class TrackState:
    """단일 트랙의 내부 상태."""
    track_id: int
    first_seen: float = 0.0
    frame_count: int = 0
    centers: deque = field(default_factory=lambda: deque(maxlen=BUFFER_SIZE))
    speeds: deque = field(default_factory=lambda: deque(maxlen=BUFFER_SIZE))
    headings: deque = field(default_factory=lambda: deque(maxlen=BUFFER_SIZE))
    areas: deque = field(default_factory=lambda: deque(maxlen=BUFFER_SIZE))
    features: deque = field(default_factory=lambda: deque(maxlen=BUFFER_SIZE))
    stop_start: float | None = None
    cls: str = ""


class FeatureStore:
    """중앙 특징 저장소 — 증분 계산 + 순환 버퍼."""

    def __init__(self, expected_heading: float | None = None):
        """
        Args:
            expected_heading: 카메라별 기대 진행 방향 (degrees).
                              None이면 교통류 평균 방향 사용.
        """
        self._tracks: dict[int, TrackState] = {}
        self._expected_heading = expected_heading
        self._flow_speeds: deque[float] = deque(maxlen=BUFFER_SIZE)

    @property
    def active_track_ids(self) -> list[int]:
        return list(self._tracks.keys())

    def get_state(self, track_id: int) -> TrackState | None:
        return self._tracks.get(track_id)

    def get_features(self, track_id: int) -> np.ndarray | None:
        """최신 15차원 특징 벡터 반환."""
        state = self._tracks.get(track_id)
        if state is None or not state.features:
            return None
        return state.features[-1]

    def get_features_ma(self, track_id: int) -> np.ndarray | None:
        """이동평균(5프레임) 적용된 특징 벡터."""
        state = self._tracks.get(track_id)
        if state is None or len(state.features) < 1:
            return None
        window = list(state.features)[-MOVING_AVG_WINDOW:]
        return np.mean(window, axis=0)

    def get_all_features(self) -> dict[int, np.ndarray]:
        """전체 활성 트랙의 최신 특징 벡터."""
        result = {}
        for tid, state in self._tracks.items():
            if state.features:
                result[tid] = state.features[-1]
        return result

    def update(
        self,
        timestamp: float,
        tracked_vehicles: list[dict[str, Any]],
        ttc_list: list[dict[str, Any]],
        dt: float = 1.0,
    ) -> dict[int, np.ndarray]:
        """프레임 데이터로 전체 트랙 특징 갱신.

        Args:
            timestamp: 현재 타임스탬프 (초).
            tracked_vehicles: VisionPipeline.process_frame() 출력의
                              "tracked_vehicles" 리스트.
                              각 항목: {track_id, bbox, center, area, conf, cls, ...}
                              speed_kmh 키가 있으면 사용, 없으면 위치 기반 추정.
            ttc_list: compute_all_ttc() 반환값.
            dt: 프레임 간 시간 간격 (초).

        Returns:
            {track_id: 15-dim feature array} 갱신된 특징.
        """
        active_ids = set()
        centers_now: dict[int, tuple[float, float]] = {}

        for v in tracked_vehicles:
            tid = v["track_id"]
            active_ids.add(tid)
            cx, cy = v["center"]
            centers_now[tid] = (cx, cy)

            if tid not in self._tracks:
                self._tracks[tid] = TrackState(
                    track_id=tid, first_seen=timestamp,
                )

            state = self._tracks[tid]
            state.frame_count += 1
            state.centers.append((cx, cy))
            state.cls = v.get("cls", "") or ""

            bbox = v["bbox"]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            state.areas.append(w * h)

        # 비활성 트랙 정리 (600프레임 이상 미갱신)
        stale = [
            tid for tid in self._tracks
            if tid not in active_ids and self._tracks[tid].frame_count > 0
        ]
        for tid in stale:
            if self._tracks[tid].features and len(self._tracks[tid].features) > 0:
                last_age = self._tracks[tid].features[-1][13]
                if last_age > 600:
                    del self._tracks[tid]

        # TTC 인덱스 구축
        ttc_by_track: dict[int, float] = {}
        for entry in ttc_list:
            for tid_key in ("track_a", "track_b"):
                tid = entry[tid_key]
                ttc_val = entry["ttc"]
                if tid not in ttc_by_track or ttc_val < ttc_by_track[tid]:
                    ttc_by_track[tid] = ttc_val

        # 교통류 평균 속도
        all_speeds_now: list[float] = []

        result: dict[int, np.ndarray] = {}

        for tid in active_ids:
            state = self._tracks[tid]
            feat = np.zeros(FEATURE_DIM, dtype=np.float32)

            # [0] speed_kmh
            speed = self._compute_speed(state, dt)
            feat[0] = speed
            state.speeds.append(speed)
            all_speeds_now.append(speed)

            # [1] acceleration
            feat[1] = self._compute_acceleration(state, dt)

            # [2] heading
            heading = self._compute_heading(state)
            feat[2] = heading
            state.headings.append(heading)

            # [3] heading_change
            feat[3] = self._compute_heading_change(state)

            # [4] heading_deviation
            feat[4] = self._compute_heading_deviation(state)

            # [5] lateral_speed
            feat[5] = self._compute_lateral_speed(state)

            # [6] ttc_min
            feat[6] = ttc_by_track.get(tid, 999.0)

            # [7,8] gap_front, gap_rear
            gf, gr = self._compute_gaps(tid, centers_now)
            feat[7] = gf
            feat[8] = gr

            # [9] speed_diff_flow (이전 프레임 평균 기준, 현재 프레임 끝에 갱신)
            flow_avg = np.mean(self._flow_speeds) if self._flow_speeds else speed
            feat[9] = speed - flow_avg

            # [10] stop_duration
            if speed <= STOP_SPEED_KMH:
                if state.stop_start is None:
                    state.stop_start = timestamp
                feat[10] = timestamp - state.stop_start
            else:
                state.stop_start = None
                feat[10] = 0.0

            # [11] area_change_rate
            if len(state.areas) >= 2:
                prev_area = state.areas[-2]
                curr_area = state.areas[-1]
                if prev_area > 0:
                    feat[11] = (curr_area - prev_area) / prev_area * 100
            else:
                feat[11] = 0.0

            # [12] aspect_ratio
            v_info = next((v for v in tracked_vehicles if v["track_id"] == tid), None)
            if v_info:
                bbox = v_info["bbox"]
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                feat[12] = w / h if h > 0 else 1.0

            # [13] track_age
            feat[13] = state.frame_count

            # [14] neighbor_count
            feat[14] = self._count_neighbors(tid, centers_now)

            state.features.append(feat.copy())
            result[tid] = feat

        # 교통류 평균 갱신
        if all_speeds_now:
            self._flow_speeds.append(np.mean(all_speeds_now))

        return result

    def _compute_speed(self, state: TrackState, dt: float) -> float:
        if len(state.centers) < 2:
            return 0.0
        c0 = state.centers[-2]
        c1 = state.centers[-1]
        dist_px = math.hypot(c1[0] - c0[0], c1[1] - c0[1])
        # px/frame → 대략적 km/h (0.05 m/px fallback, dt초 간격)
        dist_m = dist_px * 0.05
        speed_ms = dist_m / dt if dt > 0 else 0.0
        return speed_ms * 3.6

    def _compute_acceleration(self, state: TrackState, dt: float) -> float:
        if len(state.speeds) < 2:
            return 0.0
        v_curr = state.speeds[-1] / 3.6
        v_prev = state.speeds[-2] / 3.6
        return (v_curr - v_prev) / dt if dt > 0 else 0.0

    def _compute_heading(self, state: TrackState) -> float:
        if len(state.centers) < 2:
            return 0.0
        c0 = state.centers[-2]
        c1 = state.centers[-1]
        dx = c1[0] - c0[0]
        dy = c1[1] - c0[1]
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return state.headings[-1] if state.headings else 0.0
        return math.degrees(math.atan2(dx, -dy)) % 360

    def _compute_heading_change(self, state: TrackState) -> float:
        if len(state.headings) < 2:
            return 0.0
        diff = state.headings[-1] - state.headings[-2]
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        return abs(diff)

    def _compute_heading_deviation(self, state: TrackState) -> float:
        if not state.headings:
            return 0.0
        current = state.headings[-1]

        if self._expected_heading is not None:
            expected = self._expected_heading
        else:
            # 교통류 평균 방향
            all_headings = []
            for s in self._tracks.values():
                if s.headings and s.track_id != state.track_id:
                    all_headings.append(s.headings[-1])
            if len(all_headings) < 3:
                return 0.0
            expected = _circular_mean(all_headings)

        diff = abs(current - expected)
        if diff > 180:
            diff = 360 - diff
        return diff

    def _compute_lateral_speed(self, state: TrackState) -> float:
        if len(state.centers) < 2 or not state.headings:
            return 0.0
        c0 = state.centers[-2]
        c1 = state.centers[-1]
        dx = c1[0] - c0[0]
        dy = c1[1] - c0[1]
        heading_rad = math.radians(state.headings[-1])
        # 진행 방향 단위 벡터
        fwd_x = math.sin(heading_rad)
        fwd_y = -math.cos(heading_rad)
        # 횡방향 = 전체 이동 - 종방향 성분
        longitudinal = dx * fwd_x + dy * fwd_y
        lat_x = dx - longitudinal * fwd_x
        lat_y = dy - longitudinal * fwd_y
        return math.hypot(lat_x, lat_y)

    def _compute_gaps(
        self, tid: int, centers: dict[int, tuple[float, float]],
    ) -> tuple[float, float]:
        if tid not in centers:
            return 9999.0, 9999.0
        my_center = centers[tid]
        state = self._tracks.get(tid)
        heading = state.headings[-1] if (state and state.headings) else 0.0
        heading_rad = math.radians(heading)
        fwd_x = math.sin(heading_rad)
        fwd_y = -math.cos(heading_rad)

        gap_front = 9999.0
        gap_rear = 9999.0

        for other_tid, other_center in centers.items():
            if other_tid == tid:
                continue
            dx = other_center[0] - my_center[0]
            dy = other_center[1] - my_center[1]
            proj = dx * fwd_x + dy * fwd_y
            dist = math.hypot(dx, dy)
            if dist > 500:
                continue
            if proj > 0 and dist < gap_front:
                gap_front = dist
            elif proj < 0 and dist < gap_rear:
                gap_rear = dist

        return gap_front, gap_rear

    def _count_neighbors(
        self, tid: int, centers: dict[int, tuple[float, float]],
    ) -> int:
        if tid not in centers:
            return 0
        my_center = centers[tid]
        count = 0
        for other_tid, other_center in centers.items():
            if other_tid == tid:
                continue
            dist = math.hypot(
                other_center[0] - my_center[0],
                other_center[1] - my_center[1],
            )
            if dist <= NEIGHBOR_RADIUS_PX:
                count += 1
        return count

    def purge_inactive(self, max_age: int = 600) -> int:
        """비활성 트랙 일괄 제거. 제거된 수 반환."""
        to_remove = []
        for tid, state in self._tracks.items():
            if not state.features:
                continue
            if state.features[-1][13] > max_age and state.frame_count > max_age:
                to_remove.append(tid)
        for tid in to_remove:
            del self._tracks[tid]
        return len(to_remove)


def _circular_mean(angles_deg: list[float]) -> float:
    """각도의 원형 평균."""
    rads = [math.radians(a) for a in angles_deg]
    sin_sum = sum(math.sin(r) for r in rads)
    cos_sum = sum(math.cos(r) for r in rads)
    return math.degrees(math.atan2(sin_sum, cos_sum)) % 360
