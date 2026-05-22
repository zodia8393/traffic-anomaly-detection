"""MLLM 호출 트리거 판정 (7종).

프레임별 차량 상태를 분석하여 이상상황을 감지하고,
MLLM에 분석을 요청할 트리거 이벤트를 생성한다.

트리거 유형:
  T1: TTC 임계값 이하 (충돌 위험)
  T2: 급감속 (개별 차량)
  T3: 정지 차량 (일정 시간 이상)
  T4: 역주행 감지
  T5: 다수 동시 감속
  T6: 속도 분산 이상
  T7: 주기적 (5분 간격 자동)
"""

from __future__ import annotations
import sys as _sys; from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import logging
from dataclasses import dataclass, field
from typing import Any

from config_new import (
    TRIGGER_COOLDOWN_SEC,
    TRIGGER_DECEL_THRESHOLD,
    TRIGGER_MULTI_DECEL_COUNT,
    TRIGGER_PERIODIC_INTERVAL,
    TRIGGER_SPEED_VAR_SIGMA,
    TRIGGER_STOP_DURATION,
    TRIGGER_STOP_SPEED,
    TRIGGER_TTC_MIN,
    TRIGGER_TTC_STREAK_MIN,
    TRIGGER_TTC_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass
class TriggerEvent:
    """트리거 이벤트."""
    type: str                              # "T1" ~ "T7"
    frame_idx: int
    timestamp: float
    involved_tracks: list[int] = field(default_factory=list)
    severity: float = 0.0                  # 0.0 ~ 1.0
    description: str = ""


class TriggerDetector:
    """프레임별 트리거 판정기."""

    def __init__(self) -> None:
        # 쿨다운 관리: {trigger_type: last_fired_timestamp}
        self._last_fired: dict[str, float] = {}

        # 정지 차량 추적: {track_id: first_stopped_timestamp}
        self._stopped_since: dict[int, float] = {}

        # 속도 히스토리 (분산 계산용): 최근 N개 프레임의 전체 평균속도
        self._speed_history: list[float] = []
        self._speed_history_max = 300

        # 역주행 기준 방향: {track_id: initial_direction_vector}
        self._initial_directions: dict[int, tuple[float, float]] = {}

        # T1 TTC 연속 감지 카운터: {(track_a, track_b): 연속 프레임 수}
        self._ttc_streak: dict[tuple[int, int], int] = {}

        # 마지막 주기적 트리거 시각
        self._last_periodic: float = 0.0

    def update(
        self,
        frame_idx: int,
        timestamp: float,
        tracks_info: dict[int, dict[str, Any]],
        ttc_list: list[dict[str, Any]],
        speeds: dict[int, list[float]],
    ) -> list[TriggerEvent]:
        """프레임 상태를 분석하여 트리거 이벤트 목록 반환.

        Args:
            frame_idx: 현재 프레임 인덱스.
            timestamp: 현재 타임스탬프(초).
            tracks_info: {track_id: {"bbox": [...], "center": (cx,cy), "cls": str}}
            ttc_list: compute_all_ttc() 반환값.
            speeds: {track_id: [speed_kmh, ...]} 최근 속도 이력.

        Returns:
            발생한 TriggerEvent 목록.
        """
        events: list[TriggerEvent] = []

        events.extend(self._check_ttc(frame_idx, timestamp, ttc_list))
        events.extend(self._check_decel(frame_idx, timestamp, speeds))
        events.extend(self._check_stopped(frame_idx, timestamp, tracks_info, speeds))
        events.extend(self._check_wrong_way(frame_idx, timestamp, tracks_info))
        events.extend(self._check_multi_decel(frame_idx, timestamp, speeds))
        events.extend(self._check_speed_variance(frame_idx, timestamp, speeds))
        events.extend(self._check_periodic(frame_idx, timestamp))

        return events

    def _is_cooled_down(self, trigger_type: str, timestamp: float) -> bool:
        """쿨다운 검사. True면 발화 가능."""
        last = self._last_fired.get(trigger_type)
        if last is None:
            return True
        return (timestamp - last) >= TRIGGER_COOLDOWN_SEC

    def _fire(self, trigger_type: str, timestamp: float) -> None:
        """쿨다운 타이머 갱신."""
        self._last_fired[trigger_type] = timestamp

    # ── T1: TTC 임계값 이하 (연속성 필터 적용) ───────────────────────

    def _check_ttc(
        self, frame_idx: int, timestamp: float,
        ttc_list: list[dict[str, Any]],
    ) -> list[TriggerEvent]:
        if not self._is_cooled_down("T1", timestamp):
            return []

        current_pairs: set[tuple[int, int]] = set()

        for entry in ttc_list:
            if entry["ttc"] <= TRIGGER_TTC_THRESHOLD:
                if entry["ttc"] < TRIGGER_TTC_MIN:
                    continue

                pair = (min(entry["track_a"], entry["track_b"]),
                        max(entry["track_a"], entry["track_b"]))
                current_pairs.add(pair)
                self._ttc_streak[pair] = self._ttc_streak.get(pair, 0) + 1

                if self._ttc_streak[pair] >= TRIGGER_TTC_STREAK_MIN:
                    severity = max(0.0, 1.0 - entry["ttc"] / TRIGGER_TTC_THRESHOLD)
                    event = TriggerEvent(
                        type="T1",
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        involved_tracks=[entry["track_a"], entry["track_b"]],
                        severity=severity,
                        description=(
                            f"TTC={entry['ttc']:.1f}s (임계={TRIGGER_TTC_THRESHOLD}s)"
                            f" 연속{self._ttc_streak[pair]}f"
                        ),
                    )
                    self._fire("T1", timestamp)
                    del self._ttc_streak[pair]
                    return [event]

        for pair in list(self._ttc_streak):
            if pair not in current_pairs:
                del self._ttc_streak[pair]

        return []

    # ── T2: 급감속 (개별) ────────────────────────────────────────────

    def _check_decel(
        self, frame_idx: int, timestamp: float,
        speeds: dict[int, list[float]],
    ) -> list[TriggerEvent]:
        if not self._is_cooled_down("T2", timestamp):
            return []

        for tid, speed_list in speeds.items():
            if len(speed_list) < 2:
                continue
            # 최근 2프레임 속도차 -> 가속도 근사 (km/h -> m/s, /dt)
            v_curr = speed_list[-1] / 3.6
            v_prev = speed_list[-2] / 3.6
            # dt 가정: 1/fps, 여기서는 속도 리스트 간격 = 1프레임 간격으로 가정
            # 정밀한 dt는 caller가 timestamp 간격으로 넘겨야 하지만,
            # 여기서는 1초 간격 근사 (1fps CCTV 기준)
            accel = v_curr - v_prev  # m/s^2 (dt=1s 가정)

            if accel <= TRIGGER_DECEL_THRESHOLD:
                severity = min(1.0, abs(accel) / abs(TRIGGER_DECEL_THRESHOLD))
                event = TriggerEvent(
                    type="T2",
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    involved_tracks=[tid],
                    severity=severity,
                    description=f"급감속 track={tid} accel={accel:.1f}m/s^2",
                )
                self._fire("T2", timestamp)
                return [event]
        return []

    # ── T3: 정지 차량 ───────────────────────────────────────────────

    def _check_stopped(
        self, frame_idx: int, timestamp: float,
        tracks_info: dict[int, dict[str, Any]],
        speeds: dict[int, list[float]],
    ) -> list[TriggerEvent]:
        if not self._is_cooled_down("T3", timestamp):
            return []

        for tid in tracks_info:
            current_speed = speeds.get(tid, [])
            if not current_speed:
                continue

            if current_speed[-1] <= TRIGGER_STOP_SPEED:
                if tid not in self._stopped_since:
                    self._stopped_since[tid] = timestamp
                elif (timestamp - self._stopped_since[tid]) >= TRIGGER_STOP_DURATION:
                    duration = timestamp - self._stopped_since[tid]
                    severity = min(1.0, duration / (TRIGGER_STOP_DURATION * 3))
                    event = TriggerEvent(
                        type="T3",
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        involved_tracks=[tid],
                        severity=severity,
                        description=f"정지차량 track={tid} {duration:.0f}s",
                    )
                    self._fire("T3", timestamp)
                    del self._stopped_since[tid]
                    return [event]
            else:
                # 다시 움직임 -> 정지 추적 해제
                self._stopped_since.pop(tid, None)
        return []

    # ── T4: 역주행 ──────────────────────────────────────────────────

    def _check_wrong_way(
        self, frame_idx: int, timestamp: float,
        tracks_info: dict[int, dict[str, Any]],
    ) -> list[TriggerEvent]:
        """초기 이동방향 대비 180도 반전 감지."""
        if not self._is_cooled_down("T4", timestamp):
            return []

        for tid, info in tracks_info.items():
            center = info.get("center")
            if center is None:
                continue

            if tid not in self._initial_directions:
                # 아직 방향 미확정 -> 기록만
                self._initial_directions[tid] = center
                continue

            init_pos = self._initial_directions[tid]
            dx = center[0] - init_pos[0]
            dy = center[1] - init_pos[1]
            dist = (dx**2 + dy**2) ** 0.5
            if dist < 50:
                # 이동 거리 부족 -> 판단 보류
                continue

            # 이동 방향 벡터 정규화
            dir_x, dir_y = dx / dist, dy / dist

            # 주류 방향 계산 (현재 프레임 전체 차량)
            main_dirs: list[tuple[float, float]] = []
            for other_tid, other_info in tracks_info.items():
                if other_tid == tid:
                    continue
                other_center = other_info.get("center")
                if other_center is None or other_tid not in self._initial_directions:
                    continue
                oinit = self._initial_directions[other_tid]
                odx = other_center[0] - oinit[0]
                ody = other_center[1] - oinit[1]
                odist = (odx**2 + ody**2) ** 0.5
                if odist >= 50:
                    main_dirs.append((odx / odist, ody / odist))

            if len(main_dirs) < 3:
                # 비교 대상 부족
                continue

            # 주류 방향 평균
            avg_x = sum(d[0] for d in main_dirs) / len(main_dirs)
            avg_y = sum(d[1] for d in main_dirs) / len(main_dirs)
            avg_norm = (avg_x**2 + avg_y**2) ** 0.5
            if avg_norm < 0.3:
                continue
            avg_x /= avg_norm
            avg_y /= avg_norm

            # 코사인 유사도: -1에 가까울수록 역방향
            cosine = dir_x * avg_x + dir_y * avg_y
            if cosine < -0.7:
                event = TriggerEvent(
                    type="T4",
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    involved_tracks=[tid],
                    severity=min(1.0, abs(cosine)),
                    description=f"역주행 의심 track={tid} cosine={cosine:.2f}",
                )
                self._fire("T4", timestamp)
                return [event]
        return []

    # ── T5: 다수 동시 감속 ──────────────────────────────────────────

    def _check_multi_decel(
        self, frame_idx: int, timestamp: float,
        speeds: dict[int, list[float]],
    ) -> list[TriggerEvent]:
        if not self._is_cooled_down("T5", timestamp):
            return []

        decel_tracks: list[int] = []
        for tid, speed_list in speeds.items():
            if len(speed_list) < 2:
                continue
            delta = speed_list[-1] - speed_list[-2]
            # km/h 단위 급감속 (10 km/h 이상 감소)
            if delta <= -10:
                decel_tracks.append(tid)

        if len(decel_tracks) >= TRIGGER_MULTI_DECEL_COUNT:
            severity = min(1.0, len(decel_tracks) / (TRIGGER_MULTI_DECEL_COUNT * 2))
            event = TriggerEvent(
                type="T5",
                frame_idx=frame_idx,
                timestamp=timestamp,
                involved_tracks=decel_tracks,
                severity=severity,
                description=f"동시감속 {len(decel_tracks)}대",
            )
            self._fire("T5", timestamp)
            return [event]
        return []

    # ── T6: 속도 분산 이상 ──────────────────────────────────────────

    def _check_speed_variance(
        self, frame_idx: int, timestamp: float,
        speeds: dict[int, list[float]],
    ) -> list[TriggerEvent]:
        if not self._is_cooled_down("T6", timestamp):
            return []

        # 현재 프레임의 전체 차량 속도 수집
        current_speeds = [sl[-1] for sl in speeds.values() if sl]
        if len(current_speeds) < 3:
            return []

        avg = sum(current_speeds) / len(current_speeds)
        self._speed_history.append(avg)
        if len(self._speed_history) > self._speed_history_max:
            self._speed_history = self._speed_history[-self._speed_history_max:]

        if len(self._speed_history) < 10:
            return []

        # 히스토리 평균/표준편차
        hist_avg = sum(self._speed_history) / len(self._speed_history)
        hist_var = sum((s - hist_avg) ** 2 for s in self._speed_history) / len(self._speed_history)
        hist_std = hist_var ** 0.5

        if hist_std < 1.0:
            return []

        # 현재 평균속도가 히스토리 대비 sigma 배 이상 벗어나면 트리거
        deviation = abs(avg - hist_avg) / hist_std
        if deviation >= TRIGGER_SPEED_VAR_SIGMA:
            severity = min(1.0, deviation / (TRIGGER_SPEED_VAR_SIGMA * 2))
            event = TriggerEvent(
                type="T6",
                frame_idx=frame_idx,
                timestamp=timestamp,
                severity=severity,
                description=f"속도분산이상 avg={avg:.1f}km/h (hist={hist_avg:.1f}+/-{hist_std:.1f})",
            )
            self._fire("T6", timestamp)
            return [event]
        return []

    # ── T7: 주기적 (5분 간격) ───────────────────────────────────────

    def _check_periodic(
        self, frame_idx: int, timestamp: float,
    ) -> list[TriggerEvent]:
        if self._last_periodic == 0.0:
            self._last_periodic = timestamp
            return []

        if (timestamp - self._last_periodic) >= TRIGGER_PERIODIC_INTERVAL:
            self._last_periodic = timestamp
            return [TriggerEvent(
                type="T7",
                frame_idx=frame_idx,
                timestamp=timestamp,
                severity=0.1,
                description=f"주기적 스냅샷 ({TRIGGER_PERIODIC_INTERVAL}s 간격)",
            )]
        return []
