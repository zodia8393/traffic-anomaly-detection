"""YOLO 없이 이상 프레임을 탐지하는 경량 프리필터 엔진.

카메라 1대당 1개 인스턴스. 매 프레임마다 check()를 호출하면
4종 신호(MOG2 전경비율, 프레임차분, 정지객체, 히스토그램급변)를
OR 조합하여 이상 여부와 점수를 반환한다.

목표: ~1.5ms/프레임 (640x480), OpenCV만 사용.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from config_new import (
    PREFILTER_COOLDOWN_SEC,
    PREFILTER_FG_RATIO_HIGH,
    PREFILTER_FG_RATIO_LOW,
    PREFILTER_FRAME_DIFF_THRESHOLD,
    PREFILTER_HIST_DIFF_THRESHOLD,
    PREFILTER_STATIC_DURATION_SEC,
)

logger = logging.getLogger(__name__)

# 정지 객체 감지 최소 면적 (640x480의 ~0.5%)
_STATIC_MIN_AREA = 1500

# 형태학 연산 커널 (재사용)
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


@dataclass
class PreFilterResult:
    """프리필터 판정 결과."""

    is_anomaly: bool            # True면 Tier 3 승격 대상
    score: float                # 0.0~1.0 이상도 (OR 조합 최대값)
    reasons: list[str]          # 발화 이유
    fg_ratio: float             # 디버깅용: 전경 비율
    frame_diff_mean: float      # 디버깅용: 프레임 차분 평균
    hist_corr: float            # 디버깅용: 히스토그램 상관계수


class PreFilter:
    """YOLO 없이 이상 프레임을 탐지하는 경량 엔진.

    카메라 1대당 1개 인스턴스. 매 프레임마다 check()를 호출하면
    이상 여부와 점수를 반환한다.
    """

    def __init__(self) -> None:
        """MOG2 배경 모델, 이전 프레임, 히스토그램 등 초기화."""
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False,
        )
        self._prev_gray: np.ndarray | None = None
        self._prev_hist: np.ndarray | None = None

        # 정지 객체 추적: {centroid_key: first_seen_timestamp}
        # centroid_key = (cx // grid, cy // grid) 로 양자화
        self._static_blobs: dict[tuple[int, int], float] = {}
        self._static_grid = 40  # 양자화 셀 크기 (px)

        # MOG2 워밍업: 최초 N프레임은 fg_ratio 신호 비활성
        self._frame_count: int = 0
        self._warmup_frames: int = 50

        # 쿨다운
        self._last_anomaly_ts: float = 0.0

    def check(self, frame: np.ndarray, timestamp: float) -> PreFilterResult:
        """프레임 이상 여부 판별.

        Args:
            frame: BGR 이미지 (640x480).
            timestamp: 현재 타임스탬프(초).

        Returns:
            PreFilterResult(is_anomaly, score, reasons, fg_ratio,
                            frame_diff_mean, hist_corr)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._frame_count += 1
        reasons: list[str] = []
        scores: list[float] = []

        # ── 쿨다운 조기 판정 ─────────────────────────────────────
        in_cooldown = (
            self._last_anomaly_ts > 0
            and (timestamp - self._last_anomaly_ts) < PREFILTER_COOLDOWN_SEC
        )

        # ── 1. MOG2 전경 비율 ────────────────────────────────────
        fg_mask = self._mog2.apply(gray, learningRate=0.005)
        fg_count = cv2.countNonZero(fg_mask)
        total_px = fg_mask.shape[0] * fg_mask.shape[1]
        fg_ratio = fg_count / total_px

        # 워밍업 기간에는 fg_ratio 신호 비활성 (MOG2 배경 안정화)
        if self._frame_count > self._warmup_frames:
            if fg_ratio < PREFILTER_FG_RATIO_LOW:
                reasons.append("fg_ratio_low")
                # 완전 정지는 보조 신호 → score 상한 0.3
                scores.append(min(0.3, 1.0 - fg_ratio / PREFILTER_FG_RATIO_LOW))
            elif fg_ratio > PREFILTER_FG_RATIO_HIGH:
                reasons.append("fg_ratio_high")
                scores.append(min(1.0, fg_ratio / PREFILTER_FG_RATIO_HIGH - 1.0 + 0.5))

        # ── 2. 프레임 차분 강도 ──────────────────────────────────
        if self._prev_gray is not None:
            diff = cv2.absdiff(gray, self._prev_gray)
            diff_mean = float(np.mean(diff))
        else:
            diff_mean = 0.0

        if diff_mean > PREFILTER_FRAME_DIFF_THRESHOLD:
            reasons.append("frame_diff")
            scores.append(
                min(1.0, diff_mean / (PREFILTER_FRAME_DIFF_THRESHOLD * 3))
            )

        # ── 3. 정지 객체 감지 ────────────────────────────────────
        static_detected = self._check_static_blobs(fg_mask, timestamp)
        if static_detected:
            reasons.append("static_object")
            scores.append(0.7)

        # ── 4. 히스토그램 급변 ───────────────────────────────────
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)

        if self._prev_hist is not None:
            hist_corr = cv2.compareHist(
                self._prev_hist, hist, cv2.HISTCMP_CORREL
            )
        else:
            hist_corr = 1.0

        if hist_corr < PREFILTER_HIST_DIFF_THRESHOLD:
            reasons.append("hist_change")
            scores.append(
                min(1.0, 1.0 - hist_corr / PREFILTER_HIST_DIFF_THRESHOLD)
            )

        # ── 상태 갱신 ────────────────────────────────────────────
        self._prev_gray = gray
        self._prev_hist = hist

        # ── 결과 조합 (OR) ───────────────────────────────────────
        is_anomaly = len(reasons) > 0
        score = max(scores) if scores else 0.0

        # 쿨다운 적용: 직전 이상 판정으로부터 N초 이내면 억제
        if is_anomaly and in_cooldown:
            is_anomaly = False
            reasons = [f"cooldown({r})" for r in reasons]

        # 쿨다운 타이머는 score > 0.3인 강한 이상에서만 갱신
        # (fg_ratio_low 같은 보조 신호가 쿨다운을 독점하는 것 방지)
        if is_anomaly and score > 0.3:
            self._last_anomaly_ts = timestamp

        return PreFilterResult(
            is_anomaly=is_anomaly,
            score=round(score, 4),
            reasons=reasons,
            fg_ratio=round(fg_ratio, 4),
            frame_diff_mean=round(diff_mean, 2),
            hist_corr=round(hist_corr, 4),
        )

    def reset(self) -> None:
        """상태 초기화 (스트림 재연결 시)."""
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False,
        )
        self._prev_gray = None
        self._prev_hist = None
        self._static_blobs.clear()
        self._frame_count = 0
        self._last_anomaly_ts = 0.0

    # ── 내부 헬퍼 ────────────────────────────────────────────────

    def _check_static_blobs(
        self, fg_mask: np.ndarray, timestamp: float,
    ) -> bool:
        """전경 마스크에서 정지 객체(큰 blob 지속) 감지.

        Returns:
            True면 PREFILTER_STATIC_DURATION_SEC 이상 지속된
            정지 객체가 존재.
        """
        # 형태학 연산: 노이즈 제거 + 인접 영역 병합
        cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, _MORPH_KERNEL)

        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        # 정지객체 최소면적을 프레임 크기에 비례화(0.5%) — 고해상도(2K) 과소탐지 방지.
        # 640x480에서 ~1536px로 기존 고정값(1500)과 호환.
        frame_area = fg_mask.shape[0] * fg_mask.shape[1]
        static_min_area = max(_STATIC_MIN_AREA, frame_area * 0.005)

        # 현재 프레임에서 발견된 큰 blob의 양자화 좌표
        current_keys: set[tuple[int, int]] = set()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < static_min_area:
                continue
            m = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            cx = int(m["m10"] / m["m00"]) // self._static_grid
            cy = int(m["m01"] / m["m00"]) // self._static_grid
            key = (cx, cy)
            current_keys.add(key)

            if key not in self._static_blobs:
                self._static_blobs[key] = timestamp

        # 사라진 blob 제거
        expired = [k for k in self._static_blobs if k not in current_keys]
        for k in expired:
            del self._static_blobs[k]

        # 지속 시간 체크
        for key in current_keys:
            first_seen = self._static_blobs.get(key)
            if first_seen is None:
                continue
            if (timestamp - first_seen) >= PREFILTER_STATIC_DURATION_SEC:
                return True

        return False
