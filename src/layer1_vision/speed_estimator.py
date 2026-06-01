"""호모그래피 기반 속도 추정.

트래커의 픽셀 궤적을 실세계 좌표로 변환하여 속도(km/h)를 산출한다.
캘리브레이션 파일이 없으면 픽셀 이동거리 x 기본 스케일(0.05 m/px)로 근사.
"""

from __future__ import annotations
import sys as _sys; from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
import logging
import math
from pathlib import Path

import numpy as np

from config_new import SPEED_FALLBACK_SCALE

logger = logging.getLogger(__name__)


class SpeedEstimator:
    """픽셀 궤적 -> 실세계 속도(km/h) 변환기."""

    def __init__(self, calibration_file: Path | None = None) -> None:
        """캘리브레이션 JSON 로드.

        JSON 형식:
            {"homography_matrix": [[3x3]], "scale_m_per_px": float}

        Args:
            calibration_file: 캘리브레이션 JSON 경로. None이면 fallback.
        """
        self._H: np.ndarray | None = None
        self._scale_m_per_px: float = SPEED_FALLBACK_SCALE

        if calibration_file is not None and Path(calibration_file).exists():
            self._load_calibration(calibration_file)

    def _load_calibration(self, path: Path) -> None:
        """캘리브레이션 JSON 파싱."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "homography_matrix" in data:
                self._H = np.array(data["homography_matrix"], dtype=np.float64)
                if self._H.shape != (3, 3):
                    logger.warning("호모그래피 행렬 크기 오류: %s, fallback", self._H.shape)
                    self._H = None
            if "scale_m_per_px" in data:
                self._scale_m_per_px = float(data["scale_m_per_px"])
            logger.info("캘리브레이션 로드: %s (H=%s, scale=%.4f m/px)",
                        path.name, self._H is not None, self._scale_m_per_px)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("캘리브레이션 파싱 실패: %s — fallback", e)

    def _pixel_to_world(self, px: float, py: float) -> tuple[float, float]:
        """픽셀 좌표를 실세계 좌표(m)로 변환.

        호모그래피가 없으면 scale_m_per_px 적용.
        """
        if self._H is not None:
            pt = np.array([px, py, 1.0], dtype=np.float64)
            world = self._H @ pt
            # Z≈0 근처는 사영 발산(지평선 부근) — 1e-8로 상향, 발산 시 폴백
            if abs(world[2]) < 1e-8:
                return px * self._scale_m_per_px, py * self._scale_m_per_px
            return float(world[0] / world[2]), float(world[1] / world[2])
        return px * self._scale_m_per_px, py * self._scale_m_per_px

    def estimate(
        self,
        track_positions: list[tuple[float, float]],
        timestamps: list[float],
    ) -> list[float]:
        """프레임별 순간속도(km/h) 산출.

        Args:
            track_positions: [(px, py), ...] bbox 중심 좌표 시퀀스.
            timestamps: [sec, ...] 각 위치의 타임스탬프.

        Returns:
            list[float]: 각 프레임 간 속도(km/h). 길이 = len(positions) - 1.
                         첫 프레임은 속도 산출 불가이므로 제외.
        """
        if len(track_positions) < 2:
            return []
        if len(track_positions) != len(timestamps):
            raise ValueError(
                f"positions({len(track_positions)})과 timestamps({len(timestamps)}) 길이 불일치"
            )

        speeds: list[float] = []
        for i in range(1, len(track_positions)):
            dt = timestamps[i] - timestamps[i - 1]
            if dt <= 0:
                speeds.append(0.0)
                continue

            wx0, wy0 = self._pixel_to_world(*track_positions[i - 1])
            wx1, wy1 = self._pixel_to_world(*track_positions[i])
            dist_m = math.hypot(wx1 - wx0, wy1 - wy0)

            # m/s -> km/h
            speed_kmh = (dist_m / dt) * 3.6
            speeds.append(speed_kmh)

        return speeds

    @property
    def has_homography(self) -> bool:
        """호모그래피 행렬 보유 여부."""
        return self._H is not None

    @property
    def is_calibrated(self) -> bool:
        """실세계 캘리브레이션(호모그래피) 보유 여부.

        False면 산출 속도는 픽셀 스케일 폴백(상대값) — 절대 km/h로
        신뢰할 수 없으므로 MLLM 입력·보고서의 '절대속도'에서 제외해야 한다.
        """
        return self._H is not None
