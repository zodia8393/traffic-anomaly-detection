"""an1(방향지시등 불이행) 특화 감지기 — per-vehicle 깜빡이 기반.

전체 분류기는 an1을 못 잡음(0.44): an1(차선변경+무신호)·정상직진(무신호)이 clip
집계론 동일. 본 특화모델은 '차선변경 차량'에 한정해 깜빡이를 보고 판정 → AUROC 0.89.

계층 사용: 차선변경 감지(횡이동 큰 차량) 시에만 호출. 깜빡이(영상) 필요.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

try:
    from .trajectory_features import extract_features
    from .turn_signal_detector import blink_score
except ImportError:
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parent))
    from trajectory_features import extract_features
    from turn_signal_detector import blink_score

logger = logging.getLogger(__name__)
DEFAULT = Path("/workspace/prj_cctv/사고분석_설계/models/an1_specialist.pkl")
LANE_CHANGE_LAT = 0.05  # 차선변경 판정 횡이동 임계 (특화모델 발동 조건)


class An1Specialist:
    """차선변경 차량의 방향지시등 불이행 확률(0~1). 깜빡이가 결정 특징."""

    def __init__(self, model_path: str | Path = DEFAULT):
        self._model = None
        self._auroc = 0.0
        p = Path(model_path)
        if p.exists():
            try:
                import joblib
                b = joblib.load(p)
                self._model = b["model"]
                self._auroc = b.get("auroc", 0.0)
                logger.info("an1 특화모델 로드: AUROC %.3f", self._auroc)
            except Exception as e:  # noqa: BLE001
                logger.error("an1 특화 로드 실패: %s", e)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @staticmethod
    def lateral_total(track_pts: list, lat_axis: int = 0) -> float:
        """트랙의 횡방향 총이동 (차선변경 크기)."""
        if len(track_pts) < 3:
            return 0.0
        xy = np.array([(p[1], p[2]) for p in sorted(track_pts)])
        return float(np.abs(np.diff(xy[:, lat_axis])).sum())

    def score(self, tracks: dict[int, list], amber_series: list[float]) -> float | None:
        """차선변경 클립 → an1 확률.

        Args:
            tracks: {track_id: [(frame,x,y),...]} 정규화 좌표.
            amber_series: 차선변경 차량의 프레임별 amber 강도(깜빡이 검출용).
        """
        if self._model is None:
            return None
        feat = extract_features(tracks)
        if feat is None:
            return None
        # 종/횡축 결정 후 차선변경 차량 횡이동
        all_d = [np.diff(np.array([(x, y) for _, x, y in sorted(p)]), axis=0)
                 for p in tracks.values() if len(p) >= 2]
        lat_axis = 0
        if all_d:
            D = np.concatenate(all_d)
            lat_axis = 0 if D[:, 0].std() < D[:, 1].std() else 1
        lat_max = max((self.lateral_total(p, lat_axis) for p in tracks.values()), default=0.0)
        blink = blink_score(amber_series) if len(amber_series) >= 8 else 0.0
        full = np.concatenate([feat, [blink, lat_max]]).reshape(1, -1)
        try:
            return float(self._model.predict_proba(full)[0, 1])
        except Exception as e:  # noqa: BLE001
            logger.warning("an1 특화 추론 실패: %s", e)
            return None

    @staticmethod
    def has_lane_change(tracks: dict[int, list]) -> bool:
        """차선변경 발생 여부(특화모델 발동 조건)."""
        all_d = [np.diff(np.array([(x, y) for _, x, y in sorted(p)]), axis=0)
                 for p in tracks.values() if len(p) >= 2]
        if not all_d:
            return False
        D = np.concatenate(all_d)
        lat_axis = 0 if D[:, 0].std() < D[:, 1].std() else 1
        return any(An1Specialist.lateral_total(p, lat_axis) > LANE_CHANGE_LAT
                   for p in tracks.values())
