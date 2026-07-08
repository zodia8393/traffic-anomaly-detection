"""지도 사고감지기 추론 래퍼 — 엔진 통합용.

train_supervised_detector.py가 학습한 모델(궤적 운동학특징 → 비정상주행 확률)을
로드하여, 트랙 히스토리 윈도우에서 이상확률을 산출한다. STGAE(AUROC 0.49) 대체.
활성화 게이트(AUROC≥0.6)를 통과하므로 앙상블 ML 투표에 참여한다.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

try:
    from .trajectory_features import extract_features
except ImportError:  # 스크립트 직접 실행 시
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parent))
    from trajectory_features import extract_features

logger = logging.getLogger(__name__)

DEFAULT_MODEL = Path("/workspace/prj/work/AI기반 교통상황 대응 기술 개발 연구/사고분석_설계/models/supervised_detector.pkl")


class SupervisedTrajectoryDetector:
    """궤적 운동학 기반 지도 사고감지기.

    score(track_history) -> 비정상주행 확률(0~1). 모델 미로드 시 None.
    """

    def __init__(self, model_path: str | Path = DEFAULT_MODEL):
        self._model = None
        self._threshold = 0.5
        self._auroc = 0.0
        self._load(Path(model_path))

    def _load(self, path: Path) -> None:
        if not path.exists():
            logger.warning("지도감지기 모델 없음: %s", path)
            return
        try:
            import joblib
            b = joblib.load(path)
            self._model = b["model"]
            self._threshold = b.get("threshold", 0.5)
            self._auroc = b.get("auroc", 0.0)
            logger.info("지도 사고감지기 로드: AUROC %.3f, 임계 %.2f", self._auroc, self._threshold)
        except Exception as e:  # noqa: BLE001
            logger.error("지도감지기 로드 실패: %s", e)
            self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def score(self, track_history: dict[int, list]) -> float | None:
        """트랙 히스토리 → 비정상 확률.

        Args:
            track_history: {track_id: [{"bbox":[x1,y1,x2,y2],"center":(x,y),
                           "timestamp":..}, ...]} (vision_pipeline 형식)
                           또는 {track_id: [(frame,x,y),...]} (raw 형식).
        Returns:
            0~1 확률. 추출 불가/미로드 시 None.
        """
        if self._model is None:
            return None
        tracks = self._normalize(track_history)
        feat = extract_features(tracks)
        if feat is None:
            return None
        try:
            import warnings
            with warnings.catch_warnings():
                # LightGBM: numpy 입력 시 'feature names' 경고 — 무해(순서 보존), 로그 스팸 억제
                warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
                return float(self._model.predict_proba(feat.reshape(1, -1))[0, 1])
        except Exception as e:  # noqa: BLE001
            logger.warning("지도감지기 추론 실패: %s", e)
            return None

    @staticmethod
    def _normalize(track_history: dict[int, list]) -> dict[int, list]:
        """vision_pipeline 트랙형식을 (frame,x,y) 시퀀스로 변환."""
        out: dict[int, list] = {}
        for tid, pts in track_history.items():
            seq = []
            for i, p in enumerate(pts):
                if isinstance(p, dict):
                    c = p.get("center")
                    if c is None:
                        b = p.get("bbox")
                        if not b:
                            continue
                        c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                    seq.append((i, float(c[0]), float(c[1])))
                elif isinstance(p, (tuple, list)) and len(p) >= 3:
                    seq.append((int(p[0]), float(p[1]), float(p[2])))
            if seq:
                out[tid] = seq
        return out
