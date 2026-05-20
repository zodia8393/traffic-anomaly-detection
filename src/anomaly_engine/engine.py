"""이상징후 탐지 엔진 — 메인 오케스트레이터.

VisionPipeline → FeatureStore → RuleEngine → EnsembleScorer → Alerter
전체 흐름을 통합하고, 프레임 단위로 실행한다.

사용법:
    engine = AnomalyEngine(camera_id="cam001")
    # 매 프레임
    result = engine.process_frame(
        timestamp=t,
        tracked_vehicles=vision_output["tracked_vehicles"],
        ttc_list=ttc_data,
        dt=1/fps,
    )
    if result.alert:
        handle_alert(result.alert)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .alerter import AlertEvent, Alerter
from .ensemble import EnsembleResult, EnsembleScorer
from .feature_store import FeatureStore
from .roi_config import CameraConfig, default_camera_config, load_camera_config
from .rule_engine import RuleEngine, RuleViolation

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """프레임 처리 결과."""
    timestamp: float
    features_updated: int
    violations: list[RuleViolation]
    ensemble: EnsembleResult
    alert: AlertEvent | None = None


class AnomalyEngine:
    """이상징후 탐지 엔진 통합 실행기."""

    def __init__(
        self,
        camera_id: str = "default",
        camera_config: CameraConfig | None = None,
        camera_config_path: Path | None = None,
        rules_path: Path | None = None,
        log_dir: Path | None = None,
        on_alert: Callable[[AlertEvent], None] | None = None,
        on_alarm: Callable[[AlertEvent], None] | None = None,
        alert_threshold: float = 0.3,
        alarm_threshold: float = 0.7,
    ):
        # Level 0: 카메라 설정
        if camera_config:
            self.camera_config = camera_config
        elif camera_config_path and camera_config_path.exists():
            self.camera_config = load_camera_config(camera_config_path)
        else:
            self.camera_config = default_camera_config(camera_id)

        # FeatureStore
        self.feature_store = FeatureStore(
            expected_heading=self.camera_config.expected_heading,
        )

        # Level 1: 규칙 엔진
        self.rule_engine = RuleEngine(
            rules_path=rules_path,
            feature_store=self.feature_store,
        )

        # 앙상블
        self.ensemble = EnsembleScorer(
            alert_threshold=alert_threshold,
            alarm_threshold=alarm_threshold,
        )

        # 알림
        self.alerter = Alerter(
            camera_id=camera_id,
            alert_threshold=alert_threshold,
            alarm_threshold=alarm_threshold,
            log_dir=log_dir,
            on_alert=on_alert,
            on_alarm=on_alarm,
        )

        self._frame_count = 0
        logger.info(
            "AnomalyEngine 초기화: camera=%s, rules=%d개",
            camera_id, len(self.rule_engine.rules),
        )

    def process_frame(
        self,
        timestamp: float,
        tracked_vehicles: list[dict[str, Any]],
        ttc_list: list[dict[str, Any]],
        dt: float = 1.0,
        ml_scores: dict[str, float] | None = None,
    ) -> FrameResult:
        """단일 프레임 처리 — 전체 파이프라인 실행.

        Args:
            timestamp: 현재 타임스탬프 (초).
            tracked_vehicles: VisionPipeline 출력.
            ttc_list: compute_all_ttc() 결과.
            dt: 프레임 간 시간 간격 (초).
            ml_scores: Level 2~4 ML 모델 점수 (향후 확장).
        """
        self._frame_count += 1

        # 1. FeatureStore 갱신
        features = self.feature_store.update(
            timestamp=timestamp,
            tracked_vehicles=tracked_vehicles,
            ttc_list=ttc_list,
            dt=dt,
        )

        # 2. 규칙 엔진 평가
        violations = self.rule_engine.evaluate(timestamp)

        # 3. 앙상블 점수 산출
        ensemble_result = self.ensemble.score(violations, ml_scores)

        # 4. 알림 판정
        alert_event = None
        if ensemble_result.trigger or ensemble_result.final_score >= self.alerter.alert_threshold:
            alert_event = self.alerter.process(ensemble_result, timestamp)

        return FrameResult(
            timestamp=timestamp,
            features_updated=len(features),
            violations=violations,
            ensemble=ensemble_result,
            alert=alert_event,
        )

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "frames_processed": self._frame_count,
            "active_tracks": len(self.feature_store.active_track_ids),
            "alerts": self.alerter.stats,
        }
