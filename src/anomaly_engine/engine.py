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
import json
from dataclasses import dataclass
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
    shadow: bool = False


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
        emit_alerts: bool = True,
        shadow_log_dir: Path | None = None,
        shadow_min_score: float = 0.3,
    ):
        self.camera_id = camera_id
        self.emit_alerts = emit_alerts
        self.shadow_log_dir = shadow_log_dir
        self.shadow_min_score = shadow_min_score

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
            "AnomalyEngine 초기화: camera=%s, rules=%d개, emit_alerts=%s",
            camera_id, len(self.rule_engine.rules), self.emit_alerts,
        )

    def process_frame(
        self,
        timestamp: float,
        tracked_vehicles: list[dict[str, Any]],
        ttc_list: list[dict[str, Any]],
        dt: float = 1.0,
        ml_scores: dict[str, float] | None = None,
        frame_idx: int | None = None,
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
        if self.emit_alerts and (
            ensemble_result.trigger or ensemble_result.final_score >= self.alerter.alert_threshold
        ):
            alert_event = self.alerter.process(ensemble_result, timestamp)

        result = FrameResult(
            timestamp=timestamp,
            features_updated=len(features),
            violations=violations,
            ensemble=ensemble_result,
            alert=alert_event,
            shadow=not self.emit_alerts,
        )
        if not self.emit_alerts:
            self._write_shadow_event(result, frame_idx)
        return result

    def _write_shadow_event(self, result: FrameResult, frame_idx: int | None) -> None:
        """Shadow mode 진단 이벤트를 JSONL로 저장한다.

        운영 트리거와 분리된 관측 로그다. 점수·규칙·ML 신호가 모두 없으면 기록하지
        않아 장기 운영 시 로그 폭주를 줄인다.
        """
        if self.shadow_log_dir is None:
            return
        if (
            result.ensemble.final_score < self.shadow_min_score
            and not result.violations
            and not result.ensemble.ml_scores
        ):
            return
        self.shadow_log_dir.mkdir(parents=True, exist_ok=True)
        path = self.shadow_log_dir / f"{self.camera_id}.jsonl"
        row = {
            "camera_id": self.camera_id,
            "frame_idx": frame_idx,
            "timestamp": result.timestamp,
            "features_updated": result.features_updated,
            "final_score": float(result.ensemble.final_score),
            "trigger": bool(result.ensemble.trigger),
            "trigger_reason": result.ensemble.trigger_reason,
            "ml_scores": {k: float(v) for k, v in result.ensemble.ml_scores.items()},
            "violations": [
                {
                    "rule_id": v.rule_id,
                    "label": v.label,
                    "severity": v.severity,
                    "score": float(v.score),
                    "tracks": v.involved_tracks,
                    "details": v.details,
                }
                for v in result.violations[:5]
            ],
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "frames_processed": self._frame_count,
            "active_tracks": len(self.feature_store.active_track_ids),
            "alerts": self.alerter.stats,
        }
