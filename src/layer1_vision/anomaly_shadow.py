"""VisionPipeline용 AnomalyEngine shadow-mode 생성 헬퍼."""
from __future__ import annotations

import logging

from config_new import (
    ANOMALY_ALERT_THRESHOLD,
    ANOMALY_ALARM_THRESHOLD,
    ANOMALY_SHADOW_LOG_DIR,
    ANOMALY_SHADOW_MIN_SCORE,
    ENABLE_ANOMALY_SHADOW,
)

logger = logging.getLogger(__name__)


def build_shadow_anomaly_engine(camera_id: str = "default"):
    """설정이 켜진 경우 알림 없는 AnomalyEngine을 생성한다.

    Shadow engine은 final_score, trigger_reason, violations, ml_scores를 JSONL로
    남기지만 기존 TriggerDetector 기반 녹화/MLLM 흐름에는 영향을 주지 않는다.
    """
    if not ENABLE_ANOMALY_SHADOW:
        return None
    try:
        from anomaly_engine import AnomalyEngine
        return AnomalyEngine(
            camera_id=camera_id,
            alert_threshold=ANOMALY_ALERT_THRESHOLD,
            alarm_threshold=ANOMALY_ALARM_THRESHOLD,
            emit_alerts=False,
            shadow_log_dir=ANOMALY_SHADOW_LOG_DIR,
            shadow_min_score=ANOMALY_SHADOW_MIN_SCORE,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("AnomalyEngine shadow 생성 실패(%s): %s", camera_id, e)
        return None
