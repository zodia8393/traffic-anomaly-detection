"""2단계 알림 체계: Alert / Alarm.

| 수준  | 조건           | 동작                              |
|-------|----------------|-----------------------------------|
| ALERT | score 0.3~0.7  | 로그 + Ring Buffer flush          |
| ALARM | score >= 0.7   | API 조회 + 운영자 통보 + DB 저장  |

Rate limiting: 카메라당 시간당 최대 10건 (ALARM 기준).
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .ensemble import EnsembleResult

logger = logging.getLogger(__name__)

MAX_ALARMS_PER_HOUR = 10


@dataclass
class AlertEvent:
    """알림 이벤트."""
    level: str                         # "ALERT" | "ALARM"
    timestamp: float
    camera_id: str
    final_score: float
    trigger_reason: str
    involved_tracks: list[int] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


class Alerter:
    """2단계 알림 관리자."""

    def __init__(
        self,
        camera_id: str = "default",
        alert_threshold: float = 0.3,
        alarm_threshold: float = 0.7,
        log_dir: Path | None = None,
        on_alert: Callable[[AlertEvent], None] | None = None,
        on_alarm: Callable[[AlertEvent], None] | None = None,
    ):
        self.camera_id = camera_id
        self.alert_threshold = alert_threshold
        self.alarm_threshold = alarm_threshold
        self.log_dir = log_dir
        self._on_alert = on_alert
        self._on_alarm = on_alarm
        self._alarm_times: deque[float] = deque(maxlen=MAX_ALARMS_PER_HOUR)
        self._alert_count = 0
        self._alarm_count = 0

    def process(
        self, result: EnsembleResult, timestamp: float | None = None,
    ) -> AlertEvent | None:
        """앙상블 결과를 받아 알림 수준 판정 + 콜백 실행.

        Returns:
            AlertEvent (ALERT 또는 ALARM), 임계값 미달 시 None.
        """
        ts = timestamp or time.time()

        if result.final_score < self.alert_threshold:
            return None

        if result.final_score >= self.alarm_threshold:
            if self._is_rate_limited(ts):
                logger.warning(
                    "[%s] ALARM 억제 (시간당 %d건 초과)",
                    self.camera_id, MAX_ALARMS_PER_HOUR,
                )
                return self._make_alert(result, ts)
            return self._make_alarm(result, ts)

        return self._make_alert(result, ts)

    def _make_alert(self, result: EnsembleResult, ts: float) -> AlertEvent:
        event = AlertEvent(
            level="ALERT",
            timestamp=ts,
            camera_id=self.camera_id,
            final_score=result.final_score,
            trigger_reason=result.trigger_reason,
            involved_tracks=result.involved_tracks,
            details=self._extract_details(result),
        )
        self._alert_count += 1
        logger.info(
            "[%s] ALERT score=%.2f reason=%s tracks=%s",
            self.camera_id, result.final_score,
            result.trigger_reason, result.involved_tracks,
        )
        self._log_event(event)
        if self._on_alert:
            self._on_alert(event)
        return event

    def _make_alarm(self, result: EnsembleResult, ts: float) -> AlertEvent:
        event = AlertEvent(
            level="ALARM",
            timestamp=ts,
            camera_id=self.camera_id,
            final_score=result.final_score,
            trigger_reason=result.trigger_reason,
            involved_tracks=result.involved_tracks,
            details=self._extract_details(result),
        )
        self._alarm_count += 1
        self._alarm_times.append(ts)
        logger.warning(
            "[%s] ALARM score=%.2f reason=%s tracks=%s",
            self.camera_id, result.final_score,
            result.trigger_reason, result.involved_tracks,
        )
        self._log_event(event)
        if self._on_alarm:
            self._on_alarm(event)
        return event

    def _is_rate_limited(self, ts: float) -> bool:
        cutoff = ts - 3600.0
        recent = sum(1 for t in self._alarm_times if t > cutoff)
        return recent >= MAX_ALARMS_PER_HOUR

    def _extract_details(self, result: EnsembleResult) -> dict[str, Any]:
        details: dict[str, Any] = {"ml_scores": result.ml_scores}
        if result.top_violations:
            details["violations"] = [
                {
                    "rule_id": v.rule_id,
                    "label": v.label,
                    "severity": v.severity,
                    "score": v.score,
                    "tracks": v.involved_tracks,
                }
                for v in result.top_violations[:5]
            ]
        return details

    def _log_event(self, event: AlertEvent) -> None:
        if self.log_dir is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        dt = datetime.fromtimestamp(event.timestamp)
        fname = f"{event.level}_{self.camera_id}_{dt:%Y%m%d_%H%M%S}.json"
        path = self.log_dir / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(event), f, ensure_ascii=False, indent=2, default=str)

    @property
    def stats(self) -> dict[str, int]:
        return {"alerts": self._alert_count, "alarms": self._alarm_count}
