"""앙상블 스코어러: 계층적 (규칙 우선 + ML 보완).

설계 회의 결정:
  1. 규칙 CRITICAL/HIGH → 즉시 트리거
  2. ML 2/3 투표 동의 → 트리거
  3. ML 단일 확신(>0.9) → 트리거
  4. 규칙 MEDIUM + ML 보강 → 트리거

현재 Level 1만 구현. Level 2~4 추가 시 ml_scores를 채우면
앙상블 로직이 자동으로 ML 투표를 반영한다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .rule_engine import RuleViolation

logger = logging.getLogger(__name__)

SEVERITY_ORDER = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}


@dataclass
class EnsembleResult:
    """앙상블 최종 판정."""
    final_score: float
    trigger: bool
    trigger_reason: str
    top_violations: list[RuleViolation] = field(default_factory=list)
    ml_scores: dict[str, float] = field(default_factory=dict)
    involved_tracks: list[int] = field(default_factory=list)


class EnsembleScorer:
    """계층적 앙상블."""

    def __init__(
        self,
        alert_threshold: float = 0.3,
        alarm_threshold: float = 0.7,
        ml_vote_threshold: int = 2,
        ml_confidence_threshold: float = 0.9,
    ):
        self.alert_threshold = alert_threshold
        self.alarm_threshold = alarm_threshold
        self.ml_vote_threshold = ml_vote_threshold
        self.ml_confidence_threshold = ml_confidence_threshold
        # 활성화 게이트 통과 모델만 ML 투표 허용 (STGAE 등 미검증 모델 격리)
        try:
            from .activation_gate import active_ml_models
            self._active_ml = active_ml_models()
        except Exception as e:  # noqa: BLE001 — 게이트 로드 실패 시 ML 전면 차단(안전)
            logger.warning("활성화 게이트 로드 실패, ML 비활성: %s", e)
            self._active_ml = set()

    def _gate_ml(self, ml_scores: dict[str, float]) -> dict[str, float]:
        """게이트 미통과 모델(STGAE 등)을 ML 점수에서 제외."""
        if not ml_scores:
            return {}
        gated = {k: v for k, v in ml_scores.items() if k in self._active_ml}
        dropped = set(ml_scores) - set(gated)
        if dropped:
            logger.debug("ML 게이트 제외: %s", dropped)
        return gated

    def score(
        self,
        rule_violations: list[RuleViolation],
        ml_scores: dict[str, float] | None = None,
    ) -> EnsembleResult:
        """규칙 위반 + ML 점수를 종합하여 최종 판정.

        Args:
            rule_violations: RuleEngine.evaluate() 결과.
            ml_scores: {"level2_iforest": 0.8, "level3_lstm": 0.6, ...}
                       Level 2~4 구현 후 채워짐.
        Returns:
            EnsembleResult.
        """
        # 활성화 게이트: 검증 통과 모델만 투표 (STGAE 등 격리)
        ml_scores = self._gate_ml(ml_scores or {})

        if not rule_violations and not ml_scores:
            return EnsembleResult(
                final_score=0.0, trigger=False, trigger_reason="no_signal",
            )

        # 1. 규칙 CRITICAL/HIGH → 즉시 트리거
        critical_high = [
            v for v in rule_violations
            if v.severity in ("CRITICAL", "HIGH")
        ]
        if critical_high:
            top = critical_high[0]
            all_tracks = _collect_tracks(critical_high)
            score = max(v.score for v in critical_high)
            return EnsembleResult(
                final_score=min(1.0, score),
                trigger=True,
                trigger_reason=f"rule_{top.rule_id}_{top.severity}",
                top_violations=critical_high[:5],
                ml_scores=ml_scores,
                involved_tracks=all_tracks,
            )

        # 2. ML 2/3 투표 동의
        ml_triggers = {k: v for k, v in ml_scores.items() if v >= 0.5}
        if len(ml_triggers) >= self.ml_vote_threshold:
            score = sum(ml_triggers.values()) / len(ml_triggers)
            return EnsembleResult(
                final_score=min(1.0, score),
                trigger=True,
                trigger_reason=f"ml_vote_{len(ml_triggers)}",
                top_violations=rule_violations[:3],
                ml_scores=ml_scores,
                involved_tracks=_collect_tracks(rule_violations),
            )

        # 3. ML 단일 확신(>0.9)
        high_conf_ml = {k: v for k, v in ml_scores.items() if v >= self.ml_confidence_threshold}
        if high_conf_ml:
            best_k = max(high_conf_ml, key=high_conf_ml.get)
            return EnsembleResult(
                final_score=high_conf_ml[best_k],
                trigger=True,
                trigger_reason=f"ml_high_conf_{best_k}",
                top_violations=rule_violations[:3],
                ml_scores=ml_scores,
                involved_tracks=_collect_tracks(rule_violations),
            )

        # 4. 규칙 MEDIUM + ML 보강
        medium_rules = [v for v in rule_violations if v.severity == "MEDIUM"]
        ml_support = any(v >= 0.3 for v in ml_scores.values())
        if medium_rules and ml_support:
            score = max(v.score for v in medium_rules)
            ml_avg = sum(ml_scores.values()) / len(ml_scores) if ml_scores else 0
            combined = (score + ml_avg) / 2
            return EnsembleResult(
                final_score=min(1.0, combined),
                trigger=True,
                trigger_reason=f"rule_medium_ml_support",
                top_violations=medium_rules[:3],
                ml_scores=ml_scores,
                involved_tracks=_collect_tracks(medium_rules),
            )

        # 5. 규칙만 (MEDIUM/LOW) — 트리거 안 하되 점수 산출
        if rule_violations:
            score = max(v.score for v in rule_violations)
            return EnsembleResult(
                final_score=score,
                trigger=False,
                trigger_reason="below_threshold",
                top_violations=rule_violations[:3],
                ml_scores=ml_scores,
                involved_tracks=_collect_tracks(rule_violations),
            )

        # 6. ML만 (낮은 점수)
        if ml_scores:
            score = max(ml_scores.values())
            return EnsembleResult(
                final_score=score,
                trigger=score >= self.alert_threshold,
                trigger_reason="ml_only",
                ml_scores=ml_scores,
            )

        return EnsembleResult(
            final_score=0.0, trigger=False, trigger_reason="no_signal",
        )


def _collect_tracks(violations: list[RuleViolation]) -> list[int]:
    seen = set()
    result = []
    for v in violations:
        for tid in v.involved_tracks:
            if tid not in seen:
                seen.add(tid)
                result.append(tid)
    return result
