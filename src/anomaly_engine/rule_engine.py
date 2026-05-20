"""Level 1: YAML 선언적 규칙 엔진.

rules_default.yaml에서 규칙을 로드하고,
FeatureStore의 특징 벡터를 평가하여 이상징후를 판정한다.

지원 규칙 타입:
  - single: 개별 차량 특징 → 임계값 비교
  - pair:   두 차량 관계 (TTC, 정지차량 접근)
  - group:  다수 차량 집단 행동 (군집 급감속)
  - special: 커스텀 핸들러 (보행자 차도 진입)
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .feature_store import FEATURE_NAMES, FeatureStore

logger = logging.getLogger(__name__)

SEVERITY_SCORES = {"CRITICAL": 1.0, "HIGH": 0.8, "MEDIUM": 0.5, "LOW": 0.3}
OPERATORS = {
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "==": lambda a, b: a == b,
}


@dataclass
class RuleViolation:
    """규칙 위반 이벤트."""
    rule_id: str
    rule_name: str
    label: str
    severity: str
    score: float
    involved_tracks: list[int] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleSpec:
    """파싱된 규칙 명세."""
    id: str
    name: str
    label: str
    severity: str
    rule_type: str
    feature: str | None = None
    operator: str | None = None
    threshold: float | None = None
    conditions: list[dict] | None = None
    handler: str | None = None
    params: dict = field(default_factory=dict)
    filters: dict = field(default_factory=dict)
    min_count: int = 1
    spatial_radius: float = 200.0


class RuleEngine:
    """YAML 규칙 엔진."""

    def __init__(
        self,
        rules_path: Path | None = None,
        feature_store: FeatureStore | None = None,
    ):
        if rules_path is None:
            rules_path = Path(__file__).parent / "rules_default.yaml"
        self._rules = _load_rules(rules_path)
        self._store = feature_store
        # persistence 카운터: {(rule_id, track_id): 연속 위반 프레임 수}
        self._persistence: dict[tuple[str, int], int] = defaultdict(int)
        logger.info("규칙 엔진 초기화: %d개 규칙 로드", len(self._rules))

    @property
    def rules(self) -> list[RuleSpec]:
        return self._rules

    def set_feature_store(self, store: FeatureStore) -> None:
        self._store = store

    def evaluate(self, timestamp: float = 0.0) -> list[RuleViolation]:
        """전체 규칙을 현재 FeatureStore 상태에 대해 평가.

        Returns:
            발생한 위반 이벤트 리스트 (severity 내림차순).
        """
        if self._store is None:
            return []

        violations: list[RuleViolation] = []
        for rule in self._rules:
            if rule.rule_type == "single":
                violations.extend(self._eval_single(rule))
            elif rule.rule_type == "pair":
                violations.extend(self._eval_pair(rule))
            elif rule.rule_type == "group":
                violations.extend(self._eval_group(rule))
            elif rule.rule_type == "special":
                violations.extend(self._eval_special(rule))

        violations.sort(
            key=lambda v: SEVERITY_SCORES.get(v.severity, 0),
            reverse=True,
        )
        return violations

    def _eval_single(self, rule: RuleSpec) -> list[RuleViolation]:
        violations = []
        has_conditions = rule.conditions is not None

        for tid in self._store.active_track_ids:
            feat = self._store.get_features_ma(tid)
            if feat is None:
                continue

            if not self._pass_filters(rule, tid, feat):
                continue

            triggered = False

            if has_conditions:
                triggered = all(
                    self._check_condition(feat, c) for c in rule.conditions
                )
            elif rule.feature and rule.operator and rule.threshold is not None:
                triggered = self._check_feature(
                    feat, rule.feature, rule.operator, rule.threshold,
                )

            key = (rule.id, tid)
            if triggered:
                self._persistence[key] += 1
                persistence_req = rule.filters.get("persistence", 1)
                if self._persistence[key] >= persistence_req:
                    score = self._compute_score(rule, feat)
                    violations.append(RuleViolation(
                        rule_id=rule.id,
                        rule_name=rule.name,
                        label=rule.label,
                        severity=rule.severity,
                        score=score,
                        involved_tracks=[tid],
                        details=self._build_details(rule, feat, tid),
                    ))
            else:
                self._persistence[key] = 0

        return violations

    def _eval_pair(self, rule: RuleSpec) -> list[RuleViolation]:
        if rule.handler == "approach_stopped_vehicle":
            return self._handler_approach_stopped(rule)

        # R4: TTC 기반
        if rule.feature == "ttc_min":
            return self._eval_ttc_pair(rule)

        return []

    def _eval_ttc_pair(self, rule: RuleSpec) -> list[RuleViolation]:
        violations = []
        for tid in self._store.active_track_ids:
            feat = self._store.get_features_ma(tid)
            if feat is None:
                continue
            if not self._pass_filters(rule, tid, feat):
                continue

            ttc = feat[6]  # ttc_min
            if ttc < 999.0 and OPERATORS[rule.operator](ttc, rule.threshold):
                score = max(0.0, 1.0 - ttc / rule.threshold) if rule.threshold > 0 else 1.0
                violations.append(RuleViolation(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    label=rule.label,
                    severity=rule.severity,
                    score=min(1.0, score * SEVERITY_SCORES[rule.severity]),
                    involved_tracks=[tid],
                    details={"ttc": float(ttc)},
                ))
        return violations

    def _handler_approach_stopped(self, rule: RuleSpec) -> list[RuleViolation]:
        """R10: 정지 차량에 접근하는 차량 탐지."""
        violations = []
        params = rule.params
        stop_speed = params.get("stop_speed", 5.0)
        ttc_thr = params.get("ttc_threshold", 5.0)

        stopped_ids = []
        moving_ids = []
        for tid in self._store.active_track_ids:
            feat = self._store.get_features(tid)
            if feat is None:
                continue
            if feat[0] <= stop_speed and feat[10] >= 5.0:
                stopped_ids.append(tid)
            elif feat[0] > stop_speed:
                moving_ids.append(tid)

        if not stopped_ids:
            return violations

        for mid in moving_ids:
            m_feat = self._store.get_features(mid)
            if m_feat is None or not self._pass_filters(rule, mid, m_feat):
                continue
            ttc = m_feat[6]
            if ttc < ttc_thr:
                score = max(0.0, 1.0 - ttc / ttc_thr)
                violations.append(RuleViolation(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    label=rule.label,
                    severity=rule.severity,
                    score=min(1.0, score * SEVERITY_SCORES[rule.severity]),
                    involved_tracks=[mid] + stopped_ids[:1],
                    details={"ttc": float(ttc), "stopped_count": len(stopped_ids)},
                ))
        return violations

    def _eval_group(self, rule: RuleSpec) -> list[RuleViolation]:
        """R9: 군집 급감속 — 공간 범위 내 N대 이상 동시 위반."""
        if rule.feature is None or rule.operator is None or rule.threshold is None:
            return []

        violators: list[tuple[int, tuple[float, float]]] = []
        for tid in self._store.active_track_ids:
            feat = self._store.get_features_ma(tid)
            if feat is None:
                continue
            if self._check_feature(feat, rule.feature, rule.operator, rule.threshold):
                state = self._store.get_state(tid)
                if state and state.centers:
                    violators.append((tid, state.centers[-1]))

        if len(violators) < rule.min_count:
            return []

        # 공간 클러스터링 (greedy)
        clusters = _spatial_cluster(violators, rule.spatial_radius)
        violations = []
        for cluster in clusters:
            if len(cluster) >= rule.min_count:
                score = min(1.0, len(cluster) / (rule.min_count * 2))
                violations.append(RuleViolation(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    label=rule.label,
                    severity=rule.severity,
                    score=score * SEVERITY_SCORES[rule.severity],
                    involved_tracks=[tid for tid, _ in cluster],
                    details={"count": len(cluster), "radius": rule.spatial_radius},
                ))
        return violations

    def _eval_special(self, rule: RuleSpec) -> list[RuleViolation]:
        """R6: 보행자 차도 진입."""
        if rule.handler != "pedestrian_road_intrusion":
            return []

        violations = []
        for tid in self._store.active_track_ids:
            state = self._store.get_state(tid)
            if state is None:
                continue
            # 보행자 클래스 확인 (YOLO 'person' 또는 차종분류 None)
            if state.cls not in ("person", "pedestrian"):
                continue
            feat = self._store.get_features(tid)
            if feat is None:
                continue
            if not self._pass_filters(rule, tid, feat):
                continue
            violations.append(RuleViolation(
                rule_id=rule.id,
                rule_name=rule.name,
                label=rule.label,
                severity=rule.severity,
                score=SEVERITY_SCORES[rule.severity],
                involved_tracks=[tid],
                details={"cls": state.cls},
            ))
        return violations

    def _pass_filters(self, rule: RuleSpec, tid: int, feat: np.ndarray) -> bool:
        """필터 조건 확인."""
        filters = rule.filters
        if not filters:
            return True
        if feat[13] < filters.get("min_track_age", 0):
            return False
        if "min_speed" in filters and feat[0] < filters["min_speed"]:
            return False
        return True

    def _check_feature(
        self, feat: np.ndarray, feature_name: str,
        operator: str, threshold: float,
    ) -> bool:
        idx = FEATURE_NAMES.index(feature_name)
        return OPERATORS[operator](feat[idx], threshold)

    def _check_condition(self, feat: np.ndarray, cond: dict) -> bool:
        return self._check_feature(
            feat, cond["feature"], cond["operator"], cond["threshold"],
        )

    def _compute_score(self, rule: RuleSpec, feat: np.ndarray) -> float:
        """위반 정도에 비례하는 점수 산출 (0~1)."""
        base = SEVERITY_SCORES.get(rule.severity, 0.5)
        if rule.feature and rule.threshold:
            idx = FEATURE_NAMES.index(rule.feature)
            val = feat[idx]
            if rule.threshold != 0:
                ratio = abs(val / rule.threshold)
                excess = max(0.0, ratio - 1.0)
                return min(1.0, base * (1.0 + excess * 0.5))
        return base

    def _build_details(
        self, rule: RuleSpec, feat: np.ndarray, tid: int,
    ) -> dict[str, Any]:
        details: dict[str, Any] = {}
        if rule.feature:
            idx = FEATURE_NAMES.index(rule.feature)
            details[rule.feature] = float(feat[idx])
            details["threshold"] = rule.threshold
        if rule.conditions:
            for c in rule.conditions:
                idx = FEATURE_NAMES.index(c["feature"])
                details[c["feature"]] = float(feat[idx])
        details["speed_kmh"] = float(feat[0])
        details["track_age"] = int(feat[13])
        return details


def _load_rules(path: Path) -> list[RuleSpec]:
    """YAML에서 규칙 로드."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    rules = []
    for r in data.get("rules", []):
        rules.append(RuleSpec(
            id=r["id"],
            name=r["name"],
            label=r["label"],
            severity=r["severity"],
            rule_type=r["type"],
            feature=r.get("feature"),
            operator=r.get("operator"),
            threshold=r.get("threshold"),
            conditions=r.get("conditions"),
            handler=r.get("handler"),
            params=r.get("params", {}),
            filters=r.get("filters", {}),
            min_count=r.get("min_count", 1),
            spatial_radius=r.get("spatial_radius", 200.0),
        ))
    return rules


def _spatial_cluster(
    items: list[tuple[int, tuple[float, float]]],
    radius: float,
) -> list[list[tuple[int, tuple[float, float]]]]:
    """탐욕적 공간 클러스터링."""
    used = set()
    clusters = []
    for i, (tid_i, pos_i) in enumerate(items):
        if i in used:
            continue
        cluster = [(tid_i, pos_i)]
        used.add(i)
        for j, (tid_j, pos_j) in enumerate(items):
            if j in used:
                continue
            dist = math.hypot(pos_j[0] - pos_i[0], pos_j[1] - pos_i[1])
            if dist <= radius:
                cluster.append((tid_j, pos_j))
                used.add(j)
        clusters.append(cluster)
    return clusters
