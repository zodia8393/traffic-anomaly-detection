"""위험 구간 순위화 및 히트맵."""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import duckdb

from config_new import DUCKDB_PATH, L4_RISK_REPORTS
from feature_engineer import FeatureEngineer, FEATURE_NAMES

logger = logging.getLogger(__name__)


def _classify_risk(probability: float) -> str:
    """확률값을 위험도 레벨 문자열로 변환."""
    if probability >= 0.7:
        return "critical"
    if probability >= 0.5:
        return "high"
    if probability >= 0.3:
        return "medium"
    return "low"


class RiskScorer:
    """전체 IC 위험도 순위화 + 히트맵 데이터 생성."""

    def __init__(self, predictor, db_path: str = DUCKDB_PATH) -> None:
        """
        Args:
            predictor: AccidentPredictor 인스턴스 (predict 메서드 필요).
            db_path: DuckDB 경로.
        """
        self.predictor = predictor
        self.db_path = db_path
        self._fe = FeatureEngineer(db_path)

    # ------------------------------------------------------------------
    # 전체 IC 위험도 순위
    # ------------------------------------------------------------------
    def score_all_ics(self, timestamp: datetime | None = None) -> list[dict]:
        """전체 IC 위험도 순위 산출.

        가장 최근 집계 구간의 피처로 예측하고 위험도 내림차순 정렬.

        Args:
            timestamp: 기준 시각. None이면 현재 시각.

        Returns:
            [{"ic_name", "risk_score", "risk_level", "factors"}, ...]
            risk_score 내림차순.
        """
        if timestamp is None:
            timestamp = datetime.now()

        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            ic_rows = conn.execute("""
                SELECT DISTINCT ic_name FROM traffic_agg
            """).fetchall()
        finally:
            conn.close()

        if not ic_rows:
            logger.warning("traffic_agg에 IC 데이터 없음")
            return []

        results: list[dict] = []
        period_end = timestamp
        period_start = timestamp - timedelta(minutes=5)

        for (ic_name,) in ic_rows:
            features = self._fe.build_features(ic_name, period_start, period_end)
            prediction = self.predictor.predict(features)

            prob = prediction["probability"]
            risk_level = prediction["risk_level"]

            # 기여 요인 추출: 피처값이 높은 상위 3개
            factors = self._extract_factors(features)

            results.append({
                "ic_name": ic_name,
                "risk_score": prob,
                "risk_level": risk_level,
                "factors": factors,
            })

        results.sort(key=lambda x: x["risk_score"], reverse=True)
        logger.info("전체 IC 스코어링 완료: %d개 IC", len(results))
        return results

    # ------------------------------------------------------------------
    # 히트맵 데이터
    # ------------------------------------------------------------------
    def generate_heatmap_data(self, period_hours: int = 24) -> list[dict]:
        """히트맵용 시공간 위험도 데이터 생성.

        최근 period_hours 시간의 각 IC/시간대 조합에 대해
        위험도를 산출한다.

        Args:
            period_hours: 대상 기간 (시간).

        Returns:
            [{"ic_name", "hour", "risk_score"}, ...]
        """
        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            # 대상 IC + 시간대별 마지막 집계 구간 조회
            cutoff = datetime.now() - timedelta(hours=period_hours)
            rows = conn.execute("""
                SELECT ic_name,
                       EXTRACT(HOUR FROM period_start) AS hour,
                       MAX(period_start) AS latest_start
                FROM traffic_agg
                WHERE period_start >= ?
                GROUP BY ic_name, EXTRACT(HOUR FROM period_start)
                ORDER BY ic_name, hour
            """, [cutoff]).fetchall()
        finally:
            conn.close()

        if not rows:
            logger.warning("히트맵 데이터 없음 (최근 %d시간)", period_hours)
            return []

        heatmap: list[dict] = []
        for ic_name, hour, latest_start in rows:
            period_start = latest_start
            period_end = latest_start + timedelta(minutes=5)

            features = self._fe.build_features(ic_name, period_start, period_end)
            prediction = self.predictor.predict(features)

            heatmap.append({
                "ic_name": ic_name,
                "hour": int(hour),
                "risk_score": prediction["probability"],
            })

        logger.info("히트맵 데이터 생성: %d 셀", len(heatmap))
        return heatmap

    # ------------------------------------------------------------------
    # 리포트 내보내기
    # ------------------------------------------------------------------
    def export_report(self, output_path: Path | str | None = None) -> Path:
        """위험도 순위 리포트를 JSON으로 저장.

        Args:
            output_path: 저장 경로. None이면 OUTPUT_DIR/risk_report_YYYYMMDD_HHMMSS.json.

        Returns:
            저장된 파일 경로.
        """
        rankings = self.score_all_ics()
        heatmap = self.generate_heatmap_data()

        report = {
            "generated_at": datetime.now().isoformat(),
            "total_ics": len(rankings),
            "critical_count": sum(1 for r in rankings if r["risk_level"] == "critical"),
            "high_count": sum(1 for r in rankings if r["risk_level"] == "high"),
            "rankings": rankings,
            "heatmap": heatmap,
        }

        if output_path is None:
            L4_RISK_REPORTS.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = L4_RISK_REPORTS / f"risk_report_{ts}.json"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        logger.info("리포트 저장: %s (IC %d개, critical %d개)",
                     output_path, report["total_ics"], report["critical_count"])
        return output_path

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_factors(features: dict, top_k: int = 3) -> list[dict]:
        """피처 딕셔너리에서 위험 기여 요인 상위 k개 추출.

        이진/범주 피처(is_weekend 등)는 제외하고
        연속형 피처의 절대값 기준으로 순위화한다.

        Args:
            features: 피처명->값 딕셔너리.
            top_k: 반환할 요인 수.

        Returns:
            [{"feature": str, "value": float}, ...]
        """
        skip = {"is_weekend", "is_rush_hour", "ic_type_encoded",
                "hour", "day_of_week"}
        candidates = [
            (k, abs(v)) for k, v in features.items()
            if k not in skip and isinstance(v, (int, float))
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [
            {"feature": name, "value": features[name]}
            for name, _ in candidates[:top_k]
        ]
