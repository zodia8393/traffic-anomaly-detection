"""사고예측 피처 엔지니어링."""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
import logging
from datetime import datetime, timedelta

import duckdb
import numpy as np

from config_new import DUCKDB_PATH

logger = logging.getLogger(__name__)

# 피처 이름 정의 (순서 고정)
FEATURE_NAMES: list[str] = [
    # 교통류 (6)
    "volume", "avg_speed", "speed_std", "truck_ratio",
    "speed_change_rate", "volume_change_rate",
    # MLLM (6)
    "risk_score", "risk_score_ma3", "anomaly_count",
    "trigger_ttc_count", "trigger_decel_count", "trigger_stop_count",
    # 시간 (4)
    "hour", "day_of_week", "is_weekend", "is_rush_hour",
    # 공간 (3)
    "ic_type_encoded", "hist_accident_rate", "hist_avg_volume",
]


class FeatureEngineer:
    """MLLM 출력 + 교통류 데이터 -> 피처 벡터 변환기."""

    def __init__(self, db_path: str = DUCKDB_PATH) -> None:
        self.db_path = db_path

    def build_features(self, ic_name: str, period_start, period_end) -> dict:
        """특정 IC/기간의 피처 벡터 생성.

        Args:
            ic_name: IC 식별자.
            period_start: 구간 시작 시각.
            period_end: 구간 종료 시각.

        Returns:
            피처명->값 딕셔너리 (총 19개 피처).
        """
        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            features: dict = {}
            features.update(self._traffic_features(conn, ic_name, period_start, period_end))
            features.update(self._mllm_features(conn, ic_name, period_start, period_end))
            features.update(self._temporal_features(period_start))
            features.update(self._spatial_features(conn, ic_name))
            return features
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # 교통류 피처 (6개)
    # ------------------------------------------------------------------
    def _traffic_features(self, conn: duckdb.DuckDBPyConnection,
                          ic: str, start, end) -> dict:
        """traffic_agg 테이블에서 교통류 피처 추출.

        volume, avg_speed, speed_std, truck_ratio 및
        직전 구간 대비 변화율(speed_change_rate, volume_change_rate).
        """
        row = conn.execute("""
            SELECT volume, avg_speed, speed_std, truck_ratio
            FROM traffic_agg
            WHERE ic_name = ? AND period_start >= ? AND period_start < ?
            ORDER BY period_start DESC
            LIMIT 1
        """, [ic, start, end]).fetchone()

        if row is None:
            return {
                "volume": 0.0, "avg_speed": 0.0, "speed_std": 0.0,
                "truck_ratio": 0.0, "speed_change_rate": 0.0,
                "volume_change_rate": 0.0,
            }

        volume, avg_speed, speed_std, truck_ratio = (
            float(row[0] or 0), float(row[1] or 0),
            float(row[2] or 0), float(row[3] or 0),
        )

        # 직전 3구간 이동평균으로 변화율 산출
        prev_rows = conn.execute("""
            SELECT volume, avg_speed
            FROM traffic_agg
            WHERE ic_name = ? AND period_start < ?
            ORDER BY period_start DESC
            LIMIT 3
        """, [ic, start]).fetchall()

        if prev_rows:
            prev_speed_avg = np.mean([float(r[1] or 0) for r in prev_rows])
            prev_vol_avg = np.mean([float(r[0] or 0) for r in prev_rows])
            speed_change = ((avg_speed - prev_speed_avg) / prev_speed_avg
                            if prev_speed_avg > 0 else 0.0)
            volume_change = ((volume - prev_vol_avg) / prev_vol_avg
                             if prev_vol_avg > 0 else 0.0)
        else:
            speed_change = 0.0
            volume_change = 0.0

        return {
            "volume": volume,
            "avg_speed": avg_speed,
            "speed_std": speed_std,
            "truck_ratio": truck_ratio,
            "speed_change_rate": speed_change,
            "volume_change_rate": volume_change,
        }

    # ------------------------------------------------------------------
    # MLLM 피처 (6개)
    # ------------------------------------------------------------------
    def _mllm_features(self, conn: duckdb.DuckDBPyConnection,
                       ic: str, start, end) -> dict:
        """mllm_responses + traffic_agg에서 MLLM 관련 피처 추출.

        risk_score, risk_score_ma3 (3구간 이동평균),
        anomaly_count, trigger 유형별 카운트 (ttc, decel, stop).
        """
        # 현재 구간 risk_score
        risk_row = conn.execute("""
            SELECT risk_score
            FROM traffic_agg
            WHERE ic_name = ? AND period_start >= ? AND period_start < ?
            ORDER BY period_start DESC
            LIMIT 1
        """, [ic, start, end]).fetchone()
        risk_score = float(risk_row[0] or 0) if risk_row else 0.0

        # risk_score 3구간 이동평균
        ma_rows = conn.execute("""
            SELECT risk_score
            FROM traffic_agg
            WHERE ic_name = ? AND period_start <= ?
            ORDER BY period_start DESC
            LIMIT 3
        """, [ic, end]).fetchall()
        risk_scores = [float(r[0] or 0) for r in ma_rows]
        risk_score_ma3 = float(np.mean(risk_scores)) if risk_scores else 0.0

        # 해당 구간 MLLM 응답에서 이상치/트리거 카운트
        # video_id를 통해 ic_name과 연결하는 대신, 시간 범위로 필터
        mllm_rows = conn.execute("""
            SELECT trigger_type, output_json
            FROM mllm_responses
            WHERE created_at >= ? AND created_at < ?
              AND video_id IN (
                  SELECT DISTINCT video_id FROM tracks WHERE ic_name = ?
              )
        """, [start, end, ic]).fetchall()

        anomaly_count = 0
        trigger_ttc = 0
        trigger_decel = 0
        trigger_stop = 0

        for row in mllm_rows:
            trigger_type = row[0] or ""
            if trigger_type == "ttc":
                trigger_ttc += 1
            elif trigger_type == "decel":
                trigger_decel += 1
            elif trigger_type == "stop":
                trigger_stop += 1

            # output_json에서 anomaly 여부 확인
            try:
                output = json.loads(row[1]) if isinstance(row[1], str) else row[1]
                if output and output.get("is_anomaly"):
                    anomaly_count += 1
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

        return {
            "risk_score": risk_score,
            "risk_score_ma3": risk_score_ma3,
            "anomaly_count": float(anomaly_count),
            "trigger_ttc_count": float(trigger_ttc),
            "trigger_decel_count": float(trigger_decel),
            "trigger_stop_count": float(trigger_stop),
        }

    # ------------------------------------------------------------------
    # 시간 피처 (4개)
    # ------------------------------------------------------------------
    def _temporal_features(self, timestamp) -> dict:
        """시간대/요일/주말/출퇴근 피처.

        Args:
            timestamp: datetime 또는 파싱 가능한 문자열.
        """
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        hour = timestamp.hour
        dow = timestamp.weekday()  # 0=월 ~ 6=일
        is_weekend = 1.0 if dow >= 5 else 0.0
        # 출퇴근: 평일 07-09, 17-19
        is_rush = 1.0 if (not is_weekend and (7 <= hour <= 9 or 17 <= hour <= 19)) else 0.0

        return {
            "hour": float(hour),
            "day_of_week": float(dow),
            "is_weekend": is_weekend,
            "is_rush_hour": is_rush,
        }

    # ------------------------------------------------------------------
    # 공간 피처 (3개)
    # ------------------------------------------------------------------
    def _spatial_features(self, conn: duckdb.DuckDBPyConnection, ic: str) -> dict:
        """IC 유형, 과거 사고 빈도, 과거 30일 평균 교통량.

        ic_type_encoded: IC 이름 해싱으로 범주 인코딩 (임시).
        hist_accident_rate: 과거 전체 사고 건수 / 30일 데이터 건수.
        hist_avg_volume: 과거 30일 평균 교통량.
        """
        # IC 유형 — point_type 사용. 없으면 이름 해시
        pt_row = conn.execute("""
            SELECT point_type
            FROM accidents
            WHERE road_name = ? OR event_id LIKE ?
            LIMIT 1
        """, [ic, f"%{ic}%"]).fetchone()

        ic_type_map = {"IC": 0, "JC": 1, "TG": 2, "SA": 3, "본선": 4}
        if pt_row and pt_row[0] in ic_type_map:
            ic_type_encoded = float(ic_type_map[pt_row[0]])
        else:
            ic_type_encoded = float(hash(ic) % len(ic_type_map))

        # 과거 사고 빈도
        accident_count = conn.execute("""
            SELECT COUNT(*) FROM accidents
            WHERE road_name = ? OR event_id LIKE ?
        """, [ic, f"%{ic}%"]).fetchone()[0]

        traffic_count = conn.execute("""
            SELECT COUNT(*) FROM traffic_agg WHERE ic_name = ?
        """, [ic]).fetchone()[0]

        hist_accident_rate = (float(accident_count) / max(traffic_count, 1))

        # 과거 30일 평균 교통량
        avg_vol_row = conn.execute("""
            SELECT AVG(volume)
            FROM traffic_agg
            WHERE ic_name = ?
              AND period_start >= current_timestamp - INTERVAL '30' DAY
        """, [ic]).fetchone()
        hist_avg_volume = float(avg_vol_row[0] or 0) if avg_vol_row else 0.0

        return {
            "ic_type_encoded": ic_type_encoded,
            "hist_accident_rate": hist_accident_rate,
            "hist_avg_volume": hist_avg_volume,
        }

    # ------------------------------------------------------------------
    # 전체 데이터셋 구축
    # ------------------------------------------------------------------
    def build_dataset(self, period_days: int = 30) -> tuple[np.ndarray, np.ndarray]:
        """전체 학습 데이터셋 구축.

        traffic_agg 테이블의 모든 IC/구간을 순회하며 피처를 생성하고,
        해당 구간 이후 30분 내 사고 발생 여부를 라벨(y)로 설정한다.

        Args:
            period_days: 최근 N일간 데이터 사용.

        Returns:
            (X, y): X는 (n_samples, n_features) ndarray, y는 (n_samples,) 0/1 ndarray.
        """
        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            # 대상 구간 조회 (INTERVAL 플레이스홀더는 DuckDB 파서 오류 →
            # cutoff를 Python에서 계산해 파라미터로 전달)
            cutoff = datetime.now() - timedelta(days=period_days)
            rows = conn.execute("""
                SELECT DISTINCT ic_name, period_start
                FROM traffic_agg
                WHERE period_start >= ?
                ORDER BY ic_name, period_start
            """, [cutoff]).fetchall()

            if not rows:
                logger.warning("traffic_agg 데이터 없음 (최근 %d일)", period_days)
                return np.empty((0, len(FEATURE_NAMES))), np.empty(0)

            X_list: list[list[float]] = []
            y_list: list[int] = []

            for ic_name, period_start in rows:
                # 구간 길이 추정: 다음 구간까지 또는 기본 5분
                period_end = period_start + timedelta(minutes=5)

                features = self.build_features(ic_name, period_start, period_end)

                # 피처 벡터를 고정 순서로 변환
                feat_vec = [features.get(fn, 0.0) for fn in FEATURE_NAMES]
                X_list.append(feat_vec)

                # 라벨: 해당 구간 이후 30분 내 사고 발생 여부
                label_end = period_start + timedelta(minutes=30)
                accident_row = conn.execute("""
                    SELECT COUNT(*) FROM accidents
                    WHERE (road_name = ? OR event_id LIKE ?)
                      AND report_time >= ? AND report_time < ?
                """, [ic_name, f"%{ic_name}%", period_start, label_end]).fetchone()

                y_list.append(1 if accident_row[0] > 0 else 0)

            logger.info("데이터셋 구축 완료: %d 샘플 (양성 %d건)",
                        len(y_list), sum(y_list))
            return np.array(X_list, dtype=np.float64), np.array(y_list, dtype=np.int32)
        finally:
            conn.close()
