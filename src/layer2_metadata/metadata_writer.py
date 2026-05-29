"""Vision + MLLM 결과를 DuckDB에 저장하는 Writer."""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from config_new import DUCKDB_PATH, L2_EXPORTS
from .db_schema import init_db

logger = logging.getLogger(__name__)


class MetadataWriter:
    """트랙, MLLM 응답, 사고 이벤트, 교통류 집계를 DB에 적재한다."""

    def __init__(self, db_path: str = DUCKDB_PATH) -> None:
        self._conn: duckdb.DuckDBPyConnection = init_db(db_path)

    # ------------------------------------------------------------------
    # tracks
    # ------------------------------------------------------------------
    def write_tracks(self, video_id: str, tracks_data: list[dict]) -> int:
        """트랙 목록을 배치로 INSERT OR REPLACE 한다.

        Parameters
        ----------
        video_id : str
            영상 식별자.
        tracks_data : list[dict]
            각 dict는 track_id, ic_name, start_time, end_time,
            vehicle_cls_vision, vehicle_cls_mllm, vehicle_cls_final,
            confidence, avg_speed, trajectory 키를 가진다.

        Returns
        -------
        int
            적재된 행 수.
        """
        sql = """
            INSERT OR REPLACE INTO tracks (
                video_id, track_id, ic_name,
                start_time, end_time,
                vehicle_cls_vision, vehicle_cls_mllm, vehicle_cls_final,
                confidence, avg_speed, trajectory
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        rows: list[tuple[Any, ...]] = []
        for t in tracks_data:
            trajectory = t.get("trajectory")
            if trajectory is not None and not isinstance(trajectory, str):
                trajectory = json.dumps(trajectory, ensure_ascii=False)
            rows.append((
                video_id,
                t["track_id"],
                t.get("ic_name"),
                t.get("start_time"),
                t.get("end_time"),
                t.get("vehicle_cls_vision"),
                t.get("vehicle_cls_mllm"),
                t.get("vehicle_cls_final"),
                t.get("confidence"),
                t.get("avg_speed"),
                trajectory,
            ))
        self._conn.executemany(sql, rows)
        return len(rows)

    # ------------------------------------------------------------------
    # mllm_responses
    # ------------------------------------------------------------------
    def write_mllm_response(self, response_data: dict) -> None:
        """MLLM 응답 1건을 INSERT OR REPLACE 한다.

        Parameters
        ----------
        response_data : dict
            response_id, video_id, trigger_type, trigger_frame,
            task, input_summary, output_json, latency_sec, model_id
            키를 가진다. created_at 은 생략 시 DB 기본값 사용.
        """
        output_json = response_data.get("output_json")
        if output_json is not None and not isinstance(output_json, str):
            output_json = json.dumps(output_json, ensure_ascii=False)
        if isinstance(output_json, str):
            output_json = output_json.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")

        sql = """
            INSERT OR REPLACE INTO mllm_responses (
                response_id, video_id, trigger_type, trigger_frame,
                task, input_summary, output_json,
                latency_sec, model_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self._conn.execute(sql, [
            response_data["response_id"],
            response_data.get("video_id"),
            response_data.get("trigger_type"),
            response_data.get("trigger_frame"),
            response_data.get("task"),
            response_data.get("input_summary"),
            output_json,
            response_data.get("latency_sec"),
            response_data.get("model_id"),
            response_data.get("created_at"),
        ])

    # ------------------------------------------------------------------
    # accidents
    # ------------------------------------------------------------------
    def write_accident(self, accident_data: dict) -> None:
        """사고 이벤트 1건을 INSERT OR REPLACE 한다.

        Parameters
        ----------
        accident_data : dict
            event_id 필수. 나머지는 accidents 테이블 컬럼명과 동일한 키.
        """
        json_cols = ("vehicles", "casualties", "lane_damage",
                     "mllm_report_json", "facility_damage")
        vals: dict[str, Any] = {}
        for k, v in accident_data.items():
            if k in json_cols and v is not None and not isinstance(v, str):
                vals[k] = json.dumps(v, ensure_ascii=False)
            else:
                vals[k] = v

        cols = [
            "event_id", "video_id",
            "road_name", "direction", "km_post", "branch", "point_type",
            "report_time", "weather", "report_source",
            "accident_type", "cause", "fire", "rollover", "spill", "spill_type",
            "vehicles", "casualties", "lane_damage", "congestion_km",
            "severity", "mllm_response_id", "mllm_report_json", "report_path",
            "source", "blockage_type", "description", "facility_damage",
            "elapsed_sec", "mllm_confidence", "mllm_model", "mllm_latency_sec",
            "analysis_frames", "source_dir",
        ]
        placeholders = ", ".join("?" for _ in cols)
        col_names = ", ".join(cols)
        sql = f"INSERT OR REPLACE INTO accidents ({col_names}) VALUES ({placeholders})"
        self._conn.execute(sql, [vals.get(c) for c in cols])

    # ------------------------------------------------------------------
    # traffic_agg
    # ------------------------------------------------------------------
    def write_traffic_agg(self, agg_data: dict) -> None:
        """교통류 집계 1건을 INSERT OR REPLACE 한다.

        Parameters
        ----------
        agg_data : dict
            ic_name, period_start, volume, avg_speed, speed_std,
            truck_ratio, risk_score, mllm_scene 키.
        """
        sql = """
            INSERT OR REPLACE INTO traffic_agg (
                ic_name, period_start,
                volume, avg_speed, speed_std,
                truck_ratio, risk_score, mllm_scene
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        self._conn.execute(sql, [
            agg_data["ic_name"],
            agg_data["period_start"],
            agg_data.get("volume"),
            agg_data.get("avg_speed"),
            agg_data.get("speed_std"),
            agg_data.get("truck_ratio"),
            agg_data.get("risk_score"),
            agg_data.get("mllm_scene"),
        ])

    # ------------------------------------------------------------------
    # DB 내보내기
    # ------------------------------------------------------------------
    def export_tables(self, tables: list[str] | None = None) -> Path:
        """지정 테이블을 L2_EXPORTS에 JSON으로 내보내기."""
        L2_EXPORTS.mkdir(parents=True, exist_ok=True)
        if tables is None:
            tables = ["tracks", "mllm_responses", "accidents", "traffic_agg"]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = L2_EXPORTS / f"export_{ts}"
        export_dir.mkdir(parents=True, exist_ok=True)

        for table in tables:
            rows = self._conn.execute(f"SELECT * FROM {table}").fetchall()
            cols = [desc[0] for desc in self._conn.description]
            data = [dict(zip(cols, row)) for row in rows]
            fpath = export_dir / f"{table}.json"
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        logger.info("DB 내보내기 %d 테이블 → %s", len(tables), export_dir)
        return export_dir
