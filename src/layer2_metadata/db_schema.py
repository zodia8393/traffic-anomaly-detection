"""DuckDB 스키마 -- 5개 테이블 생성 및 마이그레이션."""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import duckdb

from config_new import DUCKDB_PATH


def init_db(db_path: str = DUCKDB_PATH) -> duckdb.DuckDBPyConnection:
    """DB 연결을 열고 테이블이 없으면 생성한 뒤 커넥션을 반환한다."""
    conn = duckdb.connect(db_path)
    _create_tables(conn)
    return conn


def _create_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """5개 핵심 테이블을 IF NOT EXISTS 로 생성한다."""

    conn.execute("""
        CREATE TABLE IF NOT EXISTS tracks (
            video_id            VARCHAR,
            track_id            INTEGER,
            ic_name             VARCHAR,
            start_time          TIMESTAMP,
            end_time            TIMESTAMP,
            vehicle_cls_vision  VARCHAR,
            vehicle_cls_mllm    VARCHAR,
            vehicle_cls_final   VARCHAR,
            confidence          FLOAT,
            avg_speed           FLOAT,
            trajectory          JSON,
            PRIMARY KEY (video_id, track_id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS mllm_responses (
            response_id     VARCHAR PRIMARY KEY,
            video_id        VARCHAR,
            trigger_type    VARCHAR,
            trigger_frame   INTEGER,
            task            VARCHAR,
            input_summary   TEXT,
            output_json     JSON,
            latency_sec     FLOAT,
            model_id        VARCHAR,
            created_at      TIMESTAMP DEFAULT current_timestamp
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS accidents (
            event_id            VARCHAR PRIMARY KEY,
            video_id            VARCHAR,
            road_name           VARCHAR,
            direction           VARCHAR,
            km_post             FLOAT,
            branch              VARCHAR,
            point_type          VARCHAR,
            report_time         TIMESTAMP,
            weather             VARCHAR,
            report_source       VARCHAR DEFAULT 'CCTV',
            accident_type       VARCHAR,
            cause               VARCHAR,
            fire                BOOLEAN,
            rollover            BOOLEAN,
            spill               BOOLEAN,
            spill_type          VARCHAR,
            vehicles            JSON,
            casualties          JSON,
            lane_damage         JSON,
            congestion_km       INTEGER,
            severity            VARCHAR,
            mllm_response_id    VARCHAR,
            mllm_report_json    JSON,
            report_path         VARCHAR
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS report_archive (
            report_id           VARCHAR PRIMARY KEY,
            source              VARCHAR,
            date                DATE,
            road_name           VARCHAR,
            direction           VARCHAR,
            km_post             FLOAT,
            point_type          VARCHAR,
            weather             VARCHAR,
            accident_type       VARCHAR,
            cause               VARCHAR,
            vehicles_summary    VARCHAR,
            casualties_total    INTEGER,
            fatalities          INTEGER,
            congestion_km       INTEGER,
            full_record         JSON,
            created_at          TIMESTAMP DEFAULT current_timestamp
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS traffic_agg (
            ic_name         VARCHAR,
            period_start    TIMESTAMP,
            volume          INTEGER,
            avg_speed       FLOAT,
            speed_std       FLOAT,
            truck_ratio     FLOAT,
            risk_score      FLOAT,
            mllm_scene      VARCHAR,
            PRIMARY KEY (ic_name, period_start)
        )
    """)
