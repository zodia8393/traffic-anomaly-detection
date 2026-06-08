"""Vision + MLLM 결과를 DuckDB에 저장하는 Writer."""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from config_new import DUCKDB_PATH, L2_EXPORTS
from .db_schema import init_db

logger = logging.getLogger(__name__)

# DB 쓰기 실패 시 재시도/내구성 정책
_WRITE_RETRIES = 3
_WRITE_RETRY_SLEEP = 0.3  # 초 (DB locked 등 일시적 오류 대비)


class MetadataWriter:
    """트랙, MLLM 응답, 사고 이벤트, 교통류 집계를 DB에 적재한다.

    DB 쓰기 실패 시 즉시 폐기하지 않고 재시도 후, 그래도 실패하면
    디스크 데드레터(JSONL)에 기록하여 영구 손실을 방지한다.
    recover_deadletter()로 추후 재적재 가능.
    """

    def __init__(self, db_path: str = DUCKDB_PATH) -> None:
        self._conn: duckdb.DuckDBPyConnection = init_db(db_path)
        self._deadletter_dir = L2_EXPORTS / "_deadletter"

    # ------------------------------------------------------------------
    # 내구성 있는 실행 (재시도 + 데드레터)
    # ------------------------------------------------------------------
    def _execute_durable(self, sql: str, params: list, kind: str, record: dict) -> bool:
        """execute를 재시도하고, 최종 실패 시 데드레터에 기록한다.

        Returns True if written to DB, False if routed to deadletter.
        """
        last_err: Exception | None = None
        for attempt in range(_WRITE_RETRIES):
            try:
                self._conn.execute(sql, params)
                return True
            except Exception as e:  # noqa: BLE001 — DB locked/디스크 등 모든 실패 포착
                last_err = e
                if attempt < _WRITE_RETRIES - 1:
                    time.sleep(_WRITE_RETRY_SLEEP * (attempt + 1))
        # 최종 실패 → 데드레터로 보존 (프로세스 사망에도 살아남음)
        self._to_deadletter(kind, record, last_err)
        return False

    def _to_deadletter(self, kind: str, record: dict, err: Exception | None) -> None:
        try:
            self._deadletter_dir.mkdir(parents=True, exist_ok=True)
            fp = self._deadletter_dir / f"{kind}_{datetime.now():%Y%m%d}.jsonl"
            entry = {"ts": datetime.now().isoformat(), "error": str(err), "record": record}
            with open(fp, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            logger.error("DB 쓰기 실패 → 데드레터 보존: %s (%s)", kind, err)
        except Exception as e:  # 데드레터마저 실패하면 최소한 로그로 남김
            logger.critical("데드레터 기록 실패 — 레코드 유실 위험: %s | record=%s | err=%s",
                            e, record, err)

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
        self._execute_durable(sql, [
            response_data["response_id"],
            response_data.get("video_id"),
            response_data.get("trigger_type"),
            response_data.get("trigger_frame"),
            response_data.get("task"),
            response_data.get("input_summary"),
            output_json,
            response_data.get("latency_sec"),
            response_data.get("model_id"),
            response_data.get("created_at") or datetime.now(),
        ], "mllm_responses", response_data)

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

        vals.setdefault("created_at", datetime.now())
        cols = [
            "event_id", "video_id",
            "road_name", "direction", "km_post", "branch", "point_type",
            "report_time", "weather", "report_source",
            "accident_type", "cause", "fire", "rollover", "spill", "spill_type",
            "vehicles", "casualties", "lane_damage", "congestion_km",
            "severity", "mllm_response_id", "mllm_report_json", "report_path",
            "source", "blockage_type", "description", "facility_damage",
            "elapsed_sec", "mllm_confidence", "mllm_model", "mllm_latency_sec",
            "analysis_frames", "source_dir", "created_at",
        ]
        placeholders = ", ".join("?" for _ in cols)
        col_names = ", ".join(cols)
        sql = f"INSERT OR REPLACE INTO accidents ({col_names}) VALUES ({placeholders})"
        self._execute_durable(sql, [vals.get(c) for c in cols], "accidents", accident_data)

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
            (선택) volume_dir_a, volume_dir_b, dir_a_label, dir_b_label,
            by_class(dict→json) — 방향별 교통량 계수기(CameraCounter) 적재용.
            미지정 시 None → 레거시 단일 volume 적재와 하위호환.
        """
        # by_class dict → JSON 직렬화 (write_accident json_cols 패턴)
        by_class = agg_data.get("by_class")
        by_class_json = (json.dumps(by_class, ensure_ascii=False)
                         if isinstance(by_class, (dict, list)) else agg_data.get("by_class_json"))
        sql = """
            INSERT OR REPLACE INTO traffic_agg (
                ic_name, period_start,
                volume, avg_speed, speed_std,
                truck_ratio, risk_score, mllm_scene,
                volume_dir_a, volume_dir_b, dir_a_label, dir_b_label, by_class_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self._execute_durable(sql, [
            agg_data["ic_name"],
            agg_data["period_start"],
            agg_data.get("volume"),
            agg_data.get("avg_speed"),
            agg_data.get("speed_std"),
            agg_data.get("truck_ratio"),
            agg_data.get("risk_score"),
            agg_data.get("mllm_scene"),
            agg_data.get("volume_dir_a"),
            agg_data.get("volume_dir_b"),
            agg_data.get("dir_a_label"),
            agg_data.get("dir_b_label"),
            by_class_json,
        ], "traffic_agg", agg_data)

    # ------------------------------------------------------------------
    # 데드레터 복구
    # ------------------------------------------------------------------
    def recover_deadletter(self) -> int:
        """데드레터 JSONL의 보존 레코드를 재적재 시도한다.

        성공한 레코드가 든 파일은 .done 으로 이동. 반환: 재적재 성공 건수.
        """
        if not self._deadletter_dir.exists():
            return 0
        writers = {
            "mllm_responses": self.write_mllm_response,
            "accidents": self.write_accident,
            "traffic_agg": self.write_traffic_agg,
        }
        recovered = 0
        for fp in sorted(self._deadletter_dir.glob("*.jsonl")):
            kind = fp.stem.rsplit("_", 1)[0]
            writer = writers.get(kind)
            if writer is None:
                continue
            remaining = []
            for line in fp.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)["record"]
                    writer(rec)  # 다시 _execute_durable 경유 — 실패 시 새 데드레터로
                    recovered += 1
                except Exception as e:  # noqa: BLE001
                    remaining.append(line)
                    logger.warning("데드레터 복구 실패(보류): %s", e)
            if remaining:
                fp.write_text("\n".join(remaining) + "\n", encoding="utf-8")
            else:
                fp.rename(fp.with_suffix(".jsonl.done"))
        if recovered:
            logger.info("데드레터 복구: %d건 재적재", recovered)
        return recovered

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
