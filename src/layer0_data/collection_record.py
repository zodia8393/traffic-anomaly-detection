"""사고 수집 기록 dataclass — run_realtime에서 분리.

저장 함수(save_collection_record)는 META_DIR(STREAM_DIR 의존)에 묶여 있어
run_realtime.py에 잔류한다.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CollectionRecord:
    """수집 기록."""
    event_id: str
    trigger_type: str
    trigger_description: str
    trigger_frame: int
    trigger_severity: float
    cctv_id: str
    cctv_name: str
    cctv_lat: float
    cctv_lon: float
    incident_id: str | None = None
    incident_road: str | None = None
    incident_message: str | None = None
    incident_lat: float | None = None
    incident_lon: float | None = None
    match_distance_km: float | None = None
    video_path: str | None = None
    video_size_mb: float | None = None
    video_duration_sec: float | None = None
    its_verified: bool = False
    action: str = ""               # "confirmed" | "pending_preserved" | "deleted"
    collected_at: str = ""
