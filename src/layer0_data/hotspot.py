"""사고 다발구간 선정 + 돌발 그룹화 (run_realtime 분해)."""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from realtime_bootstrap import *  # noqa: F401,F403

logger = logging.getLogger(__name__)


HOTSPOT_CANDIDATES = [
    {"name": "서해안선 발안IC-비봉IC",  "lat": 37.11, "lon": 126.89, "road": "서해안선"},
    {"name": "경부선 수원-오산",          "lat": 37.15, "lon": 127.05, "road": "경부선"},
    {"name": "영동선 여주JC",             "lat": 37.29, "lon": 127.64, "road": "영동선"},
    {"name": "경부선 안성JC",             "lat": 37.00, "lon": 127.10, "road": "경부선"},
    {"name": "서해안선 서평택IC",         "lat": 36.95, "lon": 126.90, "road": "서해안선"},
    {"name": "중부내륙선 여주JC",         "lat": 37.28, "lon": 127.60, "road": "중부내륙선"},
    {"name": "통영대전선 함양JC",         "lat": 35.50, "lon": 127.80, "road": "통영대전선"},
]


class HotspotSelector:
    """사고 다발 구간 선정기."""

    def __init__(self, cctv_client: ITSCCTVClient | None = None):
        self.cctv_client = cctv_client or ITSCCTVClient()
        self.incident_client = ITSIncidentClient()

    def select(self, radius_km: float = 5.0,
               lat: float | None = None, lon: float | None = None,
               ) -> dict:
        """최적 감시 구간 선정."""
        if lat is not None and lon is not None:
            return self._evaluate_point(
                f"사용자 지정 ({lat:.4f}, {lon:.4f})", lat, lon, radius_km,
            )

        all_events = []
        try:
            all_events = self.incident_client.fetch_incidents(event_type="all", road_type="ex")
        except Exception as e:
            logger.warning("ITS 돌발 조회 실패: %s", e)

        accidents = [ev for ev in all_events if "사고" in ev.event_type]
        cctvs = self.cctv_client.list_cctvs()
        if not cctvs:
            logger.error("CCTV 목록 조회 실패")
            return {}

        results = []
        for cand in HOTSPOT_CANDIDATES:
            score = 0.0
            reasons = []

            active = False
            for acc in accidents:
                if acc.latitude and acc.longitude:
                    d = _haversine(cand["lat"], cand["lon"], acc.latitude, acc.longitude)
                    if d <= radius_km:
                        score += 50
                        active = True
                        reasons.append(f"사고 발생 중 ({d:.1f}km)")

            nearby_events = sum(
                1 for ev in all_events
                if ev.latitude and ev.longitude
                and _haversine(cand["lat"], cand["lon"], ev.latitude, ev.longitude) <= radius_km
            )
            score += nearby_events * 5
            if nearby_events:
                reasons.append(f"돌발 {nearby_events}건")

            nearby_cctvs = [
                (c, _haversine(cand["lat"], cand["lon"], c.latitude, c.longitude))
                for c in cctvs
                if c.stream_url
                and _haversine(cand["lat"], cand["lon"], c.latitude, c.longitude) <= radius_km
            ]
            nearby_cctvs.sort(key=lambda x: x[1])
            n_cctv = len(nearby_cctvs)
            score += n_cctv * 2
            reasons.append(f"CCTV {n_cctv}대")

            if n_cctv >= MAX_CONCURRENT_STREAMS:
                score += 10
                reasons.append(f"동시감시 가능 (>={MAX_CONCURRENT_STREAMS})")

            results.append({
                "name": cand["name"],
                "lat": cand["lat"],
                "lon": cand["lon"],
                "road": cand["road"],
                "score": score,
                "reason": " | ".join(reasons),
                "cctvs": [c for c, _ in nearby_cctvs],
                "cctv_distances": [d for _, d in nearby_cctvs],
                "incidents_nearby": nearby_events,
                "active_accident": active,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        best = results[0]

        logger.info("사고 다발 구간 선정 결과:")
        for r in results[:5]:
            flag = " ★" if r is best else ""
            logger.info("  [%5.0f점] %s — %s%s",
                        r["score"], r["name"], r["reason"], flag)

        return best

    def _evaluate_point(self, name: str, lat: float, lon: float,
                        radius_km: float) -> dict:
        """단일 좌표 평가."""
        cctvs = self.cctv_client.list_cctvs()
        nearby = [
            (c, _haversine(lat, lon, c.latitude, c.longitude))
            for c in cctvs
            if c.stream_url
            and _haversine(lat, lon, c.latitude, c.longitude) <= radius_km
        ]
        nearby.sort(key=lambda x: x[1])
        return {
            "name": name,
            "lat": lat,
            "lon": lon,
            "road": "",
            "score": len(nearby) * 2,
            "reason": f"CCTV {len(nearby)}대 (반경 {radius_km}km)",
            "cctvs": [c for c, _ in nearby],
            "cctv_distances": [d for _, d in nearby],
            "incidents_nearby": 0,
            "active_accident": False,
        }


# ═══════════════════════════════════════════════════════════════════════
# 사고 그룹핑: 여러 카메라의 트리거를 같은 incident_id로 묶기
# ═══════════════════════════════════════════════════════════════════════

class IncidentGrouper:
    """동일 사고에 대한 다중 카메라 트리거를 그룹핑."""

    def __init__(self):
        self._groups: dict[str, dict] = {}
        self._its_to_group: dict[str, str] = {}
        self._seq = 0
        self._lock = threading.Lock()

    def assign_group(self, trigger_type: str, cctv: CCTVInfo,
                     incident: IncidentEvent | None,
                     video_path: Path | None) -> str:
        """트리거를 그룹에 할당. 반환: group_id."""
        with self._lock:
            now = datetime.now()

            if incident and incident.event_id:
                if incident.event_id in self._its_to_group:
                    gid = self._its_to_group[incident.event_id]
                    self._groups[gid]["cameras"].append(cctv.cctv_id)
                    if video_path:
                        self._groups[gid]["clips"].append(str(video_path))
                    logger.info("기존 사고 그룹에 추가: %s (카메라: %s)",
                                gid, cctv.name)
                    return gid

            for gid, grp in self._groups.items():
                age = (now - grp["created_at"]).total_seconds()
                if age > 180:
                    continue
                d = _haversine(cctv.latitude, cctv.longitude,
                               grp["center_lat"], grp["center_lon"])
                if d <= 5.0 and cctv.cctv_id not in grp["cameras"]:
                    grp["cameras"].append(cctv.cctv_id)
                    if video_path:
                        grp["clips"].append(str(video_path))
                    logger.info("근접 사고 그룹에 추가: %s (카메라: %s, 거리: %.1fkm)",
                                gid, cctv.name, d)
                    return gid

            self._seq += 1
            gid = f"INC_{now.strftime('%Y%m%d_%H%M%S')}_{self._seq:03d}"
            self._groups[gid] = {
                "incident_id": incident.event_id if incident else None,
                "cameras": [cctv.cctv_id],
                "clips": [str(video_path)] if video_path else [],
                "center_lat": cctv.latitude,
                "center_lon": cctv.longitude,
                "trigger_type": trigger_type,
                "created_at": now,
            }
            if incident and incident.event_id:
                self._its_to_group[incident.event_id] = gid
            logger.info("새 사고 그룹 생성: %s (카메라: %s)", gid, cctv.name)
            return gid

    def get_groups(self) -> dict[str, dict]:
        """현재 활성 그룹 목록."""
        with self._lock:
            return dict(self._groups)


