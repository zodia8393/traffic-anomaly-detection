"""ITS 사고 교차확인기 — run_realtime에서 분리 (모놀리스 2단계).

트리거 발화 좌표 인근에 실제 ITS 돌발(사고)이 있는지 확인한다.
API 오류/서킷오픈 시 api_uncertain=True로 '사고 없음' 단정을 막아 오삭제를 방지한다.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from realtime_constants import ITS_CHECK_COOLDOWN_SEC, ITS_CHECK_RADIUS_KM
from track3_api_incident import IncidentEvent, ITSIncidentClient
from track3_cctv_stream import _haversine

logger = logging.getLogger(__name__)


class IncidentVerifier:
    """트리거 발화 시 ITS API로 실제 사고 여부를 교차 확인한다."""

    def __init__(self):
        self.client = ITSIncidentClient()
        self._last_check: dict[str, float] = {}
        self._cached_incidents: list[IncidentEvent] = []
        self._cache_age: float = 0.0
        self.api_uncertain: bool = False  # 마지막 조회가 API오류/서킷오픈이면 True

    def verify(self, lat: float, lon: float,
               trigger_type: str, radius_km: float = ITS_CHECK_RADIUS_KM
               ) -> tuple[bool, IncidentEvent | None]:
        """트리거 좌표 인근에 실제 사고가 있는지 확인.

        Returns:
            (확인됨, 매칭된_사고_이벤트). 미확인이면 (False, None).
        """
        now = time.time()

        # 쿨다운 검사
        last = self._last_check.get(trigger_type, 0)
        if (now - last) < ITS_CHECK_COOLDOWN_SEC:
            logger.debug("ITS 확인 쿨다운 중: %s (%.0f초 남)",
                         trigger_type, ITS_CHECK_COOLDOWN_SEC - (now - last))
            return False, None

        self._last_check[trigger_type] = now

        # 캐시 갱신 (30초 이상 경과 시)
        if (now - self._cache_age) > 30:
            ok, events = self.client.fetch_incidents_status(event_type="acc")
            self.api_uncertain = not ok
            if not ok:
                # API 오류/서킷오픈 → '사고 없음'으로 단정 불가 (오삭제 방지)
                logger.error("ITS API 조회 불가 — 판정 미상(uncertain)")
                return False, None
            self._cached_incidents = events
            self._cache_age = now
            logger.info("ITS 사고 목록 갱신: %d건", len(self._cached_incidents))
        else:
            self.api_uncertain = False

        # 반경 내 사고 검색
        for incident in self._cached_incidents:
            if incident.latitude is None or incident.longitude is None:
                continue
            dist = _haversine(lat, lon, incident.latitude, incident.longitude)
            if dist <= radius_km:
                logger.info("사고 확인! [%s] %s %s (%.1f km) - %s",
                            incident.road_type, incident.road_name,
                            incident.direction, dist, incident.message[:50])
                return True, incident

        logger.info("ITS 사고 미확인 (반경 %.1fkm 내 매칭 없음)", radius_km)
        return False, None
