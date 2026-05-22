"""ITS 사고 발생 시 인근 CCTV를 Tier 3으로 즉시 승격하는 반응기."""
from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_new import INCIDENT_HOLD_SEC, INCIDENT_POLL_SEC, INCIDENT_REACTOR_CCTVS
from stream_manager import StreamManager
from track3_api_incident import IncidentEvent, ITSIncidentClient
from track3_cctv_stream import ITSCCTVClient

logger = logging.getLogger(__name__)


class IncidentReactor:
    """ITS 돌발상황 감지 → 인근 CCTV Tier 3 승격 반응기."""

    def __init__(self, stream_manager: StreamManager,
                 cctv_client: ITSCCTVClient,
                 stop_event: threading.Event):
        self._stream_manager = stream_manager
        self._cctv_client = cctv_client
        self._stop_event = stop_event
        self._incident_client = ITSIncidentClient()
        self._known_incidents: set[str] = set()
        self._active_reactions: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self):
        """백그라운드 폴링 스레드 시작."""
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="incident-reactor",
            daemon=True,
        )
        self._thread.start()
        logger.info("IncidentReactor 시작 (폴링 %ds)", INCIDENT_POLL_SEC)

    def stop(self):
        """폴링 중지."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        logger.info("IncidentReactor 중지")

    def _poll_loop(self):
        """INCIDENT_POLL_SEC 간격으로 ITS 사고 폴링."""
        while not self._stop_event.is_set():
            try:
                events = self._incident_client.fetch_incidents(event_type="acc")
                for ev in events:
                    with self._lock:
                        if ev.event_id in self._known_incidents:
                            continue
                        self._known_incidents.add(ev.event_id)
                    self._react(ev)
                self._cleanup_expired()
            except Exception as e:
                logger.error("폴링 오류: %s", e)

            self._stop_event.wait(INCIDENT_POLL_SEC)

    def _react(self, incident: IncidentEvent):
        """사고에 대해 인근 CCTV를 Tier 3으로 승격.

        find_nearest 결과를 StreamManager에 로드된 CCTV로 필터링하여
        '승격 실패' 0건을 보장한다. 로드된 CCTV 중 매칭이 없으면 skip.
        """
        if incident.latitude is None or incident.longitude is None:
            logger.warning(
                "좌표 없음 — 건너뜀: event_id=%s road=%s",
                incident.event_id, incident.road_name,
            )
            return

        # 여유분을 확보하여 검색한 뒤, 로드된 CCTV만 필터링
        nearest = self._cctv_client.find_nearest(
            incident.latitude, incident.longitude,
            top_k=INCIDENT_REACTOR_CCTVS * 3,
        )

        if not nearest:
            logger.warning(
                "인근 CCTV 없음: event_id=%s (%.4f, %.4f)",
                incident.event_id, incident.latitude, incident.longitude,
            )
            return

        # StreamManager에 로드된 CCTV만 대상으로 필터링
        loaded_ids = set(self._stream_manager.streams.keys())
        nearest = [
            (cctv, dist_km)
            for cctv, dist_km in nearest
            if cctv.cctv_id in loaded_ids
        ]
        nearest = nearest[:INCIDENT_REACTOR_CCTVS]

        if not nearest:
            logger.info(
                "사고 범위 밖 — skip: event_id=%s road=%s %s "
                "(로드된 %d대 중 인근 CCTV 없음)",
                incident.event_id, incident.road_name, incident.direction,
                len(loaded_ids),
            )
            return

        now = time.monotonic()
        cctv_ids: list[str] = []

        for cctv, dist_km in nearest:
            self._stream_manager.promote(cctv.cctv_id, to_tier=3)
            cctv_ids.append(cctv.cctv_id)
            logger.info(
                "  승격 T3: %s (%.1fkm) — %s",
                cctv.name, dist_km, cctv.cctv_id,
            )

        with self._lock:
            self._active_reactions[incident.event_id] = {
                "cctv_ids": cctv_ids,
                "promoted_at": now,
                "hold_until": now + INCIDENT_HOLD_SEC,
            }

        logger.info(
            "사고 반응: event_id=%s road=%s %s — CCTV %d대 승격 (유지 %ds)",
            incident.event_id, incident.road_name, incident.direction,
            len(cctv_ids), INCIDENT_HOLD_SEC,
        )

    def _cleanup_expired(self):
        """유지 시간 초과된 반응의 CCTV를 강등."""
        now = time.monotonic()
        expired_ids: list[str] = []

        with self._lock:
            for event_id, reaction in self._active_reactions.items():
                if now >= reaction["hold_until"]:
                    expired_ids.append(event_id)

        for event_id in expired_ids:
            with self._lock:
                reaction = self._active_reactions.pop(event_id, None)
            if reaction is None:
                continue

            for cctv_id in reaction["cctv_ids"]:
                self._stream_manager.demote(cctv_id)

            elapsed = now - reaction["promoted_at"]
            logger.info(
                "사고 반응 만료: event_id=%s — CCTV %d대 강등 (%.0f초 경과)",
                event_id, len(reaction["cctv_ids"]), elapsed,
            )

    def get_stats(self) -> dict:
        """현재 반응 통계."""
        with self._lock:
            active_cctv_ids = set()
            for reaction in self._active_reactions.values():
                active_cctv_ids.update(reaction["cctv_ids"])

            return {
                "known_incidents": len(self._known_incidents),
                "active_reactions": len(self._active_reactions),
                "active_cctvs": len(active_cctv_ids),
            }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    stop = threading.Event()
    cctv_client = ITSCCTVClient()
    sm = StreamManager(cctv_client=cctv_client, stop_event=stop)
    reactor = IncidentReactor(sm, cctv_client, stop)

    reactor.start()
    logger.info("Ctrl+C로 종료")

    try:
        while not stop.is_set():
            stop.wait(30)
            stats = reactor.get_stats()
            logger.info("stats: %s", stats)
    except KeyboardInterrupt:
        pass
    finally:
        reactor.stop()
