"""Track 3-A: ITS 돌발상황정보 API 폴러.

교통사고 이벤트를 주기적으로 폴링하여 Ring Buffer 녹화를 트리거한다.

실행:
  python track3_api_incident.py test          # API 연결 테스트 (인증 확인)
  python track3_api_incident.py poll          # 1회 조회
  python track3_api_incident.py watch         # 데몬 모드 (5분 간격)

사전 준비:
  its.go.kr → 오픈데이터 인증키 발급 → .env에 ITS_API_KEY=... 추가
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ITS_BASE, STREAM_DIR

logger = logging.getLogger(__name__)


@dataclass
class IncidentEvent:
    """돌발 이벤트 정규화 구조체."""
    event_id: str
    event_type: str                # "교통사고" | "공사" | "기상" | ...
    event_detail: str              # "사고" | "추돌사고" | ...
    road_type: str                 # "고속도로" | "국도" | ...
    road_name: str
    road_no: str
    direction: str
    latitude: float | None = None
    longitude: float | None = None
    occurred_at: str = ""          # "20260514172820"
    message: str = ""
    lanes_blocked: str = ""
    link_id: str = ""
    raw: dict = field(default_factory=dict)


class ITSIncidentClient:
    """ITS 국가교통정보센터 돌발상황정보 API.

    엔드포인트: https://openapi.its.go.kr:9443/eventInfo
    필수 파라미터:
      - apiKey: 인증키
      - type: 도로유형 (all/ex/its/loc/sgg/etc)
      - eventType: 이벤트유형 (all/cor/acc/wea/ete/dis/etc)
    선택 파라미터:
      - minX/maxX/minY/maxY: 좌표 범위 (type=all 또는 eventType=all 시)
      - getType: 응답형식 (json/xml)
    """

    EVENT_TYPES = {
        "all": "전체",
        "acc": "교통사고",
        "cor": "공사",
        "wea": "기상",
        "ete": "기타돌발",
        "dis": "재난",
        "etc": "기타",
    }

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("ITS_API_KEY", "")
        self.base_url = ITS_BASE

    def fetch_incidents(self, event_type: str = "acc",
                        road_type: str = "all") -> list[IncidentEvent]:
        """현재 돌발상황 조회.

        Args:
            event_type: "acc"(교통사고) | "all"(전체) | "cor"(공사) 등
            road_type: "all"(전체) | "ex"(고속도로) | "its"(국도) 등
        """
        import requests

        if not self.api_key:
            logger.warning("ITS_API_KEY 미설정")
            return []

        url = f"{self.base_url}/eventInfo"
        params = {
            "apiKey": self.api_key,
            "type": road_type,
            "eventType": event_type,
            "minX": "124.0",
            "maxX": "132.0",
            "minY": "33.0",
            "maxY": "39.0",
            "getType": "json",
        }

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("ITS 돌발상황 API 실패: %s", e)
            return []

        result_code = data.get("header", {}).get("resultCode", -1)
        if result_code != 0:
            msg = data.get("header", {}).get("resultMsg", "")
            logger.error("ITS API 오류: code=%s msg=%s", result_code, msg)
            return []

        events = []
        items = data.get("body", {}).get("items", [])
        for item in items:
            events.append(IncidentEvent(
                event_id=item.get("linkId", ""),
                event_type=item.get("eventType", ""),
                event_detail=item.get("eventDetailType", ""),
                road_type=item.get("type", ""),
                road_name=item.get("roadName", ""),
                road_no=item.get("roadNo", ""),
                direction=item.get("roadDrcType", ""),
                latitude=_safe_float(item.get("coordY")),
                longitude=_safe_float(item.get("coordX")),
                occurred_at=item.get("startDate", ""),
                message=item.get("message", ""),
                lanes_blocked=item.get("lanesBlocked", ""),
                link_id=item.get("linkId", ""),
                raw=item,
            ))

        logger.info("ITS 돌발상황: %d건 조회 (eventType=%s)", len(events), event_type)
        return events


class IncidentPoller:
    """ITS 돌발상황 통합 폴러."""

    def __init__(self):
        self.client = ITSIncidentClient()
        self._seen: set[str] = set()
        self.log_dir = STREAM_DIR / "incident_logs"

    def poll_once(self) -> list[IncidentEvent]:
        """1회 폴링 — 신규 이벤트만 반환."""
        all_events = self.client.fetch_incidents(event_type="acc")

        new_events = []
        for ev in all_events:
            key = f"{ev.link_id}_{ev.occurred_at}"
            if key not in self._seen:
                self._seen.add(key)
                new_events.append(ev)

        if new_events:
            self._save_log(new_events)
            logger.info("신규 사고 %d건 감지", len(new_events))

        return new_events

    def watch(self, interval_sec: int = 300, callback=None):
        """데몬 모드 — interval_sec 간격으로 폴링.

        callback(events: list[IncidentEvent])이 주어지면
        신규 이벤트 발생 시 호출 (Ring Buffer 트리거용).
        """
        logger.info("돌발상황 감시 시작 (간격: %d초)", interval_sec)
        while True:
            try:
                new_events = self.poll_once()
                if new_events and callback:
                    callback(new_events)
            except KeyboardInterrupt:
                logger.info("감시 중단")
                break
            except Exception as e:
                logger.error("폴링 오류: %s", e)
            time.sleep(interval_sec)

    def _save_log(self, events: list[IncidentEvent]):
        """이벤트 로그 저장 (append-only JSONL)."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.log_dir / f"incidents_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            for ev in events:
                record = asdict(ev)
                record["logged_at"] = datetime.now().isoformat()
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _safe_float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cmd = sys.argv[1] if len(sys.argv) > 1 else ""

    if cmd == "test":
        print("=== ITS 돌발상황정보 API 연결 테스트 ===")
        client = ITSIncidentClient()
        print(f"  ITS 키: {'설정됨' if client.api_key else '미설정'}")
        if client.api_key:
            events = client.fetch_incidents(event_type="acc")
            print(f"  교통사고: {len(events)}건")
            for ev in events[:5]:
                print(f"    [{ev.road_type}] {ev.road_name} {ev.direction} — {ev.message}")
                print(f"      좌표: ({ev.latitude}, {ev.longitude}) 발생: {ev.occurred_at}")
            events_all = client.fetch_incidents(event_type="all")
            print(f"  전체 돌발: {len(events_all)}건")

    elif cmd == "poll":
        poller = IncidentPoller()
        events = poller.poll_once()
        print(f"교통사고 {len(events)}건:")
        for ev in events:
            print(f"  [{ev.road_type}] {ev.road_name} {ev.direction} "
                  f"— {ev.message}")

    elif cmd == "watch":
        poller = IncidentPoller()
        poller.watch(interval_sec=300)

    else:
        print("사용법:")
        print("  python track3_api_incident.py test    # API 연결 테스트")
        print("  python track3_api_incident.py poll    # 1회 조회")
        print("  python track3_api_incident.py watch   # 데몬 모드 (5분)")


if __name__ == "__main__":
    main()
