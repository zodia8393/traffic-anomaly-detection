"""실사고 수집·검증 루프 — ITS 돌발을 상시 매칭·로깅.

녹화 중인 5개 CCTV 인근에 실제 사고가 발생하면, 그 사고를 라벨된 샘플로 기록한다.
시스템 본래 목적(사고영상 데이터셋) + 고속도로 recall 검증의 기반.

- ITS 돌발(교통사고)을 N분 간격 폴링 (회복력: backoff/서킷브레이커 내장)
- 각 사고 vs 카메라 좌표 매칭 (반경 내)
- 매칭 시: 사고 상세 + 어느 녹화파일/시각이 커버하는지 기록 (영상 추출 근거)
- 신규 사고만 기록 (event_id dedup, TTL)

실행:
  python incident_logger.py --once          # 1회
  python incident_logger.py --interval 300  # 데몬(5분 간격)
cron: */5 * * * * python incident_logger.py --once
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("/workspace/.env")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from track3_api_incident import ITSIncidentClient

HERE = Path(__file__).resolve().parent
CONFIG = str(HERE / "cameras_5.json")
REC_ROOT = Path("/DATA/cctv_recording")
LOG_DIR = REC_ROOT / "incident_matches"
SEEN_FILE = LOG_DIR / "_seen_events.json"
MATCH_RADIUS_KM = 5.0


def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def find_recording(cam_slug: str, occurred_at: str) -> dict | None:
    """사고 발생시각을 커버하는 녹화 파일 탐색.

    occurred_at: 'YYYYMMDDHHMMSS'. 해당 날짜 디렉토리의 cam .ts/.mp4 중 매칭.
    """
    if len(occurred_at) < 8:
        return None
    day = occurred_at[:8]
    cam_dir = REC_ROOT / day / f"{cam_slug}_hls"
    if not cam_dir.exists():
        return None
    files = sorted(list(cam_dir.glob("*.ts")) + list(cam_dir.glob("*.mp4")))
    if not files:
        return None
    # 자정 로테이션 단일파일 가정 — 해당 날짜 파일 반환
    return {"file": str(files[-1]), "day": day}


def load_seen() -> dict:
    if SEEN_FILE.exists():
        try:
            return json.loads(SEEN_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_seen(seen: dict):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # TTL: 24시간 경과 event 제거
    now = time.time()
    seen = {k: v for k, v in seen.items() if now - v < 86400}
    SEEN_FILE.write_text(json.dumps(seen, ensure_ascii=False))


def normalize_camera_config(cameras: list[dict]) -> list[dict]:
    normalized = []
    for i, cam in enumerate(cameras, 1):
        item = dict(cam)
        item["name"] = cam.get("name") or cam.get("cctv_name") or cam.get("slug") or f"camera_{i}"
        item["x"] = cam.get("x") or cam.get("coordx") or cam.get("lon") or cam.get("lng")
        item["y"] = cam.get("y") or cam.get("coordy") or cam.get("lat")
        item["route"] = cam.get("route") or cam.get("cctv_road") or cam.get("hotspot_road") or ""
        if item["x"] is None or item["y"] is None:
            raise ValueError(f"camera {i} has no x/y or coordx/coordy")
        normalized.append(item)
    return normalized


def poll_once(cams: list, client: ITSIncidentClient, seen: dict) -> int:
    """1회 폴링 → 신규 매칭 사고 로깅. 매칭 건수 반환."""
    ok, events = client.fetch_incidents_status(event_type="acc", road_type="ex")
    if not ok:
        print(f"[{datetime.now():%H:%M:%S}] ITS API 미상(오류/서킷) — 스킵")
        return 0

    matched = 0
    for ev in events:
        if ev.longitude is None or ev.latitude is None:
            continue
        for cam in cams:
            dist = haversine(cam["x"], cam["y"], ev.longitude, ev.latitude)
            if dist > MATCH_RADIUS_KM:
                continue
            key = f"{ev.event_id}_{cam['name']}"
            if key in seen:
                continue
            seen[key] = time.time()
            slug = cam["name"].replace("[", "").replace("]", "").replace(" ", "_")
            rec = find_recording(slug, ev.occurred_at)
            record = {
                "ts": datetime.now().isoformat(),
                "camera": cam["name"], "distance_km": round(dist, 2),
                "road_name": ev.road_name, "direction": ev.direction,
                "event_detail": ev.event_detail, "occurred_at": ev.occurred_at,
                "message": ev.message, "incident_lat": ev.latitude, "incident_lon": ev.longitude,
                "recording": rec,
            }
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            with open(LOG_DIR / f"matches_{datetime.now():%Y%m%d}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            covered = rec["file"].split("/")[-1] if rec else "녹화없음"
            print(f"🚨 [{datetime.now():%H:%M:%S}] 사고매칭! {cam['name']} {dist:.1f}km — "
                  f"{ev.road_name} {ev.event_detail} | 녹화: {covered}")
            matched += 1
    return matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=float, default=300.0)
    args = ap.parse_args()

    cams = normalize_camera_config(json.loads(Path(args.config).read_text()))
    client = ITSIncidentClient()
    seen = load_seen()

    if args.once:
        m = poll_once(cams, client, seen)
        save_seen(seen)
        if m == 0:
            print(f"[{datetime.now():%H:%M:%S}] 매칭 사고 없음 (5개 구간)")
    else:
        print(f"실사고 매칭 루프 시작 ({args.interval:.0f}초 간격, {len(cams)}대)")
        while True:
            poll_once(cams, client, seen)
            save_seen(seen)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
