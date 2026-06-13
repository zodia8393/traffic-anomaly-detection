"""ITS 돌발정보 → 녹화 중인 CCTV 구간 사고 여부 크로스체크.

현재 ITS 돌발상황(교통사고)을 조회하여, 우리가 녹화 중인 5개 CCTV 좌표
인근(기본 5km)에 사고가 있는지 매칭한다. 매칭되면 해당 녹화 영상에
사고 장면이 포함됐을 가능성이 높음을 의미한다.

실행:
  python check_incidents_in_recording.py
  python check_incidents_in_recording.py --radius 3   # 매칭 반경 3km
  python check_incidents_in_recording.py --config cameras_5.json
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("/workspace/.env")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from track3_api_incident import ITSIncidentClient

DEFAULT_CONFIG = Path(__file__).resolve().parent / "cameras_5.json"


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--radius", type=float, default=5.0, help="매칭 반경(km)")
    args = ap.parse_args()

    cams = normalize_camera_config(json.loads(Path(args.config).read_text()))
    print(f"=== ITS 돌발정보 × 녹화 CCTV 사고 체크 ({datetime.now():%Y-%m-%d %H:%M}) ===")
    print(f"녹화 카메라 {len(cams)}대, 매칭 반경 {args.radius}km\n")

    client = ITSIncidentClient()
    # 전국 고속도로 교통사고 조회 (ok=False면 API 오류 → '사고없음'으로 단정 금지)
    ok, events = client.fetch_incidents_status(event_type="acc", road_type="ex")
    if not ok:
        print("⚠ ITS API 오류/쿼터/서킷오픈 — 사고 여부 '미상'. '사고 없음'으로 단정할 수 없음.")
        return
    print(f"ITS 고속도로 교통사고 돌발: {len(events)}건\n")

    if not events:
        print("※ 돌발정보 0건 — 현재 사고 없음 (API 정상 응답).")
        return

    any_hit = False
    for cam in cams:
        cx, cy = cam.get("x"), cam.get("y")
        name = cam["name"]
        route = cam.get("route", "")
        hits = []
        for ev in events:
            if ev.longitude is None or ev.latitude is None:
                continue
            d = haversine_km(cx, cy, ev.longitude, ev.latitude)
            if d <= args.radius:
                hits.append((d, ev))
        hits.sort(key=lambda h: h[0])

        if hits:
            any_hit = True
            print(f"🚨 [{name}] 인근 사고 {len(hits)}건:")
            for d, ev in hits:
                print(f"    - {d:.1f}km | {ev.road_name} {ev.direction} | "
                      f"{ev.event_detail} | 발생 {ev.occurred_at} | {ev.message[:40]}")
        else:
            print(f"✅ [{name}] 반경 {args.radius}km 내 사고 없음")

    print()
    if any_hit:
        print("→ 사고 매칭된 카메라의 녹화 영상에 사고 장면 포함 가능성 높음. 해당 시각 구간 확인 권장.")
    else:
        print("→ 현재 5개 녹화 구간 모두 사고 없음.")


if __name__ == "__main__":
    main()
