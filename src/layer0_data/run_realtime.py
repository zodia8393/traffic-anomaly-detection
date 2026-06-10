"""실시간 사고영상 수집 통합 파이프라인 (v2 — 징조 기반 온디맨드 녹화).

설계 방향: "상시 분석, 녹화는 징조 포착 시에만"

시스템 흐름:
  Phase 1 — 상시 분석 (녹화 안 함)
    CCTV HLS 스트림 -> FrameSampler(1fps) -> Vision Pipeline(YOLO+ByteTrack+트리거)
    프레임만 분석, 디스크에 저장하지 않음

  Phase 2 — 징조 포착 -> 녹화 시작
    트리거 발화(T1/T3/T4/T5) = 사고 징조 감지
    그 즉시 ffmpeg 녹화 시작 (HLS -> mp4, -c copy)
    Vision Pipeline 분석 계속 (추가 트리거 시 녹화 연장)
    녹화 지속: 최소 3분, 최대 5분

  Phase 3 — 사고 확인 + 저장/폐기
    녹화 종료 후 ITS 돌발상황 API 교차확인
    사고 확인 -> 영상 보존 + 메타데이터 기록
    미확인 -> 영상 파일 삭제

최종 영상 내러티브:
  [0:00~0:30]  사고 징조 (급감속, TTC 임박 등)
  [0:30~1:00]  사고 발생
  [1:00~3:00+] 사고 후처리

실행:
  python run_realtime.py monitor                              # 1대 CCTV 자동 선택
  python run_realtime.py monitor --lat 37.0 --lon 127.0      # 좌표 인근 CCTV
  python run_realtime.py monitor --cctv-id <id>              # 특정 CCTV
  python run_realtime.py multi                               # 사고다발구간 다중 CCTV
  python run_realtime.py multi --lat 37.11 --lon 126.89 --radius 5.0
  python run_realtime.py hotspot                             # 사고다발구간 선정 (조회만)
  python run_realtime.py dry-run                             # 전체 흐름 시뮬레이션
  python run_realtime.py status                              # 수집 현황
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# ── 부트스트랩: 이중 config.py 충돌해소 + 공용 import는 realtime_bootstrap로 분리 ──
# (import 부수효과로 config 스왑·track3·상수·경로가 모두 준비됨. 반드시 먼저 import)
from realtime_bootstrap import *  # noqa: F401,F403

logger = logging.getLogger(__name__)

# 분해된 모듈에서 클래스 import (하위호환)
from ondemand_recorder import OnDemandRecorder, save_collection_record
from realtime_pipeline import RealtimeAccidentPipeline
from hotspot import HotspotSelector, IncidentGrouper
from camera_worker import CameraStats, CameraWorker
from multi_pipeline import MultiCCTVPipeline


def main():
    import argparse

    # dotenv 로드
    from pathlib import Path
    env_path = Path("/workspace/.env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())

    # 로그 로테이션(100MB×5) — 무한 증가(211MB+) 방지. log_setup은 src에 있음.
    try:
        from log_setup import setup_logging
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        setup_logging(logfile=str(LOG_DIR / "run_realtime.log"), level=logging.INFO)
    except Exception:  # 헬퍼 로드 실패 시 기존 방식 폴백
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    parser = argparse.ArgumentParser(
        description="실시간 사고영상 수집 파이프라인 (v2 — 징조 기반 온디맨드 녹화)",
    )
    sub = parser.add_subparsers(dest="command")

    # monitor (단일 CCTV)
    p_mon = sub.add_parser("monitor", help="단일 CCTV 모니터링 (징조 시에만 녹화)")
    p_mon.add_argument("--lat", type=float, help="CCTV 검색 위도")
    p_mon.add_argument("--lon", type=float, help="CCTV 검색 경도")
    p_mon.add_argument("--cctv-id", help="특정 CCTV ID")
    p_mon.add_argument("--max-frames", type=int, default=0,
                        help="최대 프레임 수 (0=무제한)")

    # multi (다중 CCTV 동시 감시)
    p_multi = sub.add_parser("multi", help="사고다발구간 다중 CCTV 동시 감시")
    p_multi.add_argument("--lat", type=float, help="감시 중심 위도 (미지정 시 자동 선정)")
    p_multi.add_argument("--lon", type=float, help="감시 중심 경도")
    p_multi.add_argument("--radius", type=float, default=5.0,
                         help="CCTV 검색 반경 km (기본: 5.0)")
    p_multi.add_argument("--max-cameras", type=int, default=MAX_CONCURRENT_STREAMS,
                         help=f"최대 동시 카메라 수 (기본: {MAX_CONCURRENT_STREAMS})")

    # hotspot
    p_hs = sub.add_parser("hotspot", help="사고 다발 구간 선정 (조회)")
    p_hs.add_argument("--radius", type=float, default=5.0,
                      help="CCTV 검색 반경 km")

    # dry-run
    sub.add_parser("dry-run", help="전체 흐름 시뮬레이션")

    # status
    sub.add_parser("status", help="수집 현황 확인")

    args = parser.parse_args()

    if args.command == "monitor":
        pipeline = RealtimeAccidentPipeline()
        pipeline.monitor(
            lat=args.lat,
            lon=args.lon,
            cctv_id=args.cctv_id,
            max_frames=args.max_frames,
        )
    elif args.command == "multi":
        multi = MultiCCTVPipeline()
        multi.multi_monitor(
            lat=args.lat,
            lon=args.lon,
            radius_km=args.radius,
            max_cameras=args.max_cameras,
        )
    elif args.command == "hotspot":
        selector = HotspotSelector()
        result = selector.select(radius_km=args.radius)
        if result:
            print()
            print("=" * 60)
            print("사고 다발 구간 선정 결과")
            print("=" * 60)
            print(f"  선정: {result['name']}")
            print(f"  좌표: ({result['lat']}, {result['lon']})")
            print(f"  점수: {result['score']:.0f}")
            print(f"  근거: {result['reason']}")
            print(f"  CCTV: {len(result['cctvs'])}대")
            for i, (c, d) in enumerate(zip(
                    result['cctvs'][:10], result['cctv_distances'][:10])):
                print(f"    [{i+1}] {d:.1f}km — {c.name} ({c.cctv_id})")
            print("=" * 60)
    elif args.command == "dry-run":
        pipeline = RealtimeAccidentPipeline()
        pipeline.dry_run()
    elif args.command == "status":
        pipeline = RealtimeAccidentPipeline()
        pipeline.status()
    else:
        parser.print_help()
        print()
        print("예시:")
        print("  python run_realtime.py monitor                              # 단일 CCTV (자동)")
        print("  python run_realtime.py multi                                # 다중 CCTV (자동 구간)")
        print("  python run_realtime.py multi --lat 37.11 --lon 126.89       # 다중 CCTV (지정)")
        print("  python run_realtime.py multi --max-cameras 2                # 카메라 수 제한")
        print("  python run_realtime.py hotspot                              # 구간 선정 조회")
        print("  python run_realtime.py dry-run                              # 시뮬레이션")
        print("  python run_realtime.py status                               # 현황 확인")


if __name__ == "__main__":
    main()
