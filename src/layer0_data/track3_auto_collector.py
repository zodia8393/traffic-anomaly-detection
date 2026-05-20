"""Track 3: 자동 수집 시스템 통합 — 돌발정보 + CCTV 녹화 + 품질 검증.

돌발정보 API → 사고 감지 → 인근 CCTV Ring Buffer flush → 품질 검증 → DB 적재.

실행:
  python track3_auto_collector.py run      # 전체 시스템 가동 (Ctrl+C 중단)
  python track3_auto_collector.py status   # 수집 현황 확인
  python track3_auto_collector.py dry-run  # API 없이 시뮬레이션

아키텍처:
  [돌발정보 폴러] --5분--> [사고 감지] --> [CCTV 스트림 조회]
                                        --> [Ring Buffer flush]
                                        --> [품질 검증]
                                        --> [영상 저장 + 메타 기록]
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import STREAM_DIR
from quality_gate import check_video
from track3_api_incident import IncidentEvent, IncidentPoller
from track3_cctv_stream import StreamManager

logger = logging.getLogger(__name__)


class AutoCollector:
    """돌발정보 → CCTV 녹화 → 품질 검증 통합 파이프라인."""

    def __init__(self):
        self.poller = IncidentPoller()
        self.stream_mgr = StreamManager()
        self.meta_dir = STREAM_DIR / "metadata"
        self.stats = {
            "events_detected": 0,
            "recordings_attempted": 0,
            "recordings_saved": 0,
            "quality_passed": 0,
            "quality_failed": 0,
        }

    def handle_events(self, events: list[IncidentEvent]):
        """사고 이벤트 리스트 처리 — 인근 CCTV 녹화 트리거."""
        for ev in events:
            self.stats["events_detected"] += 1
            logger.info(
                "사고 감지: [%s] %s %s — %s",
                ev.road_type, ev.road_name, ev.direction,
                ev.message[:50],
            )

            if ev.latitude is None or ev.longitude is None:
                logger.warning("좌표 없음 — 녹화 건너뜀")
                continue

            self.stats["recordings_attempted"] += 1
            video_path = self.stream_mgr.handle_incident(
                lat=ev.latitude,
                lon=ev.longitude,
                event_id=ev.event_id,
            )

            if video_path and video_path.exists():
                self.stats["recordings_saved"] += 1

                # 품질 검증
                qr = check_video(video_path)
                if qr.passed:
                    self.stats["quality_passed"] += 1
                    self._save_metadata(ev, video_path, qr)
                    logger.info("품질 통과: %s", video_path.name)
                else:
                    self.stats["quality_failed"] += 1
                    logger.warning("품질 미달: %s — %s",
                                   video_path.name, ", ".join(qr.failures))

    def run(self, poll_interval: int = 300):
        """메인 루프 — 돌발정보 폴링 + 사고 시 녹화."""
        logger.info("=" * 60)
        logger.info("자동 수집 시스템 시작")
        logger.info("  폴링 간격: %d초", poll_interval)
        logger.info("  저장 경로: %s", STREAM_DIR)
        logger.info("=" * 60)

        self.poller.watch(
            interval_sec=poll_interval,
            callback=self.handle_events,
        )

    def run_dry(self):
        """드라이런 — 가짜 이벤트로 파이프라인 흐름 확인."""
        logger.info("=== 드라이런 모드 ===")

        fake_events = [
            IncidentEvent(
                event_id="DRY001",
                event_type="교통사고",
                event_detail="추돌사고",
                road_type="고속도로",
                road_name="영동고속도로",
                road_no="50",
                direction="부산",
                latitude=37.2932,
                longitude=127.6356,
                occurred_at=datetime.now().strftime("%Y%m%d%H%M%S"),
                message="드라이런: 여주JC 인근 추돌사고 시뮬레이션",
                link_id="DRY001",
            ),
        ]

        logger.info("가짜 이벤트 %d건 생성", len(fake_events))
        for ev in fake_events:
            logger.info("  [%s] %s %s", ev.road_type, ev.road_name, ev.direction)

        logger.info("")
        logger.info("파이프라인 흐름:")
        logger.info("  1. 돌발정보 감지 → OK")
        logger.info("  2. 인근 CCTV 검색 → (API 키 필요)")
        logger.info("  3. Ring Buffer flush → (스트림 URL 필요)")
        logger.info("  4. 품질 검증 → (영상 파일 필요)")
        logger.info("  5. 메타데이터 저장 → OK")
        logger.info("")

        # 메타데이터 저장 테스트
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self.meta_dir / "dry_run_sample.json"
        meta = {
            "event_id": "DRY001",
            "source": "dry_run",
            "road_name": "영동고속도로",
            "direction": "부산",
            "km_post": 52.3,
            "latitude": 37.2932,
            "longitude": 127.6356,
            "occurred_at": datetime.now().isoformat(),
            "video_path": None,
            "quality_check": None,
            "collected_at": datetime.now().isoformat(),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info("메타데이터 샘플 → %s", meta_path)

        self.print_status()

    def print_status(self):
        """수집 현황 출력."""
        # 저장된 영상 수
        rec_dir = STREAM_DIR / "recordings"
        saved = list(rec_dir.glob("*.mp4")) if rec_dir.exists() else []
        meta_files = list(self.meta_dir.glob("*.json")) if self.meta_dir.exists() else []
        log_dir = STREAM_DIR / "incident_logs"
        log_files = list(log_dir.glob("*.jsonl")) if log_dir.exists() else []

        print()
        print("=== 수집 현황 ===")
        print(f"  녹화 영상  : {len(saved)}건 ({STREAM_DIR / 'recordings'})")
        print(f"  메타데이터 : {len(meta_files)}건 ({self.meta_dir})")
        print(f"  이벤트 로그: {len(log_files)}건 ({log_dir})")
        if self.stats["events_detected"]:
            print(f"  세션 통계  : 감지 {self.stats['events_detected']}건, "
                  f"녹화 {self.stats['recordings_saved']}건, "
                  f"품질통과 {self.stats['quality_passed']}건")
        print()

        # API 키 상태
        import os
        keys = {
            "ITS_API_KEY": "ITS 돌발상황+CCTV",
            "AIHUB_API_KEY": "AI Hub 데이터셋",
            "ROADPLUS_API_KEY": "도로공사 교통량",
        }
        print("  API 키 상태:")
        for env_var, label in keys.items():
            status = "설정됨" if os.getenv(env_var) else "미설정"
            print(f"    {label}: {status} ({env_var})")
        print()

    def _save_metadata(self, event: IncidentEvent, video_path: Path, qr):
        """녹화 메타데이터 저장."""
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        meta_path = self.meta_dir / f"{event.event_id}_{ts}.json"

        meta = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "road_type": event.road_type,
            "road_name": event.road_name,
            "direction": event.direction,
            "latitude": event.latitude,
            "longitude": event.longitude,
            "occurred_at": event.occurred_at,
            "message": event.message,
            "video_path": str(video_path),
            "quality_check": {
                "passed": qr.passed,
                "width": qr.width,
                "height": qr.height,
                "fps": qr.fps,
                "duration_sec": qr.duration_sec,
            },
            "collected_at": datetime.now().isoformat(),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


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

    collector = AutoCollector()

    if cmd == "run":
        collector.run()

    elif cmd == "status":
        collector.print_status()

    elif cmd == "dry-run":
        collector.run_dry()

    else:
        print("Track 3 자동 수집 시스템")
        print()
        print("사용법:")
        print("  python track3_auto_collector.py run      # 시스템 가동")
        print("  python track3_auto_collector.py status   # 현황 확인")
        print("  python track3_auto_collector.py dry-run  # 시뮬레이션")
        print()
        print("사전 준비:")
        print("  1. its.go.kr에서 ITS API 키 발급")
        print("  2. /workspace/.env에 추가:")
        print("     ITS_API_KEY=...")
        print("  3. ffmpeg 설치 확인")


if __name__ == "__main__":
    main()
