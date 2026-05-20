"""Track 3-B: ITS CCTV 스트림 조회 + Ring Buffer 녹화.

사고 이벤트의 좌표로 인근 CCTV를 찾고, RTSP/HLS 스트림을 녹화한다.

실행:
  python track3_cctv_stream.py list              # 전체 CCTV 목록 조회
  python track3_cctv_stream.py nearest 37.0 127.0 # 좌표 인근 CCTV 검색
  python track3_cctv_stream.py record <cctv_id>   # 단건 녹화 테스트 (30초)

사전 준비:
  data.go.kr → ITS 노드링크 CCTV API 키 발급 → .env에 ITS_CCTV_API_KEY=... 추가
  ffmpeg 설치 필수 (RTSP/HLS 녹화용)
"""
from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    ITS_BASE,
    MAX_CONCURRENT_STREAMS,
    RING_BUFFER_FPS,
    RING_BUFFER_POST_SEC,
    RING_BUFFER_PRE_SEC,
    STREAM_DIR,
)

logger = logging.getLogger(__name__)


@dataclass
class CCTVInfo:
    """CCTV 카메라 정보."""
    cctv_id: str
    name: str
    latitude: float
    longitude: float
    road_name: str = ""
    stream_url: str = ""
    resolution: str = ""


class ITSCCTVClient:
    """ITS CCTV 영상정보 API.

    엔드포인트: https://openapi.its.go.kr:9443/cctvInfo
    필수: apiKey, type(도로유형), cctvType(1:실시간/2:노선)
    선택: minX/maxX/minY/maxY, getType(json/xml)
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("ITS_API_KEY", "")
        self.base_url = ITS_BASE
        self._cache: list[CCTVInfo] = []

    def list_cctvs(self, min_x: float = 124.0, max_x: float = 132.0,
                   min_y: float = 33.0, max_y: float = 39.0) -> list[CCTVInfo]:
        """CCTV 목록 조회 (좌표 범위 지정)."""
        import requests

        if not self.api_key:
            logger.warning("ITS_API_KEY 미설정")
            return []

        url = f"{self.base_url}/cctvInfo"
        params = {
            "apiKey": self.api_key,
            "type": "all",
            "cctvType": "1",
            "minX": str(min_x),
            "maxX": str(max_x),
            "minY": str(min_y),
            "maxY": str(max_y),
            "getType": "json",
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("ITS CCTV API 실패: %s", e)
            return []

        cctvs = []
        items = data.get("response", {}).get("data", [])
        for item in items:
            name = item.get("cctvname", "")
            cctv_id = str(item.get("roadsectionid", "")) or _slugify(name)
            cctvs.append(CCTVInfo(
                cctv_id=cctv_id,
                name=name,
                latitude=float(item.get("coordy", 0)),
                longitude=float(item.get("coordx", 0)),
                stream_url=item.get("cctvurl", ""),
                resolution=item.get("cctvresolution", ""),
            ))

        self._cache = cctvs
        logger.info("CCTV %d대 조회 완료", len(cctvs))
        return cctvs

    def find_nearest(self, lat: float, lon: float, top_k: int = 3) -> list[tuple[CCTVInfo, float]]:
        """좌표 기준 인근 CCTV top_k개 검색.

        Returns:
            [(CCTVInfo, distance_km), ...] 거리순 정렬.
        """
        if not self._cache:
            self.list_cctvs()

        results = []
        for cctv in self._cache:
            dist = _haversine(lat, lon, cctv.latitude, cctv.longitude)
            results.append((cctv, dist))

        results.sort(key=lambda x: x[1])
        return results[:top_k]


class RingBuffer:
    """프레임 Ring Buffer — 항상 최근 N초 보유.

    ffmpeg subprocess로 RTSP/HLS 스트림을 읽고,
    사고 이벤트 시 pre + post 구간을 파일로 flush.
    """

    def __init__(self, cctv: CCTVInfo, pre_sec: int = RING_BUFFER_PRE_SEC,
                 post_sec: int = RING_BUFFER_POST_SEC):
        self.cctv = cctv
        self.pre_sec = pre_sec
        self.post_sec = post_sec
        self.output_dir = STREAM_DIR / "recordings"
        self._process: subprocess.Popen | None = None
        self._recording = False
        self._temp_dir = STREAM_DIR / "buffer_temp"

    def start_buffer(self):
        """Ring Buffer 시작 — ffmpeg로 세그먼트 녹화.

        ffmpeg -i <stream_url> -f segment -segment_time 10
               -segment_wrap 18 (= 180초 / 10초)
               buffer_temp/<cctv_id>_%03d.ts
        """
        if not self.cctv.stream_url:
            logger.error("스트림 URL 없음: %s", self.cctv.name)
            return

        self._temp_dir.mkdir(parents=True, exist_ok=True)
        segment_pattern = str(self._temp_dir / f"{self.cctv.cctv_id}_%03d.ts")
        segment_wrap = (self.pre_sec + self.post_sec) // 10

        cmd = [
            "ffmpeg", "-y",
            "-i", self.cctv.stream_url,
            "-c", "copy",
            "-f", "segment",
            "-segment_time", "10",
            "-segment_wrap", str(segment_wrap),
            "-reset_timestamps", "1",
            segment_pattern,
        ]

        logger.info("Ring Buffer 시작: %s (%s)", self.cctv.name, self.cctv.cctv_id)
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._recording = True

    def flush_event(self, event_id: str) -> Path | None:
        """사고 이벤트 발생 — post_sec 대기 후 세그먼트 병합.

        Returns:
            병합된 영상 파일 경로.
        """
        if not self._recording:
            logger.warning("Ring Buffer 미시작 상태")
            return None

        logger.info("이벤트 %s — %d초 후 녹화분 저장", event_id, self.post_sec)
        time.sleep(self.post_sec)

        # 세그먼트 파일 수집
        segments = sorted(self._temp_dir.glob(f"{self.cctv.cctv_id}_*.ts"))
        if not segments:
            logger.warning("세그먼트 파일 없음")
            return None

        # ffmpeg concat으로 병합
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{event_id}_{self.cctv.cctv_id}_{ts}.mp4"

        concat_list = self._temp_dir / f"concat_{self.cctv.cctv_id}.txt"
        with open(concat_list, "w") as f:
            for seg in segments:
                f.write(f"file '{seg}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(output_file),
        ]
        subprocess.run(cmd, capture_output=True)

        if output_file.exists():
            size_mb = output_file.stat().st_size / 1e6
            logger.info("녹화 저장: %s (%.1f MB)", output_file.name, size_mb)
            return output_file
        return None

    def stop(self):
        """Ring Buffer 중단."""
        if self._process:
            self._process.terminate()
            self._process.wait(timeout=5)
            self._process = None
        self._recording = False
        logger.info("Ring Buffer 중단: %s", self.cctv.name)


class StreamManager:
    """다수 CCTV 스트림 동시 관리."""

    def __init__(self):
        self.cctv_client = ITSCCTVClient()
        self.active_buffers: dict[str, RingBuffer] = {}

    def start_monitoring(self, cctvs: list[CCTVInfo]):
        """CCTV 리스트에 대해 Ring Buffer 시작."""
        for cctv in cctvs[:MAX_CONCURRENT_STREAMS]:
            if cctv.cctv_id not in self.active_buffers:
                buf = RingBuffer(cctv)
                buf.start_buffer()
                self.active_buffers[cctv.cctv_id] = buf

    def handle_incident(self, lat: float, lon: float, event_id: str) -> Path | None:
        """사고 좌표로 가장 가까운 CCTV의 녹화분 저장."""
        nearest = self.cctv_client.find_nearest(lat, lon, top_k=1)
        if not nearest:
            logger.warning("인근 CCTV 없음: lat=%f, lon=%f", lat, lon)
            return None

        cctv, dist_km = nearest[0]
        logger.info("인근 CCTV: %s (%.1f km)", cctv.name, dist_km)

        if dist_km > 5.0:
            logger.warning("CCTV 거리 %.1f km — 5km 초과, 녹화 건너뜀", dist_km)
            return None

        buf = self.active_buffers.get(cctv.cctv_id)
        if buf:
            return buf.flush_event(event_id)

        # 활성 버퍼에 없으면 즉시 시작 + 짧은 녹화
        buf = RingBuffer(cctv, pre_sec=0, post_sec=RING_BUFFER_POST_SEC)
        buf.start_buffer()
        result = buf.flush_event(event_id)
        buf.stop()
        return result

    def stop_all(self):
        """전체 버퍼 중단."""
        for buf in self.active_buffers.values():
            buf.stop()
        self.active_buffers.clear()


def _slugify(name: str) -> str:
    """CCTV 이름을 파일명 안전한 ID로 변환."""
    import re
    slug = re.sub(r"[^\w가-힣]", "_", name).strip("_")
    return slug or "unknown"


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 좌표 간 거리 (km)."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


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

    if cmd == "list":
        client = ITSCCTVClient()
        cctvs = client.list_cctvs()
        print(f"CCTV {len(cctvs)}대:")
        for c in cctvs[:20]:
            print(f"  {c.cctv_id}: {c.name} ({c.road_name}) [{c.latitude:.4f}, {c.longitude:.4f}]")
        if len(cctvs) > 20:
            print(f"  ... 외 {len(cctvs) - 20}대")

    elif cmd == "nearest":
        if len(sys.argv) < 4:
            print("사용법: python track3_cctv_stream.py nearest <lat> <lon>")
            return
        lat, lon = float(sys.argv[2]), float(sys.argv[3])
        client = ITSCCTVClient()
        results = client.find_nearest(lat, lon, top_k=5)
        print(f"({lat}, {lon}) 인근 CCTV:")
        for cctv, dist in results:
            print(f"  {dist:.1f}km — {cctv.name} ({cctv.cctv_id})")
            if cctv.stream_url:
                print(f"          URL: {cctv.stream_url[:80]}...")

    elif cmd == "record":
        if len(sys.argv) < 3:
            print("사용법: python track3_cctv_stream.py record <cctv_id>")
            return
        cctv_id = sys.argv[2]
        client = ITSCCTVClient()
        cctvs = client.list_cctvs()
        target = next((c for c in cctvs if c.cctv_id == cctv_id), None)
        if not target:
            print(f"CCTV {cctv_id} 없음")
            return
        buf = RingBuffer(target, pre_sec=0, post_sec=30)
        buf.start_buffer()
        print("30초 녹화 중...")
        result = buf.flush_event("test")
        buf.stop()
        if result:
            print(f"저장: {result}")

    else:
        print("사용법:")
        print("  python track3_cctv_stream.py list                # CCTV 목록")
        print("  python track3_cctv_stream.py nearest <lat> <lon> # 인근 검색")
        print("  python track3_cctv_stream.py record <cctv_id>    # 30초 녹화")


if __name__ == "__main__":
    main()
