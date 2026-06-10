"""HLS .ts 세그먼트 직접 다운로드 방식 CCTV 녹화.

ffmpeg 대신 Python으로 m3u8 플레이리스트를 파싱하고 .ts 세그먼트를
직접 다운로드하여 단일 파일에 append. 프래그먼트 없는 연속 녹화.

실행:
  python record_hls.py --test              # 5분 테스트
  python record_hls.py --duration 1h       # 1시간 녹화
  python record_hls.py --duration 24h      # 24시간 녹화

비교용: record_50.py (ffmpeg 방식)와 동일 CCTV를 동시에 녹화하여 비교 가능.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv

load_dotenv("/workspace/.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("record_hls")

ITS_API_KEY = os.getenv("ITS_API_KEY", "")
ITS_CCTV_URL = "https://openapi.its.go.kr:9443/cctvInfo"
OUTPUT_ROOT = Path("/DATA/cctv_recording")

_running = True


def _signal_handler(sig, frame):
    global _running
    logger.info("중단 신호 수신, 녹화 종료 중...")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def fetch_cctv_url(name: str = "경부선_남사") -> tuple[str, str] | None:
    params = {
        "apiKey": ITS_API_KEY,
        "type": "ex",
        "cctvType": "1",
        "minX": "127.0", "maxX": "127.1",
        "minY": "37.1", "maxY": "37.2",
        "getType": "json",
    }
    try:
        resp = requests.get(ITS_CCTV_URL, params=params, timeout=30)
        data = resp.json()
    except Exception as e:
        logger.error("API 호출 실패: %s", e)
        return None

    items = data.get("response", {}).get("data", [])
    if not items:
        return None

    item = items[0]
    return item.get("cctvname", "unknown"), item.get("cctvurl", "")


def resolve_media_playlist(hls_url: str) -> tuple[str, str] | None:
    """ffprobe로 실제 media playlist URL과 base URL을 추출."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "verbose", "-i", hls_url,
             "-show_entries", "format=duration", "-of", "csv=p=0"],
            capture_output=True, text=True, timeout=15,
        )
        for line in result.stderr.split("\n"):
            if "main_stream.m3u8" in line and "Opening" in line:
                match = re.search(r"'(http[^']+main_stream\.m3u8[^']*)'", line)
                if match:
                    media_url = match.group(1)
                    base_url = media_url.rsplit("/", 1)[0] + "/"
                    return media_url, base_url
    except Exception as e:
        logger.error("ffprobe 실패: %s", e)
    return None


def parse_playlist(text: str) -> list[tuple[str, float]]:
    """m3u8 텍스트에서 (세그먼트 파일명, 길이초) 리스트 추출."""
    segments = []
    duration = 0.0
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("#EXTINF:"):
            duration = float(line.split(":")[1].rstrip(","))
        elif not line.startswith("#") and line:
            segments.append((line, duration))
            duration = 0.0
    return segments


def record_single(hls_url: str, cctv_name: str, output_dir: Path,
                   duration_sec: float):
    """단일 CCTV HLS 직접 다운로드 녹화."""
    slug = cctv_name.replace("[", "").replace("]", "").replace(" ", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    ts_start = datetime.now()
    out_file = output_dir / f"{slug}_{ts_start.strftime('%Y%m%d_%H%M%S')}.ts"
    log_file = output_dir / "recording_hls.log"

    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    logger.info("HLS 직접 녹화 시작: %s → %s", cctv_name, out_file)
    logger.info("목표: %.1f시간, 방식: m3u8 세그먼트 직접 다운로드", duration_sec / 3600)

    session = requests.Session()
    session.timeout = 10

    downloaded = set()
    total_bytes = 0
    total_segments = 0
    total_duration = 0.0
    session_count = 0
    error_count = 0
    last_url_refresh = time.time()
    current_hls_url = hls_url

    with open(out_file, "wb") as f:
        while _running and (time.time() - time.mktime(ts_start.timetuple())) < duration_sec:
            # 세션 수립: media playlist URL 획득
            session_count += 1
            resolved = resolve_media_playlist(current_hls_url)
            if not resolved:
                logger.warning("media playlist 해석 실패, 5초 후 재시도")
                time.sleep(5)
                error_count += 1
                if error_count > 10:
                    logger.error("연속 실패 10회, URL 갱신 시도")
                    result = fetch_cctv_url()
                    if result:
                        _, current_hls_url = result
                        last_url_refresh = time.time()
                        error_count = 0
                    else:
                        time.sleep(30)
                continue

            media_url, base_url = resolved
            logger.info("세션 #%d 시작 (media: %s...)", session_count, media_url[:80])
            error_count = 0
            session_start = time.time()
            consecutive_empty = 0

            # 세그먼트 폴링 루프
            while _running and (time.time() - time.mktime(ts_start.timetuple())) < duration_sec:
                try:
                    resp = session.get(media_url, timeout=10)
                    if resp.status_code != 200:
                        logger.warning("playlist HTTP %d, 세션 재수립", resp.status_code)
                        break

                    segments = parse_playlist(resp.text)
                    new_count = 0

                    for seg_name, seg_dur in segments:
                        seg_key = seg_name.split("?")[0]
                        if seg_key in downloaded:
                            continue

                        seg_url = base_url + seg_name
                        try:
                            seg_resp = session.get(seg_url, timeout=10)
                            if seg_resp.status_code == 200:
                                f.write(seg_resp.content)
                                f.flush()
                                downloaded.add(seg_key)
                                total_bytes += len(seg_resp.content)
                                total_segments += 1
                                total_duration += seg_dur
                                new_count += 1
                            else:
                                logger.warning("세그먼트 HTTP %d: %s", seg_resp.status_code, seg_key[-30:])
                        except Exception as e:
                            logger.warning("세그먼트 다운로드 실패: %s", e)

                    if new_count > 0:
                        consecutive_empty = 0
                    else:
                        consecutive_empty += 1

                    # 세션 만료 감지 (연속 10회 새 세그먼트 없음)
                    if consecutive_empty > 10:
                        elapsed_session = time.time() - session_start
                        logger.info("세션 만료 (%.0f초), 재수립...", elapsed_session)
                        break

                    # 상태 로깅 (60초마다)
                    elapsed = time.time() - time.mktime(ts_start.timetuple())
                    if total_segments % 30 == 0 and total_segments > 0:
                        logger.info(
                            "진행: %.1f분, %d세그먼트, %.1fMB, 영상 %.1f분",
                            elapsed / 60, total_segments,
                            total_bytes / (1024**2), total_duration / 60,
                        )

                    time.sleep(1.5)

                except Exception as e:
                    logger.warning("폴링 에러: %s", e)
                    time.sleep(3)
                    break

            # HLS URL 갱신 (90분마다)
            if time.time() - last_url_refresh > 5400:
                logger.info("HLS URL 갱신 중 (API 1회)...")
                result = fetch_cctv_url()
                if result:
                    _, current_hls_url = result
                    last_url_refresh = time.time()
                    downloaded.clear()
                    logger.info("URL 갱신 완료, 세그먼트 캐시 초기화")

            time.sleep(2)

    elapsed_total = time.time() - time.mktime(ts_start.timetuple())
    file_size_gb = total_bytes / (1024**3)

    meta = {
        "cctv_name": cctv_name,
        "method": "hls_direct",
        "start_time": ts_start.isoformat(),
        "end_time": datetime.now().isoformat(),
        "elapsed_sec": round(elapsed_total, 1),
        "total_segments": total_segments,
        "total_duration_sec": round(total_duration, 1),
        "total_size_gb": round(file_size_gb, 3),
        "session_count": session_count,
        "output_file": str(out_file),
    }
    (output_dir / "meta_hls.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    logger.info(
        "녹화 종료. 세그먼트 %d개, 영상 %.1f분, %.2fGB, 세션 %d회",
        total_segments, total_duration / 60, file_size_gb, session_count,
    )
    return meta


def parse_duration(s: str) -> float:
    s = s.strip().lower()
    if s.endswith("h"):
        return float(s[:-1]) * 3600
    if s.endswith("m"):
        return float(s[:-1]) * 60
    if s.endswith("s"):
        return float(s[:-1])
    return float(s)


def main():
    parser = argparse.ArgumentParser(description="HLS 직접 다운로드 CCTV 녹화")
    parser.add_argument("--duration", type=str, default="5m", help="녹화 시간 (5m, 1h, 24h)")
    parser.add_argument("--test", action="store_true", help="5분 테스트")
    parser.add_argument("--output", type=str, default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    if not ITS_API_KEY:
        logger.error("ITS_API_KEY 미설정")
        sys.exit(1)

    duration_sec = 300 if args.test else parse_duration(args.duration)
    logger.info("CCTV URL 획득 중...")
    result = fetch_cctv_url()
    if not result:
        logger.error("CCTV URL 획득 실패")
        sys.exit(1)

    cctv_name, hls_url = result
    logger.info("대상: %s", cctv_name)

    today = datetime.now().strftime("%Y%m%d")
    slug = cctv_name.replace("[", "").replace("]", "").replace(" ", "_")
    output_dir = Path(args.output) / today / f"{slug}_hls"

    record_single(hls_url, cctv_name, output_dir, duration_sec)


if __name__ == "__main__":
    main()
