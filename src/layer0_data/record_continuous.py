"""사고다발 CCTV 1대 연속 녹화.

ITS API로 사고다발 구간 CCTV HLS URL을 가져와 ffmpeg로 연속 녹화한다.
1시간 단위 세그먼트 분할, 중단 명령까지 계속 녹화.

실행:
  python record_continuous.py                    # 즉시 시작
  python record_continuous.py --wait-until 00:00 # 자정까지 대기 후 시작
  python record_continuous.py --test             # API 테스트만 (녹화 안 함)

저장: /DATA/cctv_recording/{날짜}/{CCTV명}/
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv("/workspace/.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("record_continuous")

# ── 설정 ──────────────────────────────────────────────────────────────
ITS_API_KEY = os.getenv("ITS_API_KEY", "")
ITS_CCTV_URL = "https://openapi.its.go.kr:9443/cctvInfo"
OUTPUT_ROOT = Path("/DATA/cctv_recording")
SEGMENT_SEC = 0  # 0 = 통짜 (세그먼트 분할 안 함)
URL_REFRESH_SEC = 3600
DISK_CHECK_SEC = 600
DISK_MIN_GB = 50

TARGETS = [
    {
        "name": "경부선_오산IC",
        "min_x": 127.0, "max_x": 127.1,
        "min_y": 37.1, "max_y": 37.2,
        "road_type": "ex",
    },
    {
        "name": "중부선_이천",
        "min_x": 127.4, "max_x": 127.5,
        "min_y": 37.2, "max_y": 37.3,
        "road_type": "ex",
    },
    {
        "name": "영동선_여주JC",
        "min_x": 127.5, "max_x": 127.7,
        "min_y": 37.2, "max_y": 37.4,
        "road_type": "ex",
    },
]

_running = True


def _signal_handler(sig, frame):
    global _running
    logger.info("중단 신호 수신 (sig=%d), 녹화 종료 중...", sig)
    _running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def fetch_cctv_url(target: dict) -> tuple[str, str, str] | None:
    """ITS API에서 HLS URL 획득. Returns (name, url, cctv_id) or None."""
    params = {
        "apiKey": ITS_API_KEY,
        "type": target["road_type"],
        "cctvType": "1",
        "minX": str(target["min_x"]),
        "maxX": str(target["max_x"]),
        "minY": str(target["min_y"]),
        "maxY": str(target["max_y"]),
        "getType": "json",
    }
    try:
        resp = requests.get(ITS_CCTV_URL, params=params, timeout=30)
        data = resp.json()
    except Exception as e:
        logger.error("API 호출 실패: %s", e)
        return None

    header = data.get("header", data.get("response", {}).get("header", {}))
    if isinstance(header, dict):
        code = header.get("resultCode", 0)
        if code == 4001:
            logger.warning("API 일일 제한 초과 (4001)")
            return None

    items = data.get("response", {}).get("data", [])
    if not items:
        logger.warning("CCTV 없음: %s", target["name"])
        return None

    item = items[0]
    name = item.get("cctvname", "unknown")
    url = item.get("cctvurl", "")
    cctv_id = str(item.get("roadsectionid", "")) or name.replace(" ", "_")

    if not url:
        logger.warning("URL 없음: %s", name)
        return None

    logger.info("CCTV 발견: %s (URL: %s...)", name, url[:80])
    return name, url, cctv_id


def wait_until(target_time_str: str):
    """HH:MM 형식 시각까지 대기."""
    h, m = map(int, target_time_str.split(":"))
    now = datetime.now()
    target = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)

    wait_sec = (target - now).total_seconds()
    logger.info("대기: %s까지 %.0f초 (%.1f시간)", target.strftime("%Y-%m-%d %H:%M"), wait_sec, wait_sec / 3600)

    while _running and datetime.now() < target:
        time.sleep(min(30, (target - datetime.now()).total_seconds()))


def wait_for_api_ready() -> tuple[str, str, str]:
    """API 제한이 풀릴 때까지 5분 간격으로 재시도."""
    attempt = 0
    while _running:
        attempt += 1
        for target in TARGETS:
            result = fetch_cctv_url(target)
            if result:
                return result
        logger.info("API 미복구, 5분 후 재시도 (시도 #%d)", attempt)
        for _ in range(300):
            if not _running:
                sys.exit(0)
            time.sleep(1)
    sys.exit(0)


def check_disk(path: Path) -> bool:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < DISK_MIN_GB:
        logger.error("디스크 부족: %.1fGB (최소 %dGB)", free_gb, DISK_MIN_GB)
        return False
    return True


def slugify(name: str) -> str:
    return name.replace("[", "").replace("]", "").replace(" ", "_").replace("/", "_")


def record(cctv_name: str, hls_url: str, cctv_id: str):
    """ffmpeg로 연속 녹화. 1시간 세그먼트 분할."""
    today = datetime.now().strftime("%Y%m%d")
    slug = slugify(cctv_name)
    out_dir = OUTPUT_ROOT / today / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / "recording.log"
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    meta = {
        "cctv_name": cctv_name,
        "cctv_id": cctv_id,
        "start_time": datetime.now().isoformat(),
        "output_dir": str(out_dir),
        "segment_sec": SEGMENT_SEC,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    current_url = hls_url
    last_url_refresh = time.time()
    last_disk_check = time.time()
    process: subprocess.Popen | None = None
    segment_count = 0

    def start_ffmpeg(url: str) -> subprocess.Popen:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outfile = str(out_dir / f"{slug}_{ts}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "30",
            "-i", url,
            "-c", "copy",
            outfile,
        ]
        logger.info("ffmpeg 시작: %s", " ".join(cmd[:6]) + " ...")
        return subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    process = start_ffmpeg(current_url)
    logger.info("녹화 시작: %s → %s", cctv_name, out_dir)

    try:
        while _running:
            time.sleep(5)

            if process.poll() is not None:
                stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
                logger.warning("ffmpeg 종료 (code=%s), 재시작...\n%s", process.returncode, stderr[-500:])
                time.sleep(3)
                process = start_ffmpeg(current_url)
                continue

            new_segments = list(out_dir.glob(f"{slug}_*.mp4"))
            if len(new_segments) > segment_count:
                segment_count = len(new_segments)
                total_size = sum(f.stat().st_size for f in new_segments) / (1024 ** 3)
                logger.info("세그먼트 %d개, 총 %.2fGB", segment_count, total_size)

            now = time.time()
            if now - last_disk_check > DISK_CHECK_SEC:
                last_disk_check = now
                if not check_disk(OUTPUT_ROOT):
                    logger.error("디스크 부족으로 녹화 중단")
                    break

            # URL 갱신은 ffmpeg 재시작(=새 파일)이 필요하므로, 프로세스가 죽었을 때만 적용
            if now - last_url_refresh > URL_REFRESH_SEC:
                last_url_refresh = now
                for target in TARGETS:
                    result = fetch_cctv_url(target)
                    if result:
                        _, new_url, _ = result
                        if new_url != current_url:
                            current_url = new_url
                            logger.info("URL 갱신 대기 (다음 재시작 시 적용)")
                        break

    finally:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()

        meta["end_time"] = datetime.now().isoformat()
        meta["segments"] = segment_count
        segments = sorted(out_dir.glob(f"{slug}_*.mp4"))
        meta["total_size_gb"] = round(sum(f.stat().st_size for f in segments) / (1024 ** 3), 2)
        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        logger.info("녹화 종료. 세그먼트 %d개, %.2fGB", meta["segments"], meta["total_size_gb"])


def main():
    import argparse
    parser = argparse.ArgumentParser(description="사고다발 CCTV 연속 녹화")
    parser.add_argument("--wait-until", type=str, default=None, help="HH:MM 시각까지 대기 후 시작")
    parser.add_argument("--test", action="store_true", help="API 테스트만")
    args = parser.parse_args()

    if not ITS_API_KEY:
        logger.error("ITS_API_KEY 미설정 (/workspace/.env)")
        sys.exit(1)

    if args.wait_until:
        wait_until(args.wait_until)

    if not _running:
        return

    logger.info("CCTV URL 획득 시도...")
    cctv_name, hls_url, cctv_id = wait_for_api_ready()

    if args.test:
        logger.info("테스트 완료: %s → %s", cctv_name, hls_url[:80])
        return

    if not check_disk(OUTPUT_ROOT):
        sys.exit(1)

    record(cctv_name, hls_url, cctv_id)


if __name__ == "__main__":
    main()
