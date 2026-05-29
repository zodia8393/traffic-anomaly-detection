"""다중 CCTV HLS 직접 다운로드 동시 녹화.

실행:
  python record_hls_multi.py --config cameras.json --duration 1h
  python record_hls_multi.py --config cameras.json --duration 24h
  python record_hls_multi.py --config cameras.json --continuous           # 24시간 로테이션
  python record_hls_multi.py --config cameras.json --continuous --rotation 12h
  python record_hls_multi.py --test   # 5분 테스트

cameras.json 형식:
  [{"name": "[경부선] 판교분기점", "url": "http://...", "route": "경부선"}, ...]
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
import threading
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
logger = logging.getLogger("record_multi")

OUTPUT_ROOT = Path("/DATA/cctv_recording")
MIN_DISK_GB = 10  # 이 미만이면 녹화 중단
WARN_DISK_GB = 60  # 이 미만이면 경고만

ITS_API_KEY = os.getenv("ITS_API_KEY", "")
ITS_CCTV_URL = "https://openapi.its.go.kr:9443/cctvInfo"
URL_REFRESH_SEC = 5400  # 90분마다 URL 갱신

_running = True


def _signal_handler(sig, frame):
    global _running
    logger.info("중단 신호 수신, 모든 녹화 종료 중...")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def check_disk_space(path: Path) -> tuple[float, bool]:
    """가용 디스크 공간(GB) 반환 + 녹화 가능 여부."""
    import shutil
    usage = shutil.disk_usage(str(path))
    avail_gb = usage.free / (1024**3)
    if avail_gb < MIN_DISK_GB:
        logger.error("디스크 부족: %.1fGB < %dGB, 녹화 중단", avail_gb, MIN_DISK_GB)
        return avail_gb, False
    if avail_gb < WARN_DISK_GB:
        logger.warning("디스크 여유 부족: %.1fGB < %dGB, 계속 진행", avail_gb, WARN_DISK_GB)
    return avail_gb, True


def convert_to_mp4(ts_path: Path, delete_ts: bool = True):
    """TS → MP4 변환 (코덱 복사). 별도 스레드에서 호출."""
    mp4_path = ts_path.with_suffix(".mp4")
    logger.info("MP4 변환 시작: %s", ts_path.name)
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(ts_path), "-c", "copy", str(mp4_path)],
            capture_output=True, text=True, timeout=1800,
        )
        if result.returncode == 0 and mp4_path.exists():
            ts_mb = ts_path.stat().st_size / (1024**2)
            mp4_mb = mp4_path.stat().st_size / (1024**2)
            logger.info("MP4 변환 완료: %s (%.0fMB → %.0fMB)", mp4_path.name, ts_mb, mp4_mb)
            if delete_ts:
                ts_path.unlink()
                logger.info("TS 삭제 완료: %s", ts_path.name)
        else:
            logger.warning("MP4 변환 실패: %s, returncode=%d", ts_path.name, result.returncode)
            if result.stderr:
                logger.warning("ffmpeg stderr: %s", result.stderr[-500:])
    except subprocess.TimeoutExpired:
        logger.warning("MP4 변환 타임아웃: %s", ts_path.name)
    except Exception as e:
        logger.warning("MP4 변환 에러: %s — %s", ts_path.name, e)


def refresh_cctv_url(cam: dict) -> str | None:
    """ITS API로 카메라 인근 CCTV URL을 갱신."""
    if not ITS_API_KEY:
        return None
    x, y = cam.get("x", 0), cam.get("y", 0)
    if not x or not y:
        return None
    delta = 0.05  # ~5km 범위 (좁으면 대상 CCTV 누락 가능)
    params = {
        "apiKey": ITS_API_KEY,
        "type": "ex",
        "cctvType": "1",
        "minX": str(x - delta), "maxX": str(x + delta),
        "minY": str(y - delta), "maxY": str(y + delta),
        "getType": "json",
    }
    try:
        resp = requests.get(ITS_CCTV_URL, params=params, timeout=30)
        data = resp.json()
        items = data.get("response", {}).get("data", [])
        if items:
            # 이름이 가장 비슷한 CCTV 선택
            cam_name = cam.get("name", "")
            clean_name = cam_name.replace("[", "").replace("]", "").replace(" ", "")
            for item in items:
                item_clean = item.get("cctvname", "").replace("[", "").replace("]", "").replace(" ", "")
                if clean_name in item_clean:
                    return item.get("cctvurl", "")
            # 못 찾으면 첫 번째 반환
            return items[0].get("cctvurl", "")
    except Exception as e:
        logger.warning("ITS URL 갱신 실패: %s", e)
    return None


def resolve_media_playlist(hls_url: str) -> tuple[str, str] | None:
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
        logger.warning("ffprobe 실패: %s", e)
    return None


def parse_playlist(text: str) -> list[tuple[str, float]]:
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


def _record_one_file(cam_idx: int, slug: str, hls_url_ref: list,
                     cam_config: dict, out_file: Path, duration_sec: float,
                     session: requests.Session) -> dict:
    """단일 파일 녹화 (duration_sec 동안). 결과 dict 반환.

    hls_url_ref: [url] 1-element list (mutable reference for URL refresh).
    """
    downloaded: set[str] = set()
    total_bytes = 0
    total_segments = 0
    total_duration = 0.0
    session_count = 0
    error_count = 0
    start_time = time.time()
    last_url_refresh = start_time

    with open(out_file, "wb") as f:
        while _running and (time.time() - start_time) < duration_sec:
            # 디스크 체크 (30분마다)
            if session_count % 120 == 0:
                _, ok = check_disk_space(out_file.parent)
                if not ok:
                    logger.error("[%d] %s: 디스크 부족으로 녹화 중단", cam_idx, slug)
                    break

            session_count += 1
            resolved = resolve_media_playlist(hls_url_ref[0])
            if not resolved:
                error_count += 1
                logger.warning("[%d] %s: media 해석 실패 (%d회), 5초 후 재시도",
                               cam_idx, slug, error_count)
                if error_count > 10:
                    # URL 만료 가능성 → ITS API로 갱신 시도
                    logger.info("[%d] %s: URL 갱신 시도 (연속 실패 %d회)", cam_idx, slug, error_count)
                    new_url = refresh_cctv_url(cam_config)
                    if new_url:
                        hls_url_ref[0] = new_url
                        last_url_refresh = time.time()
                        error_count = 0
                        downloaded.clear()
                        logger.info("[%d] %s: URL 갱신 완료", cam_idx, slug)
                        continue
                if error_count > 20:
                    logger.error("[%d] %s: 연속 실패 20회, 녹화 중단", cam_idx, slug)
                    break
                time.sleep(5)
                continue

            media_url, base_url = resolved
            if session_count <= 3 or session_count % 10 == 0:
                logger.info("[%d] %s: 세션 #%d", cam_idx, slug, session_count)
            error_count = 0
            consecutive_empty = 0

            while _running and (time.time() - start_time) < duration_sec:
                try:
                    resp = session.get(media_url, timeout=10)
                    if resp.status_code != 200:
                        logger.warning("[%d] %s: playlist HTTP %d, 재수립",
                                       cam_idx, slug, resp.status_code)
                        break

                    segments = parse_playlist(resp.text)
                    new_count = 0

                    for seg_name, seg_dur in segments:
                        seg_key = seg_name.split("?")[0]
                        if seg_key in downloaded:
                            continue
                        seg_url = base_url + seg_name
                        try:
                            sr = session.get(seg_url, timeout=10)
                            if sr.status_code == 200:
                                f.write(sr.content)
                                f.flush()
                                downloaded.add(seg_key)
                                total_bytes += len(sr.content)
                                total_segments += 1
                                total_duration += seg_dur
                                new_count += 1
                        except Exception:
                            pass

                    if new_count > 0:
                        consecutive_empty = 0
                    else:
                        consecutive_empty += 1

                    if consecutive_empty > 10:
                        break

                    time.sleep(1.5)
                except Exception:
                    break

            # 주기적 URL 갱신 (90분마다)
            if time.time() - last_url_refresh > URL_REFRESH_SEC:
                logger.info("[%d] %s: 주기적 URL 갱신 중...", cam_idx, slug)
                new_url = refresh_cctv_url(cam_config)
                if new_url:
                    hls_url_ref[0] = new_url
                    downloaded.clear()
                    logger.info("[%d] %s: URL 갱신 완료, 세그먼트 캐시 초기화", cam_idx, slug)
                last_url_refresh = time.time()

            time.sleep(2)

    elapsed = time.time() - start_time
    size_mb = total_bytes / (1024**2)
    return {
        "file": str(out_file),
        "segments": total_segments,
        "size_mb": round(size_mb, 1),
        "duration_min": round(total_duration / 60, 1),
        "sessions": session_count,
        "elapsed_sec": round(elapsed, 1),
    }


def record_camera(cam_idx: int, cam_config: dict, hls_url: str,
                  output_dir: Path, duration_sec: float, results: dict,
                  continuous: bool = False, rotation_sec: float = 86400):
    """카메라 1대 녹화. continuous=True이면 rotation_sec마다 파일 로테이션."""
    cam_name = cam_config["name"]
    slug = cam_name.replace("[", "").replace("]", "").replace(" ", "_")
    hls_url_ref = [hls_url]  # mutable reference for URL refresh

    session = requests.Session()
    all_files = []
    convert_threads: list[threading.Thread] = []

    # 시작 전 URL 유효성 확인, 실패 시 즉시 갱신 시도
    resolved = resolve_media_playlist(hls_url_ref[0])
    if not resolved:
        logger.warning("[%d] %s: 초기 URL 무효, ITS API로 갱신 시도", cam_idx, slug)
        new_url = refresh_cctv_url(cam_config)
        if new_url:
            hls_url_ref[0] = new_url
            logger.info("[%d] %s: URL 갱신 성공", cam_idx, slug)
        else:
            logger.error("[%d] %s: URL 갱신 실패, 기존 URL로 계속 시도", cam_idx, slug)

    while _running:
        # 날짜별 디렉토리
        now = datetime.now()
        today = now.strftime("%Y%m%d")
        cam_dir = output_dir / today / f"{slug}_hls"
        cam_dir.mkdir(parents=True, exist_ok=True)

        out_file = cam_dir / f"{slug}_{now.strftime('%Y%m%d_%H%M%S')}.ts"
        logger.info("[%d] 녹화 시작: %s → %s", cam_idx, cam_name, out_file.name)

        if continuous:
            # 자정 기준 로테이션: 다음 00:00까지 남은 초
            midnight = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            rec_sec = (midnight - now).total_seconds()
            if rec_sec < 60:
                rec_sec = 86400  # 자정 직전이면 다음날 통째로
            logger.info("[%d] %s: 자정까지 %.0f분 (%.0f초)", cam_idx, slug, rec_sec / 60, rec_sec)
        else:
            rec_sec = duration_sec

        file_result = _record_one_file(
            cam_idx, slug, hls_url_ref, cam_config, out_file, rec_sec, session,
        )
        file_result["name"] = cam_name
        all_files.append(file_result)

        size_mb = file_result["size_mb"]
        logger.info(
            "[%d] %s: 파일 완료 — %d세그먼트, %.1fMB, 영상 %.1f분, %d세션, %.0f초",
            cam_idx, slug, file_result["segments"], size_mb,
            file_result["duration_min"], file_result["sessions"],
            file_result["elapsed_sec"],
        )

        # MP4 변환 (별도 스레드, 종료 시 join 대기)
        if out_file.exists() and out_file.stat().st_size > 0:
            ct = threading.Thread(target=convert_to_mp4, args=(out_file,))
            ct.start()
            convert_threads.append(ct)
            # 완료된 스레드 정리
            convert_threads = [t for t in convert_threads if t.is_alive()]

        # continuous가 아니면 1회 녹화 후 종료
        if not continuous:
            break

    # 미완료 변환 스레드 대기
    for ct in convert_threads:
        if ct.is_alive():
            logger.info("[%d] %s: MP4 변환 완료 대기 중...", cam_idx, slug)
            ct.join(timeout=300)

    # 최종 결과: 마지막 파일 기준 (continuous에서는 누적)
    if all_files:
        last = all_files[-1]
        last["total_files"] = len(all_files)
        results[cam_idx] = last


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
    parser = argparse.ArgumentParser(description="다중 CCTV HLS 직접 다운로드 동시 녹화")
    parser.add_argument("--config", type=str, help="카메라 JSON 파일")
    parser.add_argument("--duration", type=str, default="5m", help="녹화 시간 (5m, 1h, 24h)")
    parser.add_argument("--continuous", action="store_true",
                        help="연속 녹화 모드 (rotation마다 파일 로테이션, 무한 반복)")
    parser.add_argument("--rotation", type=str, default="24h",
                        help="연속 녹화 시 파일 분할 주기 (기본 24h)")
    parser.add_argument("--test", action="store_true", help="5분 테스트")
    parser.add_argument("--output", type=str, default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    # --duration과 --continuous는 상호 배타
    if args.continuous and args.duration != "5m":
        logger.error("--continuous와 --duration은 동시 사용 불가")
        sys.exit(1)

    config_path = args.config or "/tmp/cameras_5routes.json"
    if not Path(config_path).exists():
        logger.error("카메라 설정 파일 없음: %s", config_path)
        sys.exit(1)

    with open(config_path) as f:
        cameras = json.load(f)

    duration_sec = 300 if args.test else parse_duration(args.duration)
    rotation_sec = parse_duration(args.rotation)
    base_output = Path(args.output)

    # 디스크 용량 사전 확인
    avail_gb, can_record = check_disk_space(base_output)
    if not can_record:
        sys.exit(1)
    logger.info("디스크 가용: %.1fGB", avail_gb)

    mode_str = "연속(로테이션 %s)" % args.rotation if args.continuous else "1회 %.0f분" % (duration_sec / 60)
    logger.info("=== 다중 CCTV HLS 녹화 시작 ===")
    logger.info("카메라: %d대, 모드: %s, 출력: %s", len(cameras), mode_str, base_output)
    for i, cam in enumerate(cameras):
        logger.info("  [%d] %s", i + 1, cam["name"])

    results = {}
    threads = []
    for i, cam in enumerate(cameras):
        t = threading.Thread(
            target=record_camera,
            args=(i + 1, cam, cam["url"], base_output, duration_sec, results),
            kwargs={"continuous": args.continuous, "rotation_sec": rotation_sec},
        )
        threads.append(t)

    t0 = time.time()
    for t in threads:
        t.start()
        time.sleep(0.5)

    for t in threads:
        t.join()

    elapsed = time.time() - t0
    total_size = sum(r["size_mb"] for r in results.values())
    total_dur = sum(r["duration_min"] for r in results.values())
    ok_count = sum(1 for r in results.values() if r["segments"] > 0)

    logger.info("\n=== 결과 요약 ===")
    for idx in sorted(results):
        r = results[idx]
        logger.info("  [%d] %s: %.1fMB, 영상 %.1f분, %d세션",
                     idx, r["name"], r["size_mb"], r["duration_min"], r["sessions"])

    logger.info("총: %d/%d대 성공, %.1fMB, 영상 %.1f분, 소요 %.0f초",
                ok_count, len(cameras), total_size, total_dur, elapsed)

    meta = {
        "start_time": datetime.fromtimestamp(t0).isoformat(),
        "end_time": datetime.now().isoformat(),
        "mode": "continuous" if args.continuous else "oneshot",
        "rotation_sec": rotation_sec if args.continuous else None,
        "duration_target_sec": None if args.continuous else duration_sec,
        "elapsed_sec": round(elapsed, 1),
        "cameras_total": len(cameras),
        "cameras_success": ok_count,
        "total_size_mb": round(total_size, 1),
        "total_duration_min": round(total_dur, 1),
        "results": {str(k): v for k, v in results.items()},
    }
    meta_file = base_output / "meta_multi.json"
    base_output.mkdir(parents=True, exist_ok=True)
    with open(meta_file, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info("메타: %s", meta_file)


if __name__ == "__main__":
    main()
