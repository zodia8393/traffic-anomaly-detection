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
import shutil
import subprocess
import sys
import threading
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
logger = logging.getLogger("record_multi")

OUTPUT_ROOT = Path("/DATA/cctv_recording")
MIN_DISK_GB = 10  # 이 미만이면 녹화 중단
WARN_DISK_GB = 60  # 이 미만이면 경고만

ITS_API_KEY = os.getenv("ITS_API_KEY", "")
ITS_CCTV_URL = "https://openapi.its.go.kr:9443/cctvInfo"
ITS_WEB_BASE = "https://www.its.go.kr"
URL_REFRESH_SEC = 5400  # 90분마다 URL 갱신
CAMERA_RESTART_BASE_SEC = 15
CAMERA_RESTART_MAX_SEC = 300
DEFAULT_CONFIG = Path(__file__).resolve().parent / "cameras_5.json"
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) CCTVRecorder/1.0",
    "Accept": "*/*",
}
FFMPEG_BIN = os.getenv("FFMPEG_BIN") or shutil.which("ffmpeg") or "/home/ybs/.local/bin/ffmpeg"
FFPROBE_BIN = os.getenv("FFPROBE_BIN") or shutil.which("ffprobe") or "/home/ybs/.local/bin/ffprobe"

_running = True
_WEB_SESSION: requests.Session | None = None
_WEB_MARKER_CACHE = {"loaded_at": 0.0, "items": []}
_WEB_CACHE_LOCK = threading.RLock()
_OPENAPI_DISABLED_UNTIL = 0.0


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
            [FFMPEG_BIN, "-y", "-i", str(ts_path), "-c", "copy", str(mp4_path)],
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


def _cctv_type_candidates(cam: dict) -> list[str]:
    candidates = []
    for value in ("1", cam.get("cctvtype"), "4"):
        if value is None:
            continue
        value = str(value)
        if value and value not in candidates:
            candidates.append(value)
    return candidates


def _pick_cctv_url(items: list[dict], cam_name: str) -> str:
    clean_name = cam_name.replace("[", "").replace("]", "").replace(" ", "")
    for item in items:
        item_clean = item.get("cctvname", "").replace("[", "").replace("]", "").replace(" ", "")
        if clean_name and clean_name in item_clean:
            return item.get("cctvurl", "")
    return items[0].get("cctvurl", "")


def _meta_content(html: str, name: str) -> str | None:
    match = re.search(
        r'<meta[^>]+name=["\']' + re.escape(name) + r'["\'][^>]*content=["\']([^"\']+)',
        html,
    )
    return match.group(1) if match else None


def _its_web_session() -> requests.Session:
    global _WEB_SESSION
    if _WEB_SESSION is not None:
        return _WEB_SESSION

    session = requests.Session()
    session.headers.update({
        **HTTP_HEADERS,
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Content-Type": "application/json",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": ITS_WEB_BASE,
        "Referer": ITS_WEB_BASE + "/",
    })
    try:
        home = session.get(ITS_WEB_BASE + "/", timeout=30)
        header = _meta_content(home.text, "_csrf_header")
        token = _meta_content(home.text, "_csrf")
        if header and token:
            session.headers.update({header: token})
    except requests.RequestException as e:
        logger.warning("ITS 웹 세션 초기화 실패: %s", type(e).__name__)
    _WEB_SESSION = session
    return session


def _reset_its_web_session() -> None:
    global _WEB_SESSION
    with _WEB_CACHE_LOCK:
        if _WEB_SESSION is not None:
            try:
                _WEB_SESSION.close()
            except Exception:
                pass
        _WEB_SESSION = None
        _WEB_MARKER_CACHE["loaded_at"] = 0.0
        _WEB_MARKER_CACHE["items"] = []


def _load_its_web_cctvs(force: bool = False) -> list[dict]:
    now = time.time()
    with _WEB_CACHE_LOCK:
        if not force and _WEB_MARKER_CACHE["items"] and now - _WEB_MARKER_CACHE["loaded_at"] < 3600:
            return list(_WEB_MARKER_CACHE["items"])

        session = _its_web_session()
        payload = {"body": {"data": {"type": "CCTV"}}}
        try:
            resp = session.post(ITS_WEB_BASE + "/map/getMarkers", data=json.dumps(payload), timeout=40)
        except requests.RequestException as e:
            logger.warning("ITS 웹 CCTV 마커 조회 실패: %s", type(e).__name__)
            _reset_its_web_session()
            return []
        if resp.status_code != 200:
            logger.warning("ITS 웹 CCTV 마커 조회 실패: HTTP %s", resp.status_code)
            if resp.status_code in (401, 403, 429, 500, 502, 503, 504):
                _reset_its_web_session()
            return []

        infos = []
        try:
            features = resp.json().get("features", [])
        except ValueError:
            logger.warning("ITS 웹 CCTV 마커 응답 JSON 해석 실패")
            return []
        for feature in features:
            raw = (feature.get("properties") or {}).get("INFO")
            try:
                info = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                continue
            if isinstance(info, dict) and info.get("x") and info.get("y"):
                infos.append(info)

        _WEB_MARKER_CACHE["loaded_at"] = now
        _WEB_MARKER_CACHE["items"] = infos
        logger.info("ITS 웹 CCTV 마커 조회 완료: %d건", len(infos))
        return list(infos)


def _camera_match_score(cam: dict, info: dict) -> tuple[int, int, float]:
    cam_name = cam.get("name", "").replace("[", "").replace("]", "").replace(" ", "")
    info_name = str(info.get("instlLcDc", "")).replace("[", "").replace("]", "").replace(" ", "")
    route = str(cam.get("route", "")).replace(" ", "")
    info_route = (str(info.get("instlLcDc", "")) + str(info.get("detailDc", ""))).replace(" ", "")
    try:
        dx = float(info.get("x")) - float(cam.get("x"))
        dy = float(info.get("y")) - float(cam.get("y"))
        dist2 = dx * dx + dy * dy
    except Exception:
        dist2 = 999.0
    name_penalty = 0 if cam_name and cam_name in info_name else 1
    route_penalty = 0 if route and route in info_route else 1
    return name_penalty, route_penalty, dist2


def refresh_cctv_url_from_web(cam: dict) -> str | None:
    """ITS 웹 지도 경로로 현재 HLS URL을 조회한다. OpenAPI 키 장애 시 폴백."""
    try:
        items = _load_its_web_cctvs()
    except Exception as e:
        logger.warning("ITS 웹 CCTV 마커 조회 예외: %s", type(e).__name__)
        _reset_its_web_session()
        return None
    if not items:
        return None

    session = _its_web_session()
    for info in sorted(items, key=lambda item: _camera_match_score(cam, item))[:5]:
        source_url = info.get("appUrl") or info.get("webUrl") or ""
        if not source_url:
            continue
        if source_url.startswith("http://"):
            source_url = "https://" + source_url[len("http://"):]
        target_url = source_url if source_url.endswith("!hls") else source_url + "!hls"

        try:
            resp = session.post(
                ITS_WEB_BASE + "/api/cctv/hls",
                data=json.dumps({"targetUrl": target_url}),
                timeout=20,
            )
            if resp.status_code != 200:
                logger.warning("ITS 웹 HLS 조회 실패: HTTP %s", resp.status_code)
                continue
            payload = resp.json()
            stream_url = payload.get("streamUrl") if isinstance(payload, dict) else str(payload)
        except Exception as e:
            logger.warning("ITS 웹 HLS 조회 실패: %s", type(e).__name__)
            if isinstance(e, requests.RequestException):
                _reset_its_web_session()
            continue

        if stream_url and resolve_media_playlist(stream_url):
            logger.info("ITS 웹 HLS URL 선택: %s", info.get("instlLcDc", cam.get("name", "")))
            return stream_url
    return None


def refresh_cctv_url(cam: dict) -> str | None:
    """ITS API로 카메라 인근 CCTV URL을 갱신."""
    global _OPENAPI_DISABLED_UNTIL
    x, y = float(cam.get("x") or 0), float(cam.get("y") or 0)
    if ITS_API_KEY and x and y and time.time() >= _OPENAPI_DISABLED_UNTIL:
        delta = 0.05  # ~5km 범위 (좁으면 대상 CCTV 누락 가능)

        for cctv_type in _cctv_type_candidates(cam):
            params = {
                "apiKey": ITS_API_KEY,
                "type": "ex",
                "cctvType": cctv_type,
                "minX": str(x - delta), "maxX": str(x + delta),
                "minY": str(y - delta), "maxY": str(y + delta),
                "getType": "json",
            }
            try:
                resp = requests.get(ITS_CCTV_URL, params=params, headers=HTTP_HEADERS, timeout=30)
                if resp.status_code != 200:
                    logger.warning("ITS URL 갱신 HTTP %s(cctvType=%s)", resp.status_code, cctv_type)
                    if resp.status_code == 401:
                        _OPENAPI_DISABLED_UNTIL = time.time() + 3600
                        logger.warning("ITS OpenAPI 인증 실패 — 1시간 동안 웹 지도 폴백 우선 사용")
                        break
                    continue
                data = resp.json()
                items = data.get("response", {}).get("data", [])
            except Exception as e:
                logger.warning("ITS URL 갱신 실패(cctvType=%s): %s", cctv_type, type(e).__name__)
                continue

            if not items:
                logger.warning("ITS URL 후보 없음(cctvType=%s)", cctv_type)
                continue

            url = _pick_cctv_url(items, cam.get("name", ""))
            if not url:
                continue
            if resolve_media_playlist(url):
                logger.info("ITS URL 갱신 후보 선택(cctvType=%s, count=%d)", cctv_type, len(items))
                return url
            logger.warning("ITS URL 후보 해석 실패(cctvType=%s), 다음 후보 시도", cctv_type)

    return refresh_cctv_url_from_web(cam)


def resolve_media_playlist(hls_url: str) -> tuple[str, str] | None:
    try:
        result = subprocess.run(
            [FFPROBE_BIN, "-v", "verbose", "-i", hls_url,
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

    try:
        resp = requests.get(hls_url, headers=HTTP_HEADERS, timeout=10)
        if resp.status_code != 200 or "#EXTM3U" not in resp.text:
            return None
        lines = [line.strip() for line in resp.text.splitlines()
                 if line.strip() and not line.startswith("#")]
        if not lines:
            return None
        nested = [line for line in lines if ".m3u8" in line]
        if nested:
            preferred = next((line for line in nested if "main_stream" in line), nested[0])
            media_url = urljoin(hls_url, preferred)
            base_url = media_url.rsplit("/", 1)[0] + "/"
            return media_url, base_url
        base_url = hls_url.rsplit("/", 1)[0] + "/"
        return hls_url, base_url
    except Exception as e:
        logger.warning("HLS playlist 직접 해석 실패: %s", e)
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
                    try:
                        new_url = refresh_cctv_url(cam_config)
                    except Exception as e:
                        logger.warning("[%d] %s: URL 갱신 예외: %s",
                                       cam_idx, slug, type(e).__name__)
                        _reset_its_web_session()
                        new_url = None
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
                        seg_url = urljoin(base_url, seg_name)
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
                try:
                    new_url = refresh_cctv_url(cam_config)
                except Exception as e:
                    logger.warning("[%d] %s: 주기적 URL 갱신 예외: %s",
                                   cam_idx, slug, type(e).__name__)
                    _reset_its_web_session()
                    new_url = None
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
    session.headers.update(HTTP_HEADERS)
    all_files = []
    convert_threads: list[threading.Thread] = []
    restart_delay = CAMERA_RESTART_BASE_SEC

    # 시작 전 URL 유효성 확인, 실패 시 즉시 갱신 시도
    resolved = resolve_media_playlist(hls_url_ref[0])
    if not resolved:
        logger.warning("[%d] %s: 초기 URL 무효, ITS API로 갱신 시도", cam_idx, slug)
        try:
            new_url = refresh_cctv_url(cam_config)
        except Exception as e:
            logger.warning("[%d] %s: 초기 URL 갱신 예외: %s",
                           cam_idx, slug, type(e).__name__)
            _reset_its_web_session()
            new_url = None
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

        try:
            file_result = _record_one_file(
                cam_idx, slug, hls_url_ref, cam_config, out_file, rec_sec, session,
            )
            restart_delay = CAMERA_RESTART_BASE_SEC
        except Exception:
            logger.exception(
                "[%d] %s: 카메라 작업 예외, %.0f초 후 새 세션으로 재시작",
                cam_idx, slug, restart_delay,
            )
            if out_file.exists() and out_file.stat().st_size > 0:
                ct = threading.Thread(target=convert_to_mp4, args=(out_file,))
                ct.start()
                convert_threads.append(ct)
                convert_threads = [t for t in convert_threads if t.is_alive()]
            try:
                session.close()
            except Exception:
                pass
            session = requests.Session()
            session.headers.update(HTTP_HEADERS)
            if not continuous:
                break
            time.sleep(restart_delay)
            restart_delay = min(restart_delay * 2, CAMERA_RESTART_MAX_SEC)
            continue
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

    # HTTP 세션 정리 (커넥션 풀 해제)
    try:
        session.close()
    except Exception:
        pass

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


def normalize_camera_config(cameras: list[dict]) -> list[dict]:
    """Accept both legacy name/url and ITS cctv_name/cctvurl camera JSON."""
    normalized = []
    for i, cam in enumerate(cameras, 1):
        item = dict(cam)
        item.setdefault("name", cam.get("cctv_name") or cam.get("slug") or f"camera_{i}")
        item.setdefault("url", cam.get("cctvurl") or cam.get("hls_url") or "")
        item.setdefault("route", cam.get("cctv_road") or cam.get("hotspot_road") or "")
        item.setdefault("x", cam.get("coordx") or cam.get("lon") or cam.get("lng") or 0)
        item.setdefault("y", cam.get("coordy") or cam.get("lat") or 0)
        if not item["url"]:
            raise ValueError(f"camera {i} has no url/cctvurl")
        normalized.append(item)
    return normalized


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

    config_path = args.config or str(DEFAULT_CONFIG)
    if not Path(config_path).exists():
        logger.error("카메라 설정 파일 없음: %s", config_path)
        sys.exit(1)

    with open(config_path) as f:
        cameras = normalize_camera_config(json.load(f))

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
