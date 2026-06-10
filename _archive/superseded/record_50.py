"""고속도로 사고다발 CCTV 50대 일괄 녹화.

ITS API로 전국 고속도로 CCTV 4,760대를 조회하고,
사고다발 구간 50개 좌표에 최근접 CCTV를 매칭하여 일괄 녹화한다.

실행:
  # 1단계: CCTV 50대 선정
  python record_50.py --select-cameras

  # 2단계: 녹화 시작 (24시간)
  nohup python record_50.py --record --duration 24h \
      --output /DATA/cctv_recording/ \
      > /DATA/cctv_recording/record_50.log 2>&1 &

  # 3단계: 모니터링
  python record_50.py --status

  # 4단계: 영상 통합
  python record_50.py --merge --date 20260528

저장: /DATA/cctv_recording/{날짜}/{CCTV명}/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
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

# ── 설정 ──────────────────────────────────────────────────────────────
ITS_API_KEY = os.getenv("ITS_API_KEY", "")
ITS_CCTV_URL = "https://openapi.its.go.kr:9443/cctvInfo"
OUTPUT_ROOT = Path("/DATA/cctv_recording")
CAMERAS_JSON = Path(__file__).parent / "cameras_50.json"
URL_REFRESH_SEC = 90 * 60  # 90분
DISK_CHECK_SEC = 600  # 10분
DISK_WARN_GB = 50
DISK_STOP_GB = 20
STALL_CHECK_SEC = 300  # 5분간 파일 크기 변화 없으면 재시작
API_CALL_LOG: list[str] = []

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("record_50")

# ── 사고다발 구간 50개 좌표 ──────────────────────────────────────────
# 교통공학적으로 알려진 고속도로 사고다발 구간 (전국 분산, 최소 이격 ~20km)
ACCIDENT_HOTSPOTS: list[dict] = [
    # 경부선 (8대)
    {"name": "경부선_양재", "lat": 37.467, "lon": 127.035, "road": "경부선"},
    {"name": "경부선_판교", "lat": 37.380, "lon": 127.100, "road": "경부선"},
    {"name": "경부선_기흥", "lat": 37.240, "lon": 127.080, "road": "경부선"},
    {"name": "경부선_안성", "lat": 36.950, "lon": 127.050, "road": "경부선"},
    {"name": "경부선_천안", "lat": 36.800, "lon": 127.100, "road": "경부선"},
    {"name": "경부선_회덕JC", "lat": 36.380, "lon": 127.430, "road": "경부선"},
    {"name": "경부선_김천", "lat": 36.100, "lon": 128.100, "road": "경부선"},
    {"name": "경부선_경주", "lat": 35.830, "lon": 129.180, "road": "경부선"},
    # 중부선 (4대)
    {"name": "중부선_하남JC", "lat": 37.530, "lon": 127.190, "road": "중부선"},
    {"name": "중부선_이천", "lat": 37.250, "lon": 127.430, "road": "중부선"},
    {"name": "중부선_음성", "lat": 36.950, "lon": 127.700, "road": "중부선"},
    {"name": "중부선_충주", "lat": 36.950, "lon": 127.920, "road": "중부선"},
    # 영동선 (5대)
    {"name": "영동선_여주JC", "lat": 37.310, "lon": 127.550, "road": "영동선"},
    {"name": "영동선_원주", "lat": 37.340, "lon": 127.900, "road": "영동선"},
    {"name": "영동선_횡성", "lat": 37.440, "lon": 128.100, "road": "영동선"},
    {"name": "영동선_대관령", "lat": 37.690, "lon": 128.760, "road": "영동선"},
    {"name": "영동선_강릉", "lat": 37.750, "lon": 128.880, "road": "영동선"},
    # 서해안선 (5대)
    {"name": "서해안선_서평택", "lat": 36.950, "lon": 126.850, "road": "서해안선"},
    {"name": "서해안선_안중", "lat": 36.900, "lon": 126.900, "road": "서해안선"},
    {"name": "서해안선_서산", "lat": 36.750, "lon": 126.550, "road": "서해안선"},
    {"name": "서해안선_서천", "lat": 36.050, "lon": 126.700, "road": "서해안선"},
    {"name": "서해안선_목포", "lat": 34.800, "lon": 126.450, "road": "서해안선"},
    # 호남선 (4대)
    {"name": "호남선_논산JC", "lat": 36.150, "lon": 127.050, "road": "호남선"},
    {"name": "호남선_익산", "lat": 35.950, "lon": 126.950, "road": "호남선"},
    {"name": "호남선_정읍", "lat": 35.550, "lon": 126.850, "road": "호남선"},
    {"name": "호남선_광주", "lat": 35.150, "lon": 126.900, "road": "호남선"},
    # 중앙선 (4대)
    {"name": "중앙선_춘천", "lat": 37.850, "lon": 127.750, "road": "중앙선"},
    {"name": "중앙선_원주", "lat": 37.350, "lon": 127.950, "road": "중앙선"},
    {"name": "중앙선_안동", "lat": 36.550, "lon": 128.700, "road": "중앙선"},
    {"name": "중앙선_영주", "lat": 36.800, "lon": 128.650, "road": "중앙선"},
    # 남해선 (4대)
    {"name": "남해선_부산", "lat": 35.150, "lon": 129.050, "road": "남해선"},
    {"name": "남해선_김해", "lat": 35.230, "lon": 128.900, "road": "남해선"},
    {"name": "남해선_진주", "lat": 35.150, "lon": 128.100, "road": "남해선"},
    {"name": "남해선_사천", "lat": 35.050, "lon": 128.050, "road": "남해선"},
    # 제2경인 (2대)
    {"name": "제2경인_인천", "lat": 37.420, "lon": 126.750, "road": "제2경인고속도로"},
    {"name": "제2경인_안양", "lat": 37.400, "lon": 126.900, "road": "제2경인고속도로"},
    # 제2중부 (2대)
    {"name": "제2중부_마장JC", "lat": 37.200, "lon": 127.250, "road": "평택제천선"},
    {"name": "제2중부_음성", "lat": 36.900, "lon": 127.500, "road": "평택제천선"},
    # 중부내륙 (2대)
    {"name": "중부내륙_여주", "lat": 37.200, "lon": 127.600, "road": "중부내륙선"},
    {"name": "중부내륙_충주", "lat": 36.970, "lon": 127.900, "road": "중부내륙선"},
    # 수도권제1순환 (3대)
    {"name": "수도권1순환_판교", "lat": 37.390, "lon": 127.100, "road": "수도권제1순환선"},
    {"name": "수도권1순환_성남", "lat": 37.430, "lon": 127.150, "road": "수도권제1순환선"},
    {"name": "수도권1순환_하남", "lat": 37.530, "lon": 127.200, "road": "수도권제1순환선"},
    # 제2순환 (2대)
    {"name": "수도권2순환_오산", "lat": 37.130, "lon": 127.050, "road": "수도권제2순환선"},
    {"name": "수도권2순환_화성", "lat": 37.200, "lon": 126.900, "road": "수도권제2순환선"},
    # 동해선 (2대)
    {"name": "동해선_울산", "lat": 35.550, "lon": 129.300, "road": "동해선"},
    {"name": "동해선_강릉", "lat": 37.700, "lon": 128.950, "road": "동해선"},
    # 기타 (3대)
    {"name": "광주대구선_거창", "lat": 35.700, "lon": 127.900, "road": "광주대구선"},
    {"name": "익산장수선_장수", "lat": 35.700, "lon": 127.500, "road": "새만금포항선"},
    {"name": "세종포천선_양주", "lat": 37.800, "lon": 127.050, "road": "세종포천선"},
]

assert len(ACCIDENT_HOTSPOTS) == 50, f"사고다발 구간이 50개가 아님: {len(ACCIDENT_HOTSPOTS)}"


# ── API ──────────────────────────────────────────────────────────────
def fetch_all_cctvs(cctv_type: str = "4") -> list[dict]:
    """전국 고속도로 CCTV 전체 조회 (1회 API 호출).

    cctvType=4 (HTTPS HLS) 우선, 실패 시 cctvType=1 (HTTP HLS) 폴백.
    """
    params = {
        "apiKey": ITS_API_KEY,
        "type": "ex",
        "cctvType": cctv_type,
        "minX": "124", "maxX": "132",
        "minY": "33", "maxY": "39",
        "getType": "json",
    }
    try:
        resp = requests.get(ITS_CCTV_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("API 호출 실패 (cctvType=%s): %s", cctv_type, e)
        if cctv_type == "4":
            logger.info("cctvType=1 (HTTP HLS) 폴백 시도...")
            return fetch_all_cctvs("1")
        return []

    # API 제한 체크
    header = data.get("header", data.get("response", {}).get("header", {}))
    if isinstance(header, dict) and header.get("resultCode") == 4001:
        logger.error("API 일일 제한 초과 (4001)")
        return []

    items = data.get("response", {}).get("data", [])
    ts = datetime.now().isoformat()
    API_CALL_LOG.append(f"{ts}: cctvType={cctv_type}, count={len(items)}")
    logger.info("API 조회 완료: %d대 CCTV (cctvType=%s)", len(items), cctv_type)
    return items


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 좌표 간 거리 (km)."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _extract_road(cctvname: str) -> str:
    """[노선명] 에서 노선명 추출."""
    if "[" in cctvname and "]" in cctvname:
        return cctvname.split("]")[0].split("[")[1]
    return ""


def slugify(name: str) -> str:
    """파일/디렉토리명 안전 문자열 변환."""
    return (name.replace("[", "").replace("]", "")
            .replace(" ", "_").replace("/", "_")
            .replace("(", "").replace(")", ""))


# ── Phase 1: CCTV 50대 선정 ─────────────────────────────────────────
def select_cameras(output_path: Path | None = None) -> list[dict]:
    """사고다발 50개 좌표에 최근접 CCTV 매칭."""
    all_cctvs = fetch_all_cctvs()
    if not all_cctvs:
        logger.error("CCTV 데이터 조회 실패")
        sys.exit(1)

    # 좌표 파싱
    for c in all_cctvs:
        c["_lat"] = float(c.get("coordy", 0))
        c["_lon"] = float(c.get("coordx", 0))

    selected: list[dict] = []
    used_indices: set[int] = set()

    for hotspot in ACCIDENT_HOTSPOTS:
        best_idx = -1
        best_dist = float("inf")

        for i, c in enumerate(all_cctvs):
            if i in used_indices:
                continue
            dist = _haversine_km(
                hotspot["lat"], hotspot["lon"],
                c["_lat"], c["_lon"],
            )
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx < 0:
            logger.warning("매칭 실패: %s", hotspot["name"])
            continue

        cctv = all_cctvs[best_idx]
        used_indices.add(best_idx)

        entry = {
            "idx": len(selected) + 1,
            "hotspot_name": hotspot["name"],
            "hotspot_road": hotspot["road"],
            "cctv_name": cctv.get("cctvname", ""),
            "cctv_road": _extract_road(cctv.get("cctvname", "")),
            "coordx": cctv.get("coordx"),
            "coordy": cctv.get("coordy"),
            "distance_km": round(best_dist, 2),
            "cctvurl": cctv.get("cctvurl", ""),
            "cctvtype": cctv.get("cctvtype", ""),
            "cctvformat": cctv.get("cctvformat", ""),
            "slug": slugify(hotspot["name"]),
        }
        selected.append(entry)

    # 결과 출력
    roads = set(e["cctv_road"] for e in selected)
    logger.info("=== CCTV 선정 결과 ===")
    logger.info("선정: %d대 / 목표: 50대", len(selected))
    logger.info("노선 수: %d개 (%s)", len(roads), ", ".join(sorted(roads)))
    logger.info("평균 매칭 거리: %.1fkm", sum(e["distance_km"] for e in selected) / len(selected))
    logger.info("최대 매칭 거리: %.1fkm", max(e["distance_km"] for e in selected))

    for e in selected:
        logger.info("  %2d. %-20s → %-30s (%.1fkm)",
                     e["idx"], e["hotspot_name"], e["cctv_name"], e["distance_km"])

    # 저장
    save_path = output_path or CAMERAS_JSON
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2))
    logger.info("저장: %s", save_path)

    return selected


# ── Phase 2: 일괄 녹화 ──────────────────────────────────────────────
class CameraRecorder:
    """개별 CCTV ffmpeg 녹화 관리."""

    def __init__(self, cam: dict, out_root: Path, date_str: str):
        self.cam = cam
        self.slug = cam["slug"]
        self.out_dir = out_root / date_str / self.slug
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.url = cam["cctvurl"]
        self.process: asyncio.subprocess.Process | None = None
        self.current_file: Path | None = None
        self.last_size: int = 0
        self.last_size_check: float = 0
        self.start_time: float = 0
        self.restart_count: int = 0
        self.gap_count: int = 0
        self.total_gap_sec: float = 0
        self.last_stop: float = 0
        self._logger = logging.getLogger(f"cam.{self.slug}")

    async def start(self):
        """ffmpeg 프로세스 시작."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outfile = self.out_dir / f"{self.slug}_{ts}.mp4"
        self.current_file = outfile

        cmd = [
            "ffmpeg", "-y",
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "60",
            "-reconnect_on_network_error", "1",
            "-rw_timeout", "30000000",  # 30s read/write timeout (microseconds)
            "-i", self.url,
            "-c", "copy",
            "-movflags", "+faststart",
            str(outfile),
        ]

        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        self.start_time = time.time()
        self.last_size = 0
        self.last_size_check = time.time()

        if self.last_stop > 0:
            gap = time.time() - self.last_stop
            self.gap_count += 1
            self.total_gap_sec += gap

        self._logger.info("ffmpeg 시작: PID=%d → %s", self.process.pid, outfile.name)

    async def stop(self):
        """ffmpeg 프로세스 정지."""
        if self.process and self.process.returncode is None:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=15)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            self.last_stop = time.time()
            self._logger.info("ffmpeg 종료")

    async def restart(self, reason: str = ""):
        """ffmpeg 재시작 (새 파일). 재시작 간 최소 5초 쿨다운."""
        # 재시작 폭주 방지: 최소 5초 간격
        since_start = time.time() - self.start_time if self.start_time else 999
        if since_start < 5:
            return  # 5초 미만에 재시작 요청 → 무시 (아직 연결 중)
        self._logger.warning("재시작: %s", reason)
        await self.stop()
        self.restart_count += 1
        # 연속 재시작 시 백오프: 3회 이상 연속이면 30초 대기
        wait = 5 if self.restart_count < 3 else min(30, 5 * self.restart_count)
        await asyncio.sleep(wait)
        await self.start()

    def is_alive(self) -> bool:
        return self.process is not None and self.process.returncode is None

    def check_stall(self) -> bool:
        """파일 크기 정체 감지. True = 정체."""
        if not self.current_file or not self.current_file.exists():
            return False
        now = time.time()
        if now - self.last_size_check < STALL_CHECK_SEC:
            return False
        current_size = self.current_file.stat().st_size
        stalled = (current_size == self.last_size and self.last_size > 0)
        self.last_size = current_size
        self.last_size_check = now
        return stalled

    def update_url(self, new_url: str):
        """URL 갱신 (다음 재시작 시 적용)."""
        if new_url and new_url != self.url:
            self.url = new_url
            self._logger.debug("URL 갱신 예약")

    def get_stats(self) -> dict:
        """현재 녹화 통계."""
        files = sorted(self.out_dir.glob(f"{self.slug}_*.mp4"))
        total_size = sum(f.stat().st_size for f in files if f.exists())
        return {
            "name": self.cam["cctv_name"],
            "slug": self.slug,
            "road": self.cam.get("cctv_road", ""),
            "alive": self.is_alive(),
            "files": len(files),
            "total_size_gb": round(total_size / (1024 ** 3), 3),
            "restart_count": self.restart_count,
            "gap_count": self.gap_count,
            "total_gap_sec": round(self.total_gap_sec),
        }


class BatchRecorder:
    """50대 CCTV 일괄 녹화 관리자."""

    def __init__(self, cameras: list[dict], output_root: Path, duration_hours: float):
        self.cameras = cameras
        self.output_root = output_root
        self.duration_hours = duration_hours
        self.date_str = datetime.now().strftime("%Y%m%d")
        self.recorders: list[CameraRecorder] = []
        self._stop_event = asyncio.Event()
        self._api_calls = 0
        self._start_time: float = 0

    def _setup_logging(self):
        """파일 로그 설정."""
        log_dir = self.output_root / self.date_str
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_dir / "batch_record.log"), encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
        logging.getLogger().addHandler(fh)

    async def run(self):
        """메인 녹화 루프."""
        self._setup_logging()
        self._start_time = time.time()

        # 시그널 핸들링
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: self._stop_event.set())

        # 레코더 생성
        self.recorders = [
            CameraRecorder(cam, self.output_root, self.date_str)
            for cam in self.cameras
        ]

        # 초기 디스크 체크
        if not self._check_disk():
            logger.error("디스크 공간 부족, 녹화 시작 불가")
            return

        # 녹화 전 URL 갱신 (cameras_50.json URL은 만료됐을 수 있음)
        logger.info("녹화 전 URL 갱신 중 (API 1회)...")
        await self._refresh_urls()

        # 전체 시작 (5대씩 배치, HLS 연결 안정화 대기)
        logger.info("=== 녹화 시작: %d대, %.1f시간 ===", len(self.recorders), self.duration_hours)
        batch_size = 5
        for i in range(0, len(self.recorders), batch_size):
            batch = self.recorders[i:i + batch_size]
            await asyncio.gather(*(r.start() for r in batch))
            logger.info("  배치 %d/%d 시작 완료 (%d대)",
                         i // batch_size + 1,
                         math.ceil(len(self.recorders) / batch_size),
                         len(batch))
            if i + batch_size < len(self.recorders):
                await asyncio.sleep(5)  # HLS 연결 안정화 대기

        # 초기 안정화 대기 (15초) — ffmpeg HLS 연결 확립 시간
        logger.info("초기 안정화 대기 (15초)...")
        await asyncio.sleep(15)
        alive = sum(1 for r in self.recorders if r.is_alive())
        logger.info("초기 시작 완료: %d/%d 활성", alive, len(self.recorders))

        # 메인 감시 루프
        try:
            await self._watch_loop()
        finally:
            await self._shutdown()

    async def _watch_loop(self):
        """watchdog + URL 갱신 + 디스크 감시 통합 루프."""
        last_url_refresh = time.time()
        last_disk_check = time.time()
        last_status_log = time.time()

        end_time = self._start_time + self.duration_hours * 3600

        while not self._stop_event.is_set():
            now = time.time()

            # 녹화 시간 종료 체크
            if now >= end_time:
                logger.info("녹화 시간 %.1f시간 완료", self.duration_hours)
                break

            # --- watchdog: 프로세스 사망/정체 체크 (병렬 재시작) ---
            restart_tasks = []
            for r in self.recorders:
                if not r.is_alive():
                    restart_tasks.append(r.restart("프로세스 종료됨"))
                elif r.check_stall():
                    restart_tasks.append(r.restart("파일 크기 5분 정체"))
            if restart_tasks:
                await asyncio.gather(*restart_tasks)

            # --- URL 갱신 (90분 주기) ---
            if now - last_url_refresh > URL_REFRESH_SEC:
                last_url_refresh = now
                await self._refresh_urls()

            # --- 디스크 감시 (10분 주기) ---
            if now - last_disk_check > DISK_CHECK_SEC:
                last_disk_check = now
                if not self._check_disk():
                    logger.error("디스크 부족으로 녹화 중단")
                    break

            # --- 상태 로그 (30분 주기) ---
            if now - last_status_log > 1800:
                last_status_log = now
                self._log_status()

            # 5초 간격 체크 (stop 이벤트 대기)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=5)
                break  # stop event set
            except asyncio.TimeoutError:
                pass

    async def _refresh_urls(self):
        """전국 bbox 1회 호출로 50대 URL 일괄 갱신."""
        logger.info("URL 갱신 시작 (API 1회 호출)...")
        all_cctvs = await asyncio.get_event_loop().run_in_executor(
            None, fetch_all_cctvs
        )
        self._api_calls += 1

        if not all_cctvs:
            logger.warning("URL 갱신 실패, 기존 URL 유지")
            return

        # name 기반 매칭으로 URL 갱신
        cctv_by_name: dict[str, str] = {}
        for c in all_cctvs:
            name = c.get("cctvname", "")
            url = c.get("cctvurl", "")
            if name and url:
                cctv_by_name[name] = url

        updated = 0
        for r in self.recorders:
            cctv_name = r.cam["cctv_name"]
            new_url = cctv_by_name.get(cctv_name)
            if new_url:
                r.update_url(new_url)
                updated += 1

        logger.info("URL 갱신: %d/%d 업데이트, 총 API 호출 %d회",
                     updated, len(self.recorders), self._api_calls)

        # URL 변경된 카메라는 재시작하여 새 URL 적용
        for r in self.recorders:
            if r.url != r.cam.get("_last_url", r.url):
                await r.restart("URL 갱신 적용")
            r.cam["_last_url"] = r.url

    def _check_disk(self) -> bool:
        """디스크 여유 공간 확인."""
        try:
            usage = shutil.disk_usage(self.output_root)
        except OSError:
            return True
        free_gb = usage.free / (1024 ** 3)
        if free_gb < DISK_STOP_GB:
            logger.error("디스크 위험: %.1fGB (중단 기준 %dGB)", free_gb, DISK_STOP_GB)
            return False
        if free_gb < DISK_WARN_GB:
            logger.warning("디스크 경고: %.1fGB (경고 기준 %dGB)", free_gb, DISK_WARN_GB)
        return True

    def _log_status(self):
        """현재 녹화 상태 로깅."""
        alive = sum(1 for r in self.recorders if r.is_alive())
        elapsed = (time.time() - self._start_time) / 3600
        total_gb = sum(r.get_stats()["total_size_gb"] for r in self.recorders)
        logger.info("=== 상태: %.1f시간 경과 | %d/%d 활성 | %.1fGB 녹화 | API %d회 ===",
                     elapsed, alive, len(self.recorders), total_gb, self._api_calls)

    async def _shutdown(self):
        """전체 종료 + 리포트 생성."""
        logger.info("=== 종료 시작 ===")

        # 전체 ffmpeg 종료
        await asyncio.gather(*(r.stop() for r in self.recorders))

        # 녹화 리포트 생성
        report = self._generate_report()
        report_path = self.output_root / f"recording_report_{self.date_str}.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        logger.info("녹화 리포트: %s", report_path)

        # 요약
        summary = report["summary"]
        logger.info("=== 녹화 완료 ===")
        logger.info("  총 크기: %.1fGB", summary["total_size_gb"])
        logger.info("  평균 가동률: %.1f%%", summary["avg_uptime_pct"])
        logger.info("  끊김 있는 카메라: %d대", summary["cameras_with_gaps"])
        logger.info("  API 호출: %d회", summary["api_calls"])

    def _generate_report(self) -> dict:
        """녹화 리포트 JSON 생성."""
        elapsed = time.time() - self._start_time
        cam_reports = []

        for r in self.recorders:
            stats = r.get_stats()
            # uptime 계산 (총 시간 - 갭 시간)
            uptime_pct = max(0, min(100,
                (1 - stats["total_gap_sec"] / max(elapsed, 1)) * 100
            ))
            cam_reports.append({
                "name": stats["name"],
                "road": stats["road"],
                "slug": stats["slug"],
                "files": stats["files"],
                "total_size_gb": stats["total_size_gb"],
                "total_duration_sec": round(elapsed - stats["total_gap_sec"]),
                "gaps_count": stats["gap_count"],
                "gaps_total_sec": stats["total_gap_sec"],
                "restart_count": stats["restart_count"],
                "uptime_pct": round(uptime_pct, 1),
            })

        total_size = sum(c["total_size_gb"] for c in cam_reports)
        avg_uptime = sum(c["uptime_pct"] for c in cam_reports) / max(len(cam_reports), 1)
        cameras_with_gaps = sum(1 for c in cam_reports if c["gaps_count"] > 0)

        return {
            "date": self.date_str,
            "total_cameras": len(cam_reports),
            "duration_hours": round(elapsed / 3600, 2),
            "cameras": cam_reports,
            "summary": {
                "total_size_gb": round(total_size, 1),
                "avg_uptime_pct": round(avg_uptime, 1),
                "cameras_with_gaps": cameras_with_gaps,
                "api_calls": self._api_calls,
                "api_log": API_CALL_LOG.copy(),
            },
        }


# ── Phase 3: 모니터링 ───────────────────────────────────────────────
def show_status(output_root: Path, date_str: str | None = None):
    """녹화 상태 출력."""
    date = date_str or datetime.now().strftime("%Y%m%d")
    rec_dir = output_root / date

    if not rec_dir.exists():
        print(f"녹화 디렉토리 없음: {rec_dir}")
        return

    print(f"\n=== 녹화 상태: {date} ===\n")
    total_size = 0
    total_files = 0

    cam_dirs = sorted(rec_dir.iterdir())
    for d in cam_dirs:
        if not d.is_dir():
            continue
        files = list(d.glob("*.mp4"))
        size = sum(f.stat().st_size for f in files)
        total_size += size
        total_files += len(files)

        # 활성 여부: 최근 파일이 30초 내 수정됐는지
        active = ""
        if files:
            latest = max(files, key=lambda f: f.stat().st_mtime)
            if time.time() - latest.stat().st_mtime < 30:
                active = " [RECORDING]"

        print(f"  {d.name:<35} {len(files):>3}파일  {size / (1024**3):>7.2f}GB{active}")

    print(f"\n  {'합계':<35} {total_files:>3}파일  {total_size / (1024**3):>7.2f}GB")
    print()

    # 디스크 상태
    usage = shutil.disk_usage(output_root)
    print(f"  디스크: {usage.free / (1024**3):.1f}GB 여유 / {usage.total / (1024**3):.1f}GB 전체")


# ── Phase 4: 영상 통합 ──────────────────────────────────────────────
def merge_daily(output_root: Path, date_str: str):
    """CCTV별 파편 → ffmpeg concat → _full.mp4 병합."""
    rec_dir = output_root / date_str

    if not rec_dir.exists():
        logger.error("녹화 디렉토리 없음: %s", rec_dir)
        return

    merge_results: list[dict] = []

    for cam_dir in sorted(rec_dir.iterdir()):
        if not cam_dir.is_dir():
            continue

        fragments = sorted(cam_dir.glob("*.mp4"))
        # _full.mp4 와 fragments/ 내 파일 제외
        fragments = [f for f in fragments
                     if "_full.mp4" not in f.name
                     and f.parent.name != "fragments"]

        if not fragments:
            logger.info("  %s: 파편 없음, 건너뜀", cam_dir.name)
            continue

        slug = cam_dir.name
        output_file = cam_dir / f"{slug}_{date_str}_full.mp4"

        if len(fragments) == 1:
            # 파편이 1개면 rename
            fragments[0].rename(output_file)
            logger.info("  %s: 1개 파편 → rename", slug)
            result = _probe_file(output_file)
            result["slug"] = slug
            result["fragments"] = 1
            merge_results.append(result)
            continue

        # concat demuxer 목록 생성
        concat_txt = cam_dir / "concat.txt"
        with open(concat_txt, "w") as f:
            for frag in fragments:
                f.write(f"file '{frag.name}'\n")

        # ffmpeg concat (재인코딩 없음)
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_txt),
            "-c", "copy",
            "-movflags", "+faststart",
            str(output_file),
        ]
        logger.info("  %s: %d개 파편 병합 중...", slug, len(fragments))

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        concat_txt.unlink(missing_ok=True)

        if proc.returncode != 0:
            logger.error("  %s: 병합 실패\n%s", slug, proc.stderr[-500:])
            merge_results.append({
                "slug": slug, "success": False,
                "error": proc.stderr[-200:], "fragments": len(fragments),
            })
            continue

        # 원본 파편 → fragments/ 이동
        frag_dir = cam_dir / "fragments"
        frag_dir.mkdir(exist_ok=True)
        for frag in fragments:
            frag.rename(frag_dir / frag.name)

        # 무결성 검증
        result = _probe_file(output_file)
        result["slug"] = slug
        result["fragments"] = len(fragments)
        merge_results.append(result)
        logger.info("  %s: 병합 완료 (%.1fs, %.2fGB)",
                     slug, result.get("duration_sec", 0), result.get("size_gb", 0))

    # 병합 리포트
    report = {
        "date": date_str,
        "merge_time": datetime.now().isoformat(),
        "total_cameras": len(merge_results),
        "successful": sum(1 for r in merge_results if r.get("success", True)),
        "cameras": merge_results,
    }
    report_path = rec_dir / f"merge_report_{date_str}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    logger.info("병합 리포트: %s", report_path)

    # 요약
    ok = [r for r in merge_results if r.get("success", True)]
    if ok:
        total_dur = sum(r.get("duration_sec", 0) for r in ok)
        total_size = sum(r.get("size_gb", 0) for r in ok)
        logger.info("=== 병합 완료: %d/%d 성공, 총 %.1f시간, %.1fGB ===",
                     len(ok), len(merge_results), total_dur / 3600, total_size)


def _probe_file(filepath: Path) -> dict:
    """ffprobe로 MP4 파일 검증."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(filepath),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return {"success": False, "error": "ffprobe failed"}
        info = json.loads(result.stdout)
        fmt = info.get("format", {})
        return {
            "success": True,
            "duration_sec": round(float(fmt.get("duration", 0)), 1),
            "size_gb": round(int(fmt.get("size", 0)) / (1024 ** 3), 3),
            "format": fmt.get("format_long_name", ""),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── CLI ──────────────────────────────────────────────────────────────
def parse_duration(s: str) -> float:
    """'24h', '12h', '1h', '30m' → hours."""
    s = s.strip().lower()
    if s.endswith("h"):
        return float(s[:-1])
    if s.endswith("m"):
        return float(s[:-1]) / 60
    return float(s)


def main():
    parser = argparse.ArgumentParser(
        description="고속도로 사고다발 CCTV 50대 일괄 녹화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python record_50.py --select-cameras
  python record_50.py --record --duration 24h
  python record_50.py --status
  python record_50.py --merge --date 20260528
        """,
    )
    parser.add_argument("--select-cameras", action="store_true",
                        help="사고다발 50대 CCTV 선정")
    parser.add_argument("--record", action="store_true",
                        help="일괄 녹화 시작")
    parser.add_argument("--status", action="store_true",
                        help="녹화 상태 확인")
    parser.add_argument("--merge", action="store_true",
                        help="영상 통합 (파편 → _full.mp4)")
    parser.add_argument("--duration", type=str, default="24h",
                        help="녹화 시간 (기본: 24h)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_ROOT),
                        help="출력 디렉토리 (기본: /DATA/cctv_recording)")
    parser.add_argument("--date", type=str, default=None,
                        help="대상 날짜 (YYYYMMDD, --status/--merge 시)")
    parser.add_argument("--cameras-json", type=str, default=None,
                        help="cameras_50.json 경로 (기본: 같은 디렉토리)")

    args = parser.parse_args()
    output_root = Path(args.output)

    if not ITS_API_KEY:
        logger.error("ITS_API_KEY 미설정 (/workspace/.env)")
        sys.exit(1)

    if args.select_cameras:
        cam_path = Path(args.cameras_json) if args.cameras_json else None
        select_cameras(cam_path)

    elif args.record:
        # cameras_50.json 로드
        cam_json = Path(args.cameras_json) if args.cameras_json else CAMERAS_JSON
        if not cam_json.exists():
            logger.error("cameras_50.json 없음: %s (먼저 --select-cameras 실행)", cam_json)
            sys.exit(1)

        cameras = json.loads(cam_json.read_text())
        duration = parse_duration(args.duration)
        logger.info("녹화 설정: %d대, %.1f시간, 출력=%s", len(cameras), duration, output_root)

        recorder = BatchRecorder(cameras, output_root, duration)
        asyncio.run(recorder.run())

    elif args.status:
        show_status(output_root, args.date)

    elif args.merge:
        date = args.date or datetime.now().strftime("%Y%m%d")
        merge_daily(output_root, date)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
