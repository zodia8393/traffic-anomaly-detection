"""녹화 프로세스 워치독 — 죽으면 자동 재시작 + 디스크/하트비트 감시.

record_hls_multi.py(--continuous)가 OOM/예외로 죽어도 무인 복구한다.
cron 5분 간격 권장 (자체가 데몬이 아니므로 cron이 살아있는 한 복구 보장):
  매 5분: watchdog_recording.py 실행 (cron의 %는 백슬래시 이스케이프 필요)

상태/하트비트: /DATA/cctv_recording/watchdog_status.json
디스크 위험 시 stderr ERROR + status에 기록 (외부 알림 연동 지점).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] watchdog: %(message)s")
logger = logging.getLogger("watchdog")

HERE = Path(__file__).resolve().parent
CONFIG = str(HERE / "cameras_5.json")
REC_LOG = "/DATA/cctv_recording/record_continuous.log"
STATUS = Path("/DATA/cctv_recording/watchdog_status.json")
REC_ROOT = Path("/DATA/cctv_recording")
PAUSE_FILE = REC_ROOT / "pause_until.json"
PROC_PATTERN = "record_hls_multi.py"
DISK_PATH = "/DATA"
DISK_CRIT_GB = 30    # 이 미만이면 위험 (녹화 곧 중단)
DISK_WARN_GB = 100   # 이 미만이면 경고
CAMERA_STALE_SEC = 300   # 활성 .ts가 이 시간 이상 미갱신이면 카메라 비활성 판정
CAMERA_MIN_GOOD_BYTES = int(os.environ.get("CAMERA_MIN_GOOD_BYTES", str(1024 * 1024)))
ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "")  # 미설정 시 로그만


def pause_until() -> datetime | None:
    """Return the pause deadline if recording restart is temporarily disabled."""
    if not PAUSE_FILE.exists():
        return None
    try:
        data = json.loads(PAUSE_FILE.read_text(encoding="utf-8"))
        raw_until = str(data.get("pause_until", "")).strip()
        if not raw_until:
            logger.warning("pause marker has no pause_until: %s", PAUSE_FILE)
            return None
        until = datetime.fromisoformat(raw_until)
    except Exception as e:
        logger.warning("pause marker read failed: %s", e)
        return None

    now = datetime.now(until.tzinfo) if until.tzinfo else datetime.now()
    if now < until:
        return until
    return None


def is_running() -> tuple[bool, int | None]:
    """record_hls_multi.py --continuous 프로세스 생존 여부 + PID."""
    try:
        out = subprocess.run(["pgrep", "-f", f"{PROC_PATTERN} .*--continuous"],
                             capture_output=True, text=True, timeout=10)
        pids = [p for p in out.stdout.strip().split("\n") if p]
        if pids:
            return True, int(pids[0])
    except Exception as e:
        logger.warning("pgrep 실패: %s", e)
    return False, None


def restart_recording() -> int | None:
    """녹화 재시작 (백그라운드 nohup). 새 PID 반환."""
    if not Path(CONFIG).exists():
        logger.error("카메라 설정 없음: %s — 재시작 불가", CONFIG)
        return None
    try:
        log_path = Path(REC_LOG)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        with log_path.open("a", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                [sys.executable or "python3", "record_hls_multi.py", "--config", CONFIG, "--continuous"],
                cwd=str(HERE),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pid = str(proc.pid)
        logger.warning("녹화 재시작됨 PID=%s", pid)
        return proc.pid
    except Exception as e:
        logger.error("녹화 재시작 실패: %s", e)
        return None


def check_disk() -> tuple[float, str]:
    """(가용GB, 상태). 상태: ok|warn|crit."""
    free_gb = shutil.disk_usage(DISK_PATH).free / (1024 ** 3)
    if free_gb < DISK_CRIT_GB:
        logger.error("⚠ 디스크 위험: %.0fGB < %dGB — 녹화 중단 임박", free_gb, DISK_CRIT_GB)
        return free_gb, "crit"
    if free_gb < DISK_WARN_GB:
        logger.warning("디스크 경고: %.0fGB < %dGB", free_gb, DISK_WARN_GB)
        return free_gb, "warn"
    return free_gb, "ok"


def write_status(running: bool, pid, free_gb: float, disk_state: str, action: str,
                 stale_cameras: list | None = None, total_cams: int = 0,
                 active_cams: int = 0, day_bytes: float = 0.0,
                 zero_byte_files: int = 0) -> None:
    status = {
        "ts": datetime.now().isoformat(),
        "recording": running,
        "pid": pid,
        "disk_free_gb": round(free_gb, 1),
        "disk_state": disk_state,
        "action": action,
        "stale_cameras": stale_cameras or [],
        "cameras_total": total_cams,
        "cameras_active": active_cams,
        "day_bytes": int(day_bytes),
        "zero_byte_files": zero_byte_files,
    }
    try:
        STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.warning("상태 기록 실패: %s", e)


def _camera_slug(cam: dict, index: int) -> str:
    name = cam.get("name") or cam.get("cctv_name") or cam.get("slug") or f"camera_{index}"
    return str(name).replace("[", "").replace("]", "").replace(" ", "_")


def _expected_cameras() -> list[str]:
    """녹화 설정 기준 카메라 slug 목록."""
    try:
        with open(CONFIG) as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning("카메라 설정 형식 비정상: %s", CONFIG)
            return []
        return [_camera_slug(cam, i + 1) for i, cam in enumerate(data)]
    except Exception as e:
        logger.warning("카메라 설정 읽기 실패: %s", e)
        return []


def check_cameras() -> list[str]:
    """설정된 카메라 중 오늘 유효 영상 갱신이 멈춘 카메라 목록 반환.

    단순 mtime만 보면 0-byte .ts를 계속 생성하는 장애를 정상으로 오판한다.
    따라서 최신 유효 파일(기본 1MB 이상)이 stale 기준 안에 있어야 활성으로 본다.
    """
    import time
    today = datetime.now().strftime("%Y%m%d")
    day_dir = REC_ROOT / today
    stale = []
    now = time.time()

    expected = _expected_cameras()
    if expected:
        if not day_dir.exists():
            return [f"{slug}(오늘 디렉토리 없음)" for slug in expected]
        cam_dirs = [(slug, day_dir / f"{slug}_hls") for slug in expected]
    else:
        if not day_dir.exists():
            return []
        cam_dirs = [(cam_dir.name.replace("_hls", ""), cam_dir)
                    for cam_dir in day_dir.glob("*_hls")]

    for slug, cam_dir in cam_dirs:
        if not cam_dir.exists():
            stale.append(f"{slug}(디렉토리 없음)")
            continue
        ts_files = sorted(cam_dir.glob("*.ts"), key=lambda p: p.stat().st_mtime)
        media_files = ts_files + sorted(cam_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if not media_files:
            stale.append(f"{slug}(파일없음)")
            continue
        good_files = [f for f in media_files if f.stat().st_size >= CAMERA_MIN_GOOD_BYTES]
        latest_ts = ts_files[-1] if ts_files else None
        zero_note = ", 최신ts 0B" if latest_ts and latest_ts.stat().st_size == 0 else ""
        if not good_files:
            stale.append(f"{slug}(유효파일 없음{zero_note})")
            continue
        latest_good = max(good_files, key=lambda p: p.stat().st_mtime)
        if now - latest_good.stat().st_mtime > CAMERA_STALE_SEC:
            mins = (now - latest_good.stat().st_mtime) / 60
            stale.append(f"{slug}({mins:.0f}분 유효데이터 없음{zero_note})")
    return stale


def _count_active_cameras() -> tuple[int, int, float, int]:
    """(설정 기준 전체 카메라 수, 활성 카메라 수, 오늘 누적 바이트, 0-byte 파일 수)."""
    import time
    today = datetime.now().strftime("%Y%m%d")
    day_dir = REC_ROOT / today
    now = time.time()
    day_bytes = 0
    zero_byte_files = 0

    expected = _expected_cameras()
    if expected:
        total = len(expected)
        active = 0
        if not day_dir.exists():
            return total, active, 0.0, 0
        cam_dirs = [day_dir / f"{slug}_hls" for slug in expected]
    else:
        if not day_dir.exists():
            return 0, 0, 0.0, 0
        total = active = 0
        cam_dirs = list(day_dir.glob("*_hls"))

    for cam_dir in cam_dirs:
        if not expected:
            total += 1
        if not cam_dir.exists():
            continue
        files = list(cam_dir.glob("*.ts")) + list(cam_dir.glob("*.mp4"))
        stats = [(f, f.stat()) for f in files]
        day_bytes += sum(st.st_size for _, st in stats)
        zero_byte_files += sum(1 for _, st in stats if st.st_size == 0)
        good = [f for f, st in stats if st.st_size >= CAMERA_MIN_GOOD_BYTES]
        latest_good = max(good, key=lambda p: p.stat().st_mtime) if good else None
        if latest_good and (now - latest_good.stat().st_mtime) <= CAMERA_STALE_SEC:
            active += 1
    return total, active, float(day_bytes), zero_byte_files


def write_metrics(running: bool, free_gb: float, disk_state: str,
                  total_cams: int, active_cams: int, day_bytes: float,
                  zero_byte_files: int, stale_cameras: list[str]) -> None:
    """node_exporter textfile collector 형식(.prom)으로 메트릭 출력.

    METRICS_DIR(기본 /DATA/cctv_recording/metrics)에 cctv_recording.prom 기록.
    node_exporter --collector.textfile.directory=<dir>로 수집 가능.
    """
    metrics_dir = Path(os.environ.get("METRICS_DIR", str(REC_ROOT / "metrics")))
    disk_code = {"ok": 0, "warn": 1, "crit": 2}.get(disk_state, -1)
    lines = [
        "# HELP cctv_recording_up 녹화 프로세스 가동(1)/중단(0)",
        "# TYPE cctv_recording_up gauge",
        f"cctv_recording_up {1 if running else 0}",
        "# HELP cctv_recording_disk_free_gb /DATA 가용 용량(GB)",
        "# TYPE cctv_recording_disk_free_gb gauge",
        f"cctv_recording_disk_free_gb {free_gb:.1f}",
        "# HELP cctv_recording_disk_state 0=ok 1=warn 2=crit",
        "# TYPE cctv_recording_disk_state gauge",
        f"cctv_recording_disk_state {disk_code}",
        "# HELP cctv_recording_cameras_total 오늘 녹화 카메라 수",
        "# TYPE cctv_recording_cameras_total gauge",
        f"cctv_recording_cameras_total {total_cams}",
        "# HELP cctv_recording_cameras_active 활성(갱신중) 카메라 수",
        "# TYPE cctv_recording_cameras_active gauge",
        f"cctv_recording_cameras_active {active_cams}",
        "# HELP cctv_recording_day_bytes 오늘 누적 녹화 바이트",
        "# TYPE cctv_recording_day_bytes gauge",
        f"cctv_recording_day_bytes {day_bytes:.0f}",
        "# HELP cctv_recording_zero_byte_files_today 오늘 0-byte 영상 파일 수",
        "# TYPE cctv_recording_zero_byte_files_today gauge",
        f"cctv_recording_zero_byte_files_today {zero_byte_files}",
        "# HELP cctv_recording_stale_cameras 비활성/유효데이터 미갱신 카메라 수",
        "# TYPE cctv_recording_stale_cameras gauge",
        f"cctv_recording_stale_cameras {len(stale_cameras)}",
    ]
    try:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        # 원자적 쓰기 (수집 중 부분파일 방지)
        tmp = metrics_dir / "cctv_recording.prom.tmp"
        tmp.write_text("\n".join(lines) + "\n")
        tmp.rename(metrics_dir / "cctv_recording.prom")
    except Exception as e:
        logger.warning("메트릭 기록 실패: %s", e)


def send_alert(message: str) -> None:
    """경고 알림 — webhook 설정 시 POST, 항상 로그."""
    logger.error("🔔 ALERT: %s", message)
    if not ALERT_WEBHOOK_URL:
        return
    try:
        import requests
        requests.post(ALERT_WEBHOOK_URL, json={"text": f"[CCTV 녹화] {message}"}, timeout=10)
    except Exception as e:
        logger.warning("webhook 알림 실패: %s", e)


def systemd_manages() -> bool:
    """systemd cctv-recording 서비스가 관리 중인지 (재시작 충돌 방지)."""
    try:
        out = subprocess.run(["systemctl", "is-active", "cctv-recording"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() in ("active", "activating")
    except Exception:
        return False


def main():
    paused_until = pause_until()
    if paused_until:
        running, pid = is_running()
        free_gb, disk_state = check_disk()
        total_cams, active_cams, day_bytes, zero_byte_files = _count_active_cameras()
        action = f"paused_until_{paused_until.isoformat()}"
        logger.warning("녹화 재시작 일시중지: %s (running=%s pid=%s)", paused_until.isoformat(), running, pid)
        write_status(
            running, pid, free_gb, disk_state, action, stale_cameras=[],
            total_cams=total_cams, active_cams=active_cams,
            day_bytes=day_bytes, zero_byte_files=zero_byte_files,
        )
        write_metrics(
            running, free_gb, disk_state, total_cams, active_cams,
            day_bytes, zero_byte_files, [],
        )
        return

    running, pid = is_running()
    action = "none"
    if not running:
        if systemd_manages():
            # systemd가 30초 내 자동 재시작 — 워치독은 중복 기동 금지(모니터링만)
            logger.warning("녹화 프로세스 일시 부재 — systemd 재시작 대기(중복 기동 안 함)")
            action = "systemd_managed"
        else:
            logger.error("녹화 프로세스 없음 — 재시작 시도")
            pid = restart_recording()
            action = "restarted" if pid else "restart_failed"
            running = pid is not None
    else:
        logger.info("녹화 정상 가동 PID=%s", pid)

    free_gb, disk_state = check_disk()
    stale = check_cameras()
    total_cams, active_cams, day_bytes, zero_byte_files = _count_active_cameras()
    write_status(
        running, pid, free_gb, disk_state, action, stale_cameras=stale,
        total_cams=total_cams, active_cams=active_cams,
        day_bytes=day_bytes, zero_byte_files=zero_byte_files,
    )
    write_metrics(
        running, free_gb, disk_state, total_cams, active_cams,
        day_bytes, zero_byte_files, stale,
    )

    # 알림 조건: 재시작 실패 / 디스크 위험 / 비활성 카메라
    if action == "restart_failed":
        send_alert("녹화 프로세스 재시작 실패 — 수동 확인 필요")
    if disk_state == "crit":
        send_alert(f"디스크 위험: {free_gb:.0f}GB 남음 — 녹화 중단 임박")
    if stale:
        send_alert(f"비활성 카메라 {len(stale)}대: {', '.join(stale)}")

    if action == "restart_failed" or disk_state == "crit":
        sys.exit(2)


if __name__ == "__main__":
    main()
