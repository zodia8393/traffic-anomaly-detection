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
CONFIG = "/tmp/cameras_5routes.json"
REC_LOG = "/DATA/cctv_recording/record_continuous.log"
STATUS = Path("/DATA/cctv_recording/watchdog_status.json")
REC_ROOT = Path("/DATA/cctv_recording")
PROC_PATTERN = "record_hls_multi.py"
DISK_PATH = "/DATA"
DISK_CRIT_GB = 30    # 이 미만이면 위험 (녹화 곧 중단)
DISK_WARN_GB = 100   # 이 미만이면 경고
CAMERA_STALE_SEC = 300   # 활성 .ts가 이 시간 이상 미갱신이면 카메라 비활성 판정
ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "")  # 미설정 시 로그만


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
    cmd = (f"cd {HERE} && PYTHONUNBUFFERED=1 nohup python3 record_hls_multi.py "
           f"--config {CONFIG} --continuous >> {REC_LOG} 2>&1 & echo $!")
    try:
        out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=30)
        pid = out.stdout.strip().split("\n")[-1]
        logger.warning("녹화 재시작됨 PID=%s", pid)
        return int(pid) if pid.isdigit() else None
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
                 stale_cameras: list | None = None) -> None:
    status = {
        "ts": datetime.now().isoformat(),
        "recording": running,
        "pid": pid,
        "disk_free_gb": round(free_gb, 1),
        "disk_state": disk_state,
        "action": action,
        "stale_cameras": stale_cameras or [],
    }
    try:
        STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.warning("상태 기록 실패: %s", e)


def check_cameras() -> list[str]:
    """오늘 날짜 디렉토리의 활성 .ts가 갱신 멈춘 카메라(비활성) 목록 반환."""
    import time
    today = datetime.now().strftime("%Y%m%d")
    day_dir = REC_ROOT / today
    if not day_dir.exists():
        return []
    stale = []
    now = time.time()
    for cam_dir in day_dir.glob("*_hls"):
        ts_files = sorted(cam_dir.glob("*.ts"), key=lambda p: p.stat().st_mtime)
        if not ts_files:
            stale.append(cam_dir.name.replace("_hls", "") + "(파일없음)")
            continue
        latest = ts_files[-1]
        if now - latest.stat().st_mtime > CAMERA_STALE_SEC:
            mins = (now - latest.stat().st_mtime) / 60
            stale.append(f"{cam_dir.name.replace('_hls','')}({mins:.0f}분 미갱신)")
    return stale


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
    write_status(running, pid, free_gb, disk_state, action, stale_cameras=stale)

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
