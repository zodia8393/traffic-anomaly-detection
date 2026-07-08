#!/usr/bin/env python3
"""Generate a daily CCTV recording inventory report.

The OpenClaw daily status job writes this report for human review and for
cron-watchdog's daily completion check. It is read-only against recording data.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
CONFIG = HERE / "cameras_5.json"
REC_ROOT = Path("/DATA/cctv_recording")
PROJECT_LOG_DIR = PROJECT_ROOT / "logs"
INFRA_LOG_DIR = Path("/workspace/infra/codex/scripts/logs")
STALE_SECONDS = 15 * 60
VALID_STALE_SECONDS = 30 * 60


@dataclass(frozen=True)
class CameraReport:
    status: str
    camera: str
    bytes_total: int
    files_total: int
    mp4_count: int
    ts_count: int
    zero_count: int
    latest: str
    latest_valid: str
    note: str


def camera_slug(cam: dict, index: int) -> str:
    name = cam.get("name") or cam.get("cctv_name") or cam.get("slug") or f"camera_{index}"
    return str(name).replace("[", "").replace("]", "").replace(" ", "_")


def expected_cameras() -> list[str]:
    try:
        data = json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [camera_slug(cam, i + 1) for i, cam in enumerate(data) if isinstance(cam, dict)]


def format_time(path: Path | None) -> str:
    if path is None:
        return "-"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%H:%M:%S")


def status_for(latest: Path | None, latest_valid: Path | None, now_ts: float) -> tuple[str, str]:
    if latest is None:
        return "문제", "파일 없음"
    latest_age = now_ts - latest.stat().st_mtime
    if latest_age > STALE_SECONDS:
        return "문제", f"최신파일 {latest_age / 60:.0f}분 미갱신"
    if latest_valid is None:
        return "주의", "유효 파일 없음"
    valid_age = now_ts - latest_valid.stat().st_mtime
    if valid_age > VALID_STALE_SECONDS:
        return "주의", f"최신유효파일 {valid_age / 60:.0f}분 미갱신"
    return "정상", ""


def scan_camera(day_dir: Path, camera: str, now_ts: float) -> CameraReport:
    cam_dir = day_dir / f"{camera}_hls"
    if not cam_dir.exists():
        return CameraReport("문제", camera, 0, 0, 0, 0, 0, "-", "-", "디렉터리 없음")

    files = sorted(
        [path for path in cam_dir.iterdir() if path.is_file() and path.suffix.lower() in {".mp4", ".ts"}],
        key=lambda path: path.stat().st_mtime,
    )
    if not files:
        return CameraReport("문제", camera, 0, 0, 0, 0, 0, "-", "-", "파일 없음")

    sizes = [(path, path.stat().st_size) for path in files]
    latest = max(files, key=lambda path: path.stat().st_mtime)
    valid_files = [path for path, size in sizes if size > 0]
    latest_valid = max(valid_files, key=lambda path: path.stat().st_mtime) if valid_files else None
    status, note = status_for(latest, latest_valid, now_ts)
    return CameraReport(
        status=status,
        camera=camera,
        bytes_total=sum(size for _, size in sizes),
        files_total=len(files),
        mp4_count=sum(1 for path, _ in sizes if path.suffix.lower() == ".mp4"),
        ts_count=sum(1 for path, _ in sizes if path.suffix.lower() == ".ts"),
        zero_count=sum(1 for _, size in sizes if size == 0),
        latest=format_time(latest),
        latest_valid=format_time(latest_valid),
        note=note,
    )


def gb(value: int) -> float:
    return value / (1024 ** 3)


def render_report(now: datetime, reports: list[CameraReport]) -> str:
    counts = {status: sum(1 for report in reports if report.status == status) for status in ["정상", "주의", "문제"]}
    total_bytes = sum(report.bytes_total for report in reports)
    total_files = sum(report.files_total for report in reports)
    total_zero = sum(report.zero_count for report in reports)
    free_gb = shutil.disk_usage(REC_ROOT).free / (1024 ** 3)
    ymd = now.strftime("%Y%m%d")

    lines = [
        f"# CCTV 영상확보 현황 - {ymd}",
        "",
        f"- 생성시각: {now.strftime('%Y-%m-%d %H:%M:%S')}",
        (
            f"- 카메라: 정상 {counts['정상']}대 / 주의 {counts['주의']}대 / "
            f"문제 {counts['문제']}대 / 전체 {len(reports)}대"
        ),
        f"- 당일 확보량: {gb(total_bytes):.1f}GB, 파일 {total_files}개, 0-byte {total_zero}개",
        f"- 디스크 여유: {free_gb:.0f}GB",
        "",
        "| 상태 | 카메라 | 확보량(GB) | 파일 | MP4 | TS | 0-byte | 최신파일 | 최신유효파일 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        lines.append(
            "| {status} | {camera} | {gb:.2f} | {files} | {mp4} | {ts} | {zero} | {latest} | {latest_valid} |".format(
                status=report.status,
                camera=report.camera,
                gb=gb(report.bytes_total),
                files=report.files_total,
                mp4=report.mp4_count,
                ts=report.ts_count,
                zero=report.zero_count,
                latest=report.latest,
                latest_valid=report.latest_valid,
            )
        )

    needs = [report for report in reports if report.status != "정상" or report.note]
    lines.extend(["", "## 점검 필요"])
    if not needs:
        lines.append("- 없음")
    else:
        for report in needs:
            detail = report.note or report.status
            lines.append(f"- {report.camera}: {detail}")
    return "\n".join(lines) + "\n"


def main() -> int:
    now = datetime.now()
    day_dir = REC_ROOT / now.strftime("%Y%m%d")
    cameras = expected_cameras()
    if not cameras and day_dir.exists():
        cameras = sorted(path.name.removesuffix("_hls") for path in day_dir.glob("*_hls") if path.is_dir())

    now_ts = now.timestamp()
    reports = [scan_camera(day_dir, camera, now_ts) for camera in cameras]
    text = render_report(now, reports)

    outputs = [
        PROJECT_LOG_DIR / f"cctv-recording-status-{now.strftime('%Y%m%d')}.md",
        INFRA_LOG_DIR / f"cctv-recording-status-{now.strftime('%Y%m%d')}.md",
    ]
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print("wrote " + ", ".join(str(path) for path in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
