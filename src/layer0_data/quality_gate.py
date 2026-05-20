"""Bridge 데이터 품질 게이트 — 해상도/fps/길이 자동 검증."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from config import MIN_DURATION_SEC, MIN_FPS, MIN_RESOLUTION

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    path: Path
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration_sec: float = 0.0
    frame_count: int = 0
    codec: str = ""
    passed: bool = False
    failures: list[str] = field(default_factory=list)


def check_video(path: Path | str) -> QualityReport:
    """단일 영상 품질 검증.

    Returns:
        QualityReport — passed=True이면 게이트 통과.
    """
    path = Path(path)
    report = QualityReport(path=path)

    if not path.exists():
        report.failures.append("파일 없음")
        return report

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        report.failures.append("영상 열기 실패")
        return report

    report.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    report.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    report.fps = cap.get(cv2.CAP_PROP_FPS)
    report.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    report.codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
    cap.release()

    if report.fps > 0:
        report.duration_sec = report.frame_count / report.fps

    # 해상도
    if report.width < MIN_RESOLUTION[0] or report.height < MIN_RESOLUTION[1]:
        report.failures.append(
            f"해상도 부족: {report.width}x{report.height} < {MIN_RESOLUTION[0]}x{MIN_RESOLUTION[1]}"
        )

    # FPS
    if report.fps < MIN_FPS:
        report.failures.append(f"FPS 부족: {report.fps:.1f} < {MIN_FPS}")

    # 길이
    if report.duration_sec < MIN_DURATION_SEC:
        report.failures.append(f"길이 부족: {report.duration_sec:.1f}s < {MIN_DURATION_SEC}s")

    report.passed = len(report.failures) == 0
    return report


def check_directory(directory: Path | str, extensions: tuple = (".mp4", ".avi", ".mkv")) -> list[QualityReport]:
    """디렉토리 내 전체 영상 일괄 검증."""
    directory = Path(directory)
    reports = []
    for f in sorted(directory.rglob("*")):
        if f.suffix.lower() in extensions:
            reports.append(check_video(f))

    passed = sum(1 for r in reports if r.passed)
    logger.info("품질 검증: %d/%d 통과 (%s)", passed, len(reports), directory)
    return reports


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    if target.is_dir():
        results = check_directory(target)
        for r in results:
            status = "PASS" if r.passed else f"FAIL ({', '.join(r.failures)})"
            print(f"  {r.path.name}: {r.width}x{r.height} {r.fps:.0f}fps {r.duration_sec:.1f}s → {status}")
    else:
        r = check_video(target)
        print(f"{r.path.name}: {r.width}x{r.height} {r.fps:.0f}fps {r.duration_sec:.1f}s")
        print(f"  결과: {'PASS' if r.passed else 'FAIL'}")
        for f in r.failures:
            print(f"  ✗ {f}")
