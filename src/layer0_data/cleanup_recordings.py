"""녹화 보존정책 — 오래된 영상 자동 정리 + 디스크 임계 경고.

record_hls_multi.py가 /DATA/cctv_recording/YYYYMMDD/ 구조로 무제한 누적하므로
(7.3TB ÷ ~500GB/일 ≈ 15일 만석), N일 초과 날짜 디렉토리를 삭제한다.

매일 자정 cron 권장:
  30 0 * * *  cd /workspace/prj_cctv/사고분석_설계/src/layer0_data && python3 cleanup_recordings.py

실행:
  python cleanup_recordings.py                 # 기본 21일 보존, dry-run 아님
  python cleanup_recordings.py --keep-days 14
  python cleanup_recordings.py --dry-run       # 삭제 대상만 출력
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cleanup")

REC_ROOT = Path("/DATA/cctv_recording")
DATE_DIR_RE = re.compile(r"^\d{8}$")  # YYYYMMDD
WARN_PCT = 60.0   # 사용률 이상이면 경고
CRIT_PCT = 85.0   # 사용률 이상이면 강한 경고 (보존일 단축 권고)


def disk_usage_pct(path: Path) -> tuple[float, float]:
    """(사용률%, 가용GB) 반환."""
    u = shutil.disk_usage(str(path))
    return u.used / u.total * 100, u.free / (1024 ** 3)


def cleanup(root: Path, keep_days: int, dry_run: bool) -> tuple[int, float]:
    """keep_days 초과 날짜 디렉토리 삭제. (삭제 디렉토리 수, 회수 GB) 반환."""
    cutoff = (datetime.now() - timedelta(days=keep_days)).strftime("%Y%m%d")
    removed, freed_gb = 0, 0.0
    if not root.exists():
        logger.warning("녹화 루트 없음: %s", root)
        return 0, 0.0
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not DATE_DIR_RE.match(d.name):
            continue
        if d.name >= cutoff:  # 문자열 비교 = 날짜 비교 (YYYYMMDD)
            continue
        size_gb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024 ** 3)
        if dry_run:
            logger.info("[DRY] 삭제 대상: %s (%.1fGB)", d.name, size_gb)
        else:
            shutil.rmtree(d)
            logger.info("삭제: %s (%.1fGB 회수)", d.name, size_gb)
        removed += 1
        freed_gb += size_gb
    return removed, freed_gb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REC_ROOT))
    ap.add_argument("--keep-days", type=int, default=21, help="보존 일수 (기본 21)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    pct, free_gb = disk_usage_pct(root if root.exists() else Path("/DATA"))
    logger.info("디스크 사용률 %.1f%% (가용 %.0fGB) | 보존 %d일", pct, free_gb, args.keep_days)

    removed, freed = cleanup(root, args.keep_days, args.dry_run)
    logger.info("정리 완료: %d개 디렉토리, %.1fGB %s",
                removed, freed, "회수예정(dry-run)" if args.dry_run else "회수")

    pct2, free2 = disk_usage_pct(root if root.exists() else Path("/DATA"))
    if pct2 >= CRIT_PCT:
        logger.error("⚠ 디스크 위험: %.1f%% — 보존일 단축(--keep-days) 권고", pct2)
        sys.exit(2)
    elif pct2 >= WARN_PCT:
        logger.warning("⚠ 디스크 경고: %.1f%% (가용 %.0fGB)", pct2, free2)


if __name__ == "__main__":
    main()
