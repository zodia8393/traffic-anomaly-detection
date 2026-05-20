"""Track 1: AI Hub #71566 — 국도 CCTV 비정상주행 판별 데이터 다운로더.

데이터셋 규모: 약 1.9TB (Training 1,862GB + Validation 225GB)
분할 zip 형식: TS.z01~z17 + TS.zip / VS.z01~z02 + VS.zip

실행 방법:
  python track1_aihub71566.py list                   # 파일 목록 조회
  python track1_aihub71566.py download --group label  # 라벨만 다운 (6GB)
  python track1_aihub71566.py download --group all    # 전체 다운 (1.9TB)
  python track1_aihub71566.py download --group val    # Validation만 (225GB)
  python track1_aihub71566.py merge                   # 분할 zip 병합+해제
  python track1_aihub71566.py status                  # 다운로드 진행 현황

사전 준비:
  1. AI Hub 가입 → API 키 발급
  2. /workspace/.env 에 AIHUB_APIKEY=... 추가
  3. 데이터셋 #71566 다운로드 승인 완료
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import AIHUB_71566_DIR, AIHUB_71566_FILES, AIHUB_71566_ID

logger = logging.getLogger(__name__)

# ── 상수 ─────────────────────────────────────────────────────────────
AIHUBSHELL_BIN = shutil.which("aihubshell") or str(
    Path.home() / ".local" / "bin" / "aihubshell"
)
DOWNLOAD_GROUPS = {
    "all": ["train_source", "train_label", "val_source", "val_label"],
    "train": ["train_source", "train_label"],
    "val": ["val_source", "val_label"],
    "label": ["train_label", "val_label"],
    "train_source": ["train_source"],
    "val_source": ["val_source"],
}


def _get_api_key() -> str:
    """AIHUB_APIKEY를 .env 또는 환경변수에서 로드."""
    key = os.getenv("AIHUB_APIKEY", "")
    if not key:
        env_path = Path("/workspace/.env")
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("AIHUB_APIKEY="):
                    key = line.split("=", 1)[1].strip().strip("'\"")
                    break
    return key


def _ensure_dir(create: bool = True) -> Path:
    """다운로드 디렉토리 확인(및 생성) 후 반환."""
    if create:
        try:
            AIHUB_71566_DIR.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            print(f"\n[ERROR] 디렉토리 생성 권한 없음: {AIHUB_71566_DIR}")
            print("  다음 명령으로 디렉토리를 먼저 생성하세요:")
            print(f"  sudo mkdir -p {AIHUB_71566_DIR}")
            print(f"  sudo chown -R $(whoami) {AIHUB_71566_DIR}")
            raise SystemExit(1)
    return AIHUB_71566_DIR


def _file_map() -> dict[str, dict]:
    """filekey → file info 맵 생성."""
    result = {}
    for group, files in AIHUB_71566_FILES.items():
        for f in files:
            f_copy = dict(f)
            f_copy["group"] = group
            result[str(f["filekey"])] = f_copy
    return result


# ═══════════════════════════════════════════════════════════════════════
# 명령: list
# ═══════════════════════════════════════════════════════════════════════

def cmd_list(create_dir: bool = True) -> None:
    """파일 목록 및 다운로드 상태 출력."""
    dest = _ensure_dir(create=create_dir)
    total_gb = 0.0

    print(f"\n{'='*70}")
    print(f"  AI Hub #71566 — 국도 CCTV 비정상주행 판별 데이터")
    print(f"  저장 경로: {dest}")
    print(f"{'='*70}\n")

    for group, files in AIHUB_71566_FILES.items():
        group_size = sum(f["size_gb"] for f in files)
        total_gb += group_size
        print(f"  [{group}] ({group_size} GB, {len(files)}개 파일)")

        for f in files:
            fpath = dest / f["filename"]
            if fpath.exists():
                actual_gb = fpath.stat().st_size / (1024**3)
                status = f"[OK {actual_gb:.1f}GB]"
            else:
                status = "[미다운로드]"
            print(f"    {f['filename']:12s}  {f['size_gb']:>4d} GB  "
                  f"filekey={f['filekey']}  {status}")
        print()

    print(f"  총 용량: {total_gb:.0f} GB")
    print(f"  디스크 여유: ", end="")
    _print_disk_free(dest)


def _print_disk_free(path: Path) -> None:
    """디스크 여유 공간 출력."""
    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024**3)
        print(f"{free_gb:.1f} GB")
    except OSError:
        print("확인 불가")


# ═══════════════════════════════════════════════════════════════════════
# 명령: download
# ═══════════════════════════════════════════════════════════════════════

def cmd_download(group: str = "label", dry_run: bool = False) -> None:
    """aihubshell을 사용하여 파일 다운로드.

    Args:
        group: 다운로드 그룹 (all/train/val/label 등)
        dry_run: True이면 명령만 출력, 실행 안 함
    """
    api_key = _get_api_key()
    if not api_key:
        print("\n[ERROR] AIHUB_APIKEY 미설정!")
        print("  /workspace/.env 에 다음 줄을 추가하세요:")
        print("  AIHUB_APIKEY=<발급받은_API_키>")
        print("\n  API 키 발급: https://aihub.or.kr → 마이페이지 → Open API")
        return

    if not Path(AIHUBSHELL_BIN).exists():
        print(f"\n[ERROR] aihubshell 미설치: {AIHUBSHELL_BIN}")
        print("  설치: curl -o aihubshell https://api.aihub.or.kr/api/aihubshell.do")
        print("        chmod +x aihubshell && mv aihubshell ~/.local/bin/")
        return

    groups = DOWNLOAD_GROUPS.get(group)
    if not groups:
        print(f"[ERROR] 알 수 없는 그룹: {group}")
        print(f"  사용 가능: {', '.join(DOWNLOAD_GROUPS.keys())}")
        return

    dest = _ensure_dir()
    _check_disk_space(dest, groups)

    # 파일별 다운로드 (이미 존재하면 스킵)
    for grp in groups:
        files = AIHUB_71566_FILES[grp]
        print(f"\n--- [{grp}] {len(files)}개 파일 ---")

        for f in files:
            fpath = dest / f["filename"]
            if fpath.exists():
                actual_gb = fpath.stat().st_size / (1024**3)
                if actual_gb >= f["size_gb"] * 0.95:  # 95% 이상이면 완료 간주
                    print(f"  SKIP {f['filename']} (이미 존재, {actual_gb:.1f}GB)")
                    continue
                else:
                    print(f"  RESUME? {f['filename']} "
                          f"({actual_gb:.1f}/{f['size_gb']}GB, 불완전)")

            cmd = _build_download_cmd(api_key, f["filekey"])

            if dry_run:
                print(f"  [DRY RUN] {' '.join(cmd)}")
                continue

            print(f"  DOWNLOAD {f['filename']} ({f['size_gb']}GB) ...")
            _run_download(cmd, dest, f["filename"])


def _check_disk_space(dest: Path, groups: list[str]) -> None:
    """다운로드 전 디스크 여유 확인."""
    needed_gb = sum(
        sum(f["size_gb"] for f in AIHUB_71566_FILES[g])
        for g in groups
    )
    try:
        free_gb = shutil.disk_usage(dest).free / (1024**3)
    except OSError:
        return

    print(f"\n  필요 용량: {needed_gb} GB")
    print(f"  디스크 여유: {free_gb:.1f} GB")

    if free_gb < needed_gb * 1.1:
        print(f"  [WARNING] 디스크 여유가 부족할 수 있습니다!")
        print(f"            (필요 {needed_gb}GB + 여유 10% = {needed_gb*1.1:.0f}GB)")


def _build_download_cmd(api_key: str, filekey: int) -> list[str]:
    """aihubshell 다운로드 명령 생성."""
    return [
        AIHUBSHELL_BIN,
        "-mode", "d",
        "-datasetkey", str(AIHUB_71566_ID),
        "-filekey", str(filekey),
        "-aihubapikey", api_key,
    ]


def _run_download(cmd: list[str], dest: Path, filename: str) -> bool:
    """aihubshell 실행 후 download.tar를 올바른 위치로 이동.

    aihubshell은 현재 디렉토리에 download.tar를 생성하므로,
    dest 디렉토리에서 실행해야 합니다.
    """
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(dest),
            timeout=7200,  # 2시간 타임아웃 (100GB 기준)
            capture_output=False,  # 진행률 실시간 출력
        )
        elapsed = time.time() - start

        if result.returncode == 0:
            # aihubshell이 tar 해제+병합까지 자동 처리
            logger.info(
                "%s 완료 (%.0f초, %.1f분)",
                filename, elapsed, elapsed / 60,
            )
            print(f"  OK {filename} ({elapsed/60:.1f}분 소요)")
            return True
        else:
            print(f"  FAIL {filename} (returncode={result.returncode})")
            return False

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT {filename} (2시간 초과)")
        return False
    except Exception as e:
        print(f"  ERROR {filename}: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════
# 명령: merge (수동 분할 zip 병합)
# ═══════════════════════════════════════════════════════════════════════

def cmd_merge(target: str = "all") -> None:
    """분할 zip 파일 병합 및 압축 해제.

    aihubshell이 자동 병합하지만, 수동 다운로드 시 사용.
    """
    dest = _ensure_dir()

    merge_sets = {
        "TS": {
            "parts": [f"TS.z{i:02d}" for i in range(1, 18)] + ["TS.zip"],
            "output": dest / "Training" / "원천데이터",
        },
        "VS": {
            "parts": ["VS.z01", "VS.z02", "VS.zip"],
            "output": dest / "Validation" / "원천데이터",
        },
        "TL": {
            "parts": ["TL.zip"],
            "output": dest / "Training" / "라벨링데이터",
        },
        "VL": {
            "parts": ["VL.zip"],
            "output": dest / "Validation" / "라벨링데이터",
        },
    }

    targets = list(merge_sets.keys()) if target == "all" else [target.upper()]

    for name in targets:
        if name not in merge_sets:
            print(f"[SKIP] 알 수 없는 대상: {name}")
            continue

        info = merge_sets[name]
        parts = [dest / p for p in info["parts"]]
        missing = [p for p in parts if not p.exists()]

        if missing:
            print(f"\n[{name}] 미다운로드 파일 {len(missing)}개 — 건너뜀")
            for m in missing[:3]:
                print(f"  - {m.name}")
            if len(missing) > 3:
                print(f"  ... 외 {len(missing)-3}개")
            continue

        # 분할 zip → 합치기 (zip은 마지막 파일이 헤더)
        output_dir = info["output"]
        output_dir.mkdir(parents=True, exist_ok=True)

        if len(parts) > 1:
            print(f"\n[{name}] 분할 zip 해제 ({len(parts)}개 파일) → {output_dir}")
            # zip 형식의 분할: z01, z02, ..., zip 순서
            # 7z 또는 unzip 사용
            zip_main = dest / f"{name}.zip"
            try:
                subprocess.run(
                    ["7z", "x", str(zip_main), f"-o{output_dir}", "-y"],
                    check=True,
                )
                print(f"  OK {name} 해제 완료")
            except FileNotFoundError:
                print(f"  [INFO] 7z 미설치. unzip으로 시도...")
                try:
                    subprocess.run(
                        ["unzip", "-o", str(zip_main), "-d", str(output_dir)],
                        check=True,
                    )
                    print(f"  OK {name} 해제 완료")
                except Exception as e:
                    print(f"  FAIL: {e}")
                    print("  sudo apt install p7zip-full 로 7z 설치 권장")
        else:
            # 단일 zip
            zip_file = parts[0]
            print(f"\n[{name}] zip 해제 → {output_dir}")
            try:
                subprocess.run(
                    ["unzip", "-o", str(zip_file), "-d", str(output_dir)],
                    check=True,
                )
                print(f"  OK {name} 해제 완료")
            except Exception as e:
                print(f"  FAIL: {e}")


# ═══════════════════════════════════════════════════════════════════════
# 명령: status
# ═══════════════════════════════════════════════════════════════════════

def cmd_status() -> None:
    """다운로드 진행 현황 요약."""
    dest = _ensure_dir(create=False)
    total_files = 0
    done_files = 0
    total_gb = 0.0
    done_gb = 0.0

    print(f"\n{'='*60}")
    print(f"  다운로드 진행 현황 — {dest}")
    print(f"{'='*60}\n")

    for group, files in AIHUB_71566_FILES.items():
        grp_total = len(files)
        grp_done = 0
        grp_size = 0.0
        grp_actual = 0.0

        for f in files:
            fpath = dest / f["filename"]
            grp_size += f["size_gb"]
            if fpath.exists():
                actual = fpath.stat().st_size / (1024**3)
                grp_actual += actual
                if actual >= f["size_gb"] * 0.95:
                    grp_done += 1

        total_files += grp_total
        done_files += grp_done
        total_gb += grp_size
        done_gb += grp_actual

        pct = (grp_done / grp_total * 100) if grp_total else 0
        bar = _progress_bar(pct)
        print(f"  {group:15s} {bar} {grp_done}/{grp_total} "
              f"({grp_actual:.1f}/{grp_size:.0f} GB)")

    print(f"\n  {'전체':15s} {done_files}/{total_files} 파일, "
          f"{done_gb:.1f}/{total_gb:.0f} GB")
    _print_disk_free(dest)


def _progress_bar(pct: float, width: int = 20) -> str:
    """텍스트 진행률 바 생성."""
    filled = int(width * pct / 100)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {pct:5.1f}%"


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if len(sys.argv) < 2:
        print("사용법:")
        print("  python track1_aihub71566.py list                    # 파일 목록")
        print("  python track1_aihub71566.py download --group label  # 라벨만 (6GB)")
        print("  python track1_aihub71566.py download --group val    # Val (225GB)")
        print("  python track1_aihub71566.py download --group all    # 전체 (1.9TB)")
        print("  python track1_aihub71566.py download --group all --dry-run")
        print("  python track1_aihub71566.py merge                   # 분할zip 해제")
        print("  python track1_aihub71566.py status                  # 진행 현황")
        return

    cmd = sys.argv[1]

    if cmd == "list":
        cmd_list(create_dir=False)

    elif cmd == "download":
        group = "label"
        dry_run = False
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--group" and i + 1 < len(sys.argv):
                group = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--dry-run":
                dry_run = True
                i += 1
            else:
                print(f"알 수 없는 옵션: {sys.argv[i]}")
                return
        cmd_download(group=group, dry_run=dry_run)

    elif cmd == "merge":
        target = sys.argv[2] if len(sys.argv) > 2 else "all"
        cmd_merge(target=target)

    elif cmd == "status":
        cmd_status()

    else:
        print(f"알 수 없는 명령: {cmd}")


if __name__ == "__main__":
    main()
