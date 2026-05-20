"""Track 1: Bridge 데이터셋 다운로더 — AI Hub #175 + DoTA.

실행 방법:
  python track1_bridge.py aihub          # AI Hub #175 다운로드
  python track1_bridge.py dota           # DoTA 다운로드
  python track1_bridge.py verify <dir>   # 다운로드 후 품질 검증

사전 준비:
  - AI Hub: https://aihub.or.kr 가입 → API 키 발급 → .env에 AIHUB_API_KEY=... 추가
  - DoTA: GitHub에서 직접 다운로드 링크 확인
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import AIHUB_DATASET_ID, BRIDGE_DIR, DOTA_URL

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
# AI Hub #175
# ═════════════════════════════════════════════════════════════════════

class AIHubDownloader:
    """AI Hub 데이터셋 다운로더.

    AI Hub OpenAPI를 통해 데이터셋 메타 조회 → 파일 목록 취득 → 개별 다운로드.
    https://aihub.or.kr/devsport/openapi/list.do 참조.
    """

    BASE_URL = "https://api.aihub.or.kr"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("AIHUB_API_KEY", "")
        if not self.api_key:
            logger.warning("AIHUB_API_KEY 미설정 — .env 또는 환경변수에 추가 필요")
        self.output_dir = BRIDGE_DIR / "aihub_175"

    def fetch_metadata(self) -> dict:
        """데이터셋 메타 정보 조회."""
        import requests

        url = f"{self.BASE_URL}/datasets/{AIHUB_DATASET_ID}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        meta = resp.json()
        logger.info("AI Hub #%d 메타: %s", AIHUB_DATASET_ID, meta.get("title", "?"))
        return meta

    def list_files(self) -> list[dict]:
        """다운로드 가능 파일 목록 조회."""
        import requests

        url = f"{self.BASE_URL}/datasets/{AIHUB_DATASET_ID}/files"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        files = resp.json().get("files", [])
        logger.info("파일 %d개 확인", len(files))
        return files

    def download(self, file_info: dict) -> Path:
        """단일 파일 다운로드."""
        import requests

        self.output_dir.mkdir(parents=True, exist_ok=True)
        url = file_info["download_url"]
        fname = file_info.get("filename", url.split("/")[-1])
        dest = self.output_dir / fname

        if dest.exists():
            logger.info("이미 존재: %s", dest)
            return dest

        headers = {"Authorization": f"Bearer {self.api_key}"}
        logger.info("다운로드 시작: %s → %s", fname, dest)
        resp = requests.get(url, headers=headers, stream=True, timeout=300)
        resp.raise_for_status()

        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info("다운로드 완료: %s (%.1f MB)", fname, dest.stat().st_size / 1e6)
        return dest

    def download_all(self, max_files: int | None = None) -> list[Path]:
        """전체 또는 max_files개 다운로드."""
        files = self.list_files()
        if max_files:
            files = files[:max_files]

        paths = []
        for fi in files:
            paths.append(self.download(fi))
        return paths


# ═════════════════════════════════════════════════════════════════════
# DoTA (Detection of Traffic Anomaly)
# ═════════════════════════════════════════════════════════════════════

class DoTADownloader:
    """DoTA 데이터셋 다운로더.

    ICCV 2021 논문 기반 — 4677건 CCTV 시점 교통 이상 영상.
    GitHub: https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly
    다운로드는 Google Drive 링크를 통해 수행.
    """

    GDRIVE_SPLITS = {
        "train": "https://drive.google.com/drive/folders/TRAIN_FOLDER_ID",
        "val": "https://drive.google.com/drive/folders/VAL_FOLDER_ID",
        "test": "https://drive.google.com/drive/folders/TEST_FOLDER_ID",
    }
    ANNOTATIONS_URL = "https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly/raw/main/annotations"

    def __init__(self):
        self.output_dir = BRIDGE_DIR / "dota"

    def download_annotations(self) -> Path:
        """어노테이션 JSON 다운로드."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        anno_dir = self.output_dir / "annotations"
        anno_dir.mkdir(exist_ok=True)

        import requests

        for split in ("train", "val", "test"):
            url = f"{self.ANNOTATIONS_URL}/{split}.json"
            dest = anno_dir / f"{split}.json"
            if dest.exists():
                logger.info("이미 존재: %s", dest)
                continue
            logger.info("어노테이션 다운로드: %s", url)
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            dest.write_text(resp.text, encoding="utf-8")

        return anno_dir

    def download_videos(self, split: str = "val", max_count: int | None = None) -> list[Path]:
        """gdown으로 Google Drive 폴더 다운로드.

        사전 설치: pip install gdown
        """
        try:
            import gdown
        except ImportError:
            logger.error("gdown 미설치 — pip install gdown 실행 필요")
            return []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        video_dir = self.output_dir / "videos" / split
        video_dir.mkdir(parents=True, exist_ok=True)

        folder_url = self.GDRIVE_SPLITS.get(split)
        if not folder_url:
            logger.error("알 수 없는 split: %s", split)
            return []

        logger.info("DoTA %s split 다운로드 시작 → %s", split, video_dir)
        gdown.download_folder(folder_url, output=str(video_dir), quiet=False)

        paths = sorted(video_dir.glob("*.mp4"))
        if max_count:
            paths = paths[:max_count]
        logger.info("DoTA %s: %d건 다운로드 완료", split, len(paths))
        return paths


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if len(sys.argv) < 2:
        print("사용법:")
        print("  python track1_bridge.py aihub          # AI Hub #175")
        print("  python track1_bridge.py dota            # DoTA")
        print("  python track1_bridge.py verify <dir>    # 품질 검증")
        return

    cmd = sys.argv[1]

    if cmd == "aihub":
        dl = AIHubDownloader()
        if not dl.api_key:
            print("AIHUB_API_KEY를 .env에 설정하세요.")
            print(f"  저장 경로: {dl.output_dir}")
            return
        meta = dl.fetch_metadata()
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        # dl.download_all(max_files=10)  # 실제 다운로드 시 주석 해제

    elif cmd == "dota":
        dl = DoTADownloader()
        print(f"DoTA 저장 경로: {dl.output_dir}")
        print(f"GitHub: {DOTA_URL}")
        # dl.download_annotations()        # 어노테이션 먼저
        # dl.download_videos("val", 50)    # val split 50건

    elif cmd == "verify":
        from quality_gate import check_directory
        target = Path(sys.argv[2]) if len(sys.argv) > 2 else BRIDGE_DIR
        reports = check_directory(target)
        passed = sum(1 for r in reports if r.passed)
        print(f"\n품질 검증 결과: {passed}/{len(reports)} 통과")
        for r in reports:
            status = "PASS" if r.passed else f"FAIL"
            print(f"  {r.path.name}: {r.width}x{r.height} {r.fps:.0f}fps {r.duration_sec:.1f}s → {status}")
            for f in r.failures:
                print(f"    - {f}")

    else:
        print(f"알 수 없는 명령: {cmd}")


if __name__ == "__main__":
    main()
