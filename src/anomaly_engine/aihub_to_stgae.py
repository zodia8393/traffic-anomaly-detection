"""AI Hub #71566 → STGAE(MAAD) 형식 변환.

MAAD TSV 형식: frame_id  agent_id  x  y  major_label  minor_label
STGAE는 정규화된 궤적 시계열을 입력으로 사용.
"""
from __future__ import annotations

from pathlib import Path
from collections import defaultdict

from .aihub_loader import load_clips, Clip

DRIVING_TO_MINOR = {
    "정상": -1,
    "방향지시등 이행": -1,
    "실선구간 정상주행": -1,
    "정상차로변경 주행": -1,
    "차선물기의 정상주행": -1,
    "차선 물지 않음": -1,
    "정체구간 정상주행": -1,
    "안전거리 확보 차선 변경": -1,
    "2개 차로 연속 정상 변경": -1,
    "동시 차로 정상 변경": -1,
    "방향지시등 불이행": 0,
    "실선구간 차로변경": 1,
    "동시 차로 변경": 2,
    "차선 물기": 3,
    "2개 차로 연속 변경": 4,
    "정체구간 차선변경": 5,
    "안전거리 미확보 차선 변경": 6,
}

IMG_W, IMG_H = 1280, 720


def clip_to_tsv(clip: Clip) -> list[str]:
    """클립 → MAAD TSV 라인 리스트."""
    lines = []
    for frame in clip.frames:
        for v in frame.vehicles:
            vid = v.get("id", 0)
            bbox = v.get("bbox", [0, 0, 0, 0])
            cx = (bbox[0] + bbox[2] / 2) / IMG_W
            cy = (bbox[1] + bbox[3] / 2) / IMG_H
            driving = v.get("DrivingType", "정상")
            major = 0 if driving in ("정상",) or "정상" in driving else 1
            minor = DRIVING_TO_MINOR.get(driving, -1)
            lines.append(f"{frame.frame_idx}\t{vid}\t{cx:.6f}\t{cy:.6f}\t{major}\t{minor}")
    return lines


def convert_dataset(
    label_root: Path,
    output_dir: Path,
    max_clips: int | None = None,
    split: str = "val",
):
    """전체 데이터셋을 STGAE 형식으로 변환."""
    output_dir.mkdir(parents=True, exist_ok=True)
    clips = load_clips(label_root, max_clips=max_clips)

    normal_dir = output_dir / split / "normal"
    abnormal_dir = output_dir / split / "abnormal"
    normal_dir.mkdir(parents=True, exist_ok=True)
    abnormal_dir.mkdir(parents=True, exist_ok=True)

    stats = {"normal": 0, "abnormal": 0, "total_frames": 0}

    for clip in clips:
        lines = clip_to_tsv(clip)
        if not lines:
            continue

        target_dir = normal_dir if clip.is_normal else abnormal_dir
        label_key = "normal" if clip.is_normal else "abnormal"
        stats[label_key] += 1
        stats["total_frames"] += len(clip.frames)

        out_path = target_dir / f"{clip.clip_id}.txt"
        with open(out_path, "w") as f:
            f.write("\n".join(lines))

    print(f"변환 완료: {stats['normal']} 정상 + {stats['abnormal']} 비정상 클립")
    print(f"총 프레임: {stats['total_frames']:,}")
    print(f"저장 위치: {output_dir / split}")
    return stats


if __name__ == "__main__":
    convert_dataset(
        label_root=Path("/DATA/aihub_71566/labels/val_extracted"),
        output_dir=Path("/DATA/aihub_71566/stgae_format"),
    )
