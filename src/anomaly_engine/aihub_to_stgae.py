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
    """클립 → MAAD 7-column TSV 라인 리스트.

    MAAD format: frame_id  track_id  agent_id  x  y  major_label  minor_label
    MAADDataset uses col2=agent_id, col3:5=x,y, col[-2:]=labels.
    """
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
            lines.append(
                f"{frame.frame_idx}\t{vid}\t{vid}\t{cx:.6f}\t{cy:.6f}\t{major}\t{minor}"
            )
    return lines


def convert_dataset(
    label_root: Path,
    output_dir: Path,
    max_clips: int | None = None,
    train_ratio: float = 0.8,
):
    """전체 데이터셋을 STGAE 형식으로 변환.

    STGAE autoencoder 학습 구조:
    - train/ : 정상 클립 80% (autoencoder 학습)
    - test/  : 정상 20% + 비정상 전체 (평가)
    """
    import random

    output_dir.mkdir(parents=True, exist_ok=True)
    clips = load_clips(label_root, max_clips=max_clips)

    normal_clips = [c for c in clips if c.is_normal]
    abnormal_clips = [c for c in clips if not c.is_normal]

    random.seed(42)
    random.shuffle(normal_clips)
    split_idx = int(len(normal_clips) * train_ratio)
    train_normal = normal_clips[:split_idx]
    test_normal = normal_clips[split_idx:]

    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    stats = {"train_normal": 0, "test_normal": 0, "test_abnormal": 0, "total_frames": 0}

    def _write_clips(clips_list, target_dir, stat_key):
        for clip in clips_list:
            lines = clip_to_tsv(clip)
            if not lines:
                continue
            stats[stat_key] += 1
            stats["total_frames"] += len(clip.frames)
            out_path = target_dir / f"{clip.clip_id}.txt"
            with open(out_path, "w") as f:
                f.write("\n".join(lines))

    _write_clips(train_normal, train_dir, "train_normal")
    _write_clips(test_normal, test_dir, "test_normal")
    _write_clips(abnormal_clips, test_dir, "test_abnormal")

    print(f"변환 완료:")
    print(f"  train: {stats['train_normal']} 정상 클립")
    print(f"  test:  {stats['test_normal']} 정상 + {stats['test_abnormal']} 비정상 클립")
    print(f"  총 프레임: {stats['total_frames']:,}")
    print(f"  저장 위치: {output_dir}")
    return stats


if __name__ == "__main__":
    convert_dataset(
        label_root=Path("/DATA/aihub_71566/labels/val_extracted"),
        output_dir=Path("/DATA/aihub_71566/stgae_format"),
    )
