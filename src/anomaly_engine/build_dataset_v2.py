"""통합 학습셋 빌더 v2 — 궤적 운동학 + 깜빡이(영상) 특징.

val_extracted(이미지+라벨, 전 유형)에서 클립별로:
  - 궤적 특징(라벨 bbox 중심 → trajectory_features)
  - 깜빡이 점멸 스코어(이상/대상 차량 크롭 → turn_signal_detector)
를 추출하여 npz로 저장. an1(방향지시등 불이행) 감지를 영상으로 보완.

실행: python build_dataset_v2.py   (멀티프로세싱, 24코어)
"""
from __future__ import annotations

import glob
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trajectory_features import FEATURE_NAMES, extract_features

try:
    import cv2
except ImportError:
    cv2 = None
from turn_signal_detector import amber_intensity, blink_score

LBL = "/DATA/aihub_71566/labels/val_extracted"
SRC = "/DATA/aihub_71566/source/val"
OUT = "/workspace/prj/work/AI기반 교통상황 대응 기술 개발 연구/사고분석_설계/models/dataset_v2.npz"

# 비정상 유형 디렉토리 → 라벨번호
ANOM_DIRS = {
    "01.방향지시등 불이행": 1, "02.실선구간 차선변경": 2, "03.동시 차로변경": 3,
    "04.차선 물기": 4, "05.2개 차로 연속 변경": 5, "06.정체구간 차선변경": 6,
    "07.안전거리 미확보 차선변경": 7,
}


def process_clip(args):
    """(clip_lbl_dir, src_dir, is_anom, an_type) → (feat_vector, label, type) or None.

    깜빡이는 '차선변경 차량'(횡이동 최대 트랙)에서 계산 — 가설검증의 깨끗한 분리 재현.
    """
    clip_lbl, src_dir, is_anom, an_type = args
    jsons = sorted(glob.glob(os.path.join(clip_lbl, "*.json")))
    if len(jsons) < 5:
        return None

    # ── pass 1: 트랙(json만, 빠름) + bbox 보관 ──
    tracks: dict[int, list] = {}
    bbox_by_frame: dict[int, dict[int, list]] = {}  # fi -> {oid: bbox}
    for fi, jp in enumerate(jsons):
        try:
            d = json.load(open(jp))
        except Exception:
            continue
        W = d.get("imageInfo", {}).get("width", 1280)
        H = d.get("imageInfo", {}).get("height", 720)
        for a in d.get("annotation", []):
            oid, bb = a.get("id"), a.get("bbox")
            if not bb or len(bb) < 4 or oid is None:
                continue
            cx = (bb[0] + bb[2] / 2) / W
            cy = (bb[1] + bb[3] / 2) / H
            tracks.setdefault(int(oid), []).append((fi, cx, cy))
            bbox_by_frame.setdefault(fi, {})[int(oid)] = bb

    feat = extract_features(tracks)
    if feat is None:
        return None

    # ── 차선변경 차량 = 횡(분산 작은 축) 총이동 최대 트랙 ──
    all_d = [np.diff(np.array([(x, y) for _, x, y in sorted(p)]), axis=0)
             for p in tracks.values() if len(p) >= 2]
    lat_axis = 0
    if all_d:
        D = np.concatenate(all_d)
        lat_axis = 0 if D[:, 0].std() < D[:, 1].std() else 1
    changer, best_lat = None, -1.0
    for oid, p in tracks.items():
        if len(p) < 3:
            continue
        xy = np.array([(x, y) for _, x, y in sorted(p)])
        lat = float(np.abs(np.diff(xy[:, lat_axis])).sum())
        if lat > best_lat:
            best_lat, changer = lat, oid

    # ── pass 2: 차선변경 차량 크롭만 로드(프레임 절반) → amber ──
    amber: list[float] = []
    if cv2 is not None and changer is not None:
        for fi in sorted(bbox_by_frame):
            if fi % 2 != 0 or changer not in bbox_by_frame[fi]:
                continue
            jp = jsons[fi]
            img_p = os.path.join(src_dir, os.path.basename(jp).replace(".json", ".png"))
            img = cv2.imread(img_p)
            if img is None:
                continue
            x, y, w, h = [int(v) for v in bbox_by_frame[fi][changer]]
            x, y = max(0, x), max(0, y)
            crop = img[y:min(img.shape[0], y+h), x:min(img.shape[1], x+w)]
            amber.append(amber_intensity(crop))

    blink = blink_score(amber) if len(amber) >= 8 else 0.0
    full = np.concatenate([feat, [blink]])
    return full.tolist(), (1 if is_anom else 0), (f"an{an_type}" if is_anom else "normal")


def main():
    CAP = int(os.environ.get("CLIP_CAP", "180"))  # 유형당 클립 상한(속도)
    tasks = []
    # 비정상
    for dname, an in ANOM_DIRS.items():
        for clip in sorted(glob.glob(os.path.join(LBL, "비정상", dname, "*")))[:CAP]:
            tasks.append((clip, os.path.join(SRC, "비정상", dname, os.path.basename(clip)), True, an))
    # 정상
    for clip_parent in sorted(glob.glob(os.path.join(LBL, "정상", "*"))):
        dname = os.path.basename(clip_parent)
        for clip in sorted(glob.glob(os.path.join(clip_parent, "*")))[:CAP]:
            tasks.append((clip, os.path.join(SRC, "정상", dname, os.path.basename(clip)), False, 0))

    print(f"클립 {len(tasks)}개 처리 (멀티프로세싱)...")
    with Pool(20) as p:
        results = p.map(process_clip, tasks)

    X, y, types = [], [], []
    for r in results:
        if r is not None:
            f, lab, t = r
            X.append(f); y.append(lab); types.append(t)
    X = np.array(X); y = np.array(y)
    print(f"유효 {len(X)} (비정상 {int(y.sum())}/정상 {int((y==0).sum())}), 특징 {X.shape[1]}개(궤적+깜빡이)")
    np.savez(OUT, X=X, y=y, types=np.array(types),
             feature_names=np.array(FEATURE_NAMES + ["blink_score"]))
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
