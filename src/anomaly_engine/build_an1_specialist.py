"""an1 특화 데이터셋 — 방향지시등 불이행(an1) vs 정상이행 차선변경(normal-01).

전체 분류기에서 an1이 천장(0.44)인 이유: an1(차선변경+무신호)과 정상직진(무신호)이
clip 집계론 구분 불가. 본 모델은 '차선변경끼리만' 비교 — 깜빡이 점멸이 결정적
(가설검증 an1=0.0 vs 정상이행=1.9). 차선변경 차량의 blink + 횡이동 + 키네마틱.

실행: python build_an1_specialist.py   (이미지 처리, 멀티프로세싱)
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
from trajectory_features import extract_features

try:
    import cv2
except ImportError:
    cv2 = None
from turn_signal_detector import amber_intensity, blink_score

LBL = "/DATA/aihub_71566/labels/val_extracted"
SRC = "/DATA/aihub_71566/source/val"
OUT = "/workspace/prj/cctv/사고분석_설계/models/an1_specialist.npz"

# an1(불이행) vs normal-01(이행) — 둘 다 차선변경
CLASSES = [
    ("비정상/01.방향지시등 불이행", 1),       # an1
    ("정상/01.방향지시등 이행 차선변경", 0),  # 정상
]


def process_clip(args):
    clip_lbl, src_dir, label = args
    jsons = sorted(glob.glob(os.path.join(clip_lbl, "*.json")))
    if len(jsons) < 5:
        return None
    tracks: dict[int, list] = {}
    bbox_by_frame: dict[int, dict] = {}
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
            tracks.setdefault(int(oid), []).append((fi, (bb[0]+bb[2]/2)/W, (bb[1]+bb[3]/2)/H))
            bbox_by_frame.setdefault(fi, {})[int(oid)] = bb

    feat = extract_features(tracks)
    if feat is None:
        return None

    # 차선변경 차량 = 횡(분산 작은 축) 이동 최대
    all_d = [np.diff(np.array([(x, y) for _, x, y in sorted(p)]), axis=0)
             for p in tracks.values() if len(p) >= 2]
    lat_axis = 0
    if all_d:
        D = np.concatenate(all_d)
        lat_axis = 0 if D[:, 0].std() < D[:, 1].std() else 1
    changer, best = None, -1.0
    for oid, p in tracks.items():
        if len(p) < 3:
            continue
        xy = np.array([(x, y) for _, x, y in sorted(p)])
        lat = float(np.abs(np.diff(xy[:, lat_axis])).sum())
        if lat > best:
            best, changer = lat, oid

    amber = []
    if cv2 is not None and changer is not None:
        for fi in sorted(bbox_by_frame):
            if changer not in bbox_by_frame[fi]:
                continue
            img = cv2.imread(os.path.join(src_dir, os.path.basename(jsons[fi]).replace(".json", ".png")))
            if img is None:
                continue
            x, y, w, h = [int(v) for v in bbox_by_frame[fi][changer]]
            x, y = max(0, x), max(0, y)
            crop = img[y:min(img.shape[0], y+h), x:min(img.shape[1], x+w)]
            amber.append(amber_intensity(crop))
    blink = blink_score(amber) if len(amber) >= 8 else 0.0
    return np.concatenate([feat, [blink, best]]).tolist(), label


def main():
    tasks = []
    for sub, label in CLASSES:
        for clip in sorted(glob.glob(os.path.join(LBL, sub, "*"))):
            tasks.append((clip, os.path.join(SRC, sub, os.path.basename(clip)), label))
    print(f"클립 {len(tasks)} (an1+정상이행)")
    with Pool(20) as p:
        res = p.map(process_clip, tasks)
    X, y = [], []
    for r in res:
        if r is not None:
            X.append(r[0]); y.append(r[1])
    X, y = np.array(X), np.array(y)
    print(f"유효 {len(X)} (an1 {int(y.sum())}/정상이행 {int((y==0).sum())})")
    np.savez(OUT, X=X, y=y)
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
