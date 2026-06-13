"""도메인 검증 — 국도 학습 모델을 고속도로 라이브 footage에 적용.

라벨 없음 → 정상 트래픽에 대한 '오탐율'로 도메인 전이 평가.
정상인데 이상확률 높으면 도메인 갭(국도↔고속도로, fps/해상도 차이).

학습: 국도 1280x720 @10fps, 75프레임(7.5s) 클립.
라이브: 고속도로 720x480 @30fps → 10fps 샘플링으로 학습 도메인 맞춤.

실행: python validate_domain.py --video <mp4> --minutes 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from supervised_detector import SupervisedTrajectoryDetector
from trajectory_features import extract_features

DEMO = "/DATA/cctv_recording/20260528/경부선_원지동_20260528_141543.mp4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=DEMO)
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--start", type=float, default=1800.0)
    ap.add_argument("--target-fps", type=float, default=10.0, help="학습 도메인 fps")
    ap.add_argument("--clip-frames", type=int, default=75, help="클립 길이(프레임)")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO("/workspace/prj/cctv/사고분석_설계/src/yolov8n.pt")
    det = SupervisedTrajectoryDetector()
    if not det.is_loaded:
        print("모델 미로드 — 중단"); return

    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, int(round(src_fps / args.target_fps)))  # 30fps→10fps = stride 3
    start_f = int(args.start * src_fps)
    n_src = int(args.minutes * 60 * src_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    print(f"영상 {W}x{H}@{src_fps:.0f}fps → {args.target_fps:.0f}fps(stride {stride}) | "
          f"{args.minutes}분, 클립 {args.clip_frames}프레임")

    # 트랙 히스토리 누적 → clip_frames 윈도우마다 채점
    track_hist: dict[int, list] = {}
    win_idx = 0
    scores = []
    fi = 0
    for raw in range(n_src):
        ret, frame = cap.read()
        if not ret:
            break
        if raw % stride != 0:
            continue
        r = model.track(frame, persist=True, tracker="bytetrack.yaml",
                        conf=0.3, classes=[2, 5, 7], verbose=False)[0]
        if r.boxes is not None and r.boxes.id is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), tid in zip(xyxy, ids):
                cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H  # 정규화 (학습과 동일)
                track_hist.setdefault(int(tid), []).append((win_idx, cx, cy))
        win_idx += 1
        fi += 1

        # 클립 경계 → 채점 후 리셋
        if win_idx >= args.clip_frames:
            sc = det.score(track_hist)
            if sc is not None:
                scores.append(sc)
            track_hist = {}
            win_idx = 0

    cap.release()
    scores = np.array(scores)
    if len(scores) == 0:
        print("채점된 클립 없음"); return

    thr = det._threshold
    flagged = np.mean(scores >= thr)
    print(f"\n=== 도메인 검증 결과 (정상 고속도로 트래픽) ===")
    print(f"  클립 {len(scores)}개 | 이상확률 평균 {scores.mean():.3f} 중앙 {np.median(scores):.3f}")
    print(f"  임계({thr:.2f}) 초과(오탐) 비율: {flagged:.1%}")
    print(f"  분포: <0.3 {np.mean(scores<0.3):.0%} | 0.3~0.5 {np.mean((scores>=0.3)&(scores<0.5)):.0%} "
          f"| 0.5~0.7 {np.mean((scores>=0.5)&(scores<0.7)):.0%} | >=0.7 {np.mean(scores>=0.7):.0%}")
    if flagged > 0.3:
        print(f"  ⚠ 도메인 갭: 정상 트래픽 {flagged:.0%}를 이상으로 오탐 — 고속도로 재학습/캘리브 필요")
    elif flagged > 0.15:
        print(f"  △ 중간 도메인 갭: 오탐 {flagged:.0%} — 임계 상향 또는 일부 재학습 권고")
    else:
        print(f"  ✓ 도메인 전이 양호: 정상 트래픽 오탐 {flagged:.0%}")


if __name__ == "__main__":
    main()
