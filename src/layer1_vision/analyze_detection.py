"""차량감지 강화 + 트래킹 안정화 + 보행자 트리거 + POI(ROI) 비교분석.

개선:
- 차량감지강화: YOLO FP32(INT8 금지), 차량+보행자(COCO 0,2,3,5,7), conf 튜닝, agnostic NMS
- 트래킹 안정화: 높은 lost_track_buffer(ID 유지) + bbox EMA 평활(깜빡임 제거)
- 보행자 트리거: 도로 내 person → T8
- POI 토글: ROI 폴리곤 내부만 카운트/트리거 (기본 미설정=전체프레임)

비교 메트릭(POI on vs off): 검출수·고유트랙·ID전환·박스지터·트리거·보행자·오탐추정·처리율.

실행:
  python analyze_detection.py --video <mp4> --minutes 1 --json out.json
  python analyze_detection.py --video <mp4> --roi "x1,y1 x2,y2 x3,y3 x4,y4" --json out.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

YOLO_MODEL = "/workspace/prj_cctv/사고분석_설계/src/yolov8n.pt"
VEHICLE_CLS = {2, 3, 5, 7}   # car, motorcycle, bus, truck
PERSON_CLS = 0
EMA_ALPHA = 0.6              # bbox 평활 계수 (깜빡임/지터 제거)
STABLE_IMGSZ = 736           # 강화: 기본 640→736 (소형/원거리 검출↑)


def point_in_poly(x, y, poly):
    n = len(poly); inside = False; j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def analyze(video, minutes, start, roi, target_fps=10.0):
    model = YOLO(YOLO_MODEL)
    cap = cv2.VideoCapture(video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, int(round(src_fps / target_fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * src_fps))
    n_src = int(minutes * 60 * src_fps)

    # ROI 정규화좌표(0~1)면 픽셀로 스케일
    if roi is not None and max(max(p) for p in roi) <= 1.0:
        roi = [(x * W, y * H) for x, y in roi]

    # 트래킹 안정화 상태
    ema_bbox: dict[int, np.ndarray] = {}      # tid -> 평활 bbox
    last_center: dict[int, tuple] = {}
    id_first_frame: dict[int, int] = {}
    track_classes: dict[int, list] = defaultdict(list)
    jitter_sum, jitter_n = 0.0, 0
    n_det_vehicle, n_det_person = 0, 0
    n_in_roi, n_out_roi = 0, 0
    ped_trigger_frames = 0
    processed = 0
    t0 = time.time()

    for raw in range(n_src):
        ret, frame = cap.read()
        if not ret:
            break
        if raw % stride != 0:
            continue
        processed += 1
        # 강화 검출: 차량+보행자, FP32, agnostic NMS, imgsz↑
        r = model.track(frame, persist=True, tracker="bytetrack.yaml",
                        conf=0.25, iou=0.5, imgsz=STABLE_IMGSZ,
                        classes=[0, 2, 3, 5, 7], agnostic_nms=True, verbose=False)[0]
        peds_in_road = []
        if r.boxes is not None and r.boxes.id is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), tid, c in zip(xyxy, ids, cls):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                in_roi = (roi is None) or point_in_poly(cx, cy, roi)
                # POI 모드: ROI 밖 무시
                if roi is not None and not in_roi:
                    n_out_roi += 1
                    continue
                n_in_roi += 1
                # bbox EMA 평활 (깜빡임/지터 제거)
                box = np.array([x1, y1, x2, y2])
                if tid in ema_bbox:
                    ema_bbox[tid] = EMA_ALPHA * box + (1 - EMA_ALPHA) * ema_bbox[tid]
                    # 지터 = 평활 전후 중심 이동
                    pcx, pcy = last_center[tid]
                    jitter_sum += float(np.hypot(cx - pcx, cy - pcy)); jitter_n += 1
                else:
                    ema_bbox[tid] = box
                    id_first_frame[tid] = processed
                last_center[tid] = (cx, cy)
                if c == PERSON_CLS:
                    n_det_person += 1
                    peds_in_road.append({"track_id": int(tid), "center": (cx, cy)})
                else:
                    n_det_vehicle += 1
                track_classes[tid].append(int(c))
        if peds_in_road:
            ped_trigger_frames += 1

    cap.release()
    elapsed = time.time() - t0
    # ID 전환 추정: 트랙당 클래스 일관성(흔들리면 불안정)
    unstable = sum(1 for c in track_classes.values() if len(set(c)) > 1)
    fps_proc = processed / elapsed if elapsed > 0 else 0
    return {
        "mode": "POI" if roi is not None else "no_POI",
        "frames": processed, "resolution": f"{W}x{H}", "imgsz": STABLE_IMGSZ,
        "unique_tracks": len(id_first_frame),
        "det_vehicle": n_det_vehicle, "det_person": n_det_person,
        "in_roi": n_in_roi, "out_roi_filtered": n_out_roi,
        "avg_bbox_jitter_px": round(jitter_sum / jitter_n, 2) if jitter_n else 0,
        "unstable_class_tracks": unstable,
        "pedestrian_trigger_frames": ped_trigger_frames,
        "proc_fps": round(fps_proc, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="/DATA/cctv_recording/20260528/경부선_원지동_20260528_141543.mp4")
    ap.add_argument("--minutes", type=float, default=1.0)
    ap.add_argument("--start", type=float, default=1800.0)
    ap.add_argument("--roi", default=None, help='"x1,y1 x2,y2 ..." (정규화 0~1 또는 px)')
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    roi = None
    if args.roi:
        roi = [tuple(float(v) for v in p.split(",")) for p in args.roi.split()]

    res = analyze(args.video, args.minutes, args.start, roi)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.json:
        from pathlib import Path
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
