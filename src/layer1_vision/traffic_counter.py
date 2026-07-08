"""방향별 교통량 카운터 — POI=영상 전체, 이동벡터로 상/하행 구분.

설계 원칙(놓치는 것 최소화):
- POI 전체프레임: 차량이 화면을 가로지르는 수십 프레임 내내 검지 기회 → 검지율↑
  (얇은 계수선은 통과 1~2프레임에만 의존 → 누락 위험)
- 매 프레임 처리(stride=1 기본): 빠른 차량의 프레임 점프 누락 방지
  (detect_interval>1 금지 — 메모리상 -91% valid)
- FP32 + imgsz 736 + agnostic NMS: 소형/원거리 검지율↑
- 높은 track_buffer: ID 단편화로 인한 '한 차 두 번 카운트'(중복) 억제

계수 방식: 전체프레임에서 형성된 고유 트랙 = 통과 차량 1대. 각 트랙의 순이동벡터를
도로축(전체 변위 PCA)에 투영한 부호로 상/하행 분류. 차종은 트랙 내 최빈 클래스.

검증 전제: 본 카운터의 출력은 반드시 육안 정답(GT) 대조로 검지율/중복률을 입증해야 함.

실행:
  python traffic_counter.py --video <mp4> --start 49355 --seconds 60 --json out.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

YOLO_MODEL = "/workspace/prj/work/AI기반 교통상황 대응 기술 개발 연구/사고분석_설계/src/yolov8n.pt"
VEHICLE_CLS = {2: "승용차", 3: "이륜차", 5: "버스", 7: "화물차"}
IMGSZ = 736
MIN_TRACK_FRAMES = 4       # 이보다 짧은 트랙은 스푸리어스 후보(검증로그 남김)
MIN_DISPLACE_PX = 12.0     # 순이동 이 미만 = 정차/노이즈 (계수 제외, 로그)


def count_directional(video, start_sec, seconds, stride=1, track_buffer=120,
                      save_viz=None):
    model = YOLO(YOLO_MODEL)
    cap = cv2.VideoCapture(video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * src_fps))
    n_src = int(seconds * src_fps)

    # 트랙별 궤적/클래스 누적
    traj: dict[int, list] = defaultdict(list)        # tid -> [(cx,cy),...]
    tcls: dict[int, list] = defaultdict(list)        # tid -> [cls,...]
    first_frame_viz = None
    processed = 0
    t0 = time.time()

    # 높은 track_buffer로 ID 연속성(중복계수 억제). 커스텀 tracker yaml 생성.
    tracker_yaml = _make_tracker_yaml(track_buffer)

    for raw in range(n_src):
        ret, frame = cap.read()
        if not ret:
            break
        if raw % stride != 0:
            continue
        processed += 1
        r = model.track(frame, persist=True, tracker=tracker_yaml,
                        conf=0.20, iou=0.5, imgsz=IMGSZ,
                        classes=[2, 3, 5, 7], agnostic_nms=True, verbose=False)[0]
        if r.boxes is not None and r.boxes.id is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), tid, c in zip(xyxy, ids, cls):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                traj[tid].append((cx, cy))
                tcls[tid].append(int(c))
        if save_viz and first_frame_viz is None:
            first_frame_viz = frame.copy()

    cap.release()
    elapsed = time.time() - t0

    # ── 도로축 결정: 모든 트랙 순이동벡터의 PCA 주축 ──
    disps = []
    for tid, pts in traj.items():
        if len(pts) >= MIN_TRACK_FRAMES:
            d = np.array(pts[-1]) - np.array(pts[0])
            if np.hypot(*d) >= MIN_DISPLACE_PX:
                disps.append(d)
    axis = _principal_axis(disps) if disps else np.array([0.0, 1.0])

    # ── 트랙별 방향 분류 + 계수 ──
    dirA = Counter()  # axis 양(+) 방향
    dirB = Counter()  # axis 음(-) 방향
    counted, dropped_short, dropped_static = 0, 0, 0
    track_log = []
    for tid, pts in traj.items():
        n = len(pts)
        d = np.array(pts[-1]) - np.array(pts[0])
        disp = float(np.hypot(*d))
        cls_mode = Counter(tcls[tid]).most_common(1)[0][0]
        cls_name = VEHICLE_CLS.get(cls_mode, str(cls_mode))
        if n < MIN_TRACK_FRAMES:
            dropped_short += 1
            track_log.append({"tid": int(tid), "n": n, "disp": round(disp, 1),
                              "cls": cls_name, "status": "dropped_short"})
            continue
        if disp < MIN_DISPLACE_PX:
            dropped_static += 1
            track_log.append({"tid": int(tid), "n": n, "disp": round(disp, 1),
                              "cls": cls_name, "status": "dropped_static"})
            continue
        proj = float(np.dot(d, axis))
        side = "A" if proj >= 0 else "B"
        (dirA if proj >= 0 else dirB)[cls_name] += 1
        counted += 1
        track_log.append({"tid": int(tid), "n": n, "disp": round(disp, 1),
                          "cls": cls_name, "dir": side, "proj": round(proj, 1)})

    # 방향 라벨링: axis의 화면상 의미 (위/아래, 좌/우)
    label_A, label_B = _direction_labels(axis)

    result = {
        "video": video, "window": f"{start_sec}s+{seconds}s", "stride": stride,
        "resolution": f"{W}x{H}", "imgsz": IMGSZ, "track_buffer": track_buffer,
        "frames_processed": processed, "proc_fps": round(processed / elapsed, 1) if elapsed else 0,
        "road_axis": [round(float(axis[0]), 3), round(float(axis[1]), 3)],
        "directions": {
            f"{label_A}": {"total": sum(dirA.values()), "by_class": dict(dirA)},
            f"{label_B}": {"total": sum(dirB.values()), "by_class": dict(dirB)},
        },
        "counted": counted,
        "tracks_total": len(traj),
        "dropped_short": dropped_short,      # MIN_TRACK_FRAMES 미만
        "dropped_static": dropped_static,    # 정차/노이즈
        "_track_log": track_log,
    }

    if save_viz and first_frame_viz is not None:
        _draw_trajectories(first_frame_viz, traj, tcls, axis, save_viz)
        result["viz"] = save_viz

    return result


def _make_tracker_yaml(track_buffer):
    import tempfile
    cfg = f"""tracker_type: bytetrack
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25
track_buffer: {track_buffer}
match_thresh: 0.8
fuse_score: True
"""
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    f.write(cfg); f.close()
    return f.name


def _principal_axis(disps):
    """변위벡터 집합의 주축(단위벡터). 양방향 혼재라 |proj| 분산 최대 축."""
    X = np.array(disps)
    # 부호 모호성 제거 위해 outer product 평균의 최대 고유벡터 사용
    M = (X[:, :, None] * X[:, None, :]).mean(axis=0)  # 2x2
    w, v = np.linalg.eigh(M)
    axis = v[:, np.argmax(w)]
    return axis / (np.linalg.norm(axis) + 1e-9)


def _direction_labels(axis):
    """축 방향을 화면 의미로 라벨. axis 양(+)이 향하는 쪽."""
    ax, ay = axis
    # 수직성분 우선(상/하행 통상 화면 상하), 약하면 좌우
    if abs(ay) >= abs(ax):
        # 화면 위로 갈수록 y 감소 → ay<0이 '위쪽(원거리)'
        a = "방향A(화면상단행)" if ay < 0 else "방향A(화면하단행)"
        b = "방향B(화면하단행)" if ay < 0 else "방향B(화면상단행)"
    else:
        a = "방향A(화면우측행)" if ax > 0 else "방향A(화면좌측행)"
        b = "방향B(화면좌측행)" if ax > 0 else "방향B(화면우측행)"
    return a, b


def _draw_trajectories(frame, traj, tcls, axis, out):
    """검증용: 트랙 궤적을 방향별 색으로 + 축 표시."""
    for tid, pts in traj.items():
        if len(pts) < MIN_TRACK_FRAMES:
            continue
        d = np.array(pts[-1]) - np.array(pts[0])
        if np.hypot(*d) < MIN_DISPLACE_PX:
            continue
        proj = np.dot(d, axis)
        color = (0, 200, 0) if proj >= 0 else (0, 120, 255)  # A=초록 B=주황
        ptsi = np.array(pts, dtype=np.int32)
        cv2.polylines(frame, [ptsi], False, color, 2)
        cv2.circle(frame, tuple(ptsi[-1]), 4, color, -1)
        cv2.putText(frame, str(tid), tuple(ptsi[-1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1)
    cv2.imwrite(out, frame)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--track-buffer", type=int, default=120)
    ap.add_argument("--viz", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    res = count_directional(args.video, args.start, args.seconds, args.stride,
                            args.track_buffer, args.viz)
    # 콘솔 요약 (track_log 제외)
    summary = {k: v for k, v in res.items() if k != "_track_log"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json:
        from pathlib import Path
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
