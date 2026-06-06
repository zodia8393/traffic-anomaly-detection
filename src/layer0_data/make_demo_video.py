"""3종 차량 분류(승용차/버스/트럭) + 추적 데모 영상 제작.

CCTV 녹화 영상에서 차량을 검출·추적하고, COCO 클래스를 승용차/버스/트럭 3종으로
매핑하여 바운딩박스 + 한글 라벨 + 누적 집계 오버레이가 그려진 데모 영상을 출력한다.

실행:
  python make_demo_video.py \
    --input /DATA/cctv_recording/20260528/경부선_원지동_20260528_141543.mp4 \
    --output /workspace/prj_cctv/사고분석_설계/output/demo_3class_tracking.mp4 \
    --duration 300 --start 0
"""
from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# COCO id → 3종 분류
COCO_3CLASS = {2: "승용차", 5: "버스", 7: "트럭"}
# BGR 색상 (OpenCV)
COLORS_BGR = {"승용차": (0, 204, 0), "버스": (255, 102, 0), "트럭": (51, 51, 255)}
# RGB 색상 (PIL)
COLORS_RGB = {"승용차": (0, 204, 0), "버스": (0, 102, 255), "트럭": (255, 51, 51)}

FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"


def load_fonts():
    return {
        "label": ImageFont.truetype(FONT_PATH, 15),
        "hud": ImageFont.truetype(FONT_PATH, 22),
        "small": ImageFont.truetype(FONT_PATH, 16),
    }


def draw_overlay_pil(frame, boxes, finished, fonts, frame_idx, fps):
    """PIL로 박스 라벨 + HUD를 한 번에 렌더링 (cv2↔PIL 변환 1회)."""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    # 박스 + 라벨
    for (x1, y1, x2, y2, cls_name, tid) in boxes:
        color = COLORS_RGB[cls_name]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{cls_name} #{tid}"
        tw = draw.textlength(label, font=fonts["label"])
        # 라벨 배경
        ly = max(0, y1 - 18)
        draw.rectangle([x1, ly, x1 + tw + 6, ly + 18], fill=color)
        draw.text((x1 + 3, ly + 1), label, font=fonts["label"], fill=(255, 255, 255))

    # 좌상단 HUD — 누적 집계
    car = finished.get("승용차", 0)
    bus = finished.get("버스", 0)
    truck = finished.get("트럭", 0)
    total = car + bus + truck
    hud_lines = [
        ("누적 통과 차량", (255, 255, 255)),
        (f"승용차 {car}", COLORS_RGB["승용차"]),
        (f"버스 {bus}", COLORS_RGB["버스"]),
        (f"트럭 {truck}", COLORS_RGB["트럭"]),
        (f"합계 {total}", (255, 220, 0)),
    ]
    # 반투명 배경 박스
    draw.rectangle([8, 8, 200, 8 + 26 * len(hud_lines) + 8], fill=(0, 0, 0))
    y = 12
    for text, color in hud_lines:
        draw.text((16, y), text, font=fonts["hud"] if y == 12 else fonts["small"], fill=color)
        y += 26

    # 우상단 — 프레임/시간
    w = img.width
    sec = frame_idx / fps
    info = f"{int(sec // 60):02d}:{int(sec % 60):02d}  (frame {frame_idx})"
    iw = draw.textlength(info, font=fonts["small"])
    draw.rectangle([w - iw - 16, 8, w - 6, 32], fill=(0, 0, 0))
    draw.text((w - iw - 11, 11), info, font=fonts["small"], fill=(255, 255, 255))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="/workspace/prj_cctv/사고분석_설계/output/demo_3class_tracking.mp4")
    ap.add_argument("--duration", type=float, default=300.0, help="초 단위")
    ap.add_argument("--start", type=float, default=0.0, help="시작 위치(초)")
    ap.add_argument("--model", default="/workspace/prj_cctv/사고분석_설계/src/yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--fps", type=float, default=None, help="출력 fps (미지정 시 원본)")
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"영상 열기 실패: {args.input}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = args.fps or src_fps

    start_frame = int(args.start * src_fps)
    n_frames = int(args.duration * src_fps)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print(f"입력: {args.input}")
    print(f"  {w}x{h} @ {src_fps:.1f}fps | 시작 {args.start}s | {args.duration}s ({n_frames} 프레임)")
    print(f"출력: {args.output} @ {out_fps:.1f}fps")

    model = YOLO(args.model)
    fonts = load_fonts()

    # mp4v로 먼저 쓰고, 마지막에 ffmpeg로 H.264 재인코딩
    tmp_out = str(Path(args.output).with_suffix(".tmp.mp4"))
    writer = cv2.VideoWriter(tmp_out, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))

    track_classes: dict[int, Counter] = defaultdict(Counter)  # tid → 클래스 투표
    last_seen: dict[int, int] = {}                            # tid → 마지막 등장 프레임
    counted: set[int] = set()                                 # 집계 완료된 tid
    finished: Counter = Counter()                             # 최종 누적
    LOST_AFTER = int(src_fps * 1.5)                           # 1.5초 미등장 → 이탈 확정

    t0 = time.time()
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"  영상 끝 (프레임 {i})")
            break

        results = model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            conf=args.conf, classes=[2, 5, 7], verbose=False,
        )

        boxes_draw = []
        r = results[0]
        if r.boxes is not None and r.boxes.id is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), tid, c in zip(xyxy, ids, clss):
                if c not in COCO_3CLASS:
                    continue
                cls_name = COCO_3CLASS[c]
                track_classes[tid][cls_name] += 1
                last_seen[tid] = i
                # 현재 다수결 클래스로 표시 (프레임별 흔들림 안정화)
                disp = track_classes[tid].most_common(1)[0][0]
                boxes_draw.append((int(x1), int(y1), int(x2), int(y2), disp, tid))

        # 이탈 트랙 집계
        for tid, seen in list(last_seen.items()):
            if tid not in counted and (i - seen) > LOST_AFTER:
                final_cls = track_classes[tid].most_common(1)[0][0]
                finished[final_cls] += 1
                counted.add(tid)

        out_frame = draw_overlay_pil(frame, boxes_draw, finished, fonts, i, src_fps)
        writer.write(out_frame)

        if i % 500 == 0 and i > 0:
            el = time.time() - t0
            eta = el / i * (n_frames - i)
            print(f"  {i}/{n_frames} ({i/n_frames*100:.0f}%) | "
                  f"{el:.0f}s 경과, ETA {eta:.0f}s | 누적 {sum(finished.values())}대", flush=True)

    # 남은 활성 트랙도 집계 (영상 끝까지 남아있던 차량)
    for tid in last_seen:
        if tid not in counted:
            final_cls = track_classes[tid].most_common(1)[0][0]
            finished[final_cls] += 1

    cap.release()
    writer.release()

    print(f"\n최종 집계: 승용차 {finished.get('승용차',0)} | "
          f"버스 {finished.get('버스',0)} | 트럭 {finished.get('트럭',0)} | "
          f"합계 {sum(finished.values())}")

    # H.264 재인코딩 (호환성)
    import subprocess
    print("H.264 재인코딩...")
    res = subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_out, "-c:v", "libx264", "-preset", "medium",
         "-crf", "23", "-pix_fmt", "yuv420p", args.output],
        capture_output=True, text=True,
    )
    if res.returncode == 0:
        Path(tmp_out).unlink()
        print(f"완료: {args.output}")
    else:
        print(f"재인코딩 실패, tmp 보존: {tmp_out}")
        print(res.stderr[-500:])


if __name__ == "__main__":
    main()
