"""통합 데모 — 개선 MLLM + 방향별 교통량 카운터를 한 영상에.

좌: CCTV + 차량검출(박스·ID) + 방향궤적 + 유효계수선
우(패널):
  [교통량 측정] 계수선 통과 누적 — 서울방향/부산방향 × 차종별 (실시간)
  [AI 분석]     개선 MLLMClient(16스레드·greedy) 장면분석 로그 (최신 상단)

계수 설계(놓침 최소화): POI 전체프레임 검출 + 유효계수영역(near-mid)에 수평 계수선.
트랙 centroid가 계수선을 넘는 순간 1회 계수, dy 부호로 방향(상=서울 / 하=부산).
소실점 원거리는 계수 제외(검지 불안정 구간) — 표준 지점검지 방식.

실행:
  python demo_combined.py --input <mp4> --start 46800 --seconds 40 --interval 15 \
    --output output/demo_combined.mp4
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from ultralytics import YOLO
from layer3_mllm.mllm_client import MLLMClient

FONT = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
YOLO_MODEL = str(SRC / "yolov8n.pt")
COCO = {2: "승용차", 3: "이륜차", 5: "버스", 7: "화물차"}
COLOR = {"승용차": (0, 204, 0), "이륜차": (255, 200, 0), "버스": (0, 102, 255), "화물차": (255, 51, 51)}
PANEL_W = 600
IMGSZ = 736

SCENE_PROMPT = """당신은 고속도로 CCTV 교통상황 분석 전문가입니다.
이 구간은 편도 5차로이다(확정).
영상 검출 시스템 측정값(신뢰):
- 화면 내 평균 차량 {avg_on}대
- 차량 이동량 {motion:.1f} px/frame ({hint})
- 직전 {win}초 계수: 서울방향 {seoul}대, 부산방향 {busan}대

영상과 위 데이터를 종합하여 JSON으로만 답하시오. 이상징후(사고/정차/역주행)는
영상에서 명백히 보일 때만, 불확실하면 "없음".
{{"기상":"맑음/흐림/비/눈/안개","소통상태":"원활/서행/정체","주요차종":"한 단어",
"이상징후":"없음/사고/정차/역주행","요약":"15자 내외"}}"""


def fonts():
    return {k: ImageFont.truetype(FONT, s) for k, s in
            {"title": 22, "sec": 18, "ts": 15, "key": 14, "val": 15, "lab": 13, "hud": 16}.items()}


def run_scene(client, frame_bgr, g):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    msgs = [{"role": "system", "content": "당신은 교통 CCTV 분석 전문가입니다. JSON으로만 답하시오."},
            {"role": "user", "content": SCENE_PROMPT.format(**g)}]
    res = client.chat(msgs, images=[img], max_tokens=160)
    c = res.get("content")
    return (c if isinstance(c, dict) else {}), res.get("latency_sec", 0)


def draw_panel(H, traffic, entries, F, processing=False):
    p = Image.new("RGB", (PANEL_W, H), (18, 22, 30))
    d = ImageDraw.Draw(p)
    # ── 교통량 섹션 ──
    d.rectangle([0, 0, PANEL_W, 40], fill=(0, 90, 60))
    d.text((14, 9), "교통량 측정 (계수선 통과 누적)", font=F["title"], fill=(255, 255, 255))
    y = 50
    for dirname, color in [("서울방향", (90, 200, 255)), ("부산방향", (120, 255, 140))]:
        cc = traffic[dirname]
        tot = sum(cc.values())
        d.rectangle([12, y, PANEL_W - 12, y + 26], fill=(30, 44, 40))
        d.text((20, y + 4), f"{dirname}", font=F["sec"], fill=color)
        d.text((180, y + 4), f"총 {tot}대", font=F["sec"], fill=(255, 255, 255))
        y += 30
        cls_line = "  ".join(f"{k} {cc[k]}" for k in ("승용차", "버스", "화물차", "이륜차") if cc[k])
        d.text((28, y), cls_line or "-", font=F["val"], fill=(210, 216, 225))
        y += 26
    y += 6
    d.line([12, y, PANEL_W - 12, y], fill=(60, 70, 85), width=1)
    y += 10
    # ── AI 분석 섹션 ──
    d.rectangle([0, y, PANEL_W, y + 38], fill=(0, 72, 140))
    d.text((14, y + 8), "AI 분석 (Qwen2.5-VL)", font=F["title"], fill=(255, 255, 255))
    y += 48
    for idx, (ts, e) in enumerate(reversed(entries[-4:])):
        latest = idx == 0
        d.rectangle([12, y, PANEL_W - 12, y + 22],
                    fill=(60, 52, 20) if latest else (40, 46, 58))
        d.text((20, y + 2), ts, font=F["ts"], fill=(255, 192, 0) if latest else (150, 160, 175))
        y += 27
        for k in ("기상", "소통상태", "주요차종", "이상징후"):
            v = e.get(k, "-")
            vc = (255, 90, 90) if (k == "이상징후" and v not in ("없음", "-")) else (220, 226, 235)
            d.text((28, y), k, font=F["key"], fill=(130, 150, 175))
            d.text((120, y), str(v), font=F["val"], fill=vc)
            y += 19
        if e.get("요약"):
            d.text((28, y), f"“{e['요약']}”", font=F["lab"], fill=(160, 200, 240))
            y += 20
        y += 6
        if y > H - 60:
            break
    if processing:
        py = H - 30
        d.rectangle([12, py, PANEL_W - 12, py + 22], fill=(60, 40, 10))
        d.text((20, py + 2), "● AI 분석 중...", font=F["ts"], fill=(255, 200, 80))
    return cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR)


def draw_left(frame, boxes, traj, y_line, F, ts, onscreen, total_counted):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    # 방향 궤적
    for tid, (pts, color) in traj.items():
        if len(pts) >= 2:
            d.line([tuple(map(int, p)) for p in pts[-20:]], fill=color, width=2)
    # 박스 + ID
    for (x1, y1, x2, y2, cls, tid) in boxes:
        c = COLOR.get(cls, (200, 200, 200))
        d.rectangle([x1, y1, x2, y2], outline=c, width=2)
        lab = f"{cls}#{tid}"
        tw = d.textlength(lab, font=F["lab"])
        ly = max(0, y1 - 15)
        d.rectangle([x1, ly, x1 + tw + 4, ly + 15], fill=c)
        d.text((x1 + 2, ly + 1), lab, font=F["lab"], fill=(255, 255, 255))
    # 계수선 (유효계수영역)
    d.line([0, y_line, img.width, y_line], fill=(255, 80, 80), width=2)
    d.text((6, y_line - 18), "계수선", font=F["lab"], fill=(255, 120, 120))
    # HUD
    d.rectangle([0, 0, img.width, 26], fill=(0, 0, 0))
    d.text((8, 4), f"경부선 원지동 CCTV  |  {ts}  |  화면 {onscreen}대  |  누적계수 {total_counted}대",
           font=F["hud"], fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=str(SRC.parent / "output" / "demo_combined.mp4"))
    ap.add_argument("--start", type=float, default=46800)
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--interval", type=float, default=15, help="MLLM 분석 간격(초)")
    ap.add_argument("--out-fps", type=float, default=15)
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    F = fonts()
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * fps))
    n_src = int(args.seconds * fps)
    interval_f = int(args.interval * fps)
    y_line = int(H * 0.60)   # 유효계수영역(near-mid)

    out_W = W + PANEL_W
    tmp = str(Path(args.output).with_suffix(".tmp.mp4"))
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), args.out_fps, (out_W, H))
    write_stride = max(1, int(round(fps / args.out_fps)))

    print(f"입력 {W}x{H}@{fps:.0f} | {args.seconds}s | 계수선 y={y_line} | MLLM {args.interval}s", flush=True)
    det = YOLO(YOLO_MODEL)
    print("MLLMClient(개선판) 로드...", flush=True)
    t0 = time.time()
    client = MLLMClient(backend="transformers")
    import torch
    print(f"MLLM 로드 {time.time()-t0:.0f}s, threads={torch.get_num_threads()}", flush=True)

    traffic = {"서울방향": Counter(), "부산방향": Counter()}
    traj = {}                          # tid -> ([pts], color)
    track_cls = defaultdict(Counter)
    prev_cy = {}
    counted = set()
    win = {"on": [], "seoul": 0, "busan": 0}
    entries = []
    panel = draw_panel(H, traffic, [], F)

    tracker_yaml = _tracker_yaml()
    tstart = time.time()
    for i in range(n_src):
        ret, frame = cap.read()
        if not ret:
            break
        r = det.track(frame, persist=True, tracker=tracker_yaml, conf=0.20, iou=0.5,
                      imgsz=IMGSZ, classes=[2, 3, 5, 7], agnostic_nms=True, verbose=False)[0]
        boxes, on = [], 0
        if r.boxes is not None and r.boxes.id is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), tid, c in zip(xyxy, ids, cls):
                name = COCO.get(int(c), "기타")
                track_cls[tid][name] += 1
                disp = track_cls[tid].most_common(1)[0][0]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                boxes.append((int(x1), int(y1), int(x2), int(y2), disp, int(tid)))
                on += 1
                # 계수선 통과 판정
                if tid in prev_cy and tid not in counted:
                    pcy = prev_cy[tid]
                    if (pcy < y_line <= cy) or (pcy >= y_line > cy):
                        direction = "부산방향" if cy > pcy else "서울방향"  # 하행=부산, 상행=서울
                        traffic[direction][disp] += 1
                        counted.add(tid)
                        win["seoul" if direction == "서울방향" else "busan"] += 1
                # 궤적(방향 색)
                col = (90, 200, 255) if (tid in prev_cy and cy < prev_cy[tid]) else (120, 255, 140)
                if tid not in traj:
                    traj[tid] = ([], col)
                traj[tid][0].append((cx, cy))
                traj[tid] = (traj[tid][0], col)
                prev_cy[tid] = cy
        win["on"].append(on)
        sec = (int(args.start * fps) + i) / fps
        ts = f"{int(sec//3600):02d}:{int(sec%3600//60):02d}:{int(sec%60):02d}"

        # MLLM 분석
        if i > 0 and i % interval_f == 0:
            proc_panel = draw_panel(H, traffic, entries, F, processing=True)
            speeds = []
            for tid, (pts, _) in traj.items():
                if len(pts) >= 2:
                    dd = [np.hypot(pts[k+1][0]-pts[k][0], pts[k+1][1]-pts[k][1]) for k in range(len(pts)-1)]
                    speeds.append(np.mean(dd))
            motion = float(np.percentile(speeds, 75)) if speeds else 0.0
            hint = "빠름=원활" if motion >= 6 else ("느림=서행" if motion >= 2.5 else "거의정지=정체")
            g = {"avg_on": round(np.mean(win["on"]), 1) if win["on"] else 0, "motion": motion,
                 "hint": hint, "win": int(args.interval), "seoul": win["seoul"], "busan": win["busan"]}
            scene, lat = run_scene(client, frame, g)
            if scene:
                entries.append((ts, scene))
            panel = draw_panel(H, traffic, entries, F)
            print(f"  [{ts}] MLLM {lat:.0f}s motion {motion:.1f}({hint}) → "
                  f"{scene.get('소통상태','?')}/{scene.get('이상징후','?')} | "
                  f"누적 서울 {sum(traffic['서울방향'].values())} 부산 {sum(traffic['부산방향'].values())}", flush=True)
            win = {"on": [], "seoul": 0, "busan": 0}
        else:
            panel = draw_panel(H, traffic, entries, F)

        if i % write_stride == 0:
            left = draw_left(frame, boxes, traj, y_line, F, ts, on,
                             sum(traffic['서울방향'].values()) + sum(traffic['부산방향'].values()))
            writer.write(np.hstack([left, panel]))

    cap.release(); writer.release()
    total = sum(traffic['서울방향'].values()) + sum(traffic['부산방향'].values())
    print(f"\n총 계수 {total}대 | 서울 {dict(traffic['서울방향'])} | 부산 {dict(traffic['부산방향'])}", flush=True)
    print(f"MLLM 분석 {len(entries)}회 | 처리 {time.time()-tstart:.0f}s", flush=True)

    import subprocess
    print("H.264 재인코딩...", flush=True)
    res = subprocess.run(["ffmpeg", "-y", "-i", tmp, "-c:v", "libx264", "-preset", "medium",
                          "-crf", "23", "-pix_fmt", "yuv420p", args.output], capture_output=True, text=True)
    if res.returncode == 0:
        Path(tmp).unlink()
        print(f"완료: {args.output}", flush=True)
    else:
        print(f"재인코딩 실패: {res.stderr[-300:]}", flush=True)


def _tracker_yaml():
    import tempfile
    cfg = "tracker_type: bytetrack\ntrack_high_thresh: 0.25\ntrack_low_thresh: 0.1\nnew_track_thresh: 0.25\ntrack_buffer: 120\nmatch_thresh: 0.8\nfuse_score: True\n"
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    f.write(cfg); f.close()
    return f.name


if __name__ == "__main__":
    main()
