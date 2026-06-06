"""VLM 실시간 상황정보 추출 PoC 영상.

좌측: CCTV 영상 + 3종 차량 검출 박스 (yolov8n, 매 프레임)
우측: 'AI CCTV 상황 정보' 패널 — Qwen2.5-VL이 키프레임을 검출결과 grounding과
      함께 판독하여 도공 양식 구조화 필드를 타임스탬프 순으로 누적.

검출(빠름)과 VLM(키프레임만, ~28s/frame)을 단일 패스로 합성. 오프라인 렌더이므로
VLM 추론 중 타임라인이 멈춰도 무방 (프레임 순차 기록).

실행:
  python make_vlm_poc.py --input <mp4> --output <mp4> --duration 300 --start 1800 --interval 40
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from ultralytics import YOLO

COCO_3CLASS = {2: "승용차", 5: "버스", 7: "화물차"}
COLORS_RGB = {"승용차": (0, 204, 0), "버스": (0, 102, 255), "화물차": (255, 51, 51)}
FONT = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

PANEL_W = 560  # 우측 패널 폭

PROMPT_TMPL = """당신은 고속도로 CCTV 교통상황 분석 전문가입니다.
이 CCTV 구간은 {lanes}차로이다 (확정 정보, 변경 금지).
영상 검출 시스템이 측정한 데이터 (신뢰 가능):
- 화면 내 평균 차량: {avg_onscreen}대
- 차량 평균 이동속도: {motion:.1f} px/frame ({flow_hint})
- 직전 {win}초간 주요 통과 차종: {top_class}

소통상태 판단 기준 (이동속도 우선):
- 차량이 빠르게 이동(이동량 큼) → "원활" (대수가 많아도 흐르면 원활)
- 느리게 가다 서다 → "서행"
- 거의 정지 → "정체"

이 기준과 영상을 종합하여 JSON으로만 답하시오. 이상징후(사고/정차/역주행)는
영상에서 명백히 보일 때만 적고, 불확실하면 "없음"으로 하시오.
{{"기상":"맑음/흐림/비/눈/안개","소통상태":"원활/서행/정체",
"주요차종":"한 단어","이상징후":"없음/사고/정차/역주행","요약":"15자 내외 한 문장"}}"""


def load_fonts():
    return {
        "title": ImageFont.truetype(FONT, 20),
        "ts": ImageFont.truetype(FONT, 16),
        "key": ImageFont.truetype(FONT, 15),
        "val": ImageFont.truetype(FONT, 15),
        "label": ImageFont.truetype(FONT, 14),
        "hud": ImageFont.truetype(FONT, 16),
    }


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def run_vlm(model, proc, frame_bgr, grounding: dict) -> dict:
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    prompt = PROMPT_TMPL.format(**grounding)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": prompt},
    ]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    gen = out[0][inputs.input_ids.shape[1]:]
    resp = proc.decode(gen, skip_special_tokens=True)
    return parse_json(resp)


def draw_panel(panel_img, entries, fonts, processing=False):
    """우측 패널 렌더 (PIL). entries: [(ts_str, dict), ...]"""
    draw = ImageDraw.Draw(panel_img)
    draw.rectangle([0, 0, PANEL_W, panel_img.height], fill=(18, 22, 30))
    # 타이틀 바
    draw.rectangle([0, 0, PANEL_W, 46], fill=(0, 72, 140))
    draw.text((16, 12), "AI CCTV 상황 정보", font=fonts["title"], fill=(255, 255, 255))

    y = 60
    show = list(reversed(entries[-6:]))  # 최근 6개, 최신이 맨 위
    for idx, (ts, d) in enumerate(show):
        is_latest = idx == 0  # 맨 위 = 최신
        # 타임스탬프 헤더
        bar = (255, 192, 0) if is_latest else (90, 100, 115)
        draw.rectangle([12, y, PANEL_W - 12, y + 24], fill=(60, 52, 20) if is_latest else (40, 46, 58))
        draw.text((20, y + 3), f"{ts}", font=fonts["ts"], fill=bar)
        y += 30
        fields = [
            ("기상", d.get("기상", "-")),
            ("차로수", f"{d.get('차로수','-')}차로" if d.get("차로수") else "-"),
            ("소통상태", d.get("소통상태", "-")),
            ("주요차종", d.get("주요차종", "-")),
            ("이상징후", d.get("이상징후", "-")),
        ]
        for k, v in fields:
            vc = (255, 90, 90) if (k == "이상징후" and v not in ("없음", "-")) else (220, 226, 235)
            draw.text((28, y), f"{k}", font=fonts["key"], fill=(130, 150, 175))
            draw.text((120, y), str(v), font=fonts["val"], fill=vc)
            y += 21
        summ = d.get("요약", "")
        if summ:
            draw.text((28, y), f"“{summ}”", font=fonts["label"], fill=(160, 200, 240))
            y += 22
        y += 8
        if y > panel_img.height - 90:
            break

    if processing:
        py = panel_img.height - 34
        draw.rectangle([12, py, PANEL_W - 12, py + 24], fill=(60, 40, 10))
        draw.text((20, py + 3), "● AI 분석 중...", font=fonts["ts"], fill=(255, 200, 80))
    return panel_img


def draw_left(frame, boxes, cur_count, fonts, ts_str):
    """좌측 CCTV 프레임에 박스+라벨+상단 HUD (PIL)."""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for (x1, y1, x2, y2, cls, tid) in boxes:
        c = COLORS_RGB[cls]
        draw.rectangle([x1, y1, x2, y2], outline=c, width=2)
        lab = f"{cls}#{tid}"
        tw = draw.textlength(lab, font=fonts["label"])
        ly = max(0, y1 - 16)
        draw.rectangle([x1, ly, x1 + tw + 5, ly + 16], fill=c)
        draw.text((x1 + 2, ly + 1), lab, font=fonts["label"], fill=(255, 255, 255))
    # 상단 HUD
    draw.rectangle([0, 0, img.width, 26], fill=(0, 0, 0))
    draw.text((8, 4), f"경부선 원지동 CCTV  |  {ts_str}  |  화면 내 {cur_count}대",
              font=fonts["hud"], fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="/workspace/prj_cctv/사고분석_설계/output/demo_vlm_poc.mp4")
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--start", type=float, default=1800.0)
    ap.add_argument("--interval", type=float, default=40.0, help="VLM 키프레임 간격(초)")
    ap.add_argument("--lanes", type=int, default=5, help="CCTV 구간 차로수 (확정 메타데이터)")
    ap.add_argument("--model", default="/workspace/prj_cctv/사고분석_설계/src/yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fonts = load_fonts()

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_f = int(args.start * fps)
    n_frames = int(args.duration * fps)
    interval_f = int(args.interval * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    out_W = W + PANEL_W
    tmp = str(Path(args.output).with_suffix(".tmp.mp4"))
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_W, H))

    print(f"입력 {W}x{H}@{fps:.0f} | {args.duration}s | VLM 간격 {args.interval}s "
          f"(~{n_frames//interval_f}회)", flush=True)

    print("YOLO 로드...", flush=True)
    det = YOLO(args.model)
    print("Qwen2.5-VL 로드...", flush=True)
    t0 = time.time()
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, device_map="cpu")
    vproc = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256*28*28, max_pixels=512*28*28)
    print(f"VLM 로드 {time.time()-t0:.0f}s", flush=True)

    track_cls = defaultdict(Counter)
    win_stats = Counter()           # 윈도우 통과 차종
    win_centroids = defaultdict(list)  # tid -> [(cx,cy) per frame] (윈도우 한정)
    win_onscreen = []               # 프레임별 화면 내 대수
    last_seen = {}
    entries = []                    # [(ts_str, dict)]
    panel = Image.new("RGB", (PANEL_W, H))
    panel_bgr = cv2.cvtColor(np.array(draw_panel(panel.copy(), [], fonts)), cv2.COLOR_RGB2BGR)

    tstart = time.time()
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"영상 끝 {i}", flush=True)
            break

        r = det.track(frame, persist=True, tracker="bytetrack.yaml",
                      conf=args.conf, classes=[2, 5, 7], verbose=False)[0]
        boxes = []
        cur = 0
        if r.boxes is not None and r.boxes.id is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), tid, c in zip(xyxy, ids, cls):
                if c not in COCO_3CLASS:
                    continue
                name = COCO_3CLASS[c]
                track_cls[tid][name] += 1
                last_seen[tid] = i
                disp = track_cls[tid].most_common(1)[0][0]
                boxes.append((int(x1), int(y1), int(x2), int(y2), disp, tid))
                cur += 1
                win_stats[disp] += 1
                win_centroids[tid].append(((x1 + x2) / 2, (y1 + y2) / 2))
        win_onscreen.append(cur)

        sec = (start_f + i) / fps
        ts_str = f"{int(sec//3600):02d}:{int(sec%3600//60):02d}:{int(sec%60):02d}"

        # VLM 키프레임
        if i > 0 and i % interval_f == 0:
            # 분석중 표시 프레임 1장 먼저
            proc_panel = cv2.cvtColor(
                np.array(draw_panel(panel.copy(), entries, fonts, processing=True)),
                cv2.COLOR_RGB2BGR)
            # 차량 이동량: 각 트랙 연속프레임 centroid 변위의 중앙값
            speeds = []
            for cents in win_centroids.values():
                if len(cents) >= 2:
                    d = [((cents[k+1][0]-cents[k][0])**2 + (cents[k+1][1]-cents[k][1])**2)**0.5
                         for k in range(len(cents)-1)]
                    speeds.append(sum(d) / len(d))
            # p75: 원근감으로 느려보이는 원거리 차량 배제, '흐를 수 있는가'를 근거리 기준 판단
            motion = float(np.percentile(speeds, 75)) if speeds else 0.0
            flow_hint = "빠름=원활" if motion >= 6 else ("느림=서행" if motion >= 2.5 else "거의정지=정체")
            avg_on = round(sum(win_onscreen) / max(1, len(win_onscreen)), 1)
            top = win_stats.most_common(1)[0][0] if win_stats else "승용차"
            grounding = {
                "avg_onscreen": avg_on, "motion": motion, "flow_hint": flow_hint,
                "win": int(args.interval), "top_class": top, "lanes": args.lanes,
            }
            t_v = time.time()
            result = run_vlm(vlm, vproc, frame, grounding)
            if result:
                result["차로수"] = args.lanes  # 확정 메타데이터로 강제
                entries.append((ts_str, result))
            panel_bgr = cv2.cvtColor(
                np.array(draw_panel(panel.copy(), entries, fonts)), cv2.COLOR_RGB2BGR)
            print(f"  [{ts_str}] VLM {time.time()-t_v:.0f}s | motion {motion:.1f}px "
                  f"({flow_hint}) → {result.get('소통상태','?')}/{result.get('이상징후','?')}",
                  flush=True)
            win_stats.clear()
            win_centroids.clear()
            win_onscreen.clear()

        left = draw_left(frame, boxes, cur, fonts, ts_str)
        composite = np.hstack([left, panel_bgr])
        writer.write(composite)

        if i % 600 == 0 and i > 0:
            el = time.time() - tstart
            print(f"  {i}/{n_frames} ({i*100//n_frames}%) {el:.0f}s", flush=True)

    cap.release()
    writer.release()

    print(f"\n총 VLM 분석 {len(entries)}회", flush=True)
    import subprocess
    print("H.264 재인코딩...", flush=True)
    res = subprocess.run(
        ["ffmpeg", "-y", "-i", tmp, "-c:v", "libx264", "-preset", "medium",
         "-crf", "23", "-pix_fmt", "yuv420p", args.output],
        capture_output=True, text=True)
    if res.returncode == 0:
        Path(tmp).unlink()
        print(f"완료: {args.output}", flush=True)
    else:
        print(f"재인코딩 실패: {res.stderr[-400:]}", flush=True)


if __name__ == "__main__":
    main()
