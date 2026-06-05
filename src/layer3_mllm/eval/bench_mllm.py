"""MLLM 평가 하네스 — 지연·비전토큰·메모리·출력신뢰성·환각(FP) 측정.

워크플로우 감사(P1) 후속. 단발 벤치를 메트릭 적재형 재사용 모듈로 승격.
실제 코드경로(MLLMClient + AccidentDetectorMLLM)를 그대로 사용하므로 설정/프롬프트
변경이 그대로 반영된다 → before/after 회귀 비교에 사용.

측정 항목:
  - latency_sec        : 태스크별 추론 지연 (콜드 + 반복)
  - prompt_tokens      : 입력 토큰(비전토큰 포함) — vision_token 제어 효과 확인
  - completion_tokens  : 출력 토큰
  - peak_rss_gb        : 피크 메모리
  - json_parsed        : JSON 파싱 성공 여부 (str fallback이면 False)
  - deterministic      : 2회 greedy 출력이 동일한가 (재현성)
  - false_positive     : 정상 장면 + 가짜 트리거에서 사고로 오판했는가 (환각 지표)

실행:
  python -m layer3_mllm.eval.bench_mllm --label after --video <mp4>
  python -m layer3_mllm.eval.bench_mllm --compare before after
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import cv2

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

RESULT_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_VIDEO = "/DATA/cctv_recording/20260528/경인선_도당2_20260528_141545.mp4"


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB→GB (linux)


def extract_keyframes(video: str, n: int, start: float, gap: float):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((start + i * gap) * fps))
        ret, f = cap.read()
        if ret:
            frames.append(f)
    cap.release()
    return frames, (W, H)


def run_bench(label: str, video: str, n_frames: int = 3,
              start: float = 60.0, gap: float = 1.0, repeats: int = 2) -> dict:
    from layer3_mllm.mllm_client import MLLMClient
    from layer3_mllm.accident_detector import AccidentDetectorMLLM

    frames, (W, H) = extract_keyframes(video, n_frames, start, gap)
    print(f"[bench:{label}] 키프레임 {len(frames)}장, {W}x{H}")

    t0 = time.perf_counter()
    client = MLLMClient(backend="transformers")
    load_sec = time.perf_counter() - t0
    detector = AccidentDetectorMLLM(client)
    print(f"[bench:{label}] 모델로드 {load_sec:.1f}초, RSS {peak_rss_gb():.1f}GB, "
          f"torch_threads={__import__('torch').get_num_threads()}")

    # 정상 장면 + 가짜 트리거 → 사고로 오판하면 환각(FP). 시각근거 우선 프롬프트면 false 기대.
    trigger = {"trigger_type": "T1_TTC", "ttc_values": [1.8, 1.5],
               "speed_changes": [{"track_id": 3, "delta": -12}]}
    tracks = {"track_summaries": [
        {"track_id": 3, "class": "승용차", "avg_speed": 88},
        {"track_id": 7, "class": "화물차", "avg_speed": 72},
    ], "speed_calibrated": True}

    runs, reasonings = [], []
    for r in range(repeats):
        t = time.perf_counter()
        res = detector.detect(frames, trigger, tracks)
        dt = time.perf_counter() - t
        usage = res.get("raw_response", {}).get("usage", {})
        content = res.get("raw_response", {}).get("content")
        runs.append({
            "run": r, "latency_sec": round(dt, 1),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "json_parsed": not isinstance(content, str),
            "accident_detected": res.get("accident_detected"),
            "confidence": res.get("confidence"),
            "parse_error": res.get("parse_error", False),
        })
        reasonings.append(json.dumps(res.get("involved_vehicles", [])) +
                          f"|{res.get('accident_detected')}|{res.get('accident_type')}")
        print(f"[bench:{label}] run{r}: {dt:.1f}초 tok={usage.get('prompt_tokens')}→"
              f"{usage.get('completion_tokens')} parsed={runs[-1]['json_parsed']} "
              f"detected={res.get('accident_detected')} conf={res.get('confidence')}")

    deterministic = len(set(reasonings)) == 1 if len(reasonings) > 1 else None
    # 정상 장면이므로 detected=True는 false positive(환각)
    fp = any(rr["accident_detected"] for rr in runs)

    out = {
        "label": label,
        "config": {
            "model": client.model_name, "n_frames": n_frames, "resolution": f"{W}x{H}",
            "torch_threads": __import__("torch").get_num_threads(),
        },
        "model_load_sec": round(load_sec, 1),
        "peak_rss_gb": round(peak_rss_gb(), 1),
        "avg_latency_sec": round(sum(rr["latency_sec"] for rr in runs) / len(runs), 1),
        "avg_prompt_tokens": round(sum(rr["prompt_tokens"] or 0 for rr in runs) / len(runs)),
        "deterministic": deterministic,
        "false_positive_normal_scene": fp,
        "runs": runs,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / f"{label}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[bench:{label}] 저장 → {RESULT_DIR / f'{label}.json'}")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def compare(a: str, b: str) -> None:
    da = json.loads((RESULT_DIR / f"{a}.json").read_text())
    db = json.loads((RESULT_DIR / f"{b}.json").read_text())
    print(f"\n=== 비교: {a} → {b} ===")
    rows = [
        ("평균지연(초)", da["avg_latency_sec"], db["avg_latency_sec"], "↓"),
        ("프롬프트토큰", da["avg_prompt_tokens"], db["avg_prompt_tokens"], "↓"),
        ("피크RSS(GB)", da["peak_rss_gb"], db["peak_rss_gb"], "↓"),
        ("torch_threads", da["config"]["torch_threads"], db["config"]["torch_threads"], "↑"),
        ("결정성", da["deterministic"], db["deterministic"], ""),
        ("정상장면FP", da["false_positive_normal_scene"], db["false_positive_normal_scene"], ""),
    ]
    for name, va, vb, good in rows:
        delta = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va:
            pct = (vb - va) / va * 100
            delta = f"  ({pct:+.0f}%)"
        print(f"  {name:14s}: {va} → {vb}{delta}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--start", type=float, default=60.0)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return
    run_bench(args.label, args.video, args.frames, args.start, repeats=args.repeats)


if __name__ == "__main__":
    main()
