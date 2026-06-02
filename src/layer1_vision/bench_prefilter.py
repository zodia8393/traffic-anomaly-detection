"""프리필터 처리시간 벤치마크 — '~1.5ms/프레임' 주장을 실측화.

PPT/제안서의 "프레임당 ~1.5ms (640x480)" 주장은 코드에 타이밍 측정이 없어
검증 불가였다. 이 스크립트로 실제 CPU 환경 처리시간을 측정해 근거를 만든다.

실행:
  python bench_prefilter.py                # 640x480, 500프레임
  python bench_prefilter.py --w 1920 --h 1080 --n 300
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from layer1_vision.prefilter import PreFilter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=60, help="MOG2 워밍업 프레임")
    args = ap.parse_args()

    pf = PreFilter()
    rng = np.random.default_rng(42)

    def make_frame():
        # 배경 + 움직이는 블롭 (현실적 전경 비율)
        f = rng.integers(40, 80, (args.h, args.w, 3), dtype=np.uint8)
        x = int(rng.integers(0, args.w - 80))
        y = int(rng.integers(0, args.h - 80))
        f[y:y+80, x:x+80] = rng.integers(150, 255)
        return f

    # 워밍업 (MOG2 배경 학습)
    for _ in range(args.warmup):
        pf.check(make_frame(), time.time())

    # 측정
    times = []
    for _ in range(args.n):
        fr = make_frame()
        t0 = time.perf_counter()
        pf.check(fr, time.time())
        times.append((time.perf_counter() - t0) * 1000)  # ms

    times.sort()
    p50 = statistics.median(times)
    p95 = times[int(len(times) * 0.95)]
    mean = statistics.mean(times)
    print(f"=== PreFilter 벤치마크 ({args.w}x{args.h}, {args.n}프레임, CPU) ===")
    print(f"  평균   : {mean:.2f} ms")
    print(f"  중앙값 : {p50:.2f} ms")
    print(f"  p95    : {p95:.2f} ms")
    print(f"  최소   : {times[0]:.2f} ms / 최대 {times[-1]:.2f} ms")
    print(f"  처리율 : {1000/mean:.0f} fps/코어")
    print(f"\n  → 제안서 '~1.5ms' 주장 대비: 실측 {mean:.2f}ms "
          f"({'부합' if mean <= 2.0 else '초과 — 수치 정정 필요'})")


if __name__ == "__main__":
    main()
