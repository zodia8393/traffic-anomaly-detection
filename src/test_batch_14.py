"""Step 3: 14건 배치 테스트 — 정상 7 + 비정상 7 유형별 1건.

V3: MLLM 싱글턴 + 클립별 try/except + 멀티프레임 입력.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/workspace/prj_cctv/pipeline/src")

from run_pipeline import run
from layer3_mllm.mllm_client import MLLMClient

BASE = Path("/DATA/aihub_71566/source/val")

CLIPS = [
    # 정상 7건
    ("N1", "정상/01.방향지시등 이행 차선변경"),
    ("N2", "정상/02.실선구간 정상주행"),
    ("N3", "정상/03.정상차로변경 주행"),
    ("N4", "정상/04.차선물기의 정상주행"),
    ("N5", "정상/05.2개 차로 연속변경의 정상주행"),
    ("N6", "정상/06.정체구간 정상주행"),
    ("N7", "정상/07.안전거리 확보 차선변경"),
    # 비정상 7건
    ("A1", "비정상/01.방향지시등 불이행"),
    ("A2", "비정상/02.실선구간 차선변경"),
    ("A3", "비정상/03.동시 차로변경"),
    ("A4", "비정상/04.차선 물기"),
    ("A5", "비정상/05.2개 차로 연속 변경"),
    ("A6", "비정상/06.정체구간 차선변경"),
    ("A7", "비정상/07.안전거리 미확보 차선변경"),
]

print("MLLM 모델 로드 (1회)...")
t_load = time.time()
mllm = MLLMClient(backend="transformers")
print(f"MLLM 로드 완료: {time.time() - t_load:.1f}s")

results = []

for label, subdir in CLIPS:
    cat_dir = BASE / subdir
    clip_dirs = sorted(cat_dir.iterdir())
    if not clip_dirs:
        print(f"SKIP {label}: no clips in {subdir}")
        continue
    clip = clip_dirs[0]
    video_id = f"{label}_{clip.name}"

    print(f"\n{'='*50}")
    print(f"[{label}] {subdir.split('/')[-1]}")
    print(f"  clip: {clip.name}")
    t0 = time.time()
    try:
        summary = run(clip, video_id, max_mllm_calls=1, force_trigger=True, mllm=mllm)
    except Exception as e:
        print(f"  ERROR: {e}")
        summary = {
            "video_id": video_id, "clip": clip.name, "frames": 0,
            "tracks": 0, "triggers": 0, "trigger_types": [],
            "mllm_calls": 0, "mllm_anomaly": [], "mllm_vehicles_real": [],
            "mllm_latency_avg": None, "error": str(e),
        }
    elapsed = time.time() - t0
    summary["label"] = label
    summary["category"] = subdir.split("/")[-1]
    summary["is_abnormal"] = label.startswith("A")
    summary["elapsed_sec"] = round(elapsed, 1)
    results.append(summary)

    anom_list = summary.get("mllm_anomaly", [])
    anomaly = anom_list[0] if anom_list else None
    veh_list = summary.get("mllm_vehicles_real", [])
    vehicles_ok = veh_list[0] if veh_list else False
    print(f"  anomaly={anomaly} vehicles_real={vehicles_ok} "
          f"triggers={summary['triggers']} elapsed={elapsed:.0f}s")

# 요약 테이블
print("\n" + "=" * 70)
print("배치 테스트 결과 요약 (14건)")
print("=" * 70)
print(f"{'Label':<5} {'Type':<6} {'Category':<30} {'Anomaly':<8} {'Veh OK':<7} {'Triggers'}")
print("-" * 70)
correct = 0
veh_ok_count = 0
for r in results:
    anom_list = r.get("mllm_anomaly", [])
    anomaly = anom_list[0] if anom_list else None
    veh_list = r.get("mllm_vehicles_real", [])
    veh_ok = veh_list[0] if veh_list else False
    is_abn = r["is_abnormal"]
    match = (anomaly is True and is_abn) or (anomaly is False and not is_abn)
    if match:
        correct += 1
    if veh_ok:
        veh_ok_count += 1
    mark = "O" if match else "X"
    print(f"{r['label']:<5} {'비정상' if is_abn else '정상':<6} "
          f"{r['category']:<30} {str(anomaly):<8} {str(veh_ok):<7} {r['triggers']}  {mark}")

n = len(results)
print("-" * 70)
print(f"판별 정확도: {correct}/{n} ({100*correct/n:.0f}%)")
print(f"Vehicles 실제값: {veh_ok_count}/{n} ({100*veh_ok_count/n:.0f}%)")
print(f"총 소요: {sum(r['elapsed_sec'] for r in results):.0f}s")

out = Path("/workspace/prj_cctv/사고분석_설계/src/layer1_vision/output/batch_14_results_v3.json")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)
print(f"\n결과 저장: {out}")
