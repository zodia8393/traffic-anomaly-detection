"""Level 2 — Isolation Forest 기반 이상 탐지.

AI Hub #71566 라벨에서 추출한 클립별 특성벡터로 학습.
정상 클립만으로 학습 → 비정상 클립에서 높은 anomaly score 기대.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

from .aihub_loader import load_clips, extract_clip_features, FEATURE_NAMES_CLIP


def build_dataset(label_root: Path, max_clips: int | None = None):
    """클립 로딩 → 특성벡터 추출 → X, y 반환."""
    print(f"클립 로딩: {label_root}")
    clips = load_clips(label_root, max_clips=max_clips)
    print(f"  총 {len(clips)}개 클립 로딩 완료")

    X, y, clip_ids, subtypes = [], [], [], []
    for clip in clips:
        feat = extract_clip_features(clip)
        if np.all(feat == 0):
            continue
        X.append(feat)
        y.append(0 if clip.is_normal else 1)  # 0=정상, 1=비정상
        clip_ids.append(clip.clip_id)
        subtypes.append(clip.subtype)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"  유효 클립: {len(X)} (정상: {(y==0).sum()}, 비정상: {(y==1).sum()})")
    return X, y, clip_ids, subtypes


def train_and_evaluate(
    label_root: Path,
    model_dir: Path,
    max_clips: int | None = None,
    contamination: float = 0.05,
):
    """학습 + 평가 파이프라인."""
    t0 = time.time()

    X, y, clip_ids, subtypes = build_dataset(label_root, max_clips=max_clips)
    if len(X) == 0:
        print("유효 데이터 없음")
        return

    # 정상 데이터만으로 학습
    X_normal = X[y == 0]
    print(f"\n학습 데이터: 정상 {len(X_normal)}개 클립")

    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_normal_scaled)
    print(f"  Isolation Forest 학습 완료 ({time.time()-t0:.1f}s)")

    # 전체 데이터 평가
    X_scaled = scaler.transform(X)
    scores = model.decision_function(X_scaled)
    preds = model.predict(X_scaled)  # 1=정상, -1=이상

    # sklearn IForest: 1=inlier, -1=outlier → 변환
    y_pred = (preds == -1).astype(int)  # 1=이상

    print(f"\n=== 평가 결과 ===")
    print(classification_report(
        y, y_pred,
        target_names=["정상", "비정상"],
        digits=3,
    ))

    if len(np.unique(y)) > 1:
        auc = roc_auc_score(y, -scores)
        print(f"ROC-AUC: {auc:.3f}")

    # 비정상 서브타입별 탐지율
    print("\n=== 비정상 서브타입별 탐지율 ===")
    from collections import Counter
    subtype_stats: dict[str, dict] = {}
    for i, st in enumerate(subtypes):
        if y[i] == 0:
            continue
        if st not in subtype_stats:
            subtype_stats[st] = {"total": 0, "detected": 0}
        subtype_stats[st]["total"] += 1
        if y_pred[i] == 1:
            subtype_stats[st]["detected"] += 1

    for st, stats in sorted(subtype_stats.items(), key=lambda x: -x[1]["detected"]/max(x[1]["total"],1)):
        rate = stats["detected"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {st}: {stats['detected']}/{stats['total']} ({rate:.1f}%)")

    # 특성 중요도 (anomaly score 상관)
    print("\n=== 특성 중요도 (|상관계수|) ===")
    for j, fname in enumerate(FEATURE_NAMES_CLIP):
        if X[:, j].std() > 0:
            corr = abs(np.corrcoef(X[:, j], -scores)[0, 1])
            print(f"  {fname}: {corr:.3f}")

    # 모델 저장
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "iforest_l2.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    print(f"\n모델 저장: {model_dir / 'iforest_l2.pkl'}")
    print(f"총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    label_root = Path("/DATA/aihub_71566/labels/val_extracted")
    model_dir = Path("/workspace/prj_cctv/사고분석_설계/models")
    train_and_evaluate(label_root, model_dir)
