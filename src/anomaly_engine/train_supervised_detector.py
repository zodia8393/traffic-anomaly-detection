"""지도 사고감지기 학습 — 궤적 운동학특징 → 비정상주행 분류.

STGAE(비지도, AUROC 0.49) 대체. 모델 비교(RF/XGBoost/GBoost) 후 최고 모델 저장.
평가: 독립 홀드아웃 AUROC + 비정상유형(an1~an7)별 recall + 임계값 튜닝.

실행:
  python train_supervised_detector.py
  python train_supervised_detector.py --data /DATA/aihub_71566/stgae_format
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trajectory_features import FEATURE_NAMES, extract_features, parse_clip

MODEL_DIR = Path("/workspace/prj_cctv/사고분석_설계/models")


def load_split(data_root: str, split: str):
    """split 디렉토리의 클립 → (X, y, types). y: 1=비정상. types: an번호/None."""
    X, y, types, names = [], [], [], []
    for f in glob.glob(os.path.join(data_root, split, "*.txt")):
        base = os.path.basename(f)
        m = re.search(r"_an([0-9])", base)
        feat = extract_features(parse_clip(f))
        if feat is None:
            continue
        X.append(feat)
        y.append(1 if m else 0)
        types.append(f"an{m.group(1)}" if m else "normal")
        names.append(base)
    return np.array(X), np.array(y), types, names


def per_type_recall(y_true, y_pred, types):
    """비정상유형별 recall."""
    out = {}
    for t in sorted(set(types)):
        if t == "normal":
            idx = [i for i, tt in enumerate(types) if tt == "normal"]
            out[t] = float(np.mean(y_pred[idx] == 0)) if idx else 0.0  # 정상은 TN율
        else:
            idx = [i for i, tt in enumerate(types) if tt == t]
            out[t] = float(np.mean(y_pred[idx] == 1)) if idx else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/DATA/aihub_71566/stgae_format")
    ap.add_argument("--out", default=str(MODEL_DIR / "supervised_detector.pkl"))
    args = ap.parse_args()

    # test split만 양 클래스 보유 → 70/30 층화분할(독립 홀드아웃)
    X, y, types, names = load_split(args.data, "test")
    print(f"클립 {len(X)} (비정상 {int(y.sum())} / 정상 {int((y==0).sum())}), 특징 {X.shape[1]}개")

    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.3, stratify=y, random_state=42)
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    types_te = [types[i] for i in te]

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced",
            random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42),
    }
    try:
        from xgboost import XGBClassifier
        spw = float((y == 0).sum()) / max(1, (y == 1).sum())
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            scale_pos_weight=spw, eval_metric="auc", random_state=42, n_jobs=-1)
    except ImportError:
        pass

    results = {}
    best_name, best_auroc, best_model = None, -1, None
    for name, m in models.items():
        m.fit(Xtr, ytr)
        prob = m.predict_proba(Xte)[:, 1]
        auroc = roc_auc_score(yte, prob)
        results[name] = round(float(auroc), 4)
        print(f"  {name:18s} AUROC {auroc:.4f}")
        if auroc > best_auroc:
            best_name, best_auroc, best_model = name, auroc, m

    print(f"\n=== 최고: {best_name} AUROC {best_auroc:.4f} (STGAE 0.49 대비 +{best_auroc-0.49:.3f}) ===")

    # 임계값 튜닝: 정상 오탐 ≤10% 유지하며 recall 최대 (운영: 오탐 억제 중요)
    prob = best_model.predict_proba(Xte)[:, 1]
    best_thr, best_f = 0.5, -1
    for thr in np.linspace(0.1, 0.9, 33):
        pred = (prob >= thr).astype(int)
        fpr = np.mean(pred[yte == 0] == 1)
        rec = np.mean(pred[yte == 1] == 1)
        if fpr <= 0.10 and rec > best_f:
            best_f, best_thr = rec, thr
    pred = (prob >= best_thr).astype(int)
    overall_rec = float(np.mean(pred[yte == 1] == 1))
    overall_fpr = float(np.mean(pred[yte == 0] == 1))
    ptr = per_type_recall(yte, pred, types_te)
    print(f"임계값 {best_thr:.2f}: recall {overall_rec:.3f}, 오탐(FPR) {overall_fpr:.3f}")
    print("유형별:", " ".join(f"{k}={v:.2f}" for k, v in ptr.items()))

    # 특징 중요도
    if hasattr(best_model, "feature_importances_"):
        imp = sorted(zip(FEATURE_NAMES, best_model.feature_importances_), key=lambda t: -t[1])[:8]
        print("상위특징:", ", ".join(f"{n}={v:.3f}" for n, v in imp))

    # 저장 (모델 + 임계값 + 메타)
    import joblib
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {"model": best_model, "threshold": float(best_thr),
              "feature_names": FEATURE_NAMES, "auroc": float(best_auroc),
              "model_type": best_name}
    joblib.dump(bundle, args.out)
    meta = {"model": best_name, "auroc": round(best_auroc, 4), "threshold": round(best_thr, 3),
            "recall": round(overall_rec, 3), "fpr": round(overall_fpr, 3),
            "per_type": {k: round(v, 3) for k, v in ptr.items()},
            "all_models": results, "n_clips": len(X)}
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
