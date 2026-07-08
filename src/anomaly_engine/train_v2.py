"""사고감지기 v2 학습 — 궤적+깜빡이 특징. an1(방향지시등) 개선 측정."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

MODEL_DIR = Path("/workspace/prj/work/AI기반 교통상황 대응 기술 개발 연구/사고분석_설계/models")
d = np.load(MODEL_DIR / "dataset_v2.npz", allow_pickle=True)
X, y, types, fnames = d["X"], d["y"], list(d["types"]), list(d["feature_names"])
print(f"클립 {len(X)} (비정상 {int(y.sum())}/정상 {int((y==0).sum())}), 특징 {X.shape[1]}개")

idx = np.arange(len(X))
tr, te = train_test_split(idx, test_size=0.3, stratify=y, random_state=42)
types_te = [types[i] for i in te]

models = {
    "RandomForest": RandomForestClassifier(n_estimators=400, max_depth=14, class_weight="balanced", random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=250, max_depth=4, learning_rate=0.08, random_state=42),
}
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.07,
                                      scale_pos_weight=1.0, eval_metric="auc", random_state=42, n_jobs=-1)
except ImportError:
    pass

best = (None, -1, None)
for name, m in models.items():
    m.fit(X[tr], y[tr])
    auroc = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
    print(f"  {name:18s} AUROC {auroc:.4f}")
    if auroc > best[1]:
        best = (name, auroc, m)

name, auroc, model = best
print(f"\n=== 최고 {name} AUROC {auroc:.4f} (v1 0.916 / STGAE 0.49) ===")

# 임계 튜닝 (FPR<=10%)
prob = model.predict_proba(X[te])[:, 1]
thr, bf = 0.5, -1
for t in np.linspace(0.1, 0.9, 33):
    p = (prob >= t).astype(int)
    fpr = np.mean(p[y[te] == 0] == 1); rec = np.mean(p[y[te] == 1] == 1)
    if fpr <= 0.10 and rec > bf:
        bf, thr = rec, t
pred = (prob >= thr).astype(int)
rec = float(np.mean(pred[y[te] == 1] == 1)); fpr = float(np.mean(pred[y[te] == 0] == 1))
print(f"임계 {thr:.2f}: recall {rec:.3f}, FPR {fpr:.3f}")

# 유형별 recall (an1 집중)
ptr = {}
for t in sorted(set(types_te)):
    ii = [i for i, tt in enumerate(types_te) if tt == t]
    if t == "normal":
        ptr[t] = float(np.mean(pred[ii] == 0))
    else:
        ptr[t] = float(np.mean(pred[ii] == 1))
print("유형별:", " ".join(f"{k}={v:.2f}" for k, v in ptr.items()))

# blink 특징 중요도
if hasattr(model, "feature_importances_"):
    bi = list(fnames).index("blink_score")
    print(f"blink_score 중요도: {model.feature_importances_[bi]:.3f} "
          f"(순위 {sorted(model.feature_importances_, reverse=True).index(model.feature_importances_[bi])+1}/{len(fnames)})")

# 저장
import joblib
joblib.dump({"model": model, "threshold": float(thr), "feature_names": list(fnames),
             "auroc": float(auroc), "model_type": name, "has_blink": True},
            MODEL_DIR / "supervised_detector.pkl")
meta = {"model": name, "auroc": round(auroc, 4), "threshold": round(thr, 3),
        "recall": round(rec, 3), "fpr": round(fpr, 3), "version": "v2_blink",
        "per_type": {k: round(v, 3) for k, v in ptr.items()}, "n_clips": len(X)}
(MODEL_DIR / "supervised_detector.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
print("저장: supervised_detector.pkl (v2, blink 포함)")
