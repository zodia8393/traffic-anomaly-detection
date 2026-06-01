"""XGBoost 사고 확률 예측."""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
import logging
from pathlib import Path

import numpy as np

from config_new import L4_MODELS, L4_EVALUATIONS

logger = logging.getLogger(__name__)

# 위험도 레벨 기준
_RISK_THRESHOLDS: list[tuple[float, str]] = [
    (0.7, "critical"),
    (0.5, "high"),
    (0.3, "medium"),
]
_RISK_DEFAULT = "low"


def _classify_risk(probability: float) -> str:
    """확률값을 위험도 레벨 문자열로 변환."""
    for threshold, level in _RISK_THRESHOLDS:
        if probability >= threshold:
            return level
    return _RISK_DEFAULT


class AccidentPredictor:
    """XGBoost 기반 사고 확률 예측 모델."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model = None
        self.feature_names: list[str] | None = None
        self._default_model_dir = L4_MODELS

        if model_path and model_path.exists():
            self.load(model_path)

    # ------------------------------------------------------------------
    # 학습
    # ------------------------------------------------------------------
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        **xgb_params,
    ) -> dict:
        """XGBoost 모델 학습.

        클래스 불균형을 scale_pos_weight로 자동 보정하며,
        5-fold CV + early stopping을 수행한다.

        Args:
            X: (n_samples, n_features) 학습 피처.
            y: (n_samples,) 이진 라벨 (0/1).
            feature_names: 피처 이름 리스트.
            **xgb_params: XGBoost 파라미터 오버라이드.

        Returns:
            5-fold CV 결과 {"cv_auc_mean", "cv_auc_std", "best_iteration"}.
        """
        import xgboost as xgb

        self.feature_names = feature_names

        # 클래스 불균형 보정
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        default_spw = n_neg / max(n_pos, 1)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "scale_pos_weight": default_spw,
            "tree_method": "hist",
            "random_state": 42,
            "verbosity": 0,
        }
        params.update(xgb_params)

        # n_estimators를 꺼내 num_boost_round로 전환 (CV용)
        n_estimators = params.pop("n_estimators", 200)
        random_state = params.pop("random_state", 42)

        dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)

        # 5-fold CV + early stopping
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=n_estimators,
            nfold=5,
            seed=random_state,
            early_stopping_rounds=20,
            verbose_eval=False,
        )

        best_iteration = len(cv_results)
        cv_auc_mean = float(cv_results["test-auc-mean"].iloc[-1])
        cv_auc_std = float(cv_results["test-auc-std"].iloc[-1])

        logger.info("CV 완료: AUC=%.4f (+-%.4f), best_iter=%d",
                     cv_auc_mean, cv_auc_std, best_iteration)

        # 전체 데이터로 최종 모델 학습
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=best_iteration,
        )

        return {
            "cv_auc_mean": cv_auc_mean,
            "cv_auc_std": cv_auc_std,
            "best_iteration": best_iteration,
        }

    # ------------------------------------------------------------------
    # 예측
    # ------------------------------------------------------------------
    def predict(self, features: dict) -> dict:
        """단일 예측.

        Args:
            features: 피처명->값 딕셔너리.

        Returns:
            {"probability": float, "risk_level": str}.
            Layer4 미활성(ENABLE_LAYER4_PREDICTION=False) 또는 모델 미로드 시
            안전 폴백(probability=0, risk_level="unavailable")을 반환한다.
        """
        import xgboost as xgb

        try:
            from config_new import ENABLE_LAYER4_PREDICTION
        except Exception:  # noqa: BLE001
            ENABLE_LAYER4_PREDICTION = False

        # 미활성/미로드 시 크래시 대신 안전 폴백 (실수 배선 방지)
        if not ENABLE_LAYER4_PREDICTION or self.model is None:
            return {"probability": 0.0, "risk_level": "unavailable"}

        if self.feature_names:
            vec = [float(features.get(fn, 0.0)) for fn in self.feature_names]
        else:
            vec = list(features.values())

        dmat = xgb.DMatrix(
            np.array([vec], dtype=np.float64),
            feature_names=self.feature_names,
        )
        prob = float(self.model.predict(dmat)[0])

        return {
            "probability": round(prob, 4),
            "risk_level": _classify_risk(prob),
        }

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """배치 예측.

        Args:
            X: (n_samples, n_features) 피처 행렬.

        Returns:
            (n_samples,) 사고 확률 배열.

        Raises:
            RuntimeError: 모델 미로드 시.
        """
        import xgboost as xgb

        if self.model is None:
            raise RuntimeError("모델이 로드되지 않음. load() 또는 train() 먼저 호출")

        dmat = xgb.DMatrix(X, feature_names=self.feature_names)
        return self.model.predict(dmat)

    # ------------------------------------------------------------------
    # 평가
    # ------------------------------------------------------------------
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """테스트셋 평가.

        Args:
            X_test: (n_samples, n_features) 테스트 피처.
            y_test: (n_samples,) 이진 라벨.

        Returns:
            {"auc", "precision", "recall", "f1", "confusion_matrix"}.
        """
        from sklearn.metrics import (
            confusion_matrix,
            precision_recall_fscore_support,
            roc_auc_score,
        )

        probs = self.predict_batch(X_test)
        preds = (probs >= 0.5).astype(int)

        auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.0
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="binary", zero_division=0,
        )
        cm = confusion_matrix(y_test, preds).tolist()

        results = {
            "auc": round(float(auc), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "confusion_matrix": cm,
        }
        logger.info("평가: AUC=%.4f, P=%.4f, R=%.4f, F1=%.4f",
                     results["auc"], results["precision"],
                     results["recall"], results["f1"])
        return results

    # ------------------------------------------------------------------
    # 저장 / 로드
    # ------------------------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        """모델 + 피처명 저장.

        Args:
            path: 저장 디렉토리. None이면 MODEL_DIR/xgb_accident/.

        Returns:
            저장된 디렉토리 경로.
        """
        if self.model is None:
            raise RuntimeError("저장할 모델이 없음")

        save_dir = Path(path) if path else self._default_model_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        model_file = save_dir / "model.json"
        self.model.save_model(str(model_file))

        meta_file = save_dir / "meta.json"
        meta = {"feature_names": self.feature_names}
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info("모델 저장: %s", save_dir)
        return save_dir

    def load(self, path: Path) -> None:
        """모델 로드.

        Args:
            path: 모델 디렉토리 또는 model.json 파일 경로.
        """
        import xgboost as xgb

        path = Path(path)
        if path.is_dir():
            model_file = path / "model.json"
            meta_file = path / "meta.json"
        else:
            model_file = path
            meta_file = path.parent / "meta.json"

        if not model_file.exists():
            raise FileNotFoundError(f"모델 파일 없음: {model_file}")

        self.model = xgb.Booster()
        self.model.load_model(str(model_file))

        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.feature_names = meta.get("feature_names")

        logger.info("모델 로드: %s (피처 %d개)",
                     model_file, len(self.feature_names or []))

    # ------------------------------------------------------------------
    # 피처 중요도
    # ------------------------------------------------------------------
    def feature_importance(self, top_k: int = 10) -> list[tuple[str, float]]:
        """피처 중요도 상위 k개.

        Args:
            top_k: 반환할 피처 수.

        Returns:
            [(피처명, 중요도), ...] 중요도 내림차순.

        Raises:
            RuntimeError: 모델 미로드 시.
        """
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않음")

        importance = self.model.get_score(importance_type="gain")

        if not importance:
            return []

        # 정규화 (합계=1)
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_imp[:top_k]

    def save_evaluation(self, eval_results: dict) -> Path:
        """평가 결과를 L4_EVALUATIONS에 JSON 저장."""
        from datetime import datetime
        L4_EVALUATIONS.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fpath = L4_EVALUATIONS / f"eval_{ts}.json"
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)
        logger.info("평가 결과 저장 → %s", fpath)
        return fpath
