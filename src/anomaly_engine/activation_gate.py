"""ML 모델 활성화 게이트 — 미검증/저성능 모델의 사고판정 오염 차단.

설계만 되고 미배선된 Level 2~4 모델이 실수로 켜져 사고판정을 왜곡하는 것을
구조적으로 막는다. 각 모델은 전제조건(파일 존재 + AUROC ≥ 임계)을 통과해야
앙상블 투표에 참여할 수 있다.

특히 STGAE(Level 4)는 AUROC≈0.495(랜덤 수준)이므로 명시적으로 격리한다.

사용:
  from anomaly_engine.activation_gate import active_ml_models
  enabled = active_ml_models()   # 통과한 모델 key 집합 (예: {"level2_iforest"})
  # ENABLE_ML_ENSEMBLE=False면 항상 빈 집합
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_new import ENABLE_ML_ENSEMBLE, ML_MIN_AUROC, ML_MODEL_GATES

logger = logging.getLogger(__name__)


def _best_auroc(eval_path: Path | None) -> float | None:
    """eval json에서 AUROC 추출 (두 형식 지원, 없으면 None = 검사 생략).

    형식1(STGAE): {epoch: {"auroc": x, ...}, ...} → 최고값
    형식2(지도감지기): {"auroc": x, ...} → 그 값
    """
    if eval_path is None or not Path(eval_path).exists():
        return None
    try:
        data = json.loads(Path(eval_path).read_text())
        # 형식2: 최상위에 auroc
        if "auroc" in data and isinstance(data["auroc"], (int, float)):
            return float(data["auroc"])
        # 형식1: epoch별 dict
        aurocs = [v["auroc"] for v in data.values()
                  if isinstance(v, dict) and "auroc" in v]
        return max(aurocs) if aurocs else None
    except Exception as e:  # noqa: BLE001
        logger.warning("AUROC 파싱 실패 %s: %s", eval_path, e)
        return None


def gate_report() -> dict[str, dict]:
    """각 모델의 게이트 통과 여부와 사유를 반환 (진단용)."""
    report: dict[str, dict] = {}
    for key, cfg in ML_MODEL_GATES.items():
        path = Path(cfg["path"])
        reasons = []
        ok = True
        if not cfg.get("enabled", False):
            ok = False
            reasons.append("config disabled")
        if not path.exists():
            ok = False
            reasons.append(f"model missing: {path.name}")
        auroc = _best_auroc(cfg.get("eval"))
        if auroc is not None and auroc < ML_MIN_AUROC:
            ok = False
            reasons.append(f"AUROC {auroc:.3f} < {ML_MIN_AUROC}")
        report[key] = {"active": ok, "auroc": auroc, "reasons": reasons or ["ok"]}
    return report


def active_ml_models() -> set[str]:
    """전제조건을 통과해 앙상블에 참여 가능한 ML 모델 key 집합.

    ENABLE_ML_ENSEMBLE=False면 빈 집합(규칙 Level1만 동작).
    """
    if not ENABLE_ML_ENSEMBLE:
        return set()
    report = gate_report()
    active = {k for k, r in report.items() if r["active"]}
    blocked = {k: r["reasons"] for k, r in report.items() if not r["active"]}
    if blocked:
        logger.info("ML 게이트 차단: %s", blocked)
    logger.info("ML 활성 모델: %s", active or "(없음)")
    return active


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(gate_report(), ensure_ascii=False, indent=2, default=str))
    print("활성:", active_ml_models())
