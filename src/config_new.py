"""사고분석 시스템 설정.

기존 파이프라인 config.py를 import하고, 사고분석 전용 설정을 추가한다.
"""

from pathlib import Path
import sys

# 기존 파이프라인 경로 추가
PIPELINE_SRC = Path("/workspace/prj_cctv/pipeline/src")
sys.path.insert(0, str(PIPELINE_SRC))

ROOT = Path("/workspace/prj_cctv")
NEW_ROOT = ROOT / "사고분석_설계"
SRC_DIR = NEW_ROOT / "src"
DATA_DIR = NEW_ROOT / "data"
MODEL_DIR = NEW_ROOT / "models"
OUTPUT_DIR = NEW_ROOT / "output"

# ── 레이어별 결과물 경로 ─────────────────────────────────────────────
L1_OUTPUT = SRC_DIR / "layer1_vision" / "output"
L1_KEYFRAMES = L1_OUTPUT / "keyframes"      # 핵심 장면 이미지
L1_PACKAGES = L1_OUTPUT / "packages"        # MLLM 입력 패키지 JSON

L2_OUTPUT = SRC_DIR / "layer2_metadata" / "output"
L2_EXPORTS = L2_OUTPUT / "exports"          # DB 덤프/내보내기

L3_OUTPUT = SRC_DIR / "layer3_mllm" / "output"
L3_SCENES = L3_OUTPUT / "scenes"            # 장면 분석 결과
L3_ACCIDENTS = L3_OUTPUT / "accidents"      # 사고 감지 결과
L3_CORRECTIONS = L3_OUTPUT / "corrections"  # 차종 보정 결과
L3_REPORTS = L3_OUTPUT / "reports"          # 93컬럼 보고서

L4_OUTPUT = SRC_DIR / "layer4_prediction" / "output"
L4_MODELS = L4_OUTPUT / "models"            # XGBoost 모델
L4_EVALUATIONS = L4_OUTPUT / "evaluations"  # 평가 결과
L4_RISK_REPORTS = L4_OUTPUT / "risk_reports" # 위험도 순위

# 13종 분류 (도공 전체)
VEHICLE_13CLASS = {
    "T1": "승용차", "T2": "버스", "T3": "소형화물", "T4": "중형화물",
    "T5": "대형화물", "T6": "대형특수4축", "T7": "대형특수5축",
    "T8": "세미트레일러4축", "T9": "풀트레일러4축", "T10": "세미트레일러5축",
    "T11": "풀트레일러5축", "T12": "세미트레일러6축", "T13": "이륜차",
}

# ── 트리거 임계값 ────────────────────────────────────────────────────
TRIGGER_TTC_THRESHOLD = 3.0        # 초
TRIGGER_DECEL_THRESHOLD = -3.0     # m/s^2
TRIGGER_STOP_SPEED = 5.0           # km/h
TRIGGER_STOP_DURATION = 3.0        # 초
TRIGGER_MULTI_DECEL_COUNT = 3      # 대
TRIGGER_SPEED_VAR_SIGMA = 2.0      # 배수
TRIGGER_PERIODIC_INTERVAL = 300    # 초 (5분)

# 트리거 쿨다운
TRIGGER_COOLDOWN_SEC = 30.0        # 동일 유형 재발 무시 시간

# ── MLLM 설정 ────────────────────────────────────────────────────────
MLLM_BACKEND = "llama_cpp"         # "llama_cpp" | "openai_api"
MLLM_MODEL_PATH = str(MODEL_DIR / "mllm" / "qwen2.5-vl-7b-q4.gguf")
MLLM_API_URL = "http://localhost:8080/v1"
MLLM_MAX_TOKENS = 2048
MLLM_TEMPERATURE = 0.1

# ── 키프레임 ─────────────────────────────────────────────────────────
KEYFRAME_MAX = 5
KEYFRAME_MIN_INTERVAL_SEC = 1.0

# ── 속도 추정 ────────────────────────────────────────────────────────
SPEED_HOMOGRAPHY_DEFAULT = None    # IC별 calibrations/ 에서 로드
SPEED_FALLBACK_SCALE = 0.05        # 캘리브 없을 때 m/px

# ── 이상징후 탐지 엔진 ───────────────────────────────────────────────
ANOMALY_ENGINE_DIR = SRC_DIR / "anomaly_engine"
ANOMALY_RULES_PATH = ANOMALY_ENGINE_DIR / "rules_default.yaml"
ANOMALY_LOG_DIR = NEW_ROOT / "output" / "anomaly_logs"
ANOMALY_CAMERA_CONFIGS = NEW_ROOT / "configs" / "cameras"
ANOMALY_ALERT_THRESHOLD = 0.3
ANOMALY_ALARM_THRESHOLD = 0.7

# ── DuckDB ───────────────────────────────────────────────────────────
DUCKDB_PATH = str(DATA_DIR / "accident.duckdb")
