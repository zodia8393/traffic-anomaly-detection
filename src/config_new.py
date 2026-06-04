"""사고분석 시스템 설정.

기존 파이프라인 config.py를 import하고, 사고분석 전용 설정을 추가한다.
"""

from pathlib import Path
import os
import sys

# 루트 경로 — 환경변수로 오버라이드 가능 (배포 이식성). 미설정 시 기본값.
ROOT = Path(os.environ.get("CCTV_PRJ_ROOT", "/workspace/prj_cctv"))
NEW_ROOT = Path(os.environ.get("CCTV_ANALYSIS_ROOT", str(ROOT / "사고분석_설계")))

# 기존 파이프라인 경로 추가
PIPELINE_SRC = Path(os.environ.get("CCTV_PIPELINE_SRC", str(ROOT / "pipeline" / "src")))
sys.path.insert(0, str(PIPELINE_SRC))
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
TRIGGER_TTC_THRESHOLD = 2.0        # 초 (v3: 3.0 → v4: 2.0)
TRIGGER_DECEL_THRESHOLD = -3.0     # m/s^2
TRIGGER_STOP_SPEED = 5.0           # km/h
TRIGGER_STOP_DURATION = 10.0       # 초
TRIGGER_MULTI_DECEL_COUNT = 3      # 대
TRIGGER_SPEED_VAR_SIGMA = 2.0      # 배수
TRIGGER_PERIODIC_INTERVAL = 300    # 초 (5분)

# 트리거 쿨다운
TRIGGER_COOLDOWN_SEC = 60.0        # 동일 유형 재발 무시 시간 (v3: 30 → v4: 60)
TRIGGER_TTC_MIN = 0.5              # TTC < 이 값은 bbox 아티팩트 무시 (v3: 0.3 → v4: 0.5)
TRIGGER_TTC_STREAK_MIN = 4         # 연속 N프레임 유지 시에만 T1 발화 (v3: 2 → v4: 4)

# 다중 트리거 합의
CONSENSUS_WINDOW_SEC = 5.0         # 합의 윈도우 (초)
CONSENSUS_MIN_TYPES = 2            # 최소 트리거 종류 수

# ── MLLM 설정 ────────────────────────────────────────────────────────
MLLM_BACKEND = "transformers"      # "llama_cpp" | "openai_api" | "transformers"
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

# ── ML 앙상블 활성화 게이트 (Level 2~4 + 지도감지기) ───────────────────
# 미검증/저성능 모델이 사고판정을 오염시키지 않도록 전제조건 통과 모델만 투표.
ENABLE_ML_ENSEMBLE = True            # 지도감지기(AUROC 0.918) 통과로 활성화
ML_MIN_AUROC = 0.60                  # 활성화 최소 AUROC (미달 시 격리)
# 각 레벨: (모델파일, eval파일(있으면 AUROC 검사), 기본활성)
ML_MODEL_GATES = {
    # 지도 사고감지기: 궤적 운동학특징 RandomForest, AUROC 0.918 (STGAE 0.49 대체)
    "supervised_traj": {"path": MODEL_DIR / "supervised_detector.pkl",
                        "eval": MODEL_DIR / "supervised_detector.json", "enabled": True},
    # an1 특화: 차선변경 차량 깜빡이 검출, AUROC 0.89 (방향지시등 불이행 전용)
    "an1_specialist": {"path": MODEL_DIR / "an1_specialist.pkl",
                       "eval": MODEL_DIR / "an1_specialist.json", "enabled": True},
    "level2_iforest": {"path": MODEL_DIR / "iforest_l2.pkl", "eval": None, "enabled": False},
    "level3_lstm":    {"path": MODEL_DIR / "lstm_ae_l3.pt",  "eval": None, "enabled": False},
    # STGAE: AUROC 0.495(랜덤) — 명시적 격리. 지도감지기로 대체됨.
    "level4_stgae":   {"path": MODEL_DIR / "stgae_aihub",
                       "eval": MODEL_DIR / "stgae_aihub" / "eval_summary.json", "enabled": False},
}

# ── 미배선 모듈 활성화 플래그 (실수 활성화 방지) ──────────────────────
ENABLE_LAYER4_PREDICTION = False     # XGBoost 사고예측 — 학습데이터/모델 준비 후 True
ENABLE_RAG = False                   # RAG 보강 — 미배선
ENABLE_SELF_TRAINING = False         # 자기강화 루프 — 미배선

# ── 전국 스케일업 (3-Tier) ────────────────────────────────────────────
NATIONWIDE_MAX_CAMERAS = 1600        # 전체 가동 기본 상한 (RAM 40GB 기준, ITS 4760대 중)
TIER2_HOTSPOT_COUNT = 50             # Tier 2 핫스팟 상시 정밀 감시 대수
TIER3_MAX_CONCURRENT = 50            # Tier 3 동시 정밀 분석 최대 대수
STREAM_SAMPLE_FPS = 1                # 전체 스트림 분석 fps
STREAM_URL_REFRESH_SEC = 3600        # HLS URL 갱신 주기 (1시간)
STREAM_RECONNECT_DELAY_SEC = 10      # 스트림 끊김 시 재연결 대기
STREAM_MAX_FAILURES = 15             # 연속 실패 시 스트림 비활성화 (5→15, ITS HLS 간헐적 불안정 대응)
STREAM_REPROBE_SEC = 1800            # 비활성 카메라 재탐지 주기 (30분)

# 프리필터 (Tier 1)
PREFILTER_FG_RATIO_LOW = 0.02        # MOG2 전경 비율 하한 (전체 정지)
PREFILTER_FG_RATIO_HIGH = 0.40       # MOG2 전경 비율 상한 (급변)
PREFILTER_FRAME_DIFF_THRESHOLD = 30  # 프레임 차분 평균 픽셀값 임계
PREFILTER_HIST_DIFF_THRESHOLD = 0.4  # 히스토그램 상관계수 하한 (급변)
PREFILTER_STATIC_DURATION_SEC = 10   # 정지 객체 지속 시간 (초)
PREFILTER_COOLDOWN_SEC = 30          # 동일 카메라 Tier 3 승격 쿨다운
PREFILTER_TIER3_HOLD_SEC = 60        # Tier 3 유지 시간 (정밀 분석 후 복귀)

# ITS 사고 연동
INCIDENT_POLL_SEC = 60               # ITS 돌발상황 폴링 주기
INCIDENT_REACTOR_CCTVS = 3           # 사고 시 편입할 인근 CCTV 수
INCIDENT_HOLD_SEC = 600              # 사고 편입 CCTV 유지 시간 (10분)

# ── 돌발정보 CCTV 캡처 ──────────────────────────────────────────────
OUTBREAK_DWELL_SEC = 30              # CCTV 체류 시간 (초)
OUTBREAK_DWELL_EXTEND_SEC = 60       # 트리거 발화 시 연장 (초)
OUTBREAK_CAPTURE_FPS = 1.0           # 체류 중 캡처 fps
OUTBREAK_CCTV_DELAY = 5.0            # CCTV 전환 간 대기 (초)
OUTBREAK_INCIDENT_DELAY = 3.0        # 사고 전환 간 대기 (초)
OUTBREAK_POLL_INTERVAL = 300         # 전체 스캔 주기 (초)
OUTBREAK_MAX_PER_CYCLE = 20          # 사이클당 최대 체류 CCTV 수
OUTBREAK_MAX_PER_INCIDENT = 2        # 사고당 최대 CCTV 수
OUTBREAK_ENABLED = True              # 활성화 플래그
OUTBREAK_RECORDING_DIR = OUTPUT_DIR / "outbreak_recordings"

# ── 돌발정보 MLLM 분석 ─────────────────────────────────────────────
OUTBREAK_MLLM_ENABLED = True             # MLLM 분석 활성화
OUTBREAK_MLLM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"  # CPU 추론용 3B
OUTBREAK_MLLM_MAX_IMAGES = 5             # MLLM에 전달할 대표 프레임 수
OUTBREAK_MLLM_QUEUE_SIZE = 50            # 분석 큐 최대 크기

# ── DuckDB ───────────────────────────────────────────────────────────
DUCKDB_PATH = str(DATA_DIR / "accident.duckdb")
