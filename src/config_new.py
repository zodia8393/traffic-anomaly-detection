"""사고분석 시스템 설정.

기존 파이프라인 config.py를 import하고, 사고분석 전용 설정을 추가한다.
"""

from pathlib import Path
import os
import sys

# 루트 경로 — 환경변수로 오버라이드 가능 (배포 이식성). 미설정 시 기본값.
ROOT = Path(os.environ.get("CCTV_PRJ_ROOT", "/workspace/prj/cctv"))
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

# CPU 추론 최적화 (i9-285K 24코어). 엔트리포인트 OMP_NUM_THREADS=4가 torch intra-op을
# 4코어로 묶어 MLLM forward가 24코어 중 4개만 사용 → 명시 상향. 단발성 호출(시간당 ~12회)
# 이라 16스레드 점유가 안전. 환경변수로 오버라이드.
MLLM_TORCH_THREADS = int(os.environ.get("MLLM_TORCH_THREADS", "16"))
# 비전토큰 상한 (지연 직결). 720x480 CCTV는 native 유지, 고해상 카메라만 다운스케일.
# Qwen2.5-VL: 28x28 패치당 1토큰, 2x2 병합. max_pixels로 patch 수 캡.
MLLM_MAX_PIXELS = int(os.environ.get("MLLM_MAX_PIXELS", str(768 * 768)))   # ≈ 589k px
MLLM_MIN_PIXELS = int(os.environ.get("MLLM_MIN_PIXELS", str(256 * 256)))
# 결정성: temp가 이 값 이하면 greedy(do_sample=False)로 — 0.1은 사실상 greedy 의도.
MLLM_GREEDY_TEMP_THRESHOLD = 0.3
# 태스크별 출력 토큰 상한 (512 하드캡 제거 — 93컬럼 보고서 잘림 방지)
MLLM_MAX_NEW_TOKENS = {"accident": 512, "scene": 384, "classify": 384,
                       "report": 2048, "outbreak": 1024, "default": 768}

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

# Shadow mode: 운영 녹화/MLLM 트리거는 기존 TriggerDetector 경로를 유지하고,
# AnomalyEngine 점수와 규칙 위반만 JSONL로 수집한다. 실알림 전환은 별도 검증 후 활성화.
ENABLE_ANOMALY_SHADOW = os.environ.get("ENABLE_ANOMALY_SHADOW", "1") == "1"
ANOMALY_SHADOW_LOG_DIR = Path(os.environ.get(
    "ANOMALY_SHADOW_LOG_DIR", str(NEW_ROOT / "output" / "anomaly_shadow"),
))
ANOMALY_SHADOW_MIN_SCORE = float(os.environ.get("ANOMALY_SHADOW_MIN_SCORE", "0.30"))
ANOMALY_PROMOTE_TO_TRIGGER = os.environ.get("ANOMALY_PROMOTE_TO_TRIGGER", "0") == "1"
ANOMALY_PROMOTE_MIN_STREAK = int(os.environ.get("ANOMALY_PROMOTE_MIN_STREAK", "3"))
ANOMALY_PROMOTE_COOLDOWN_SEC = float(os.environ.get("ANOMALY_PROMOTE_COOLDOWN_SEC", "60"))
ANOMALY_PROMOTE_MIN_SCORE = float(os.environ.get("ANOMALY_PROMOTE_MIN_SCORE", "0.85"))
ANOMALY_ALLOW_ML_ONLY_PROMOTION = os.environ.get("ANOMALY_ALLOW_ML_ONLY_PROMOTION", "0") == "1"

# an1 특화모델은 차선변경+깜빡이 후보가 명확할 때만 점수 반영한다.
AN1_MIN_TRACK_POINTS = int(os.environ.get("AN1_MIN_TRACK_POINTS", "8"))
AN1_MIN_LATERAL_TOTAL = float(os.environ.get("AN1_MIN_LATERAL_TOTAL", "0.08"))
AN1_MIN_AMBER_HISTORY = int(os.environ.get("AN1_MIN_AMBER_HISTORY", "12"))
AN1_SCORE_COOLDOWN_SEC = float(os.environ.get("AN1_SCORE_COOLDOWN_SEC", "20"))
AN1_EMA_ALPHA = float(os.environ.get("AN1_EMA_ALPHA", "0.45"))
AN1_SCORE_ON_THRESHOLD = float(os.environ.get("AN1_SCORE_ON_THRESHOLD", "0.75"))

# T8 보행자 감지는 차량 경로와 분리한다. ROI 확정 전 운영 기본값은 shadow/비활성이다.
ENABLE_T8_PEDESTRIAN = os.environ.get("ENABLE_T8_PEDESTRIAN", "0") == "1"
T8_PEDESTRIAN_CONF = float(os.environ.get("T8_PEDESTRIAN_CONF", "0.25"))

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

# ── 외부 RAG 지식베이스 연계 (DB 파일 플러그인) ───────────────────────
# 외부 협업자가 구축한 RAG DB(사고/차단시간 지식)를 파일로 받아 즉시 결합.
# RAG_DB_PATH에 파일이 존재하면 자동 활성(없으면 안전 무동작 → similar_cases=[]).
RAG_DB_PATH = os.environ.get("RAG_DB_PATH", str(DATA_DIR / "rag_knowledge.duckdb"))
# 청크 스키마 매핑 (JHJ RAG 호환 기본값, 협업자 스키마 다르면 env로 오버라이드)
RAG_CHUNKS_TABLE = os.environ.get("RAG_CHUNKS_TABLE", "chunks")
RAG_EMBED_TABLE = os.environ.get("RAG_EMBED_TABLE", "chunk_embeddings")
RAG_DOCS_TABLE = os.environ.get("RAG_DOCS_TABLE", "documents")
# 쿼리 임베더 — DB 임베딩과 모델 일치 필요. 불가 시 키워드 전용으로 자동 강등(DB만 있어도 작동).
RAG_EMBED_PROVIDER = os.environ.get("RAG_EMBED_PROVIDER", "auto")  # auto|ollama|sbert|none
RAG_OLLAMA_URL = os.environ.get("RAG_OLLAMA_URL", "http://localhost:11434")
RAG_EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "")            # 미지정 시 DB에서 자동 감지
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "3"))
RAG_RRF_K = int(os.environ.get("RAG_RRF_K", "60"))                 # RRF 융합 상수(스케일 무관)
RAG_CACHE_TTL_SEC = int(os.environ.get("RAG_CACHE_TTL_SEC", "300"))  # 임베딩 행렬 캐시 갱신주기
# 관련도 게이트 — 키워드/벡터 신호가 전혀 없는 청크는 반환 금지(무관 사례 주입 방지)
RAG_SIM_FLOOR = float(os.environ.get("RAG_SIM_FLOOR", "0.25"))     # 벡터 cosine 최소 관련도
# 쿼리 임베더는 DB 임베딩 모델과 정확히 일치할 때만 벡터검색(차원우연 cross-space 방지)
RAG_REQUIRE_EXACT_MODEL = os.environ.get("RAG_REQUIRE_EXACT_MODEL", "1") == "1"
RAG_MAX_CHUNKS = int(os.environ.get("RAG_MAX_CHUNKS", "500000"))   # 캐시 행렬 상한(OOM 방지)

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
