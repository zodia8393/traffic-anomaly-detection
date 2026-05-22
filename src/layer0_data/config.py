"""데이터 수집 공통 설정."""
from pathlib import Path

COLLECTOR_ROOT = Path(__file__).resolve().parent
DATA_ROOT = COLLECTOR_ROOT.parent.parent / "data"

# ── 저장 경로 (HDD 7.3TB) ───────────────────────────────────────────
HDD_ROOT = Path("/DATA/aihub_71566")
EXPANSION = Path("/media/ybs/Expansion/CCTV차종분류/accident_data")
BRIDGE_DIR = EXPANSION / "bridge"      # Track 1
STREAM_DIR = EXPANSION / "stream"      # Track 3

# ── API 키 (.env에서 로드) ───────────────────────────────────────────
# /workspace/.env 에 추가:
#   ROADPLUS_API_KEY=...  (data.ex.co.kr 교통량 데이터)
#   ITS_API_KEY=...       (its.go.kr 돌발상황+CCTV)

# ── Track 1: 공개 데이터셋 ───────────────────────────────────────────
AIHUB_DATASET_ID = 175          # AI Hub "교통사고 CCTV 영상"
AIHUB_71566_ID = 71566          # AI Hub "국도 CCTV 비정상주행 판별"
AIHUB_71566_DIR = HDD_ROOT
DOTA_URL = "https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly"

# ── AI Hub #71566 파일 맵 (filekey → filename) ───────────────────────
AIHUB_71566_FILES: dict[str, list[dict]] = {
    "train_source": [
        {"filekey": k, "filename": f"TS.z{i:02d}", "size_gb": 100}
        for i, k in enumerate(range(551679, 551696), start=1)
    ] + [{"filekey": 551696, "filename": "TS.zip", "size_gb": 57}],
    "train_label": [
        {"filekey": 551697, "filename": "TL.zip", "size_gb": 5},
    ],
    "val_source": [
        {"filekey": 551698, "filename": "VS.z01", "size_gb": 100},
        {"filekey": 551699, "filename": "VS.z02", "size_gb": 100},
        {"filekey": 551700, "filename": "VS.zip", "size_gb": 24},
    ],
    "val_label": [
        {"filekey": 551701, "filename": "VL.zip", "size_gb": 1},
    ],
}

# ── Track 3: API 엔드포인트 (2026-05-14 연결 확인) ────────────────────
# 1) 도로공사 교통 데이터 — data.ex.co.kr, 쿼리 key=  (교통량/VMS/AVC, 돌발정보 없음)
EXKOR_BASE = "http://data.ex.co.kr/openapi"
# 2) ITS 돌발상황정보 + CCTV — its.go.kr, 쿼리 apiKey=  (핵심 채널)
ITS_BASE = "https://openapi.its.go.kr:9443"

# ── Ring Buffer 설정 (legacy — OnDemandRecorder로 대체) ───────────────
RING_BUFFER_PRE_SEC = 60       # 사고 전 보관 (초)
RING_BUFFER_POST_SEC = 120     # 사고 후 녹화 (초)
RING_BUFFER_FPS = 15           # 버퍼 FPS (원본 대비 절반)
MAX_CONCURRENT_STREAMS = 4     # 동시 모니터링 카메라 수

# ── OnDemand 녹화 설정 (징조 기반) ───────────────────────────────────
RECORD_DURATION_MIN_SEC = 180   # 최소 녹화 시간 (3분)
RECORD_DURATION_MAX_SEC = 300   # 최대 녹화 시간 (5분)
RECORD_EXTEND_SEC = 60          # 추가 트리거 시 연장 (1분)
RECORD_COOLDOWN_SEC = 600       # 동일 CCTV 재녹화 쿨다운 (10분)
ITS_VERIFY_DELAY_SEC = 30       # 녹화 종료 후 ITS 확인 대기 (30초)

# ── 영상 품질 게이트 ─────────────────────────────────────────────────
MIN_RESOLUTION = (640, 480)
MIN_FPS = 15
MIN_DURATION_SEC = 10
MAX_SYNTHETIC_RATIO = 0.2      # 합성 데이터 최대 비율
