"""실시간 파이프라인 부트스트랩 — 이중 config.py 충돌 해소 + 공용 import.

run_realtime 계열 모듈이 검출기/트래커를 import하기 전에 **반드시 먼저 import**해야 한다.
import 시 부수효과로:
  1) layer0_data/config.py에서 녹화 상수 로드 (Phase 1)
  2) track3 모듈(ITS 사고/CCTV 클라이언트) import
  3) sys.modules["config"]를 pipeline/src/config.py로 교체 (Phase 2 — detector/tracker용)
  4) realtime_constants / frame_sampler / collection_record 로드
공용 심볼(CCTVInfo, ITSCCTVClient, 상수, 경로 등)을 re-export한다.

분리 이유: 이 config 스왑 순서가 깨지면 vision pipeline이 잘못된 config를 받는다.
단일 모듈에 격리해 순서를 보장하고, 각 클래스 모듈은 여기서만 공용 심볼을 가져온다.
"""
from __future__ import annotations

import importlib.util as _ilu
import logging
import os
import sys
from pathlib import Path

# ── 경로 설정 ────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
PIPELINE_SRC = Path("/workspace/prj/work/AI기반 교통상황 대응 기술 개발 연구/pipeline/src")
ACCIDENT_SRC = Path("/workspace/prj/work/AI기반 교통상황 대응 기술 개발 연구/사고분석_설계/src")

os.environ.setdefault("OMP_NUM_THREADS", "4")

# ── 이중 config.py 충돌 해소 ────────────────────────────────────────
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(1, str(ACCIDENT_SRC))

# Phase 1: layer0_data/config.py 기반으로 track3 모듈 import
from config import (
    MAX_CONCURRENT_STREAMS,
    RECORD_COOLDOWN_SEC,
    RECORD_DURATION_MAX_SEC,
    RECORD_DURATION_MIN_SEC,
    RECORD_EXTEND_SEC,
    ITS_VERIFY_DELAY_SEC,
    STREAM_DIR,
)
from track3_api_incident import IncidentEvent, ITSIncidentClient
from track3_cctv_stream import CCTVInfo, ITSCCTVClient, _haversine

# Phase 2: config 모듈을 pipeline/src/config.py로 교체
_spec = _ilu.spec_from_file_location("config", str(PIPELINE_SRC / "config.py"))
_pipeline_cfg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pipeline_cfg)
sys.modules["config"] = _pipeline_cfg

# pipeline/src를 path에 추가 (detector, tracker import용)
sys.path.insert(0, str(PIPELINE_SRC))

# 임계값 상수(단일 출처) + 프레임 샘플러 + 수집 레코드
from realtime_constants import (
    SAMPLE_FPS,
    ITS_CHECK_RADIUS_KM, ITS_CHECK_COOLDOWN_SEC,
    SEVERITY_GATE, SEVERITY_PENDING, SEVERITY_FORCE_PRESERVE,
    PENDING_MAX_RETRIES, PENDING_RETRY_INTERVAL_SEC,
    RECORD_TRIGGER_TYPES,
    CONSENSUS_WINDOW_SEC, CONSENSUS_MIN_TYPES, INSTANT_RECORD_TYPES,
)
from frame_sampler import FrameSampler
from collection_record import CollectionRecord

# ── 파생 경로 (STREAM_DIR 의존) ─────────────────────────────────────
SAVE_DIR = STREAM_DIR / "accident_clips"
META_DIR = STREAM_DIR / "metadata"
LOG_DIR = STREAM_DIR / "realtime_logs"
MULTI_STATUS_FILE = LOG_DIR / "multi_status.json"
PENDING_DIR = STREAM_DIR / "pending"

logger = logging.getLogger("realtime")

__all__ = [
    "PIPELINE_SRC", "ACCIDENT_SRC",
    "MAX_CONCURRENT_STREAMS", "RECORD_COOLDOWN_SEC", "RECORD_DURATION_MAX_SEC",
    "RECORD_DURATION_MIN_SEC", "RECORD_EXTEND_SEC", "ITS_VERIFY_DELAY_SEC", "STREAM_DIR",
    "IncidentEvent", "ITSIncidentClient", "CCTVInfo", "ITSCCTVClient", "_haversine",
    "SAMPLE_FPS", "ITS_CHECK_RADIUS_KM", "ITS_CHECK_COOLDOWN_SEC",
    "SEVERITY_GATE", "SEVERITY_PENDING", "SEVERITY_FORCE_PRESERVE",
    "PENDING_MAX_RETRIES", "PENDING_RETRY_INTERVAL_SEC", "RECORD_TRIGGER_TYPES",
    "CONSENSUS_WINDOW_SEC", "CONSENSUS_MIN_TYPES", "INSTANT_RECORD_TYPES",
    "FrameSampler", "CollectionRecord",
    "SAVE_DIR", "META_DIR", "LOG_DIR", "MULTI_STATUS_FILE", "PENDING_DIR",
]
