"""run_realtime 실시간 파이프라인 임계/설정 상수 (STREAM_DIR 비의존).

2167줄 모놀리스에서 분리 — 상수를 단일 출처로 관리하여 워커/파이프라인 모듈이
공유한다. STREAM_DIR 의존 경로 상수(SAVE_DIR 등)는 config 설정 이후 산출되므로
run_realtime.py에 잔류한다.
"""
from __future__ import annotations

SAMPLE_FPS = 1                   # Vision Pipeline 입력 fps (CPU 부하 관리)
ITS_CHECK_RADIUS_KM = 10.0       # 트리거 발화 시 ITS 사고 매칭 반경 (km)
ITS_CHECK_COOLDOWN_SEC = 60      # 동일 트리거 유형 ITS 확인 쿨다운

SEVERITY_GATE = 0.5              # 녹화 시작 최소 severity (v3: 0.3 → v4: 0.5)
SEVERITY_PENDING = 0.7           # ITS 미확인 시 pending 보존 최소 severity
SEVERITY_FORCE_PRESERVE = 0.8    # 재확인 실패해도 보존하는 최소 severity
PENDING_MAX_RETRIES = 3          # pending 재확인 최대 횟수
PENDING_RETRY_INTERVAL_SEC = 600 # pending 재확인 간격 (10분)

# 녹화 트리거 대상 (T7 주기적 스냅샷 제외)
RECORD_TRIGGER_TYPES = {"T1", "T2", "T3", "T4", "T5", "T6"}

# 다중 트리거 합의 (v4)
CONSENSUS_WINDOW_SEC = 5.0       # 합의 윈도우 (초)
CONSENSUS_MIN_TYPES = 2          # 최소 트리거 종류 수
INSTANT_RECORD_TYPES = {"T5"}    # 단독 즉시 녹화 (다수 동시 감속)
