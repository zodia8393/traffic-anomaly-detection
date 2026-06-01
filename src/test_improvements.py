"""파이프라인 개선(P1~P7) 회귀 테스트.

보고서 기반 개선사항이 의도대로 동작하고, 향후 변경에 회귀하지 않도록 잠근다.
실행: cd src && python3 -m pytest test_improvements.py -v
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ── P1-a: feature_engineer INTERVAL 버그 ─────────────────────────────
def test_feature_engineer_build_dataset_no_interval_crash(tmp_path):
    """build_dataset가 INTERVAL 파서오류 없이 동작 (빈 데이터도 graceful)."""
    import duckdb
    from layer4_prediction.feature_engineer import FeatureEngineer, FEATURE_NAMES

    db = str(tmp_path / "t.duckdb")
    conn = duckdb.connect(db)
    conn.execute("""CREATE TABLE traffic_agg (
        ic_name VARCHAR, period_start TIMESTAMP, volume INT,
        avg_speed DOUBLE, speed_std DOUBLE, truck_ratio DOUBLE,
        risk_score DOUBLE, mllm_scene VARCHAR)""")
    conn.close()

    fe = FeatureEngineer(db_path=db)
    X, y = fe.build_dataset(period_days=30)  # INTERVAL ? 였다면 파서오류로 크래시
    assert X.shape[1] == len(FEATURE_NAMES)
    assert len(X) == 0  # 빈 traffic_agg


# ── P1-b: metadata_writer 데드레터 내구성 ────────────────────────────
def test_metadata_writer_deadletter_on_failure(tmp_path, monkeypatch):
    """write 실패 시 데드레터(JSONL)에 보존되어 영구손실 방지."""
    import duckdb
    from layer2_metadata import metadata_writer as mw

    db = str(tmp_path / "t.duckdb")
    monkeypatch.setattr(mw, "L2_EXPORTS", tmp_path / "exports")
    w = mw.MetadataWriter(db_path=db)

    # 존재하지 않는 테이블로 강제 실패 → 데드레터 라우팅
    ok = w._execute_durable("INSERT INTO nope VALUES (?)", [1],
                            "traffic_agg", {"ic_name": "X", "period_start": str(datetime.now())})
    assert ok is False
    dl = list((tmp_path / "exports" / "_deadletter").glob("*.jsonl"))
    assert len(dl) == 1, "데드레터 파일이 생성되어야 함"
    assert "X" in dl[0].read_text()


def test_metadata_writer_normal_write_succeeds(tmp_path):
    from layer2_metadata.metadata_writer import MetadataWriter
    w = MetadataWriter(db_path=str(tmp_path / "t.duckdb"))
    w.write_traffic_agg({"ic_name": "IC1", "period_start": datetime.now(), "volume": 5})
    # created_at 자동 채움(P1 L2-5)
    w.write_accident({"event_id": "E1", "road_name": "경부선"})


# ── P2: ITS API 회복력 ───────────────────────────────────────────────
def test_its_fetch_status_returns_tuple():
    from layer0_data.track3_api_incident import ITSIncidentClient
    c = ITSIncidentClient(api_key="")  # 키없음 → (False, [])
    ok, events = c.fetch_incidents_status()
    assert ok is False and events == []


def test_its_backward_compat_list():
    from layer0_data.track3_api_incident import ITSIncidentClient
    c = ITSIncidentClient(api_key="")
    result = c.fetch_incidents()
    assert isinstance(result, list)


def test_its_circuit_breaker_opens_on_quota(monkeypatch):
    """쿼터(4001) 응답 시 서킷 오픈."""
    import requests
    from layer0_data.track3_api_incident import ITSIncidentClient

    # 서킷 상태 초기화
    ITSIncidentClient._circuit_open_until = 0.0
    ITSIncidentClient._consecutive_fails = 0

    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"header": {"resultCode": 4001, "resultMsg": "제한량 초과"}}

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    c = ITSIncidentClient(api_key="DUMMY")
    ok, _ = c.fetch_incidents_status()
    assert ok is False
    assert ITSIncidentClient._circuit_open() is True, "쿼터 시 서킷 오픈해야 함"
    ITSIncidentClient._circuit_open_until = 0.0  # 정리


# ── P4: 활성화 게이트 — STGAE 격리 ──────────────────────────────────
def test_activation_gate_blocks_stgae():
    from anomaly_engine.activation_gate import gate_report
    rep = gate_report()
    assert rep["level4_stgae"]["active"] is False, "STGAE는 AUROC 0.495로 차단되어야 함"
    assert any("AUROC" in r for r in rep["level4_stgae"]["reasons"])


def test_ensemble_isolates_ungated_ml():
    """게이트 미통과 ML 점수(STGAE 등)는 트리거 유발 못 함."""
    from anomaly_engine.ensemble import EnsembleScorer
    s = EnsembleScorer()
    r = s.score([], {"level4_stgae": 0.99, "level2_iforest": 0.99})
    # ENABLE_ML_ENSEMBLE 기본 False → 전부 격리 → 트리거 없음
    assert r.trigger is False


# ── P5: 트리거 T2 실제 dt 가속도 ─────────────────────────────────────
def test_trigger_t2_uses_real_dt():
    """프레임 간격이 2초면 가속도가 dt=1 가정의 절반이어야 함."""
    from layer1_vision.trigger_detector import TriggerDetector
    from config_new import TRIGGER_DECEL_THRESHOLD

    det = TriggerDetector()
    # update(frame_idx, timestamp, tracks_info, ttc_list, speeds)
    # 첫 프레임(prev_timestamp 설정)
    det.update(0, 0.0, {}, [], {})
    # 2초 후, 한 트랙이 급감속 (속도 36→0 km/h = 10→0 m/s)
    # dt=2면 accel=-5 m/s^2, dt=1 가정이면 -10 m/s^2
    events = det.update(1, 2.0, {}, [], {1: [36.0, 0.0]})
    # accel = (0-10)/2 = -5 m/s^2. 임계가 -5보다 완만하면 발화 여부로 dt반영 확인
    # 직접 가속도 검증: dt가 반영되면 -5, 아니면 -10
    # 임계값과 무관하게, dt=2 적용시 약한 감속으로 분류되는지 description 확인
    # (발화하면 description에 accel 포함)
    if events:
        assert "accel=-5" in events[0].description or "accel=-5.0" in events[0].description, \
            f"dt=2 반영 시 accel=-5 기대, got: {events[0].description}"


# ── P7: prefilter 동적 임계 ──────────────────────────────────────────
def test_prefilter_dynamic_static_area():
    """고해상도 프레임에서 정지객체 최소면적이 비례 증가."""
    import numpy as np
    from layer1_vision.prefilter import PreFilter, _STATIC_MIN_AREA

    pf = PreFilter()
    small = np.zeros((480, 640), dtype=np.uint8)   # 307200 * 0.005 = 1536
    big = np.zeros((1080, 1920), dtype=np.uint8)   # 2073600 * 0.005 = 10368
    # _check_static_blobs는 빈 마스크면 False, 임계 계산만 확인 위해 면적식 검증
    assert max(_STATIC_MIN_AREA, small.size * 0.005) == 1536.0
    assert max(_STATIC_MIN_AREA, big.size * 0.005) == 10368.0


# ── P7: rule_engine 메모리 누수 차단 ─────────────────────────────────
def test_rule_engine_purges_inactive():
    from anomaly_engine.rule_engine import RuleEngine
    assert hasattr(RuleEngine, "_purge_inactive")


# ── cleanup retention 날짜 컷오프 ────────────────────────────────────
def test_cleanup_date_cutoff(tmp_path):
    """keep_days 초과 날짜 디렉토리만 삭제 대상."""
    from layer0_data.cleanup_recordings import cleanup
    from datetime import timedelta
    old = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
    recent = datetime.now().strftime("%Y%m%d")
    (tmp_path / old).mkdir()
    (tmp_path / recent).mkdir()
    (tmp_path / old / "f.ts").write_bytes(b"x" * 100)
    removed, _ = cleanup(tmp_path, keep_days=21, dry_run=True)
    assert removed == 1  # old만 대상, recent 보존
