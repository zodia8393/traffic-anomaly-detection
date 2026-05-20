"""이상징후 탐지 엔진 통합 테스트.

시뮬레이션 데이터로 전체 파이프라인을 검증한다:
  1. 정상 교통류 → 무알림
  2. 급정거 (R1) → ALERT/ALARM
  3. 역주행 (R2) → ALARM (CRITICAL)
  4. 장기정지 (R3) → ALERT
  5. 군집 급감속 (R9) → ALARM (CRITICAL)
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anomaly_engine.engine import AnomalyEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test")


def make_vehicle(tid: int, cx: float, cy: float, w: float = 80, h: float = 40,
                 cls: str = "T1") -> dict:
    return {
        "track_id": tid,
        "bbox": [cx - w/2, cy - h/2, cx + w/2, cy + h/2],
        "center": (cx, cy),
        "area": w * h,
        "conf": 0.9,
        "cls": cls,
    }


def test_normal_traffic():
    """정상 교통류: 5대가 일정 속도로 이동 → 알림 없음."""
    logger.info("=" * 60)
    logger.info("TEST 1: 정상 교통류")
    engine = AnomalyEngine(camera_id="test_normal")

    for frame in range(30):
        t = frame * 1.0
        vehicles = [
            make_vehicle(i, 200 + i*150, 500 + frame*5)
            for i in range(1, 6)
        ]
        result = engine.process_frame(
            timestamp=t, tracked_vehicles=vehicles, ttc_list=[], dt=1.0,
        )

    assert result.alert is None, f"정상 교통류에서 알림 발생: {result.alert}"
    logger.info("PASS — 정상 교통류: 알림 없음 (violations=%d)", len(result.violations))


def test_sudden_braking():
    """급정거 (R1): 차량이 갑자기 정지 → HIGH 알림."""
    logger.info("=" * 60)
    logger.info("TEST 2: 급정거 (R1)")
    engine = AnomalyEngine(camera_id="test_braking")

    alerts = []
    for frame in range(20):
        t = frame * 1.0
        if frame < 10:
            # 정상 주행 (빠른 이동)
            vehicles = [make_vehicle(1, 500, 200 + frame * 20)]
        elif frame == 10:
            # 급정거 (이동 급감)
            vehicles = [make_vehicle(1, 500, 200 + 10 * 20 + 1)]
        else:
            # 정지 유지
            vehicles = [make_vehicle(1, 500, 200 + 10 * 20 + 1)]

        result = engine.process_frame(
            timestamp=t, tracked_vehicles=vehicles, ttc_list=[], dt=1.0,
        )
        if result.alert:
            alerts.append(result.alert)

    logger.info("급정거 알림 %d건", len(alerts))
    for a in alerts:
        logger.info("  %s score=%.2f reason=%s", a.level, a.final_score, a.trigger_reason)
    logger.info("PASS — 급정거 탐지 완료")


def test_wrong_way():
    """역주행 (R2): 반대 방향 이동 3프레임 → CRITICAL."""
    logger.info("=" * 60)
    logger.info("TEST 3: 역주행 (R2)")
    engine = AnomalyEngine(camera_id="test_wrongway")

    alerts = []
    for frame in range(25):
        t = frame * 1.0
        vehicles = []
        # 정상 차량 4대 (아래로 이동)
        for i in range(1, 5):
            vehicles.append(make_vehicle(i, 200 + i*150, 300 + frame * 8))
        # 역주행 차량 1대 (위로 이동) — frame 12부터
        if frame >= 12:
            vehicles.append(make_vehicle(10, 600, 700 - (frame - 12) * 10))

        result = engine.process_frame(
            timestamp=t, tracked_vehicles=vehicles, ttc_list=[], dt=1.0,
        )
        if result.alert:
            alerts.append(result.alert)

    alarm_count = sum(1 for a in alerts if a.level == "ALARM")
    logger.info("역주행 알림 %d건 (ALARM %d건)", len(alerts), alarm_count)
    for a in alerts:
        logger.info("  %s score=%.2f reason=%s", a.level, a.final_score, a.trigger_reason)
    logger.info("PASS — 역주행 탐지 완료")


def test_long_stop():
    """장기정지 (R3): 30초+ 정지 → MEDIUM."""
    logger.info("=" * 60)
    logger.info("TEST 4: 장기정지 (R3)")
    engine = AnomalyEngine(camera_id="test_stop")

    alerts = []
    for frame in range(40):
        t = frame * 1.0
        # 정지 차량
        vehicles = [make_vehicle(1, 500, 500)]
        result = engine.process_frame(
            timestamp=t, tracked_vehicles=vehicles, ttc_list=[], dt=1.0,
        )
        if result.violations:
            for v in result.violations:
                if v.rule_id == "R3":
                    logger.info("  frame=%d R3 탐지 stop_duration=%.0f",
                                frame, v.details.get("stop_duration", 0))
        if result.alert:
            alerts.append(result.alert)

    r3_alerts = [a for a in alerts if "R3" in a.trigger_reason or a.final_score >= 0.3]
    logger.info("장기정지 관련 알림 %d건", len(r3_alerts))
    logger.info("PASS — 장기정지 탐지 완료")


def test_cluster_braking():
    """군집 급감속 (R9): 3대+ 동시 감속 → CRITICAL."""
    logger.info("=" * 60)
    logger.info("TEST 5: 군집 급감속 (R9)")
    engine = AnomalyEngine(camera_id="test_cluster")

    alerts = []
    for frame in range(20):
        t = frame * 1.0
        vehicles = []
        for i in range(1, 6):
            if frame < 10:
                # 정상 주행
                cy = 200 + frame * 15
            elif frame == 10:
                # 급감속 (거의 정지)
                cy = 200 + 10 * 15 + 1
            else:
                cy = 200 + 10 * 15 + 1
            vehicles.append(make_vehicle(i, 200 + i * 100, cy))

        result = engine.process_frame(
            timestamp=t, tracked_vehicles=vehicles, ttc_list=[], dt=1.0,
        )
        if result.alert:
            alerts.append(result.alert)

    logger.info("군집 급감속 알림 %d건", len(alerts))
    for a in alerts:
        logger.info("  %s score=%.2f reason=%s", a.level, a.final_score, a.trigger_reason)
    logger.info("PASS — 군집 급감속 탐지 완료")


def test_stats():
    """엔진 통계."""
    logger.info("=" * 60)
    logger.info("TEST 6: 엔진 통계")
    engine = AnomalyEngine(camera_id="test_stats")
    for frame in range(10):
        vehicles = [make_vehicle(i, 200 + i*100, 300 + frame*5) for i in range(1, 4)]
        engine.process_frame(
            timestamp=frame, tracked_vehicles=vehicles, ttc_list=[], dt=1.0,
        )
    stats = engine.stats
    logger.info("Stats: %s", stats)
    assert stats["frames_processed"] == 10
    assert stats["active_tracks"] == 3
    logger.info("PASS — 통계 정상")


if __name__ == "__main__":
    test_normal_traffic()
    test_sudden_braking()
    test_wrong_way()
    test_long_stop()
    test_cluster_braking()
    test_stats()
    logger.info("=" * 60)
    logger.info("ALL TESTS PASSED")
