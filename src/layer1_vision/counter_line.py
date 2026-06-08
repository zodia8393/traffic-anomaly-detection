"""방향별 교통량 계수기 — 계수선 통과 방식, 상태 격리.

사고감지 파이프라인과 **동일 검출/추적 패스를 공유**하되, VisionPipeline이 매 프레임
생성하는 tracked_vehicles의 center만 소비한다. 자체 상태(prev_center·counted)로 격리되어
사고감지 데이터경로(triggers/anomaly/_track_history)를 전혀 건드리지 않는다.

계수 설계(놓침 최소화):
- POI 전체프레임 검출 + 카메라가 잘 보는 '유효계수영역(validity_band)'에 수평 계수선.
- 트랙 center가 계수선을 straddle하는 순간 1회 계수(counted set으로 중복 차단).
- dy 부호로 방향(상행/하행) 분류. 카메라별 label로 서울/부산 등 절대방향 부여 가능.
- 소실점 원거리는 validity_band 밖이라 계수 제외(검지 불안정 구간).

미설정(계수선 없음)이면 인스턴스를 생성하지 않음 → VisionPipeline에서 None = 계수 OFF.

검증 전제: 본 계수기 출력은 반드시 육안 GT 대조로 검지율/중복률/방향정확도를 입증해야 한다.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

# 13종 T코드 → 계수용 대분류 (classifier 결과가 있을 때)
_T_BUCKET = {
    "T1": "승용차", "T2": "버스", "T13": "이륜차",
    **{f"T{i}": "화물차" for i in range(3, 13)},
}
# COCO 클래스 → 대분류 (classifier 미주입 시 fallback)
_COCO_BUCKET = {2: "승용차", 3: "이륜차", 5: "버스", 7: "화물차"}


class CameraCounter:
    """계수선 통과 기반 방향별·차종별 교통량 계수기 (카메라 1대).

    Args:
        y_line: 수평 계수선의 y좌표(px). validity_band 안에 위치해야 함.
        label_up: dy<0(화면 위로 이동) 방향 라벨. 예: '서울방향'. 미지정 시 '방향A'.
        label_down: dy>0(화면 아래로 이동) 방향 라벨. 예: '부산방향'. 미지정 시 '방향B'.
        validity_band: (y_near, y_far) 계수 유효 구간. None이면 전체. 소실점 제외용.
        calibrated: 절대방향 라벨이 캘리브로 부여됐는지(서울/부산) vs 자동 A/B인지.
    """

    def __init__(self, y_line: float, label_up: str = "방향A", label_down: str = "방향B",
                 validity_band: tuple[float, float] | None = None,
                 calibrated: bool = False) -> None:
        self.y_line = float(y_line)
        self.label_up = label_up
        self.label_down = label_down
        self.validity_band = validity_band
        self.calibrated = calibrated
        # 격리 상태
        self._prev_center: dict[int, tuple[float, float]] = {}
        self._counted: set[int] = set()
        self._track_cls: dict[int, Counter] = defaultdict(Counter)
        self._counts: dict[str, Counter] = {label_up: Counter(), label_down: Counter()}

    # ── 차종 추출 ────────────────────────────────────────────────────
    @staticmethod
    def _bucket(vehicle: dict) -> str | None:
        cls = vehicle.get("cls")
        if isinstance(cls, str) and cls in _T_BUCKET:
            return _T_BUCKET[cls]
        coco = vehicle.get("coco_cls")
        if isinstance(coco, int) and coco in _COCO_BUCKET:
            return _COCO_BUCKET[coco]
        return None  # classifier 미주입(DummyClassifier) 등 → 방향별 총량만

    def _in_band(self, y: float) -> bool:
        if self.validity_band is None:
            return True
        lo, hi = self.validity_band
        return lo <= y <= hi

    # ── 매 프레임 갱신 ───────────────────────────────────────────────
    def update(self, tracked_vehicles: list[dict[str, Any]]) -> list[dict]:
        """tracked_vehicles의 center로 계수선 통과 판정. 신규 계수 이벤트 리스트 반환."""
        events: list[dict] = []
        for v in tracked_vehicles:
            tid = v.get("track_id")
            center = v.get("center")
            if tid is None or center is None:
                continue
            cx, cy = float(center[0]), float(center[1])
            b = self._bucket(v)
            if b:
                self._track_cls[tid][b] += 1
            prev = self._prev_center.get(tid)
            self._prev_center[tid] = (cx, cy)
            if prev is None or tid in self._counted:
                continue
            pcy = prev[1]
            crossed = (pcy < self.y_line <= cy) or (pcy >= self.y_line > cy)
            if not crossed or not self._in_band(cy):
                continue
            # 방향: 위로(서울/up) vs 아래로(부산/down)
            direction = self.label_down if cy > pcy else self.label_up
            cls_name = (self._track_cls[tid].most_common(1)[0][0]
                        if self._track_cls[tid] else "미분류")
            self._counts[direction][cls_name] += 1
            self._counted.add(tid)
            events.append({"track_id": tid, "direction": direction, "cls": cls_name})
        return events

    # ── 상태 조회/배출 ───────────────────────────────────────────────
    def snapshot(self) -> dict:
        """현재 누적 (리셋 없음)."""
        return {
            "calibrated": self.calibrated,
            "directions": {
                d: {"volume": sum(c.values()), "by_class": dict(c)}
                for d, c in self._counts.items()
            },
        }

    def flush(self) -> dict:
        """누적 배출 후 계수 리셋 (5분 interval 적재용). prev_center/counted는 유지."""
        snap = self.snapshot()
        self._counts = {self.label_up: Counter(), self.label_down: Counter()}
        return snap

    def prune(self, active_track_ids: set[int]) -> None:
        """소실 트랙의 격리상태 정리 — 메모리 무한증가 방지."""
        stale = [t for t in self._prev_center if t not in active_track_ids]
        for t in stale:
            self._prev_center.pop(t, None)
            self._counted.discard(t)
            self._track_cls.pop(t, None)

    @classmethod
    def from_config(cls, cfg: dict | None, frame_h: int) -> "CameraCounter | None":
        """카메라 설정 dict → CameraCounter. 설정 없으면 None(계수 OFF).

        cfg 예: {"count_line_y": 0.60, "validity_band": [0.4, 0.85],
                 "direction_map": {"up": "서울방향", "down": "부산방향"}}
        count_line_y가 0~1이면 비율로 해석(× frame_h).
        """
        if not cfg or "count_line_y" not in cfg:
            return None
        yl = cfg["count_line_y"]
        yl = yl * frame_h if yl <= 1.0 else yl
        band = cfg.get("validity_band")
        if band and band[0] <= 1.0:
            band = (band[0] * frame_h, band[1] * frame_h)
        dm = cfg.get("direction_map", {})
        return cls(
            y_line=yl,
            label_up=dm.get("up", "방향A"),
            label_down=dm.get("down", "방향B"),
            validity_band=tuple(band) if band else None,
            calibrated=bool(dm),
        )
