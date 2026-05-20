"""5분 단위 교통류 집계기."""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import math
from datetime import datetime, timedelta
from typing import Any

from config_new import VEHICLE_13CLASS

# 화물차 범주: T3 ~ T12
_TRUCK_CLASSES: set[str] = {
    f"T{i}" for i in range(3, 13)
}


class TrafficAggregator:
    """실시간 트랙 스냅샷을 누적하여 고정 interval 마다 교통류 지표를 산출한다."""

    def __init__(self, interval_sec: int = 300) -> None:
        self._interval: timedelta = timedelta(seconds=interval_sec)
        self._period_start: datetime | None = None
        self._seen_track_ids: set[int] = set()
        self._speeds: list[float] = []
        self._classes: list[str] = []

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------
    def update(self, timestamp: datetime, tracks_snapshot: list[dict]) -> dict | None:
        """매 프레임(또는 주기)마다 호출하여 트랙 정보를 누적한다.

        Parameters
        ----------
        timestamp : datetime
            현재 프레임 시각.
        tracks_snapshot : list[dict]
            현재 활성 트랙 목록. 각 dict 는 최소한
            ``track_id``, ``speed`` (km/h), ``vehicle_cls`` 키를 가진다.

        Returns
        -------
        dict | None
            interval 이 경과하면 flush() 결과를 반환하고 버퍼를 초기화한다.
            아직 경과하지 않았으면 None.
        """
        if self._period_start is None:
            self._period_start = self._align_start(timestamp)

        # interval 경과 시 flush 후 새 구간 시작
        result: dict | None = None
        if timestamp >= self._period_start + self._interval:
            result = self.flush()
            self._period_start = self._align_start(timestamp)

        for t in tracks_snapshot:
            tid: int = t["track_id"]
            if tid not in self._seen_track_ids:
                self._seen_track_ids.add(tid)
                speed = t.get("speed")
                if speed is not None:
                    self._speeds.append(float(speed))
                cls_label = t.get("vehicle_cls")
                if cls_label:
                    self._classes.append(str(cls_label))

        return result

    def flush(self) -> dict:
        """현재 구간의 집계를 반환하고 내부 버퍼를 초기화한다.

        Returns
        -------
        dict
            period_start, volume, avg_speed, speed_std, truck_ratio 키.
        """
        volume: int = len(self._seen_track_ids)
        avg_speed: float = _mean(self._speeds)
        speed_std: float = _std(self._speeds)
        truck_count: int = sum(1 for c in self._classes if c in _TRUCK_CLASSES)
        truck_ratio: float = truck_count / volume if volume > 0 else 0.0

        result: dict[str, Any] = {
            "period_start": self._period_start,
            "volume": volume,
            "avg_speed": round(avg_speed, 2),
            "speed_std": round(speed_std, 2),
            "truck_ratio": round(truck_ratio, 4),
        }

        # 버퍼 초기화
        self._seen_track_ids.clear()
        self._speeds.clear()
        self._classes.clear()

        return result

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------
    def _align_start(self, ts: datetime) -> datetime:
        """timestamp 를 interval 경계로 내림한다."""
        epoch = ts.timestamp()
        interval_sec = self._interval.total_seconds()
        aligned = math.floor(epoch / interval_sec) * interval_sec
        return datetime.fromtimestamp(aligned)


# ------------------------------------------------------------------
# 통계 유틸
# ------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(variance)
