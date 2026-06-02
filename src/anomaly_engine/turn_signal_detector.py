"""방향지시등(깜빡이) 점멸 검출 — an1(방향지시등 불이행) 감지용.

CCTV 차량 크롭 시퀀스의 후미등 영역에서 amber(주황) 점멸을 검출한다.
검증결과: an1(불이행)=점멸스코어~0.0 vs 정상차선변경=~1.9 (명확 분리).

궤적(x,y)만으로는 an1을 잡을 수 없던 한계(AUROC 0.48)를 영상기반으로 보완.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

# amber/orange 색범위 (HSV)
_AMBER_LO = (8, 80, 80)
_AMBER_HI = (35, 255, 255)


def amber_intensity(crop) -> float:
    """크롭 후미(하단 절반)의 amber 강도(0~255 평균). 점멸 검출용 단일프레임값."""
    if cv2 is None or crop is None or crop.size == 0:
        return 0.0
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return 0.0
    rear = crop[crop.shape[0] // 2:, :]
    hsv = cv2.cvtColor(rear, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _AMBER_LO, _AMBER_HI)
    return float(mask.mean())


def blink_score(amber_series: list[float] | np.ndarray) -> float:
    """amber 강도 시계열 → 점멸 스코어. 높을수록 깜빡이 켜짐.

    점멸 = 주기적 진동(자기상관 피크 × 진폭). 10fps 기준 주기 4~10프레임(1~2.5Hz).
    """
    s = np.asarray(amber_series, dtype=np.float64)
    if len(s) < 8:
        return 0.0
    s = s - s.mean()
    if s.std() < 1e-6:
        return 0.0
    ac = np.correlate(s, s, mode="full")[len(s) - 1:]
    ac = ac / (ac[0] + 1e-9)
    peak = float(np.max(ac[3:11])) if len(ac) > 11 else 0.0
    return peak * float(s.std())


def blink_score_from_crops(crops: list) -> float:
    """차량 크롭 시퀀스 → 점멸 스코어 (라이브 추론용)."""
    series = [amber_intensity(c) for c in crops if c is not None]
    return blink_score(series)
