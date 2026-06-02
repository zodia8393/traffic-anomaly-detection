"""MLLM용 키프레임 선정.

트리거 이벤트 발생 시점 전후의 프레임 버퍼에서
MLLM 분석에 가장 적합한 키프레임을 선정한다.
"""

from __future__ import annotations
import sys as _sys; from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from config_new import KEYFRAME_MAX, KEYFRAME_MIN_INTERVAL_SEC


def _quality_score(
    bbox_area: float,
    det_conf: float,
    center_x: float,
    center_y: float,
    frame_w: float,
    frame_h: float,
) -> float:
    """키프레임 품질 스코어.

    = bbox면적(정규화) x 검출신뢰도 x 화면중앙근접도

    Args:
        bbox_area: bbox 면적 (px^2).
        det_conf: 검출 confidence (0~1).
        center_x, center_y: bbox 중심 좌표.
        frame_w, frame_h: 프레임 크기.
    """
    # 면적 정규화: frame 면적 대비 비율 (0~1)
    area_norm = min(1.0, bbox_area / (frame_w * frame_h)) if frame_w * frame_h > 0 else 0.0

    # 중앙 근접도: 중심에 가까울수록 1, 가장자리면 0
    dx = abs(center_x - frame_w / 2) / (frame_w / 2) if frame_w > 0 else 1.0
    dy = abs(center_y - frame_h / 2) / (frame_h / 2) if frame_h > 0 else 1.0
    centrality = max(0.0, 1.0 - (dx + dy) / 2)

    return area_norm * det_conf * centrality


def select_keyframes(
    track_buffer: list[dict],
    trigger_frame: int,
    fps: float,
    window_sec: float = 10.0,
    max_frames: int = KEYFRAME_MAX,
    min_interval_sec: float = KEYFRAME_MIN_INTERVAL_SEC,
    frame_w: float = 1920.0,
    frame_h: float = 1080.0,
) -> list[int]:
    """MLLM용 키프레임 인덱스 선정.

    Args:
        track_buffer: 프레임별 트랙 정보 리스트.
            각 원소: {"frame_idx": int, "timestamp": float,
                     "tracks": [{"bbox": [x1,y1,x2,y2], "conf": float, "track_id": int}, ...]}
        trigger_frame: 트리거 발생 프레임 인덱스. 필수 포함.
        fps: 영상 FPS.
        window_sec: 트리거 전후 탐색 윈도우(초).
        max_frames: 최대 키프레임 수.
        min_interval_sec: 연속 키프레임 간 최소 간격(초).
        frame_w: 프레임 너비(px).
        frame_h: 프레임 높이(px).

    Returns:
        선정된 프레임 인덱스 목록 (시간순).
    """
    if not track_buffer:
        return [trigger_frame]

    # 윈도우 범위 필터링
    window_frames = int(window_sec * fps)
    candidates = [
        entry for entry in track_buffer
        if abs(entry["frame_idx"] - trigger_frame) <= window_frames
    ]

    if not candidates:
        return [trigger_frame]

    # 프레임별 최고 품질 스코어 산출
    scored: list[tuple[int, float, float]] = []  # (frame_idx, score, timestamp)
    for entry in candidates:
        frame_idx = entry["frame_idx"]
        timestamp = entry.get("timestamp", frame_idx / fps)
        tracks = entry.get("tracks", [])

        if not tracks:
            scored.append((frame_idx, 0.0, timestamp))
            continue

        # 해당 프레임 내 모든 트랙의 최고 품질
        best_score = 0.0
        for t in tracks:
            bbox = t.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            area = max(0.0, (x2 - x1) * (y2 - y1))
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            conf = t.get("conf", 0.5)
            score = _quality_score(area, conf, cx, cy, frame_w, frame_h)
            best_score = max(best_score, score)

        scored.append((frame_idx, best_score, timestamp))

    # 스코어 내림차순 정렬
    scored.sort(key=lambda x: x[1], reverse=True)

    # 트리거 프레임 필수 포함
    selected_frames: list[int] = [trigger_frame]
    selected_timestamps: list[float] = []

    # 트리거 프레임의 timestamp 찾기
    trigger_ts = trigger_frame / fps
    for fidx, _, ts in scored:
        if fidx == trigger_frame:
            trigger_ts = ts
            break
    selected_timestamps.append(trigger_ts)

    # 나머지 프레임 선정: 최소 간격 준수
    for fidx, score, ts in scored:
        if len(selected_frames) >= max_frames:
            break
        if fidx == trigger_frame:
            continue

        # 이미 선정된 프레임과 최소 간격 확인
        too_close = False
        for sel_ts in selected_timestamps:
            if abs(ts - sel_ts) < min_interval_sec:
                too_close = True
                break
        if too_close:
            continue

        selected_frames.append(fidx)
        selected_timestamps.append(ts)

    # 시간순 정렬
    selected_frames.sort()
    return selected_frames


def select_keyframes_histogram(frames: list, max_frames: int = KEYFRAME_MAX) -> list:
    """히스토그램 거리(Bhattacharyya) 기반 대표 프레임 선정.

    제안서 13.4 "대표 프레임 선정(30→5, 히스토그램 거리)"의 realtime 구현.
    연속 프레임 간 Bhattacharyya 거리가 큰(장면 변화) 프레임을 우선 선택하고,
    첫·마지막 프레임을 필수 포함하여 시간 범위를 확보한다.

    Args:
        frames: 프레임 이미지(np.ndarray) 리스트.
        max_frames: 최대 선정 수.

    Returns:
        선정된 프레임 이미지 리스트 (시간순).
    """
    import cv2

    n = len(frames)
    if n <= max_frames:
        return frames

    diffs = []
    for i in range(1, n):
        h1 = cv2.calcHist([frames[i - 1]], [0], None, [64], [0, 256])
        h2 = cv2.calcHist([frames[i]], [0], None, [64], [0, 256])
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        diff = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
        diffs.append((i, diff))

    diffs.sort(key=lambda x: x[1], reverse=True)
    selected_idx = {0, n - 1}  # 첫·마지막 필수
    for idx, _ in diffs:
        if len(selected_idx) >= max_frames:
            break
        selected_idx.add(idx)

    return [frames[i] for i in sorted(selected_idx)]
