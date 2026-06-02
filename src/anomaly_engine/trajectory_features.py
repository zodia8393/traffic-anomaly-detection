"""궤적 운동학 특징 추출 — 지도 사고감지기용.

STGAE(비지도 reconstruction)가 AUROC 0.49(랜덤)인 근본 원인은 라벨을 버리는
구조였다. 본 모듈은 궤적에서 물리적 운동학 특징(속도·가속도·방향변화·정지)을
추출하여 지도 분류기 입력으로 쓴다. 트리거 7종(급감속·역주행·정차 등)과 정합.

클립 형식(stgae_format .txt): frame_id track_id track_id x_norm y_norm ...
"""
from __future__ import annotations

import numpy as np

# 특징 이름 (순서 고정 — 모델 입력/해석 일관성)
# an1~an7 = 차선변경 위반 → 횡방향(lateral) + 차량간(inter-vehicle) 특징이 핵심
FEATURE_NAMES = [
    "sp_mean", "sp_std", "sp_max", "sp_p90", "sp_min",
    "acc_mean", "acc_std", "acc_min", "acc_max", "acc_p95_abs",
    "jerk_std",                                   # 가속도 변화율(급격성)
    "dc_mean", "dc_std", "dc_max", "dc_reversal",  # 방향변화/역주행
    "stop_mean", "stop_max", "stop_frac_tracks",   # 정지
    "speed_var_across", "n_tracks", "track_len_mean",
    # ── 차선변경/관계 특징 (an2~an7 겨냥) ──
    "lat_max", "lat_mean", "lat_total",            # 횡방향 이동(차선변경 크기)
    "lat_changers", "lat_simul_max",               # 차선변경 차량수 / 동시변경(an3)
    "min_inter_dist", "close_pair_frac",           # 차량간 최소거리 / 근접(an7 안전거리)
]
N_FEATURES = len(FEATURE_NAMES)

_STOP_SPEED = 0.002      # 정지 판정 속도(정규화 변위/frame)
_REVERSAL_ANGLE = 2.0    # 역주행 판정 각도(rad, ~115도 이상 급반전)


def parse_clip(path_or_lines) -> dict[int, list]:
    """클립 파일/라인 → {track_id: [(frame, x, y), ...]}."""
    tracks: dict[int, list] = {}
    lines = open(path_or_lines) if isinstance(path_or_lines, str) else path_or_lines
    for line in lines:
        p = line.split()
        if len(p) < 5:
            continue
        try:
            fr, tid, x, y = int(p[0]), int(p[1]), float(p[3]), float(p[4])
        except ValueError:
            continue
        tracks.setdefault(tid, []).append((fr, x, y))
    return tracks


def extract_features(tracks: dict[int, list]) -> np.ndarray | None:
    """클립 단위 특징 벡터(N_FEATURES). 유효 트랙 없으면 None.

    주 진행축(분산 큰 축)을 종방향, 직교축을 횡방향으로 보고 차선변경(횡이동)을 측정.
    """
    speeds, accels, jerks, dirchg = [], [], [], []
    reversals, stop_ratio, mean_speeds, track_lens = [], [], [], []
    lat_moves = []                          # 트랙별 횡방향 총이동
    frames_xy: dict[int, list] = {}         # frame -> [(x,y),...] (차량간 거리용)
    lat_change_frames: dict[int, int] = {}  # frame -> 횡이동 큰 트랙 수 (동시변경)
    n = 0

    # 종/횡 축 결정: 전체 변위의 분산이 큰 축 = 종방향
    all_d = []
    for pts in tracks.values():
        if len(pts) >= 2:
            xy = np.array([(x, y) for _, x, y in sorted(pts)], dtype=np.float64)
            all_d.append(np.diff(xy, axis=0))
    if all_d:
        D = np.concatenate(all_d)
        lat_axis = 0 if D[:, 0].std() < D[:, 1].std() else 1  # 분산 작은 축 = 횡
    else:
        lat_axis = 0

    for pts in tracks.values():
        if len(pts) < 3:
            continue
        pts = sorted(pts)
        xy = np.array([(x, y) for _, x, y in pts], dtype=np.float64)
        frs = [f for f, _, _ in pts]
        d = np.diff(xy, axis=0)
        sp = np.linalg.norm(d, axis=1)
        if len(sp) < 2:
            continue
        acc = np.diff(sp)
        jerk = np.diff(acc) if len(acc) >= 2 else np.array([0.0])
        ang = np.arctan2(d[:, 1], d[:, 0])
        dang = np.abs(np.diff(ang))
        dang = np.minimum(dang, 2 * np.pi - dang)

        speeds.append(sp); accels.append(acc); jerks.append(jerk); dirchg.append(dang)
        reversals.append(np.sum(dang > _REVERSAL_ANGLE))
        stop_ratio.append(np.mean(sp < _STOP_SPEED))
        mean_speeds.append(sp.mean())
        track_lens.append(len(pts))

        lat_disp = np.abs(d[:, lat_axis])
        lat_total = float(lat_disp.sum())
        lat_moves.append(lat_total)
        # 차량간 거리용 프레임별 위치
        for (f, x, y) in pts:
            frames_xy.setdefault(f, []).append((x, y))
        # 동시 차선변경: 횡이동 큰 프레임 집계
        for k, f in enumerate(frs[1:]):
            if lat_disp[k] > 0.015:  # 유의미 횡이동
                lat_change_frames[f] = lat_change_frames.get(f, 0) + 1
        n += 1

    if n == 0:
        return None

    # 차량간 최소거리 / 근접 프레임 비율
    min_inter = 1.0
    close_frames = 0
    multi_frames = 0
    for f, pos in frames_xy.items():
        if len(pos) < 2:
            continue
        multi_frames += 1
        P = np.array(pos)
        dists = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        md = dists.min()
        min_inter = min(min_inter, md)
        if md < 0.08:
            close_frames += 1
    close_frac = close_frames / multi_frames if multi_frames else 0.0
    lat_changers = float(np.sum(np.array(lat_moves) > 0.05))   # 차선변경 차량수
    lat_simul_max = float(max(lat_change_frames.values())) if lat_change_frames else 0.0

    sp = np.concatenate(speeds); acc = np.concatenate(accels)
    jr = np.concatenate(jerks); dc = np.concatenate(dirchg); lm = np.array(lat_moves)
    feat = [
        sp.mean(), sp.std(), sp.max(), np.percentile(sp, 90), sp.min(),
        acc.mean(), acc.std(), acc.min(), acc.max(), np.percentile(np.abs(acc), 95),
        jr.std(),
        dc.mean(), dc.std(), dc.max(), float(np.sum(reversals)),
        float(np.mean(stop_ratio)), float(np.max(stop_ratio)),
        float(np.mean(np.array(stop_ratio) > 0.3)),
        float(np.std(mean_speeds)) if n > 1 else 0.0,
        float(n), float(np.mean(track_lens)),
        lm.max(), lm.mean(), lm.sum(),
        lat_changers, lat_simul_max,
        min_inter, close_frac,
    ]
    arr = np.array(feat, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return None
    return arr
