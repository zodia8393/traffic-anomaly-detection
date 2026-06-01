"""카메라 호모그래피 캘리브레이션 생성기.

speed_estimator/ttc는 호모그래피가 있어야 픽셀→실세계(m) 변환으로 절대속도(km/h)를
산출한다. calibrations/ 가 비어 폴백(0.05m/px)만 쓰이면 속도/TTC가 물리적으로 무의미하다.

영상에서 도로 위 4점(예: 차로 경계·정지선)을 픽셀로 찍고, 그 실세계 좌표(m)를
입력하면 호모그래피를 계산해 JSON으로 저장한다. 실세계 기준은 차로폭 3.5m,
차선 점선 길이/간격 등 표준 도로요소를 활용한다.

사용 (대화형 좌표 대신 인자로):
  python make_calibration.py --camera 경부선_원지동 \
    --img-pts "100,400 600,400 580,250 150,250" \
    --world-pts "0,0 7,0 7,30 0,30"     # 2차로폭(7m) x 30m 사다리꼴

출력: configs/cameras/<camera>.json  {"homography_matrix": [[3x3]], "scale_m_per_px": ...}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_new import ANOMALY_CAMERA_CONFIGS


def parse_pts(s: str) -> np.ndarray:
    pts = []
    for pair in s.split():
        x, y = pair.split(",")
        pts.append([float(x), float(y)])
    arr = np.array(pts, dtype=np.float64)
    if arr.shape != (4, 2):
        raise ValueError(f"4점(x,y)이 필요: {arr.shape}")
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True, help="카메라 식별자(파일명)")
    ap.add_argument("--img-pts", required=True,
                    help='이미지 픽셀 4점 "x1,y1 x2,y2 x3,y3 x4,y4"')
    ap.add_argument("--world-pts", required=True,
                    help='대응 실세계 4점(m) "X1,Y1 ..." (차로폭 3.5m 기준)')
    ap.add_argument("--out-dir", default=str(ANOMALY_CAMERA_CONFIGS))
    args = ap.parse_args()

    img = parse_pts(args.img_pts)
    world = parse_pts(args.world_pts)

    H, status = cv2.findHomography(img, world)
    if H is None:
        raise SystemExit("호모그래피 계산 실패 — 4점이 동일선상이 아닌지 확인")

    # 대각 픽셀 거리 vs 실세계 거리로 평균 스케일(폴백용) 추정
    img_diag = float(np.hypot(*(img[2] - img[0])))
    world_diag = float(np.hypot(*(world[2] - world[0])))
    scale = world_diag / img_diag if img_diag > 0 else 0.05

    # 재투영 오차 검증
    proj = cv2.perspectiveTransform(img.reshape(-1, 1, 2), H).reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum((proj - world) ** 2, axis=1))))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.camera}.json"
    out.write_text(json.dumps({
        "camera": args.camera,
        "homography_matrix": H.tolist(),
        "scale_m_per_px": round(scale, 5),
        "reproj_rmse_m": round(rmse, 3),
        "img_pts": img.tolist(),
        "world_pts": world.tolist(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"저장: {out}")
    print(f"  재투영 RMSE: {rmse:.3f}m (작을수록 정확, 1m 이내 권장)")
    print(f"  폴백 스케일: {scale:.4f} m/px")
    if rmse > 1.0:
        print("  ⚠ RMSE>1m — 4점 좌표를 재확인하세요")


if __name__ == "__main__":
    main()
