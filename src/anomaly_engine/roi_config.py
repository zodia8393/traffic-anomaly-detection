"""Level 0: 카메라별 ROI / 차로 설정.

카메라마다 YAML 파일로 관심영역(ROI)과 차로 정보를 정의한다.
YAML이 없는 카메라는 전체 프레임을 ROI로, 방향은 교통류 추정값을 사용.

YAML 예시 (configs/camera_001.yaml):
  camera_id: "001"
  frame_size: [1920, 1080]
  expected_heading: 90       # 주 진행방향 (degrees, 0=북, 시계방향)
  road_type: expressway      # expressway | national | urban
  roi:
    - name: "main_road"
      polygon: [[100,200],[1800,200],[1800,900],[100,900]]
      type: road
    - name: "shoulder"
      polygon: [[0,900],[1920,900],[1920,1080],[0,1080]]
      type: shoulder
  lanes:
    - id: 1
      center_line: [[200,200],[200,900]]
      direction: 180
    - id: 2
      center_line: [[500,200],[500,900]]
      direction: 180
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ROIRegion:
    name: str
    polygon: list[tuple[int, int]]
    region_type: str = "road"


@dataclass
class LaneInfo:
    lane_id: int
    center_line: list[tuple[int, int]]
    direction: float = 0.0


@dataclass
class CameraConfig:
    camera_id: str
    frame_size: tuple[int, int] = (1920, 1080)
    expected_heading: float | None = None
    road_type: str = "expressway"
    roi_regions: list[ROIRegion] = field(default_factory=list)
    lanes: list[LaneInfo] = field(default_factory=list)
    scale_m_per_px: float = 0.05

    def point_in_roi(self, x: float, y: float, region_type: str = "road") -> bool:
        """점이 지정 타입의 ROI 내부에 있는지 판정."""
        for roi in self.roi_regions:
            if roi.region_type == region_type:
                if _point_in_polygon(x, y, roi.polygon):
                    return True
        if not self.roi_regions:
            w, h = self.frame_size
            return 0 <= x <= w and 0 <= y <= h
        return False

    def nearest_lane(self, x: float, y: float) -> LaneInfo | None:
        """가장 가까운 차로 반환."""
        if not self.lanes:
            return None
        best = None
        best_dist = float("inf")
        for lane in self.lanes:
            dist = _point_to_polyline_dist(x, y, lane.center_line)
            if dist < best_dist:
                best_dist = dist
                best = lane
        return best


def load_camera_config(yaml_path: Path) -> CameraConfig:
    """YAML 파일에서 카메라 설정 로드."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    roi_regions = []
    for roi in data.get("roi", []):
        roi_regions.append(ROIRegion(
            name=roi["name"],
            polygon=[tuple(p) for p in roi["polygon"]],
            region_type=roi.get("type", "road"),
        ))

    lanes = []
    for lane in data.get("lanes", []):
        lanes.append(LaneInfo(
            lane_id=lane["id"],
            center_line=[tuple(p) for p in lane["center_line"]],
            direction=lane.get("direction", 0.0),
        ))

    config = CameraConfig(
        camera_id=data.get("camera_id", "unknown"),
        frame_size=tuple(data.get("frame_size", [1920, 1080])),
        expected_heading=data.get("expected_heading"),
        road_type=data.get("road_type", "expressway"),
        roi_regions=roi_regions,
        lanes=lanes,
        scale_m_per_px=data.get("scale_m_per_px", 0.05),
    )
    logger.info("카메라 설정 로드: %s (ROI %d개, 차로 %d개)",
                config.camera_id, len(roi_regions), len(lanes))
    return config


def default_camera_config(camera_id: str = "default") -> CameraConfig:
    """기본 카메라 설정 (ROI 없음, 전체 프레임 사용)."""
    return CameraConfig(camera_id=camera_id)


def _point_in_polygon(x: float, y: float, polygon: list[tuple[int, int]]) -> bool:
    """Ray casting 알고리즘."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _point_to_polyline_dist(
    x: float, y: float, polyline: list[tuple[int, int]],
) -> float:
    """점에서 폴리라인까지 최소 거리."""
    import math
    min_dist = float("inf")
    for i in range(len(polyline) - 1):
        ax, ay = polyline[i]
        bx, by = polyline[i + 1]
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            dist = math.hypot(x - ax, y - ay)
        else:
            t = max(0, min(1, ((x - ax) * dx + (y - ay) * dy) / length_sq))
            proj_x = ax + t * dx
            proj_y = ay + t * dy
            dist = math.hypot(x - proj_x, y - proj_y)
        min_dist = min(min_dist, dist)
    return min_dist
