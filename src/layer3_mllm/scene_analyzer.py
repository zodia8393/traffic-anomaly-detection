"""Task 1: 장면 이해 — MLLM 기반 교통 상황 분석."""
import sys as _sys; from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from config_new import L3_SCENES
from .mllm_client import MLLMClient
from .prompts import build_messages

logger = logging.getLogger(__name__)

# 유효 traffic_state 값
_VALID_STATES = {"free_flow", "congested", "incident", "stopped"}
_VALID_RISK = {"low", "medium", "high", "critical"}


class SceneAnalyzer:
    """CCTV 키프레임에서 교통 장면을 이해하고 구조화 결과 반환."""

    def __init__(self, mllm_client: MLLMClient) -> None:
        self.client = mllm_client

    def analyze(
        self,
        keyframes: list[np.ndarray],
        metadata: dict[str, Any],
    ) -> dict:
        """키프레임과 메타데이터로 장면 분석 수행.

        Args:
            keyframes: 키프레임 이미지 리스트 (1~5장).
            metadata: ic_name, lane_count, road_type, vehicle_count,
                      class_distribution, avg_speed, trigger_type,
                      trigger_description 등.

        Returns:
            {
                "scene_description": str,
                "traffic_state": str,
                "anomalies": list[str],
                "risk_level": str,
                "raw_response": dict,  # MLLM 원본 응답
            }
        """
        messages = build_messages("scene", keyframes, metadata)
        response = self.client.chat(messages, images=keyframes)

        content = response["content"]
        result = self._validate(content)
        result["raw_response"] = response

        logger.info(
            "장면 분석 완료: state=%s, risk=%s (%.1f초)",
            result.get("traffic_state", "?"),
            result.get("risk_level", "?"),
            response.get("latency_sec", 0),
        )

        return result

    @staticmethod
    def _validate(content: dict | str) -> dict:
        """MLLM 응답을 검증하고 기본값 보완."""
        if isinstance(content, str):
            return {
                "scene_description": content,
                "traffic_state": "free_flow",
                "anomalies": [],
                "risk_level": "low",
            }

        result = dict(content)

        # traffic_state 검증
        if result.get("traffic_state") not in _VALID_STATES:
            logger.warning(
                "유효하지 않은 traffic_state: %s → free_flow",
                result.get("traffic_state"),
            )
            result["traffic_state"] = "free_flow"

        # risk_level 검증
        if result.get("risk_level") not in _VALID_RISK:
            logger.warning(
                "유효하지 않은 risk_level: %s → low",
                result.get("risk_level"),
            )
            result["risk_level"] = "low"

        # anomalies 보장 (리스트)
        if not isinstance(result.get("anomalies"), list):
            result["anomalies"] = []

        # scene_description 보장
        if not result.get("scene_description"):
            result["scene_description"] = ""

        return result

    def save_result(self, result: dict, ic_name: str = "") -> Path:
        """장면 분석 결과를 L3_SCENES에 JSON 저장."""
        L3_SCENES.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"scene_{ic_name}_{ts}.json" if ic_name else f"scene_{ts}.json"
        fpath = L3_SCENES / fname
        save_data = {k: v for k, v in result.items() if k != "raw_response"}
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        logger.info("장면 분석 저장 → %s", fpath)
        return fpath
