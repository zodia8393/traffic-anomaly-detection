"""Task 2: MLLM 사고 감지 — 키프레임 + 차량 궤적으로 사고 판단."""
import sys as _sys; from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from config_new import L3_ACCIDENTS
from .mllm_client import MLLMClient
from .prompts import build_messages

logger = logging.getLogger(__name__)

_VALID_TYPES = {
    "rear_end", "sideswipe", "rollover",
    "head_on", "fixed_object", "none",
}
_VALID_SEVERITY = {"minor", "moderate", "severe", "fatal"}


class AccidentDetectorMLLM:
    """MLLM 기반 사고 감지기 (기존 rule-based video_consistency와 별개)."""

    def __init__(self, mllm_client: MLLMClient) -> None:
        self.client = mllm_client

    def detect(
        self,
        keyframes: list[np.ndarray],
        trigger_event: dict[str, Any],
        tracks_metadata: dict[str, Any],
    ) -> dict:
        """사고 감지 수행.

        Args:
            keyframes: 트리거 시점 키프레임 3~5장 (시간순).
            trigger_event: {
                "trigger_type": str,  # T1~T7
                "ttc_values": list[float],
                "speed_changes": list[dict],
            }
            tracks_metadata: {
                "track_summaries": str | list[dict],
                기타 궤적 정보
            }

        Returns:
            {
                "accident_detected": bool,
                "confidence": float,
                "accident_type": str,
                "severity": str,
                "involved_vehicles": list[dict],
                "timestamp_estimated": str,
                "reasoning": str,
                "raw_response": dict,
            }
        """
        # 메타데이터 조립 (speed_calibrated 전파 — 미보정 속도는 build_messages에서 제외)
        metadata = {
            "trigger_type": trigger_event.get("trigger_type", ""),
            "ttc_values": _format_list(trigger_event.get("ttc_values", [])),
            "speed_changes": _format_list(trigger_event.get("speed_changes", [])),
            "track_summaries": _format_tracks(
                tracks_metadata.get("track_summaries", [])
            ),
            "speed_calibrated": tracks_metadata.get("speed_calibrated", True),
        }

        messages = build_messages("accident", keyframes, metadata)
        response = self.client.chat(messages, images=keyframes)

        content = response["content"]
        result = self._validate(content)
        result["raw_response"] = response

        logger.info(
            "사고 감지 완료: detected=%s, type=%s, conf=%.2f (%.1f초)",
            result.get("accident_detected"),
            result.get("accident_type"),
            result.get("confidence", 0),
            response.get("latency_sec", 0),
        )

        return result

    @staticmethod
    def _validate(content: dict | str) -> dict:
        """MLLM 응답 검증 + 기본값 보완."""
        if isinstance(content, str):
            return {
                "accident_detected": False,
                "confidence": 0.0,
                "accident_type": "none",
                "severity": "minor",
                "involved_vehicles": [],
                "timestamp_estimated": "",
                "reasoning": content,
            }

        result = dict(content)

        # accident_detected: bool 보장
        detected = result.get("accident_detected")
        if isinstance(detected, str):
            result["accident_detected"] = detected.lower() in ("true", "yes", "1")
        elif not isinstance(detected, bool):
            result["accident_detected"] = False

        # confidence: float 범위
        try:
            conf = float(result.get("confidence", 0))
            result["confidence"] = max(0.0, min(1.0, conf))
        except (TypeError, ValueError):
            result["confidence"] = 0.0

        # accident_type 검증
        if result.get("accident_type") not in _VALID_TYPES:
            result["accident_type"] = "none"

        # severity 검증
        if result.get("severity") not in _VALID_SEVERITY:
            result["severity"] = "minor"

        # involved_vehicles 보장
        if not isinstance(result.get("involved_vehicles"), list):
            result["involved_vehicles"] = []

        # 미감지 시 정합성 보정
        if not result["accident_detected"]:
            result["accident_type"] = "none"

        if not result.get("reasoning"):
            result["reasoning"] = ""
        if not result.get("timestamp_estimated"):
            result["timestamp_estimated"] = ""

        return result


    def save_result(self, result: dict, trigger_type: str = "") -> Path:
        """사고 감지 결과를 L3_ACCIDENTS에 JSON 저장."""
        L3_ACCIDENTS.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"accident_{trigger_type}_{ts}.json" if trigger_type else f"accident_{ts}.json"
        fpath = L3_ACCIDENTS / fname
        save_data = {k: v for k, v in result.items() if k != "raw_response"}
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        logger.info("사고 감지 저장 → %s", fpath)
        return fpath


def _format_list(items: list) -> str:
    """리스트를 프롬프트 텍스트로 변환."""
    if not items:
        return "없음"
    return ", ".join(str(x) for x in items)


def _format_tracks(tracks: str | list) -> str:
    """궤적 요약을 프롬프트 텍스트로 변환."""
    if isinstance(tracks, str):
        return tracks
    if not tracks:
        return "없음"
    lines = []
    for t in tracks:
        if isinstance(t, dict):
            tid = t.get("track_id", "?")
            cls = t.get("class", "?")
            speed = t.get("avg_speed", "?")
            lines.append(f"  - Track {tid}: {cls}, 평균속도 {speed} km/h")
        else:
            lines.append(f"  - {t}")
    return "\n".join(lines)
