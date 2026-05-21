"""Task 3: 차종 분류 보정 (한국도로공사 13종)."""
import sys as _sys; from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from config_new import VEHICLE_13CLASS, L3_CORRECTIONS
from .mllm_client import MLLMClient
from .prompts import build_messages

logger = logging.getLogger(__name__)

_VALID_T_CODES = set(VEHICLE_13CLASS.keys())
_VALID_UNIT_COUNTS = {1, 2}
_VALID_AXLE_RANGE = range(2, 7)  # 2~6


class ClassCorrector:
    """Vision 1차 분류를 MLLM이 검증 및 보정."""

    def __init__(self, mllm_client: MLLMClient) -> None:
        self.client = mllm_client

    def correct(
        self,
        crop_image: np.ndarray,
        vision_class: str,
        confidence: float,
        bbox_meta: dict[str, Any],
    ) -> dict:
        """단일 차량 crop에 대한 차종 보정.

        Args:
            crop_image: 차량 crop 이미지 (numpy).
            vision_class: Vision 모델의 1차 분류 결과 (예: "T4").
            confidence: Vision 모델 신뢰도 (0.0~1.0).
            bbox_meta: {
                "area": float,          # bbox 면적 (px^2)
                "aspect_ratio": float,  # 가로/세로
                "track_id": int,        # (선택)
            }

        Returns:
            {
                "corrected_class": str,    # T코드
                "confidence": float,
                "unit_count": int,         # 1 또는 2
                "axle_count": int,         # 2~6
                "vision_agreed": bool,
                "reasoning": str,
            }
        """
        metadata = {
            "vision_class": vision_class,
            "confidence": f"{confidence:.2f}",
            "area": str(bbox_meta.get("area", 0)),
            "aspect_ratio": f"{bbox_meta.get('aspect_ratio', 1.0):.2f}",
        }

        messages = build_messages("classify", [crop_image], metadata)
        response = self.client.chat(messages, images=[crop_image])

        content = response["content"]
        result = self._validate(content, vision_class)
        result["raw_response"] = response

        action = "유지" if result["vision_agreed"] else "보정"
        logger.info(
            "차종 보정: %s → %s (%s, conf=%.2f, %.1f초)",
            vision_class,
            result["corrected_class"],
            action,
            result["confidence"],
            response.get("latency_sec", 0),
        )

        return result

    def correct_batch(self, items: list[dict]) -> list[dict]:
        """여러 차량 일괄 보정.

        Args:
            items: [
                {
                    "crop_image": np.ndarray,
                    "vision_class": str,
                    "confidence": float,
                    "bbox_meta": dict,
                },
                ...
            ]

        Returns:
            보정 결과 리스트 (correct() 반환값 + "index" 키).
        """
        results = []
        for idx, item in enumerate(items):
            try:
                result = self.correct(
                    crop_image=item["crop_image"],
                    vision_class=item["vision_class"],
                    confidence=item["confidence"],
                    bbox_meta=item.get("bbox_meta", {}),
                )
                result["index"] = idx
                results.append(result)
            except Exception as e:
                logger.error("배치 보정 실패 (idx=%d): %s", idx, e)
                results.append({
                    "index": idx,
                    "corrected_class": item["vision_class"],
                    "confidence": item["confidence"],
                    "unit_count": 1,
                    "axle_count": 2,
                    "vision_agreed": True,
                    "reasoning": f"MLLM 호출 실패: {e}",
                    "error": str(e),
                })

        agreed = sum(1 for r in results if r.get("vision_agreed", False))
        logger.info(
            "배치 보정 완료: %d건 중 %d건 Vision 동의",
            len(results), agreed,
        )

        return results

    @staticmethod
    def _validate(content: dict | str, vision_class: str) -> dict:
        """MLLM 응답 검증 + 기본값 보완."""
        if isinstance(content, str):
            return {
                "corrected_class": vision_class,
                "confidence": 0.5,
                "unit_count": 1,
                "axle_count": 2,
                "vision_agreed": True,
                "reasoning": content,
            }

        result = dict(content)

        # corrected_class 검증
        corrected = result.get("corrected_class", "")
        if corrected not in _VALID_T_CODES:
            # "T4" 형식이 아닌 경우 정제 시도
            corrected = corrected.upper().strip()
            if corrected not in _VALID_T_CODES:
                logger.warning(
                    "유효하지 않은 T코드: %s → Vision 원본 %s 유지",
                    result.get("corrected_class"), vision_class,
                )
                corrected = vision_class
        result["corrected_class"] = corrected

        # confidence
        try:
            conf = float(result.get("confidence", 0.5))
            result["confidence"] = max(0.0, min(1.0, conf))
        except (TypeError, ValueError):
            result["confidence"] = 0.5

        # unit_count
        try:
            uc = int(result.get("unit_count", 1))
            result["unit_count"] = uc if uc in _VALID_UNIT_COUNTS else 1
        except (TypeError, ValueError):
            result["unit_count"] = 1

        # axle_count
        try:
            ac = int(result.get("axle_count", 2))
            result["axle_count"] = ac if ac in _VALID_AXLE_RANGE else 2
        except (TypeError, ValueError):
            result["axle_count"] = 2

        # vision_agreed
        agreed = result.get("vision_agreed")
        if isinstance(agreed, str):
            result["vision_agreed"] = agreed.lower() in ("true", "yes", "1")
        elif not isinstance(agreed, bool):
            result["vision_agreed"] = (corrected == vision_class)

        if not result.get("reasoning"):
            result["reasoning"] = ""

        return result

    def save_result(self, results: list[dict], video_id: str = "") -> Path:
        """차종 보정 결과를 L3_CORRECTIONS에 JSON 저장."""
        L3_CORRECTIONS.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"correction_{video_id}_{ts}.json" if video_id else f"correction_{ts}.json"
        fpath = L3_CORRECTIONS / fname
        save_data = [
            {k: v for k, v in r.items() if k != "raw_response"}
            for r in results
        ]
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        logger.info("차종 보정 %d건 저장 → %s", len(save_data), fpath)
        return fpath
