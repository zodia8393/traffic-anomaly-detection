"""MLLM 추상 인터페이스 — 백엔드 교체 시 이것만 변경."""
import sys as _sys; from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import base64
import json
import logging
import time
from typing import Any

import numpy as np
import requests

from config_new import (
    MLLM_API_URL,
    MLLM_BACKEND,
    MLLM_MAX_TOKENS,
    MLLM_MODEL_PATH,
    MLLM_TEMPERATURE,
)

logger = logging.getLogger(__name__)


class MLLMClient:
    """멀티모달 LLM 클라이언트 — llama_cpp / openai_api 백엔드 지원."""

    def __init__(
        self,
        backend: str = MLLM_BACKEND,
        model_path: str | None = None,
        api_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.backend = backend
        self.api_url = ""
        self.model_name = "default"
        self._session = requests.Session()

        self._hf_model = None
        self._hf_processor = None

        if backend == "llama_cpp":
            self._init_llama_cpp(model_path or MLLM_MODEL_PATH)
        elif backend == "openai_api":
            self._init_openai_api(api_url or MLLM_API_URL)
        elif backend == "transformers":
            self._init_transformers(model_path or "Qwen/Qwen2.5-VL-3B-Instruct")
        else:
            raise ValueError(f"지원하지 않는 백엔드: {backend}")

    def _init_llama_cpp(self, model_path: str) -> None:
        """llama.cpp 서버 모드 — localhost OpenAI 호환 API."""
        self.api_url = "http://localhost:8080/v1/chat/completions"
        self.model_name = model_path
        logger.info("llama_cpp 백엔드 초기화: %s", self.api_url)

    def _init_openai_api(self, api_url: str) -> None:
        """외부 OpenAI 호환 API (vLLM, TGI 등)."""
        self.api_url = api_url.rstrip("/") + "/chat/completions"
        self.model_name = "remote"
        logger.info("openai_api 백엔드 초기화: %s", self.api_url)

    def _init_transformers(self, model_id: str) -> None:
        """HuggingFace transformers 직접 로딩 (CPU/GPU)."""
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        logger.info("transformers 백엔드 로딩: %s", model_id)
        self._hf_processor = AutoProcessor.from_pretrained(model_id)
        self._hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        self._hf_model.eval()
        self.model_name = model_id
        logger.info("transformers 모델 로드 완료: %s", model_id)

    # ── 이미지 인코딩 ────────────────────────────────────────────────

    @staticmethod
    def _encode_image(image: np.ndarray | str) -> str:
        """numpy 배열 또는 base64 문자열 → base64 문자열."""
        if isinstance(image, str):
            return image
        # numpy array → JPEG → base64
        import cv2

        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    @staticmethod
    def _build_image_content(b64: str) -> dict:
        """OpenAI 호환 image_url content 블록 생성."""
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        }

    # ── 메시지 변환 ──────────────────────────────────────────────────

    def _prepare_messages(
        self, messages: list[dict], images: list | None
    ) -> list[dict]:
        """이미지를 첫 번째 user 메시지의 content 배열에 삽입."""
        if not images:
            return messages

        prepared = []
        image_inserted = False
        for msg in messages:
            if msg["role"] == "user" and not image_inserted:
                content_parts: list[dict] = []
                # 이미지 삽입
                for img in images:
                    b64 = self._encode_image(img)
                    content_parts.append(self._build_image_content(b64))
                # 텍스트
                text = msg.get("content", "")
                if text:
                    content_parts.append({"type": "text", "text": text})
                prepared.append({"role": "user", "content": content_parts})
                image_inserted = True
            else:
                prepared.append(msg)
        return prepared

    # ── JSON 파싱 ────────────────────────────────────────────────────

    @staticmethod
    def _parse_json_response(text: str) -> dict | str:
        """응답 텍스트에서 JSON 추출 시도. 실패 시 raw text 반환."""
        # 1) 코드블록 안의 JSON
        import re

        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        candidate = match.group(1).strip() if match else text.strip()

        # 2) 직접 파싱
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # 3) 첫 번째 { ... } 블록 추출
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass

        logger.warning("JSON 파싱 실패, raw text 반환")
        return text

    # ── 핵심 API ─────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        images: list | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict:
        """멀티모달 채팅 — 이미지+텍스트 -> JSON 응답.

        Args:
            messages: OpenAI 형식 메시지 리스트.
            images: numpy 배열 또는 base64 문자열 리스트.
            max_tokens: 최대 생성 토큰 수.
            temperature: 샘플링 온도.

        Returns:
            {"content": str|dict, "usage": dict, "latency_sec": float}
        """
        if self.backend == "transformers":
            return self._chat_transformers(messages, images, max_tokens, temperature)

        prepared = self._prepare_messages(messages, images)

        payload = {
            "model": self.model_name,
            "messages": prepared,
            "max_tokens": max_tokens or MLLM_MAX_TOKENS,
            "temperature": temperature if temperature is not None else MLLM_TEMPERATURE,
        }

        t0 = time.perf_counter()
        try:
            resp = self._session.post(self.api_url, json=payload, timeout=300)
            resp.raise_for_status()
        except requests.RequestException as e:
            latency = time.perf_counter() - t0
            logger.error("MLLM API 호출 실패 (%.1f초): %s", latency, e)
            return {
                "content": f"[ERROR] {e}",
                "usage": {},
                "latency_sec": latency,
            }
        latency = time.perf_counter() - t0

        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        parsed = self._parse_json_response(raw_text)

        logger.info(
            "MLLM 응답 수신: %.1f초, tokens=%s",
            latency,
            usage.get("total_tokens", "?"),
        )

        return {
            "content": parsed,
            "usage": usage,
            "latency_sec": latency,
        }

    def _chat_transformers(
        self,
        messages: list[dict],
        images: list | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> dict:
        """transformers 백엔드로 직접 추론."""
        import torch
        from qwen_vl_utils import process_vision_info

        # 메시지를 Qwen2.5-VL 형식으로 변환
        qwen_messages = []
        for msg in messages:
            if msg["role"] == "system":
                qwen_messages.append({"role": "system", "content": [{"type": "text", "text": msg["content"]}]})
            elif msg["role"] == "user":
                content_parts = []
                if images:
                    for img in images:
                        b64 = self._encode_image(img)
                        content_parts.append({
                            "type": "image",
                            "image": f"data:image/jpeg;base64,{b64}",
                        })
                content_parts.append({"type": "text", "text": msg.get("content", "")})
                qwen_messages.append({"role": "user", "content": content_parts})
            else:
                qwen_messages.append(msg)

        text_input = self._hf_processor.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(qwen_messages)

        inputs = self._hf_processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._hf_model.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self._hf_model.generate(
                **inputs,
                max_new_tokens=max_tokens or MLLM_MAX_TOKENS,
                temperature=temperature if temperature is not None else MLLM_TEMPERATURE,
                do_sample=(temperature or MLLM_TEMPERATURE) > 0,
            )

        # 입력 토큰 이후의 생성 부분만 디코딩
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_text = self._hf_processor.decode(generated, skip_special_tokens=True)
        latency = time.perf_counter() - t0

        input_tokens = int(inputs["input_ids"].shape[1])
        output_tokens = int(len(generated))
        usage = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

        parsed = self._parse_json_response(raw_text)

        logger.info(
            "MLLM 응답 수신 (transformers): %.1f초, tokens=%d→%d",
            latency, input_tokens, output_tokens,
        )

        return {
            "content": parsed,
            "usage": usage,
            "latency_sec": latency,
        }
