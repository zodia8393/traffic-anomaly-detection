"""Task 4: 사고 원인 추론 + 도공 93컬럼 보고서 자동 생성."""
import sys as _sys; from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from config_new import L3_REPORTS, VEHICLE_13CLASS
from .mllm_client import MLLMClient
from .prompts import build_messages

logger = logging.getLogger(__name__)


class ReportGenerator:
    """MLLM 기반 사고 원인 추론 + 93컬럼 보고서 생성."""

    def __init__(
        self,
        mllm_client: MLLMClient,
        report_indexer: Any | None = None,
    ) -> None:
        self.client = mllm_client
        self.report_indexer = report_indexer

    def _fetch_similar_cases(self, accident_type: str, road_name: str,
                             top_k: int = 3) -> list[dict]:
        """유사 사례: 외부 RAG 지식베이스(DB 파일 플러그인) 우선, 우리 인덱서 폴백.

        RAG_DB_PATH에 협업자 DB가 있으면 그쪽 지식(차단시간 통계 등)을, 없으면 우리
        과거 보고서(report_indexer)를 사용. 둘 다 실패하면 빈 리스트(안전 무동작).
        """
        # 1) 외부 RAG 지식베이스 (있을 때만)
        try:
            from layer4_rag.rag_retriever import get_retriever
            rag = get_retriever()
            if rag.available:
                hits = rag.search_similar(accident_type=accident_type,
                                          road_name=road_name, top_k=top_k)
                if hits:
                    return hits
        except Exception as e:  # noqa: BLE001 — RAG 실패가 보고서 생성을 막지 않음
            logger.warning("RAG 지식베이스 검색 실패: %s", e)
        # 2) 우리 과거 보고서 인덱서 (폴백)
        if self.report_indexer is not None:
            try:
                return self.report_indexer.search_similar(
                    accident_type=accident_type, road_name=road_name, top_k=top_k)
            except Exception as e:  # noqa: BLE001
                logger.warning("유사 사례 검색 실패: %s", e)
        return []

    def generate(
        self,
        keyframes: list[np.ndarray],
        accident_info: dict[str, Any],
        tracks_meta: dict[str, Any],
        ic_info: dict[str, Any],
        similar_cases: list[dict] | None = None,
    ) -> dict:
        """사고 원인 추론 + 93컬럼 서식 보고서 생성.

        Args:
            keyframes: 사고 전후 키프레임 (최대 5장, 시간순).
            accident_info: {
                "accident_type": str,
                "severity": str,
                "involved_vehicles": list[dict],
                "timestamp_estimated": str,
            }
            tracks_meta: {
                "pre_accident_traffic": str | dict,
                "involved_vehicles_detail": str | list,
            }
            ic_info: {
                "ic_name": str,
                "road_name": str,
                "direction": str,
                "km": float,
                "branch": str,
            }
            similar_cases: 유사 과거 사고 사례 리스트 (report_indexer 검색 결과).

        Returns:
            93컬럼 서식의 JSON dict + raw_response.
        """
        # 유사 사례 검색: 외부 RAG 지식베이스(DB 파일 있으면) 우선 → 우리 보고서 인덱서 폴백
        if similar_cases is None:
            similar_cases = self._fetch_similar_cases(
                accident_info.get("accident_type", ""),
                ic_info.get("road_name", ""),
            )

        # 메타데이터 조립
        metadata = {
            "ic_name": ic_info.get("ic_name", ""),
            "road_name": ic_info.get("road_name", ""),
            "direction": ic_info.get("direction", ""),
            "km": ic_info.get("km", 0),
            "branch": ic_info.get("branch", ""),
            "accident_type": accident_info.get("accident_type", ""),
            "involved_vehicles_detail": _format_vehicles(
                tracks_meta.get("involved_vehicles_detail",
                                accident_info.get("involved_vehicles", []))
            ),
            "pre_accident_traffic": _format_traffic(
                tracks_meta.get("pre_accident_traffic", {})
            ),
            "weather": ic_info.get("weather", "맑음"),
            "similar_cases": _format_similar(similar_cases or []),
        }

        messages = build_messages("report", keyframes, metadata)
        response = self.client.chat(messages, images=keyframes)

        content = response["content"]
        report = self._validate(content, ic_info, accident_info)
        report["raw_response"] = response

        logger.info(
            "보고서 생성 완료: %s %s방향 %.1fkm, 유형=%s (%.1f초)",
            ic_info.get("road_name", "?"),
            ic_info.get("direction", "?"),
            ic_info.get("km", 0),
            accident_info.get("accident_type", "?"),
            response.get("latency_sec", 0),
        )

        return report

    def save_report(self, report_data: dict, output_path: Path | str | None = None) -> Path:
        """보고서 JSON 저장.

        Args:
            report_data: generate() 반환값.
            output_path: 저장 경로 (None이면 자동 생성).

        Returns:
            저장된 파일 경로.
        """
        if output_path is None:
            L3_REPORTS.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            road = report_data.get("위치", {}).get("노선명", "unknown")
            output_path = L3_REPORTS / f"report_{road}_{ts}.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # raw_response 제거 (저장 불필요한 대형 데이터)
        save_data = {k: v for k, v in report_data.items() if k != "raw_response"}

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        logger.info("보고서 저장: %s", output_path)
        return output_path

    @staticmethod
    def _validate(
        content: dict | str,
        ic_info: dict,
        accident_info: dict,
    ) -> dict:
        """MLLM 응답 검증 + 누락 필드 보완."""
        if isinstance(content, str):
            # JSON 파싱 실패 시 최소 골격 생성
            content = _build_skeleton(ic_info, accident_info, narrative=content)

        report = dict(content)

        # 필수 섹션 보장
        for section in ("시간", "위치", "사고 특성", "피해차량", "인명피해", "차로 피해", "교통 영향", "추론"):
            if section not in report:
                report[section] = {}

        # 위치 정보 — ic_info로 보완
        loc = report["위치"]
        if not loc.get("노선명"):
            loc["노선명"] = ic_info.get("road_name", "")
        if not loc.get("방향"):
            loc["방향"] = ic_info.get("direction", "")
        if not loc.get("이정"):
            loc["이정"] = ic_info.get("km", 0)
        if not loc.get("사고 관할 지사"):
            loc["사고 관할 지사"] = ic_info.get("branch", "")

        # 접보 유형 — 항상 CCTV
        characteristics = report["사고 특성"]
        characteristics["접보 유형"] = "CCTV"

        # 추론 섹션 보장
        reasoning = report["추론"]
        if not reasoning.get("cause_primary"):
            reasoning["cause_primary"] = ""
        if not isinstance(reasoning.get("cause_contributing"), list):
            reasoning["cause_contributing"] = []
        if not reasoning.get("narrative"):
            reasoning["narrative"] = ""
        if not isinstance(reasoning.get("recommendations"), list):
            reasoning["recommendations"] = []

        # confidence 범위
        try:
            conf = float(reasoning.get("confidence", 0.5))
            reasoning["confidence"] = max(0.0, min(1.0, conf))
        except (TypeError, ValueError):
            reasoning["confidence"] = 0.5

        return report


def _build_skeleton(
    ic_info: dict, accident_info: dict, narrative: str = ""
) -> dict:
    """JSON 파싱 실패 시 최소 보고서 골격 생성."""
    now = datetime.now()
    return {
        "시간": {
            "사고접보 시각(날짜)": now.strftime("%Y. %m. %d").replace(" 0", " "),
            "사고접보 시각(요일)": ["월", "화", "수", "목", "금", "토", "일"][now.weekday()],
            "사고접보 시각(시간)": now.strftime("%H:%M"),
            "안전순찰발 도착시각": None,
            "119 도착시각": None,
            "사고처리 완료시간": None,
        },
        "위치": {
            "노선명": ic_info.get("road_name", ""),
            "방향": ic_info.get("direction", ""),
            "이정": ic_info.get("km", 0),
            "사고 관할 지사": ic_info.get("branch", ""),
            "사고지점 유형": "본선",
        },
        "사고 특성": {
            "기상": ic_info.get("weather", "맑음"),
            "접보 유형": "CCTV",
            "사고 유형": "",
            "사고원인": "",
            "화재 여부": "미발생",
            "차량 전복/전도 여부": "미 전복/전도",
            "적재물 유출 여부": "미 유출",
            "적재물 유형_1": None,
        },
        "피해차량": {},
        "인명피해": {
            "인명피해(총수)": None,
            "사망자수": None,
            "중상자수": None,
            "경상자수": None,
        },
        "차로 피해": {
            "일방향 여부": "일방",
        },
        "교통 영향": {
            "정체길이": None,
            "도로피해_1": None,
            "시설물 피해_1": None,
        },
        "추론": {
            "cause_primary": "",
            "cause_contributing": [],
            "narrative": narrative,
            "recommendations": [],
            "confidence": 0.3,
            "reasoning": "MLLM JSON 파싱 실패 — 수동 검토 필요",
        },
    }


def _format_vehicles(vehicles: str | list) -> str:
    """관련 차량 정보를 프롬프트 텍스트로 변환."""
    if isinstance(vehicles, str):
        return vehicles
    if not vehicles:
        return "정보 없음"
    lines = []
    for v in vehicles:
        if isinstance(v, dict):
            tid = v.get("track_id", "?")
            cls = v.get("class", v.get("corrected_class", "?"))
            role = v.get("role", "?")
            speed = v.get("speed", "?")
            lines.append(f"  - Track {tid}: {cls} ({role}), 속도 {speed} km/h")
        else:
            lines.append(f"  - {v}")
    return "\n".join(lines)


def _format_traffic(traffic: str | dict) -> str:
    """사고 전 교통류 정보 변환."""
    if isinstance(traffic, str):
        return traffic
    if not traffic:
        return "정보 없음"
    parts = []
    for k, v in traffic.items():
        parts.append(f"{k}: {v}")
    return ", ".join(parts)


def _format_similar(cases: list[dict]) -> str:
    """유사 과거 사고 사례 변환."""
    if not cases:
        return "유사 사례 없음"
    lines = []
    for i, c in enumerate(cases, 1):
        date = c.get("date", "?")
        road = c.get("road_name", "?")
        cause = c.get("cause", "?")
        vtype = c.get("vehicles_summary", "?")
        lines.append(f"  사례 {i}: {date} {road}, 원인: {cause}, 차종: {vtype}")
    return "\n".join(lines)
