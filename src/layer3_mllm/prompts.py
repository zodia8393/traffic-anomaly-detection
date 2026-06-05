"""MLLM 프롬프트 템플릿 — 4개 태스크."""
import sys as _sys; from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from config_new import VEHICLE_13CLASS

# ── Task 1: 장면 이해 ────────────────────────────────────────────────

SCENE_UNDERSTANDING_PROMPT = """\
당신은 교통 CCTV 영상 분석 전문가입니다.

아래는 현재 CCTV 영상에서 추출한 교통 상황 데이터입니다:
- 위치: {ic_name} ({lane_count}차로, {road_type})
- 검출 차량: {vehicle_count}대
- 차종 분포: {class_distribution}
- 평균 속도: {avg_speed} km/h
- 트리거: {trigger_type} — {trigger_description}

이 장면의 교통 상황을 분석하고 JSON으로 응답하시오:
{{
  "scene_description": "장면 설명",
  "traffic_state": "free_flow | congested | incident | stopped",
  "anomalies": ["이상 상황 목록"],
  "risk_level": "low | medium | high | critical"
}}"""

# ── Task 2: 사고 감지 ────────────────────────────────────────────────

ACCIDENT_DETECTION_PROMPT = """\
당신은 교통사고 감지 전문가입니다.

제공된 CCTV 키프레임은 **시간순으로 정렬**되어 있습니다(첫 장→마지막 장이 시간 경과).
프레임 간 차량 위치 변화를 따라가며 충돌 순간을 찾으시오.

아래 센서 데이터는 **참고용 힌트**이며 사고를 의미하지 않습니다.
트리거는 "이상 의심" 신호일 뿐이므로, **반드시 영상(키프레임)에서 충돌·전복·정차·파손이
실제로 보이는지 확인**한 뒤 판정하시오:
- 트리거: {trigger_type}
- 관련 차량 궤적:
  {track_summaries}
- TTC: {ttc_values}
- 속도 변화: {speed_changes}

판정 규칙 (엄수):
1) 영상에서 충돌/접촉/전복/이상정차/파손이 **명백히 보일 때만** accident_detected=true.
2) 힌트만 있고 영상에 사고 증거가 없으면 accident_detected=false, accident_type="none".
3) 불확실하면 confidence를 낮게(<0.5) 주고, 추측으로 사고를 단정하지 마시오.
4) involved_vehicles의 track_id는 위 궤적 데이터에 존재하는 ID만 사용(없는 ID 생성 금지).
5) reasoning에는 **몇 번째 프레임에서 무엇을 보고** 판단했는지 시각적 근거를 명시.

아래 JSON 형식으로만 응답(필드/enum 정확히 준수):
{{
  "accident_detected": true 또는 false,
  "confidence": 0.0~1.0 사이 숫자,
  "accident_type": "rear_end | sideswipe | rollover | head_on | fixed_object | none 중 하나",
  "severity": "minor | moderate | severe | fatal 중 하나",
  "involved_vehicles": [{{"track_id": 정수, "role": "striking | struck"}}],
  "timestamp_estimated": "HH:MM:SS",
  "reasoning": "프레임 기반 시각적 판단 근거"
}}"""

# ── Task 3: 차종 분류 보정 ───────────────────────────────────────────

# 13종 분류 체계 텍스트 (프롬프트 삽입용)
_CLASS_TABLE = """한국도로공사 13종 분류 체계 (차축 기반):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1단위(단일차체):
  T1  = 승용차·미니트럭 (2축, 소형)
  T2  = 버스 (2축, 창문 다수)
  T3  = 소형화물 1~2.5t 미만 (2축, 포터/봉고급)
  T4  = 중형화물 2.5~8.5t 미만 (2축, 마이티급)
  T5  = 대형화물 8.5t+ (3축, 후축 복륜)
  T6  = 대형특수 (4축, 레미콘/대형덤프)
  T7  = 대형특수 (5축)

2단위(트랙터+트레일러 또는 트럭+트레일러):
  T8  = 세미트레일러 (4축)
  T9  = 풀트레일러 (4축)
  T10 = 세미트레일러 (5축, 40ft 컨테이너)
  T11 = 풀트레일러 (5축)
  T12 = 세미트레일러 (6축, 초대형)

이륜:
  T13 = 이륜차 (오토바이, 스쿠터)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

CLASS_CORRECTION_PROMPT = f"""\
당신은 한국도로공사 교통량조사 차종분류 전문가입니다.

Vision 모델의 1차 분류 결과: {{vision_class}} (신뢰도: {{confidence}})
차량 메타데이터: bbox 면적 {{area}}px², 종횡비 {{aspect_ratio}}

{_CLASS_TABLE}

판단 순서:
1) 2단위(연결부 꺾임) 여부 → T8~T12
2) 이륜(2륜, 매우 작음) → T13
3) 창문 패턴(승객용) → T2(버스)
4) 적재함 유무·축 수로 T1/T3~T7 구분

JSON으로 응답:
{{{{
  "corrected_class": "T코드",
  "confidence": 0.0~1.0,
  "unit_count": 1 | 2,
  "axle_count": 2~6,
  "vision_agreed": true/false,
  "reasoning": "판단 근거 (차축, 차체형태, 크기 기반)"
}}}}"""

# ── Task 4: 사고 원인 추론 + 보고서 (93컬럼) ────────────────────────

REPORT_GENERATION_PROMPT = """\
당신은 교통사고 분석 전문가입니다. 한국도로공사 사고보고서 표준 서식에 맞춰 분석합니다.

사고 상황 데이터:
- CCTV 위치: {ic_name} ({road_name}, {direction}방향, 이정 {km}km)
- 사고 유형: {accident_type}
- 관련 차량:
  {involved_vehicles_detail}
- 사고 전 30초 교통류:
  {pre_accident_traffic}
- 기상: {weather}
- 유사 과거 사고 사례:
  {similar_cases}

도로공사 사고보고서 표준 서식(93컬럼)에 맞춰 JSON으로 작성하시오.
CCTV 영상에서 판단 가능한 항목만 채우고, 현장 확인 필요 항목은 null로 표기:

{{
  "시간": {{
    "사고접보 시각(날짜)": "YYYY. M. D",
    "사고접보 시각(요일)": "월~일",
    "사고접보 시각(시간)": "HH:MM",
    "안전순찰발 도착시각": null,
    "119 도착시각": null,
    "사고처리 완료시간": null
  }},
  "위치": {{
    "노선명": "{road_name}",
    "방향": "{direction}",
    "이정": {km},
    "사고 관할 지사": "{branch}",
    "사고지점 유형": "JC | IC | 터널 | 본선 | 진출입"
  }},
  "사고 특성": {{
    "기상": "맑음 | 흐림 | 비 | 눈 | 안개",
    "접보 유형": "CCTV",
    "사고 유형": "부분차단 | 전면차단 | 갓길",
    "사고원인": "주시태만 | 졸음운전 | 과속 | 안전거리 미확보 | 차로변경 | 기타",
    "화재 여부": "미발생 | 화재",
    "차량 전복/전도 여부": "미 전복/전도 | 전복 | 전도",
    "적재물 유출 여부": "미 유출 | 유출",
    "적재물 유형_1": null
  }},
  "피해차량": {{
    "피해차량_1": "승용차 | 화물차 | 버스 | 승합차",
    "피해차량_2": "...",
    "피해차량_N": "... (최대 10대)"
  }},
  "인명피해": {{
    "인명피해(총수)": null,
    "사망자수": null,
    "중상자수": null,
    "경상자수": null
  }},
  "차로 피해": {{
    "일방향 여부": "일방 | 양방",
    "사고방향 피해 차로_1": "유 | 무",
    "사고방향 피해 차로_2": "유 | 무",
    "사고방향 피해 차로_3": "유 | 무",
    "사고방향 피해 차로_갓길": "유 | 무"
  }},
  "교통 영향": {{
    "정체길이": "km (정수)",
    "도로피해_1": "차로명",
    "시설물 피해_1": null
  }},
  "추론": {{
    "cause_primary": "주 원인",
    "cause_contributing": ["기여 요인 목록"],
    "narrative": "사고 경위 서술 (3~5문장)",
    "recommendations": ["재발 방지 권고사항"],
    "confidence": 0.0~1.0,
    "reasoning": "MLLM 판단 근거"
  }}
}}"""

# ── Task 5: 돌발정보 사고 현장 분석 (Outbreak) ──────────────────────

OUTBREAK_REPORT_PROMPT = """\
당신은 한국도로공사 교통사고 분석 전문가입니다.
ITS 돌발정보 시스템에서 사고로 확인된 현장의 CCTV 영상 프레임을 분석합니다.

사고 기본정보 (ITS 돌발정보):
- 노선명: {road_name}
- 방향: {direction}
- 돌발 유형: {incident_type}
- 사고 접보 시각: {incident_time}
- 사고 후 경과: {elapsed_min}분
- CCTV: {cctv_name} ({cctv_distance})

제공된 CCTV 프레임({frame_count}장)을 분석하여 아래 항목을 JSON으로 작성하시오.
영상에서 판단 불가능한 항목은 null로 표기:

{{
  "weather": "맑음 | 흐림 | 비 | 눈 | 안개",
  "blockage_type": "전면차단 | 부분차단 | 갓길",
  "blocked_lanes": [1, 2],
  "lane_count": 2,
  "road_geometry": "직선 | 곡선 | 터널 | 교량",
  "vehicle_count": 1,
  "vehicles": [
    {{
      "type": "승용차 | 화물차 | 버스 | 승합차 | 이륜차 | 특수차",
      "size": "소형 | 중형 | 대형",
      "cargo": "적재물 종류 또는 null",
      "damage_state": "파손 | 전복 | 전도 | 정차 | 기타"
    }}
  ],
  "fire": false,
  "rollover": false,
  "cargo_spill": false,
  "cargo_type": null,
  "facility_damage": [
    {{"type": "가드레일 | 중분대 | 방음벽 | 표지판 | 기타", "detail": "설명"}}
  ],
  "cause_estimated": "주시태만 | 졸음운전 | 과속 | 안전거리미확보 | 차로변경 | 차량결함 | 낙하물 | 기상 | 기타",
  "description": "사고 경위 추정 (3~5문장, 영상 기반)",
  "severity": "경미 | 보통 | 중대 | 심각",
  "confidence": 0.7
}}"""

# ── 시스템 프롬프트 ──────────────────────────────────────────────────

_SYSTEM_PROMPTS: dict[str, str] = {
    "scene": (
        "당신은 교통 CCTV 영상 분석 전문가입니다. "
        "제공된 키프레임 이미지와 메타데이터를 기반으로 현재 교통 상황을 정확히 분석합니다. "
        "반드시 JSON 형식으로만 응답하시오."
    ),
    "accident": (
        "당신은 교통사고 감지 전문가입니다. "
        "CCTV 키프레임(시간순)에서 보이는 시각적 증거를 최우선으로 사고를 판단합니다. "
        "센서/트리거 데이터는 참고용 힌트일 뿐이며, 영상에 사고가 보이지 않으면 사고가 아닙니다. "
        "추측으로 사고를 단정하지 말고, 불확실하면 낮은 confidence로 표기하시오. "
        "반드시 JSON 형식으로만 응답하시오."
    ),
    "classify": (
        "당신은 한국도로공사 교통량조사 차종분류 전문가입니다. "
        "Vision 모델의 1차 분류를 검증하고, 차축 수와 차체 형태를 기반으로 "
        "한국도로공사 13종(T1~T13) 분류 체계에 맞게 보정합니다. "
        "반드시 JSON 형식으로만 응답하시오."
    ),
    "report": (
        "당신은 교통사고 분석 전문가입니다. "
        "한국도로공사 사고보고서 표준 서식(93컬럼)에 맞춰 사고 원인을 추론하고 보고서를 작성합니다. "
        "CCTV에서 판단 가능한 항목만 채우고, 현장 확인 필요 항목은 null로 표기하시오. "
        "반드시 JSON 형식으로만 응답하시오."
    ),
    "outbreak": (
        "당신은 한국도로공사 교통사고 현장 분석 전문가입니다. "
        "ITS 돌발정보에서 사고로 확인된 CCTV 영상을 분석하여 "
        "전면차단 사고 보고서 양식에 맞는 현장 정보를 추출합니다. "
        "반드시 JSON 형식으로만 응답하시오."
    ),
}

# 태스크 → 프롬프트 템플릿 매핑
_TASK_PROMPTS: dict[str, str] = {
    "scene": SCENE_UNDERSTANDING_PROMPT,
    "accident": ACCIDENT_DETECTION_PROMPT,
    "classify": CLASS_CORRECTION_PROMPT,
    "report": REPORT_GENERATION_PROMPT,
    "outbreak": OUTBREAK_REPORT_PROMPT,
}


def build_messages(
    task: str,
    images: list | None = None,
    metadata: dict | None = None,
) -> list[dict]:
    """태스크별 프롬프트를 메시지 리스트로 조립.

    Args:
        task: "scene" | "accident" | "classify" | "report"
        images: 키프레임 이미지 리스트 (numpy 또는 base64).
        metadata: 프롬프트 플레이스홀더에 채울 메타데이터.

    Returns:
        OpenAI 호환 messages 리스트.
    """
    if task not in _TASK_PROMPTS:
        valid = ", ".join(_TASK_PROMPTS)
        raise ValueError(f"알 수 없는 태스크: {task} (유효: {valid})")

    meta = dict(metadata or {})
    template = _TASK_PROMPTS[task]

    # 미보정 속도 제외: 캘리브레이션이 없으면 절대속도(km/h)는 물리적 무의미하므로
    # MLLM 입력에서 '측정불가(미보정)'로 표시 (오판 유발 방지).
    # speed_calibrated 미지정(None)이면 기존 동작 유지(하위호환).
    if meta.get("speed_calibrated") is False:
        meta["avg_speed"] = "측정불가(미보정)"
        meta["speed_changes"] = meta.get("speed_changes") or "측정불가(미보정)"

    # 메타데이터로 플레이스홀더 채우기 (누락 키는 빈 문자열)
    try:
        user_text = template.format_map(_SafeDict(meta))
    except (KeyError, IndexError):
        user_text = template

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPTS[task]},
        {"role": "user", "content": user_text},
    ]

    return messages


class _SafeDict(dict):
    """format_map에서 누락 키를 빈 문자열로 대체."""

    def __missing__(self, key: str) -> str:
        return ""
