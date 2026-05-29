# /goal 프롬프트: AI Hub #597 교통사고 영상 데이터 활용 파이프라인

아래 내용을 `/goal` 커맨드의 `$ARGUMENTS`로 사용.

---

## 목표

AI Hub #597 「교통사고 영상 데이터」를 다운로드하여, 우리 CCTV 사고분석 파이프라인에서 활용할 수 있도록 정제하고, 제공 모델(DetectoRS, VTN)을 포함한 사고 검출 + 차종 3분류(승용차/버스/트럭) 통합 파이프라인을 구축한다.

## 산출물

1. **데이터 정제 스크립트**: CCTV+3인칭만 필터링, 1인칭 제외
2. **모델 평가 코드**: DetectoRS 검출 + 차종 3분류 정확도 측정
3. **통합 추론 스크립트**: 사고 영상 입력 → 사고 검출 + 차종 분류 → 결과 JSON
4. **평가 리포트**: 검증셋 기준 성능 수치

## 배경

### 데이터셋 정보 (AI Hub #597)

- **이름**: 교통사고 영상 데이터
- **규모**: 영상 21,895건, 이미지 3,284,250장 (2021 구축, 2024-01 갱신)
- **포맷**: MP4 (10초 단위) + PNG 프레임 (15fps) + JSON 어노테이션
- **해상도**: ~3840×2160
- **분할**: train 80% / val 10% / test 10%
- **촬영**: CCTV (`filming_way="cc"`) + 블랙박스 (`filming_way="bb"`)
- **시점**: 1인칭 (`video_point_of_view=1`) / 3인칭 (`video_point_of_view=3`)
- **다운로드**: 분할 압축 → `find "폴더" -name "*.zip.part*" | xargs -0 cat > "파일.zip"` 으로 병합

### 어노테이션 구조

**영상 JSON**:
```json
{
  "video_name": "...",
  "filming_way": "cc",           // cc=CCTV, bb=블랙박스
  "video_point_of_view": 3,      // 1=1인칭, 3=3인칭
  "accident_type": 0~433,        // 사고유형 434종
  "accident_object": 0~3,        // 차대차/차대보행자/차대자전거/차대이륜차
  "accident_place": 0~14,        // 사고장소
  "accidental_negligence_rateA": 0~100,
  "accidental_negligence_rateB": 0~100
}
```

**이미지 JSON**:
```json
{
  "file_name": "...",
  "width": 3840, "height": 2160,
  "video_file_name": "...",
  "sequence_frame_number": 0~150,
  "objects": [
    {
      "bbox": [x, y, w, h],
      "category": "차량",         // 9종: 차량/보행자/이륜차/자전거/표지판/신호등(적/녹/기타)/횡단보도
      "isObjectA": true,          // 사고 당사자 A 여부
      "isObjectB": false,
      "isOtherObjects": false
    }
  ]
}
```

### 제공 모델

| 모델 | 용도 | 성능 |
|------|------|------|
| **DetectoRS** | 객체 검출 (bbox) | mAP 83.4% |
| **VTN** (Video Transformer Network) | 사고 상황 분류 (과실비율) | F1 0.7591 |

### 핵심 문제

데이터셋의 차량 클래스는 **"차량" 단일** — 승용차/버스/트럭 세분류가 없다. 따라서:
- 사고 검출 → DetectoRS (제공 모델) 또는 YOLO 활용
- 차종 3분류 → 별도 분류기 필요 (우리 기존 모델 활용 또는 새로 학습)

### 우리 기존 자산

| 자산 | 위치 | 내용 |
|------|------|------|
| YOLO 차량 검출 | `src/yolov8n.pt` | YOLOv8n, 범용 검출 |
| 차종분류 Triple Ensemble | `pipeline/` | 13종 분류 (v8lb_self 기반) |
| 사고분석 파이프라인 | `사고분석_설계/src/` | YOLO+ByteTrack+트리거+VLM |
| AI Hub #71566 (VL) | `/DATA/aihub_71566/` | 차선변경 비정상주행 (225GB, 21만장) |

## 데이터 필터 조건

### 사용할 것
- `filming_way == "cc"` (CCTV 영상)
- `filming_way == "bb"` AND `video_point_of_view == 3` (블랙박스 3인칭)

### 제외할 것
- `video_point_of_view == 1` (1인칭 블랙박스) — **전부 제외**

### 이유
우리 파이프라인은 고속도로 CCTV(고정 카메라, 3인칭 조감)에서 운용. 1인칭 블랙박스 영상은 뷰 특성이 완전히 달라 학습/검증 모두 부적합.

## 차종 3분류 전략

데이터셋에 차종 세분류가 없으므로, 다음 전략을 순서대로 검토:

### 전략 1: DetectoRS bbox → crop → 기존 차종분류기
1. DetectoRS로 "차량" bbox 검출
2. bbox crop
3. 우리 Triple Ensemble (13종) 또는 경량 분류기로 승용차/버스/트럭 3분류
4. 장점: 추가 학습 불필요, 즉시 적용 가능

### 전략 2: YOLO 기반 통합 (기존 파이프라인 유지)
1. YOLOv8n으로 차량 검출 (기존 파이프라인 그대로)
2. crop → 차종 분류
3. 사고 여부는 `isObjectA/B` 라벨로 별도 판정
4. 장점: 기존 파이프라인 호환

### 전략 3: 신규 학습 (차종 세분류 라벨 구축)
1. 데이터셋의 "차량" bbox를 crop
2. 일부(500~1000장)에 수동/반자동으로 승용차/버스/트럭 라벨 부여
3. 분류기 fine-tune
4. 장점: 사고 영상 특화 분류기

**우선 검토 순서**: 전략1 → 전략2 → 필요 시 전략3

## 실행 단계

### Phase 1: 데이터 다운로드 + 정제
1. AI Hub에서 데이터 다운로드 (분할 압축 병합)
2. AI Hub에서 **모델 파일** 다운로드 (DetectoRS, VTN)
3. 1인칭 필터링 스크립트 작성 — JSON의 `video_point_of_view` 체크
4. CCTV + 3인칭만 추출하여 정리
5. 통계: 필터 후 영상 수, 프레임 수, 사고유형 분포

### Phase 2: 모델 평가
1. DetectoRS 모델 로드 + 검증셋 추론
2. 검출 성능 측정 (mAP, 클래스별 AP)
3. 차종 3분류 전략 1 시도: DetectoRS crop → 기존 분류기 적용
4. 3분류 정확도 측정 (수동 샘플링 100장 검증)

### Phase 3: 통합 파이프라인
1. 사고 검출 + 차종 분류 통합 추론 스크립트
2. 입력: 영상/이미지 → 출력: 사고 bbox + 차종(승용차/버스/트럭) + 사고유형
3. 기존 사고분석 파이프라인(`run_nationwide.py`)과 연동 가능한 인터페이스

## 파일 구조

```
/workspace/prj_cctv/사고분석_설계/
├── data/
│   └── aihub_597/
│       ├── raw/              ← 다운로드 원본
│       ├── filtered/         ← CCTV+3인칭만 필터링
│       │   ├── videos/
│       │   ├── images/
│       │   └── labels/
│       ├── models/           ← DetectoRS, VTN 모델 파일
│       └── stats.json        ← 필터 후 통계
├── src/
│   ├── filter_aihub597.py    ← 1인칭 제외 필터 스크립트
│   ├── eval_detectors.py     ← DetectoRS 평가
│   ├── eval_vehicle_cls.py   ← 차종 3분류 평가
│   └── infer_accident.py     ← 통합 추론
└── docs/
    └── aihub597_평가리포트.md
```

## 저장 경로

- 원본 다운로드: `/media/ybs/Expansion/CCTV차종분류/accident_data/aihub_597/` (외장 15TB)
- 작업 데이터: `/workspace/prj_cctv/사고분석_설계/data/aihub_597/`
- 대용량 원본은 외장에, 필터링된 서브셋만 workspace로 심볼릭 링크

## 하지 말 것

- 1인칭 영상 포함 금지
- 전체 데이터 workspace에 복사 금지 (NVMe 용량 주의, 심볼릭 링크 사용)
- DetectoRS/VTN 모델을 처음부터 재학습하지 말 것 — 제공 모델 먼저 평가
- 차종 분류를 위해 전체 데이터에 수동 라벨링하지 말 것 — 전략 1,2 먼저 시도

## 성공 기준

1. CCTV+3인칭 영상만 필터링 완료 + 통계 산출
2. DetectoRS 모델 로드 + 검증셋 10건 이상 추론 성공
3. 차종 3분류 (승용차/버스/트럭) 추론 가능한 파이프라인 동작
4. 통합 추론 스크립트: 영상 1건 입력 → 사고 검출 + 차종 분류 JSON 출력
5. 평가 리포트 작성
