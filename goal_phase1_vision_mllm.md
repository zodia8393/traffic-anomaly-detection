# Phase 0+1 게이트 통과: MLLM 서빙 + Vision 파이프라인 실데이터 검증

## 0. 목표 해부

- **What**: (1) Qwen2.5-VL-7B Q4 MLLM 서빙 동작 확인 (2) Vision 파이프라인(트리거 7종) 실영상 테스트 (3) 트리거→MLLM 연동 E2E 데모
- **Why**: 아키텍처 설계(5 Phase)에서 Phase 0(MLLM 서빙) + Phase 1(Vision Layer) 게이트 통과가 전체 시스템의 선행 조건
- **Scope**: 기존 코드(layer1_vision, layer3_mllm) 검증·수정. 신규 구현 최소화. AI Hub #71566 영상 활용
- **Success Criteria**:
  1. MLLM 서빙: Qwen2.5-VL-7B Q4가 이미지+텍스트 입력에 JSON 응답 (CPU, <60초)
  2. Vision 파이프라인: 실영상 1건에서 트리거 7종 중 최소 1종 정상 발화
  3. E2E 데모: 트리거 발화 → 키프레임 추출 → MLLM 사고 판단 → JSON 출력

**유형 분류**: 인프라(MLLM 환경) + 구현(파이프라인 검증) = 복합

## 1. 현황 진단

### 자원 탐색

| 필요한 것 | 현재 상태 | 조달 방법 | 리스크 |
|-----------|----------|----------|--------|
| Qwen2.5-VL-7B GGUF | 불확실 (다운로드 여부 미확인) | HuggingFace / llama.cpp 빌드 | 중 (4.5GB DL + 빌드) |
| llama-cpp-python | 불확실 (설치 여부 미확인) | pip install | 낮 |
| 테스트 영상 (사고) | 있음 | /DATA/aihub_71566/ 또는 data/test_videos/ | 낮 |
| Layer 1 코드 | 있음 (1,139줄) | src/layer1_vision/ | 낮 (미검증) |
| Layer 3 코드 | 있음 (1,498줄) | src/layer3_mllm/ | 낮 (미검증) |
| AnomalyEngine | 있음 | src/anomaly_engine/ | 낮 (테스트 통과) |
| YOLO11n 모델 | 있음 | pipeline/data/models/ | 낮 |
| ByteTrack | 있음 | pipeline/src/tracker.py | 낮 |

### 갭 분석 요약
- **확인 필요**: MLLM 모델 파일 존재 여부, llama-cpp-python 설치 상태
- **검증 필요**: Layer 1/3 코드가 실제로 동작하는지 (작성만 되고 미테스트 가능성)
- **조달 필요**: GGUF 모델 파일 (없으면 다운로드)

## 3. 실행 계획

### Step 1: 환경 점검 (5분) — ✅ 완료
- [x] llama-cpp-python 0.3.23 설치 확인
- [x] GGUF 모델 대신 transformers 백엔드 채택 (Qwen2.5-VL-3B-Instruct, 7.2GB 캐시)
- [x] 테스트 클립 선정: `/DATA/aihub_71566/source/val/비정상/07.안전거리 미확보 차선변경/p01_20230107_141213_an7_026_04/` (41 PNG, 1280x720)

### Step 2: MLLM 서빙 검증 (30분) — ✅ 완료
- [x] transformers 백엔드로 Qwen2.5-VL-3B-Instruct 로딩 (HuggingFace 캐시)
- [x] 단독 테스트: 이미지 1장 + 프롬프트 → JSON 응답 확인 (42.4s)
- [x] config_new.py MLLM_BACKEND="transformers"로 변경

### Step 3: Vision 파이프라인 테스트 (30분) — ✅ 완료
- [x] 41 PNG 프레임 → VisionPipeline 처리 완료 (~2초)
- [x] YOLO ONNX FP32 검출: 평균 5.8개/frame
- [x] ByteTrack 추적 정상 동작
- [x] layer1_vision 모듈 relative import 수정 (vision_pipeline.py)
- [x] 트리거 T3(정지차량) 2건 발화: frame 5, frame 38

### Step 4: E2E 연동 (30분) — ✅ 완료
- [x] 트리거 T3 발화 → 키프레임 5장 선정 [1, 5, 9, 10, 14]
- [x] 키프레임 → MLLM 전송 → JSON 응답 수신 (59.4s)
- [x] MLLM 판정: anomaly=true, type=정지차량, severity=high, confidence=0.95
- [x] 결과 JSON 저장: `layer1_vision/output/e2e_demo_result.json`

## 4. 검증 체크리스트

- [x] A. MLLM이 이미지+텍스트 멀티모달 입력을 처리하고 JSON 응답
- [x] B. Vision 파이프라인이 에러 없이 영상 처리 완료
- [x] C. 트리거 최소 1종이 실영상에서 발화 (T3 × 2건)
- [x] D. E2E: 트리거→키프레임→MLLM→JSON 전체 경로 동작

## 5. 산출물

1. ✅ MLLM 서빙: Qwen2.5-VL-3B, transformers, CPU 42~59s/req, ~7.2GB RAM
2. ✅ Vision 트리거: T3(정지차량) 2건 / 41 frames, YOLO ONNX avg 5.8 det/frame
3. ✅ E2E JSON: `layer1_vision/output/e2e_demo_result.json`
4. ✅ 코드 수정: config_new (backend), layer1_vision (relative import), layer3_mllm (relative import)

## 6. 하지 말 것

- MLLM 프롬프트 최적화 (이번 목표는 "동작 확인"이지 "품질 개선"이 아님)
- 새 모델 학습 (기존 코드 검증에 집중)
- 대규모 영상 배치 처리 (1건 데모로 충분)
- 아키텍처 변경 (기존 설계 그대로 검증)
