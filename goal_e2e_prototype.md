# E2E 프로토타입: 4개 레이어 연결 + DuckDB 적재

## 0. 목표 해부

- **What**: Vision(L1) → MLLM(L3) → DuckDB(L2) → 피처(L4) 4개 레이어를 하나의 실행 파이프라인으로 연결
- **Why**: Phase 0+1 게이트 통과. 개별 컴포넌트 동작은 확인됨. 이제 전체 흐름을 통합하여 프로토타입 완성
- **Scope**: 기존 코드(layer1~4) 통합 + import 수정 + 통합 러너 작성. 신규 모듈 최소화
- **Success Criteria**:
  1. 통합 러너: 클립 디렉토리 → Vision → MLLM → DuckDB 적재 → 1개 스크립트로 실행
  2. DuckDB에 tracks, mllm_responses, accidents 3개 테이블 적재 확인
  3. Layer4 FeatureEngineer가 DB에서 피처 벡터 추출 가능 (구조 검증)

**유형 분류**: 통합(기존 코드 연결) + 검증(DB 적재 확인)

## 1. 현황 진단

| 레이어 | 모듈 | 상태 | 잔여 이슈 |
|--------|------|------|----------|
| L1 Vision | vision_pipeline.py | ✅ 검증됨 | 분류기 더미 (EnsembleClassifier 미연결) |
| L2 Metadata | db_schema.py, metadata_writer.py | 미검증 | import 미테스트, DuckDB 연결 미확인 |
| L3 MLLM | mllm_client.py | ✅ 검증됨 | vehicles 필드 placeholder (3B 한계) |
| L4 Prediction | feature_engineer.py | 미검증 | DB 적재 후 테스트 가능 |

### 갭 분석
- **L2 import**: `from db_schema import init_db` — 패키지 import 시 relative import 필요할 수 있음
- **통합 러너 부재**: 각 레이어를 연결하는 오케스트레이터가 없음
- **DuckDB 파일**: `data/accident.duckdb` 미생성 상태

## 3. 실행 계획

### Step 1: Layer 2 검증 (10분) — ✅ 완료
- [x] DuckDB 연결 + 5개 테이블 스키마 생성 확인
- [x] MetadataWriter import 성공
- [x] relative import 수정: metadata_writer.py, report_indexer.py (layer2), risk_scorer.py (layer4)

### Step 2: 통합 러너 작성 (30분) — ✅ 완료
- [x] `run_pipeline.py` 작성: clip_dir → L1 Vision → L3 MLLM → L2 DuckDB
- [x] 처리 순서: PNG 로드 → YOLO/ByteTrack → 트리거 → MLLM → tracks/mllm_responses/accidents 적재
- [x] argparse CLI: `--video-id`, `--max-mllm` 옵션

### Step 3: 통합 테스트 (20분) — ✅ 완료
- [x] 클립 1건 (안전거리 미확보): 41 frames → 10 tracks, 2 triggers, MLLM 49s
- [x] DuckDB: tracks 10건, mllm_responses 1건, accidents 1건 적재
- [x] FeatureEngineer.build_features(): 19 피처 추출 성공 (hist_accident_rate=1.0)

### Step 4: 멀티 클립 검증 (15분) — ✅ 완료
- [x] 클립 2건 누적 실행 (안전거리 미확보 + 방향지시등 불이행)
- [x] DB 누적: tracks 12건(2 video), mllm_responses 2건, accidents 2건

## 4. 검증 체크리스트

- [x] A. DuckDB 5개 테이블 스키마 생성 확인
- [x] B. 통합 러너 에러 없이 클립 처리 완료
- [x] C. tracks 테이블 12건 적재 (2개 영상)
- [x] D. mllm_responses 테이블 2건 적재
- [x] E. FeatureEngineer.build_features() 에러 없이 19 피처 반환

## 5. 산출물

1. ✅ `run_pipeline.py` — 통합 러너 (CLI)
2. ✅ `data/accident.duckdb` — 2건 클립 적재
3. ✅ Layer4 FeatureEngineer 구조 검증 완료

## 6. 하지 말 것

- MLLM 프롬프트 최적화 (vehicles placeholder 개선 등)
- XGBoost 학습 (데이터 축적 전)
- EnsembleClassifier 연결 (분류기는 후속 과제)
- 대규모 배치 처리 (1~3건 데모로 충분)
- 성능 최적화 (MLLM 레이턴시 개선 등)
