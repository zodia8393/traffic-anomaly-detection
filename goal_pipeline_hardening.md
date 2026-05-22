# E2E 파이프라인 경화: 정확도·속도·안정성 동시 개선

## 0. 목표 해부

- **What**: 현재 4계층 파이프라인(L1 Vision → L3 MLLM → L2 DuckDB → L4 Prediction)의 3대 축 개선
  - 정확도: 비정상 판별력 0% → 50%+ (MLLM 프롬프트 + 멀티프레임)
  - 속도: 클립당 70~425s → 30s 이하 (모델 재로드 제거 + max_tokens 제한)
  - 안정성: UTF-8 크래시, 빈 리스트 인덱싱, 저해상도 폭주 등 런타임 에러 제로화
- **Why**: V2 배치 결과 — 정상 판별 6/7 성공이나 비정상 판별 0/7 실패. N6 저해상도 클립에서 425초 폭주 + UTF-8 크래시. MLLM 모델 매 클립 재로드로 8s×14=112s 낭비
- **Scope**: run_pipeline.py + mllm_client.py + test_batch_14.py + metadata_writer.py 수정. 모델 교체·학습 없음
- **Success Criteria**:
  1. 14건 배치 에러 0건 완주
  2. 정상 7건 중 5건+ anomaly=false
  3. 비정상 7건 중 3건+ anomaly=true (현재 0건 → 개선)
  4. 총 소요시간 ≤ 15분 (현재 ~20분+)
  5. 전 클립 vehicles 실제값 채워짐 (placeholder 없음)

**유형 분류**: 코드 경화 + 프롬프트 엔지니어링

## 1. 현황 진단

### V2 배치 결과 (14건 중 10건 확인)

| 구분 | 결과 | 문제 |
|------|------|------|
| N1~N5, N7 정상 6건 | 전부 anomaly=False ✓ | 없음 |
| N6 정체구간 정상 | 크래시 → None | 352x288 저해상도 → 2048토큰 폭주 → UTF-8 에러 (425초) |
| A1 방향지시등 불이행 | anomaly=False ✗ | 단일 프레임으로 방향지시등 판별 불가 |
| A2 실선구간 차선변경 | anomaly=False ✗ | 차선 유형(실선/점선) 판별 실패 |
| A3~ | 진행중 | 대부분 False 예상 |

### 결함 3축 분류

| 축 | 결함 | 원인 | 해결 방향 |
|----|------|------|----------|
| **정확도** | 비정상 전부 False | 단일 프레임 분석 한계, 프롬프트에 비교 관점 부재 | 멀티프레임(3장) 입력 + 차선변경 전후 비교 프롬프트 |
| **정확도** | 트리거 T3만 발화 | 속도 fallback 0.05m/px → 전부 저속 | 트리거 불요, force_trigger 기본화 |
| **속도** | 클립당 모델 재로드 8s | MLLMClient 매번 new | 모델 싱글턴 (루프 밖 1회 로드) |
| **속도** | N6 425초 폭주 | 저해상도 + max_tokens=2048 제한 없음 | max_new_tokens=512 제한 + 타임아웃 |
| **안정성** | UTF-8 크래시 | MLLM raw 텍스트에 invalid 바이트 | metadata_writer에서 sanitize |
| **안정성** | 빈 리스트 IndexError | mllm_anomaly 빈 리스트 [0] 접근 | 이미 수정됨 (V3) |
| **안정성** | JSON 파싱 실패 시 raw text 그대로 저장 | 파싱 실패 = 무의미 데이터 | 파싱 실패 시 에러 마커 dict 반환 |

## 2. 하지 말 것

- MLLM 모델 교체 (3B → 7B 등) — CPU 환경에서 비현실적
- 속도 캘리브레이션 — 이번 목표 아님
- EnsembleClassifier 연결 — 다음 단계
- 새로운 테이블/스키마 추가 — 기존 구조 활용
- 14건 넘는 대규모 배치 — 유형별 1건으로 충분

## 3. 실행 계획

### Step 1: MLLM 속도 최적화 (15분)

**목표**: 모델 재로드 제거 + max_tokens 축소 → 클립당 MLLM 시간 60s → 30s

1-1. `run_pipeline.py` — MLLMClient를 루프 밖에서 1회만 생성
```
현재: for clip in clips: mllm = MLLMClient(backend="transformers")  # 매번 로드
개선: mllm = MLLMClient(backend="transformers")  # 1회
      for clip in clips: mllm.chat(...)
```

1-2. `config_new.py` — MLLM_MAX_TOKENS 2048 → 512 (JSON 응답은 200토큰이면 충분)

1-3. `mllm_client.py` — transformers 백엔드에 timeout 장치 추가
- generate() 호출 시 max_new_tokens=min(max_tokens, 512)
- 120초 초과 시 강제 중단 (threading.Timer)

**검증**: N6 클립 단독 실행 → 60초 이내 완료, 크래시 없음

### Step 2: MLLM 안정성 경화 (15분)

**목표**: 어떤 입력이든 크래시 없이 완주

2-1. `metadata_writer.py:write_mllm_response()` — output_json 필드 sanitize
```python
if isinstance(output_json, str):
    output_json = output_json.encode("utf-8", errors="replace").decode("utf-8")
```

2-2. `mllm_client.py:_parse_json_response()` — 파싱 실패 시 에러 마커 dict 반환
```python
# 현재: return text (raw string)
# 개선: return {"error": "json_parse_failed", "raw_length": len(text), "anomaly": None}
```

2-3. `run_pipeline.py` — MLLM 호출 전 이미지 최소 해상도 체크
- 352x288 이하 → 640x480으로 리사이즈 후 전달 (3B 모델 입력 품질 향상)

**검증**: N6 클립 정상 완주, DB 적재 성공

### Step 3: 비정상 판별력 개선 (20분)

**목표**: 멀티프레임 입력 + 프롬프트 고도화 → 비정상 3/7+ 판별

3-1. `run_pipeline.py` — MLLM에 멀티프레임 입력 (단일 → 3장)
- 트리거 프레임 기준 -5, 0, +5 프레임 3장을 한번에 전달
- "이전/현재/이후 프레임을 비교하여 변화를 분석하세요" 지시

3-2. SCENE_PROMPT 개선
```
현재: "이 영상 프레임을 분석하세요"
개선: "3장의 연속 프레임(이전→현재→이후)을 비교 분석하세요.
      특히 다음을 확인하세요:
      - 차량의 차선변경 여부와 방향지시등 사용
      - 차선 유형(실선/점선) 위반
      - 급격한 속도 변화나 비정상 거동
      정상 교통류라면 anomaly=false, 위험 상황이면 anomaly=true"
```

3-3. few-shot 예시 2건 추가 (정상 1건 + 비정상 1건)
- 현재 비정상 예시만 있어 모델이 True 편향 또는 예시 복사

**검증**: A2(실선 차선변경), A4(차선 물기) 등 시각적으로 명확한 유형에서 True 출력

### Step 4: test_batch_14.py 경화 + 재실행 (20분)

4-1. `test_batch_14.py` 개선
- MLLMClient 외부 1회 생성 → run() 함수에 전달 (시그니처 확장)
- 각 클립별 try/except + 에러 시 continue
- 결과 JSON에 에러 정보 포함

4-2. DB 초기화 → 14건 전체 재실행

4-3. 결과 분석
- 정확도 테이블, 소요시간 비교 (V2 vs V3)

**검증**: 14건 전체 완주, 에러 0건

### Step 5: 결과 정리 + 커밋 (10분)

5-1. V2 vs V3 비교표 출력
5-2. goal_prototype_v2.md 완료 표시
5-3. 변경 파일 전체 커밋

## 4. 검증 체크리스트

- [ ] A. 14건 배치 에러 0건 완주
- [ ] B. 정상 클립 anomaly=false ≥ 5/7
- [ ] C. 비정상 클립 anomaly=true ≥ 3/7
- [ ] D. 총 소요시간 ≤ 15분
- [ ] E. N6 저해상도 클립 정상 처리 (크래시 없음)
- [ ] F. vehicles 실제값 채워진 비율 ≥ 10/14
- [ ] G. DuckDB 14건 적재 완료

## 5. 산출물

1. 경화된 run_pipeline.py (멀티프레임, 싱글턴 MLLM, 에러 핸들링)
2. 경화된 mllm_client.py (타임아웃, JSON 에러 마커, max_tokens 제한)
3. 경화된 metadata_writer.py (UTF-8 sanitize)
4. 경화된 test_batch_14.py (에러 내성, 외부 MLLM 인스턴스)
5. 14건 배치 결과 JSON + V2 vs V3 비교표
6. 커밋
