# /goal 프롬프트: CCTV 사고영상 자동 수집 — 징조 기반 녹화

아래 내용을 `/goal` 커맨드의 `$ARGUMENTS`로 사용.

---

## 목표

사고 다발 구간의 CCTV를 상시 **분석만** 하다가, AI가 사고 징조를 포착한 순간부터 녹화를 시작하여 **징조 → 사고 발생 → 사고 후처리** 전체 시퀀스를 담은 영상을 자동 수집한다.

## 핵심 설계: 분석과 녹화의 분리

```
[Phase 1: 상시 분석 — 녹화 안 함]
CCTV HLS 스트림 → FrameSampler(1fps) → Vision Pipeline(YOLO+ByteTrack+트리거)
  - 프레임만 분석, 디스크에 저장하지 않음
  - 트리거 발화 감시 (T1 TTC, T3 정지, T4 역주행, T5 급차선변경 등)

[Phase 2: 징조 포착 → 녹화 시작]
트리거 발화 = "사고 징조 감지"
  - 그 즉시 ffmpeg 녹화 시작 (HLS 스트림 → mp4)
  - Vision Pipeline 분석 계속 (추가 트리거, 사고 확대 감시)
  - 녹화 지속 시간: 최소 3분, 최대 5분 (사고 후처리까지 포착)

[Phase 3: 사고 확인 + 저장/폐기]
녹화 종료 후:
  - ITS 돌발상황 API로 해당 구간 실제 사고 교차확인
  - 사고 확인 → 영상 보존 + 메타데이터 기록 (징조→사고→후처리 완전 영상)
  - 미확인 → 영상 파일 삭제 (폐기)
  - 쿨다운: 동일 CCTV에서 10분 이내 재녹화 방지
```

## 최종 영상 내러티브

```
[0:00 ~ 0:30]  사고 징조: 급감속, 차간거리 급축, 급차선변경 등
[0:30 ~ 1:00]  사고 발생: 충돌, 추돌, 전복 등
[1:00 ~ 3:00+] 사고 후처리: 정차, 대기, 긴급차량 진입 등
```

AI가 징조를 포착한 시점이 영상의 시작이므로, 영상만 봐도 "왜 사고가 났는지"부터 알 수 있다.

## 해야 할 것

### 1. run_realtime.py 재설계
- 기존 Ring Buffer 상시녹화 방식 → **징조 기반 온디맨드 녹화** 방식으로 전환
- `FrameSampler`: 기존과 동일 (1fps 프레임 추출, 분석 전용)
- `OnDemandRecorder`: 신규 — 트리거 시점에 ffmpeg 녹화 시작, N분 후 종료
  - 입력: CCTV 스트림 URL, 녹화 시간
  - 출력: mp4 파일 (징조 시점 ~ N분 후)
  - Ring Buffer 불필요 (징조 포착 = 녹화 시작이므로 "이전 영상"이 필요 없음)
- `IncidentVerifier`: 기존과 동일 (ITS API 교차확인)
- 녹화 중에도 Vision Pipeline 분석 계속 (추가 트리거 발화 시 녹화 연장 가능)

### 2. 다중 CCTV 동시 감시 유지
- 사고 다발 구간 반경 내 CCTV N대 동시 **분석**
- 녹화는 트리거 발화한 카메라만 (전체 녹화 X)
- 동일 사고에 여러 카메라 트리거 → incident_id 그룹핑 유지

### 3. 트리거 = 사고 징조 (기존 트리거 활용)
- T1: TTC 임박 (충돌 위험) → 사고 전조
- T3: 장기 정지 (고속도로 위 정차) → 사고 또는 고장
- T4: 역주행 → 즉시 위험
- T5: 급차선변경 → 사고 전조
- T7: 주기적 스냅샷 → 녹화 트리거로 사용 안 함 (제외)

### 4. 검증
- 트리거 발화 시 녹화 시작 → mp4 파일 생성 확인
- 녹화 종료 후 ITS 확인 → 사고면 보존, 아니면 삭제 확인
- 다중 카메라 동시 분석 중 특정 카메라만 녹화 시작 확인
- 최종 영상에 징조 시점부터 담겨있는지 확인

## 기존 코드

| 파일 | 역할 | 수정 범위 |
|------|------|----------|
| `src/layer0_data/run_realtime.py` | 통합 파이프라인 | SegmentRingBuffer → OnDemandRecorder로 교체 |
| `src/layer0_data/track3_api_incident.py` | ITS API | 변경 없음 |
| `src/layer0_data/track3_cctv_stream.py` | CCTV 목록 | 변경 없음 |
| `src/layer0_data/config.py` | 설정 | 녹화 시간 상수 추가 (RECORD_DURATION_SEC 등) |

## 제약

- CPU 전용 — CCTV당 1fps 분석
- 녹화는 ffmpeg -c copy (재인코딩 없음, CPU 부하 거의 없음)
- 저장: `/media/ybs/Expansion/CCTV차종분류/accident_data/stream/` (3.2TB 여유)
- ITS_API_KEY 설정 완료

## 성공 기준

1. 트리거 발화 시 즉시 녹화 시작 → mp4 생성 (Ring Buffer 없이)
2. 녹화 지속 3~5분 후 ITS 교차확인 → 사고면 보존, 아니면 삭제
3. 다중 카메라 분석 중 트리거 카메라만 녹화
4. 최종 영상이 징조 시점부터 시작 (사고 전 행동이 담김)
5. 10분+ 안정 가동 에러 없음

## 하지 말 것

- Ring Buffer / 상시 녹화 (징조 감지 전에는 녹화하지 않음)
- MLLM 연동
- DuckDB 연동
- 기존 track3_auto_collector.py / track3_cctv_stream.py 수정
