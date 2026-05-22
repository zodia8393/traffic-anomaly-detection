# /goal 프롬프트: 사고영상 수집 시스템 개선 — 오탐 감소 + 보존 정책

아래 내용을 `/goal` 커맨드의 `$ARGUMENTS`로 사용.

---

## 목표

현재 가동 중인 징조 기반 온디맨드 녹화 시스템의 **실전 운영 데이터**에서 드러난 문제를 개선한다.

핵심 문제: 55분 가동 → 트리거 167건, 녹화 4건, **보존 0건** (전부 삭제)

## 현황 진단 (2026-05-22 실 가동 데이터)

### 문제 1: T1 TTC 오탐 과다

```
RT_T1_121538_0 | sev=0.06 | TTC=2.8s | deleted
RT_T1_123108_0 | sev=0.97 | TTC=0.1s | deleted
RT_T1_124616_0 | sev=0.99 | TTC=0.0s | deleted
RT_T1_130143_0 | sev=0.99 | TTC=0.0s | deleted
```

- TTC=0.0s (severity 0.99)가 반복 발생 → **167건 트리거 중 대부분이 T1**
- 원인: CCTV 원근 투영으로 다른 차선 차량의 바운딩 박스가 2D 좌표상 겹침 → TTC 0 계산
- 현재 임계값: `TRIGGER_TTC_THRESHOLD = 3.0s`, 쿨다운 30초
- 단발성 프레임 겹침 1건으로도 즉시 녹화 시작 → 불필요한 녹화 + 쿨다운 소진

### 문제 2: 이진 보존 정책 (confirmed/deleted만)

```python
# 현재 로직 (run_realtime.py:665-707)
if confirmed and incident:
    # 보존
else:
    video_path.unlink()  # 즉시 삭제
```

- ITS API가 사고를 반영하기까지 시간차 존재 (수분~수십분)
- API 장애 시에도 무조건 삭제 → 실제 사고 영상 유실 위험
- severity 0.99 영상도 ITS 미확인이면 삭제

### 문제 3: cam0만 Vision Pipeline

- 4대 중 cam0만 YOLO 분석, cam1~3은 프레임 샘플링만
- cam1~3에서 사고가 발생해도 감지 불가
- CPU 여유 있음 (i9-285K, YOLO 21ms/frame × 4대 = 84ms < 1000ms)

### 문제 4: T3 정지차량 임계값 과민

- `TRIGGER_STOP_DURATION = 3.0s` → 고속도로 정체 시 정상 정차도 트리거
- 고속도로 사고 정지는 보통 10초 이상 지속

## 해야 할 것

### 1. T1 TTC 오탐 필터링 (trigger_detector.py)

**연속성 요구**: 단발성 프레임 겹침 제거
- TTC 임계 이하가 **연속 N프레임(≥2) 이상** 지속될 때만 트리거 발화
- 1fps 분석이므로 연속 2프레임 = 실제 2초간 위험 지속 의미
- 구현: `_ttc_streak: dict[tuple[int,int], int]` — 트랙 쌍별 연속 카운트

**TTC 최소값 필터**: TTC=0.0s는 박스 겹침(artifact)
- `TTC < 0.3s`이면서 두 트랙의 IoU > 0.5면 무시 (실제 충돌이면 IoU 높을 수 없음)
- 또는 TTC < 0.3s 자체를 무시 (실제 TTC 0초 = 이미 충돌 = 별도 감지 필요)

**녹화 시작 severity 게이트**:
- severity < 0.3인 트리거는 로그만 남기고 녹화 시작하지 않음
- 현재: severity 무관하게 모든 트리거가 녹화 시작

### 2. 3단계 보존 정책 (run_realtime.py)

기존 이진(confirmed/deleted) → 3단계(confirmed/pending/deleted):

```
트리거 발화 → 녹화 → 녹화 종료
  ├─ ITS 확인 성공 → confirmed (영상 보존, 메타 기록)
  ├─ ITS 미확인 + severity ≥ 0.7 → pending (영상 보존, 재확인 예약)
  ├─ ITS 미확인 + severity < 0.7 → deleted (영상 삭제)
  └─ ITS API 장애 → pending (영상 보존, 재확인 예약)
```

pending 처리:
- `pending/` 서브디렉토리에 영상 + 메타 저장
- 다음 ITS 폴링 주기에 재확인 (최대 3회, 각 10분 간격)
- 3회 재확인 후에도 미확인 → severity ≥ 0.8이면 보존, 아니면 삭제
- `action` 필드: "confirmed" / "pending" / "deleted" / "pending_expired"

### 3. 다중 카메라 Vision 확장 (run_realtime.py)

현재 cam0만 → **전체 카메라 Vision Pipeline 적용**:
- CPU 여유 확인됨: 4대 × 21ms = 84ms/초 (< 1000ms)
- 각 CameraWorker에 독립 VisionPipeline 인스턴스 할당
- YOLO 모델은 공유 (메모리 절약), ByteTrack+TriggerDetector는 카메라별 독립
- `--max-cameras` 기본값 유지하되, vision 적용 카메라 수 별도 옵션 추가하지 않음 (전체 적용)

### 4. T3 정지차량 임계값 조정 (config_new.py)

- `TRIGGER_STOP_DURATION`: 3초 → **10초** (고속도로 정체와 구분)
- `TRIGGER_STOP_SPEED`: 5 km/h → 유지 (적절)

## 수정 대상 파일

| 파일 | 수정 내용 |
|------|----------|
| `src/layer1_vision/trigger_detector.py` | T1 연속성 필터, TTC 최소값 필터 |
| `src/layer0_data/run_realtime.py` | 3단계 보존 정책, 다중 카메라 Vision 확장, severity 게이트 |
| `src/config_new.py` | T3 STOP_DURATION 3→10초 |
| `src/layer0_data/config.py` | pending 디렉토리 경로 추가 |

## 제약

- CPU 전용 — 카메라 추가 시 21ms × N < 500ms 유지 (50% 마진)
- 기존 트리거 로직(T1~T7)의 구조는 유지, 임계값/필터만 수정
- track3_api_incident.py / track3_cctv_stream.py 수정 금지
- 현재 가동 중인 프로세스(PID 1898419) 종료 후 재시작 필요

## 검증

### A. T1 오탐 감소 확인
- [ ] 수정 전: 55분간 167건 트리거 → 수정 후: 동일 시간 대비 80% 이상 감소
- [ ] TTC=0.0s 트리거가 녹화를 시작하지 않는 것 확인
- [ ] 연속 2프레임 미만 TTC 이벤트가 필터링되는 것 확인

### B. 보존 정책 동작 확인
- [ ] severity ≥ 0.7 + ITS 미확인 → pending/ 디렉토리에 영상 보존 확인
- [ ] pending 재확인 로직 (최대 3회) 동작 확인
- [ ] ITS 확인 → confirmed 정상 동작 확인
- [ ] severity < 0.7 + ITS 미확인 → 삭제 확인

### C. 다중 카메라 Vision 확인
- [ ] cam0~cam3 모두 트리거 발화하는 것 확인
- [ ] CPU 사용률 50% 미만 유지 확인 (4대 기준)

### D. T3 임계값 확인
- [ ] 10초 미만 정지 → 트리거 미발화 확인
- [ ] 10초 이상 정지 → 트리거 발화 확인

### E. 안정성
- [ ] 30분+ 연속 가동 에러 없음
- [ ] 메모리 사용량 안정 (증가 추세 없음)

## 하지 말 것

- MLLM(Qwen2.5-VL) 연동
- DuckDB 연동
- 새로운 트리거 유형 추가
- YOLO 모델 교체/재학습
- track3_auto_collector.py / track3_cctv_stream.py 수정
- HotspotSelector 로직 변경
