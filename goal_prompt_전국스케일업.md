# /goal 프롬프트: 전국 CCTV 363대 사고영상 수집 스케일업

아래 내용을 `/goal` 커맨드의 `$ARGUMENTS`로 사용.

---

## 목표

현재 4대 카메라 한정인 v4 사고영상 수집 시스템을 **ITS 전국 CCTV 363대 전수 감시**로 확장하는 설계 + 구현.
서버 증설 없이 현재 하드웨어(i9-285K 24코어, 122GB RAM)로 처리한다.

## 현황

### v4 아키텍처 (2026-05-22 가동 중)

```
CCTV 4대 → FrameSampler(1fps) → YOLO(21ms) → ByteTrack → TriggerDetector
    → 3중 방어(차선필터 + 임계값 + 다중합의)
    → 트리거 시 녹화(3~5분) → ITS API 사고 확인 → 보존/삭제
```

- 40분 가동: 트리거 28건, 녹화 0건 (정상 교통 오탐 제거 확인)
- YOLO 21ms/프레임 × 4대 = 84ms/초 (CPU 여유 충분)

### CPU 병목

```
YOLO 21ms/프레임 × 363대 × 1fps = 7.6초/초  ← 실시간 불가
현재 CPU 한계 = ~47대 (1000ms / 21ms)
```

363대 전부에 YOLO를 돌리는 것은 물리적으로 불가능.

### 핵심 통찰

전체 영상의 **99%+ 가 정상 교통**. 정상 프레임에 YOLO를 돌리는 것은 CPU 낭비.
경량 프리필터(~1.5ms/프레임)로 이상 후보만 걸러내면 YOLO는 소수 카메라에만 적용.

## 설계: 3-Tier 아키텍처

### Tier 1 — 전수 프리필터 (363대, CPU 1~2코어)

363대 CCTV 스트림을 1fps로 수신하고, **YOLO 없이** 이상 여부를 판별.

프리필터 방법 (OR 조합, 하나라도 해당하면 Tier 3으로):
1. **프레임 차분 (MOG2)**: 배경 대비 급격한 변화 (정지 차량 출현, 충돌 순간 변형)
2. **광학 흐름 급변**: 전체 프레임의 흐름 패턴이 갑자기 바뀜 (다수 급정거)
3. **정지 객체 감지**: 배경 모델에 없던 정지 물체 지속 출현 (사고 차량/잔해)
4. **프레임 간 히스토그램 차이**: 연기/화재 등 전체적 밝기/색상 변화

기대 성능:
- 처리 시간: ~1.5ms/프레임 (CPU)
- 363대 × 1fps × 1.5ms = **545ms/초** → 1코어로 처리 가능
- 목표: 재현율 95%+ (오경보는 허용, 못 잡는 것이 치명적)
- 이상 후보: 전체의 1~5% → 4~18대 → Tier 3 처리 가능

### Tier 2 — 핫스팟 정밀 감시 (30~50대, CPU ~10코어)

사고다발구간 상위 30~50개소 CCTV는 프리필터 없이 **직접 정밀 분석**.

- 현재 v4 파이프라인 그대로 적용 (YOLO + ByteTrack + TriggerDetector)
- 프리필터 미탐지 보완 역할 (저속 추돌, 정차 후 2차 사고 등)
- 핫스팟 선정: ITS 사고 이력 기반 상위 구간

CPU: 50대 × 21ms = 1,050ms → **10코어** (코어당 ~5대)

### Tier 3 — 정밀 분석 (CPU ~10코어)

Tier 1 이상 후보 + Tier 2 트리거 → 현 anomaly_engine 파이프라인.

- Tier 1에서 올라온 카메라에 YOLO + ByteTrack + TriggerDetector 즉시 가동
- 트리거 발화 → 녹화 시작 (v4와 동일)
- ITS API 사고 확인 → 보존/삭제

CPU: 동시 최대 ~50대 × 21ms = ~10코어

### 총 CPU 예산

```
Tier 1:  363대 프리필터     → ~1코어
Tier 2:  50대 정밀 상시     → ~10코어
Tier 3:  ~50대 정밀 온디맨드 → ~10코어
스트림 관리 + ITS 폴링      → ~1코어
────────────────────────────────────
합계:                        ~22코어 (24코어 내)
```

## 구현해야 할 것

### 1. 스트림 매니저 (`stream_manager.py`, 신규)

363대 CCTV의 HLS 스트림을 효율적으로 관리하는 계층.

역할:
- ITS CCTV API로 전국 CCTV 목록 + 스트림 URL 조회 (주기적 갱신)
- 363개 FrameSampler를 스레드풀로 관리
- 스트림 건강 상태 모니터링 (끊김 감지/재연결)
- Tier 등급(1/2/3) 관리: 프리필터 결과에 따라 동적 승격/강등

핵심 설계:
```python
class StreamManager:
    def __init__(self, max_streams=400):
        self.streams: dict[str, StreamWorker] = {}
        self.tier_assignment: dict[str, int] = {}  # cctv_id → 1/2/3
        
    def start_all(self):
        """363대 스트림 시작 (Tier 1 기본)"""
        
    def promote_to_tier3(self, cctv_id: str):
        """프리필터 이상 감지 → Tier 3 정밀 분석 전환"""
        
    def demote_to_tier1(self, cctv_id: str):
        """정밀 분석 종료 → Tier 1 복귀"""
```

제약:
- 363개 동시 HLS 연결의 네트워크 대역폭: 363 × ~500Kbps(640×480) = ~180Mbps
- 메모리: 363 × ~5MB(프레임 버퍼) = ~1.8GB
- 스트림 URL 만료 대응: 1시간마다 URL 갱신

### 2. 프리필터 엔진 (`prefilter.py`, 신규)

YOLO 없이 이상 프레임을 탐지하는 경량 엔진.

```python
class PreFilter:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.prev_frame = None
        self.prev_hist = None
        
    def check(self, frame: np.ndarray) -> tuple[bool, float]:
        """이상 여부 + 이상도 점수 반환.
        
        Returns:
            (is_anomaly, score): True면 Tier 3 승격 대상
        """
```

프리필터 신호 4종:
1. **MOG2 전경 비율**: `fg_ratio = foreground_pixels / total_pixels`
   - 정상: 5~20% (움직이는 차량)
   - 이상: < 3% (전체 정지) 또는 > 40% (급격한 변화)
2. **프레임 차분 강도**: `diff = cv2.absdiff(curr, prev)` 평균값
   - 정상: 안정적 흐름
   - 이상: 급격한 변화 (충돌/전복)
3. **정지 영역 출현**: MOG2 배경에 없는 새로운 정지 객체 지속
   - 사고 차량/잔해가 10초+ 정지
4. **히스토그램 급변**: `cv2.compareHist()` 값 급등
   - 연기/화재/조명 급변

임계값 전략:
- **낮게 시작** (오경보 허용, 재현율 99% 목표)
- OR 조합: 하나라도 이상이면 통과
- 운영 데이터로 점진적 조정

### 3. 스케일 파이프라인 (`run_nationwide.py`, 신규)

363대 통합 파이프라인. `run_realtime.py`의 MultiCCTVPipeline을 확장.

```
시작
  ↓
ITS API → CCTV 363대 목록 + 스트림 URL
  ↓
핫스팟 선정 → 상위 50대 = Tier 2, 나머지 313대 = Tier 1
  ↓
StreamManager.start_all() → 363개 스트림 시작
  ↓
메인 루프:
  [Tier 1] 313대 → PreFilter.check() → 이상 시 promote_to_tier3()
  [Tier 2] 50대 → VisionPipeline 상시 분석
  [Tier 3] 승격 카메라 → VisionPipeline 분석 → 트리거 → 녹화
  [ITS] 60초 폴링 → 사고 발생 시 해당 CCTV 즉시 Tier 3 편입
  ↓
녹화 종료 → ITS 확인 → 보존/삭제
```

### 4. ITS 사고 연동 강화 (`incident_reactor.py`, 신규)

ITS API에서 사고 감지 시 **즉시 해당 CCTV 편입**:

```python
class IncidentReactor:
    def on_new_incident(self, incident: IncidentEvent):
        """ITS 사고 발생 → 가장 가까운 CCTV를 Tier 3으로 즉시 승격 + 녹화 시작"""
        nearest = self.find_nearest_cctvs(incident.lat, incident.lon, n=3)
        for cctv in nearest:
            self.stream_manager.promote_to_tier3(cctv.cctv_id)
            self.start_recording(cctv)  # 사고 처리 영상이라도 확보
```

이렇게 하면:
- AI가 먼저 감지 → 사고 순간 영상 (Tier 1/2 → Tier 3)
- ITS가 늦게 알림 → 사고 처리 영상이라도 확보 (보완)

## 구현 로드맵

| Phase | 기간 | 내용 | 병렬 |
|:---:|:---:|------|:---:|
| P1 | 2주 | PreFilter 프로토타입 (기존 4대로 검증) | ← |
| P2 | 2주 | StreamManager (363대 스트림 관리) | ← P1과 병렬 |
| P3 | 2주 | 통합 (run_nationwide.py + Tier 2 핫스팟) | |
| P4 | 2주 | 부하 테스트 + 임계값 튜닝 + 안정화 | |

**실질 6주** (P1~P2 병렬)

## 품질 게이트

| 게이트 | 기준 | 시점 |
|--------|------|:---:|
| G1 프리필터 재현율 | >= 95% (사고 놓치지 않기) | P1 |
| G2 스트림 안정성 | 363대 24h 무중단 | P2 |
| G3 CPU 사용률 | 정상 <= 60%, 피크 <= 85% | P4 |
| G4 오경보율 | Tier 3 승격 중 실제 이상 >= 10% | P4 |
| G5 E2E 포착률 | >= 90% (ITS 사고 사후 대조) | P4 |

## 폴백

G1/G5 미달 시 → 서버 1대 추가 → Tier 2를 100대로 확대 (방안 C 최소 적용)

## 수정/생성 대상 파일

| 파일 | 유형 | 역할 |
|------|:---:|------|
| `src/layer0_data/stream_manager.py` | 신규 | 363대 스트림 수명주기 관리 |
| `src/layer1_vision/prefilter.py` | 신규 | 경량 이상 판별 (MOG2 + 차분 + 히스토그램) |
| `src/layer0_data/run_nationwide.py` | 신규 | 3-Tier 통합 파이프라인 |
| `src/layer0_data/incident_reactor.py` | 신규 | ITS 사고 → 즉시 CCTV 편입 |
| `src/config_new.py` | 수정 | Tier/프리필터/스트림 설정 추가 |
| `src/layer0_data/run_realtime.py` | 참조 | 기존 v4 로직 재사용 (수정 최소) |

## 제약

- CPU 전용 (GPU 없음) — 프리필터는 반드시 CPU 경량
- 서버 1대 — 24코어 / 122GB RAM / NVMe 1.8TB + HDD 7.3TB + 외장 15TB
- ITS CCTV 스트림 URL은 일정 시간 후 만료 → 주기적 갱신 필요
- HLS 연결 363개의 네트워크 안정성 미검증 (P2에서 확인)
- 사고 영상 학습 데이터 없음 (이 시스템으로 축적하는 것이 목적)

## 기대 효과

| 지표 | 현재 (v4) | 스케일업 후 |
|------|:---:|:---:|
| 감시 카메라 | 4대 | **363대** |
| 감시 공백 | 359대 | **0대** |
| 사고 포착 확률 | ~1% (4/363) | **90%+** |
| 추가 비용 | - | **0원** |
| 사고 영상 축적 | 0건/일 | 예상 2~5건/일 |

## 하지 말 것

- GPU 구매/클라우드 사용
- YOLO 모델 교체/경량화 (기존 YOLO11n 유지)
- MLLM 연동
- DuckDB 연동
- 실시간 대시보드/웹 UI
- track3_api_incident.py / track3_cctv_stream.py 직접 수정 (래핑만)
