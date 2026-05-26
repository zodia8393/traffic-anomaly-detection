# /goal 프롬프트: 돌발정보 CCTV 캡처 → 사고분석 파이프라인 통합

아래 내용을 `/goal` 커맨드의 `$ARGUMENTS`로 사용.

---

## 목표

ITS 돌발정보 페이지 기반 CCTV 프레임 캡처(`OutbreakCCTVCapture`)를 전국 사고분석 파이프라인(`NationwidePipeline`)에 통합하여, 사고/고장 인근 CCTV 영상을 **자동으로 분석·트리거·녹화**하는 E2E 파이프라인을 완성한다.

## 배경: 왜 이 통합이 필요한가

### 현재 상황

```
StreamManager (ffmpeg)                OutbreakCCTVCapture (Playwright)
─────────────────────                ──────────────────────────────────
ITS API → cctvurl → ffmpeg           ITS 돌발정보 페이지 UI 클릭
        ↓                                     ↓
    403 Forbidden (3중 보호)          hls.js 자동 재생 → canvas 캡처
    → 0프레임                         → 10/10 성공 (720×480)
```

- **StreamManager**: ITS HLS 프록시 3중 보호(도메인 화이트리스트 + IP 속도 제한 + wmsAuth)로 **모든 카메라 0프레임**
- **OutbreakCCTVCapture**: 돌발정보 페이지 UI 경로로 프록시 우회, **사고/고장 인근 CCTV 100% 캡처 성공**
- 현재 OutbreakCCTVCapture는 **스냅샷만** 캡처 (CCTV당 1장) — VisionPipeline(ByteTrack) 연동 불가

### 해결해야 할 핵심 문제

1. **스냅샷 → 연속 프레임**: ByteTrack/TriggerDetector는 연속 프레임이 필요. CCTV당 1장 스냅샷으로는 추적·트리거 불가.
2. **파이프라인 연동**: 캡처된 프레임을 NationwidePipeline의 Vision 큐에 주입하는 경로 구축.
3. **병행 운영**: Outbreak 캡처와 기존 StreamManager(향후 복구 시)가 공존할 수 있는 구조.

## 설계: 체류형 캡처 + 파이프라인 콜백

### 핵심 아이디어: "순찰 체류" 모드

기존 OutbreakCCTVCapture의 "1장 스냅샷" 방식을 **"체류 캡처"로 확장**:

```
사고 클릭 → 인근 CCTV 클릭 → 영상 재생
                                  ↓
                        30초간 1fps 연속 캡처 (체류)
                        ├─ 매 프레임 → VisionPipeline
                        ├─ 트리거 발화 → 체류 연장 (60초)
                        └─ 체류 종료 → 다음 CCTV로 이동
```

### 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│  NationwidePipeline (기존)                                         │
│                                                                     │
│  StreamManager ──→ _on_frame(frame, cctv_id, tier)                 │
│  (ffmpeg, 현재 0프레임)      ↓                                      │
│                     ┌──────────────────────┐                        │
│                     │  Vision 큐 워커       │                        │
│  OutbreakFeeder ──→ │  YOLO → ByteTrack    │──→ 트리거 → 녹화      │
│  (신규, 브라우저)    │  → TriggerDetector   │                        │
│                     └──────────────────────┘                        │
│                                                                     │
│  OutbreakFeeder = OutbreakCCTVCapture + 체류 캡처 + 프레임 주입     │
└─────────────────────────────────────────────────────────────────────┘
```

### 데이터 흐름 상세

```
OutbreakCCTVCapture                   NationwidePipeline
───────────────────                   ──────────────────
1. /map/outbreak 스캔
2. 사고/고장 필터
3. 인근 CCTV 클릭
4. 영상 재생 확인
5. 체류 루프 시작:                    
   ├─ canvas 캡처 (1fps)  ─────→     _on_frame(frame, outbreak_cctv_id, tier=3)
   │                                        ↓
   │                                  Vision 큐 → YOLO → ByteTrack → 트리거
   │                                        ↓
   ├─ 트리거 콜백 수신     ←─────     trigger_callback(trigger)
   │  → 체류 연장 (60초)
   │  → 사고 확인 메타 기록
   │
   └─ 체류 시간 종료
6. 다음 CCTV로 이동
7. 사이클 종료 → poll_interval 대기
```

## 해야 할 것

### 1. OutbreakCCTVCapture 확장: 체류 캡처 모드

`browser_stream.py`에 체류(dwelling) 캡처 기능 추가:

```python
class OutbreakCCTVCapture:
    def capture_dwelling(
        self,
        cctv: dict,
        incident: OutbreakIncident,
        dwell_sec: float = 30.0,       # 기본 체류 30초
        fps: float = 1.0,              # 캡처 fps
        on_frame: Callable = None,      # 프레임 콜백 (파이프라인 주입)
        extend_event: Event = None,     # 체류 연장 신호
    ) -> list[CapturedFrame]:
        """CCTV 클릭 후 dwell_sec 동안 연속 캡처, 매 프레임마다 on_frame 호출."""
```

- CCTV 클릭 → 영상 재생 확인 → dwell_sec 동안 1fps canvas 캡처
- 매 프레임마다 `on_frame(frame_bgr, cctv_id, timestamp)` 콜백 호출
- `extend_event.set()` 수신 시 체류 시간 60초 추가 (트리거 발화 시)
- 체류 중에도 `_stop_event` 확인하여 graceful shutdown

### 2. OutbreakFeeder: 파이프라인 연결 어댑터

`run_nationwide.py`에 추가할 브릿지 클래스:

```python
class OutbreakFeeder:
    """OutbreakCCTVCapture → NationwidePipeline 프레임 브릿지."""

    def __init__(self, pipeline: NationwidePipeline, stop_event: Event):
        self._pipeline = pipeline
        self._capture = OutbreakCCTVCapture(
            target_types=("사고", "고장"),
            cctv_delay=5.0,
            incident_delay=3.0,
            max_cctvs_per_incident=2,
        )
        self._stop_event = stop_event
        self._extend_events: dict[str, Event] = {}  # cctv_id → extend signal

    def start(self):
        """브라우저 시작 + 백그라운드 순찰 루프."""

    def _patrol_loop(self):
        """사고 스캔 → 인근 CCTV 체류 캡처 → 프레임 주입."""

    def _on_dwelling_frame(self, frame_bgr, cctv_id, timestamp):
        """프레임을 NationwidePipeline Vision 큐에 주입."""
        self._pipeline._on_frame(frame_bgr, cctv_id, tier=3)

    def _on_trigger(self, trigger, cctv_id):
        """트리거 발화 시 해당 CCTV 체류 연장."""
        ev = self._extend_events.get(cctv_id)
        if ev:
            ev.set()
```

### 3. NationwidePipeline 수정

```python
class NationwidePipeline:
    def __init__(self, ...):
        ...
        self._outbreak_feeder: OutbreakFeeder | None = None

    def start(self):
        ...
        # 기존 StreamManager 시작 (향후 복구 시 작동)
        self._stream_manager.start(...)

        # OutbreakFeeder 병행 시작
        self._outbreak_feeder = OutbreakFeeder(
            pipeline=self,
            stop_event=self._stop_event,
        )
        self._outbreak_feeder.start()
```

핵심 수정 포인트:
- `_on_frame()`: outbreak feeder에서 들어온 프레임도 기존 Vision 큐로 처리
- CCTV ID 규칙: outbreak 경로는 `OB_{도로명}_{위치}` 형식으로 구분
- `_handle_recording_end()`: outbreak 경로는 ITS API 교차확인 생략 (이미 사고/고장 확인됨)
- 트리거 콜백: VisionPipeline 트리거 발화 시 OutbreakFeeder에 체류 연장 신호

### 4. 녹화 방식

Outbreak 경로에서 트리거 발화 시:
- ffmpeg 녹화 대신 **프레임 시퀀스 저장** (JPEG) — 브라우저 캡처 경유이므로 HLS URL 직접 접근 불가
- 프레임 시퀀스 → 트리거 전후 키프레임 선정 → 메타데이터 기록
- 또는: 트리거 시점의 영상 URL을 추출하여 ffmpeg 시도 (실패 시 프레임 시퀀스 fallback)

```
output/outbreak_recordings/
├── OB_경부선_오산_20260526_143000/
│   ├── frames/           # 체류 중 캡처된 전체 프레임
│   │   ├── 000.jpg
│   │   ├── 001.jpg
│   │   └── ...
│   ├── keyframes/        # 트리거 전후 키프레임
│   ├── metadata.json     # 사고정보 + 트리거 + CCTV 메타
│   └── trigger_log.jsonl # 트리거 이력
```

### 5. 단독 실행 모드

테스트 및 독립 운용을 위한 단독 모드:

```bash
# 테스트: 3건 사고 × 체류 30초
python browser_stream.py --dwell 30 --max-incidents 3

# 독립 수집: 연속 루프
python browser_stream.py --loop --dwell 30 --output /path/to/output
```

## 기존 코드 (수정/참조)

| 파일 | 역할 | 수정 범위 |
|------|------|----------|
| `src/layer0_data/browser_stream.py` | OutbreakCCTVCapture | **확장**: 체류 캡처 모드 추가 |
| `src/layer0_data/run_nationwide.py` | NationwidePipeline | **수정**: OutbreakFeeder 통합 |
| `src/layer0_data/stream_manager.py` | StreamManager | 변경 없음 (병행 유지) |
| `src/layer0_data/incident_reactor.py` | IncidentReactor | 변경 없음 |
| `src/layer1_vision/vision_pipeline.py` | VisionPipeline | 변경 없음 |
| `src/layer1_vision/trigger_detector.py` | TriggerDetector | 변경 없음 |
| `src/config_new.py` | 설정 | **추가**: Outbreak 관련 설정 상수 |

## 설정 상수 (config_new.py 추가)

```python
# ── 돌발정보 CCTV 캡처 ──────────────────────────────────────────────
OUTBREAK_DWELL_SEC = 30              # CCTV 체류 시간 (초)
OUTBREAK_DWELL_EXTEND_SEC = 60       # 트리거 발화 시 연장 (초)
OUTBREAK_CAPTURE_FPS = 1.0           # 체류 중 캡처 fps
OUTBREAK_CCTV_DELAY = 5.0            # CCTV 전환 간 대기 (초)
OUTBREAK_INCIDENT_DELAY = 3.0        # 사고 전환 간 대기 (초)
OUTBREAK_POLL_INTERVAL = 300         # 전체 스캔 주기 (초)
OUTBREAK_MAX_PER_CYCLE = 20          # 사이클당 최대 체류 CCTV 수
OUTBREAK_MAX_PER_INCIDENT = 2        # 사고당 최대 CCTV 수
OUTBREAK_ENABLED = True              # 활성화 플래그
```

## 제약

- CPU 전용 (GPU 없음) — YOLO 21ms/f, 체류 캡처 1fps 충분
- Playwright headless Chrome 필요 (`pip install playwright && playwright install chromium`)
- 브라우저 인스턴스 1개 → CCTV 순차 체류 (동시 재생 불가)
- ITS 돌발정보 페이지가 접속 불가하면 전체 Outbreak 경로 비활성화 (StreamManager fallback)
- 사고/고장이 없는 시간대에는 Outbreak 프레임 0건 (정상 — 사고 없으면 수집할 것도 없음)
- 체류 30초 × CCTV당 1fps = 30프레임 — ByteTrack 추적 가능 최소 단위

## 성공 기준

1. **체류 캡처 작동**: CCTV 클릭 후 30초간 1fps 연속 캡처 → 30프레임 numpy 배열 반환
2. **파이프라인 주입**: 캡처 프레임이 NationwidePipeline Vision 큐에 도착 → YOLO 처리 확인
3. **트리거 발화**: 체류 중 ByteTrack 추적 → 트리거 1종 이상 발화 (T1/T2/T5/T6 중)
4. **체류 연장**: 트리거 발화 시 체류 시간 60초 자동 연장 확인
5. **E2E**: `python run_nationwide.py start --outbreak-only` — 5분 가동 에러 없음, 프레임 수집 + 트리거 로그 확인
6. **독립 실행**: `python browser_stream.py --dwell 30 --max-incidents 2` — 단독 테스트 통과

## 하지 말 것

- StreamManager(ffmpeg) 제거/대체 (향후 ITS 화이트리스트 확보 시 복원 필요)
- MLLM 연동 (이 단계는 Vision Pipeline 트리거만)
- DuckDB 연동 (메타데이터는 JSON/JSONL)
- 기존 track3_auto_collector.py 수정
- browser_stream.py의 기존 run_cycle/run_loop API 제거 (하위 호환 유지)
- Playwright를 async로 전환 (sync API 유지 — NationwidePipeline과 threading 호환)

## 검증 순서

```
1단계: browser_stream.py 체류 캡처 단위 테스트
       → 30프레임 연속 캡처 + extend 동작 확인

2단계: OutbreakFeeder 단독 테스트
       → 프레임 → Vision 큐 주입 → YOLO 처리 확인

3단계: NationwidePipeline 통합 테스트 (--outbreak-only)
       → E2E: 스캔→체류→분석→트리거→로그

4단계: 병행 운영 테스트
       → StreamManager + OutbreakFeeder 동시 가동 (충돌 없음 확인)
```
