# /goal 프롬프트: 전국 스케일업 3-Tier 파이프라인 디버깅 + 안정화

아래 내용을 `/goal` 커맨드의 `$ARGUMENTS`로 사용.

---

## 목표

P4 통합 테스트(4대, 90초)에서 발견된 **5개 이슈**를 해결하고, 363대 장시간 가동 전 시스템을 안정화한다.

## P4 테스트 결과 요약 (2026-05-22)

```
python run_nationwide.py start --max-cameras 4
```

### 정상 동작 확인
- 4대 HLS 스트림 연결 (수도권제1순환선: 판교분기점, 성남, 성남요금소, 송파)
- VisionPipeline 4개 생성 (VehicleDetector YOLO 싱글턴 공유)
- ITS IncidentReactor 가동: 6건 실시간 사고 감지 → 18대 CCTV 승격 시도
- 트리거 정상 발화: T1(TTC 0.7s), T2(급감속 -3.3m/s²), T3(정지 23s/61s), T6(속도분산)
- 합의 필터 동작: 녹화 0건 (단독 트리거 차단)
- 60초 주기 상태 로깅 정상

### 발견 이슈 5건

## 이슈 1: Tier 1 미테스트 — 핫스팟 로직 버그

**증상**: `max_cameras=4`인데 4대 전부 Tier 2로 배정, Tier 1 프리필터 경로 미검증.

```
스트림 4대 시작 완료 (T1=0, T2=4)
```

**원인**: `_select_hotspots()`가 CCTV 목록 첫 50대를 핫스팟으로 지정. `max_cameras=4`면 로드된 4대가 모두 핫스팟 범위에 포함.

**수정 방향**: `max_cameras`가 핫스팟 수보다 적을 때, 비율을 유지하거나 최소 1대는 Tier 1로 배정.

```python
# 예시: max_cameras=4일 때 T2=3, T1=1 (비율 유지)
# 또는: max_cameras <= hotspot_count 일 때 T2 = max_cameras * (hotspot_count / total_cctvs)
```

**검증**: `max_cameras=4`로 실행 시 T1 >= 1, T2 >= 1 확인. 프리필터 이상 감지 → Tier 3 승격 → VisionPipeline 생성 흐름 검증.

## 이슈 2: HLS 스트림 불안정 — 0프레임 반복 끊김

**증상**: 성남, 송파 2대가 연결 후 0프레임 → 끊김 → 재연결 반복.

```
16:16:48 stream.수도권제1순환선__성남: 스트림 연결 [T2]
16:16:48 FrameSampler 종료 (총 0프레임)
16:16:58 stream.수도권제1순환선__성남: 스트림 연결 [T2]   ← 10초 후 재시도
16:16:58 FrameSampler 종료 (총 0프레임)                   ← 또 0프레임
```

판교분기점/성남요금소는 안정적 (프레임 지속 수신).

**가능한 원인**:
1. ITS HLS 서버 로드밸런싱 — 특정 CCTV URL이 일시적 불안정
2. FrameSampler ffmpeg 옵션 — HLS 전용 옵션 부재
3. 동시 4개 ffmpeg 프로세스의 리소스 경합

**수정 방향**:
```python
# FrameSampler.start()에 HLS 안정성 옵션 추가
cmd = [
    "ffmpeg",
    "-loglevel", "error",
    "-reconnect", "1",
    "-reconnect_streamed", "1",
    "-reconnect_at_eof", "1",           # 추가
    "-reconnect_on_network_error", "1", # 추가
    "-reconnect_delay_max", "10",       # 5 → 10
    "-rw_timeout", "15000000",          # 10s → 15s
    "-i", stream_url,
    ...
]
```

**검증**: 동일 4대 CCTV로 5분 가동 후, 스트림별 `worker.stats.frames` > 0 확인. `consecutive_failures < STREAM_MAX_FAILURES` 유지.

## 이슈 3: VisionPipeline CPU 경합 — 콜백 스레드 모델

**증상**: 직접 에러는 없으나, 구조적 리스크 존재.

**문제**: StreamWorker의 `on_frame` 콜백이 **각 워커 스레드**에서 실행됨. Tier 2/3 카메라의 VisionPipeline.process_frame()이 YOLO(21ms)를 호출하므로, 동시 50대 Tier 2 + 50대 Tier 3 = 100개 스레드가 동시에 YOLO.detect()를 호출하면 GIL 경합 + CPU 과포화.

```python
# stream_manager.py StreamWorker._run()
self._on_frame(frame, self.cctv.cctv_id, self.tier)  # 워커 스레드에서 실행
```

**현재 YOLO 처리량**: 21ms/프레임 × 1코어 = ~47FPS.
100대 × 1fps = 100FPS 필요 → 최소 3코어 (GIL 고려 시 더 필요).

**수정 방향**:
1. **YOLO 호출 직렬화**: 프로듀서-컨슈머 패턴. 워커 스레드는 프레임을 큐에 넣고, YOLO 처리 스레드 N개가 순서대로 처리.
2. **또는 OMP_NUM_THREADS 조정**: YOLO가 내부적으로 멀티스레드 → 동시 호출 수 제한.
3. **또는 ThreadPoolExecutor(max_workers=N)**: 콜백 실행을 풀로 제한.

```python
# 방안 1: 직렬화 큐
class NationwidePipeline:
    def __init__(self):
        self._frame_queue = queue.Queue(maxsize=200)
        self._workers = [Thread(target=self._process_loop) for _ in range(4)]
    
    def _on_frame(self, frame, cctv_id, tier):
        self._frame_queue.put((frame, cctv_id, tier), block=False)
    
    def _process_loop(self):
        while not self._stop_event.is_set():
            frame, cctv_id, tier = self._frame_queue.get()
            # tier 분기 처리
```

**검증**: `--max-cameras 10`으로 실행. CPU 사용률 모니터링 (`top -p PID`). 10대 동시 처리 시 CPU < 50%.

## 이슈 4: IncidentReactor 범위 불일치

**증상**: find_nearest가 반환한 CCTV가 StreamManager.streams에 없어 승격 실패.

```
16:16:18 incident_reactor:   승격 T3: [청주영덕선] 회인 (0.3km)
16:16:18 stream_manager: 승격 실패: 청주영덕선__회인 없음
```

**원인**: `max_cameras=4`로 수도권 4대만 로드. ITS 사고는 청주/광주/당진 등 전국에서 발생. find_nearest는 전체 4,768대에서 검색하므로 로드되지 않은 CCTV 반환.

**전체 가동 시**: 363대 로드 → 대부분 해결. 단, 363대에 포함되지 않은 CCTV가 가장 가까울 수 있음.

**수정 방향**:
```python
# IncidentReactor._react()에서 StreamManager에 로드된 CCTV만 대상으로 검색
# 또는 find_nearest 결과를 streams.keys()로 필터링
nearest = self._cctv_client.find_nearest(lat, lon, top_k=INCIDENT_REACTOR_CCTVS * 3)
nearest = [(c, d) for c, d in nearest if c.cctv_id in self._stream_manager.streams]
nearest = nearest[:INCIDENT_REACTOR_CCTVS]
```

**검증**: `max_cameras=4`로 실행, 사고 반응 시 "승격 실패" 대신 로드된 CCTV 중 가장 가까운 것을 승격. 또는 해당 사고와 로드된 CCTV 거리가 너무 멀면 skip 로깅.

## 이슈 5: 메모리 프로파일링 미수행

**증상**: 363대 장시간 가동 시 메모리 증가 패턴 미확인.

**우려 사항**:
- PreFilter 363개: MOG2 배경 모델(640×480) × 363 = 추정 ~500MB
- VisionPipeline 최대 100개: ByteTrack 상태 + trigger history = 추정 ~200MB
- FrameSampler 프레임 버퍼: 640×480×3 × 363 = ~320MB
- 합계: ~1GB+ (122GB RAM 대비 여유)

**수정 방향**: 실제 메모리 측정.
```python
import psutil, os
process = psutil.Process(os.getpid())
mem_mb = process.memory_info().rss / 1e6
logger.info("메모리: %.0f MB (prefilters=%d, pipelines=%d)",
            mem_mb, len(self._prefilters), len(self._vision_pipelines))
```

**검증**: `--max-cameras 50` 으로 실행. 30분 후 메모리 < 10GB. PreFilter/VisionPipeline 생성/삭제 시 메모리 변화 확인.

## 수정 우선순위

| # | 이슈 | 심각도 | 난이도 | 우선순위 |
|:---:|------|:---:|:---:|:---:|
| 1 | Tier 1 미테스트 (핫스팟 비율) | 높음 | 낮음 | **P0** |
| 4 | IncidentReactor 범위 불일치 | 중간 | 낮음 | **P0** |
| 2 | HLS 스트림 불안정 | 중간 | 낮음 | **P1** |
| 3 | VisionPipeline CPU 경합 | 높음 | 높음 | **P1** |
| 5 | 메모리 프로파일링 | 낮음 | 낮음 | **P2** |

## 수정/검증 대상 파일

| 파일 | 이슈 | 수정 내용 |
|------|:---:|------|
| `src/layer0_data/run_nationwide.py` | 1,3,5 | 핫스팟 비율 수정, 콜백 큐 도입, 메모리 로깅 |
| `src/layer0_data/incident_reactor.py` | 4 | 로드된 CCTV 필터링 |
| `src/layer0_data/run_realtime.py` | 2 | FrameSampler HLS 옵션 강화 |
| `src/layer0_data/stream_manager.py` | — | (변경 없음, 참조) |

## 검증 시나리오

### 시나리오 A: Tier 1→3 승격 흐름 (이슈 1)
```
max_cameras=6 → T1 >= 2, T2 >= 2
→ Tier 1 카메라에서 프리필터 이상 감지
→ Tier 3 승격 → VisionPipeline lazy 생성
→ 정밀 분석 → 트리거 → (합의) → 녹화 or 강등
```

### 시나리오 B: 스트림 안정성 (이슈 2)
```
max_cameras=10, 5분 가동
→ 전체 스트림 frames > 0
→ disabled 스트림 0대
→ reconnect 횟수 < 스트림 수 × 2
```

### 시나리오 C: CPU 부하 (이슈 3)
```
max_cameras=20, 3분 가동
→ CPU 사용률 < 60% (정상)
→ 프레임 처리 지연 없음 (큐 오버플로 0건)
```

### 시나리오 D: IncidentReactor (이슈 4)
```
max_cameras=10
→ ITS 사고 발생 시 로드된 CCTV 중 가장 가까운 것 승격
→ "승격 실패" 로그 0건 (로드 범위 밖 사고는 skip 로그)
```

### 시나리오 E: 메모리 (이슈 5)
```
max_cameras=50, 10분 가동
→ RSS < 10GB
→ Tier 3→1 강등 시 메모리 감소 확인
```

## 품질 게이트

| 게이트 | 기준 | 시점 |
|--------|------|:---:|
| G1 | Tier 1→3 승격 흐름 E2E 동작 | 이슈 1 수정 후 |
| G2 | 10대 5분 가동, disabled 0 | 이슈 2 수정 후 |
| G3 | 20대 3분 가동, CPU < 60% | 이슈 3 수정 후 |
| G4 | IncidentReactor "승격 실패" 0건 | 이슈 4 수정 후 |
| G5 | 50대 10분 가동, RSS < 10GB | 이슈 5 수정 후 |

## 하지 말 것

- track3_api_incident.py / track3_cctv_stream.py 직접 수정
- YOLO 모델 교체/경량화
- DuckDB/MLLM 연동
- 웹 UI/대시보드
- 363대 전체 가동 (이 goal은 디버깅 전용, 전체 가동은 G1~G5 통과 후)
- run_realtime.py의 기존 v4 로직 변경 (FrameSampler 옵션만 수정)
