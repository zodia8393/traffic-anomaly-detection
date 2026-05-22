# /goal 프롬프트: 사고영상 수집 시스템 통합 파이프라인

아래 내용을 `/goal` 커맨드의 `$ARGUMENTS`로 사용.

---

## 목표

CCTV 실시간 스트림에서 비정상 주행을 AI로 먼저 감지하고, ITS 돌발상황 API로 실제 사고 여부를 교차 확인하여, 사고가 확인된 영상만 자동 저장하는 통합 파이프라인 구축.

## 시스템 흐름

```
CCTV HLS 스트림
  ├─ [FrameSampler] 1fps 프레임 추출 → Vision Pipeline (YOLO+ByteTrack+트리거)
  └─ [RingBuffer] 원본 스트림 세그먼트 녹화 (항상 최근 60+120초 보유)
        ↓
트리거 발화 (T1 급정거, T3 장기정지, T4 TTC 등)
        ↓
ITS 돌발상황 API 교차확인 (CCTV 좌표 반경 10km 내 교통사고 매칭)
        ↓
  사고 확인 → Ring Buffer flush → mp4 저장 + 메타데이터 기록
  미확인   → 폐기 (링버퍼만 계속 유지)
```

## 기존 코드 (전부 구현 완료, 통합만 필요)

| 모듈 | 위치 | 역할 |
|------|------|------|
| ITS 돌발상황 API | `src/layer0_data/track3_api_incident.py` | 교통사고 실시간 조회 (테스트 완료, 현재 4건) |
| CCTV 스트림+Ring Buffer | `src/layer0_data/track3_cctv_stream.py` | CCTV 목록(363대), 인근 검색, ffmpeg 세그먼트 녹화 |
| 자동수집기 (기존) | `src/layer0_data/track3_auto_collector.py` | API→CCTV 방향 (반대 방향이라 참고만) |
| 품질 게이트 | `src/layer0_data/quality_gate.py` | 해상도/FPS/길이 검증 |
| 설정 | `src/layer0_data/config.py` | RING_BUFFER 60/120초, FPS 15, 동시 4대, 저장경로 |
| Vision Pipeline | `pipeline/src/detector.py`, `tracker.py` | YOLO11n, ByteTrack |
| 트리거 7종 | `사고분석_설계/src/layer1_vision/trigger_detector.py` | T1~T7 |
| VisionPipeline | `사고분석_설계/src/layer1_vision/vision_pipeline.py` | 통합 파이프라인 |
| E2E 러너 (클립용) | `사고분석_설계/src/run_pipeline.py` | 오프라인 클립 처리 (참고) |

## 제약

- CPU 전용 (GPU 없음) → 프레임 샘플링 1fps로 부하 관리
- ITS_API_KEY 설정 완료 (`/workspace/.env`)
- ffmpeg 사용 가능
- 저장 경로: `/media/ybs/Expansion/CCTV차종분류/accident_data/stream/` (3.2TB 여유)
- 이중 config.py 충돌 주의: `layer0_data/config.py` (Track3 설정) vs `pipeline/src/config.py` (YOLO 설정)

## 성공 기준

1. `python run_realtime.py dry-run` — 전체 흐름 시뮬레이션 통과 (CCTV 조회, ITS 조회, Vision Pipeline, 교차확인, 저장)
2. `python run_realtime.py monitor --max-frames 30` — 실제 CCTV 스트림 30프레임 처리 에러 없이 완료
3. `python run_realtime.py status` — 수집 현황 정상 출력
4. 트리거 발화 → ITS 교차확인 → 사고면 flush, 아니면 폐기 흐름이 로그로 확인 가능

## 하지 말 것

- MLLM(Qwen2.5-VL) 연동 (이 단계에서는 Vision Pipeline 트리거만 사용)
- 모델 학습/교체
- DuckDB 연동 (메타데이터는 JSON/JSONL로 저장)
- 14건 배치 테스트 코드 수정
- 기존 track3_auto_collector.py 수정 (신규 파일로 작성)
