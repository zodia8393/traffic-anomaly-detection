# /goal 프롬프트: 사고다발 CCTV 24시간 녹화 → 사고 영상 데이터셋 자동 구축

아래 내용을 `/goal` 커맨드의 `$ARGUMENTS`로 사용.

---

## 목표

사고다발 고속도로 CCTV를 24시간 연속 녹화하고, ITS 돌발상황정보와 시간 매칭하여 **실제 사고 영상 클립을 자동 추출**하는 파이프라인을 구축한다. AI Hub 다운로드 없이 **우리가 직접 사고 영상 데이터셋을 만든다**.

## 산출물

1. **24시간 녹화 스크립트**: `record_24h.py` — 사고다발 CCTV N대 동시 녹화
2. **사고 클립 추출 스크립트**: `extract_accident_clips.py` — 돌발상황정보 매칭 → 사고 전후 클립 자동 추출
3. **실행 래퍼**: `run_dataset_builder.sh` — 녹화+추출 원커맨드 실행
4. **데이터셋 메타데이터**: 사고 클립별 JSON (시각, 위치, 사고유형, CCTV명)

## 배경

### 왜 직접 수집인가

- AI Hub #597 (교통사고 영상)은 다운로드 대기 + 차종 세분류 없음
- ITS API로 실시간 CCTV 스트림 접근 가능 (인증키 승인 완료)
- ITS 돌발상황정보 API로 사고 발생 시각·위치 정확히 알 수 있음
- 두 API를 조합하면 **사고 시점 ± 전후 영상**을 정확히 추출 가능

### 기존 자산

| 자산 | 위치 | 상태 |
|------|------|------|
| ITS API 키 | `/workspace/.env` → `ITS_API_KEY` | 승인 완료, 10개 서비스 |
| CCTV 스트림 코드 | `src/layer0_data/track3_cctv_stream.py` | HLS→ffmpeg 녹화 구현됨 |
| 돌발상황 API | `src/layer0_data/incident_reactor.py` | 사고/고장 필터링 구현됨 |
| Tier2 핫스팟 목록 | `src/layer0_data/run_nationwide.py` | 사고다발 50대 CCTV 정의됨 |
| 기존 녹화 클립 | `accident_data/stream/accident_clips/` | 11건 (트리거 기반 단편) |
| 외장 스토리지 | `/media/ybs/Expansion/` | 15TB (충분) |

### ITS API 스펙

**CCTV 화상자료 API**:
```
GET https://openapi.its.go.kr:9443/cctvInfo
  ?apiKey={KEY}&type=ex&cctvType=4&minX=...&maxX=...&minY=...&maxY=...&getType=json
```
- `type=ex` 고속도로, `type=its` 국도
- `cctvType=4` HLS HTTPS 실시간 스트림
- 응답: `cctvurl`(HLS m3u8), `cctvname`, `coordx/y`

**돌발상황정보 API**:
```
GET https://openapi.its.go.kr:9443/eventInfo
  ?apiKey={KEY}&type=ex&eventType=all&getType=json
```
- 응답: 사고 위치(노선/이정), 발생시각, 사고유형, 상태

### 저장 용량 추정

| 항목 | 값 |
|------|-----|
| CCTV 1대 HLS 비트레이트 | ~1.5 Mbps (일반), ~3 Mbps (고화질) |
| 1대 × 24시간 | ~16GB (1.5Mbps 기준) |
| 10대 × 24시간 | ~160GB |
| 20대 × 24시간 | ~320GB |
| 외장 여유 | 15TB → 충분 |
| 사고 클립 (전후 5분) | ~100MB/건 |

## 설계

### 전체 흐름

```
① CCTV 목록 선정 (사고다발 10~20대)
     ↓
② 24시간 연속 녹화 (ffmpeg -c copy, 1시간 단위 분할)
     ↓  (동시에)
③ 돌발상황정보 API 5분 간격 폴링 → 사고 이벤트 로깅
     ↓
④ 녹화 완료 후: 사고 이벤트 시각 매칭
     ↓
⑤ ffmpeg로 사고 전후 ±5분 클립 추출
     ↓
⑥ 메타데이터 JSON 생성 (시각, 위치, 사고유형, CCTV명)
```

### record_24h.py 설계

```python
# 핵심 기능:
# 1. ITS CCTV API로 대상 CCTV의 HLS URL 획득
# 2. ffmpeg subprocess로 동시 녹화 (asyncio 또는 multiprocessing)
# 3. 1시간 단위 segment 분할 (-f segment -segment_time 3600)
# 4. 동시에 돌발상황정보 5분 폴링 → incident_log.jsonl 적재
# 5. 24시간 후 자동 종료 + 요약 출력

# CCTV 선정 기준:
# - Tier2 핫스팟 50대 중 사고 빈도 상위 10~20대
# - 또는 사용자가 직접 CCTV명/좌표 지정

# 녹화 파라미터:
# ffmpeg -i {hls_url} -c copy -f segment -segment_time 3600
#        -strftime 1 "{output_dir}/{cctv_id}_%Y%m%d_%H%M%S.mp4"

# 장애 대응:
# - HLS URL 만료 시 API 재호출 (URL은 주기적으로 갱신됨)
# - ffmpeg 프로세스 watchdog (죽으면 재시작)
# - 디스크 용량 체크 (여유 10GB 미만이면 경고+중단)
```

### extract_accident_clips.py 설계

```python
# 입력:
# - 녹화된 1시간 segment 파일들: {cctv_id}_YYYYMMDD_HHMMSS.mp4
# - 돌발상황 로그: incident_log.jsonl

# 처리:
# 1. incident_log에서 녹화 CCTV와 매칭되는 사고 이벤트 필터링
#    - 매칭 기준: CCTV 좌표 ↔ 사고 위치 (반경 5km 이내)
# 2. 사고 발생 시각 기준 전후 5분 (총 10분) 클립 추출
#    - ffmpeg -ss {start} -t 600 -i {segment.mp4} -c copy {clip.mp4}
# 3. 메타데이터 JSON 생성

# 출력 구조:
# accident_dataset/
# ├── clips/
# │   ├── ACC_20260527_143022_경부선_오산_T2.mp4
# │   └── ...
# └── metadata/
#     ├── ACC_20260527_143022_경부선_오산_T2.json
#     └── dataset_summary.json
```

### 메타데이터 JSON 구조

```json
{
  "clip_id": "ACC_20260527_143022_경부선_오산_T2",
  "cctv_name": "[경부선] 오산",
  "cctv_id": "0010",
  "coord": {"x": 127.05, "y": 37.15},
  "accident_time": "2026-05-27T14:30:22",
  "clip_start": "2026-05-27T14:25:22",
  "clip_end": "2026-05-27T14:35:22",
  "duration_sec": 600,
  "incident_type": "사고",
  "incident_detail": "추돌",
  "road_name": "경부선",
  "direction": "부산방향",
  "source_segments": ["경부선_오산_20260527_140000.mp4", "경부선_오산_20260527_150000.mp4"],
  "file_size_mb": 95.2
}
```

## 파일 구조

```
/workspace/prj_cctv/사고분석_설계/
├── src/layer0_data/
│   ├── record_24h.py           ← 24시간 녹화 스크립트
│   ├── extract_accident_clips.py  ← 사고 클립 추출
│   └── run_dataset_builder.sh  ← 원커맨드 래퍼

/media/ybs/Expansion/CCTV차종분류/accident_data/
├── recording_24h/
│   ├── 20260527/               ← 날짜별
│   │   ├── 경부선_오산/        ← CCTV별 1시간 segment
│   │   │   ├── 경부선_오산_20260527_000000.mp4
│   │   │   ├── 경부선_오산_20260527_010000.mp4
│   │   │   └── ...
│   │   └── 중부선_이천/
│   │       └── ...
│   └── incident_log_20260527.jsonl   ← 당일 돌발상황 로그
│
├── accident_dataset/            ← 최종 산출물
│   ├── clips/                  ← 사고 클립 MP4
│   ├── metadata/               ← 클립별 JSON
│   └── dataset_summary.json    ← 전체 통계
│
└── stream/                     ← 기존 실시간 수집 (유지)
```

## CCTV 선정 기준

Tier2 핫스팟 또는 사고다발 구간 CCTV 중 다음 조건으로 10~20대 선정:

1. **고속도로** (`type=ex`) 우선 — 해상도·안정성 우수
2. **사고 빈도 상위** — 경부선, 중부선, 영동선 주요 구간
3. **다양한 도로 조건** — 직선, 커브, IC 부근, 터널 전후
4. **HLS 스트림 안정성** — 사전 테스트로 끊김 없는 것 확인

선정은 스크립트 내에서 좌표 bbox로 CCTV 목록을 가져온 뒤, 사용자가 확인·수정할 수 있게 JSON 설정 파일로 관리.

## 구현 지침

### 필수

1. **ffmpeg -c copy** — 재인코딩 없이 원본 그대로 저장 (CPU 부하 최소)
2. **1시간 segment 분할** — `-f segment -segment_time 3600 -strftime 1`
3. **HLS URL 갱신** — ITS API의 HLS URL은 일정 시간 후 만료될 수 있음. 1시간마다 API 재호출하여 URL 갱신
4. **돌발상황 로그** — 5분 간격 폴링, JSONL 형식으로 append
5. **디스크 감시** — `shutil.disk_usage()` 로 여유 확인, 10GB 미만이면 중단
6. **프로세스 watchdog** — ffmpeg 프로세스가 죽으면 자동 재시작
7. **API 호출 제한 고려** — ITS API 일일 제한 있음. CCTV URL은 1시간 1회만 갱신

### 돌발상황정보 매칭 로직

```
for each incident in incident_log:
    for each recording_cctv:
        distance = haversine(incident.coord, cctv.coord)
        if distance < 5km AND incident.road == cctv.road:
            matched!
            clip_start = incident.time - 5min
            clip_end = incident.time + 5min
            extract_clip(segments, clip_start, clip_end)
```

### 하지 말 것

- ffmpeg 재인코딩 금지 (`-c copy`만 사용)
- 녹화 중 AI 분석 금지 — 녹화에만 집중 (분석은 별도 파이프라인)
- workspace NVMe에 녹화 파일 저장 금지 — **외장 15TB만 사용**
- API 과도 호출 금지 — CCTV URL 갱신 1시간 1회, 돌발상황 5분 1회

## 실행 방법

```bash
# 1단계: CCTV 목록 확인 + 설정
python3 record_24h.py --list-cameras --road ex --region 수도권

# 2단계: 24시간 녹화 시작 (백그라운드)
nohup python3 record_24h.py --cameras cameras_config.json --duration 24h \
    --output /media/ybs/Expansion/CCTV차종분류/accident_data/recording_24h/ \
    > recording.log 2>&1 &

# 3단계: 녹화 완료 후 사고 클립 추출
python3 extract_accident_clips.py \
    --recordings /media/ybs/Expansion/.../recording_24h/20260527/ \
    --incidents /media/ybs/Expansion/.../recording_24h/incident_log_20260527.jsonl \
    --output /media/ybs/Expansion/.../accident_dataset/ \
    --margin 300  # 전후 5분(300초)
```

## 성공 기준

1. **녹화 스크립트** 동작: CCTV 10대 이상 동시 녹화, 1시간 segment 생성
2. **ffmpeg watchdog**: 프로세스 죽으면 자동 재시작
3. **돌발상황 로그**: 5분 간격 폴링, JSONL 적재
4. **클립 추출**: 사고 이벤트 ↔ CCTV 매칭 → 전후 5분 MP4 클립
5. **메타데이터 JSON**: 클립마다 시각/위치/사고유형 기록
6. **디스크 안전**: 여유 10GB 미만 시 자동 중단
7. **API 호출 제한 준수**: CCTV 1시간 1회, 돌발상황 5분 1회
