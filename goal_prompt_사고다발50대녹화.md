# /goal 프롬프트: 고속도로 사고다발 CCTV 50대 하루치 일괄 녹화

아래 내용을 `/goal` 커맨드의 `$ARGUMENTS`로 사용.

---

## 목표

고속도로 CCTV 4,760대 중 **사고다발 구간 50대**를 지역 비중복으로 선정하고, **하루(24시간) 일괄 녹화**하여 사고 영상 데이터셋 구축의 원본 소스를 확보한다.

## 산출물

1. **CCTV 50대 선정 JSON**: `cameras_50.json` — 이름, 좌표, 노선, HLS URL
2. **일괄 녹화 스크립트**: `record_50.py` — 50대 동시 녹화, watchdog, URL 갱신, 디스크 감시
3. **녹화 데이터**: `/DATA/cctv_recording/{날짜}/{CCTV별}/` — 통짜 MP4
4. **영상 통합**: 24시간 녹화 후 CCTV당 파편 → `ffmpeg concat` → `{CCTV}_full.mp4` 1개로 병합
5. **녹화 리포트**: 녹화 완료 후 요약 JSON (CCTV별 녹화시간, 파일크기, 끊김 횟수, 병합 결과)

## 배경

### 현재 상태

| 자산 | 상태 |
|------|------|
| ITS API 키 (신규) | `883c4ddddab648b2bc91dc66395bc6ea` — `/workspace/.env` `ITS_API_KEY` |
| ITS CCTV API | 고속도로 4,760대, HLS 실시간 스트림 |
| API 일일 제한 | ~1,000회/키 (추정) |
| 오산 테스트 녹화 | `/DATA/cctv_recording/20260527/경부선_오산/` — 36MB (테스트 완료) |
| 기존 코드 | `record_continuous.py` — 1대 녹화 구현 완료 |
| 인프라 | i9-285K 24코어, 122GB RAM, `/DATA` 6.3TB 여유 |

### 인프라 적합성 (검증 완료)

| 항목 | 50대 × 24시간 | 판정 |
|------|--------------|------|
| CPU | ~25% (ffmpeg -c copy) | OK |
| RAM | ~1.5GB | OK |
| 네트워크 | 50Mbps | OK |
| 저장 | **540GB** | `/DATA` 6.3TB → OK |
| API | URL갱신 600회/일 | 1,000회 한도 내 |

### API 사용 예산

| 용도 | 호출 수 |
|------|---------|
| 전국 CCTV 목록 조회 (1회) | 1 |
| 50대 URL 초기 획득 | 이미 목록에 포함 (추가 0) |
| URL 갱신 (90분 주기 × 50대 × 16회) | ~800 |
| 여유 | ~199 |
| **합계** | ~801 / 1,000 |

**핵심**: CCTV 목록 API 1회 호출로 4,760대 전체 + URL을 한번에 받고, 이후는 URL 갱신만.

### ITS CCTV API cctvType 구분

| cctvType | 설명 | 비고 |
|----------|------|------|
| 1 | 실시간 스트리밍 (HLS) | HTTP |
| 2 | 동영상 (MP4) | HTTP |
| 3 | 정지 영상 | - |
| 4 | 실시간 스트리밍 (HLS) | **HTTPS** — 우선 사용 |
| 5 | 동영상 (MP4) | **HTTPS** |

**cctvType=4 (HTTPS HLS)** 를 기본으로 사용. 실패 시 cctvType=1 (HTTP HLS) 폴백.

## 설계

### Phase 1: CCTV 50대 선정 (API 1회)

```python
# 1. ITS API로 전국 고속도로 CCTV 전체 조회 (1회 호출)
#    GET cctvInfo?type=ex&cctvType=4&minX=124&maxX=132&minY=33&maxY=39
#    → cctvType=4 (HTTPS HLS) 우선, 실패 시 cctvType=1 폴백
#    → 4,760대 (이름, 좌표, HLS URL 포함)

# 2. 사고다발 구간 좌표 50개 (하드코딩)
#    - 교통공학적으로 알려진 고속도로 사고다발 구간
#    - 전국 주요 노선별 분산 배치
#    - 최소 이격 20km

# 3. 각 사고다발 좌표에 최근접 CCTV 1대 매칭
#    → cameras_50.json 생성
```

### 사고다발 구간 50개 선정 기준

**전국 분산 + 노선 다양성 + 사고 빈도** 기반으로 아래 노선에서 배분:

| 노선 | 배정 | 대표 구간 |
|------|------|----------|
| 경부선 | 8대 | 양재~판교, 천안~안성, 신탄진~회덕, 김천~구미, 경주~울산 |
| 중부선 | 4대 | 하남JC~이천, 음성~충주 |
| 영동선 | 5대 | 여주JC~원주, 횡성~강릉, 대관령 |
| 서해안선 | 5대 | 서평택~안중, 서천~군산, 목포 부근 |
| 호남선 | 4대 | 논산JC~익산, 광주 부근 |
| 중앙선 | 4대 | 춘천~원주, 안동~영주 |
| 남해선 | 4대 | 부산~김해, 진주~사천 |
| 제2경인 | 2대 | 인천~안양 |
| 제2중부 | 2대 | 마장JC~음성 |
| 중부내륙 | 2대 | 여주~충주 |
| 수도권제1순환 | 3대 | 판교~성남, 하남~구리 |
| 제2순환 | 2대 | 오산~화성 |
| 동해선 | 2대 | 울산~포항, 강릉~속초 |
| 기타 | 3대 | 광주대구, 익산장수, 세종포천 등 |
| **합계** | **50대** | |

### Phase 2: 일괄 녹화 (`record_50.py`)

```python
# 핵심 구조:
# - asyncio 기반 50개 ffmpeg subprocess 관리
# - 각 CCTV마다 독립 Task: ffmpeg 실행 + watchdog + 로깅
# - 공유 자원: URL 갱신 스케줄러 (90분 주기), 디스크 감시 (10분 주기)

# ffmpeg 명령 (통짜 MP4, 재인코딩 없음):
# ffmpeg -y -reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 30
#        -i {hls_url} -c copy {output_dir}/{slug}_{timestamp}.mp4

# Watchdog:
# - ffmpeg 프로세스 poll() → 죽으면 자동 재시작
# - 파일 크기 5분간 증가 없으면 → 재시작
# - 재시작 시 새 파일 생성 (이어붙이기 아님)

# URL 갱신:
# - 90분마다 ITS API 재호출 (만료 2시간 전 선제 갱신)
# - 갱신 시 ffmpeg 재시작 → 새 파일
# - API 1회 호출로 50대 전체 URL 갱신 (bbox 전국)
#   → 90분 × 16회 = 16 API 호출/일

# 디스크 감시:
# - shutil.disk_usage() 10분 주기 체크
# - 50GB 미만 시 경고 로깅
# - 20GB 미만 시 녹화 중단

# 시그널 핸들링:
# - SIGINT/SIGTERM → 전체 ffmpeg graceful 종료
# - 녹화 리포트 JSON 출력 후 종료
```

### Phase 3: URL 갱신 최적화

```python
# 문제: 50대 개별 API 호출 → 50회/갱신 → 800회/일
# 해결: 전국 bbox 1회 호출 → 4,760대 응답 → 50대 URL 추출
#
# refresh_all_urls():
#   all_cctvs = api.get(minX=124, maxX=132, minY=33, maxY=39)  # 1회
#   for cam in cameras_50:
#       matched = find_by_id(all_cctvs, cam.cctv_id)
#       cam.url = matched.url
#   return cameras_50
#
# → 90분 × 16회 = 16 API 호출/일 (800회 → 16회로 절감!)
```

### Phase 4: 녹화 완료 후 영상 통합

```python
# URL 갱신마다 새 파일이 생기므로, 24시간 녹화 후 CCTV당 ~16개 파일 존재
# 이를 CCTV당 1개 통짜 MP4로 병합
#
# merge_daily():
#   for each cctv_dir in recording_dir:
#       fragments = sorted(glob("*.mp4"))  # 시간순 정렬
#       if len(fragments) <= 1:
#           continue
#
#       # ffmpeg concat demuxer (재인코딩 없음)
#       concat_list = write_concat_txt(fragments)
#       output = f"{cctv_slug}_20260528_full.mp4"
#       ffmpeg -f concat -safe 0 -i concat.txt -c copy {output}
#
#       # 무결성 검증
#       ffprobe {output} → duration, size 확인
#
#       # 원본 파편 정리: merged/ 하위로 이동 (삭제는 사용자 판단)
#       move fragments → {cctv_dir}/fragments/

# 출력 구조 (병합 후):
# /DATA/cctv_recording/20260528/경부선_양재/
# ├── 경부선_양재_20260528_full.mp4          ← 통합본 (약 10GB)
# └── fragments/                            ← 원본 파편 보관
#     ├── 경부선_양재_20260528_000500.mp4
#     ├── 경부선_양재_20260528_013500.mp4
#     └── ...

# 실행:
#   python3 record_50.py --merge --date 20260528
#   → CCTV 50대 × concat → 50개 _full.mp4 생성
#   → 병합 리포트: 원본 파편 수, 통합 duration, gap 유무
```

## 파일 구조

```
/workspace/prj_cctv/사고분석_설계/
├── src/layer0_data/
│   ├── record_continuous.py      ← 기존 1대 녹화 (유지)
│   ├── record_50.py              ← 50대 일괄 녹화 (신규)
│   └── cameras_50.json           ← 선정된 50대 CCTV 정보

/DATA/cctv_recording/
├── 20260527/                     ← 날짜별
│   ├── 경부선_오산/              ← 테스트 녹화 (36MB, 완료)
│   └── ...
├── 20260528/                     ← 50대 녹화일 (예시)
│   ├── 경부선_양재/
│   │   ├── 경부선_양재_20260528_full.mp4         ← 통합본 (병합 후)
│   │   └── fragments/                           ← 원본 파편 보관
│   │       ├── 경부선_양재_20260528_000500.mp4
│   │       ├── 경부선_양재_20260528_013500.mp4
│   │       └── ...
│   ├── 경부선_천안/
│   ├── 영동선_여주JC/
│   └── ... (50개 디렉토리)
└── recording_report_20260528.json  ← 녹화 리포트
```

### 녹화 리포트 JSON 구조

```json
{
  "date": "2026-05-28",
  "total_cameras": 50,
  "duration_hours": 24,
  "cameras": [
    {
      "name": "[경부선] 양재",
      "road": "경부선",
      "files": 16,
      "total_size_gb": 10.2,
      "total_duration_sec": 85200,
      "gaps_count": 2,
      "gaps_total_sec": 1200
    }
  ],
  "summary": {
    "total_size_gb": 512,
    "avg_uptime_pct": 97.5,
    "cameras_with_gaps": 8
  }
}
```

## 실행 방법

```bash
# 1단계: CCTV 50대 선정 (API 1회)
python3 record_50.py --select-cameras
# → cameras_50.json 생성, 선정 결과 출력

# 2단계: 녹화 시작 (백그라운드)
nohup python3 record_50.py --record --duration 24h \
    --output /DATA/cctv_recording/ \
    > /DATA/cctv_recording/record_50.log 2>&1 &

# 3단계: 모니터링
python3 record_50.py --status
# → 각 CCTV 녹화 상태, 파일 크기, 끊김 횟수 실시간 출력

# 4단계: 수동 중단 (필요 시)
kill -TERM <PID>
# → graceful 종료 + 리포트 생성

# 5단계: 녹화 완료 후 영상 통합
python3 record_50.py --merge --date 20260528
# → CCTV당 파편 → ffmpeg concat → _full.mp4 1개
# → 원본 파편은 fragments/ 하위로 이동
```

## 구현 지침

### 필수

1. **ffmpeg -c copy** — 재인코딩 없이 원본 그대로
2. **통짜 MP4** — 세그먼트 분할 안 함 (URL 갱신 시만 새 파일)
3. **URL 갱신 최적화** — 전국 bbox 1회 호출로 50대 전체 갱신 (16회/일)
4. **Watchdog** — ffmpeg 죽으면 자동 재시작, 파일 크기 정체 감지
5. **디스크 감시** — 50GB 미만 경고, 20GB 미만 중단
6. **Graceful 종료** — SIGINT/SIGTERM → 전체 정리 + 리포트
7. **로깅** — CCTV별 독립 로그 + 전체 요약 로그

### 하지 말 것

- ffmpeg 재인코딩 금지 (`-c copy`만 사용)
- 녹화 중 AI 분석 금지 — 녹화에만 집중
- workspace NVMe에 녹화 파일 저장 금지 — **`/DATA` HDD만 사용**
- API 과도 호출 금지 — 전국 bbox 1회로 50대 URL 일괄 갱신
- 1인칭 세그먼트 분할 금지 — 통짜 MP4 (URL 갱신 시만 새 파일)

## 성공 기준

1. **cameras_50.json**: 50대 CCTV 선정, 전국 분산, 최소 10개 노선 포함
2. **동시 녹화**: 50대 ffmpeg 프로세스 동시 실행
3. **24시간 녹화**: 각 CCTV 평균 uptime 90% 이상
4. **API 예산**: 일일 100회 이내 (전국 bbox 갱신 방식)
5. **파일 무결성**: ffprobe로 검증 가능한 MP4
6. **영상 통합**: CCTV당 파편 → `_full.mp4` 1개, ffprobe duration 정합성 확인
7. **녹화 리포트**: JSON으로 CCTV별 녹화시간/크기/끊김/병합결과 기록
8. **디스크 안전**: 20GB 미만 시 자동 중단
