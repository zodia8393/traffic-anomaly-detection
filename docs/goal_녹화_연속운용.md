# Goal: 기존 녹화 MP4 변환 + 24시간 자동 로테이션 연속 녹화

## 작업 순서 (시간 긴급)
1. **즉시**: 새 녹화 먼저 시작 (기존과 동일 카메라 5대, 동일 설정)
2. **백그라운드**: 기존 .ts → .mp4 변환
3. **코드 수정**: 24시간 자동 로테이션 기능 추가
4. **로테이션 적용**: 현재 녹화를 중단하고 로테이션 버전으로 재시작

## 1단계: 즉시 녹화 재시작

기존 `record_hls_multi.py`로 즉시 시작. 로테이션 코드 완성 전까지 임시로 돌림.

```bash
cd /workspace/prj_cctv/사고분석_설계/src/layer0_data
PYTHONUNBUFFERED=1 nohup python3 record_hls_multi.py \
  --config /tmp/cameras_5routes.json \
  --duration 24h \
  > /DATA/cctv_recording/record_multi_24h.log 2>&1 &
```

출력 디렉토리: `/DATA/cctv_recording/20260529/` (날짜 자동 생성)

## 2단계: 기존 .ts → .mp4 변환

### 대상 파일 (5개)
```
/DATA/cctv_recording/20260528/경부선_원지동_hls/경부선_원지동_20260528_141543.ts     → 10.7GB
/DATA/cctv_recording/20260528/남해선_망덕교_hls/남해선_망덕교_20260528_141543.ts     → 20.6GB
/DATA/cctv_recording/20260528/영동선_주천강교_hls/영동선_주천강교_20260528_141544.ts → 10.4GB
/DATA/cctv_recording/20260528/중앙선_제천휴게소_hls/중앙선_제천휴게소_20260528_141544.ts → 5.5GB
/DATA/cctv_recording/20260528/경인선_도당2_hls/경인선_도당2_20260528_141545.ts       → 10.8GB
```

### 변환 방법
```bash
ffmpeg -i input.ts -c copy output.mp4
```
- 코덱 복사(`-c copy`)이므로 재인코딩 없이 컨테이너만 변환 → 빠름
- 변환 완료 후 원본 .ts 파일 삭제 (용량 절약, 58GB 회수)
- **백그라운드에서 5개 순차 실행** (동시 실행 시 I/O 부하)

### 출력 파일명 규칙
```
{노선}_{CCTV명}_{시작일}_{시작시각}.mp4
```
예: `경부선_원지동_20260528_141543.mp4`

같은 디렉토리에 저장하되, `_hls` 서브디렉토리 안이 아닌 날짜 디렉토리(`20260528/`) 직하에 배치.

## 3단계: record_hls_multi.py 자동 로테이션 기능 추가

### 핵심 변경
현재: `--duration 24h` → 24시간 후 종료, 수동 재시작 필요
변경: `--continuous` 모드 추가 → 24시간마다 파일을 닫고 새 파일을 열어 무한 녹화

### 로테이션 로직

```python
# record_camera() 내부 변경 (개념)
while _running:
    # 24시간 단위 파일 생성
    ts_start = datetime.now()
    out_file = output_dir / f"{slug}_{ts_start.strftime('%Y%m%d_%H%M%S')}.ts"
    
    with open(out_file, "wb") as f:
        segment_start = time.time()
        while _running and (time.time() - segment_start) < rotation_sec:
            # 기존 HLS 세그먼트 다운로드 로직 (변경 없음)
            ...
    
    # 24시간 경과 → 파일 닫힘 → 자동으로 mp4 변환 → 새 파일 시작
    convert_to_mp4(out_file)  # 백그라운드 스레드로 변환
    # while 루프 상단으로 돌아가 새 파일 생성
```

### 자동 MP4 변환
- 24시간 파일이 닫히면 **별도 스레드**에서 `ffmpeg -c copy` 변환
- 변환 완료 후 원본 .ts 삭제
- 변환 실패 시 .ts 보존 + WARNING 로그

### CLI 인터페이스 변경

```bash
# 기존 (1회 녹화)
python record_hls_multi.py --config cameras.json --duration 24h

# 신규 (연속 녹화, 24시간 로테이션)
python record_hls_multi.py --config cameras.json --continuous
python record_hls_multi.py --config cameras.json --continuous --rotation 24h
python record_hls_multi.py --config cameras.json --continuous --rotation 12h  # 12시간 단위도 가능
```

- `--continuous`: 무한 녹화 모드. `--rotation`으로 파일 분할 주기 지정 (기본 24h)
- `--duration`과 `--continuous`는 상호 배타
- SIGINT/SIGTERM 수신 시 현재 파일 정상 종료 후 mp4 변환까지 완료하고 종료

### 파일 구조

```
/DATA/cctv_recording/
├── 20260528/                          ← 변환 완료
│   ├── 경부선_원지동_20260528_141543.mp4
│   ├── 남해선_망덕교_20260528_141543.mp4
│   └── ...
├── 20260529/                          ← 현재 녹화 중
│   ├── 경부선_원지동_hls/
│   │   └── 경부선_원지동_20260529_HHMMSS.ts   ← 녹화 진행
│   └── ...
├── 20260530/                          ← 내일 자동 생성
│   └── ...
└── meta_multi.json                    ← 누적 메타데이터
```

### 날짜 디렉토리 정책
- 파일명의 날짜는 **녹화 시작 시각** 기준
- 24시간 로테이션이므로 자연스럽게 날짜별 디렉토리 분리
- 자정을 넘기는 경우: 시작 시각 기준이므로 `20260529/` 디렉토리에 저장

### 디스크 용량 관리
- 녹화 시작 전 `df -h /DATA` 확인
- 가용 공간 < 녹화 예상 용량(~60GB)이면 WARNING 후 계속 (중단하지 않음)
- 가용 공간 < 10GB이면 ERROR 후 녹화 중단

## 4단계: 로테이션 버전으로 재시작

1. 1단계에서 시작한 임시 녹화 종료 (SIGTERM)
2. 임시 녹화 .ts도 mp4 변환
3. 로테이션 버전으로 `--continuous` 시작

```bash
PYTHONUNBUFFERED=1 nohup python3 record_hls_multi.py \
  --config /tmp/cameras_5routes.json \
  --continuous \
  > /DATA/cctv_recording/record_continuous.log 2>&1 &
```

## 수정 대상 파일
- **수정**: `/workspace/prj_cctv/사고분석_설계/src/layer0_data/record_hls_multi.py`
  - `--continuous`, `--rotation` 인자 추가
  - `record_camera()` 내 로테이션 로직
  - `convert_to_mp4()` 함수 추가
  - 디스크 용량 체크 함수 추가
- **변경 없음**: `record_hls.py` (단일 카메라용, 별도 유지)

## 검증 항목
- [ ] 기존 20260528 .ts 5개 → .mp4 변환 완료, .ts 삭제, 용량 회수
- [ ] 새 녹화 즉시 시작되어 진행 중
- [ ] `--continuous` 모드 동작: 파일 로테이션 테스트 (짧은 주기로 확인)
- [ ] 로테이션 시 끊김 없음 (세션 유지, 새 파일만 열림)
- [ ] mp4 자동 변환 후 .ts 삭제
- [ ] SIGTERM 시 정상 종료 + 미변환 .ts mp4 변환 완료
- [ ] 디스크 용량 체크 동작
- [ ] 기존 `--duration` 모드 호환성 유지 (회귀 없음)

## 산출물
- `/DATA/cctv_recording/20260528/*.mp4` (5개, 변환 완료)
- `/workspace/prj_cctv/사고분석_설계/src/layer0_data/record_hls_multi.py` (로테이션 기능 추가)
- 연속 녹화 프로세스 실행 중 상태
