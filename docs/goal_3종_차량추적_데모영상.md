# Goal: 3종 차량 분류 + 추적 데모 영상 제작

## 목적

제안서 12~13장 시각자료용 **데모 영상** 제작. CCTV 녹화 영상에서 차량을 검출·추적하고, 승용차/버스/트럭 3종으로 분류하여 바운딩박스와 라벨이 표시된 ~5분 영상을 출력한다.

## 입력

- **영상**: `/DATA/cctv_recording/20260528/경부선_원지동_20260528_141543.mp4`
  - 720×480, H.264, ~30fps, 24시간 원본 (10.7GB)
  - 이 중 교통량이 적당한 구간 **5분(9,000프레임)** 만 사용
  - 시작점은 영상 앞부분(0:00)부터. 차량이 너무 적으면 30분~1시간 지점으로 이동
- **검출 모델**: `/workspace/prj_cctv/사고분석_설계/src/yolov8n.pt` (COCO pretrained)
- **추적**: ByteTrack (기존 `tracker.py` 참조 가능)

## 3종 분류 매핑

YOLO COCO 검출 결과를 3종으로 매핑. 별도 학습 불필요.

| COCO class | COCO id | 3종 분류 | 표시 색상 |
|------------|---------|---------|----------|
| car | 2 | 승용차 | 초록 `#00CC00` |
| bus | 5 | 버스 | 파랑 `#0066FF` |
| truck | 7 | 트럭 | 빨강 `#FF3300` |
| motorcycle | 3 | (제외) | — |

- COCO car(2) 이외의 소형 차량도 car에 포함
- motorcycle(3)은 제외 (고속도로 희귀)
- 그 외 COCO 클래스(사람 등)도 제외

## 출력

### 데모 영상
- **경로**: `/workspace/prj_cctv/사고분석_설계/output/demo_3class_tracking.mp4`
- **길이**: 약 5분 (300초)
- **해상도**: 원본 유지 (720×480)
- **코덱**: H.264, mp4 컨테이너
- **fps**: 원본 유지 (~30fps) 또는 처리 속도에 따라 15fps도 수용

### 영상 위 표시 요소

1. **바운딩박스**: 차량마다 색상 구분된 사각형
2. **라벨**: 박스 상단에 `{종류} #{트랙ID}` (예: `승용차 #12`, `트럭 #5`)
   - 한글 라벨 (PIL ImageFont 또는 cv2.putText + 한글 폰트)
   - 한글 폰트 경로: `/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf` (없으면 설치)
3. **실시간 집계**: 화면 좌상단에 현재까지 누적 통과 차량 수
   ```
   승용차: 142  버스: 8  트럭: 23
   ```
4. **프레임 정보**: 우상단에 프레임 번호 / 경과 시간

### 집계 로직

- ByteTrack으로 차량별 고유 ID 부여
- 트랙이 화면에서 사라지면(lost) 해당 트랙의 **최빈 클래스**를 최종 분류로 확정 → 누적 카운트 +1
- 동일 트랙 내 프레임별 분류가 흔들려도 다수결로 안정화

## 구현 방향

### 스크립트: `make_demo_video.py`
- **위치**: `/workspace/prj_cctv/사고분석_설계/src/layer0_data/make_demo_video.py`
- 단일 파일, 독립 실행 가능

### 핵심 로직 (의사코드)

```python
import cv2
from ultralytics import YOLO
from collections import defaultdict, Counter

model = YOLO("yolov8n.pt")
COCO_3CLASS = {2: "승용차", 5: "버스", 7: "트럭"}
COLORS = {"승용차": (0,204,0), "버스": (255,102,0), "트럭": (0,51,255)}

cap = cv2.VideoCapture(input_video)
# 시작점 seek (필요시)
# cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

writer = cv2.VideoWriter(output_path, ...)
track_classes = defaultdict(list)  # {track_id: [cls, cls, ...]}
finished_counts = Counter()       # 화면 이탈한 트랙의 최종 분류 집계

for frame_idx in range(total_frames):
    ret, frame = cap.read()
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    for box in results[0].boxes:
        cls_id = int(box.cls)
        if cls_id not in COCO_3CLASS:
            continue
        cls_name = COCO_3CLASS[cls_id]
        track_id = int(box.id) if box.id is not None else -1

        track_classes[track_id].append(cls_name)

        # 바운딩박스 + 라벨 그리기
        draw_box(frame, box.xyxy, cls_name, track_id, COLORS[cls_name])

    # 사라진 트랙 처리 → 누적 집계
    update_finished_tracks(...)

    # 좌상단 집계, 우상단 프레임 정보
    draw_overlay(frame, finished_counts, frame_idx, fps)
    writer.write(frame)
```

### YOLO 내장 tracker 사용

- `model.track(frame, persist=True, tracker="bytetrack.yaml")`로 YOLO 내장 ByteTrack 사용
- 별도 tracker.py 의존 없이 ultralytics 내장 기능 활용
- `box.id`로 트랙 ID 직접 접근

### 한글 렌더링

- OpenCV `putText`는 한글 미지원 → **PIL ImageDraw** 사용
- cv2 frame ↔ PIL Image 변환하여 한글 텍스트 렌더링

```python
from PIL import Image, ImageDraw, ImageFont
font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf", 16)

def draw_korean_text(frame, text, pos, color):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
```

### 성능 고려

- 720×480 @ yolov8n: ~21ms/프레임 (CPU) → 5분 영상 처리 약 3~4분
- 9,000프레임 × 21ms ≈ 189초
- 메모리: yolov8n ~30MB, 프레임 버퍼 ~1MB → 여유

## CLI

```bash
cd /workspace/prj_cctv/사고분석_설계/src/layer0_data

python make_demo_video.py \
  --input "/DATA/cctv_recording/20260528/경부선_원지동_20260528_141543.mp4" \
  --output "/workspace/prj_cctv/사고분석_설계/output/demo_3class_tracking.mp4" \
  --duration 300 \
  --start 0
```

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--input` | (필수) | 입력 영상 경로 |
| `--output` | `output/demo_3class_tracking.mp4` | 출력 경로 |
| `--duration` | 300 | 초 단위 (5분) |
| `--start` | 0 | 시작 위치 (초) |
| `--model` | `yolov8n.pt` | YOLO 모델 경로 |
| `--conf` | 0.3 | 검출 신뢰도 임계값 |
| `--fps` | (원본) | 출력 fps (미지정 시 원본) |

## 검증 항목

- [ ] 출력 영상이 ~5분이고 재생 가능
- [ ] 바운딩박스가 차량을 정확히 추적 (ID 유지)
- [ ] 한글 라벨(승용차/버스/트럭)이 정상 표시
- [ ] 색상 구분이 명확 (초록/파랑/빨강)
- [ ] 좌상단 누적 집계가 점진적으로 증가
- [ ] 트랙 ID가 동일 차량에 일관 유지
- [ ] 영상 품질이 제안서 시각자료로 사용 가능한 수준

## 산출물

- `/workspace/prj_cctv/사고분석_설계/src/layer0_data/make_demo_video.py`
- `/workspace/prj_cctv/사고분석_설계/output/demo_3class_tracking.mp4`
