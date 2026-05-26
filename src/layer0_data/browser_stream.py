"""ITS 돌발정보 페이지 기반 CCTV 프레임 자동 캡처.

ITS 웹사이트 돌발정보(/map/outbreak) 페이지를 headless Chrome으로 열어
사고/고장 인근 CCTV 영상을 자동 순회하며 프레임을 캡처한다.

ITS HLS 프록시 속도 제한(~24회/세션)을 우회:
  돌발정보 페이지의 CCTV 재생은 UI 클릭 경로로 제한 없음.

Workflow:
  /map/outbreak 페이지 → 돌발정보 목록 스캔 → 사고/고장 필터
  → 인근 CCTV 순회 클릭 → hls.js 영상 재생 → canvas 프레임 캡처
  → numpy BGR 배열 반환 (파이프라인 연동)
"""
from __future__ import annotations

import base64
import io
import json
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright, Browser, Page, Playwright

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("outbreak_capture")

_OUTBREAK_URL = "https://its.go.kr/map/outbreak"
_PAGE_SETTLE_SEC = 8
_VIDEO_LOAD_TIMEOUT = 15
_MIN_READY_STATE = 2
_DEFAULT_CCTV_DELAY = 5.0
_DEFAULT_INCIDENT_DELAY = 3.0
_DEFAULT_POLL_INTERVAL = 300
_DEFAULT_MAX_PER_CYCLE = 50
_TARGET_TYPES = ("사고", "고장")


@dataclass
class OutbreakIncident:
    """돌발정보 페이지에서 추출한 사고 정보."""
    dom_index: int
    event_type: str
    road_name: str
    direction: str
    message: str
    incident_time: str = ""
    raw_text: str = ""
    cctvs: list[dict] = field(default_factory=list)


@dataclass
class CapturedFrame:
    """캡처된 CCTV 프레임 + 메타데이터."""
    incident_type: str
    road_name: str
    cctv_name: str
    distance: str
    frame: np.ndarray
    timestamp: float
    width: int
    height: int
    jpeg_size: int = 0


# ═══════════════════════════════════════════════════════════════════════
# JavaScript — 돌발정보 페이지 DOM 조작
# ═══════════════════════════════════════════════════════════════════════

_JS_SCAN_INCIDENTS = """() => {
    const lists = document.querySelectorAll('ul.info_list');
    if (lists.length === 0) return { error: 'ul.info_list not found', total: 0 };

    const items = [];
    for (const ul of lists) {
        for (const li of ul.querySelectorAll(':scope > li[data-idx]')) {
            const road = (li.querySelector('.title strong') || {}).textContent?.trim() || '';
            const type = (li.querySelector('.title .label') || {}).textContent?.trim() || '';
            const date = (li.querySelector('.sub_date') || {}).textContent?.trim() || '';
            items.push({
                index: parseInt(li.getAttribute('data-idx')),
                dataId: li.getAttribute('data-id') || '',
                road: road,
                type: type,
                date: date,
                text: road + type + ' ' + date,
            });
        }
    }

    return { total: items.length, items: items, lists: lists.length };
}"""

_JS_CLICK_INCIDENT = """(dataIdx) => {
    const li = document.querySelector('ul.info_list li[data-idx=\"' + dataIdx + '\"]');
    if (!li) return { ok: false, error: 'li[data-idx=' + dataIdx + '] not found' };

    const anchor = li.querySelector('a.item') || li.querySelector('a') || li;
    anchor.click();
    return { ok: true, text: (li.textContent || '').trim().substring(0, 100) };
}"""

_JS_GET_NEARBY_CCTVS = """() => {
    const cctvs = [];
    const seen = new Set();

    for (const el of document.querySelectorAll('a, span, li, button')) {
        const text = (el.textContent || '').trim();
        if (text.length > 60 || text.length < 3) continue;
        if (text.includes('\\n')) continue;
        if (!text.match(/\\d+\\.\\d+\\s*km/)) continue;
        if (seen.has(text)) continue;
        if (text.includes('사고') || text.includes('공사') || text.includes('통제') ||
            text.includes('고장') || text.includes('작업')) continue;

        seen.add(text);
        cctvs.push({
            index: cctvs.length,
            name: text,
            distance: (text.match(/(\\d+\\.\\d+\\s*km)/) || [''])[0],
        });
    }

    return { cctvs: cctvs, total: cctvs.length };
}"""

_JS_CLICK_CCTV = """(targetIndex) => {
    const cctvEls = [];
    const seen = new Set();

    for (const el of document.querySelectorAll('a, span, li, button')) {
        const text = (el.textContent || '').trim();
        if (text.length > 60 || text.length < 3) continue;
        if (text.includes('\\n')) continue;
        if (!text.match(/\\d+\\.\\d+\\s*km/)) continue;
        if (seen.has(text)) continue;
        if (text.includes('사고') || text.includes('공사') || text.includes('통제') ||
            text.includes('고장') || text.includes('작업')) continue;
        seen.add(text);
        cctvEls.push({ el, text });
    }

    if (targetIndex >= cctvEls.length) {
        return { ok: false, error: 'index ' + targetIndex + '/' + cctvEls.length };
    }

    cctvEls[targetIndex].el.click();
    return { ok: true, name: cctvEls[targetIndex].text };
}"""

_JS_CHECK_VIDEO = """() => {
    for (const v of document.querySelectorAll('video')) {
        if (v.readyState >= 2 && v.videoWidth > 0 && v.videoHeight > 0) {
            return {
                ready: true,
                width: v.videoWidth,
                height: v.videoHeight,
                readyState: v.readyState,
                src: (v.currentSrc || v.src || '').substring(0, 80),
            };
        }
    }
    return { ready: false };
}"""

_JS_CAPTURE_FRAME = """() => {
    for (const video of document.querySelectorAll('video')) {
        if (video.readyState < 2 || video.videoWidth === 0) continue;

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);

        const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
        return {
            dataUrl: dataUrl,
            width: video.videoWidth,
            height: video.videoHeight,
            size: Math.round(dataUrl.length * 0.75),
        };
    }
    return null;
}"""

_JS_DISCOVER_DOM = """() => {
    const info = { title: document.title, url: location.href, lists: [], videos: [] };

    for (const ul of document.querySelectorAll('ul, ol')) {
        if (ul.children.length > 2) {
            info.lists.push({
                tag: ul.tagName,
                id: ul.id || '',
                class: (ul.className || '').substring(0, 80),
                items: ul.children.length,
                sample: (ul.textContent || '').trim().substring(0, 150),
                parent: (ul.parentElement?.className || '').substring(0, 50),
            });
        }
    }
    for (const v of document.querySelectorAll('video')) {
        info.videos.push({
            id: v.id || '',
            readyState: v.readyState,
            w: v.videoWidth, h: v.videoHeight,
            src: (v.src || '').substring(0, 80),
        });
    }
    return info;
}"""


# ═══════════════════════════════════════════════════════════════════════
# OutbreakCCTVCapture
# ═══════════════════════════════════════════════════════════════════════

class OutbreakCCTVCapture:
    """ITS 돌발정보 페이지에서 사고 인근 CCTV 프레임을 자동 캡처.

    Workflow:
      /map/outbreak → 돌발정보 스캔 → 사고/고장 필터
      → 인근 CCTV 순회 클릭 → hls.js 영상 재생 → canvas 프레임 캡처
    """

    def __init__(
        self,
        target_types: tuple[str, ...] = _TARGET_TYPES,
        cctv_delay: float = _DEFAULT_CCTV_DELAY,
        incident_delay: float = _DEFAULT_INCIDENT_DELAY,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        max_per_cycle: int = _DEFAULT_MAX_PER_CYCLE,
        max_cctvs_per_incident: int = 2,
        output_dir: str | Path | None = None,
    ):
        self.target_types = target_types
        self.cctv_delay = cctv_delay
        self.incident_delay = incident_delay
        self.poll_interval = poll_interval
        self.max_per_cycle = max_per_cycle
        self.max_cctvs_per_incident = max_cctvs_per_incident
        self.output_dir = Path(output_dir) if output_dir else None

        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._page: Page | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._cycle_captured = 0
        self._stats = {
            "cycles": 0,
            "incidents_scanned": 0,
            "cctvs_attempted": 0,
            "cctvs_captured": 0,
            "total_frames": 0,
        }

    # ── Public API ──────────────────────────────────────────────────

    def start(self) -> bool:
        """브라우저 시작 + 돌발정보 페이지 로드."""
        try:
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-gpu",
                    "--autoplay-policy=no-user-gesture-required",
                ],
            )
            self._page = self._browser.new_page()
            self._page.set_viewport_size({"width": 1920, "height": 1080})
            logger.info("headless Chrome 시작")
        except Exception as e:
            logger.error("브라우저 시작 실패: %s", e)
            return False

        return self._navigate_to_outbreak()

    def stop(self):
        """브라우저 종료."""
        self._stop_event.set()
        if self._browser:
            try:
                self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._pw:
            try:
                self._pw.stop()
            except Exception:
                pass
            self._pw = None
        logger.info("OutbreakCCTVCapture 종료 | %s", self._stats)

    def scan_incidents(self) -> list[OutbreakIncident]:
        """돌발정보 목록 스캔 → target_types 필터링."""
        if not self._page:
            return []

        try:
            raw = self._page.evaluate(_JS_SCAN_INCIDENTS)
        except Exception as e:
            logger.error("돌발정보 스캔 JS 오류: %s", e)
            return []

        if not raw or raw.get("error"):
            logger.warning("돌발정보 스캔 실패: %s", raw)
            self._discover_dom()
            return []

        incidents = []
        for item in raw.get("items", []):
            event_type = item.get("type", "")
            if not any(t in event_type for t in self.target_types):
                continue
            incidents.append(OutbreakIncident(
                dom_index=item["index"],
                event_type=event_type,
                road_name=item.get("road", ""),
                direction="",
                message=f"{item.get('road', '')} {event_type} {item.get('date', '')}",
                incident_time=item.get("date", ""),
                raw_text=item.get("text", ""),
            ))

        with self._lock:
            self._stats["incidents_scanned"] += len(incidents)

        logger.info(
            "돌발정보 스캔: 전체 %d건 → 대상(%s) %d건",
            raw.get("total", 0), "/".join(self.target_types), len(incidents),
        )
        return incidents

    def capture_incident_cctvs(
        self, incident: OutbreakIncident,
    ) -> list[CapturedFrame]:
        """사고 클릭 → 인근 CCTV 순회 → 프레임 캡처."""
        if not self._page:
            return []

        click_ok = self._click_incident(incident.dom_index)
        if not click_ok:
            logger.warning(
                "사고 클릭 실패: [%d] %s",
                incident.dom_index, incident.message[:50],
            )
            return []

        self._page.wait_for_timeout(2000)

        cctvs = self._get_nearby_cctvs()
        if not cctvs:
            logger.info("인근 CCTV 없음: %s", incident.message[:50])
            return []

        incident.cctvs = cctvs
        logger.info(
            "[%s] %s — CCTV %d대",
            incident.event_type,
            incident.road_name or incident.message[:30],
            len(cctvs),
        )

        frames = []
        limit = min(len(cctvs), self.max_cctvs_per_incident)
        for i, cctv in enumerate(cctvs[:limit]):
            if self._stop_event.is_set():
                break
            if self._cycle_captured >= self.max_per_cycle:
                logger.info("사이클 상한 도달 (%d)", self.max_per_cycle)
                break

            with self._lock:
                self._stats["cctvs_attempted"] += 1

            frame = self._click_cctv_and_capture(cctv, incident)
            if frame:
                frames.append(frame)
                self._cycle_captured += 1
                with self._lock:
                    self._stats["cctvs_captured"] += 1
                    self._stats["total_frames"] += 1
                if self.output_dir:
                    self._save_frame(frame)

            if i < limit - 1 and not self._stop_event.is_set():
                time.sleep(self.cctv_delay)

        return frames

    def run_cycle(self) -> list[CapturedFrame]:
        """1회 수집 사이클: 스캔 → 필터 → 순회 캡처."""
        with self._lock:
            self._stats["cycles"] += 1
        self._cycle_captured = 0

        incidents = self.scan_incidents()
        all_frames = []

        for i, inc in enumerate(incidents):
            if self._stop_event.is_set():
                break
            if self._cycle_captured >= self.max_per_cycle:
                break

            frames = self.capture_incident_cctvs(inc)
            all_frames.extend(frames)

            if i < len(incidents) - 1 and not self._stop_event.is_set():
                time.sleep(self.incident_delay)

        logger.info(
            "사이클 완료: 사고 %d건 → 프레임 %d장",
            len(incidents), len(all_frames),
        )
        return all_frames

    def run_loop(
        self,
        on_frame: Callable[[CapturedFrame], None] | None = None,
        stop_event: threading.Event | None = None,
    ):
        """연속 수집 루프."""
        if stop_event:
            self._stop_event = stop_event

        logger.info(
            "수집 루프 시작 | 대상=%s CCTV간격=%.0fs 폴링=%ds",
            "/".join(self.target_types), self.cctv_delay, int(self.poll_interval),
        )

        while not self._stop_event.is_set():
            try:
                frames = self.run_cycle()
                if on_frame:
                    for f in frames:
                        on_frame(f)
            except Exception as e:
                logger.error("수집 사이클 오류: %s", e)
                try:
                    self._navigate_to_outbreak()
                except Exception:
                    pass

            if not self._stop_event.is_set():
                self._stop_event.wait(self.poll_interval)

    def capture_dwelling(
        self,
        cctv: dict,
        incident: OutbreakIncident,
        dwell_sec: float = 30.0,
        fps: float = 1.0,
        on_frame: Callable[[CapturedFrame], None] | None = None,
        extend_event: threading.Event | None = None,
        extend_sec: float = 60.0,
    ) -> list[CapturedFrame]:
        """CCTV 클릭 후 dwell_sec 동안 연속 캡처, 매 프레임마다 on_frame 호출.

        extend_event.set() 수신 시 체류 시간 extend_sec 추가.
        """
        if not self._page:
            return []

        page = self._page
        cctv_name = cctv.get("name", "")
        distance = cctv.get("distance", "")

        prev_src = page.evaluate(
            "document.querySelector('video')?.currentSrc || ''"
        )
        try:
            click_result = page.evaluate(_JS_CLICK_CCTV, cctv.get("index", 0))
        except Exception as e:
            logger.debug("CCTV 클릭 오류: %s (%s)", cctv_name, e)
            return []
        if not click_result or not click_result.get("ok"):
            logger.debug("CCTV 클릭 실패: %s", cctv_name)
            return []

        for attempt in range(_VIDEO_LOAD_TIMEOUT):
            time.sleep(1)
            try:
                state = page.evaluate(_JS_CHECK_VIDEO)
            except Exception:
                continue
            if state and state.get("ready"):
                cur_src = state.get("src", "")
                if cur_src != prev_src or attempt >= 3:
                    break
        else:
            logger.debug("영상 로드 타임아웃 (체류): %s", cctv_name)
            return []

        logger.info(
            "체류 시작: %s (%ds, %.1ffps)", cctv_name[:40], int(dwell_sec), fps,
        )

        interval = 1.0 / fps
        deadline = time.monotonic() + dwell_sec
        frames: list[CapturedFrame] = []

        while time.monotonic() < deadline and not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                capture = page.evaluate(_JS_CAPTURE_FRAME)
            except Exception:
                capture = None

            if capture and capture.get("dataUrl"):
                frame_bgr, w, h = _b64_to_bgr(capture["dataUrl"])
                if frame_bgr is not None:
                    cf = CapturedFrame(
                        incident_type=incident.event_type,
                        road_name=incident.road_name,
                        cctv_name=cctv_name,
                        distance=distance,
                        frame=frame_bgr,
                        timestamp=time.time(),
                        width=w,
                        height=h,
                        jpeg_size=capture.get("size", 0),
                    )
                    frames.append(cf)
                    with self._lock:
                        self._stats["total_frames"] += 1
                    if on_frame:
                        on_frame(cf)

            if extend_event and extend_event.is_set():
                extend_event.clear()
                deadline = time.monotonic() + extend_sec
                logger.info(
                    "체류 연장: %s (+%ds)", cctv_name[:30], int(extend_sec),
                )

            elapsed = time.monotonic() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info(
            "체류 종료: %s — %d프레임 (%ds)",
            cctv_name[:40], len(frames), int(dwell_sec),
        )
        return frames

    def get_stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    # ── 내부 메서드 ──────────────────────────────────────────────────

    def _navigate_to_outbreak(self) -> bool:
        if not self._page:
            return False
        try:
            logger.info("돌발정보 페이지 로드 중...")
            self._page.goto(
                _OUTBREAK_URL, wait_until="domcontentloaded", timeout=30000,
            )
            self._page.wait_for_timeout(int(_PAGE_SETTLE_SEC * 1000))
            self._page.evaluate("window.openFullscreen = function() {};")
            logger.info("페이지 로드 완료: %s", self._page.title())
            return True
        except Exception as e:
            logger.error("페이지 로드 실패: %s", e)
            return False

    def _click_incident(self, dom_index: int) -> bool:
        try:
            result = self._page.evaluate(_JS_CLICK_INCIDENT, dom_index)
            return bool(result and result.get("ok"))
        except Exception as e:
            logger.debug("사고 클릭 JS 오류: %s", e)
            return False

    def _get_nearby_cctvs(self) -> list[dict]:
        try:
            result = self._page.evaluate(_JS_GET_NEARBY_CCTVS)
            if not result or result.get("error"):
                return []
            return result.get("cctvs", [])
        except Exception as e:
            logger.debug("CCTV 목록 JS 오류: %s", e)
            return []

    def _click_cctv_and_capture(
        self, cctv: dict, incident: OutbreakIncident,
    ) -> CapturedFrame | None:
        page = self._page
        cctv_name = cctv.get("name", "")
        distance = cctv.get("distance", "")

        prev_src = page.evaluate(
            "document.querySelector('video')?.currentSrc || ''"
        )

        try:
            click_result = page.evaluate(_JS_CLICK_CCTV, cctv.get("index", 0))
        except Exception as e:
            logger.debug("CCTV 클릭 JS 오류: %s (%s)", cctv_name, e)
            return None

        if not click_result or not click_result.get("ok"):
            logger.debug("CCTV 클릭 실패: %s", cctv_name)
            return None

        for attempt in range(_VIDEO_LOAD_TIMEOUT):
            time.sleep(1)
            try:
                state = page.evaluate(_JS_CHECK_VIDEO)
            except Exception:
                continue
            if state and state.get("ready"):
                cur_src = state.get("src", "")
                if cur_src != prev_src or attempt >= 3:
                    break
        else:
            logger.debug("영상 로드 타임아웃: %s", cctv_name)
            return None

        try:
            capture = page.evaluate(_JS_CAPTURE_FRAME)
        except Exception as e:
            logger.debug("프레임 캡처 JS 오류: %s (%s)", cctv_name, e)
            return None

        if not capture or not capture.get("dataUrl"):
            logger.debug("프레임 캡처 실패: %s", cctv_name)
            return None

        frame, w, h = _b64_to_bgr(capture["dataUrl"])
        if frame is None:
            return None

        logger.info(
            "  캡처: %s %dx%d (%s)",
            cctv_name[:40], w, h, distance,
        )
        return CapturedFrame(
            incident_type=incident.event_type,
            road_name=incident.road_name,
            cctv_name=cctv_name,
            distance=distance,
            frame=frame,
            timestamp=time.time(),
            width=w,
            height=h,
            jpeg_size=capture.get("size", 0),
        )

    def _save_frame(self, frame: CapturedFrame):
        if not self.output_dir:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.fromtimestamp(frame.timestamp).strftime("%Y%m%d_%H%M%S")
        safe = frame.cctv_name.replace("/", "_").replace(" ", "_")[:50]
        fname = f"{ts}_{frame.incident_type}_{safe}.jpg"
        path = self.output_dir / fname
        Image.fromarray(frame.frame[:, :, ::-1]).save(
            str(path), "JPEG", quality=85,
        )

    @staticmethod
    def _classify_type(text: str) -> str:
        for t in ("사고", "고장", "공사", "기상", "재난"):
            if t in text:
                return t
        return "기타"

    @staticmethod
    def _extract_road(text: str) -> str:
        import re
        m = re.search(r'\[([^\]]+)\]', text)
        if m:
            return m.group(1)
        for kw in ("고속도로", "국도", "선"):
            idx = text.find(kw)
            if idx >= 0:
                start = max(0, idx - 10)
                return text[start:idx + len(kw)].strip()
        return ""

    def _discover_dom(self):
        if not self._page:
            return
        try:
            info = self._page.evaluate(_JS_DISCOVER_DOM)
            logger.info(
                "DOM 구조:\n%s",
                json.dumps(info, ensure_ascii=False, indent=2)[:2000],
            )
        except Exception as e:
            logger.error("DOM 탐색 오류: %s", e)


# ═══════════════════════════════════════════════════════════════════════
# 유틸리티
# ═══════════════════════════════════════════════════════════════════════

def _b64_to_bgr(data_url: str) -> tuple[np.ndarray | None, int, int]:
    """base64 data URL → BGR numpy 배열."""
    try:
        _, b64 = data_url.split(",", 1)
        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes))
        arr = np.array(img)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1].copy()
        return arr, img.width, img.height
    except Exception as e:
        logger.debug("이미지 디코딩 오류: %s", e)
        return None, 0, 0


# ═══════════════════════════════════════════════════════════════════════
# 테스트
# ═══════════════════════════════════════════════════════════════════════

def _test_snapshot(max_incidents: int = 3, max_per_cycle: int = 10):
    """돌발정보 페이지 1사이클 스냅샷 캡처 테스트."""
    output_dir = Path(
        "/workspace/prj_cctv/사고분석_설계/output/outbreak_captures"
    )
    cap = OutbreakCCTVCapture(
        target_types=("사고", "고장"),
        cctv_delay=5.0,
        incident_delay=3.0,
        max_per_cycle=max_per_cycle,
        output_dir=output_dir,
    )

    if not cap.start():
        print("브라우저 시작 실패")
        return

    try:
        incidents = cap.scan_incidents()
        print(f"\n돌발정보 스캔: {len(incidents)}건 (사고/고장)")
        for inc in incidents[:5]:
            print(f"  [{inc.event_type}] {inc.message[:60]}")

        if not incidents:
            print("대상 사고 없음 — DOM 구조 탐색 결과는 로그 참조")
            return

        all_frames = []
        for i, inc in enumerate(incidents[:max_incidents]):
            print(f"\n>>> [{inc.event_type}] {inc.message[:60]}")
            frames = cap.capture_incident_cctvs(inc)
            all_frames.extend(frames)
            for f in frames:
                print(
                    f"  캡처: {f.cctv_name} "
                    f"({f.width}x{f.height}, {f.jpeg_size:,}B)"
                )
            if i < min(max_incidents - 1, len(incidents) - 1):
                time.sleep(cap.incident_delay)

        stats = cap.get_stats()
        print(f"\n=== 테스트 결과 ===")
        print(f"  스캔 사고: {stats['incidents_scanned']}건")
        print(f"  CCTV 시도: {stats['cctvs_attempted']}")
        print(f"  CCTV 성공: {len(all_frames)}")
        if output_dir.exists():
            saved = list(output_dir.glob("*.jpg"))
            print(f"  저장 이미지: {len(saved)}장 → {output_dir}")
    finally:
        cap.stop()


def _test_dwelling(
    dwell_sec: float = 30.0,
    fps: float = 1.0,
    max_incidents: int = 2,
    output: str | None = None,
    loop: bool = False,
):
    """체류 캡처 테스트: CCTV당 dwell_sec 동안 fps로 연속 캡처."""
    import re

    output_dir = Path(
        output or "/workspace/prj_cctv/사고분석_설계/output/outbreak_dwelling"
    )
    cap = OutbreakCCTVCapture(
        target_types=("사고", "고장"),
        cctv_delay=5.0,
        incident_delay=3.0,
        max_cctvs_per_incident=2,
    )

    if not cap.start():
        print("브라우저 시작 실패")
        return

    def _make_cctv_id(incident: OutbreakIncident, cctv: dict) -> str:
        raw = cctv.get("name", "unknown")
        sanitized = re.sub(r'[\[\]\s/]', '_', raw).strip('_')[:40]
        return f"OB_{sanitized}"

    try:
        while True:
            incidents = cap.scan_incidents()
            print(f"\n돌발정보 스캔: {len(incidents)}건 (사고/고장)")

            if not incidents:
                print("대상 사고 없음")
                if not loop:
                    return
                print(f"다음 스캔까지 {cap.poll_interval}초 대기...")
                time.sleep(cap.poll_interval)
                continue

            total_frames = 0
            for i, inc in enumerate(incidents[:max_incidents]):
                if cap._stop_event.is_set():
                    break
                print(f"\n>>> [{inc.event_type}] {inc.message[:60]}")

                click_ok = cap._click_incident(inc.dom_index)
                if not click_ok:
                    print("  사고 클릭 실패")
                    continue
                cap._page.wait_for_timeout(2000)

                cctvs = cap._get_nearby_cctvs()
                if not cctvs:
                    print("  인근 CCTV 없음")
                    continue

                for cctv in cctvs[:cap.max_cctvs_per_incident]:
                    cctv_id = _make_cctv_id(inc, cctv)
                    cctv_dir = output_dir / cctv_id
                    cctv_dir.mkdir(parents=True, exist_ok=True)

                    frame_idx = [0]

                    def save_frame(cf: CapturedFrame, d=cctv_dir, idx=frame_idx):
                        path = d / f"{idx[0]:04d}.jpg"
                        Image.fromarray(cf.frame[:, :, ::-1]).save(
                            str(path), "JPEG", quality=85,
                        )
                        idx[0] += 1

                    frames = cap.capture_dwelling(
                        cctv=cctv,
                        incident=inc,
                        dwell_sec=dwell_sec,
                        fps=fps,
                        on_frame=save_frame,
                    )
                    total_frames += len(frames)
                    print(
                        f"  {cctv.get('name', '')[:40]} "
                        f"→ {len(frames)}프레임 → {cctv_dir}",
                    )
                    time.sleep(cap.cctv_delay)

                if i < min(max_incidents - 1, len(incidents) - 1):
                    time.sleep(cap.incident_delay)

            print(f"\n=== 체류 캡처 결과: {total_frames}프레임 ===")
            if not loop:
                break
            print(f"다음 사이클까지 {cap.poll_interval}초 대기...")
            time.sleep(cap.poll_interval)
    finally:
        cap.stop()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="ITS 돌발정보 CCTV 프레임 캡처",
    )
    parser.add_argument(
        "--dwell", type=float, default=0,
        help="체류 시간 (초, 0=스냅샷 모드)",
    )
    parser.add_argument("--fps", type=float, default=1.0, help="캡처 fps")
    parser.add_argument(
        "--max-incidents", type=int, default=3, help="최대 사고 수",
    )
    parser.add_argument("--loop", action="store_true", help="연속 루프")
    parser.add_argument("--output", type=str, default=None, help="출력 디렉토리")
    args = parser.parse_args()

    if args.dwell > 0:
        _test_dwelling(
            dwell_sec=args.dwell,
            fps=args.fps,
            max_incidents=args.max_incidents,
            output=args.output,
            loop=args.loop,
        )
    else:
        _test_snapshot(max_incidents=args.max_incidents)
