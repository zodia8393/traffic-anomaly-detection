"""363대 CCTV HLS 스트림 수명주기 관리자.

3-Tier 아키텍처:
  Tier 1 (프리필터): 전체 363대. 경량 프리필터만 적용 (MOG2/프레임차분).
         이상 감지 시 Tier 3으로 승격.
  Tier 2 (핫스팟):   사고다발구간 50대. 상시 정밀 분석 (YOLO+트리거).
  Tier 3 (이상후보): 동적 승격. Tier 1에서 이상 감지된 카메라.
         PREFILTER_TIER3_HOLD_SEC 후 원래 Tier로 강등.

StreamWorker: FrameSampler를 감싸는 얇은 래퍼. tier 정보를 콜백에 전달.
StreamManager: 전체 스트림 시작/정지, 승격/강등, URL 갱신, 건강 모니터링.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# 부모 디렉토리(src/)를 path에 추가하여 config_new import 가능하게 함
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from track3_cctv_stream import CCTVInfo, ITSCCTVClient
from run_realtime import FrameSampler
from config_new import (
    STREAM_MAX_FAILURES,
    STREAM_RECONNECT_DELAY_SEC,
    STREAM_REPROBE_SEC,
    STREAM_SAMPLE_FPS,
    STREAM_URL_REFRESH_SEC,
    TIER2_HOTSPOT_COUNT,
    TIER3_MAX_CONCURRENT,
    PREFILTER_TIER3_HOLD_SEC,
)

logger = logging.getLogger(__name__)


@dataclass
class WorkerStats:
    """개별 워커 통계."""
    frames: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    zero_frame_sessions: int = 0
    reconnects: int = 0
    last_frame_at: float = 0.0
    started_at: float = 0.0
    disabled: bool = False
    disabled_reason: str = ""
    disabled_at: float = 0.0


class StreamWorker:
    """단일 CCTV 스트림 워커.

    FrameSampler를 내부에 갖고, tier에 따라 동일한 on_frame 콜백을 호출한다.
    콜백에 tier 정보를 함께 전달하여 상위 파이프라인에서 분기 처리.
    """

    def __init__(self, cctv: CCTVInfo, tier: int,
                 on_frame: Callable, stop_event: threading.Event):
        """
        Args:
            cctv: CCTV 정보 (cctv_id, name, stream_url 등).
            tier: 1(프리필터) / 2(핫스팟 정밀) / 3(이상 후보 정밀).
            on_frame: 프레임 도착 시 호출할 콜백 (frame, cctv_id, tier).
            stop_event: 전역 종료 시그널.
        """
        self.cctv = cctv
        self.tier = tier
        self._on_frame = on_frame
        self._stop_event = stop_event
        self._sampler: FrameSampler | None = None
        self._thread: threading.Thread | None = None
        self.stats = WorkerStats()
        self._logger = logging.getLogger(
            f"stream.{cctv.cctv_id[:20]}"
        )

    def start(self):
        """백그라운드 스레드에서 스트림 시작."""
        if self._thread and self._thread.is_alive():
            return
        self.stats.started_at = time.monotonic()
        self.stats.disabled = False
        self.stats.consecutive_failures = 0
        self.stats.zero_frame_sessions = 0
        self._thread = threading.Thread(
            target=self._run,
            name=f"stream-{self.cctv.cctv_id[:20]}",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        """스트림 정지."""
        if self._sampler:
            self._sampler.stop()
            self._sampler = None

    @property
    def is_alive(self) -> bool:
        """스트림 활성 여부."""
        return (
            self._thread is not None
            and self._thread.is_alive()
            and not self.stats.disabled
        )

    def update_url(self, new_url: str):
        """스트림 URL 갱신 (다음 재연결 시 반영)."""
        self.cctv.stream_url = new_url

    def _run(self):
        """스트림 읽기 루프 (스레드에서 실행)."""
        while not self._stop_event.is_set() and not self.stats.disabled:
            self._sampler = FrameSampler(
                stream_url=self.cctv.stream_url,
                sample_fps=STREAM_SAMPLE_FPS,
                width=640,
                height=480,
            )
            if not self._sampler.start():
                self._handle_failure("FrameSampler 시작 실패")
                if self.stats.disabled:
                    break
                self._wait_reconnect()
                continue

            self.stats.consecutive_failures = 0
            self._logger.info(
                "스트림 연결 [T%d]: %s", self.tier, self.cctv.name
            )

            # 프레임 읽기 루프
            session_frames = 0
            while not self._stop_event.is_set() and not self.stats.disabled:
                ok, frame = self._sampler.read_frame()
                if not ok:
                    self._handle_failure("프레임 읽기 실패")
                    self._sampler.stop()
                    self._sampler = None
                    break

                session_frames += 1
                self.stats.frames += 1
                self.stats.last_frame_at = time.monotonic()
                self.stats.consecutive_failures = 0
                self.stats.zero_frame_sessions = 0

                try:
                    self._on_frame(frame, self.cctv.cctv_id, self.tier)
                except Exception as e:
                    self._logger.error("콜백 오류: %s", e)

            # 0프레임 세션 처리: 연결 성공했으나 프레임 0건
            if session_frames == 0 and not self._stop_event.is_set():
                self.stats.zero_frame_sessions += 1
                if self.stats.zero_frame_sessions >= STREAM_MAX_FAILURES:
                    self.stats.disabled = True
                    self.stats.disabled_at = time.monotonic()
                    self.stats.disabled_reason = (
                        f"연속 {self.stats.zero_frame_sessions}회 0프레임"
                    )
                    self._logger.warning(
                        "스트림 비활성화 (0프레임): %s — %s",
                        self.cctv.name, self.stats.disabled_reason,
                    )
                    break
                backoff = min(
                    STREAM_RECONNECT_DELAY_SEC
                    * (2 ** (self.stats.zero_frame_sessions - 1)),
                    300,
                )
                n = self.stats.zero_frame_sessions
                log_fn = (
                    self._logger.warning
                    if n in (1, STREAM_MAX_FAILURES // 2, STREAM_MAX_FAILURES - 1)
                    else self._logger.debug
                )
                log_fn(
                    "0프레임 세션 %d/%d: %s — %.0fs 후 재시도",
                    n, STREAM_MAX_FAILURES,
                    self.cctv.name, backoff,
                )
                self._stop_event.wait(backoff)
                continue

            if self.stats.disabled or self._stop_event.is_set():
                break
            self._wait_reconnect()

        # 정리
        if self._sampler:
            self._sampler.stop()
            self._sampler = None
        self._logger.info("스트림 종료: %s (총 %d프레임)",
                          self.cctv.name, self.stats.frames)

    def _handle_failure(self, reason: str):
        """실패 처리. 연속 실패 초과 시 비활성화."""
        self.stats.failures += 1
        self.stats.consecutive_failures += 1
        if self.stats.consecutive_failures >= STREAM_MAX_FAILURES:
            self.stats.disabled = True
            self.stats.disabled_at = time.monotonic()
            self.stats.disabled_reason = (
                f"{reason} (연속 {self.stats.consecutive_failures}회)"
            )
            self._logger.warning(
                "스트림 비활성화: %s — %s",
                self.cctv.name, self.stats.disabled_reason,
            )
        else:
            self._logger.debug(
                "스트림 실패 %d/%d: %s — %s",
                self.stats.consecutive_failures, STREAM_MAX_FAILURES,
                self.cctv.name, reason,
            )

    def _wait_reconnect(self):
        """재연결 대기."""
        self.stats.reconnects += 1
        self._stop_event.wait(STREAM_RECONNECT_DELAY_SEC)


class StreamManager:
    """363대 CCTV 스트림 수명주기 관리.

    - 전체 스트림 시작/정지
    - Tier 승격/강등 (1->3, 3->1)
    - URL 주기적 갱신
    - 건강 상태 모니터링
    """

    def __init__(self, cctv_client: ITSCCTVClient | None = None,
                 stop_event: threading.Event | None = None):
        self._client = cctv_client or ITSCCTVClient()
        self._stop_event = stop_event or threading.Event()
        self.streams: dict[str, StreamWorker] = {}
        self.tier_map: dict[str, int] = {}  # cctv_id -> tier
        self._hotspot_ids: set[str] = set()
        self._on_frame: Callable | None = None
        self._url_thread: threading.Thread | None = None
        self._health_thread: threading.Thread | None = None
        self._lock = threading.Lock()  # tier_map/streams 동시 접근 보호
        self._tier3_expiry: dict[str, float] = {}  # cctv_id -> 만료 시각
        self._all_cctvs: list[CCTVInfo] = []

    def start_all(self, on_frame: Callable,
                  hotspot_ids: set[str] | None = None,
                  max_cameras: int = 0) -> int:
        """모든 CCTV 스트림 시작.

        Args:
            on_frame: 프레임 콜백 (frame, cctv_id, tier).
            hotspot_ids: Tier 2로 지정할 CCTV ID 집합 (나머지는 Tier 1).
            max_cameras: 최대 카메라 수 (0=전체, 테스트용).

        Returns:
            시작된 스트림 수.
        """
        self._on_frame = on_frame
        self._hotspot_ids = hotspot_ids or set()

        # CCTV 목록 조회
        cctvs = self._client.list_cctvs()
        if not cctvs:
            logger.error("CCTV 목록 조회 실패 — 스트림 시작 불가")
            return 0

        if max_cameras > 0:
            cctvs = cctvs[:max_cameras]

        self._all_cctvs = cctvs
        logger.info("CCTV %d대 조회 완료, 스트림 시작 중...", len(cctvs))

        started = 0
        for cctv in cctvs:
            if not cctv.stream_url:
                continue

            # Tier 결정
            if cctv.cctv_id in self._hotspot_ids:
                tier = 2
            else:
                tier = 1

            worker = StreamWorker(
                cctv=cctv,
                tier=tier,
                on_frame=on_frame,
                stop_event=self._stop_event,
            )
            with self._lock:
                self.streams[cctv.cctv_id] = worker
                self.tier_map[cctv.cctv_id] = tier

            worker.start()
            started += 1

        tier1 = sum(1 for t in self.tier_map.values() if t == 1)
        tier2 = sum(1 for t in self.tier_map.values() if t == 2)
        logger.info(
            "스트림 %d대 시작 완료 (T1=%d, T2=%d)", started, tier1, tier2
        )

        # URL 갱신 백그라운드 스레드
        self._url_thread = threading.Thread(
            target=self._url_refresh_loop,
            name="url-refresh",
            daemon=True,
        )
        self._url_thread.start()

        # 건강 모니터링 백그라운드 스레드
        self._health_thread = threading.Thread(
            target=self._health_check_loop,
            name="health-check",
            daemon=True,
        )
        self._health_thread.start()

        return started

    def promote(self, cctv_id: str, to_tier: int = 3, hold_sec: float | None = None):
        """Tier 승격 (1->3 또는 2->3).

        Tier 3 동시 분석 상한(TIER3_MAX_CONCURRENT) 초과 시 거부.
        승격된 카메라는 hold_sec(기본 PREFILTER_TIER3_HOLD_SEC) 후 자동 강등.
        ITS 사고 편입처럼 더 길게 유지하려면 hold_sec를 명시(예: INCIDENT_HOLD_SEC).
        """
        with self._lock:
            current = self.tier_map.get(cctv_id)
            if current is None:
                logger.warning("승격 실패: %s 없음", cctv_id)
                return
            if current >= to_tier:
                return  # 이미 해당 티어 이상

            # Tier 3 상한 검사
            if to_tier == 3:
                tier3_count = sum(
                    1 for t in self.tier_map.values() if t == 3
                )
                if tier3_count >= TIER3_MAX_CONCURRENT:
                    logger.warning(
                        "Tier 3 상한 도달 (%d/%d) — %s 승격 거부",
                        tier3_count, TIER3_MAX_CONCURRENT, cctv_id,
                    )
                    return

            self.tier_map[cctv_id] = to_tier
            worker = self.streams.get(cctv_id)
            if worker:
                worker.tier = to_tier

            # 자동 강등 타이머 설정 (Tier 3만). hold_sec 미지정 시 프리필터 기본값.
            if to_tier == 3:
                hold = PREFILTER_TIER3_HOLD_SEC if hold_sec is None else hold_sec
                self._tier3_expiry[cctv_id] = time.monotonic() + hold

        logger.info("승격: %s T%d -> T%d", cctv_id, current, to_tier)

    def demote(self, cctv_id: str):
        """Tier 1로 강등 (핫스팟이면 Tier 2로)."""
        with self._lock:
            current = self.tier_map.get(cctv_id)
            if current is None:
                return

            if cctv_id in self._hotspot_ids:
                target = 2
            else:
                target = 1

            if current <= target:
                return

            self.tier_map[cctv_id] = target
            worker = self.streams.get(cctv_id)
            if worker:
                worker.tier = target

            self._tier3_expiry.pop(cctv_id, None)

        logger.info("강등: %s T%d -> T%d", cctv_id, current, target)

    def refresh_urls(self):
        """ITS API로 스트림 URL 일괄 갱신."""
        cctvs = self._client.list_cctvs()
        if not cctvs:
            logger.warning("URL 갱신 실패: CCTV 목록 조회 불가")
            return

        url_map = {c.cctv_id: c.stream_url for c in cctvs if c.stream_url}
        updated = 0

        with self._lock:
            for cctv_id, worker in self.streams.items():
                new_url = url_map.get(cctv_id)
                if new_url and new_url != worker.cctv.stream_url:
                    worker.update_url(new_url)
                    updated += 1

        logger.info("URL 갱신 완료: %d/%d대 업데이트", updated, len(url_map))

    def get_stats(self) -> dict:
        """Tier별 스트림 수, 활성/비활성 상태 등."""
        with self._lock:
            tier_counts = {1: 0, 2: 0, 3: 0}
            active = 0
            disabled = 0
            total_frames = 0
            zero_frame = 0

            for cctv_id, worker in self.streams.items():
                tier = self.tier_map.get(cctv_id, 1)
                tier_counts[tier] = tier_counts.get(tier, 0) + 1

                if worker.stats.disabled:
                    disabled += 1
                elif worker.is_alive:
                    active += 1

                total_frames += worker.stats.frames
                if worker.stats.zero_frame_sessions > 0:
                    zero_frame += 1

            return {
                "total": len(self.streams),
                "active": active,
                "disabled": disabled,
                "zero_frame": zero_frame,
                "tier_counts": dict(tier_counts),
                "total_frames": total_frames,
                "tier3_pending_demote": len(self._tier3_expiry),
            }

    def stop_all(self):
        """전체 스트림 정지."""
        logger.info("전체 스트림 정지 중... (%d대)", len(self.streams))
        self._stop_event.set()

        for worker in self.streams.values():
            worker.stop()

        # 스레드 종료 대기
        for worker in self.streams.values():
            if worker._thread and worker._thread.is_alive():
                worker._thread.join(timeout=5)

        stats = self.get_stats()
        logger.info(
            "전체 스트림 정지 완료 (총 %d프레임 처리)",
            stats["total_frames"],
        )

    # ── 내부 백그라운드 루프 ──────────────────────────────────────────

    def _url_refresh_loop(self):
        """STREAM_URL_REFRESH_SEC마다 URL 갱신."""
        while not self._stop_event.is_set():
            if self._stop_event.wait(STREAM_URL_REFRESH_SEC):
                break
            try:
                self.refresh_urls()
            except Exception as e:
                logger.error("URL 갱신 루프 오류: %s", e)

    def _health_check_loop(self):
        """30초마다 건강 검사 + Tier 3 자동 강등."""
        while not self._stop_event.is_set():
            if self._stop_event.wait(30):
                break
            try:
                self._expire_tier3()
                self._retry_disabled()
            except Exception as e:
                logger.error("건강 검사 루프 오류: %s", e)

    def _expire_tier3(self):
        """Tier 3 유지 시간 초과 카메라 자동 강등."""
        now = time.monotonic()
        expired: list[str] = []

        with self._lock:
            for cctv_id, expiry in list(self._tier3_expiry.items()):
                if now >= expiry:
                    expired.append(cctv_id)

        for cctv_id in expired:
            self.demote(cctv_id)

    def _retry_disabled(self):
        """비활성 카메라 재탐지 (STREAM_REPROBE_SEC마다)."""
        now = time.monotonic()
        to_retry = []
        with self._lock:
            for w in self.streams.values():
                if w.stats.disabled and (now - w.stats.disabled_at) >= STREAM_REPROBE_SEC:
                    to_retry.append(w)

        if not to_retry:
            return

        self.refresh_urls()
        logger.info("비활성 카메라 재탐지: %d대 재시도", len(to_retry))

        for worker in to_retry:
            worker.start()
