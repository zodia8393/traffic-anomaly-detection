"""llama.cpp 서버 관리 — 시작/중지/상태 확인."""

import logging
import shutil
import subprocess
import time

import requests

logger = logging.getLogger(__name__)


class MLLMServer:
    """llama-server 프로세스 래퍼."""

    def __init__(
        self,
        model_path: str,
        port: int = 8080,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
    ) -> None:
        self.model_path = model_path
        self.port = port
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._process: subprocess.Popen | None = None
        self._base_url = f"http://localhost:{port}"

    # ── 서버 시작 ────────────────────────────────────────────────────

    def start(self) -> bool:
        """llama-server 프로세스 시작. 이미 실행 중이면 False."""
        if self._process is not None and self._process.poll() is None:
            logger.warning("서버가 이미 실행 중입니다 (PID %d)", self._process.pid)
            return False

        exe = shutil.which("llama-server")
        if exe is None:
            logger.error("llama-server 실행 파일을 찾을 수 없습니다")
            return False

        cmd = [
            exe,
            "--model", self.model_path,
            "--port", str(self.port),
            "--ctx-size", str(self.n_ctx),
            "--n-gpu-layers", str(self.n_gpu_layers),
        ]

        logger.info("llama-server 시작: %s", " ".join(cmd))
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as e:
            logger.error("llama-server 시작 실패: %s", e)
            return False

        logger.info("llama-server 시작됨 (PID %d)", self._process.pid)
        return True

    # ── 서버 중지 ────────────────────────────────────────────────────

    def stop(self) -> None:
        """서버 프로세스 종료."""
        if self._process is None:
            return

        if self._process.poll() is None:
            logger.info("llama-server 종료 중 (PID %d)", self._process.pid)
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("SIGTERM 무응답, SIGKILL 전송")
                self._process.kill()
                self._process.wait(timeout=5)

        self._process = None
        logger.info("llama-server 종료 완료")

    # ── 상태 확인 ────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        """health 엔드포인트로 서버 준비 상태 확인."""
        try:
            resp = requests.get(f"{self._base_url}/health", timeout=3)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def wait_ready(self, timeout: int = 120) -> bool:
        """서버가 준비될 때까지 폴링 대기.

        Args:
            timeout: 최대 대기 시간(초).

        Returns:
            준비 완료 시 True, 타임아웃 시 False.
        """
        logger.info("llama-server 준비 대기 (최대 %d초)", timeout)
        deadline = time.monotonic() + timeout
        interval = 1.0

        while time.monotonic() < deadline:
            # 프로세스가 비정상 종료된 경우
            if self._process is not None and self._process.poll() is not None:
                rc = self._process.returncode
                logger.error("llama-server 비정상 종료 (rc=%d)", rc)
                return False

            if self.is_ready():
                logger.info("llama-server 준비 완료")
                return True

            time.sleep(interval)
            # 점진적 간격 증가 (최대 5초)
            interval = min(interval * 1.5, 5.0)

        logger.error("llama-server 준비 대기 타임아웃 (%d초)", timeout)
        return False

    # ── 컨텍스트 매니저 ──────────────────────────────────────────────

    def __enter__(self) -> "MLLMServer":
        self.start()
        self.wait_ready()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @property
    def pid(self) -> int | None:
        """현재 프로세스 PID (실행 중이 아니면 None)."""
        if self._process is not None and self._process.poll() is None:
            return self._process.pid
        return None
