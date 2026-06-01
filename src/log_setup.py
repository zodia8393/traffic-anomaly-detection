"""구조화 로깅 + 로그 로테이션 헬퍼.

기존 모듈은 logging.StreamHandler만 사용해 (1) 단일 로그가 무한 증가(130MB+),
(2) 구조화(컨텍스트) 부재로 RCA가 어렵다. 이 헬퍼로 RotatingFileHandler +
선택적 JSON 포맷을 일괄 적용한다.

사용:
  from log_setup import setup_logging
  setup_logging("record_multi", logfile="/DATA/cctv_recording/record.log", json_fmt=True)
  logger.info("녹화 시작", extra={"camera_id": cid, "route": route})
"""
from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_MAX_BYTES = 100 * 1024 * 1024  # 100MB
_BACKUPS = 5                     # 총 ~600MB 상한


class JsonFormatter(logging.Formatter):
    """로그를 JSON 한 줄로 직렬화 (extra 컨텍스트 포함)."""

    _RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
        "message", "asctime", "taskName"}

    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # extra= 로 넘어온 사용자 필드 병합
        for k, v in record.__dict__.items():
            if k not in self._RESERVED and not k.startswith("_"):
                base[k] = v
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False, default=str)


def setup_logging(name: str | None = None, *, logfile: str | None = None,
                  level: int = logging.INFO, json_fmt: bool = False,
                  console: bool = True) -> logging.Logger:
    """루트(또는 named) 로거에 로테이션 파일 + 콘솔 핸들러 설치.

    Args:
        name: None이면 루트 로거.
        logfile: 지정 시 RotatingFileHandler 추가.
        json_fmt: True면 파일 핸들러에 JSON 포맷.
        console: True면 StreamHandler 유지.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    text_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if console:
        sh = logging.StreamHandler()
        sh.setFormatter(text_fmt)
        logger.addHandler(sh)

    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(logfile, maxBytes=_MAX_BYTES,
                                 backupCount=_BACKUPS, encoding="utf-8")
        fh.setFormatter(JsonFormatter() if json_fmt else text_fmt)
        logger.addHandler(fh)

    return logger
