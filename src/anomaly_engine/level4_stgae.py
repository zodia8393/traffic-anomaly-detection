"""Level 4: STGAE (Spatio-Temporal Graph Auto-Encoder) 학습 및 평가.

subprocess로 external/stgae/train_aihub.py를 실행.
src/ 패키지 충돌 회피를 위해 직접 import하지 않음.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

STGAE_ROOT = Path(__file__).resolve().parent.parent.parent / "external" / "stgae"


def train_stgae(
    data_dir: str | Path = "/DATA/aihub_71566/stgae_format",
    output_dir: str | Path | None = None,
    config: str | None = None,
    skip_eval: bool = False,
) -> int:
    """STGAE 학습을 subprocess로 실행."""
    cmd = ["python3", str(STGAE_ROOT / "train_aihub.py"), "--data_dir", str(data_dir)]
    if output_dir:
        cmd += ["--output_dir", str(output_dir)]
    if config:
        cmd += ["--config", str(config)]
    if skip_eval:
        cmd.append("--skip_eval")

    result = subprocess.run(cmd, cwd=str(STGAE_ROOT))
    return result.returncode


if __name__ == "__main__":
    train_stgae(
        output_dir=Path(__file__).resolve().parent.parent.parent / "models" / "stgae_aihub",
    )
