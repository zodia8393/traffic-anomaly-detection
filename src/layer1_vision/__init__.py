"""Layer 1: 영상 분석 — 탐지·추적·속도·트리거·키프레임."""
from .speed_estimator import SpeedEstimator
from .ttc_calculator import compute_ttc, compute_all_ttc
from .trigger_detector import TriggerDetector, TriggerEvent
from .keyframe_selector import select_keyframes
from .vision_pipeline import VisionPipeline
