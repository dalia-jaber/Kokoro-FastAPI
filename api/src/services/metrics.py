from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class TTSMetrics:
    """Metrics collected during TTS generation."""
    request_id: str
    timestamp: float
    ttfb: float
    duration: float
    audio_duration: float
    characters_count: int


