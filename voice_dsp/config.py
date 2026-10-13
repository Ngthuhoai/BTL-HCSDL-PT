"""Cấu hình phân tích âm thanh; mọi hàm dùng chung lưới STFT 16 kHz."""
from dataclasses import dataclass


@dataclass(frozen=True)
class DSPConfig:
    sample_rate: int = 16000
    n_fft: int = 1024       # 64 ms; độ phân giải tần số 15.625 Hz
    hop_length: int = 160   # 10 ms
    n_mfcc: int = 13
    n_mels: int = 40
    min_duration_s: float = 0.25
    silence_floor_dbfs: float = -60.0
    silence_relative_db: float = 35.0
    f0_min_hz: float = 65.0
    f0_max_hz: float = 500.0


CONFIG = DSPConfig()
