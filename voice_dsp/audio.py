"""Đọc, kiểm tra và chuẩn bị tín hiệu mono."""
from pathlib import Path

import librosa
import numpy as np

from config import CONFIG


def load_audio(path: str | Path) -> tuple[np.ndarray, dict]:
    """Đọc thực tế bằng librosa, trộn kênh về mono và đổi SR bằng resampling."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp âm thanh: {path}")
    y, sr = librosa.load(path, sr=CONFIG.sample_rate, mono=True, dtype=np.float32)
    if y.size == 0 or not np.isfinite(y).all():
        raise ValueError("Tín hiệu rỗng hoặc có mẫu NaN/Inf.")
    peak = float(np.max(np.abs(y)))
    if peak < 1e-7:
        raise ValueError("Tín hiệu im lặng hoặc biên độ gần bằng 0.")
    # RMS cửa sổ 25 ms, bước 10 ms; ngưỡng sàn và ngưỡng tương đối đều áp dụng.
    rms = librosa.feature.rms(y=y, frame_length=400, hop_length=160, center=True)[0]
    dbfs = librosa.amplitude_to_db(rms + 1e-12, ref=1.0)
    threshold = max(CONFIG.silence_floor_dbfs, float(np.max(dbfs)) - CONFIG.silence_relative_db)
    active = dbfs > threshold
    duration = len(y) / CONFIG.sample_rate
    active_duration = float(np.sum(active) * 0.01)
    status = "ok" if active_duration >= CONFIG.min_duration_s else "insufficient_activity"
    meta = {
        "path": str(path), "sample_rate_hz": CONFIG.sample_rate,
        "duration_s": duration, "peak_amplitude": peak,
        "clipped_fraction": float(np.mean(np.abs(y) >= 0.999)),
        "silence_threshold_dbfs": threshold,
        "active_fraction": float(np.mean(active)), "active_duration_s": active_duration,
        "status": status,
    }
    return y, meta
