"""Đặc trưng phổ, cepstral, cao độ và formant trích rút từ tiếng nói."""
import numpy as np
import librosa
import parselmouth

from config import CONFIG


def _stats(prefix: str, values: np.ndarray, out: dict, percentiles=()):
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    out[prefix + "_mean"] = float(np.mean(v))
    out[prefix + "_std"] = float(np.std(v))
    out[prefix + "_min"] = float(np.min(v))
    out[prefix + "_max"] = float(np.max(v))
    for q in percentiles:
        out[f"{prefix}_p{q}"] = float(np.percentile(v, q))


def extract(y: np.ndarray, metadata: dict) -> tuple[dict, dict]:
    """Trả vector thống kê một hàng và ma trận chuỗi dùng cho đồ thị sidecar."""
    c = CONFIG
    if len(y) < int(c.sample_rate * c.min_duration_s):
        raise ValueError("Âm thanh ngắn hơn 0,25 giây.")
    # center=True căn tâm các khung tại i*hop/sr trên cả ma trận đặc trưng.
    D = np.abs(librosa.stft(y, n_fft=c.n_fft, hop_length=c.hop_length,
                            win_length=c.n_fft, window="hann", center=True))
    times = librosa.frames_to_time(np.arange(D.shape[1]), sr=c.sample_rate,
                                   hop_length=c.hop_length)
    rms = librosa.feature.rms(S=D, frame_length=c.n_fft)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=c.n_fft,
                                              hop_length=c.hop_length, center=True)[0]
    db = librosa.amplitude_to_db(rms + 1e-12, ref=1.0)
    voice_thr = max(c.silence_floor_dbfs, float(np.max(db)) - c.silence_relative_db)
    voiced_activity = (db > voice_thr) & (rms > 1e-6)

    centroid = librosa.feature.spectral_centroid(S=D, sr=c.sample_rate)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=D, sr=c.sample_rate)[0]
    rolloff = librosa.feature.spectral_rolloff(S=D, sr=c.sample_rate, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(S=D)[0]
    mel = librosa.feature.melspectrogram(S=D**2, sr=c.sample_rate, n_fft=c.n_fft,
                                         hop_length=c.hop_length, n_mels=c.n_mels,
                                         power=2.0)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel + 1e-12),
                                sr=c.sample_rate, n_mfcc=c.n_mfcc)
    contrast = librosa.feature.spectral_contrast(S=D, sr=c.sample_rate,
                                                  n_fft=c.n_fft, hop_length=c.hop_length,
                                                  fmin=125, n_bands=5)
    chroma = librosa.feature.chroma_stft(S=D**2, sr=c.sample_rate,
                                          n_fft=c.n_fft, hop_length=c.hop_length,
                                          n_chroma=12)
    # Praat autocorrelation F0: phương pháp truyền thống nhanh cho kho lớn.
    # Xử lý trực tiếp trên cùng lưới 10 ms; frame vô thanh là NaN.
    sound = parselmouth.Sound(y, sampling_frequency=c.sample_rate)
    pitch = sound.to_pitch_ac(time_step=c.hop_length/c.sample_rate,
                              pitch_floor=c.f0_min_hz,
                              pitch_ceiling=c.f0_max_hz,
                              very_accurate=False)
    pitch_times = pitch.xs()
    pitch_values = pitch.selected_array["frequency"].astype(float)
    pitch_values[pitch_values <= 0] = np.nan
    f0 = np.full(len(times), np.nan)
    # Lấy frame Praat gần nhất (lưới 10 ms đồng pha với STFT).
    nearest = np.floor((times - pitch_times[0])/(c.hop_length/c.sample_rate) + 0.5).astype(int)
    valid_pitch = (nearest >= 0) & (nearest < len(pitch_values))
    f0[valid_pitch] = pitch_values[nearest[valid_pitch]]

    # Burg formants ước lượng riêng theo các tâm khung 10 ms, cửa sổ 25 ms.
    formant = sound.to_formant_burg(time_step=0.01, max_number_of_formants=5,
                                    maximum_formant=5500, window_length=0.025)
    formants = np.full((3, len(times)), np.nan)
    for j, t in enumerate(times):
        if not voiced_activity[j]:
            continue
        for k in range(1, 4):
            value = formant.get_value_at_time(k, float(t))
            if np.isfinite(value) and (150 <= value <= 5500):
                formants[k-1, j] = value
        a, b, d = formants[:, j]
        # Loại các nghiệm đảo thứ tự/ngoài dải thực dụng để không gây nhiễu DB.
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(d)
                and a < b < d and a >= 150 and d <= 5500):
            formants[:, j] = np.nan

    # Tính FFT một cửa sổ Hann từ đoạn hoạt động dài nhất, cho phổ đại diện.
    active_idx = np.flatnonzero(voiced_activity)
    center = len(y) // 2 if not len(active_idx) else int(times[active_idx[len(active_idx)//2]] * c.sample_rate)
    lo, hi = max(0, center-c.n_fft//2), min(len(y), center+c.n_fft//2)
    segment = np.zeros(c.n_fft)
    piece = y[lo:hi]
    seg_lo = max(0, c.n_fft//2 - center)
    segment[seg_lo:seg_lo+len(piece)] = piece
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(c.n_fft)))
    frequencies = np.fft.rfftfreq(c.n_fft, 1 / c.sample_rate)

    row = dict(metadata)
    row["voiced_frame_fraction"] = float(np.mean(np.isfinite(f0)))
    row["silence_frame_fraction"] = float(1.0 - np.mean(voiced_activity))
    row["rms_dbfs_mean"] = float(np.mean(db[voiced_activity])) if voiced_activity.any() else np.nan
    _stats("rms", rms[voiced_activity], row)
    _stats("zcr", zcr[voiced_activity], row)
    for name, values in [("centroid_hz", centroid), ("bandwidth_hz", bandwidth),
                         ("rolloff_hz", rolloff), ("flatness", flatness)]:
        _stats(name, values[voiced_activity], row)
    _stats("f0_hz", f0, row, (5, 25, 50, 75, 95))
    if "f0_hz_min" in row and "f0_hz_max" in row:
        row["f0_hz_range"] = row["f0_hz_max"] - row["f0_hz_min"]
    _stats("formant_f1_hz", formants[0], row)
    _stats("formant_f2_hz", formants[1], row)
    _stats("formant_f3_hz", formants[2], row)
    for i in range(c.n_mfcc):
        _stats(f"mfcc_{i+1:02d}", mfcc[i, voiced_activity], row)
    for i in range(contrast.shape[0]):
        _stats(f"contrast_band_{i+1}", contrast[i, voiced_activity], row)

    # Các chuỗi biến thiên nằm ở NPZ nén riêng, không nhồi ma trận vào CSV.
    sequence = {"time_s": times, "rms": rms, "zcr": zcr, "f0_hz": f0,
                "formants_hz": formants, "mfcc": mfcc, "mel_db": librosa.power_to_db(mel+1e-12),
                "spectrogram_db": librosa.amplitude_to_db(D+1e-12),
                "contrast": contrast, "chroma": chroma,
                "spectrum_hz": frequencies, "spectrum_magnitude": spectrum}
    return row, sequence
