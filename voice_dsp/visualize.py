"""Matplotlib dashboard, jossa aika-akselit seuraavat alkuperäistä signaalia."""
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / "output" / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import librosa.display
import numpy as np


def dashboard(y, sequence: dict, metadata: dict, output: str | Path):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    t = sequence["time_s"]
    sr = 16000
    fig, ax = plt.subplots(6, 2, figsize=(17, 23), constrained_layout=True)
    fig.suptitle(Path(metadata["path"]).name + " — DSP voice analysis", fontsize=16)
    sample_t = np.arange(len(y))/sr
    ax[0, 0].plot(sample_t, y, lw=.45); ax[0, 0].set(xlabel="Thời gian (s)", ylabel="Biên độ", title="Waveform")
    ax[0, 1].plot(t, sequence["rms"], color="tab:blue", label="RMS")
    ax[0, 1].set(xlabel="Thời gian (s)", ylabel="RMS (biên độ tuyến tính)", title="RMS Energy và Zero-Crossing Rate")
    zax = ax[0, 1].twinx(); zax.plot(t, sequence["zcr"], color="tab:orange", alpha=.8, label="ZCR")
    zax.set_ylabel("ZCR (số lần đổi dấu / mẫu)")
    lines = ax[0,1].lines + zax.lines
    ax[0,1].legend(lines, [line.get_label() for line in lines], loc="upper right")
    freq = sequence["spectrum_hz"]; mag = sequence["spectrum_magnitude"]
    centroid = float(np.sum(freq*mag)/max(np.sum(mag), 1e-12))
    bandwidth = float(np.sqrt(np.sum(((freq-centroid)**2)*mag)/max(np.sum(mag),1e-12)))
    cumulative = np.cumsum(mag**2)
    rolloff = float(freq[min(np.searchsorted(cumulative, .85*cumulative[-1]), len(freq)-1)])
    ax[1, 0].plot(freq, mag, color="tab:blue", label="|FFT|")
    ax[1, 0].axvspan(max(0,centroid-bandwidth),centroid+bandwidth,alpha=.13,color="tab:green",label="Centroid ± bandwidth")
    ax[1, 0].axvline(centroid,color="tab:green",ls="--",label=f"Centroid {centroid:.0f} Hz")
    ax[1, 0].axvline(rolloff,color="tab:red",ls=":",label=f"Rolloff 85% {rolloff:.0f} Hz")
    ax[1, 0].set(xlim=(0, 8000), xlabel="Tần số (Hz)", ylabel="Biên độ FFT", title="Phổ FFT")
    ax[1,0].legend(fontsize=8)
    ax[1, 0].plot(sequence["spectrum_hz"], sequence["spectrum_magnitude"]); ax[1, 0].set(xlim=(0, 8000), xlabel="Tần số (Hz)", ylabel="Biên độ FFT", title="Phổ FFT")
    im = ax[1, 1].imshow(sequence["spectrogram_db"], origin="lower", aspect="auto", extent=[t[0], t[-1], 0, 8000], cmap="magma")
    ax[1, 1].set(title="Spectrogram (dB)", xlabel="Thời gian (s)", ylabel="Hz"); fig.colorbar(im, ax=ax[1, 1], label="dB")
    im = ax[2, 0].imshow(sequence["mel_db"], origin="lower", aspect="auto", extent=[t[0], t[-1], 0, sequence["mel_db"].shape[0]], cmap="magma")
    ax[2, 0].set(title="Mel spectrogram (dB)", xlabel="Thời gian (s)", ylabel="Mel band"); fig.colorbar(im, ax=ax[2, 0], label="dB")
    im = ax[2, 1].imshow(sequence["mfcc"], origin="lower", aspect="auto", extent=[t[0], t[-1], 1, sequence["mfcc"].shape[0]+1], cmap="coolwarm")
    ax[2, 1].set(title="MFCC (13 hệ số)", xlabel="Thời gian (s)", ylabel="Hệ số"); fig.colorbar(im, ax=ax[2, 1], label="Giá trị MFCC")
    ax[3, 0].plot(t, sequence["f0_hz"], ".-", ms=2); ax[3, 0].set(title="Pitch contour Praat autocorrelation", xlabel="Thời gian (s)", ylabel="F0 (Hz)", ylim=(0, 550))
    for i in range(3): ax[3, 1].plot(t, sequence["formants_hz"][i], ".", ms=2, label=f"F{i+1}")
    ax[3, 1].set(title="Formant contour Burg (ước lượng)", xlabel="Thời gian (s)", ylabel="Tần số (Hz)", ylim=(0, 6000)); ax[3, 1].legend()
    im = ax[4, 0].imshow(sequence["contrast"], origin="lower", aspect="auto", extent=[t[0], t[-1], 0, sequence["contrast"].shape[0]], cmap="viridis")
    ax[4, 0].set(title="Spectral contrast", xlabel="Thời gian (s)", ylabel="Dải tần"); fig.colorbar(im, ax=ax[4, 0], label="dB")
    im = ax[4, 1].imshow(sequence["chroma"], origin="lower", aspect="auto", extent=[t[0], t[-1], 0, 12], cmap="viridis")
    ax[4, 1].set(title="Chroma (12 lớp pitch class)", xlabel="Thời gian (s)", ylabel="Lớp chroma"); fig.colorbar(im, ax=ax[4, 1], label="Năng lượng tương đối")
    facts = ["Công thức / ý nghĩa đồ thị:", "Waveform: x[n], biên độ theo giây.", "RMS: sqrt(mean(x²)); ZCR: dấu đổi giữa mẫu kề nhau.", "FFT: biên độ theo Hz. Spectrogram: năng lượng STFT theo Hz và dB.", "Mel: gộp năng lượng qua bank Mel; MFCC: DCT log-Mel.", "F0: chu kỳ thanh quản; chấm chỉ khung hữu thanh.", "F1–F3: cộng hưởng ống thanh; Burg là ước lượng nhạy nguyên âm.", "Contrast: chênh peak–valley mỗi dải; chroma gộp octave vào 12 lớp.", "RMS/ZCR chịu ảnh hưởng âm lượng, nhiễu và âm vị; chroma hữu hạn với lời nói.", "Đặc trưng hỗ trợ so khớp; không một đại lượng riêng nào xác minh danh tính."]
    ax[5, 0].axis("off"); ax[5, 0].text(0, 1, "\n\n".join(facts), va="top", wrap=True, fontsize=10)
    ax[5, 1].axis("off"); ax[5, 1].text(0, 1, f"Duration: {metadata['duration_s']:.2f} s\nActive: {metadata['active_duration_s']:.2f} s\nPeak: {metadata['peak_amplitude']:.4f}\nClipped fraction: {metadata['clipped_fraction']:.6g}\nStatus: {metadata['status']}\n\nNote: F0, formant and activity are estimates; compare whole groups, not one value.", va="top", fontsize=12)
    fig.canvas.draw()
    fig.savefig(output/"dashboard.png", dpi=300)
    names = {(0,0):"waveform",(0,1):"energy_zcr",(1,0):"spectrum",(1,1):"spectrogram",(2,0):"mel_spectrogram",(2,1):"mfcc",(3,0):"f0",(3,1):"formants",(4,0):"contrast",(4,1):"chroma"}
    fig_width, fig_height = fig.get_size_inches()
    for (r,c), name in names.items():
        bbox = ax[r,c].get_position()
        left, bottom = bbox.x0*fig_width, bbox.y0*fig_height
        width, height = bbox.width*fig_width, bbox.height*fig_height
        # Giữ colorbar gắn với heatmap trong PNG riêng.
        for extra in fig.axes:
            pos = extra.get_position()
            if pos.x0 >= bbox.x1 and pos.x0-bbox.x1 < .08 and abs(pos.y0-bbox.y0) < .04 and abs(pos.height-bbox.height) < .1:
                width = max(width, (pos.x1-bbox.x0)*fig_width)
        right_pad = .65 if (r,c)==(0,1) else .12
        fig.savefig(output/f"{name}.png", dpi=300, bbox_inches=Bbox.from_bounds(left-.12,bottom-.12,width+right_pad+.12,height+.24))
    plt.close(fig)
    return output/"dashboard.png"
