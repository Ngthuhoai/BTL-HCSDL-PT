# VoiceDSP: phân tích và tìm kiếm giọng nói tương đồng

Ứng dụng Python dùng xử lý tín hiệu truyền thống, không dùng mạng nơ-ron. Kho dữ liệu LSVSC đi kèm có 2.000 tệp WAV đơn kênh, 16 kHz; metadata có thể nối vào CSV đặc trưng.

## Cài đặt và chạy (PowerShell, từ thư mục Dataset)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r voice_dsp\requirements.txt
.\.venv\Scripts\python voice_dsp\run.py analyze LSVSC_female_stratified_2000\audio --metadata LSVSC_female_stratified_2000\metadata.csv --output voice_dsp\output
```

Để tạo dashboard cho một WAV, thêm `--plots`; với cả kho, tùy chọn này tạo chín PNG 300 DPI và NPZ chuỗi đặc trưng cho mỗi tệp, nên cần nhiều dung lượng. Mặc định batch chỉ ghi vector gọn và tiếp tục ghi CSV theo lô 25 tệp, phù hợp kho 1.000+ tệp. Lỗi từng WAV được ghi thành hàng `status=error`, tiến trình vẫn tiếp tục.

Tìm top 5 bằng vector query và thốngיקות corpus:

```powershell
.\.venv\Scripts\python voice_dsp\run.py search path\to\query.wav --library voice_dsp\output\features.csv --output voice_dsp\output\search --top 5
```

So sánh cặp và vẽ hai dashboard:

```powershell
.\.venv\Scripts\python voice_dsp\run.py compare a.wav b.wav --reference voice_dsp\output\features.csv --output voice_dsp\output\pair
```

Kết quả nằm trong thư mục output: `features.csv` (một hàng/tệp), `sequences/<id>.npz` khi yêu cầu đồ thị, các dashboard PNG, và báo cáo JSON/CSV so khớp. NPZ gồm các ma trận theo khung thay vì nhồi chúng trong ô CSV.

## Mô hình xử lý

```text
WAV -> mono/resample 16 kHz -> kiểm tra & đánh dấu frame im lặng
    -> STFT 1024 mẫu / hop 160 mẫu (64 ms / 10 ms)
    -> đặc trưng theo frame -> thống kê gọn CSV + chuỗi NPZ
    -> scaler mean/std học trên thư viện -> khoảng cách theo nhóm -> top-k
```

Thời gian frame của librosa được tính bằng `i × hop / 16000` (center padding); dashboard giữ nguyên trục thời gian của toàn tệp, không ghép tiếng nói rời nhau. Cửa sổ Hann. FFT bin rộng 15,625 Hz. ZCR/RMS dùng cùng nhịp 10 ms. Ngưỡng hoạt động là giá trị lớn hơn `max(-60 dBFS, max(RMS dBFS)-35 dB)`; đây là phát hiện hoạt động âm học, không phải nhận dạng tiếng nói/VAD và không đo SNR. Bản ghi dưới 0,25 giây hoạt động được đánh dấu, không đưa vào tìm kiếm.

## Diễn giải và giới hạn khoa học

- **Waveform:** mẫu x[n], biên độ số và thời gian; trị tuyệt đối tối đa báo clipping gần full scale. Không tự chuẩn hóa biên độ vì sẽ mất thông tin gain.
- **RMS:** `sqrt(mean(x²))` trên frame; năng lượng/độ lớn ngắn hạn, chịu ảnh hưởng gain, khoảng cách mic và nguyên âm. **ZCR:** số lần dấu mẫu kề nhau đổi, chia độ dài frame; hữu ích phân biệt tiếng xát/vô thanh với nguyên âm nhưng nhạy nhiễu.
- **FFT:** `|FFT(x·Hann)|`; trục Hz, biên độ phụ thuộc cửa sổ và kích thước frame. Centroid `ΣfA/ΣA`, bandwidth là căn bậc hai trung bình bình phương khoảng cách centroid, rolloff là phân vị 85% tổng năng lượng, flatness là trung bình nhân/chia trung bình phổ. Chúng mô tả độ sáng/phân bố phổ, không phải định danh.
- **Spectrogram:** `20log10(|STFT|)` theo thời gian, Hz và dB tương đối; harmonic là vệt cách nhau xấp xỉ F0. Formant có thể hiện như vùng tăng cường phổ; âm học phòng và âm vị cũng tạo hình tương tự.
- **Mel:** năng lượng qua 40 bộ lọc tam giác thang Mel (ở đây `m = 2595 log10(1+f/700)`); gom độ phân giải tần cao cho gần cảm nhận hơn. **MFCC:** DCT-II của log năng lượng Mel; hệ số 1–13 tóm lược bao phổ và âm sắc. CSV lưu mean/std trên frame hoạt động thành 26 chiều; nhạy nội dung câu, kênh và noise.
- **F0 Praat autocorrelation:** chu kỳ cơ bản của tiếng hữu thanh; Praat tự phân loại voiced/unvoiced và contour giữ NaN ở frame vô thanh. Mean/std/min/max/range/percentile Hz mô tả độ cao và biến thiên ngữ điệu. Không đồng nhất F0 với giới tính/người nói; phương pháp tự tương quan có thể lỗi octave, dropout hoặc bị tiếng ồn ảnh hưởng. Tùy chọn pYIN của librosa có thể thay thế nhưng chậm hơn đáng kể khi trích xuất hàng nghìn tệp.
- **Formant Burg:** đỉnh cộng hưởng ống thanh F1–F3 (Hz); xấp xỉ cộng hưởng nguyên âm, thường phụ thuộc âm vị mạnh. Hàng không đúng thứ tự hoặc ngoài 150–5500 Hz thành NaN; ngay cả giá trị còn lại là ước lượng, đặc biệt nhạy giọng cao, giọng thở và cài đặt maximum formant.
- **Spectral contrast:** chênh mức peak/valley theo sáu dải (dB), gợi ý phổ có đỉnh hẹp hay nhiễu rộng; chịu ảnh hưởng âm vị và tần số lấy mẫu.
- **Chroma:** 12 pitch class sau gộp octave, màu là năng lượng chuẩn hóa; khác F0 vì có thể cộng các harmonic ở nhiều octave. Hữu ích cho hòa âm nhạc, yếu cho lời nói vốn không theo 12 nốt đều.

API tham khảo: [librosa pYIN](https://librosa.org/doc/0.11.0/generated/librosa.pyin.html), [librosa spectral contrast](https://librosa.org/doc/main/generated/librosa.feature.spectral_contrast.html), [Parselmouth Burg formants](https://parselmouth.readthedocs.io/en/stable/api_reference.html#parselmouth.Sound.to_formant_burg).

Trong heatmap, X là thời gian giây, Y là frame/hệ số/dải và màu là đại lượng ghi trên colorbar. Tìm kiếm dùng z-score mean/std của các hàng hợp lệ trong corpus; mỗi nhóm pitch, MFCC, formant, spectral góp một khoảng cách RMS cân bằng. Cosine và Euclidean theo nhóm là phép so vector chuẩn hóa. Cột thiếu >20%, hằng hoặc thiếu ở query bị bỏ qua; query cần ít nhất hai cột nhóm. Các điểm gần nhất là hàng xóm đặc trưng, không phải xác suất hay khẳng định cùng người nói. Chỉ kết luận hiệu năng nhận diện sau đánh giá theo speaker-disjoint, các mẫu ghi âm khác phiên, ROC/EER và kiểm soát nội dung/kênh.

**Lưu ý benchmark:** nếu corpus không có mã người nói, không thể loại bản ghi cùng người nói khi đánh giá truy hồi. Nối metadata hiện có chỉ cung cấp `file_id`, tuổi và vùng miền; không bịa speaker ID. Các tệp phải chứa một người nói theo điều kiện dữ liệu của đề tài.

## Khắc phục lỗi

- Lỗi cài đặt: kích hoạt internet/proxy PyPI rồi cài `requirements.txt` trong Python 3.10–3.12; không đổi sang Python global nếu muốn giữ môi trường dự án.
- Tệp hỏng/codec không hỗ trợ: chuyển tệp sang WAV PCM trước; lỗi cụ thể ghi trong CSV.
- Query quá ngắn/im lặng: dùng bản ghi có ít nhất 0,25 giây hoạt động. Ngưỡng phát hiện có thể cấu hình trong `config.py` theo microphone, nhưng không nên giảm nếu nền nhiễu cao.
- Formant/F0 thiếu: đây là ước lượng từ tín hiệu; kiểm tra âm thanh và phổ thủ công, không thay NaN bằng số 0.

## Kiến trúc

`audio.py` đọc/tiền xử lý; `features.py` trích đặc trưng; `visualize.py` vẽ; `compare.py` chuẩn hóa, so sánh và truy hồi; `run.py` CLI và xuất kết quả; `config.py` tham số chung.
