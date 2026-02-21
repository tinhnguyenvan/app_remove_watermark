# 🎬 Công cụ xoá Watermark Video

Phát hiện và xoá **text watermark** + **logo** khỏi video tự động.

Sử dụng **EasyOCR** để nhận diện chữ, **template matching** để tìm logo, và **OpenCV inpainting** (Navier-Stokes) để xoá sạch.

---

## ✨ Tính năng

- 🔍 **Nhận diện text bằng OCR** — tìm và xoá text watermark (vd: `@tinh.nguyenvan`, `Sora`)
- 🖼️ **Nhận diện logo bằng template matching** — tìm logo ở mọi vị trí, kể cả khi logo di chuyển ngẫu nhiên
- 🎨 **Xoá bằng inpainting Navier-Stokes** — chất lượng cao, giữ nguyên nền
- 🔊 **Giữ nguyên âm thanh** — tự động ghép audio từ video gốc bằng ffmpeg
- ⚡ **Tối ưu tốc độ** — OCR chạy mỗi N frame, logo dò mỗi frame

---

## 📋 Yêu cầu

- Python 3.10+
- ffmpeg (để ghép âm thanh)

---

## 🚀 Cài đặt

### 1. Cài dependencies

```bash
pip3 install -r requirements.txt
```

### 2. Cài ffmpeg (nếu chưa có)

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

---

## 📖 Hướng dẫn sử dụng

### Cơ bản — Xoá text watermark + logo

```bash
python3 main.py video.mp4 -o output/clean.mp4
```

Mặc định xoá text `@tinh.nguyenvan`, `Sora` và logo Sora (`media/logo_sora.png`).

### Dùng logo khác

```bash
python3 main.py video.mp4 -o output/clean.mp4 -l media/logo_sora.png
```

### Xoá text tuỳ chỉnh

```bash
python3 main.py video.mp4 -o output/clean.mp4 -t "@username,watermark_text"
```

Nhiều text phân cách bằng dấu phẩy.

---

## ⚙️ Tham số dòng lệnh

| Tham số | Mô tả | Mặc định |
|---------|--------|----------|
| `input` | Đường dẫn video đầu vào | *(bắt buộc)* |
| `-o`, `--output` | Đường dẫn video đầu ra | `{input}_clean.mp4` |
| `-t`, `--text` | Text watermark cần xoá (phân cách bằng `,`) | `@tinh.nguyenvan,sora` |
| `-l`, `--logo` | Đường dẫn ảnh logo để dò | `media/logo_sora.png` |
| `--logo-threshold` | Ngưỡng khớp logo 0-1 (thấp hơn = nhạy hơn) | `0.65` |
| `-e`, `--expand` | Số pixel mở rộng vùng xoá | `15` |
| `-d`, `--detect-every` | Chạy OCR mỗi N frame | `5` |
| `--lang` | Ngôn ngữ OCR (phân cách bằng `,`) | `en` |

---

## 📂 Cấu trúc thư mục

```
app_remove_watermark/
├── main.py              # Script chính
├── requirements.txt     # Dependencies
├── README.md            # Hướng dẫn (file này)
├── media/               # Video đầu vào & ảnh logo
│   ├── demo.mp4
│   └── logo_sora.png
└── output/              # Video đã xử lý
    └── clean.mp4
```

---

## 💡 Ví dụ nâng cao

### Tăng độ chính xác (OCR mỗi frame, chậm hơn)

```bash
python3 main.py video.mp4 -o output/clean.mp4 -l media/logo_sora.png -d 1
```

### Mở rộng vùng xoá (watermark lớn hoặc bị sót viền)

```bash
python3 main.py video.mp4 -o output/clean.mp4 -e 25
```

### Giảm ngưỡng logo (nếu logo không bị phát hiện)

```bash
python3 main.py video.mp4 -o output/clean.mp4 -l media/logo_sora.png --logo-threshold 0.5
```

### Xoá nhiều loại text cùng lúc

```bash
python3 main.py video.mp4 -o output/clean.mp4 -t "@user1,@user2,watermark,Sora"
```

---

## 🔧 Cách hoạt động

1. **Tải mô hình OCR** (EasyOCR, lần đầu tải ~100MB)
2. **Tải template logo** ở 8 tỷ lệ khác nhau (nếu có)
3. **Duyệt từng frame:**
   - Dò **text** bằng OCR mỗi N frame → xoá text trước
   - Dò **logo** mỗi frame bằng template matching → xoá logo sau (trên frame đã sạch text)
4. **Tạo mask** cho vùng text + logo (logo dùng vùng mở rộng gấp đôi)
5. **Xoá watermark** bằng inpainting Navier-Stokes
6. **Ghép âm thanh** từ video gốc bằng ffmpeg

---

## ⚠️ Lưu ý

- Lần chạy đầu tiên sẽ tải mô hình OCR (~100MB), các lần sau dùng cache
- Logo di chuyển ngẫu nhiên giữa các frame → dò mỗi frame, không dùng vị trí cố định
- Chất lượng inpainting phụ thuộc vào nền video — nền phức tạp có thể bị mờ nhẹ
- Cần **ffmpeg** để giữ âm thanh, nếu không có thì video xuất ra sẽ không có tiếng
- Hỗ trợ CPU, không bắt buộc GPU

---

## 📊 Hiệu suất tham khảo

| Cấu hình | Tốc độ | Ghi chú |
|----------|--------|---------|
| `-d 5` (mặc định) | ~2-3 fps | Cân bằng tốc độ/chất lượng |
| `-d 1` (mỗi frame) | ~1-1.5 fps | Chính xác nhất |
| `-d 10` | ~3-4 fps | Nhanh, phù hợp text cố định |

*Đo trên MacBook, CPU only, video 704x1280*
