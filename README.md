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
- 📂 **Batch processing** — tự động xử lý tất cả video trong thư mục `media/`, lưu vào `output/`, bỏ qua video đã xử lý

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

### Batch — Xử lý hàng loạt (khuyên dùng)

Tự động xử lý **tất cả video** trong `media/` và lưu vào `output/`:

```bash
python3 main.py --batch
```

- Quét tất cả file video trong `media/` (`.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, `.flv`, `.webm`, `.m4v`)
- **Bỏ qua tự động** nếu file cùng tên đã tồn tại trong `output/` — chạy lại không xử lý lại video đã xong
- Khởi tạo mô hình OCR **một lần** rồi tái sử dụng cho tất cả video

Tuỳ chỉnh thư mục đầu vào/đầu ra:

```bash
python3 main.py --batch --media-dir /path/to/input --output-dir /path/to/output
```

### Đơn lẻ — Xoá watermark một video

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
| `input` | Đường dẫn video đầu vào (bỏ trống nếu dùng `--batch`) | — |
| `-o`, `--output` | Đường dẫn video đầu ra | `{input}_clean.mp4` |
| `--batch` | Xử lý hàng loạt tất cả video trong `media/` → `output/` | — |
| `--media-dir` | Thư mục chứa video đầu vào (batch mode) | `media/` |
| `--output-dir` | Thư mục lưu video đầu ra (batch mode) | `output/` |
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
│   ├── catim1.mp4       #   Đặt video cần xử lý vào đây
│   ├── catim2.mp4
│   └── logo_sora.png    #   Template logo
└── output/              # Video đã xử lý (tự động tạo)
    ├── catim1.mp4       #   Cùng tên file → batch sẽ bỏ qua
    └── catim2.mp4
```

---

## 💡 Ví dụ nâng cao

### Batch với tuỳ chỉnh

```bash
# Xử lý hàng loạt với OCR chính xác hơn và ngưỡng logo thấp hơn
python3 main.py --batch -d 1 --logo-threshold 0.5

# Xử lý thư mục tuỳ chỉnh
python3 main.py --batch --media-dir ~/Videos/input --output-dir ~/Videos/output
```

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
