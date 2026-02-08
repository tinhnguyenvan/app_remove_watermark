# 🎬 Sora Watermark Remover

Ứng dụng Python xóa watermark từ video được tạo bởi Sora (OpenAI). Hỗ trợ giao diện web (Gradio) và CLI.

## 📁 Cấu trúc Project

```
app_remove_watermark/
├── app.py                      # Giao diện web Gradio (entry point chính)
├── main.py                     # CLI entry point
├── run.sh                      # Script khởi chạy nhanh
├── requirements.txt            # Dependencies
├── .gitignore
│
├── config/
│   ├── __init__.py             # Config loader
│   └── settings.yaml           # Cấu hình ứng dụng
│
├── core/
│   ├── __init__.py
│   ├── video_processor.py      # Đọc/ghi video, trích xuất frame
│   ├── watermark_detector.py   # Phát hiện vị trí watermark
│   ├── mask_generator.py       # Tạo mask chính xác cho vùng watermark
│   ├── watermark_remover.py    # Xóa watermark bằng inpainting
│   └── deep_inpainter.py       # Deep learning inpainting (optional)
│
├── utils/
│   ├── __init__.py
│   ├── file_utils.py           # Quản lý file/path
│   └── logger.py               # Cấu hình logging
│
├── models/                     # Thư mục chứa model weights (nếu dùng deep learning)
├── output/                     # Video đã xử lý
├── logs/                       # Log files
└── temp/                       # File tạm trong quá trình xử lý
```

## 🚀 Cài đặt & Chạy

### Cách 1: Script tự động
```bash
chmod +x run.sh
./run.sh
```

### Cách 2: Thủ công
```bash
# Tạo virtual environment
python3 -m venv venv
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy giao diện web
python app.py
```

Mở trình duyệt tại: **http://localhost:7860**

### Cách 3: CLI (Command Line)
```bash
# Tự động phát hiện và xóa watermark
python main.py video_input.mp4

# Chỉ định vị trí watermark
python main.py video_input.mp4 --position bottom-right

# Chỉ định vùng thủ công (X Y Width Height)
python main.py video_input.mp4 --region 800 600 200 50

# Sử dụng Navier-Stokes (chất lượng cao hơn)
python main.py video_input.mp4 --method ns

# Chỉ định output
python main.py video_input.mp4 -o output/clean.mp4
```

## ⚙️ Giải pháp kỹ thuật

### 1. Phát hiện Watermark (Detection)
- **Auto-detect**: Phân tích góc frame, tìm vùng tĩnh qua nhiều frame
- **Template matching**: So khớp mẫu watermark (multi-scale)
- **Edge analysis**: Phát hiện text/logo qua phân tích cạnh và tần số
- **Static region**: So sánh nhiều frame để tìm vùng không đổi (watermark)

### 2. Tạo Mask
- **Region-based**: Mask từ vùng phát hiện + mở rộng
- **Pixel-precise**: Phân tích contour để mask chính xác từng pixel
- **Feathering**: Làm mượt biên mask để blend tự nhiên

### 3. Xóa Watermark (Inpainting)
| Method | Tốc độ | Chất lượng | Yêu cầu |
|--------|--------|------------|----------|
| **TELEA** | ⚡ Nhanh | ⭐⭐⭐ | OpenCV |
| **Navier-Stokes** | 🐢 Chậm hơn | ⭐⭐⭐⭐ | OpenCV |
| **Deep Learning** | 🐢🐢 Chậm nhất | ⭐⭐⭐⭐⭐ | PyTorch + GPU |

### 4. Hậu xử lý
- **Temporal smoothing**: Giảm nhấp nháy giữa các frame
- **Mask feathering**: Blend mượt vùng đã xóa với xung quanh

## 💡 Mẹo sử dụng

1. **Bắt đầu với Auto**: Để ứng dụng tự phát hiện watermark trước
2. **Preview trước khi xử lý**: Kiểm tra vùng phát hiện đúng chưa
3. **Dùng Manual nếu cần**: Nếu auto-detect sai, chỉ định vùng thủ công
4. **Mask expansion 5-15px**: Mở rộng mask một chút cho kết quả tốt hơn
5. **NS cho chất lượng cao**: Phương pháp Navier-Stokes cho kết quả mượt hơn TELEA

## 📦 Dependencies chính

- **OpenCV**: Xử lý ảnh/video, inpainting
- **NumPy**: Xử lý mảng số
- **Gradio**: Giao diện web
- **Loguru**: Logging
- **PyTorch** (optional): Deep learning inpainting

## ⚠️ Lưu ý

- Chất lượng kết quả phụ thuộc vào độ phức tạp của vùng bên dưới watermark
- Video có nền đơn giản (trời, tường...) sẽ cho kết quả tốt nhất
- Video có chi tiết phức tạp dưới watermark có thể cần deep learning method
- Deep learning method yêu cầu PyTorch và GPU để chạy nhanh
