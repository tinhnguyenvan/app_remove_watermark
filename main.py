"""
Công cụ xoá watermark video
Phát hiện và xoá text "@tinh.nguyenvan" + logo Sora khỏi video.
Sử dụng EasyOCR để nhận diện chữ, template matching để tìm logo, và OpenCV inpainting để xoá.
"""

import sys
import os
import argparse
import subprocess
import warnings
import cv2
import numpy as np

# Tắt cảnh báo pin_memory của PyTorch (không ảnh hưởng trên Apple Silicon)
warnings.filterwarnings("ignore", message=".*pin_memory.*")

import easyocr
from tqdm import tqdm


# Đường dẫn logo mặc định
DEFAULT_LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media", "logo_sora.png")


class TextWatermarkRemover:
    """Phát hiện text/logo watermark trong video bằng OCR và xoá bằng inpainting."""

    def __init__(
        self,
        target_texts: list[str] = None,
        logo_path: str | None = None,
        logo_threshold: float = 0.55,
        languages: list[str] = None,
    ):
        """
        Tham số:
            target_texts: Danh sách text watermark cần tìm và xoá.
            logo_path: Đường dẫn ảnh logo để dò bằng template matching.
                        Mặc định: media/logo_sora.png
            logo_threshold: Ngưỡng khớp logo (0-1).
            languages: Ngôn ngữ OCR. Mặc định: ['en'].
        """
        self.target_texts = [t.lower() for t in (target_texts or ["@tinh.nguyenvan", "Sora"])]
        self.languages = languages or ["en"]
        self.logo_threshold = logo_threshold

        # Tải template logo (dùng đường dẫn mặc định nếu không truyền)
        self.logo_templates = []  # danh sách (template, w, h) ở nhiều tỷ lệ
        effective_logo = logo_path or DEFAULT_LOGO_PATH
        if effective_logo and os.path.isfile(effective_logo):
            self._load_logo(effective_logo)
        
        print(f"🔍 Text cần xoá: {self.target_texts}")
        if self.logo_templates:
            print(f"🖼️  Đã tải template logo ({len(self.logo_templates)} tỷ lệ)")
        print(f"⏳ Đang tải mô hình OCR...")
        self.reader = easyocr.Reader(self.languages, gpu=False)
        print(f"✅ Đã tải mô hình OCR")

    def _load_logo(self, path: str) -> None:
        """Tải logo và tạo template nhiều tỷ lệ để dò chính xác hơn."""
        logo = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if logo is None:
            print(f"⚠️  Không đọc được logo: {path}")
            return

        # Chuyển sang BGR nếu ảnh có kênh alpha (BGRA)
        if logo.shape[2] == 4:
            self.logo_alpha = logo[:, :, 3]
            logo = logo[:, :, :3]
        else:
            self.logo_alpha = None

        # Tạo template ở nhiều tỷ lệ cho các độ phân giải video khác nhau
        for scale in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
            new_w = max(16, int(logo.shape[1] * scale))
            new_h = max(16, int(logo.shape[0] * scale))
            tmpl = cv2.resize(logo, (new_w, new_h), interpolation=cv2.INTER_AREA)
            self.logo_templates.append((tmpl, new_w, new_h))

    def find_text_regions(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Tìm vùng chứa text watermark trong frame.

        Tham số:
            frame: Ảnh BGR.

        Trả về:
            Danh sách toạ độ (x, y, w, h) của các vùng text khớp.
        """
        # EasyOCR trả về danh sách (bbox, text, confidence)
        results = self.reader.readtext(frame)
        
        regions = []
        for bbox, text, conf in results:
            text_lower = text.lower().strip()
            matched = self._is_match(text_lower)
            
            # Log tất cả text OCR tìm thấy
            status = "✅ MATCH" if matched else "❌ skip"
            print(f"    [OCR] {status} | conf={conf:.2f} | \"{text}\"")
            
            if matched:
                # bbox có dạng [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                xs = [int(p[0]) for p in bbox]
                ys = [int(p[1]) for p in bbox]
                x = min(xs)
                y = min(ys)
                w = max(xs) - x
                h = max(ys) - y
                regions.append((x, y, w, h))
        
        return regions

    def find_logo_regions(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Tìm logo trong frame bằng template matching đa tỷ lệ.
        Logo có thể di chuyển ngẫu nhiên giữa các frame nên phải dò mỗi frame.

        Tham số:
            frame: Ảnh BGR.

        Trả về:
            Danh sách toạ độ (x, y, w, h) của logo tìm thấy.
        """
        if not self.logo_templates:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        best_val = 0
        best_region = None

        for tmpl_bgr, tw, th in self.logo_templates:
            if tw >= frame.shape[1] or th >= frame.shape[0]:
                continue

            tmpl_gray = cv2.cvtColor(tmpl_bgr, cv2.COLOR_BGR2GRAY)
            result = cv2.matchTemplate(gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_val:
                best_val = max_val
                best_region = (int(max_loc[0]), int(max_loc[1]), tw, th)

        # Trả về best match + score để caller quyết định dùng hay không
        if best_region is not None:
            return [best_region], best_val

        return [], best_val

    def _is_match(self, detected_text: str) -> bool:
        """Kiểm tra text phát hiện có khớp với watermark cần xoá không."""
        detected = detected_text.lower()
        
        for target in self.target_texts:
            # Khớp chính xác hoặc chứa chuỗi con
            if target in detected or detected in target:
                return True
            
            # Khớp từng phần: "tinh", "nguyenvan", "@tinh", "sora"
            clean_target = target.replace("@", "").replace(".", " ").replace("_", " ")
            key_parts = [p for p in clean_target.split() if len(p) >= 3]
            
            clean_detected = detected.replace("@", "").replace(".", " ").replace("_", " ")
        
            for part in key_parts:
                if part in clean_detected:
                    return True
        
        return False

    def create_inpaint_mask(
        self, frame_shape: tuple, regions: list[tuple[int, int, int, int]], expand: int = 15
    ) -> np.ndarray:
        """
        Tạo mask nhị phân từ các vùng text đã phát hiện.

        Tham số:
            frame_shape: Kích thước frame (height, width, channels).
            regions: Danh sách toạ độ (x, y, w, h).
            expand: Số pixel mở rộng mỗi vùng.

        Trả về:
            Ảnh mask (uint8), 255 = vùng cần xoá.
        """
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for x, y, rw, rh in regions:
            x1 = max(0, x - expand)
            y1 = max(0, y - expand)
            x2 = min(w, x + rw + expand)
            y2 = min(h, y + rh + expand)
            mask[y1:y2, x1:x2] = 255
        
        return mask

    def remove_text_from_frame(
        self, frame: np.ndarray, regions: list[tuple[int, int, int, int]], expand: int = 15
    ) -> np.ndarray:
        """
        Xoá vùng text watermark khỏi frame bằng inpainting.

        Tham số:
            frame: Ảnh BGR.
            regions: Danh sách toạ độ (x, y, w, h) cần xoá.
            expand: Số pixel mở rộng mask.

        Trả về:
            Frame đã xoá watermark.
        """
        if not regions:
            return frame
        
        mask = self.create_inpaint_mask(frame.shape, regions, expand)
        # Dùng thuật toán Navier-Stokes cho chất lượng tốt nhất
        result = cv2.inpaint(frame, mask, inpaintRadius=12, flags=cv2.INPAINT_NS)
        return result

    def process_video(
        self,
        input_path: str,
        output_path: str,
        expand: int = 15,
        detect_every: int = 5,
    ) -> str:
        """
        Xử lý toàn bộ video: phát hiện text bằng OCR và xoá.

        Tham số:
            input_path: Đường dẫn video đầu vào.
            output_path: Đường dẫn video đầu ra.
            expand: Số pixel mở rộng mask.
            detect_every: Chạy OCR mỗi N frame (dùng lại kết quả cũ ở giữa).

        Trả về:
            Đường dẫn video đầu ra.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Không thể mở video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"📹 Video: {width}x{height}, {fps:.1f} fps, {total} frame")

        # Đường dẫn tạm cho video chưa có âm thanh
        temp_path = output_path + ".temp.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Không thể tạo video đầu ra: {temp_path}")

        last_text_regions = []
        last_logo_region = None  # Vị trí logo frame trước (fallback)
        fallback_threshold = 0.40  # Ngưỡng thấp hơn cho fallback
        removed_count = 0
        logo_stats = {"high": 0, "fallback": 0, "last_pos": 0, "miss": 0}

        for idx in tqdm(range(total), desc="Đang xử lý"):
            ret, frame = cap.read()
            if not ret:
                break

            # --- Bước 1: Xoá TEXT trước ---
            if idx % detect_every == 0:
                print(f"\n  🔍 Frame {idx}: Chạy OCR...")
                text_regions = self.find_text_regions(frame)
                if text_regions:
                    last_text_regions = text_regions
                    print(f"    → Tìm thấy {len(text_regions)} vùng text")
                else:
                    print(f"    → Không tìm thấy text khớp")

            if last_text_regions:
                h_frame, w_frame = frame.shape[:2]
                text_mask = np.zeros((h_frame, w_frame), dtype=np.uint8)
                for x, y, rw, rh in last_text_regions:
                    x1 = max(0, x - expand)
                    y1 = max(0, y - expand)
                    x2 = min(w_frame, x + rw + expand)
                    y2 = min(h_frame, y + rh + expand)
                    text_mask[y1:y2, x1:x2] = 255
                frame = cv2.inpaint(frame, text_mask, inpaintRadius=12, flags=cv2.INPAINT_NS)

            # --- Bước 2: Xoá LOGO sau (trên frame đã xoá text) ---
            # Fallback 2 tầng để tránh chớp:
            #   score ≥ threshold (0.65)  → tin cậy cao, dùng ngay
            #   score ≥ fallback (0.40)   → tin cậy thấp nhưng likely đúng
            #   score < fallback           → dùng vị trí frame trước
            logo_candidates, logo_score = self.find_logo_regions(frame)
            use_logo = None

            if logo_candidates and logo_score >= self.logo_threshold:
                use_logo = logo_candidates[0]
                last_logo_region = use_logo
                logo_stats["high"] += 1
                print(f"  🎯 Frame {idx}: Logo HIGH  score={logo_score:.3f} tại ({use_logo[0]},{use_logo[1]}) {use_logo[2]}x{use_logo[3]}")
            elif logo_candidates and logo_score >= fallback_threshold:
                use_logo = logo_candidates[0]
                last_logo_region = use_logo
                logo_stats["fallback"] += 1
                print(f"  🟡 Frame {idx}: Logo FALL  score={logo_score:.3f} tại ({use_logo[0]},{use_logo[1]}) {use_logo[2]}x{use_logo[3]}")
            elif last_logo_region is not None:
                use_logo = last_logo_region
                logo_stats["last_pos"] += 1
                print(f"  🔵 Frame {idx}: Logo PREV  score={logo_score:.3f} → dùng vị trí trước ({use_logo[0]},{use_logo[1]})")
            elif self.logo_templates:
                logo_stats["miss"] += 1
                print(f"  ⚪ Frame {idx}: Logo MISS  score={logo_score:.3f}")

            if use_logo:
                h_frame, w_frame = frame.shape[:2]
                logo_mask = np.zeros((h_frame, w_frame), dtype=np.uint8)
                logo_expand = expand * 2
                x, y, rw, rh = use_logo
                x1 = max(0, x - logo_expand)
                y1 = max(0, y - logo_expand)
                x2 = min(w_frame, x + rw + logo_expand)
                y2 = min(h_frame, y + rh + logo_expand)
                logo_mask[y1:y2, x1:x2] = 255
                frame = cv2.inpaint(frame, logo_mask, inpaintRadius=12, flags=cv2.INPAINT_NS)

            # Đếm frame đã xử lý
            if last_text_regions or use_logo:
                removed_count += 1

            writer.write(frame)

        cap.release()
        writer.release()

        print(f"📊 Đã xoá watermark {removed_count}/{total} frame")
        if self.logo_templates:
            print(f"📊 Logo: {logo_stats['high']} high + {logo_stats['fallback']} fallback + {logo_stats['last_pos']} prev_pos + {logo_stats['miss']} miss")

        # Ghép âm thanh từ video gốc
        final_path = self._merge_audio(input_path, temp_path, output_path)
        
        # Dọn file tạm
        if os.path.exists(temp_path) and final_path != temp_path:
            os.remove(temp_path)

        print(f"✅ Đã lưu: {final_path}")
        return final_path

    def _merge_audio(self, original: str, video_no_audio: str, output: str) -> str:
        """Ghép âm thanh từ video gốc vào video đã xử lý bằng ffmpeg."""
        try:
            # Kiểm tra ffmpeg có sẵn không
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("⚠️  Không tìm thấy ffmpeg - video sẽ không có âm thanh")
            os.rename(video_no_audio, output)
            return output

        # Kiểm tra video gốc có âm thanh không
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", original],
            capture_output=True, text=True,
        )
        if not probe.stdout.strip():
            print("ℹ️  Video gốc không có âm thanh")
            os.rename(video_no_audio, output)
            return output

        # Ghép: video đã xử lý + âm thanh từ video gốc
        print("🔊 Đang ghép âm thanh từ video gốc...")
        result = subprocess.run(
            ["ffmpeg", "-y",
             "-i", video_no_audio,
             "-i", original,
             "-c:v", "copy",
             "-c:a", "aac",
             "-map", "0:v:0",
             "-map", "1:a:0",
             "-shortest",
             output],
            capture_output=True, text=True,
        )

        if result.returncode == 0 and os.path.exists(output):
            print("🔊 Ghép âm thanh thành công")
            return output
        else:
            print(f"⚠️  Ghép âm thanh thất bại: {result.stderr[:200]}")
            os.rename(video_no_audio, output)
            return output


def main():
    parser = argparse.ArgumentParser(
        description="Xoá watermark (text + logo) khỏi video bằng OCR và template matching"
    )
    parser.add_argument("input", help="Đường dẫn video đầu vào")
    parser.add_argument("-o", "--output", help="Đường dẫn video đầu ra", default=None)
    parser.add_argument(
        "-t", "--text",
        help='Text watermark cần xoá, phân cách bằng dấu phẩy (mặc định: "@tinh.nguyenvan,Sora")',
        default="@tinh.nguyenvan,Sora",
    )
    parser.add_argument(
        "-l", "--logo",
        help="Đường dẫn ảnh logo (mặc định: media/logo_sora.png)",
        default=None,
    )
    parser.add_argument(
        "--logo-threshold",
        help="Ngưỡng khớp logo 0-1 (mặc định: 0.65, thấp hơn = nhạy hơn)",
        type=float, default=0.65,
    )
    parser.add_argument(
        "-e", "--expand",
        help="Số pixel mở rộng vùng xoá (mặc định: 15)",
        type=int, default=15,
    )
    parser.add_argument(
        "-d", "--detect-every",
        help="Chạy OCR mỗi N frame (mặc định: 5, dùng 1 để chính xác nhất)",
        type=int, default=5,
    )
    parser.add_argument(
        "--lang",
        help='Ngôn ngữ OCR, phân cách bằng dấu phẩy (mặc định: "en")',
        default="en",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"❌ Không tìm thấy file: {args.input}")
        sys.exit(1)

    # Đường dẫn đầu ra mặc định
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_clean{ext}"

    # Tạo thư mục đầu ra nếu chưa có
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    languages = [l.strip() for l in args.lang.split(",")]

    target_texts = [t.strip() for t in args.text.split(",")]
    remover = TextWatermarkRemover(
        target_texts=target_texts,
        logo_path=args.logo,
        logo_threshold=args.logo_threshold,
        languages=languages,
    )
    remover.process_video(
        input_path=args.input,
        output_path=args.output,
        expand=args.expand,
        detect_every=args.detect_every,
    )


if __name__ == "__main__":
    main()
