"""
Gradio Web UI for Sora Watermark Remover.
Provides an interactive interface for uploading videos and removing watermarks.
"""

import os
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from loguru import logger

from config import CONFIG
from core.watermark_remover import WatermarkRemover
from core.watermark_detector import WatermarkDetector
from core.video_processor import VideoProcessor
from core.mask_generator import MaskGenerator
from utils.file_utils import get_output_path, ensure_dir
from utils.logger import setup_logging


# Initialize logging
setup_logging()


def get_first_frame(video_path: str) -> np.ndarray | None:
    """Extract first frame from video for preview."""
    if not video_path:
        return None
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def preview_detection(
    video_path: str,
    position: str,
    manual_x: int,
    manual_y: int,
    manual_w: int,
    manual_h: int,
) -> np.ndarray | None:
    """Preview watermark detection on the first frame."""
    if not video_path:
        return None

    frame = get_first_frame(video_path)
    if frame is None:
        return None

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    detector = WatermarkDetector()
    mask_gen = MaskGenerator()

    if position == "manual":
        if manual_w > 0 and manual_h > 0:
            region = (manual_x, manual_y, manual_w, manual_h)
        else:
            return frame
    elif position == "auto":
        region = detector.detect(frame_bgr)
    else:
        detector.position = position
        region = detector.detect(frame_bgr)

    if region is None:
        return frame

    # Draw detection visualization
    mask = mask_gen.create_mask(frame_bgr, region)
    overlay = mask_gen.visualize_mask(frame_bgr, mask, color=(0, 0, 255), alpha=0.4)

    x, y, w, h = region
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        overlay, f"Watermark: {w}x{h}", (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
    )

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def process_video(
    video_path: str,
    position: str,
    method: str,
    manual_x: int,
    manual_y: int,
    manual_w: int,
    manual_h: int,
    mask_expansion: int,
    mask_feather: int,
    inpaint_radius: int,
    temporal_smoothing: bool,
    progress=gr.Progress(),
) -> str | None:
    """Process video and remove watermark."""
    if not video_path:
        gr.Warning("Please upload a video file")
        return None

    try:
        # Setup remover with custom settings
        config = {
            "method": method.lower(),
            "radius": inpaint_radius,
            "mask_expansion": mask_expansion,
            "mask_feather": mask_feather,
            "temporal_smoothing": temporal_smoothing,
            "temporal_window": 5,
        }
        remover = WatermarkRemover(config=config)

        # Determine watermark region
        region = None
        if position == "manual":
            if manual_w > 0 and manual_h > 0:
                region = (manual_x, manual_y, manual_w, manual_h)
            else:
                gr.Warning("Please set manual region dimensions (W and H must be > 0)")
                return None
        elif position != "auto":
            remover.detector.position = position

        # Generate output path
        output_dir = ensure_dir("output")
        input_name = Path(video_path).stem
        output_path = str(output_dir / f"{input_name}_no_watermark.mp4")

        # Process with progress tracking
        def progress_callback(current, total):
            progress(current / total, desc=f"Processing frame {current}/{total}")

        result_path = remover.remove_from_video(
            input_path=video_path,
            output_path=output_path,
            region=region,
            method=method.lower(),
            progress_callback=progress_callback,
        )

        gr.Info("Watermark removal complete!")
        return result_path

    except Exception as e:
        logger.error(f"Processing error: {e}")
        gr.Warning(f"Error: {str(e)}")
        return None


def build_ui() -> gr.Blocks:
    """Build the Gradio UI."""
    
    with gr.Blocks(
        title="Sora Watermark Remover",
        theme=gr.themes.Soft(),
        css="""
        .main-title { text-align: center; margin-bottom: 20px; }
        .info-box { padding: 15px; border-radius: 8px; background: #f0f7ff; }
        """,
    ) as app:
        
        gr.Markdown(
            """
            # 🎬 Sora Watermark Remover
            
            Remove watermarks from Sora-generated videos. Upload your video, 
            configure detection settings, and get a clean output.
            """,
            elem_classes="main-title",
        )

        with gr.Row():
            # ====== LEFT COLUMN: Input & Settings ======
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input Video")
                video_input = gr.Video(
                    label="Upload Video",
                    sources=["upload"],
                )

                gr.Markdown("### ⚙️ Detection Settings")
                
                position = gr.Dropdown(
                    label="Watermark Position",
                    choices=[
                        "auto",
                        "bottom-right",
                        "bottom-left",
                        "bottom-center",
                        "top-right",
                        "top-left",
                        "manual",
                    ],
                    value="auto",
                    info="Select watermark location or 'auto' for automatic detection",
                )

                with gr.Accordion("Manual Region (if position = manual)", open=False):
                    with gr.Row():
                        manual_x = gr.Number(label="X", value=0, precision=0)
                        manual_y = gr.Number(label="Y", value=0, precision=0)
                    with gr.Row():
                        manual_w = gr.Number(label="Width", value=0, precision=0)
                        manual_h = gr.Number(label="Height", value=0, precision=0)

                preview_btn = gr.Button(
                    "👁 Preview Detection", variant="secondary",
                )

                gr.Markdown("### 🔧 Processing Settings")

                method = gr.Radio(
                    label="Inpainting Method",
                    choices=["TELEA", "NS", "Deep"],
                    value="TELEA",
                    info="TELEA: Fast | NS: Better quality | Deep: Best (requires PyTorch)",
                )

                with gr.Accordion("Advanced Settings", open=False):
                    mask_expansion = gr.Slider(
                        label="Mask Expansion (px)",
                        minimum=0, maximum=50, value=10, step=1,
                        info="Expand mask beyond detected region",
                    )
                    mask_feather = gr.Slider(
                        label="Mask Feather (px)",
                        minimum=0, maximum=20, value=5, step=1,
                        info="Smooth mask edges for seamless blending",
                    )
                    inpaint_radius = gr.Slider(
                        label="Inpaint Radius",
                        minimum=1, maximum=20, value=5, step=1,
                        info="Radius for inpainting algorithm",
                    )
                    temporal_smoothing = gr.Checkbox(
                        label="Temporal Smoothing",
                        value=True,
                        info="Reduce flickering across frames",
                    )

                process_btn = gr.Button(
                    "🚀 Remove Watermark", variant="primary", size="lg",
                )

            # ====== RIGHT COLUMN: Output & Preview ======
            with gr.Column(scale=1):
                gr.Markdown("### 👁 Detection Preview")
                preview_output = gr.Image(
                    label="Detection Preview",
                    type="numpy",
                )

                gr.Markdown("### 📥 Output Video")
                video_output = gr.Video(label="Processed Video")

        # ====== Footer ======
        gr.Markdown(
            """
            ---
            ### 📖 How to Use
            1. **Upload** your Sora video with watermark
            2. **Select position** or use 'auto' for automatic detection
            3. Click **Preview Detection** to verify the detected region
            4. Adjust settings if needed
            5. Click **Remove Watermark** to process
            
            ### 💡 Tips
            - **Auto detection** works best for standard Sora watermarks
            - Use **manual region** if auto-detection misses the watermark
            - **NS method** gives better quality than TELEA but is slower
            - **Temporal smoothing** reduces flickering in the cleaned region
            - For best results, use **mask expansion** of 5-15px
            """,
        )

        # ====== Event Handlers ======
        preview_btn.click(
            fn=preview_detection,
            inputs=[video_input, position, manual_x, manual_y, manual_w, manual_h],
            outputs=preview_output,
        )

        # Auto-preview on video upload
        video_input.change(
            fn=preview_detection,
            inputs=[video_input, position, manual_x, manual_y, manual_w, manual_h],
            outputs=preview_output,
        )

        process_btn.click(
            fn=process_video,
            inputs=[
                video_input, position, method,
                manual_x, manual_y, manual_w, manual_h,
                mask_expansion, mask_feather, inpaint_radius,
                temporal_smoothing,
            ],
            outputs=video_output,
        )

    return app


# Entry point
if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
