"""
Watermark remover module.
Implements multiple inpainting strategies for watermark removal.
"""

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

from config import CONFIG
from core.video_processor import VideoProcessor
from core.watermark_detector import WatermarkDetector
from core.mask_generator import MaskGenerator


class WatermarkRemover:
    """
    Main class for removing watermarks from video frames.
    
    Supports multiple inpainting methods:
    - TELEA: Fast, good for small regions
    - Navier-Stokes (NS): Better quality, slower
    - Deep Learning: Best quality (requires model weights)
    """

    INPAINT_METHODS = {
        "telea": cv2.INPAINT_TELEA,
        "ns": cv2.INPAINT_NS,
    }

    def __init__(self, config: dict | None = None):
        self.config = config or CONFIG.get("inpainting", {})
        self.method = self.config.get("method", "telea")
        self.radius = self.config.get("radius", 5)
        self.temporal_smoothing = self.config.get("temporal_smoothing", True)
        self.temporal_window = self.config.get("temporal_window", 5)

        self.detector = WatermarkDetector()
        self.mask_gen = MaskGenerator(
            expansion=self.config.get("mask_expansion", 10),
            feather=self.config.get("mask_feather", 5),
        )

        self._previous_frames: list[np.ndarray] = []

    def remove_from_frame(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        method: str | None = None,
    ) -> np.ndarray:
        """
        Remove watermark from a single frame using inpainting.
        
        Args:
            frame: Input frame (BGR).
            mask: Binary mask (255 = watermark region).
            method: Inpainting method override.
            
        Returns:
            Frame with watermark removed.
        """
        method_name = method or self.method
        
        # Ensure mask is single channel uint8
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.astype(np.uint8)

        if method_name in self.INPAINT_METHODS:
            result = cv2.inpaint(
                frame, mask, self.radius, self.INPAINT_METHODS[method_name]
            )
        elif method_name == "deep":
            result = self._deep_inpaint(frame, mask)
        else:
            logger.warning(f"Unknown method '{method_name}', falling back to TELEA")
            result = cv2.inpaint(frame, mask, self.radius, cv2.INPAINT_TELEA)

        # Apply temporal smoothing if enabled
        if self.temporal_smoothing:
            result = self._apply_temporal_smoothing(result, mask)

        return result

    def remove_from_video(
        self,
        input_path: str,
        output_path: str,
        region: tuple[int, int, int, int] | None = None,
        template: np.ndarray | None = None,
        method: str | None = None,
        progress_callback=None,
    ) -> str:
        """
        Remove watermark from entire video.
        
        Args:
            input_path: Path to input video.
            output_path: Path for output video.
            region: Manual watermark region (x, y, w, h). Auto-detect if None.
            template: Watermark template image for detection.
            method: Inpainting method override.
            progress_callback: Callable(current_frame, total_frames) for progress.
            
        Returns:
            Path to the output video.
        """
        processor = VideoProcessor()

        try:
            # Open input video
            info = processor.open(input_path)
            logger.info(f"Processing video: {info.filepath}")

            # Detect watermark region
            if region is not None:
                self.detector.set_manual_region(*region)
                detected_region = region
            else:
                # Read a few frames for detection
                sample_frames = []
                sample_indices = np.linspace(
                    0, info.total_frames - 1, min(10, info.total_frames), dtype=int
                )
                for idx in sample_indices:
                    frame = processor.get_frame_at(int(idx))
                    if frame is not None:
                        sample_frames.append(frame)

                detected_region = self.detector.detect_from_frames(
                    sample_frames, sample_count=10
                )

                if detected_region is None:
                    logger.error("Could not detect watermark region")
                    raise RuntimeError("Watermark detection failed")

            logger.info(f"Watermark region: {detected_region}")

            # Create output video writer
            output_config = CONFIG.get("output", {})
            codec = CONFIG.get("video", {}).get("codec", "mp4v")
            processor.create_writer(output_path, codec=codec)

            # Process frames
            processor._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._previous_frames = []

            for frame_idx in tqdm(range(info.total_frames), desc="Removing watermark"):
                ret, frame = processor.read_frame()
                if not ret or frame is None:
                    break

                # Generate mask for this frame
                mask = self.mask_gen.create_mask(frame, detected_region)

                # Remove watermark
                clean_frame = self.remove_from_frame(frame, mask, method)

                # Write output frame
                processor.write_frame(clean_frame)

                # Progress callback
                if progress_callback:
                    progress_callback(frame_idx + 1, info.total_frames)

            logger.info(f"Video saved: {output_path}")
            return output_path

        finally:
            processor.close()

    def remove_batch(
        self,
        frames: list[np.ndarray],
        region: tuple[int, int, int, int],
        method: str | None = None,
    ) -> list[np.ndarray]:
        """
        Remove watermark from a batch of frames.
        
        Args:
            frames: List of input frames.
            region: Watermark region (x, y, w, h).
            method: Inpainting method override.
            
        Returns:
            List of clean frames.
        """
        self._previous_frames = []
        results = []

        for frame in tqdm(frames, desc="Processing frames"):
            mask = self.mask_gen.create_mask(frame, region)
            clean = self.remove_from_frame(frame, mask, method)
            results.append(clean)

        return results

    def _apply_temporal_smoothing(
        self, current: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply temporal smoothing to reduce flickering in the inpainted region.
        Blends the current inpainted result with previous frames' results.
        """
        self._previous_frames.append(current.copy())

        if len(self._previous_frames) > self.temporal_window:
            self._previous_frames.pop(0)

        if len(self._previous_frames) < 2:
            return current

        # Weighted average of recent frames in the mask region
        mask_bool = mask > 0
        if not np.any(mask_bool):
            return current

        result = current.copy().astype(np.float64)
        total_weight = 0.0

        for i, prev_frame in enumerate(self._previous_frames):
            # More recent frames get higher weight
            weight = (i + 1) / len(self._previous_frames)
            result[mask_bool] += prev_frame[mask_bool].astype(np.float64) * weight
            total_weight += weight

        # Add current frame weight
        total_weight += 1.0
        result[mask_bool] /= total_weight

        return result.astype(np.uint8)

    def _deep_inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Deep learning-based inpainting.
        Falls back to TELEA if model is not available.
        """
        try:
            from core.deep_inpainter import DeepInpainter
            inpainter = DeepInpainter()
            return inpainter.inpaint(frame, mask)
        except ImportError:
            logger.warning(
                "Deep inpainting model not available. "
                "Install torch and download model weights. "
                "Falling back to TELEA method."
            )
            return cv2.inpaint(frame, mask, self.radius, cv2.INPAINT_TELEA)

    def preview_detection(
        self,
        frame: np.ndarray,
        region: tuple[int, int, int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Preview detection and mask for debugging.
        
        Args:
            frame: Input frame.
            region: Manual region or None for auto-detect.
            
        Returns:
            Tuple of (frame_with_overlay, mask).
        """
        if region is None:
            region = self.detector.detect(frame)

        if region is None:
            return frame, np.zeros(frame.shape[:2], dtype=np.uint8)

        mask = self.mask_gen.create_mask(frame, region)
        overlay = self.mask_gen.visualize_mask(frame, mask)

        # Draw bounding box
        x, y, w, h = region
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            overlay, "Watermark Region", (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

        return overlay, mask
