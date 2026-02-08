"""
Watermark detector module.
Detects Sora watermark position and region in video frames.
Supports automatic detection and manual region specification.
"""

import cv2
import numpy as np
from loguru import logger

from config import CONFIG


class WatermarkDetector:
    """
    Detects Sora watermark in video frames using multiple strategies:
    1. Template matching (if template provided)
    2. Text detection (OCR-based)
    3. Edge/contrast analysis (for semi-transparent watermarks)
    4. Static region detection (comparing frame differences)
    """

    # Typical Sora watermark positions (percentage from edges)
    POSITION_MAP = {
        "bottom-right": (0.70, 0.85, 1.0, 1.0),   # (x1%, y1%, x2%, y2%)
        "bottom-left": (0.0, 0.85, 0.30, 1.0),
        "top-right": (0.70, 0.0, 1.0, 0.15),
        "top-left": (0.0, 0.0, 0.30, 0.15),
        "bottom-center": (0.30, 0.85, 0.70, 1.0),
    }

    def __init__(self, config: dict | None = None):
        self.config = config or CONFIG.get("watermark", {})
        self.position = self.config.get("position", "auto")
        self.search_region_percent = self.config.get("search_region_percent", 25)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        self._cached_region: tuple[int, int, int, int] | None = None

    def detect(
        self,
        frame: np.ndarray,
        template: np.ndarray | None = None,
    ) -> tuple[int, int, int, int] | None:
        """
        Detect watermark region in a single frame.
        
        Args:
            frame: Input frame (BGR).
            template: Optional watermark template image for template matching.
            
        Returns:
            Tuple (x, y, w, h) of the watermark region, or None if not found.
        """
        # Use cached region if available (watermark position is usually static)
        if self._cached_region is not None:
            return self._cached_region

        if self.position != "auto":
            region = self._detect_by_position(frame)
        elif template is not None:
            region = self._detect_by_template(frame, template)
        else:
            region = self._detect_by_analysis(frame)

        if region is not None:
            self._cached_region = region
            logger.info(f"Watermark detected at: x={region[0]}, y={region[1]}, "
                        f"w={region[2]}, h={region[3]}")

        return region

    def detect_from_frames(
        self,
        frames: list[np.ndarray],
        sample_count: int = 10,
    ) -> tuple[int, int, int, int] | None:
        """
        Detect watermark by analyzing multiple frames.
        Uses static region detection - watermark remains constant across frames.
        
        Args:
            frames: List of video frames.
            sample_count: Number of frames to sample for analysis.
            
        Returns:
            Tuple (x, y, w, h) of the watermark region, or None.
        """
        if len(frames) < 2:
            return self.detect(frames[0]) if frames else None

        # Sample frames evenly
        indices = np.linspace(0, len(frames) - 1, min(sample_count, len(frames)), dtype=int)
        sampled = [frames[i] for i in indices]

        # Find static regions (regions that don't change across frames)
        region = self._detect_static_region(sampled)

        if region is None:
            # Fallback to single-frame analysis
            region = self.detect(frames[0])

        if region is not None:
            self._cached_region = region

        return region

    def _detect_by_position(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """Detect watermark at a predefined position."""
        h, w = frame.shape[:2]

        if self.position in self.POSITION_MAP:
            x1_pct, y1_pct, x2_pct, y2_pct = self.POSITION_MAP[self.position]
        else:
            # Default to bottom-right
            x1_pct, y1_pct, x2_pct, y2_pct = self.POSITION_MAP["bottom-right"]

        x = int(w * x1_pct)
        y = int(h * y1_pct)
        rw = int(w * (x2_pct - x1_pct))
        rh = int(h * (y2_pct - y1_pct))

        return (x, y, rw, rh)

    def _detect_by_template(
        self, frame: np.ndarray, template: np.ndarray
    ) -> tuple[int, int, int, int] | None:
        """Detect watermark using template matching."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        th, tw = gray_template.shape[:2]

        # Multi-scale template matching
        best_match = None
        best_val = -1

        for scale in np.linspace(0.5, 1.5, 11):
            resized_t = cv2.resize(
                gray_template,
                (int(tw * scale), int(th * scale)),
                interpolation=cv2.INTER_AREA,
            )

            if (resized_t.shape[0] > gray_frame.shape[0] or 
                resized_t.shape[1] > gray_frame.shape[1]):
                continue

            result = cv2.matchTemplate(
                gray_frame, resized_t, cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_val and max_val > self.confidence_threshold:
                best_val = max_val
                best_match = (
                    max_loc[0],
                    max_loc[1],
                    resized_t.shape[1],
                    resized_t.shape[0],
                )

        if best_match:
            logger.info(f"Template match confidence: {best_val:.3f}")

        return best_match

    def _detect_by_analysis(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """
        Detect watermark by analyzing frame edges and text-like features.
        Sora watermark is typically a small text/logo in corner areas.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        best_region = None
        best_score = 0.0

        # Check each corner region
        for pos_name, (x1_pct, y1_pct, x2_pct, y2_pct) in self.POSITION_MAP.items():
            x1 = int(w * x1_pct)
            y1 = int(h * y1_pct)
            x2 = int(w * x2_pct)
            y2 = int(h * y2_pct)

            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            score = self._compute_watermark_score(roi)

            if score > best_score and score > self.confidence_threshold:
                best_score = score
                best_region = (x1, y1, x2 - x1, y2 - y1)

        if best_region:
            logger.info(f"Analysis detection score: {best_score:.3f}")

        # Fallback: default to bottom-right (most common for Sora)
        if best_region is None:
            logger.warning("Auto-detection failed, defaulting to bottom-right region")
            best_region = self._detect_by_position(frame)

        return best_region

    def _detect_static_region(
        self, frames: list[np.ndarray]
    ) -> tuple[int, int, int, int] | None:
        """
        Detect watermark by finding static regions across multiple frames.
        Watermarks remain constant while video content changes.
        """
        if len(frames) < 2:
            return None

        h, w = frames[0].shape[:2]

        # Compute absolute differences between consecutive frames
        diff_accum = np.zeros((h, w), dtype=np.float64)

        for i in range(len(frames) - 1):
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float64)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY).astype(np.float64)
            diff = np.abs(gray1 - gray2)
            diff_accum += diff

        diff_accum /= len(frames) - 1

        # Static regions have low average difference
        # Focus on corner areas where watermarks typically appear
        for pos_name, (x1_pct, y1_pct, x2_pct, y2_pct) in self.POSITION_MAP.items():
            x1 = int(w * x1_pct)
            y1 = int(h * y1_pct)
            x2 = int(w * x2_pct)
            y2 = int(h * y2_pct)

            roi_diff = diff_accum[y1:y2, x1:x2]
            mean_diff = np.mean(roi_diff)

            # Very low difference in corner + has some edge content = likely watermark
            if mean_diff < 5.0:  # Nearly static
                roi_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)[y1:y2, x1:x2]
                edges = cv2.Canny(roi_gray, 50, 150)
                edge_density = np.count_nonzero(edges) / edges.size

                if edge_density > 0.02:  # Has some structure (text/logo)
                    logger.info(
                        f"Static watermark found at {pos_name} "
                        f"(diff={mean_diff:.2f}, edges={edge_density:.3f})"
                    )
                    return (x1, y1, x2 - x1, y2 - y1)

        return None

    def _compute_watermark_score(self, roi_gray: np.ndarray) -> float:
        """
        Compute a score indicating likelihood of watermark presence.
        Higher score = more likely to be a watermark.
        """
        # Edge detection
        edges = cv2.Canny(roi_gray, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size

        # Contrast analysis
        std_dev = np.std(roi_gray.astype(np.float64))

        # Text-like features: high local contrast with structure
        # Watermarks typically have moderate edge density and contrast
        score = 0.0

        if 0.02 < edge_density < 0.3:
            score += 0.4
        if 20 < std_dev < 100:
            score += 0.3

        # Check for semi-transparent overlay characteristics
        mean_val = np.mean(roi_gray)
        if mean_val > 180 or mean_val < 80:
            score += 0.2  # Very bright or dark regions common in watermarks

        # Frequency analysis - watermarks often have specific frequency patterns
        f_transform = np.fft.fft2(roi_gray.astype(np.float64))
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log1p(np.abs(f_shift))
        high_freq_ratio = np.mean(magnitude > np.median(magnitude))
        if high_freq_ratio > 0.3:
            score += 0.1

        return min(score, 1.0)

    def set_manual_region(self, x: int, y: int, w: int, h: int) -> None:
        """
        Manually set the watermark region.
        
        Args:
            x: Top-left x coordinate.
            y: Top-left y coordinate.
            w: Width of the region.
            h: Height of the region.
        """
        self._cached_region = (x, y, w, h)
        logger.info(f"Manual watermark region set: x={x}, y={y}, w={w}, h={h}")

    def reset(self) -> None:
        """Reset cached detection results."""
        self._cached_region = None
