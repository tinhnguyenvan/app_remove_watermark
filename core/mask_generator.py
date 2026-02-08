"""
Mask generator module.
Creates precise masks for watermark regions to guide inpainting.
"""

import cv2
import numpy as np
from loguru import logger


class MaskGenerator:
    """
    Generates masks for watermark removal.
    Supports multiple mask refinement strategies for better inpainting results.
    """

    def __init__(
        self,
        expansion: int = 10,
        feather: int = 5,
    ):
        """
        Args:
            expansion: Pixels to expand the mask beyond detected region.
            feather: Pixels for mask edge feathering (Gaussian blur).
        """
        self.expansion = expansion
        self.feather = feather

    def create_mask(
        self,
        frame: np.ndarray,
        region: tuple[int, int, int, int],
        refine: bool = True,
    ) -> np.ndarray:
        """
        Create a binary mask for the watermark region.
        
        Args:
            frame: Input frame (BGR).
            region: Watermark region (x, y, w, h).
            refine: Whether to refine mask based on pixel analysis.
            
        Returns:
            Binary mask (uint8) same size as frame. 255 = watermark area.
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        x, y, rw, rh = region

        # Expand region
        x1 = max(0, x - self.expansion)
        y1 = max(0, y - self.expansion)
        x2 = min(w, x + rw + self.expansion)
        y2 = min(h, y + rh + self.expansion)

        # Fill the region
        mask[y1:y2, x1:x2] = 255

        if refine:
            mask = self._refine_mask(frame, mask, (x1, y1, x2, y2))

        # Feather edges for smoother blending
        if self.feather > 0:
            ksize = self.feather * 2 + 1
            mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
            # Re-threshold to keep it mostly binary with soft edges
            _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            # Apply final feathering
            mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

        return mask

    def create_precise_mask(
        self,
        frame: np.ndarray,
        region: tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Create a pixel-precise mask by analyzing watermark pixels.
        Better for semi-transparent watermarks.
        
        Args:
            frame: Input frame (BGR).
            region: Watermark region (x, y, w, h).
            
        Returns:
            Refined mask with per-pixel accuracy.
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        x, y, rw, rh = region
        x1 = max(0, x - self.expansion)
        y1 = max(0, y - self.expansion)
        x2 = min(w, x + rw + self.expansion)
        y2 = min(h, y + rh + self.expansion)

        roi = frame[y1:y2, x1:x2]

        # Convert to multiple color spaces for analysis
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Detect watermark pixels using edge detection
        edges = cv2.Canny(gray, 30, 100)

        # Dilate edges to connect nearby components
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours and filter by area
        contours, _ = cv2.findContours(
            edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        roi_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        min_area = roi.shape[0] * roi.shape[1] * 0.01  # At least 1% of ROI

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                cv2.drawContours(roi_mask, [contour], -1, 255, -1)

        # If contour detection found something, use it; otherwise use full region
        if np.count_nonzero(roi_mask) > 0:
            # Expand the contour mask slightly
            roi_mask = cv2.dilate(roi_mask, kernel, iterations=2)
            mask[y1:y2, x1:x2] = roi_mask
        else:
            mask[y1:y2, x1:x2] = 255

        # Feather
        if self.feather > 0:
            ksize = self.feather * 2 + 1
            mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

        return mask

    def create_mask_from_difference(
        self,
        frame_with_wm: np.ndarray,
        frame_without_wm: np.ndarray,
        threshold: int = 30,
    ) -> np.ndarray:
        """
        Create mask by comparing a frame with and without watermark.
        Useful when you have a clean reference.
        
        Args:
            frame_with_wm: Frame with watermark.
            frame_without_wm: Clean frame (same content, no watermark).
            threshold: Difference threshold.
            
        Returns:
            Binary mask of watermark region.
        """
        diff = cv2.absdiff(frame_with_wm, frame_without_wm)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def _refine_mask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Refine mask using GrabCut-like approach within the bounding box.
        """
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        roi_mask = mask[y1:y2, x1:x2]

        # Use adaptive thresholding on the ROI
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Detect high-contrast areas (likely watermark text/logo)
        local_mean = cv2.blur(gray, (21, 21))
        diff = cv2.absdiff(gray, local_mean)
        _, text_mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

        # Combine with original mask
        combined = cv2.bitwise_or(roi_mask, text_mask)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        mask[y1:y2, x1:x2] = combined
        return mask

    def visualize_mask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        color: tuple[int, int, int] = (0, 0, 255),
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Overlay mask visualization on frame for debugging.
        
        Args:
            frame: Input frame.
            mask: Binary mask.
            color: Overlay color (BGR).
            alpha: Overlay transparency.
            
        Returns:
            Frame with mask visualization overlaid.
        """
        overlay = frame.copy()
        mask_bool = mask > 0

        overlay[mask_bool] = (
            np.array(color) * alpha + overlay[mask_bool] * (1 - alpha)
        ).astype(np.uint8)

        return overlay
