"""
Video processor module.
Handles video I/O operations: reading, writing, frame extraction, and reconstruction.
"""

from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm


@dataclass
class VideoInfo:
    """Stores video metadata."""
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    codec: str
    filepath: str


class VideoProcessor:
    """
    Handles video reading, writing, and frame-level processing.
    """

    def __init__(self):
        self._cap: cv2.VideoCapture | None = None
        self._writer: cv2.VideoWriter | None = None
        self._info: VideoInfo | None = None

    @property
    def info(self) -> VideoInfo | None:
        return self._info

    def open(self, video_path: str) -> VideoInfo:
        """
        Open a video file and extract metadata.
        
        Args:
            video_path: Path to the input video file.
            
        Returns:
            VideoInfo with video metadata.
            
        Raises:
            FileNotFoundError: If video file doesn't exist.
            RuntimeError: If video cannot be opened.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self._cap = cv2.VideoCapture(str(path))

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        self._info = VideoInfo(
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration=total_frames / fps if fps > 0 else 0,
            codec=codec,
            filepath=str(path),
        )

        logger.info(
            f"Opened video: {path.name} | "
            f"{width}x{height} | {fps:.1f}fps | "
            f"{total_frames} frames | {self._info.duration:.1f}s"
        )

        return self._info

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """
        Read the next frame from the video.
        
        Returns:
            Tuple of (success, frame). Frame is None if reading fails.
        """
        if self._cap is None:
            raise RuntimeError("No video opened. Call open() first.")

        ret, frame = self._cap.read()
        return ret, frame if ret else (False, None)

    def read_all_frames(self, max_frames: int | None = None) -> list[np.ndarray]:
        """
        Read all frames from the video into memory.
        
        Args:
            max_frames: Maximum number of frames to read. None for all.
            
        Returns:
            List of frames as numpy arrays.
        """
        if self._cap is None:
            raise RuntimeError("No video opened. Call open() first.")

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames = []
        total = max_frames or self._info.total_frames

        for _ in tqdm(range(total), desc="Reading frames"):
            ret, frame = self._cap.read()
            if not ret:
                break
            frames.append(frame)

        logger.info(f"Read {len(frames)} frames")
        return frames

    def create_writer(
        self,
        output_path: str,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        codec: str = "mp4v",
    ) -> None:
        """
        Create a video writer for output.
        
        Args:
            output_path: Path for the output video file.
            width: Output width (default: same as input).
            height: Output height (default: same as input).
            fps: Output FPS (default: same as input).
            codec: FourCC codec string.
        """
        if self._info is None:
            raise RuntimeError("No video info available. Open a video first.")

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        w = width or self._info.width
        h = height or self._info.height
        f = fps or self._info.fps
        fourcc = cv2.VideoWriter_fourcc(*codec)

        self._writer = cv2.VideoWriter(str(out_path), fourcc, f, (w, h))

        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot create video writer: {output_path}")

        logger.info(f"Writer created: {out_path.name} | {w}x{h} | {f:.1f}fps")

    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single frame to the output video."""
        if self._writer is None:
            raise RuntimeError("No writer created. Call create_writer() first.")
        self._writer.write(frame)

    def write_frames(self, frames: list[np.ndarray]) -> None:
        """Write multiple frames to the output video."""
        for frame in tqdm(frames, desc="Writing frames"):
            self.write_frame(frame)

    def get_frame_at(self, frame_index: int) -> np.ndarray | None:
        """
        Get a specific frame by index.
        
        Args:
            frame_index: Zero-based frame index.
            
        Returns:
            Frame as numpy array, or None if invalid index.
        """
        if self._cap is None:
            raise RuntimeError("No video opened. Call open() first.")

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self._cap.read()
        return frame if ret else None

    def close(self) -> None:
        """Release video capture and writer resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        if self._writer is not None:
            self._writer.release()
            self._writer = None

        logger.info("Video resources released")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
