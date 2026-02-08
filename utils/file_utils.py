"""
File utility functions.
Handles file path generation, validation, and management.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

from loguru import logger


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_output_path(
    input_path: str,
    output_dir: str = "output",
    suffix: str = "_no_watermark",
    output_format: str | None = None,
) -> str:
    """
    Generate output file path based on input file.
    
    Args:
        input_path: Path to input video.
        output_dir: Output directory.
        suffix: Filename suffix before extension.
        output_format: Override output format (e.g., "mp4").
        
    Returns:
        Full output file path.
    """
    input_p = Path(input_path)
    ext = f".{output_format}" if output_format else input_p.suffix
    
    output_name = f"{input_p.stem}{suffix}{ext}"
    output_p = ensure_dir(output_dir) / output_name

    # Avoid overwriting existing files
    counter = 1
    while output_p.exists():
        output_name = f"{input_p.stem}{suffix}_{counter}{ext}"
        output_p = output_p.parent / output_name
        counter += 1

    return str(output_p)


def get_temp_dir() -> Path:
    """Get/create temporary directory for processing."""
    temp_dir = Path("temp") / datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(str(temp_dir))


def cleanup_temp(temp_dir: str | Path) -> None:
    """Remove temporary directory and all contents."""
    path = Path(temp_dir)
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
        logger.debug(f"Cleaned up temp dir: {path}")


def get_file_size_mb(path: str) -> float:
    """Get file size in megabytes."""
    return os.path.getsize(path) / (1024 * 1024)


def validate_video_file(path: str) -> bool:
    """Check if file exists and has a valid video extension."""
    valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
    p = Path(path)
    return p.exists() and p.suffix.lower() in valid_extensions
