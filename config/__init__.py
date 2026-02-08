"""
Configuration loader for Sora Watermark Remover.
"""

import os
from pathlib import Path

import yaml


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "settings.yaml"


def load_config(config_path: str | None = None) -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path) if config_path else CONFIG_PATH
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def get_default_config() -> dict:
    """Return default configuration dictionary."""
    return {
        "watermark": {
            "position": "auto",
            "text_pattern": "SORA",
            "search_region_percent": 25,
            "confidence_threshold": 0.5,
            "estimated_width_percent": 15,
            "estimated_height_percent": 5,
        },
        "video": {
            "codec": "mp4v",
            "quality": 95,
            "keep_fps": True,
            "keep_resolution": True,
        },
        "inpainting": {
            "method": "telea",
            "radius": 5,
            "mask_expansion": 10,
            "mask_feather": 5,
            "temporal_smoothing": True,
            "temporal_window": 5,
        },
        "output": {
            "directory": "output",
            "suffix": "_no_watermark",
            "format": "mp4",
        },
        "logging": {
            "level": "INFO",
            "file": "logs/app.log",
        },
    }


# Global config instance
try:
    CONFIG = load_config()
except FileNotFoundError:
    CONFIG = get_default_config()
