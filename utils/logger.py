"""
Logging configuration module.
"""

import sys
from pathlib import Path

from loguru import logger

from config import CONFIG


def setup_logging():
    """Configure loguru logging based on config settings."""
    log_config = CONFIG.get("logging", {})
    level = log_config.get("level", "INFO")
    log_file = log_config.get("file", "logs/app.log")

    # Remove default handler
    logger.remove()

    # Console handler with color
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        str(log_path),
        level=level,
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        ),
    )

    logger.info("Logging initialized")
