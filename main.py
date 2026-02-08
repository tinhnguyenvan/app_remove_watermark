"""
CLI entry point for Sora Watermark Remover.
Run processing from the command line without the web UI.
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

from config import CONFIG
from core.watermark_remover import WatermarkRemover
from utils.file_utils import get_output_path, validate_video_file
from utils.logger import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove watermarks from Sora-generated videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect and remove watermark
  python main.py input.mp4

  # Specify watermark position
  python main.py input.mp4 --position bottom-right

  # Manual region
  python main.py input.mp4 --region 800 600 200 50

  # Use Navier-Stokes inpainting
  python main.py input.mp4 --method ns

  # Custom output path
  python main.py input.mp4 -o output/clean_video.mp4
        """,
    )

    parser.add_argument(
        "input", type=str, help="Path to input video file"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output video path (default: auto-generated)",
    )
    parser.add_argument(
        "--position", type=str, default="auto",
        choices=["auto", "bottom-right", "bottom-left", "bottom-center",
                 "top-right", "top-left"],
        help="Watermark position (default: auto)",
    )
    parser.add_argument(
        "--region", type=int, nargs=4, default=None,
        metavar=("X", "Y", "W", "H"),
        help="Manual watermark region: X Y Width Height",
    )
    parser.add_argument(
        "--method", type=str, default="telea",
        choices=["telea", "ns", "deep"],
        help="Inpainting method (default: telea)",
    )
    parser.add_argument(
        "--radius", type=int, default=5,
        help="Inpainting radius in pixels (default: 5)",
    )
    parser.add_argument(
        "--expansion", type=int, default=10,
        help="Mask expansion in pixels (default: 10)",
    )
    parser.add_argument(
        "--feather", type=int, default=5,
        help="Mask feather in pixels (default: 5)",
    )
    parser.add_argument(
        "--no-temporal", action="store_true",
        help="Disable temporal smoothing",
    )

    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    # Validate input
    if not validate_video_file(args.input):
        logger.error(f"Invalid video file: {args.input}")
        sys.exit(1)

    # Determine output path
    output_path = args.output or get_output_path(
        args.input,
        output_dir=CONFIG.get("output", {}).get("directory", "output"),
        suffix=CONFIG.get("output", {}).get("suffix", "_no_watermark"),
    )

    # Setup remover
    config = {
        "method": args.method,
        "radius": args.radius,
        "mask_expansion": args.expansion,
        "mask_feather": args.feather,
        "temporal_smoothing": not args.no_temporal,
        "temporal_window": 5,
    }
    remover = WatermarkRemover(config=config)

    # Set detector position
    if args.position != "auto":
        remover.detector.position = args.position

    # Determine region
    region = tuple(args.region) if args.region else None

    # Process video
    logger.info(f"Input:  {args.input}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Method: {args.method}")

    try:
        result = remover.remove_from_video(
            input_path=args.input,
            output_path=output_path,
            region=region,
            method=args.method,
        )
        logger.info(f"Done! Output saved to: {result}")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
