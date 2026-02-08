#!/bin/bash
# ==============================================
# Sora Watermark Remover - Quick Start Script
# ==============================================

set -e

echo "=========================================="
echo "  Sora Watermark Remover"
echo "=========================================="
echo ""

# Check Python version
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "[1/3] Checking Python..."
$PYTHON_CMD --version

# Install dependencies
echo ""
echo "[2/3] Installing dependencies..."
$PYTHON_CMD -m pip install -r requirements.txt --quiet

# Create necessary directories
mkdir -p output logs models temp

# Launch app
echo ""
echo "[3/3] Starting Sora Watermark Remover..."
echo ""
echo "  Web UI: http://localhost:7860"
echo "  Press Ctrl+C to stop"
echo ""

$PYTHON_CMD app.py
