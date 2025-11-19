#!/bin/bash
# Quick script to check checkpoint status

CKPT_DIR="${1:-./ckpt}"

echo "=== Checkpoint Directory Status ==="
echo "Directory: $CKPT_DIR"
echo ""

if [ ! -d "$CKPT_DIR" ]; then
    echo "❌ Directory does not exist!"
    echo "   The directory will be created when training starts."
    exit 1
fi

echo "✓ Directory exists"
echo ""

echo "=== Files in checkpoint directory ==="
ls -lht "$CKPT_DIR" 2>/dev/null | head -20

echo ""
echo "=== Checkpoint count ==="
CKPT_COUNT=$(ls "$CKPT_DIR"/*.ckpt 2>/dev/null | wc -l)
echo "Total .ckpt files: $CKPT_COUNT"

if [ $CKPT_COUNT -eq 0 ]; then
    echo ""
    echo "⚠️  No checkpoints found yet."
    echo "   If training is running, checkpoints will appear after the first epoch completes."
    echo "   Check that:"
    echo "   1. Training has completed at least 1 epoch"
    echo "   2. You passed the correct --saving_path to train.py"
    echo "   3. Training hasn't crashed"
else
    echo ""
    echo "✓ Checkpoints found!"
    echo ""
    echo "=== Most recent checkpoints ==="
    ls -lt "$CKPT_DIR"/*.ckpt 2>/dev/null | head -5
fi

echo ""
echo "=== Config file ==="
if [ -f "$CKPT_DIR/config.json" ]; then
    echo "✓ config.json exists"
    echo "   save_every_n_epochs: $(cat "$CKPT_DIR/config.json" | grep -o '"save_every_n_epochs": [0-9]*' | awk '{print $2}')"
else
    echo "❌ config.json not found (will be created when training starts)"
fi

echo ""
echo "=== Disk usage ==="
du -sh "$CKPT_DIR" 2>/dev/null || echo "N/A"

