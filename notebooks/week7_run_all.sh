#!/usr/bin/env bash
# Week 7 — Run all experiments
# Usage: bash notebooks/week7_run_all.sh
# Run from the project root directory.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
export PYTHONPATH="$ROOT/src:$PYTHONPATH"

echo "========================================================"
echo "  Week 7 — Threshold Sensitivity & Cascade Benchmark"
echo "  $(date)"
echo "========================================================"

echo ""
echo "── Script 1/2: Threshold Sensitivity Analysis ────────────"
"$PYTHON" "$ROOT/notebooks/week7_threshold_sensitivity.py"

echo ""
echo "── Script 2/2: Cascade Architecture Benchmark ────────────"
"$PYTHON" "$ROOT/notebooks/week7_cascade_benchmark.py"

echo ""
echo "========================================================"
echo "  Week 7 computational scripts complete. Output files:"
ls -lh "$ROOT/data/week7/"
echo "========================================================"
