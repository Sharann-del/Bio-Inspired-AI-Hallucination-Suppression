#!/bin/bash
# Week 2 full pipeline: MC2 eval → generation
# Run from project root:  bash notebooks/week2_run_all.sh

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

echo "=== Week 2 Pipeline ==="
echo "Step 1/2: TruthfulQA MC2 Evaluation"
python3 notebooks/week2_mc2_eval.py

echo ""
echo "Step 2/2: TruthfulQA Generation (250 outputs per model)"
python3 notebooks/week2_generate.py

echo ""
echo "=== Week 2 Complete ==="
echo "Results:"
echo "  Baselines : data/week2/baselines/"
echo "  Generations: data/week2/generations/"
