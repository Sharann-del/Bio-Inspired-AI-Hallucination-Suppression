#!/usr/bin/env bash
# Week 6 — Run all experiments
# Usage: bash notebooks/week6_run_all.sh
# Run from the project root directory.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
export PYTHONPATH="$ROOT/src:$PYTHONPATH"

echo "========================================================"
echo "  Week 6 — Blockchain Integration & RAG Baseline"
echo "  $(date)"
echo "========================================================"

# ── 1. Blockchain benchmark (write times, gas, testnet estimate) ──────────────
echo ""
echo "── Script 1/3: Blockchain Benchmark ──────────────────────"
"$PYTHON" "$ROOT/notebooks/week6_blockchain_benchmark.py"

# ── 2. Audit completeness test ────────────────────────────────────────────────
echo ""
echo "── Script 2/3: Audit Completeness ────────────────────────"
"$PYTHON" "$ROOT/notebooks/week6_audit_completeness.py"

# ── 3. RAG baseline ───────────────────────────────────────────────────────────
echo ""
echo "── Script 3/3: RAG Baseline ──────────────────────────────"
"$PYTHON" "$ROOT/notebooks/week6_rag_baseline.py"

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  Week 6 complete. Output files:"
ls -lh "$ROOT/data/week6/"
echo "========================================================"
