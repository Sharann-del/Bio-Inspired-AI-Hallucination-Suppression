#!/usr/bin/env bash
# Week 4 — Run All Tests + Wikidata Diagnostic
# Usage:  bash notebooks/week4_run_all.sh
# Run from the repo root directory.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"

echo "======================================================"
echo "  Week 4 — Bio-Inspired AI Hallucination Suppression"
echo "  Running all pipeline layer tests + diagnostic"
echo "======================================================"
echo ""

run() {
    local label="$1"
    local script="$2"
    echo "------------------------------------------------------"
    echo "  $label"
    echo "------------------------------------------------------"
    $PYTHON "$script"
    echo ""
}

# Layer tests (fast — no large model downloads except layers 3 & 4)
run "LAYER 1: NER + Routing"         "notebooks/week4_ner_routing_test.py"
run "LAYER 2: Evidence Retrieval"    "notebooks/week4_evidence_test.py"
run "LAYER 3: Confidence Scorer"     "notebooks/week4_confidence_test.py"
run "LAYER 4: Contradiction Detect"  "notebooks/week4_contradiction_test.py"

# Wikidata reliability diagnostic (200 queries, ~2 min with rate limiting)
run "WIKIDATA DIAGNOSTIC (200 queries)" "notebooks/week4_wikidata_diagnostic.py"

echo "======================================================"
echo "  Week 4 — ALL DONE"
echo "======================================================"
echo ""
echo "  Outputs:"
echo "    data/week4/tests/ner_routing_test.json"
echo "    data/week4/tests/evidence_retrieval_test.json"
echo "    data/week4/tests/confidence_scorer_test.json"
echo "    data/week4/tests/contradiction_detector_test.json"
echo "    data/week4/wikidata_reliability_report.json"
echo "    data/week4/wikidata_diagnostic_log.jsonl"
