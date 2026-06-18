"""
Week 4 — Evidence Retrieval: Individual Test

Tests the evidence_retrieval module against known queries:
  - Wikidata hit expected for famous entities
  - Fallback expected for obscure/partial queries
  - No-evidence expected for nonsense queries
  - Verifies reliability logging

Output: data/week4/tests/evidence_retrieval_test.json
"""

import sys, json, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline.evidence_retrieval import retrieve, reliability_summary, clear_log, get_log

OUT_DIR = ROOT / "data/week4/tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_CASES = [
    # (query, strategy, expect_source_in)
    ("Albert Einstein",       "entity_lookup",   ["wikidata"]),
    ("Eiffel Tower",          "entity_lookup",   ["wikidata"]),
    ("Marie Curie",           "entity_lookup",   ["wikidata"]),
    ("World War II",          "text_search",     ["wikidata", "wikipedia_page", "wikipedia_search"]),
    ("speed of light",        "structured_fact", ["wikidata", "wikipedia_page", "wikipedia_search"]),
    ("Shakespeare Hamlet",    "text_search",     ["wikidata", "wikipedia_page", "wikipedia_search"]),
    ("xyzzy_fnord_blorb_999", "keyword_search",  ["none"]),
]

print("=" * 60)
print("  Week 4 — Evidence Retrieval Test")
print("=" * 60)

clear_log()
results  = []
correct  = 0

for query, strat, expected_sources in TEST_CASES:
    r = retrieve(query, strategy=strat)
    match = r.source in expected_sources
    correct += int(match)
    status = "PASS" if match else "FAIL"
    print(f"\n  [{status}] Query: {query!r}")
    print(f"       source    : {r.source}  (expected one of: {expected_sources})")
    print(f"       qid       : {r.wikidata_qid}")
    print(f"       fallback  : {r.fallback_used}  no_evidence: {r.no_evidence}")
    print(f"       evidence  : {r.evidence[:100]}{'...' if len(r.evidence) > 100 else ''}")
    print(f"       latency   : {r.latency_ms} ms")
    results.append({
        "query":            query,
        "strategy":         strat,
        "expected_sources": expected_sources,
        "actual_source":    r.source,
        "match":            match,
        "fallback_used":    r.fallback_used,
        "no_evidence":      r.no_evidence,
        "wikidata_qid":     r.wikidata_qid,
        "evidence_snippet": r.evidence[:200],
        "latency_ms":       r.latency_ms,
    })
    time.sleep(0.3)

summary = reliability_summary(get_log())
accuracy = correct / len(TEST_CASES)

print(f"\n  Accuracy : {correct}/{len(TEST_CASES)} = {accuracy:.0%}")
print(f"  Reliability summary:")
for k, v in summary.items():
    print(f"    {k}: {v}")

out = {
    "accuracy":   accuracy,
    "correct":    correct,
    "total":      len(TEST_CASES),
    "reliability_summary": summary,
    "results":    results,
    "full_log":   get_log(),
}
(OUT_DIR / "evidence_retrieval_test.json").write_text(json.dumps(out, indent=2))
print(f"\n  Saved → {OUT_DIR}/evidence_retrieval_test.json")

assert accuracy >= 0.7, f"Evidence retrieval accuracy too low: {accuracy:.0%}"
print("\n  LAYER 2 — EVIDENCE RETRIEVAL: PASS")
