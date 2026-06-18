"""
Week 4 — NER + Routing Layer: Individual Test

Tests the ner_router module against:
  - 10 hand-crafted claims covering all 4 routing strategies
  - strategy distribution check
  - batch routing

Output: data/week4/tests/ner_routing_test.json
"""

import sys, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline.ner_router import route, route_batch, strategy_summary

OUT_DIR = ROOT / "data/week4/tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_CLAIMS = [
    # entity_lookup expected
    ("Albert Einstein was born in Ulm, Germany in 1879.",             "entity_lookup"),
    ("Marie Curie received two Nobel Prizes.",                        "entity_lookup"),
    ("The Eiffel Tower stands in Paris, France.",                     "entity_lookup"),
    ("NASA was founded in 1958 by the United States government.",     "entity_lookup"),
    # structured_fact expected
    ("The speed of light is approximately 299,792 km per second.",    "structured_fact"),
    ("World War II ended in 1945.",                                   "structured_fact"),
    # text_search expected
    ("Hamlet was written by Shakespeare.",                            "text_search"),
    ("The Magna Carta was signed in 1215.",                           "text_search"),
    # keyword_search expected (no named entities)
    ("Poverty has declined significantly in recent decades.",         "keyword_search"),
    ("This answer is truthful and concise.",                          "keyword_search"),
]

print("=" * 60)
print("  Week 4 — NER + Routing Layer Test")
print("=" * 60)

claims   = [c for c, _ in TEST_CLAIMS]
expected = [e for _, e in TEST_CLAIMS]

decisions = route_batch(claims)

results  = []
correct  = 0

for d, exp in zip(decisions, expected):
    match = d.strategy == exp
    correct += int(match)
    ents = [(e.text, e.label) for e in d.entities]
    status = "PASS" if match else "FAIL"
    print(f"\n  [{status}] {d.claim[:60]}")
    print(f"       entities : {ents}")
    print(f"       strategy : {d.strategy}  (expected: {exp})")
    print(f"       query    : {d.primary_query}")
    results.append({
        "claim":    d.claim,
        "entities": ents,
        "strategy": d.strategy,
        "expected": exp,
        "match":    match,
        "primary_query": d.primary_query,
        "keywords": d.keywords[:5],
    })

summary = strategy_summary(decisions)
accuracy = correct / len(TEST_CLAIMS)

print(f"\n  Accuracy : {correct}/{len(TEST_CLAIMS)} = {accuracy:.0%}")
print(f"  Strategy distribution: {summary}")

out = {
    "accuracy": accuracy,
    "correct":  correct,
    "total":    len(TEST_CLAIMS),
    "strategy_distribution": summary,
    "results":  results,
}
(OUT_DIR / "ner_routing_test.json").write_text(json.dumps(out, indent=2))
print(f"\n  Saved → {OUT_DIR}/ner_routing_test.json")

assert accuracy >= 0.7, f"NER routing accuracy too low: {accuracy:.0%}"
print("\n  LAYER 1 — NER + ROUTING: PASS")
