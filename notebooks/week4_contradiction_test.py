"""
Week 4 — Contradiction Detector: Individual Test

Tests the contradiction_detector module against known (premise, hypothesis) pairs
where the expected label is known.

Expected:
  - Clear factual contradictions detected (e.g. wrong birthplace)
  - Clear entailments detected (paraphrase of the premise)
  - Empty premise → neutral
  - Aggregate helper works correctly

Output: data/week4/tests/contradiction_detector_test.json
"""

import sys, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline.contradiction_detector import detect, detect_batch, aggregate_claim_results

OUT_DIR = ROOT / "data/week4/tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_PAIRS = [
    # (premise, hypothesis, expected_label)
    (
        "Albert Einstein was born on 14 March 1879 in Ulm, Kingdom of Württemberg, Germany.",
        "Einstein was born in Berlin.",
        "contradiction",
    ),
    (
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
        "The Eiffel Tower is located in Paris.",
        "entailment",
    ),
    (
        "The moon is a natural satellite of Earth, orbiting at an average distance of 384,400 km.",
        "The moon is made of green cheese.",
        "contradiction",
    ),
    (
        "Marie Curie was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.",
        "Marie Curie was a scientist who studied radioactivity.",
        "entailment",
    ),
    (
        "",   # no evidence
        "Barack Obama was the 44th president of the United States.",
        "neutral",   # no evidence → neutral by design
    ),
    (
        "Water freezes at 0 degrees Celsius at standard atmospheric pressure.",
        "Water boils at 0 degrees Celsius.",
        "contradiction",
    ),
]

print("=" * 60)
print("  Week 4 — Contradiction Detector Test (RoBERTa-large-MNLI)")
print("=" * 60)
print("  (Loading roberta-large-mnli — may download ~1.4 GB on first run…)")

results = []
correct = 0

for prem, hypo, expected in TEST_PAIRS:
    r = detect(prem, hypo)
    match = r.label == expected
    correct += int(match)
    status = "PASS" if match else "FAIL"
    print(f"\n  [{status}] expected={expected}")
    print(f"  Premise   : {prem[:80] or '(empty)'}")
    print(f"  Hypothesis: {hypo[:80]}")
    print(f"  Label     : {r.label}  (score={r.score:.3f})")
    print(f"  Scores    : E={r.entailment_score:.3f}  N={r.neutral_score:.3f}  C={r.contradiction_score:.3f}")
    results.append({
        "premise":               prem[:200],
        "hypothesis":            hypo,
        "expected_label":        expected,
        "predicted_label":       r.label,
        "match":                 match,
        "score":                 r.score,
        "entailment_score":      r.entailment_score,
        "neutral_score":         r.neutral_score,
        "contradiction_score":   r.contradiction_score,
        "is_contradiction":      r.is_contradiction,
    })

# test aggregate helper
agg = aggregate_claim_results([
    detect(p, h) for p, h, _ in TEST_PAIRS[:4]
])
print(f"\n  Aggregate (first 4 pairs): {agg}")

accuracy = correct / len(TEST_PAIRS)
print(f"\n  Accuracy : {correct}/{len(TEST_PAIRS)} = {accuracy:.0%}")

out = {
    "accuracy": accuracy,
    "correct":  correct,
    "total":    len(TEST_PAIRS),
    "aggregate_example": agg,
    "results":  results,
}
(OUT_DIR / "contradiction_detector_test.json").write_text(json.dumps(out, indent=2))
print(f"\n  Saved → {OUT_DIR}/contradiction_detector_test.json")

assert accuracy >= 0.6, f"Contradiction detection accuracy too low: {accuracy:.0%}"
print("\n  LAYER 4 — CONTRADICTION DETECTION: PASS")
