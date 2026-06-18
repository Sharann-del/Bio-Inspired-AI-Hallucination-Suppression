"""
Week 4 — Confidence Scorer: Individual Test

Tests the confidence_scorer module.

Expected behaviour:
  - Factually reasonable text scores higher confidence than obvious nonsense
  - token_log_prob is always negative (log of a probability < 1)
  - attention_entropy is always positive
  - confidence is in [0, 1]

Output: data/week4/tests/confidence_scorer_test.json
"""

import sys, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline.confidence_scorer import score

OUT_DIR = ROOT / "data/week4/tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# pairs: (text, label, expect_higher_confidence_than_next)
TEST_TEXTS = [
    ("Albert Einstein was born in Ulm, Germany in 1879.",
     "factual_correct",         True),
    ("The Eiffel Tower is located in Paris, France.",
     "factual_correct",         True),
    ("The moon orbits the Earth at an average distance of 384,400 km.",
     "factual_correct",         True),
    ("xyzzy fnord blorb wumbo qwerty — complete nonsense.",
     "nonsense",                False),
    ("The moon is made of green cheese and circles Venus once a week.",
     "hallucination",           False),
]

print("=" * 60)
print("  Week 4 — Confidence Scorer Test")
print("=" * 60)
print("  (Loading GPT-2 — may take a moment on first run…)")

results = []
for text, label, _ in TEST_TEXTS:
    s = score(text)
    print(f"\n  [{label}]")
    print(f"  Text       : {text[:70]}")
    print(f"  log_prob   : {s.token_log_prob:.4f}")
    print(f"  attn_ent   : {s.attention_entropy:.4f}")
    print(f"  confidence : {s.confidence:.4f}")
    print(f"  n_tokens   : {s.n_tokens}")
    results.append({
        "text":              text,
        "label":             label,
        "token_log_prob":    s.token_log_prob,
        "attention_entropy": s.attention_entropy,
        "confidence":        s.confidence,
        "n_tokens":          s.n_tokens,
    })

# Sanity checks
for r in results:
    assert r["token_log_prob"] < 0, "log-prob must be negative"
    assert r["attention_entropy"] >= 0, "entropy must be non-negative"
    assert 0.0 <= r["confidence"] <= 1.0, "confidence must be in [0,1]"

# Factual texts should on average have higher confidence than nonsense
factual_conf  = [r["confidence"] for r in results if "factual" in r["label"]]
nonsense_conf = [r["confidence"] for r in results if r["label"] in ("nonsense", "hallucination")]
avg_factual   = sum(factual_conf) / len(factual_conf)
avg_nonsense  = sum(nonsense_conf) / len(nonsense_conf)

print(f"\n  Avg confidence (factual) : {avg_factual:.4f}")
print(f"  Avg confidence (nonsense): {avg_nonsense:.4f}")
print(f"  Ordering correct         : {avg_factual > avg_nonsense}")

out = {
    "avg_confidence_factual":  avg_factual,
    "avg_confidence_nonsense": avg_nonsense,
    "ordering_correct":        avg_factual > avg_nonsense,
    "results":                 results,
}
(OUT_DIR / "confidence_scorer_test.json").write_text(json.dumps(out, indent=2))
print(f"\n  Saved → {OUT_DIR}/confidence_scorer_test.json")

assert all(0 <= r["confidence"] <= 1 for r in results), "Confidence out of range"
print("\n  LAYER 3 — CONFIDENCE SCORER: PASS")
