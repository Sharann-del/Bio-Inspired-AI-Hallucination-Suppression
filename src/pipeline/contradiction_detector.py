"""
Week 4 — Contradiction Detector (RoBERTa-large-MNLI)

Uses the HuggingFace zero-shot NLI pipeline backed by roberta-large-mnli
to classify each (evidence, claim) pair as:
  - ENTAILMENT    → evidence supports the claim
  - NEUTRAL       → inconclusive
  - CONTRADICTION → evidence contradicts the claim (hallucination signal)

The model runs on CPU (device=-1) to avoid MPS initialisation issues on
Apple Silicon Macs with < 16 GB unified memory.

Public API:
    result  = detect(premise=evidence_text, hypothesis=claim_text)
    results = detect_batch(pairs)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

_pipeline = None   # lazy-loaded

# label strings returned by the MNLI pipeline (order may vary)
_LABEL_MAP = {
    "ENTAILMENT":    "entailment",
    "NEUTRAL":       "neutral",
    "CONTRADICTION": "contradiction",
    # some model checkpoints use lowercase or title-case
    "entailment":    "entailment",
    "neutral":       "neutral",
    "contradiction": "contradiction",
}

# Confidence threshold below which we treat the verdict as uncertain
_CONFIDENCE_THRESHOLD = 0.5


# ── result dataclass ─────────────────────────────────────────────────────────

@dataclass
class ContradictionResult:
    premise: str                    # evidence text
    hypothesis: str                 # claim text
    label: str                      # "entailment" | "neutral" | "contradiction"
    score: float                    # model confidence for the predicted label
    entailment_score: float = 0.0
    neutral_score: float = 0.0
    contradiction_score: float = 0.0
    is_contradiction: bool = False  # True when label=="contradiction" and score≥threshold
    model_used: str = "roberta-large-mnli"


# ── model loader ─────────────────────────────────────────────────────────────

def _load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    from transformers import pipeline
    print("  [contradiction_detector] Loading roberta-large-mnli on CPU …")
    print("  (first run downloads ~1.4 GB — subsequent runs use local cache)")
    _pipeline = pipeline(
        "zero-shot-classification",
        model="roberta-large-mnli",
        device=-1,          # CPU — avoids MPS issues on Apple Silicon
    )
    print("  [contradiction_detector] Model loaded.")
    return _pipeline


# ── core detection ────────────────────────────────────────────────────────────

def detect(premise: str, hypothesis: str) -> ContradictionResult:
    """
    Classify whether *premise* (evidence) contradicts *hypothesis* (claim).

    Uses zero-shot-classification with the three NLI labels so that
    the model scores all three in a single forward pass.
    """
    if not premise.strip():
        # no evidence → cannot determine contradiction
        return ContradictionResult(
            premise=premise, hypothesis=hypothesis,
            label="neutral", score=0.0,
            entailment_score=0.0, neutral_score=1.0, contradiction_score=0.0,
            is_contradiction=False,
        )

    pipe = _load_pipeline()

    # truncate to avoid exceeding the 512-token limit of RoBERTa
    prem_trunc = premise[:700]
    hypo_trunc = hypothesis[:200]

    out = pipe(
        sequences=prem_trunc,
        candidate_labels=["entailment", "neutral", "contradiction"],
        hypothesis_template="This text {}s the statement: " + hypo_trunc,
        multi_label=False,
    )

    # collect all three scores
    score_map: dict[str, float] = {}
    for lbl, sc in zip(out["labels"], out["scores"]):
        norm_lbl = _LABEL_MAP.get(lbl.upper(), lbl.lower())
        score_map[norm_lbl] = round(sc, 4)

    predicted_label = _LABEL_MAP.get(out["labels"][0].upper(), out["labels"][0].lower())
    predicted_score = round(out["scores"][0], 4)

    return ContradictionResult(
        premise=premise,
        hypothesis=hypothesis,
        label=predicted_label,
        score=predicted_score,
        entailment_score=score_map.get("entailment", 0.0),
        neutral_score=score_map.get("neutral", 0.0),
        contradiction_score=score_map.get("contradiction", 0.0),
        is_contradiction=(
            predicted_label == "contradiction"
            and predicted_score >= _CONFIDENCE_THRESHOLD
        ),
    )


def detect_batch(
    pairs: list[tuple[str, str]],
    verbose: bool = False,
) -> list[ContradictionResult]:
    """
    Classify a list of (premise, hypothesis) pairs.
    Loads the model once and processes sequentially.
    """
    _load_pipeline()
    results = []
    for i, (prem, hypo) in enumerate(pairs):
        r = detect(prem, hypo)
        results.append(r)
        if verbose and (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(pairs)}] {r.label} ({r.score:.3f})")
    return results


# ── aggregate helpers ─────────────────────────────────────────────────────────

def aggregate_claim_results(results: list[ContradictionResult]) -> dict:
    """
    Summarise contradiction results across multiple (evidence, claim) pairs
    for a single answer.
    """
    n = len(results)
    if n == 0:
        return {"n_claims": 0, "contradiction_count": 0, "has_contradiction": False,
                "max_contradiction_score": 0.0, "mean_contradiction_score": 0.0}

    contra_scores = [r.contradiction_score for r in results]
    return {
        "n_claims":                n,
        "contradiction_count":     sum(1 for r in results if r.is_contradiction),
        "has_contradiction":       any(r.is_contradiction for r in results),
        "max_contradiction_score": round(max(contra_scores), 4),
        "mean_contradiction_score": round(sum(contra_scores) / n, 4),
        "entailment_count":        sum(1 for r in results if r.label == "entailment"),
        "neutral_count":           sum(1 for r in results if r.label == "neutral"),
    }


if __name__ == "__main__":
    test_pairs = [
        (
            "Albert Einstein was born on 14 March 1879 in Ulm, Kingdom of Württemberg.",
            "Einstein was born in Berlin.",
        ),
        (
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
            "The Eiffel Tower is located in Paris.",
        ),
        (
            "The moon is a natural satellite of Earth, orbiting at an average distance of 384,400 km.",
            "The moon is made of green cheese.",
        ),
        (
            "",  # no evidence
            "Barack Obama was the 44th president of the United States.",
        ),
    ]

    print("Contradiction Detector — smoke test")
    print("=" * 55)
    for prem, hypo in test_pairs:
        r = detect(prem, hypo)
        print(f"\n  Premise   : {prem[:80] or '(empty)'}")
        print(f"  Hypothesis: {hypo[:80]}")
        print(f"  Label     : {r.label}  (score={r.score:.3f})")
        print(f"  Contra?   : {r.is_contradiction}")
        print(f"  Scores    : E={r.entailment_score:.3f}  N={r.neutral_score:.3f}  C={r.contradiction_score:.3f}")
