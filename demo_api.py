#!/usr/bin/env python3
"""
Live cascade hallucination check for a single (question, answer) pair.

Runs: confidence_scorer → ner_router → evidence_retrieval → contradiction_detector
with early-exit gating (week 7 thresholds), then writes to the local blockchain
audit layer (AuditLog + GovernanceDecision).

Usage:
    python demo_api.py "Who wrote Hamlet?" "Shakespeare wrote Hamlet in 1603."
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# Week 7 cascade thresholds (same as notebooks/week7_cascade_benchmark.py)
CONF_EARLY_REJECT = 0.500
CONF_EARLY_ACCEPT = 0.570
CONF_THRESHOLD = 0.520
CONTRA_THRESHOLD = 0.020
MAX_CLAIMS = 4

_HEDGE = re.compile(
    r"^(i (think|believe|feel|would|am not sure)|it (seems|appears|might|may|could)|"
    r"in my opinion|as far as i know|i('m| am) not|sorry|i don't know|i cannot)",
    re.IGNORECASE,
)


@dataclass
class CascadeResult:
    stage: int
    exit_type: str
    flagged: bool
    confidence: float
    max_contra_score: float
    has_evidence: bool
    claims: list[str] = field(default_factory=list)
    stage_log: list[str] = field(default_factory=list)


def _run_id(question: str, answer: str) -> str:
    digest = hashlib.sha256(f"{question}\0{answer}".encode()).hexdigest()
    return f"demo-{digest[:12]}"


def extract_claims(answer: str) -> list[str]:
    """Split answer into verifiable sentences (spaCy, same heuristic as week 3)."""
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess

        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            check=True,
        )
        nlp = spacy.load("en_core_web_sm")

    claims: list[str] = []
    for sent in nlp(answer[:1000]).sents:
        text = sent.text.strip()
        if len(text) < 15 or _HEDGE.match(text) or text.endswith("?"):
            continue
        claims.append(text)
    return claims if claims else [answer[:300]]


def run_cascade(question: str, answer: str) -> CascadeResult:
    from pipeline.confidence_scorer import score as conf_score
    from pipeline.contradiction_detector import detect
    from pipeline.evidence_retrieval import clear_log, retrieve
    from pipeline.ner_router import route

    log: list[str] = []
    clear_log()

    # ── Stage 1: Confidence gate ──────────────────────────────────────────────
    cs = conf_score(answer)
    log.append(
        f"Stage 1 [confidence_scorer]  conf={cs.confidence:.4f}  "
        f"(log_prob={cs.token_log_prob:.4f}, attn_entropy={cs.attention_entropy:.4f})"
    )

    if cs.confidence < CONF_EARLY_REJECT:
        log.append(f"  → EXIT Stage 1 REJECT  (conf < {CONF_EARLY_REJECT})")
        return CascadeResult(
            stage=1, exit_type="REJECT", flagged=True,
            confidence=cs.confidence, max_contra_score=0.0,
            has_evidence=False, stage_log=log,
        )

    if cs.confidence > CONF_EARLY_ACCEPT:
        log.append(f"  → EXIT Stage 1 ACCEPT  (conf > {CONF_EARLY_ACCEPT})")
        return CascadeResult(
            stage=1, exit_type="ACCEPT", flagged=False,
            confidence=cs.confidence, max_contra_score=0.0,
            has_evidence=False, stage_log=log,
        )

    log.append(f"  → PASS to Stage 2  ({CONF_EARLY_REJECT} ≤ conf ≤ {CONF_EARLY_ACCEPT})")

    # ── Stage 2: NER routing + evidence retrieval ─────────────────────────────
    claims = extract_claims(answer)[:MAX_CLAIMS]
    log.append(f"Stage 2 [ner_router + evidence_retrieval]  {len(claims)} claim(s)")

    evidences: list[str] = []
    for i, claim in enumerate(claims, 1):
        rd = route(claim)
        query = rd.primary_query or claim[:80]
        ev = retrieve(query, strategy=rd.strategy)
        evidences.append(ev.evidence)
        log.append(
            f"  claim {i}: strategy={rd.strategy!r}  query={query!r}  "
            f"source={ev.source}  evidence_len={len(ev.evidence)}"
        )

    has_evidence = any(evidences)
    if not has_evidence:
        flagged = cs.confidence < CONF_THRESHOLD
        log.append(
            f"  → EXIT Stage 2 NO_EVIDENCE  "
            f"(conf {'<' if flagged else '≥'} {CONF_THRESHOLD} → "
            f"{'REJECT' if flagged else 'ACCEPT'})"
        )
        return CascadeResult(
            stage=2, exit_type="NO_EVIDENCE", flagged=flagged,
            confidence=cs.confidence, max_contra_score=0.0,
            has_evidence=False, claims=claims, stage_log=log,
        )

    log.append("  → PASS to Stage 3  (evidence found)")

    # ── Stage 3: NLI contradiction gate ───────────────────────────────────────
    max_contra = 0.0
    log.append("Stage 3 [contradiction_detector]")
    for i, (claim, evidence) in enumerate(zip(claims, evidences), 1):
        if not evidence:
            continue
        r = detect(evidence, claim)
        max_contra = max(max_contra, r.contradiction_score)
        log.append(
            f"  claim {i}: label={r.label}  contra_score={r.contradiction_score:.4f}  "
            f"is_contradiction={r.is_contradiction}"
        )

    flagged = max_contra >= CONTRA_THRESHOLD
    exit_type = "REJECT" if flagged else "ACCEPT"
    log.append(
        f"  → EXIT Stage 3 {exit_type}  "
        f"(max_contra {'≥' if flagged else '<'} {CONTRA_THRESHOLD})"
    )

    return CascadeResult(
        stage=3, exit_type=exit_type, flagged=flagged,
        confidence=cs.confidence, max_contra_score=max_contra,
        has_evidence=True, claims=claims, stage_log=log,
    )


def write_blockchain(
    run_id: str, question: str, answer: str, result: CascadeResult
) -> None:
    from blockchain.audit_writer import AuditWriter

    writer = AuditWriter.create_local()

    parts: list[str] = []
    if result.max_contra_score >= CONTRA_THRESHOLD:
        parts.append("contra")
    if result.confidence < CONF_THRESHOLD:
        parts.append("lowconf")
    if result.stage == 1 and result.exit_type == "REJECT":
        parts.append("early_reject")
    if result.stage == 1 and result.exit_type == "ACCEPT":
        parts.append("early_accept")
    if result.stage == 2 and result.exit_type == "NO_EVIDENCE":
        parts.append("no_evidence")
    verdict_reason = "+".join(parts) if parts else "clean"

    r1 = writer.write_audit_record(run_id, question, answer)
    r2 = writer.write_governance_decision(
        run_id,
        flagged=result.flagged,
        confidence_score=result.confidence,
        contradiction_score=result.max_contra_score,
        verdict_reason=verdict_reason,
    )

    print("\n── Blockchain audit ──")
    print(f"  run_id              : {run_id}")
    print(f"  AuditLog (L1)       : {writer.audit_address()}")
    print(f"  Governance (L2)     : {writer.governance_address()}")
    print(f"  L1 tx               : {r1.tx_hash}  ({r1.write_time_ms:.1f} ms, gas={r1.gas_used})")
    print(f"  L2 tx               : {r2.tx_hash}  ({r2.write_time_ms:.1f} ms, gas={r2.gas_used})")

    reconstructed = writer.reconstruct_run(run_id)
    print(f"  audit complete      : {reconstructed.complete}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the live cascade pipeline on one Q/A pair and audit on-chain.",
    )
    parser.add_argument("question", help="Input question text")
    parser.add_argument("answer", help="Model answer text to verify")
    args = parser.parse_args()

    question = args.question.strip()
    answer = args.answer.strip()
    if not question or not answer:
        parser.error("question and answer must be non-empty")

    run_id = _run_id(question, answer)

    print("=" * 60)
    print("  Cascade pipeline — live run (no cache)")
    print("=" * 60)
    print(f"  run_id   : {run_id}")
    print(f"  question : {question[:120]}{'…' if len(question) > 120 else ''}")
    print(f"  answer   : {answer[:120]}{'…' if len(answer) > 120 else ''}")
    print()

    result = run_cascade(question, answer)

    for line in result.stage_log:
        print(line)

    verdict = "HALLUCINATION" if result.flagged else "OK"
    print()
    print("── Final verdict ──")
    print(f"  stage      : {result.stage}")
    print(f"  exit_type  : {result.exit_type}")
    print(f"  flagged    : {result.flagged}  ({verdict})")
    print(f"  confidence : {result.confidence:.4f}")
    print(f"  max_contra : {result.max_contra_score:.4f}")

    write_blockchain(run_id, question, answer, result)


if __name__ == "__main__":
    main()
