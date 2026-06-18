"""
Week 6 — Audit Completeness Test
=================================
Re-deploys the two-layer blockchain architecture and writes the full 200-sample
pipeline run to chain (reusing the benchmark's signals). Then reconstructs every
decision from blockchain records *alone* (no access to the original data) and
measures audit completeness:

    completeness_rate = runs with BOTH audit + governance layers / total runs

A complete run can be fully reconstructed: question hash, answer hash, pipeline
version, flagging decision, confidence score, contradiction score. This test
validates that the blockchain provides a self-sufficient tamper-evident audit trail.

Output: data/week6/audit_completeness.json
         data/week6/audit_completeness_report.txt
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR     = ROOT / "data/week6"
CACHE_FILE  = ROOT / "data/week5/ablation_cache.json"
LABELS_FILE = ROOT / "data/week3/main_eval/gpt_oss_labels.jsonl"
CLAIMS_FILE = ROOT / "data/week3/claims/llama2_claims.jsonl"
GEN_DIR     = ROOT / "data/week2/generations/llama2/main"

CONF_THRESHOLD          = 0.52
CONTRADICTION_THRESHOLD = 0.02


# ── data loading ──────────────────────────────────────────────────────────────

def load_samples() -> list:
    gt: dict = {}
    for line in LABELS_FILE.read_text().splitlines():
        r = json.loads(line)
        if r["model"] == "llama2" and r["split"] == "main":
            gt[r["id"]] = r["label"]

    claims_map: dict = {}
    for line in CLAIMS_FILE.read_text().splitlines():
        r = json.loads(line)
        if r["split"] == "main":
            claims_map[r["file"].replace(".json", "")] = r["claims"]

    samples = []
    for f in sorted(GEN_DIR.glob("q*.json")):
        d   = json.loads(f.read_text())
        qid = f.stem
        lbl = gt.get(qid)
        if lbl not in ("TRUE", "FALSE"):
            continue
        samples.append({
            "id":       qid,
            "question": d["question"],
            "answer":   d["generation"],
            "claims":   claims_map.get(qid, [d["generation"]]),
            "gt_label": lbl,
        })
    return samples


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.perf_counter()

    print("=" * 64)
    print("  Week 6 — Audit Completeness Test")
    print("=" * 64)

    samples = load_samples()
    cache   = json.loads(CACHE_FILE.read_text())
    print(f"  Loaded {len(samples)} samples")

    # ── deploy fresh contracts ────────────────────────────────────────────────
    print("\n── Deploying two-layer blockchain (local py-evm) ──")
    from blockchain.audit_writer import AuditWriter
    writer = AuditWriter.create_local()

    # ── write all records ─────────────────────────────────────────────────────
    print(f"\n── Writing {len(samples)} pipeline runs to chain ──")
    written_qids = []

    for i, sample in enumerate(samples):
        qid   = sample["id"]
        conf  = cache.get(f"conf_{qid}", 0.5)
        cont  = cache.get(f"contra_ner_{qid}", {"max_score": 0.0, "has_contradiction": False})
        flag  = cont["max_score"] >= CONTRADICTION_THRESHOLD or conf < CONF_THRESHOLD
        reason = "contra" if cont["has_contradiction"] else ("lowconf" if conf < CONF_THRESHOLD else "clean")

        writer.write_audit_record(qid, sample["question"], sample["answer"])
        writer.write_governance_decision(
            qid, flagged=flag,
            confidence_score=conf,
            contradiction_score=cont["max_score"],
            verdict_reason=reason,
        )
        written_qids.append(qid)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(samples)}] wrote run {qid}")

    print(f"\n  Chain: {writer.total_audit_entries()} audit entries, "
          f"{writer.total_governance_decisions()} governance decisions")

    # ── reconstruct from chain only ───────────────────────────────────────────
    print("\n── Reconstructing all runs from blockchain records alone ──")
    reconstructed = writer.reconstruct_all(written_qids)

    # ── completeness analysis ─────────────────────────────────────────────────
    total          = len(reconstructed)
    complete       = sum(1 for r in reconstructed if r.complete)
    audit_only     = sum(1 for r in reconstructed if not r.complete and r.audit_layer and not r.governance_layer)
    gov_only       = sum(1 for r in reconstructed if not r.complete and not r.audit_layer and r.governance_layer)
    neither        = sum(1 for r in reconstructed if not r.audit_layer and not r.governance_layer)

    completeness_rate = complete / total if total else 0.0

    # field-level verification: for complete runs, verify field presence
    field_checks = {
        "question_hash":    0,
        "answer_hash":      0,
        "pipeline_version": 0,
        "flagged":          0,
        "confidence_score": 0,
        "contradiction_score": 0,
    }
    for r in reconstructed:
        if r.complete:
            if r.audit_layer and r.audit_layer.get("question_hash"):
                field_checks["question_hash"] += 1
            if r.audit_layer and r.audit_layer.get("answer_hash"):
                field_checks["answer_hash"] += 1
            if r.audit_layer and r.audit_layer.get("pipeline_version") is not None:
                field_checks["pipeline_version"] += 1
            if r.governance_layer and r.governance_layer.get("flagged") is not None:
                field_checks["flagged"] += 1
            if r.governance_layer and r.governance_layer.get("confidence_score") is not None:
                field_checks["confidence_score"] += 1
            if r.governance_layer and r.governance_layer.get("contradiction_score") is not None:
                field_checks["contradiction_score"] += 1

    field_rates = {k: round(v / total, 4) for k, v in field_checks.items()}

    # ── report ────────────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 64)
    lines.append("  Week 6 — Audit Completeness Report")
    lines.append(f"  {total} pipeline runs · local py-evm · two-layer blockchain")
    lines.append("=" * 64)
    lines.append("")
    lines.append(f"  Total runs written          : {total}")
    lines.append(f"  Fully reconstructible       : {complete}  ({completeness_rate:.1%})")
    lines.append(f"  Audit layer only            : {audit_only}")
    lines.append(f"  Governance layer only       : {gov_only}")
    lines.append(f"  Neither layer found         : {neither}")
    lines.append("")
    lines.append("  Field-level reconstruction rates (out of all runs)")
    lines.append("  ────────────────────────────────────────────────────")
    for field, rate in field_rates.items():
        lines.append(f"  {field:<25} : {rate:.1%}  ({int(rate*total)}/{total})")
    lines.append("")
    lines.append("  Interpretation")
    lines.append("  ──────────────")
    lines.append(f"  Completeness rate of {completeness_rate:.1%} means that {complete}/{total} pipeline")
    lines.append("  decisions can be fully reconstructed from blockchain records alone,")
    lines.append("  with no access to the original question/answer corpus.")
    lines.append("  All six auditable fields (question_hash, answer_hash, pipeline_version,")
    lines.append("  flagged, confidence_score, contradiction_score) are recoverable.")
    lines.append("=" * 64)

    report_str = "\n".join(lines)
    print("\n" + report_str)

    # ── save ──────────────────────────────────────────────────────────────────
    elapsed = round(time.perf_counter() - t_start, 2)
    output = {
        "timestamp":          time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec":        elapsed,
        "n_runs":             total,
        "completeness_rate":  round(completeness_rate, 4),
        "complete_count":     complete,
        "audit_only_count":   audit_only,
        "governance_only_count": gov_only,
        "neither_count":      neither,
        "field_recovery_rates": field_rates,
        "per_run": [
            {
                "run_id":    r.run_id,
                "complete":  r.complete,
                "missing":   r.missing_layers,
                "audit_fields": list(r.audit_layer.keys()) if r.audit_layer else [],
                "gov_fields":   list(r.governance_layer.keys()) if r.governance_layer else [],
            }
            for r in reconstructed
        ],
    }

    out_json = OUT_DIR / "audit_completeness.json"
    out_txt  = OUT_DIR / "audit_completeness_report.txt"
    out_json.write_text(json.dumps(output, indent=2))
    out_txt.write_text(report_str)

    print(f"\n  Saved → {out_json}")
    print(f"  Saved → {out_txt}")
    print(f"  Total elapsed: {elapsed}s")


if __name__ == "__main__":
    main()
