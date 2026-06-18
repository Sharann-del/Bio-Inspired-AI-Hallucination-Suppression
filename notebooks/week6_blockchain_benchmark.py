"""
Week 6 — Blockchain Integration Benchmark
==========================================
Deploys the two-layer blockchain architecture on a local in-process EVM
(py-evm via eth-tester) and runs the full pipeline on all 200 evaluation
samples, measuring:

  1. Pipeline stage latencies — NER, evidence, contradiction, confidence
     (signals loaded from week-5 cache; no re-inference needed)
  2. Blockchain write times — Layer 1 (AuditLog) and Layer 2 (GovernanceDecision)
     per sample on local node
  3. Testnet estimate — derives expected Sepolia confirmation time from
     gas consumed and current reference block times

Output: data/week6/blockchain_benchmark.json
         data/week6/blockchain_benchmark_summary.txt
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import mean, median, quantiles

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "data/week6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILE     = ROOT / "data/week5/ablation_cache.json"
LABELS_FILE    = ROOT / "data/week3/main_eval/gpt_oss_labels.jsonl"
CLAIMS_FILE    = ROOT / "data/week3/claims/llama2_claims.jsonl"
GEN_DIR        = ROOT / "data/week2/generations/llama2/main"

# C4 thresholds (from week 5)
CONF_THRESHOLD          = 0.52
CONTRADICTION_THRESHOLD = 0.02

# Sepolia reference: 12s avg block time, gas price reference
SEPOLIA_BLOCK_TIME_S    = 12.0
SEPOLIA_GAS_PRICE_GWEI  = 1.5   # conservative low-priority estimate


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
        d    = json.loads(f.read_text())
        qid  = f.stem
        lbl  = gt.get(qid)
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


def load_cache() -> dict:
    return json.loads(CACHE_FILE.read_text())


# ── pipeline signal lookup (from week-5 cache) ────────────────────────────────

def get_pipeline_signals(qid: str, cache: dict) -> dict:
    """
    Pull precomputed signals from week-5 ablation cache.
    Returns timing-annotated dict with all four pipeline stage outputs.
    """
    conf_score     = cache.get(f"conf_{qid}", 0.5)
    contra_result  = cache.get(f"contra_ner_{qid}", {"has_contradiction": False, "max_score": 0.0})
    ev_results     = cache.get(f"ev_ner_{qid}", [])

    flagged = (
        contra_result["max_score"] >= CONTRADICTION_THRESHOLD
        or conf_score < CONF_THRESHOLD
    )

    return {
        "confidence_score":    conf_score,
        "contradiction_score": contra_result["max_score"],
        "has_contradiction":   contra_result["has_contradiction"],
        "flagged":             flagged,
        "n_evidence_found":    sum(1 for e in ev_results if e),
    }


# ── stage timing simulation ───────────────────────────────────────────────────
# Week-4/5 measured timings (from the evidence retrieval log and model load times)
# reproduced here as reference latencies for the per-sample breakdown table.

_STAGE_LATENCY_MS = {
    "ner":           3.1,    # spaCy en_core_web_sm per claim
    "evidence":      612.0,  # Wikidata/Wikipedia per query (avg from week-4 log)
    "contradiction": 890.0,  # RoBERTa-large-MNLI per (evidence, claim) pair
    "confidence":    210.0,  # GPT-2 log-prob + attn entropy per answer
}


def simulate_stage_latency(sample: dict, cache: dict) -> dict:
    """
    Return per-stage latency for one sample, derived from:
    - fixed per-claim NER overhead
    - number of claims × evidence retrieval latency
    - number of claims with evidence × contradiction latency
    - fixed confidence scoring overhead
    """
    n_claims  = len(sample["claims"][:4])
    ev_list   = cache.get(f"ev_ner_{sample['id']}", [])
    n_with_ev = sum(1 for e in ev_list if e)

    ner_ms   = n_claims * _STAGE_LATENCY_MS["ner"]
    ev_ms    = n_claims * _STAGE_LATENCY_MS["evidence"]
    contra_ms = n_with_ev * _STAGE_LATENCY_MS["contradiction"]
    conf_ms   = _STAGE_LATENCY_MS["confidence"]

    return {
        "ner_ms":            round(ner_ms, 1),
        "evidence_ms":       round(ev_ms, 1),
        "contradiction_ms":  round(contra_ms, 1),
        "confidence_ms":     round(conf_ms, 1),
        "total_pipeline_ms": round(ner_ms + ev_ms + contra_ms + conf_ms, 1),
    }


# ── testnet estimate ──────────────────────────────────────────────────────────

def estimate_testnet_latency(gas_used_l1: float, gas_used_l2: float) -> dict:
    """
    Estimate expected Sepolia write latency from gas measurements.
    Assumes 1-block confirmation target (12s avg block time on Sepolia/mainnet).
    """
    total_gas = gas_used_l1 + gas_used_l2
    tx_fee_gwei = total_gas * SEPOLIA_GAS_PRICE_GWEI
    return {
        "total_gas_per_run":         total_gas,
        "estimated_fee_gwei":        round(tx_fee_gwei, 2),
        "estimated_fee_eth":         round(tx_fee_gwei / 1e9, 8),
        "expected_confirmation_s":   SEPOLIA_BLOCK_TIME_S,
        "note": (
            "Sepolia testnet not benchmarked live — requires funded wallet. "
            f"Expected confirmation: 1 block ≈ {SEPOLIA_BLOCK_TIME_S}s. "
            "Gas estimates derived from local py-evm measurements."
        ),
    }


# ── main benchmark ────────────────────────────────────────────────────────────

def main():
    t_start = time.perf_counter()

    print("=" * 64)
    print("  Week 6 — Blockchain Integration Benchmark")
    print("=" * 64)

    # ── load data ─────────────────────────────────────────────────────────────
    print("\n── Loading eval set and week-5 cache ──")
    samples = load_samples()
    cache   = load_cache()
    print(f"  Loaded {len(samples)} samples")

    # ── deploy contracts ──────────────────────────────────────────────────────
    print("\n── Deploying two-layer blockchain (local py-evm) ──")
    from blockchain.audit_writer import AuditWriter
    writer = AuditWriter.create_local()
    print(f"  Layer 1 (AuditLog)          : {writer.audit_address()}")
    print(f"  Layer 2 (GovernanceDecision): {writer.governance_address()}")

    # ── benchmark loop ────────────────────────────────────────────────────────
    print(f"\n── Benchmarking {len(samples)} samples ──")
    records = []

    for i, sample in enumerate(samples):
        qid = sample["id"]

        # pipeline signals (pre-computed, week 5 cache)
        signals    = get_pipeline_signals(qid, cache)
        stage_time = simulate_stage_latency(sample, cache)

        # Layer 1 write
        r1 = writer.write_audit_record(
            qid, sample["question"], sample["answer"]
        )

        # Layer 2 write
        verdict = ("contra" if signals["has_contradiction"] else "") + \
                  ("lowconf" if signals["confidence_score"] < CONF_THRESHOLD else "")
        r2 = writer.write_governance_decision(
            qid,
            flagged=signals["flagged"],
            confidence_score=signals["confidence_score"],
            contradiction_score=signals["contradiction_score"],
            verdict_reason=verdict or "clean",
        )

        records.append({
            "qid":           qid,
            "gt_label":      sample["gt_label"],
            "flagged":       signals["flagged"],
            "signals":       signals,
            "stage_time_ms": stage_time,
            "blockchain": {
                "l1_write_ms":    r1.write_time_ms,
                "l1_gas":         r1.gas_used,
                "l1_block":       r1.block_number,
                "l2_write_ms":    r2.write_time_ms,
                "l2_gas":         r2.gas_used,
                "l2_block":       r2.block_number,
                "total_write_ms": round(r1.write_time_ms + r2.write_time_ms, 3),
            },
        })

        if (i + 1) % 50 == 0 or i == 0:
            total_w = r1.write_time_ms + r2.write_time_ms
            print(f"  [{i+1:3d}/{len(samples)}]  {qid}  "
                  f"L1={r1.write_time_ms:.1f}ms  L2={r2.write_time_ms:.1f}ms  "
                  f"total_write={total_w:.1f}ms  pipeline={stage_time['total_pipeline_ms']:.0f}ms")

    print(f"\n  Chain state: {writer.total_audit_entries()} audit entries, "
          f"{writer.total_governance_decisions()} governance decisions")

    # ── aggregate stats ───────────────────────────────────────────────────────
    l1_times  = [r["blockchain"]["l1_write_ms"]    for r in records]
    l2_times  = [r["blockchain"]["l2_write_ms"]    for r in records]
    tot_times = [r["blockchain"]["total_write_ms"] for r in records]
    pipe_ms   = [r["stage_time_ms"]["total_pipeline_ms"] for r in records]
    l1_gas    = [r["blockchain"]["l1_gas"] for r in records]
    l2_gas    = [r["blockchain"]["l2_gas"] for r in records]

    def stats(xs: list) -> dict:
        qs = quantiles(xs, n=100)
        return {
            "mean":  round(mean(xs), 3),
            "p50":   round(median(xs), 3),
            "p95":   round(qs[94], 3),
            "min":   round(min(xs), 3),
            "max":   round(max(xs), 3),
        }

    blockchain_stats = {
        "l1_write_ms":    stats(l1_times),
        "l2_write_ms":    stats(l2_times),
        "total_write_ms": stats(tot_times),
        "l1_gas":         stats(l1_gas),
        "l2_gas":         stats(l2_gas),
    }
    pipeline_stats = {
        "total_pipeline_ms": stats(pipe_ms),
        "stage_breakdown_ms": _STAGE_LATENCY_MS,
    }

    testnet_est = estimate_testnet_latency(
        mean(l1_gas), mean(l2_gas)
    )

    # ── print summary table ───────────────────────────────────────────────────
    lines = []
    lines.append("=" * 70)
    lines.append("  Week 6 — Blockchain Benchmark Summary")
    lines.append(f"  {len(samples)} samples  ·  two-layer architecture  ·  local py-evm")
    lines.append("=" * 70)
    lines.append("")
    lines.append("  Blockchain Write Times (local in-process EVM)")
    lines.append("  ─────────────────────────────────────────────")
    for label, key in [("Layer 1 (AuditLog)     ", "l1_write_ms"),
                        ("Layer 2 (GovernanceDec)", "l2_write_ms"),
                        ("Total (L1 + L2)        ", "total_write_ms")]:
        s = blockchain_stats[key]
        lines.append(f"  {label}  mean={s['mean']:.1f}ms  "
                     f"p50={s['p50']:.1f}ms  p95={s['p95']:.1f}ms")
    lines.append("")
    lines.append("  Gas Consumption (per run)")
    lines.append("  ──────────────────────────")
    for label, key in [("Layer 1 gas", "l1_gas"), ("Layer 2 gas", "l2_gas")]:
        s = blockchain_stats[key]
        lines.append(f"  {label:<14}  mean={s['mean']:.0f}  "
                     f"p50={s['p50']:.0f}  p95={s['p95']:.0f}")
    lines.append("")
    lines.append("  Pipeline Stage Latencies (reference, from week-4/5 measurements)")
    lines.append("  ──────────────────────────────────────────────────────────────────")
    for stage, ms in _STAGE_LATENCY_MS.items():
        lines.append(f"  {stage:<15}  {ms:.1f} ms / claim")
    s = pipeline_stats["total_pipeline_ms"]
    lines.append(f"  {'total (avg)':<15}  mean={s['mean']:.0f}ms  "
                 f"p50={s['p50']:.0f}ms  p95={s['p95']:.0f}ms")
    lines.append("")
    lines.append("  Testnet Estimate (Sepolia)")
    lines.append("  ─────────────────────────")
    lines.append(f"  Total gas / run      : {testnet_est['total_gas_per_run']:.0f}")
    lines.append(f"  Est. fee             : {testnet_est['estimated_fee_gwei']:.2f} Gwei "
                 f"({testnet_est['estimated_fee_eth']:.8f} ETH)")
    lines.append(f"  Confirmation time    : ~{testnet_est['expected_confirmation_s']:.0f}s (1 block)")
    lines.append(f"  Note: {testnet_est['note']}")
    lines.append("=" * 70)

    table_str = "\n".join(lines)
    print("\n" + table_str)

    # ── save output ───────────────────────────────────────────────────────────
    elapsed = round(time.perf_counter() - t_start, 2)
    output = {
        "timestamp":           time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec":         elapsed,
        "n_samples":           len(samples),
        "local_evm":           "eth-tester / py-evm (in-process)",
        "contracts": {
            "audit_log":          writer.audit_address(),
            "governance_decision": writer.governance_address(),
        },
        "blockchain_stats":    blockchain_stats,
        "pipeline_stats":      pipeline_stats,
        "testnet_estimate":    testnet_est,
        "sample_records":      records,
    }

    out_json = OUT_DIR / "blockchain_benchmark.json"
    out_txt  = OUT_DIR / "blockchain_benchmark_summary.txt"
    out_json.write_text(json.dumps(output, indent=2))
    out_txt.write_text(table_str)

    print(f"\n  Saved → {out_json}")
    print(f"  Saved → {out_txt}")
    print(f"  Total elapsed: {elapsed}s")


if __name__ == "__main__":
    main()
