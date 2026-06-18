"""
Week 7 — Cascade Architecture Benchmark
=========================================
Implements a three-stage cascade pipeline inspired by the layered structure
of biological immune response, and measures the exit distribution — what
fraction of samples are resolved at each stage — along with accuracy at
each exit point.

The cascade avoids running expensive computations (NLI, full retrieval) on
"easy" samples that can be resolved with fast signals alone.

Cascade Stages
--------------
Stage 1  Confidence Gate  (fast, always runs)
  Signal  : GPT-2 token log-prob + attention entropy combined score
  Exit REJECT : conf < CONF_EARLY_REJECT  → flag immediately
  Exit ACCEPT : conf > CONF_EARLY_ACCEPT  → accept immediately
  Pass        : otherwise                 → proceed to Stage 2

Stage 2  Evidence Gate  (moderate cost — NER + retrieval)
  Signal  : whether NER-routed evidence was retrieved
  Exit NO-EVIDENCE : no evidence found   → apply confidence gate (conf < CONF_THRESHOLD)
  Pass             : evidence found      → proceed to Stage 3

Stage 3  NLI Gate  (expensive — RoBERTa-large-MNLI)
  Signal  : RoBERTa contradiction score
  Exit REJECT : contradiction_score ≥ CONTRA_THRESHOLD → flag
  Exit ACCEPT : otherwise                               → accept

All signals are pre-computed (from week-5 cache); wall-clock times are
derived from per-stage reference latencies measured in weeks 4–5.

Output:
  data/week7/cascade_benchmark.json
  data/week7/cascade_report.txt
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR    = ROOT / "data/week7"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = ROOT / "data/week5/ablation_cache.json"
LABELS_FILE = ROOT / "data/week3/main_eval/gpt_oss_labels.jsonl"
GEN_DIR     = ROOT / "data/week2/generations/llama2/main"

# ── cascade thresholds ────────────────────────────────────────────────────────
# Chosen based on the confidence distribution observed in week 5:
#   range = [0.483, 0.599];  operative threshold = 0.52
# Stage 1 carves out the bottom (very low) and top (very high) tails.

CONF_EARLY_REJECT = 0.500   # conf below → immediate flag (innate-response)
CONF_EARLY_ACCEPT = 0.570   # conf above → immediate accept (self-tolerance)
CONF_THRESHOLD    = 0.520   # used for Stage 2 no-evidence exits
CONTRA_THRESHOLD  = 0.020   # NLI contradiction threshold (Stage 3)

# ── reference latencies (ms per claim, from week-4/5 measurements) ────────────
_LAT = {
    "confidence":    210.0,   # GPT-2, runs once per answer
    "ner":           3.1,     # spaCy, per claim
    "evidence":      612.0,   # Wikidata/Wikipedia retrieval, per claim
    "nli":           890.0,   # RoBERTa-large-MNLI, per claim with evidence
}


# ── data loading ──────────────────────────────────────────────────────────────

def load_samples() -> list:
    gt: dict = {}
    for line in LABELS_FILE.read_text().splitlines():
        r = json.loads(line)
        if r["model"] == "llama2" and r["split"] == "main":
            gt[r["id"]] = r["label"]

    samples = []
    for f in sorted(GEN_DIR.glob("q*.json")):
        d   = json.loads(f.read_text())
        qid = f.stem
        lbl = gt.get(qid)
        if lbl in ("TRUE", "FALSE"):
            samples.append({
                "id":       qid,
                "question": d["question"],
                "answer":   d["generation"],
                "gt_label": lbl,
            })
    return samples


def load_signals(samples: list, cache: dict) -> list:
    enriched = []
    for s in samples:
        qid     = s["id"]
        conf    = cache.get(f"conf_{qid}", 0.5)
        cont    = cache.get(f"contra_ner_{qid}", {"max_score": 0.0, "has_contradiction": False})
        ev_list = cache.get(f"ev_ner_{qid}", [])
        n_claims = len(cache.get(f"ev_ner_{qid}", [])) or 1

        enriched.append({
            "id":            qid,
            "gt_label":      s["gt_label"],
            "is_hall":       s["gt_label"] == "FALSE",
            "conf":          conf,
            "contra_score":  cont["max_score"],
            "has_evidence":  any(e for e in ev_list),
            "n_claims":      n_claims,
        })
    return enriched


# ── cascade runner ────────────────────────────────────────────────────────────

def run_cascade(samples: list) -> list:
    """
    Apply the three-stage cascade to every sample.
    Returns list of per-sample result dicts.
    """
    results = []

    for s in samples:
        conf         = s["conf"]
        contra_score = s["contra_score"]
        has_evidence = s["has_evidence"]
        n_claims     = s["n_claims"]

        # ── Stage 1: Confidence Gate ──────────────────────────────────────────
        if conf < CONF_EARLY_REJECT:
            stage       = 1
            exit_type   = "REJECT"
            flagged     = True
            lat_ms      = _LAT["confidence"]

        elif conf > CONF_EARLY_ACCEPT:
            stage       = 1
            exit_type   = "ACCEPT"
            flagged     = False
            lat_ms      = _LAT["confidence"]

        # ── Stage 2: Evidence Gate ────────────────────────────────────────────
        elif not has_evidence:
            stage       = 2
            exit_type   = "NO_EVIDENCE"
            flagged     = conf < CONF_THRESHOLD
            lat_ms      = (_LAT["confidence"]
                           + n_claims * (_LAT["ner"] + _LAT["evidence"]))

        # ── Stage 3: NLI Gate ─────────────────────────────────────────────────
        else:
            stage       = 3
            flagged     = contra_score >= CONTRA_THRESHOLD
            exit_type   = "REJECT" if flagged else "ACCEPT"
            lat_ms      = (_LAT["confidence"]
                           + n_claims * (_LAT["ner"] + _LAT["evidence"] + _LAT["nli"]))

        results.append({
            "id":          s["id"],
            "gt_label":    s["gt_label"],
            "is_hall":     s["is_hall"],
            "stage":       stage,
            "exit_type":   exit_type,
            "flagged":     flagged,
            "conf":        s["conf"],
            "contra_score": contra_score,
            "lat_ms":      round(lat_ms, 1),
            "correct":     (flagged == s["is_hall"]),
        })

    return results


# ── metrics ───────────────────────────────────────────────────────────────────

def cascade_metrics(results: list) -> dict:
    tp = fp = tn = fn = 0
    for r in results:
        pred    = r["flagged"]
        is_hall = r["is_hall"]
        if pred and is_hall:         tp += 1
        elif pred and not is_hall:   fp += 1
        elif not pred and not is_hall: tn += 1
        else:                        fn += 1

    n    = len(results)
    dr   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = 2 * prec * dr / (prec + dr) if (prec + dr) > 0 else 0.0

    return {
        "n": n,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "detection_rate":      round(dr,   4),
        "false_positive_rate": round(fpr,  4),
        "precision":           round(prec, 4),
        "f1":                  round(f1,   4),
    }


def stage_breakdown(results: list) -> dict:
    """Per-stage exit counts, accuracy, and latency stats."""
    breakdown = {}
    for stage in [1, 2, 3]:
        stage_results = [r for r in results if r["stage"] == stage]
        if not stage_results:
            continue
        n      = len(stage_results)
        n_hall = sum(1 for r in stage_results if r["is_hall"])
        n_flag = sum(1 for r in stage_results if r["flagged"])
        n_corr = sum(1 for r in stage_results if r["correct"])
        lats   = [r["lat_ms"] for r in stage_results]

        # per exit type within stage
        exit_types = {}
        for exit_t in set(r["exit_type"] for r in stage_results):
            subset = [r for r in stage_results if r["exit_type"] == exit_t]
            exit_types[exit_t] = {
                "count": len(subset),
                "pct":   round(len(subset) / len(results) * 100, 1),
                "n_hall": sum(1 for r in subset if r["is_hall"]),
                "n_flag": sum(1 for r in subset if r["flagged"]),
                "accuracy": round(sum(1 for r in subset if r["correct"]) / len(subset), 4),
            }

        breakdown[stage] = {
            "n":         n,
            "pct_total": round(n / len(results) * 100, 1),
            "n_hall":    n_hall,
            "n_flagged": n_flag,
            "accuracy":  round(n_corr / n, 4),
            "avg_lat_ms": round(sum(lats) / n, 1),
            "exit_types": exit_types,
        }
    return breakdown


# ── efficiency analysis ───────────────────────────────────────────────────────

def efficiency_vs_flat(cascade_results: list) -> dict:
    """
    Compare cascade average latency to flat C4 pipeline (all stages always run).
    """
    # flat C4 average latency
    flat_lats = []
    for r in cascade_results:
        # flat: always runs all 4 stages; evidence retrieval on all claims
        # use same n_claims derived from cascade result (contained in lat calc)
        # estimate: conf + NER + evidence (per claim, avg 2.4 claims from week3)
        n_avg = 2.4
        flat_lat = (_LAT["confidence"]
                    + n_avg * (_LAT["ner"] + _LAT["evidence"])
                    + n_avg * _LAT["nli"])
        flat_lats.append(flat_lat)

    cascade_lats = [r["lat_ms"] for r in cascade_results]
    flat_avg     = round(sum(flat_lats) / len(flat_lats), 1)
    cascade_avg  = round(sum(cascade_lats) / len(cascade_lats), 1)
    speedup      = round(flat_avg / cascade_avg, 2)

    return {
        "flat_avg_lat_ms":     flat_avg,
        "cascade_avg_lat_ms":  cascade_avg,
        "speedup":             speedup,
        "pct_reduction":       round((1 - cascade_avg / flat_avg) * 100, 1),
    }


# ── C4 flat for reference ─────────────────────────────────────────────────────

C4_REF = {
    "detection_rate":      0.6230,
    "false_positive_rate": 0.6835,
    "precision":           0.2857,
    "f1":                  0.3917,
}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.perf_counter()
    print("=" * 64)
    print("  Week 7 — Cascade Architecture Benchmark")
    print("=" * 64)
    print(f"\n  Thresholds:")
    print(f"    Stage 1 early-reject : conf < {CONF_EARLY_REJECT}")
    print(f"    Stage 1 early-accept : conf > {CONF_EARLY_ACCEPT}")
    print(f"    Stage 2 conf-only    : conf < {CONF_THRESHOLD}")
    print(f"    Stage 3 NLI          : contra ≥ {CONTRA_THRESHOLD}")

    samples  = load_samples()
    cache    = json.loads(CACHE_FILE.read_text())
    samples  = load_signals(samples, cache)

    print(f"\n  {len(samples)} samples  "
          f"({sum(1 for s in samples if s['is_hall'])} hallucinatory)")

    # ── run cascade ───────────────────────────────────────────────────────────
    cascade_results = run_cascade(samples)

    # ── metrics ───────────────────────────────────────────────────────────────
    overall   = cascade_metrics(cascade_results)
    breakdown = stage_breakdown(cascade_results)
    efficiency = efficiency_vs_flat(cascade_results)

    # ── report ────────────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 68)
    lines.append("  Week 7 — Cascade Architecture Benchmark")
    lines.append("  200 samples · llama2/main · judge: gpt-oss-120b")
    lines.append("=" * 68)
    lines.append("")
    lines.append("  Cascade Parameters")
    lines.append(f"  Stage 1 early-reject : GPT-2 confidence < {CONF_EARLY_REJECT}"
                 "  (innate response)")
    lines.append(f"  Stage 1 early-accept : GPT-2 confidence > {CONF_EARLY_ACCEPT}"
                 "  (self-tolerance)")
    lines.append(f"  Stage 2 no-evidence  : confidence < {CONF_THRESHOLD}  → flag"
                 "  (adaptive fail-safe)")
    lines.append(f"  Stage 3 NLI          : contradiction ≥ {CONTRA_THRESHOLD}"
                 "  → flag  (adaptive response)")
    lines.append("")
    lines.append("  Exit Distribution")
    lines.append("  " + "─" * 64)
    lines.append(f"  {'Stage':<8} {'Exit Type':<20} {'N':>5} {'%':>6} "
                 f"{'Hallucinatory':>14} {'Flagged':>9} {'Acc':>7}")
    lines.append("  " + "─" * 64)

    for stage_n, stage_data in sorted(breakdown.items()):
        for exit_t, ed in sorted(stage_data["exit_types"].items()):
            label = (f"Stage {stage_n}"
                     if exit_t == sorted(stage_data["exit_types"].keys())[0]
                     else "")
            lines.append(
                f"  {label:<8} {exit_t:<20} {ed['count']:>5} {ed['pct']:>5.1f}% "
                f"{ed['n_hall']:>14} {ed['n_flag']:>9} {ed['accuracy']:>6.1%}"
            )

    lines.append("  " + "─" * 64)
    lines.append(f"  {'Total':<28} {len(cascade_results):>5} {'100.0':>6}%")
    lines.append("")
    lines.append("  Overall Performance")
    lines.append("  " + "─" * 64)
    lines.append(f"  {'System':<28} {'DR':>7} {'FPR':>7} {'Prec':>7} {'F1':>7}")
    lines.append("  " + "─" * 64)
    lines.append(f"  {'C4: Flat pipeline (ref)':<28} "
                 f"{C4_REF['detection_rate']:>6.1%} "
                 f"{C4_REF['false_positive_rate']:>6.1%} "
                 f"{C4_REF['precision']:>6.1%} "
                 f"{C4_REF['f1']:>6.1%}")
    lines.append(f"  {'Cascade pipeline':<28} "
                 f"{overall['detection_rate']:>6.1%} "
                 f"{overall['false_positive_rate']:>6.1%} "
                 f"{overall['precision']:>6.1%} "
                 f"{overall['f1']:>6.1%}")
    lines.append("  " + "─" * 64)
    lines.append("")
    lines.append("  Computational Efficiency")
    lines.append("  " + "─" * 64)
    lines.append(f"  Flat C4 avg latency    : {efficiency['flat_avg_lat_ms']:.0f} ms / sample")
    lines.append(f"  Cascade avg latency    : {efficiency['cascade_avg_lat_ms']:.0f} ms / sample")
    lines.append(f"  Speedup                : {efficiency['speedup']:.2f}×  "
                 f"({efficiency['pct_reduction']:.1f}% reduction)")
    lines.append("")
    lines.append("  Stage-level latency (avg ms / sample reaching each stage)")
    for sn, sd in sorted(breakdown.items()):
        lines.append(f"  Stage {sn} ({sd['pct_total']:.0f}% of samples): "
                     f"{sd['avg_lat_ms']:.0f} ms avg")
    lines.append("")
    lines.append("  Biological Analogy")
    lines.append("  " + "─" * 64)
    s1_pct = sum(sd["pct_total"] for sn, sd in breakdown.items() if sn == 1)
    s2_pct = sum(sd["pct_total"] for sn, sd in breakdown.items() if sn == 2)
    s3_pct = sum(sd["pct_total"] for sn, sd in breakdown.items() if sn == 3)
    lines.append(f"  Stage 1 (innate gate):   {s1_pct:.1f}% of queries resolved by fast signal")
    lines.append(f"  Stage 2 (evidence gate): {s2_pct:.1f}% resolved by retrieval coverage")
    lines.append(f"  Stage 3 (NLI gate):      {s3_pct:.1f}% require full adaptive response")
    lines.append("  Only the uncertain middle fraction triggers expensive NLI computation,")
    lines.append("  mirroring the immune system's staged escalation from innate to adaptive.")
    lines.append("=" * 68)

    report_str = "\n".join(lines)
    print("\n" + report_str)

    # ── save ──────────────────────────────────────────────────────────────────
    elapsed = round(time.perf_counter() - t_start, 3)
    output = {
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": elapsed,
        "n_samples":   len(samples),
        "thresholds": {
            "conf_early_reject": CONF_EARLY_REJECT,
            "conf_early_accept": CONF_EARLY_ACCEPT,
            "conf_threshold":    CONF_THRESHOLD,
            "contra_threshold":  CONTRA_THRESHOLD,
        },
        "overall_metrics":   overall,
        "c4_reference":      C4_REF,
        "stage_breakdown":   breakdown,
        "efficiency":        efficiency,
        "per_sample":        cascade_results,
    }

    (OUT_DIR / "cascade_benchmark.json").write_text(json.dumps(output, indent=2))
    (OUT_DIR / "cascade_report.txt").write_text(report_str)

    print(f"\n  Saved → {OUT_DIR}/cascade_benchmark.json")
    print(f"  Saved → {OUT_DIR}/cascade_report.txt")
    print(f"  Total elapsed: {elapsed}s")


if __name__ == "__main__":
    main()
