"""
Week 4 — Wikidata Reliability Diagnostic

Runs 200 queries through the full evidence retrieval pipeline and measures:
  - success_rate      : fraction resolved via Wikidata directly
  - fallback_rate     : fraction that fell back to Wikipedia
  - no_evidence_rate  : fraction with no evidence found

Query source: named entities extracted by the NER router from week 3 claims.
We collect the first 200 unique entity queries across both models/splits.

Output:
  data/week4/wikidata_reliability_report.json   ← main report
  data/week4/wikidata_diagnostic_log.jsonl       ← per-query log
"""

import sys, json, time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline.ner_router        import route
from pipeline.evidence_retrieval import retrieve, reliability_summary, clear_log, get_log

OUT_DIR = ROOT / "data/week4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_N  = 200
RATE_LIMIT = 0.4   # seconds between requests (be polite to Wikidata/Wikipedia)

print("=" * 65)
print("  Week 4 — Wikidata Reliability Diagnostic (200 queries)")
print("=" * 65)

# ── Step 1: collect entity queries from week 3 claims ─────────────────────────

claims_dir = ROOT / "data/week3/claims"
claims_files = list(claims_dir.glob("*_claims.jsonl"))
assert claims_files, f"No claims files found in {claims_dir}"

print(f"\n  Loading claims from: {[f.name for f in claims_files]}")

all_claims: list[str] = []
for cf in sorted(claims_files):
    for line in cf.read_text().splitlines():
        rec = json.loads(line)
        all_claims.extend(rec.get("claims", []))

print(f"  Total claims: {len(all_claims)}")

# ── Step 2: NER → extract unique entity queries ───────────────────────────────

print("\n  Running NER to extract entity queries …")

seen_queries: set[str] = set()
query_list: list[tuple[str, str]] = []   # (query_text, strategy)

for claim in all_claims:
    if len(query_list) >= TARGET_N:
        break
    decision = route(claim)
    if decision.primary_query:
        q = decision.primary_query.strip()
        if q and q.lower() not in seen_queries and len(q) > 2:
            seen_queries.add(q.lower())
            query_list.append((q, decision.strategy))

# If we still need more, fall back to keywords from remaining claims
if len(query_list) < TARGET_N:
    for claim in all_claims:
        if len(query_list) >= TARGET_N:
            break
        decision = route(claim)
        for kw in decision.keywords:
            if len(query_list) >= TARGET_N:
                break
            kl = kw.lower()
            if kl not in seen_queries and len(kw) > 3:
                seen_queries.add(kl)
                query_list.append((kw, "keyword_search"))

print(f"  Collected {len(query_list)} unique queries (target: {TARGET_N})")

# ── Step 3: run retrieval on each query ───────────────────────────────────────

clear_log()

print(f"\n  Running {len(query_list)} retrieval queries (est. {len(query_list)*RATE_LIMIT/60:.1f} min)…\n")

detailed_results: list[dict] = []
source_counts: dict[str, int] = defaultdict(int)

for i, (query, strategy) in enumerate(query_list, 1):
    result = retrieve(query, strategy=strategy)
    source_counts[result.source] += 1

    detailed_results.append({
        "i":            i,
        "query":        query,
        "strategy":     strategy,
        "source":       result.source,
        "fallback":     result.fallback_used,
        "no_evidence":  result.no_evidence,
        "wikidata_qid": result.wikidata_qid,
        "latency_ms":   result.latency_ms,
        "evidence_len": len(result.evidence),
        "evidence_snippet": result.evidence[:150] if result.evidence else "",
    })

    if i % 25 == 0 or i == 1 or i == len(query_list):
        wd_so_far = sum(1 for r in detailed_results if r["source"] == "wikidata")
        fb_so_far = sum(1 for r in detailed_results
                        if r["source"] in ("wikipedia_page", "wikipedia_search"))
        no_so_far = sum(1 for r in detailed_results if r["no_evidence"])
        print(f"  [{i:3d}/{len(query_list)}]  wikidata={wd_so_far}  fallback={fb_so_far}  no_evidence={no_so_far}  last={result.source}")

    time.sleep(RATE_LIMIT)

# ── Step 4: compute reliability metrics ──────────────────────────────────────

log = get_log()
summary = reliability_summary(log)
n = len(detailed_results)

# per-strategy breakdown
strategy_stats: dict[str, dict] = defaultdict(lambda: defaultdict(int))
for r in detailed_results:
    strategy_stats[r["strategy"]]["total"] += 1
    strategy_stats[r["strategy"]][r["source"]] += 1

# latency percentiles
latencies = sorted(r["latency_ms"] for r in detailed_results)
p50 = latencies[n // 2]
p90 = latencies[int(n * 0.9)]
p99 = latencies[min(int(n * 0.99), n - 1)]

# ── Step 5: build and save report ────────────────────────────────────────────

report = {
    "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%S"),
    "n_queries":        n,
    "target":           TARGET_N,

    # core metrics (the deliverable)
    "success_rate":     summary["success_rate"],       # wikidata direct hit
    "fallback_rate":    summary["fallback_rate"],       # fell back to Wikipedia
    "no_evidence_rate": summary["no_evidence_rate"],    # nothing found

    # source breakdown
    "wikidata_hits":         summary["wikidata_hits"],
    "wikipedia_page_hits":   summary["wikipedia_page_hits"],
    "wikipedia_search_hits": summary["wikipedia_search_hits"],
    "no_evidence_count":     summary["no_evidence"],

    # latency
    "avg_latency_ms": summary["avg_latency_ms"],
    "p50_latency_ms": p50,
    "p90_latency_ms": p90,
    "p99_latency_ms": p99,

    # per-strategy breakdown
    "by_strategy": {k: dict(v) for k, v in strategy_stats.items()},
}

(OUT_DIR / "wikidata_reliability_report.json").write_text(json.dumps(report, indent=2))
(OUT_DIR / "wikidata_diagnostic_log.jsonl").write_text(
    "\n".join(json.dumps(r) for r in detailed_results)
)

# ── Step 6: print report ──────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  WIKIDATA RELIABILITY REPORT")
print("=" * 65)
print(f"\n  Queries run       : {n}")
print(f"  Success rate      : {summary['success_rate']:.1%}  ({summary['wikidata_hits']}/{n} via Wikidata)")
print(f"  Fallback rate     : {summary['fallback_rate']:.1%}  ({summary['wikipedia_page_hits']+summary['wikipedia_search_hits']}/{n} via Wikipedia)")
print(f"    ↳ page extract  : {summary['wikipedia_page_hits']}")
print(f"    ↳ search hits   : {summary['wikipedia_search_hits']}")
print(f"  No-evidence rate  : {summary['no_evidence_rate']:.1%}  ({summary['no_evidence']}/{n})")
print(f"\n  Latency p50/p90/p99: {p50} / {p90} / {p99} ms")
print(f"  Avg latency        : {summary['avg_latency_ms']:.0f} ms")

print("\n  By strategy:")
for strat, counts in strategy_stats.items():
    total = counts["total"]
    wd    = counts.get("wikidata", 0)
    no    = counts.get("none", 0)
    print(f"    {strat:<20s}: n={total}  wikidata={wd}  no_ev={no}")

print(f"\n  Report saved → {OUT_DIR}/wikidata_reliability_report.json")
print(f"  Log    saved → {OUT_DIR}/wikidata_diagnostic_log.jsonl")
print("\n  WIKIDATA DIAGNOSTIC: COMPLETE")
