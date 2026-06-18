"""
Week 6 — Controlled RAG Baseline
==================================
Implements a retrieval-augmented detection baseline on the same 200
evaluation samples (llama2/main) with the same ground-truth labels
(gpt-oss-120b) used by the main pipeline.

The baseline replaces the bio-inspired multi-signal pipeline with the
simplest possible retrieval approach:
  1. For each answer, retrieve Wikipedia/Wikidata evidence
     (reuses the NER-routed evidence already cached from week 5)
  2. Compute TF-IDF cosine similarity between the answer and the
     concatenated retrieved evidence
  3. Flag the answer as hallucinated if similarity < threshold
     (low evidence overlap → answer not grounded → suspect)

Threshold is swept over [0.01, 0.99] to find the F1-maximising value.
Final metrics are compared against C4 (full pipeline) from week 5.

This baseline models what a simpler RAG-style detection system achieves
with the same external knowledge sources, demonstrating the added value
of NLI + confidence scoring in our bio-inspired architecture.

Output: data/week6/rag_baseline.json
         data/week6/rag_baseline_report.txt
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

# C4 reference (week 5)
C4_RESULTS = {
    "detection_rate":      0.6230,
    "false_positive_rate": 0.6835,
    "precision":           0.2857,
    "f1":                  0.3917,
    "tp": 38, "fp": 95, "tn": 44, "fn": 23,
}
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


# ── TF-IDF retrieval overlap scorer ──────────────────────────────────────────

def build_tfidf_scores(samples: list, cache: dict) -> tuple:
    """
    For each sample, compute cosine similarity between the answer text
    and the concatenated NER-retrieved evidence (from week-5 cache).

    Returns ({qid: similarity_score}, n_with_evidence) where score=0 means
    no evidence retrieved or zero textual overlap. Uses scikit-learn TfidfVectorizer.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    print("  Building TF-IDF corpus…")

    # build parallel lists: answer text and evidence text per sample
    answers  = []
    evidence = []
    qids     = []

    for s in samples:
        qid      = s["id"]
        ev_list  = cache.get(f"ev_ner_{qid}", [])
        ev_text  = " ".join(e for e in ev_list if e).strip()

        answers.append(s["answer"])
        evidence.append(ev_text if ev_text else "")
        qids.append(qid)

    n_with_evidence = sum(1 for e in evidence if e)
    print(f"  {n_with_evidence}/{len(samples)} samples have retrieved evidence "
          f"({n_with_evidence/len(samples):.1%})")

    # fit TF-IDF on the union of all answers + evidence texts
    corpus = answers + evidence
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20_000,
        ngram_range=(1, 2),
        min_df=1,
    )
    vectorizer.fit(corpus)

    scores = {}
    for i, qid in enumerate(qids):
        if not evidence[i]:
            # no evidence retrieved → similarity is 0 (flag as suspect)
            scores[qid] = 0.0
            continue
        a_vec = vectorizer.transform([answers[i]])
        e_vec = vectorizer.transform([evidence[i]])
        sim   = float(cosine_similarity(a_vec, e_vec)[0, 0])
        scores[qid] = round(sim, 6)

    return scores, n_with_evidence


# ── threshold sweep + metrics ─────────────────────────────────────────────────

def sweep_thresholds(samples: list, scores: dict) -> list:
    """
    For each threshold t in [0.00, 1.00]:
      flag = similarity < t  (low overlap → suspect)
    Compute DR, FPR, Prec, F1 and return the sweep table.
    """
    import numpy as np

    thresholds = [round(t, 3) for t in [x / 200 for x in range(201)]]
    results    = []

    for t in thresholds:
        tp = fp = tn = fn = 0
        for s in samples:
            pred    = scores[s["id"]] < t   # low overlap = flagged
            is_hall = s["gt_label"] == "FALSE"
            if pred and is_hall:       tp += 1
            elif pred and not is_hall: fp += 1
            elif not pred and not is_hall: tn += 1
            else:                      fn += 1

        dr   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1   = 2 * prec * dr / (prec + dr) if (prec + dr) > 0 else 0.0

        results.append({
            "threshold": t,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "detection_rate":      round(dr,   4),
            "false_positive_rate": round(fpr,  4),
            "precision":           round(prec, 4),
            "f1":                  round(f1,   4),
        })

    return results


def find_best(sweep: list, metric: str = "f1") -> dict:
    return max(sweep, key=lambda r: r[metric])


# ── additional baseline: unigram overlap ─────────────────────────────────────

def unigram_overlap_scores(samples: list, cache: dict) -> dict:
    """
    Token-level F1 overlap between answer and evidence.
    Complementary baseline — word overlap without TF-IDF weighting.
    """
    import re

    def tokens(text: str) -> set:
        return {w.lower() for w in re.findall(r"\w+", text)
                if len(w) > 2 and w.lower() not in _STOPWORDS}

    scores = {}
    for s in samples:
        qid     = s["id"]
        ev_list = cache.get(f"ev_ner_{qid}", [])
        ev_text = " ".join(e for e in ev_list if e)

        if not ev_text:
            scores[qid] = 0.0
            continue

        ans_toks = tokens(s["answer"])
        ev_toks  = tokens(ev_text)

        if not ans_toks or not ev_toks:
            scores[qid] = 0.0
            continue

        inter    = len(ans_toks & ev_toks)
        prec     = inter / len(ans_toks)
        rec      = inter / len(ev_toks)
        f1       = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        scores[qid] = round(f1, 6)

    return scores


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "that", "this",
    "it", "its", "they", "them", "their", "he", "she", "his", "her", "we",
    "our", "you", "your", "i", "my", "me", "as", "if", "so", "no", "more",
    "also", "about", "than", "into", "through", "during", "before", "after",
    "above", "below", "between", "each", "other", "such", "than",
}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.perf_counter()

    print("=" * 68)
    print("  Week 6 — Controlled RAG Baseline")
    print("  Same 200 samples · same judge labels · retrieval-only detection")
    print("=" * 68)

    samples = load_samples()
    cache   = json.loads(CACHE_FILE.read_text())
    print(f"\n  Loaded {len(samples)} samples  "
          f"({sum(1 for s in samples if s['gt_label']=='FALSE')} hallucinatory, "
          f"{sum(1 for s in samples if s['gt_label']=='TRUE')} truthful)")

    # ── compute retrieval similarity scores ───────────────────────────────────
    print("\n── RAG Baseline 1: TF-IDF cosine similarity ──")
    tfidf_scores, n_with_evidence = build_tfidf_scores(samples, cache)

    print("\n── RAG Baseline 2: Unigram token overlap (F1) ──")
    overlap_scores = unigram_overlap_scores(samples, cache)

    n_zero_tfidf   = sum(1 for s in tfidf_scores.values() if s == 0.0)
    n_zero_overlap = sum(1 for s in overlap_scores.values() if s == 0.0)
    n_no_evidence  = len(samples) - n_with_evidence   # truly no evidence retrieved
    print(f"  TF-IDF:  {n_zero_tfidf}/{len(samples)} samples with zero similarity "
          f"({n_no_evidence} no evidence, {n_zero_tfidf - n_no_evidence} evidence but zero overlap)")
    print(f"  Overlap: {n_zero_overlap}/{len(samples)} samples with zero overlap")

    # ── threshold sweep ───────────────────────────────────────────────────────
    print("\n── Sweeping thresholds ──")
    tfidf_sweep   = sweep_thresholds(samples, tfidf_scores)
    overlap_sweep = sweep_thresholds(samples, overlap_scores)

    best_tfidf   = find_best(tfidf_sweep,   "f1")
    best_overlap = find_best(overlap_sweep, "f1")

    # also find best by detection_rate (to compare recall)
    best_tfidf_dr   = find_best(tfidf_sweep,   "detection_rate")
    best_overlap_dr = find_best(overlap_sweep, "detection_rate")

    # ── comparison table ──────────────────────────────────────────────────────
    rows = [
        ("C4: Full pipeline (week 5)",          C4_RESULTS,         "—"),
        ("RAG-TF-IDF (best F1)",                best_tfidf,         f"sim<{best_tfidf['threshold']}"),
        ("RAG-TF-IDF (best DR)",                best_tfidf_dr,      f"sim<{best_tfidf_dr['threshold']}"),
        ("RAG-Overlap (best F1)",               best_overlap,       f"ovlp<{best_overlap['threshold']}"),
        ("RAG-Overlap (best DR)",               best_overlap_dr,    f"ovlp<{best_overlap_dr['threshold']}"),
    ]

    lines = []
    lines.append("=" * 80)
    lines.append("  Week 6 — RAG Baseline vs Full Pipeline")
    lines.append("  200 samples · llama2/main · judge: gpt-oss-120b")
    lines.append("=" * 80)
    hdr = (f"  {'System':<35} {'Threshold':<15} {'DR':>7} {'FPR':>7} "
           f"{'Prec':>7} {'F1':>7}")
    lines.append(hdr)
    lines.append("  " + "─" * 76)
    for name, m, thresh in rows:
        lines.append(
            f"  {name:<35} {thresh:<15} "
            f"{m['detection_rate']:>6.1%} "
            f"{m['false_positive_rate']:>6.1%} "
            f"{m['precision']:>6.1%} "
            f"{m['f1']:>6.1%}"
        )
    lines.append("  " + "─" * 76)
    lines.append("")
    lines.append("  Evidence coverage (NER-retrieved from Wikipedia/Wikidata)")
    lines.append(f"  Samples with retrieved evidence     : {n_with_evidence}/{len(samples)}  "
                 f"({n_with_evidence/len(samples):.1%})")
    lines.append(f"  Samples with non-zero TF-IDF overlap: {len(samples)-n_zero_tfidf}/{len(samples)}  "
                 f"({(len(samples)-n_zero_tfidf)/len(samples):.1%})")
    lines.append(f"  Samples with no evidence            : {n_no_evidence}/{len(samples)}  "
                 f"({n_no_evidence/len(samples):.1%})")
    lines.append(f"  Evidence retrieved but zero overlap  : {n_zero_tfidf - n_no_evidence}/{len(samples)}")
    lines.append("")
    lines.append("  Key findings")
    lines.append("  ────────────")

    c4_f1    = C4_RESULTS["f1"]
    rag_f1   = best_tfidf["f1"]
    delta_f1 = c4_f1 - rag_f1
    lines.append(f"  • RAG-TF-IDF best F1 = {rag_f1:.1%} vs C4 F1 = {c4_f1:.1%} "
                 f"(Δ = {delta_f1:+.1%})")

    c4_dr    = C4_RESULTS["detection_rate"]
    lines.append(f"  • RAG-TF-IDF best DR = {best_tfidf_dr['detection_rate']:.1%} vs C4 DR = {c4_dr:.1%}")

    lines.append(f"  • {n_no_evidence}/{len(samples)} samples ({n_no_evidence/len(samples):.0%}) "
                 f"have no retrieved evidence, forcing them to be flagged.")
    lines.append(f"  • Even among {n_with_evidence} samples with evidence, "
                 f"{n_zero_tfidf - n_no_evidence} show zero TF-IDF overlap,")
    lines.append(f"    demonstrating the lexical gap between retrieved context and answer text.")
    lines.append("  • TF-IDF overlap cannot distinguish factual precision from")
    lines.append("    topic relevance, explaining the high FPR.")
    lines.append("  • Our pipeline adds two independent signals — NLI entailment")
    lines.append("    and GPT-2 confidence — that improve precision over retrieval alone.")
    lines.append("=" * 80)

    report_str = "\n".join(lines)
    print("\n" + report_str)

    # ── save ──────────────────────────────────────────────────────────────────
    elapsed = round(time.perf_counter() - t_start, 2)
    output = {
        "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec":   elapsed,
        "n_samples":     len(samples),
        "n_hallucinatory": sum(1 for s in samples if s["gt_label"] == "FALSE"),
        "n_truthful":    sum(1 for s in samples if s["gt_label"] == "TRUE"),
        "evidence_coverage": {
            "with_retrieved_evidence":    n_with_evidence,
            "with_nonzero_tfidf_overlap": len(samples) - n_zero_tfidf,
            "no_evidence":                n_no_evidence,
            "evidence_but_zero_overlap":  n_zero_tfidf - n_no_evidence,
            "retrieval_rate":             round(n_with_evidence / len(samples), 4),
            "overlap_rate":               round((len(samples) - n_zero_tfidf) / len(samples), 4),
        },
        "c4_reference":  C4_RESULTS,
        "tfidf_baseline": {
            "best_f1":       best_tfidf,
            "best_dr":       best_tfidf_dr,
            "sweep":         tfidf_sweep,
        },
        "overlap_baseline": {
            "best_f1":       best_overlap,
            "best_dr":       best_overlap_dr,
            "sweep":         overlap_sweep,
        },
        "per_sample": [
            {
                "qid":           s["id"],
                "gt_label":      s["gt_label"],
                "tfidf_score":   tfidf_scores[s["id"]],
                "overlap_score": overlap_scores[s["id"]],
            }
            for s in samples
        ],
    }

    out_json = OUT_DIR / "rag_baseline.json"
    out_txt  = OUT_DIR / "rag_baseline_report.txt"
    out_json.write_text(json.dumps(output, indent=2))
    out_txt.write_text(report_str)

    print(f"\n  Saved → {out_json}")
    print(f"  Saved → {out_txt}")
    print(f"  Total elapsed: {elapsed}s")


if __name__ == "__main__":
    main()
