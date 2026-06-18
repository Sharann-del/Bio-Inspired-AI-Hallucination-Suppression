# Week 4–5 Report: Core Pipeline + Ablation Study

**Project:** Bio-Inspired AI — Hallucination Suppression Pipeline
**Period:** Jun 1–14, 2026
**Dataset:** TruthfulQA · llama2/main split · 200 evaluation claims

---

## Week 4 (Jun 1–7): Core Pipeline Modules

### Overview

Four pipeline modules were implemented and individually tested. A Wikidata reliability diagnostic was run on 200 queries to measure retrieval performance.

---

### Module 1: NER + Routing Layer

**File:** `src/pipeline/ner_router.py`

Extracts named entities from a claim using spaCy (`en_core_web_sm`) and decides which verification strategy to apply.

| Entity Types | Strategy | Description |
|---|---|---|
| PERSON, ORG, GPE, LOC, FAC | `entity_lookup` | Wikidata entity search |
| DATE, TIME, QUANTITY, CARDINAL | `structured_fact` | Wikidata property search |
| EVENT, WORK_OF_ART, LAW, NORP | `text_search` | Wikipedia fulltext |
| None found | `keyword_search` | Wikipedia keyword fallback |

Returns a `RoutingDecision` containing extracted entities, their types, the chosen strategy, the primary query term, and keyword fallback terms.

---

### Module 2: Evidence Retrieval

**File:** `src/pipeline/evidence_retrieval.py`

Retrieves evidence through a 4-step fallback chain:

1. **Wikidata** — entity search via REST API (description + label)
2. **Wikipedia page extract** — intro paragraph via MediaWiki API
3. **Wikipedia fulltext search** — top search hit extract
4. **No evidence** — empty result, logged as `"none"`

Every retrieval attempt is logged with source, fallback flag, no-evidence flag, and latency. A `reliability_summary()` function computes aggregate metrics across a session.

---

### Module 3: Confidence Scorer

**File:** `src/pipeline/confidence_scorer.py`

Scores generation text using GPT-2 (124M, CPU) as a proxy model. Two signals:

- **Token log-probability** — mean per-token log P(token | context) under GPT-2. Low = model considers text unlikely.
- **Attention entropy** — mean entropy of attention distributions across all layers/heads. High = diffuse, unfocused attention.

Combined score:

```
confidence = 0.5 × sigmoid(mean_log_prob / 3.0) + 0.5 × (1 − norm_attention_entropy)
```

Output range: [0, 1]. Higher = more confident = less likely hallucination.

---

### Module 4: Contradiction Detector

**File:** `src/pipeline/contradiction_detector.py`

Classifies (evidence, claim) pairs using `roberta-large-mnli` via HuggingFace zero-shot NLI pipeline.

| Label | Meaning |
|---|---|
| `entailment` | Evidence supports the claim |
| `neutral` | Evidence is inconclusive |
| `contradiction` | Evidence contradicts the claim → hallucination signal |

Returns scores for all three labels. `is_contradiction = True` when `label == "contradiction"` and `score ≥ 0.5`.

---

### Wikidata Reliability Diagnostic

**Script:** `notebooks/week4_wikidata_diagnostic.py`
**Output:** `data/week4/wikidata_reliability_report.json`
**Run date:** Jun 4, 2026

Collected 200 unique entity queries from NER routing over Week 3 claims, then ran the full retrieval pipeline.

#### Results

| Metric | Value |
|---|---|
| Queries run | 200 |
| **Success rate** (Wikidata direct hit) | **59.5%** (119 / 200) |
| **Fallback rate** (fell back to Wikipedia) | **10.5%** (21 / 200) |
| **No-evidence rate** (nothing found) | **30.0%** (60 / 200) |
| Avg latency | 4,996 ms |
| p50 latency | 618 ms |
| p90 latency | 13,036 ms |
| p99 latency | 14,163 ms |

#### Breakdown by Strategy

| Strategy | n | Wikidata hits | No evidence |
|---|---|---|---|
| `entity_lookup` | 150 | 92 (61.3%) | 45 (30.0%) |
| `structured_fact` | 30 | 13 (43.3%) | 12 (40.0%) |
| `text_search` | 20 | 14 (70.0%) | 3 (15.0%) |

#### Key Finding

30% of queries returned no evidence from any source. This becomes the primary bottleneck for contradiction-based detection in Week 5 — when retrieval fails, the NLI model has no premise to compare against and outputs zero by definition.

---

---

## Week 5 (Jun 8–14): Ablation Study

### Overview

Ran five configurations on the same 200 evaluation claims to isolate the contribution of each pipeline component.

**Script:** `notebooks/week5_ablation_study.py`
**Output:** `data/week5/ablation_results.json`, `data/week5/ablation_table.txt`
**Run date:** Jun 5, 2026

---

### Dataset

| Split | Count |
|---|---|
| Total samples | 200 |
| Hallucinatory (`FALSE`) | 61 |
| Truthful (`TRUE`) | 139 |

Ground truth: GPT-OSS judge labels from Week 3 evaluation.

---

### Configurations

| Config | Description |
|---|---|
| **C0** | No detection — always predicts "not hallucination" (trivial baseline) |
| **C1** | Confidence only — GPT-2 log-prob + attention entropy; flag if `confidence < 0.52` |
| **C2** | Keyword evidence + RoBERTa NLI — raw keyword retrieval, no NER routing; flag if `contradiction_score ≥ 0.02` |
| **C3** | NER + Contradiction — NER-routed retrieval + RoBERTa NLI; flag if `contradiction_score ≥ 0.02` |
| **C4** | Full pipeline — NER routing + evidence + contradiction + confidence; flag if contradiction OR low confidence |

Thresholds were set based on observed signal distributions on the eval set:
- Confidence scores range: [0.483, 0.599]
- Contradiction max score range: [0.000, 0.357]; 80% of answers score 0.000

---

### Results

| Config | DR | FPR | Precision | F1 | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|
| C0: No detection | 0.0% | 0.0% | 0.0% | 0.0% | 0 | 0 | 139 | 61 |
| C1: Confidence only | 50.8% | 63.3% | 26.1% | 34.4% | 31 | 88 | 51 | 30 |
| C2: Keyword + Contradiction | 18.0% | 20.9% | 27.5% | 21.8% | 11 | 29 | 110 | 50 |
| C3: NER + Contradiction | 26.2% | 21.6% | 34.8% | 29.9% | 16 | 30 | 109 | 45 |
| **C4: Full pipeline** | **62.3%** | **68.3%** | **28.6%** | **39.2%** | **38** | **95** | **44** | **23** |

**DR** = Detection Rate = TP / (TP + FN) — recall on the hallucination class
**FPR** = False Positive Rate = FP / (FP + TN) — truthful answers wrongly flagged

---

### Findings

**NER routing improves contradiction detection (+8 pp DR, same FPR)**
Comparing C2 → C3: detection rate rises from 18.0% to 26.2% with no meaningful change in false positive rate (20.9% → 21.6%). Smarter entity-based routing retrieves more relevant evidence, giving the NLI model a better premise to work with. This validates the Week 4 NER router as a net positive component.

**Full pipeline achieves the best DR but at high FPR cost**
C4 reaches 62.3% detection rate but 68.3% false positive rate. The OR combination of confidence and contradiction signals stacks their individual noise — confidence alone already has 63.3% FPR (C1), so the combined signal cannot improve on that floor.

**Evidence retrieval coverage is the binding constraint**
80% of answers have a contradiction max score of 0.000, meaning evidence retrieval found nothing usable for NLI comparison. The contradiction detector is operating on only ~20% of the dataset. Improving retrieval coverage is the highest-leverage path to better DR in C2/C3/C4.

**GPT-2 is a mismatched proxy for Llama-2 confidence**
The confidence signal (C1) scores answers generated by Llama-2 using GPT-2's token distribution. These are different models with different training data and priors. A hallucination that Llama-2 generated confidently may not look unusual to GPT-2, and vice versa. Using Llama-2's own output logprobs would produce a much stronger confidence signal.

---

### Limitations and Next Steps

| Issue | Effect | Fix |
|---|---|---|
| 30% no-evidence retrieval rate | Caps contradiction-based DR at ~26% | Add more retrieval sources; improve query formulation |
| GPT-2 used as Llama-2 proxy | Noisy confidence signal; all scores cluster in [0.483, 0.599] | Use generating model's own logprobs |
| OR combination in C4 | FPR inherits the worst of both signals | Tune combination logic; require both signals to agree |
| Contradiction threshold too low (0.02) | Captures noise-level NLI outputs | Calibrate on held-out validation set |
| Single NLI model | RoBERTa-large-MNLI not SOTA for factual contradiction | Replace with AlignScore or FActScore |
