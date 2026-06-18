# Bio-Inspired Cascaded Hallucination Suppression in Large Language Models with Blockchain Audit Governance

**Draft — Sections 1–5**  
*Dhriti Vaz · Summer Research in Progress (SRIP) · June 2026*

---

## Abstract

Large language models (LLMs) generate factually incorrect statements — commonly called hallucinations — at rates that limit their reliability in high-stakes applications. We present a bio-inspired, cascade-structured hallucination suppression pipeline that combines four independently motivated detection signals: token-level confidence scoring (GPT-2 log-probability and attention entropy), named entity recognition (NER) routing, external evidence retrieval (Wikidata/Wikipedia), and natural language inference (NLI) contradiction detection. Inspired by the staged escalation of the mammalian immune system from innate to adaptive response, the cascade resolves 80% of queries at early stages and requires expensive NLI computation for only 20%, achieving approximately 2× latency reduction over a flat pipeline with negligible accuracy loss (F1 39.4% vs. 39.2%). All pipeline decisions are recorded on a two-layer blockchain architecture providing tamper-evident governance with 100% audit completeness across 200 evaluation samples. A controlled retrieval-augmented generation (RAG) detection baseline demonstrates the added value of our multi-signal approach over simple retrieval overlap, which achieves comparable recall only at an unacceptably high false-positive rate (91.4%). We evaluate on TruthfulQA [Lin et al., 2022] with LLaMA-2-7B [Touvron et al., 2023] as the generator and gpt-oss-120b as the judge, achieving a detection rate of 62.3% on a benchmark with 30.5% baseline hallucination rate.

---

## 1. Introduction

Large language models exhibit a well-documented tendency to produce confident-sounding but factually incorrect outputs [Ji et al., 2023; Maynez et al., 2020]. This phenomenon — hallucination — arises from the autoregressive training objective, which optimizes for fluency and contextual coherence rather than factual accuracy. On TruthfulQA [Lin et al., 2022], a benchmark designed to elicit hallucinations through questions that exploit common misconceptions, LLaMA-2-7B hallucinates 30.5% of the time (61/200 answers judged FALSE by gpt-oss-120b), and Mistral-7B hallucinates 17.0% of the time under identical evaluation conditions.

Existing approaches to hallucination mitigation fall broadly into two categories. *Preventive* approaches, such as retrieval-augmented generation (RAG) [Lewis et al., 2020], condition the LLM's generation on retrieved context, reducing — but not eliminating — hallucination. *Detective* approaches post-process generated text using external knowledge to identify factual inconsistencies. Both approaches face a fundamental challenge: the heterogeneity of factual claims. Different entity types (persons, quantities, events, laws) require different verification strategies, and no single detection mechanism covers all categories with equal accuracy.

We observe a structural parallel between this challenge and the vertebrate immune system. The immune system does not apply a single, uniform detection mechanism to all pathogens; instead, it operates a *staged cascade* that escalates from fast, broad-spectrum innate responses to slow, highly specific adaptive responses. The innate system provides rapid, general-purpose threat detection at low metabolic cost; the adaptive system provides precise, antigen-specific response when the innate signal is insufficient. Crucially, most threats are resolved at the innate stage, reserving the expensive adaptive machinery for genuinely uncertain cases.

We propose mapping this cascade structure directly onto hallucination detection:

| Immune Stage | Detection Analog | Mechanism |
|---|---|---|
| Pattern recognition receptors (innate) | Confidence scorer | GPT-2 token log-probability + attention entropy |
| Antigen presentation | NER routing | spaCy entity extraction → verification strategy |
| Lymphocyte activation | Evidence retrieval | Wikidata + Wikipedia fallback chain |
| Effector cell response (adaptive) | NLI contradiction | RoBERTa-large-MNLI zero-shot classification |
| Immunological memory | Blockchain audit | Two-layer smart contract governance |

This biological framing has concrete computational consequences. The "innate" confidence scorer resolves 13.5% of queries with sub-second latency; the "no-evidence" gate handles another 67%; only 19.5% reach the expensive NLI stage. The result is a 1.99× speedup over a flat pipeline with essentially unchanged accuracy.

A governance layer records every pipeline decision on an Ethereum-compatible blockchain, providing an immutable, reconstructible audit trail. In the context of enterprise LLM deployment, this addresses the regulatory requirement for explainability and accountability [Doshi-Velez & Kim, 2017].

### 1.1 Contributions

1. **Bio-inspired cascade architecture**: A three-stage hallucination detection pipeline with biologically motivated stage transitions, achieving 1.99× latency reduction over a flat pipeline at negligible accuracy cost (ΔF1 = +0.2%).

2. **NER-guided verification routing**: A claim routing system that maps detected entity types to appropriate verification strategies (Wikidata entity lookup, structured fact query, Wikipedia text search, keyword fallback), improving detection rate over keyword-only retrieval by 8 percentage points (18.0% → 26.2% DR).

3. **Two-layer blockchain governance**: Immutable Layer-1 audit log (AuditLog.sol) combined with mutable Layer-2 governance decisions (GovernanceDecision.sol), validated with 100% audit completeness across 200 evaluated pipeline runs.

4. **Controlled RAG baseline**: Comparison against TF-IDF retrieval overlap detection on identical inputs and judge labels, demonstrating that simple retrieval overlap requires a 91.4% false-positive rate to achieve comparable recall, motivating the multi-signal design.

5. **Threshold sensitivity analysis**: A 49-point grid sweep of both governance thresholds, identifying the production-viable operating region (FPR < 50%) and the confidence threshold as the dominant control variable.

---

## 2. Related Work

### 2.1 Hallucination in Large Language Models

The phenomenon of LLM hallucination has been extensively documented [Ji et al., 2023; Bang et al., 2023]. Survey work categorizes hallucinations as intrinsic (contradicting the source) or extrinsic (unverifiable against the source) [Maynez et al., 2020]. On open-domain question answering, hallucination rates of 20–40% have been reported for instruction-tuned 7B models [Lin et al., 2022; Touvron et al., 2023]. Our LLaMA-2-7B baseline of 30.5% on TruthfulQA/main aligns with published figures (MC2 = 0.452, compared to published range [0.39, 0.45]).

Mitigation strategies include: RLHF-based truthfulness alignment [Ouyang et al., 2022], decoding-time interventions [Chuang et al., 2023], and chain-of-thought prompting [Wei et al., 2022]. Our work focuses on post-hoc detection rather than prevention, as detection can be applied to any existing model without retraining and provides an audit record of identified errors.

### 2.2 Fact-Checking and Natural Language Inference

Automated fact-checking using NLI has been studied since the FEVER dataset [Thorne et al., 2018] introduced evidence-supported claim verification. Subsequent work [Popat et al., 2018; Wadden et al., 2020] established the pipeline of claim extraction → evidence retrieval → NLI classification as the dominant paradigm. RoBERTa-large fine-tuned on MNLI [Williams et al., 2018; Liu et al., 2019] achieves strong zero-shot NLI performance [Yin et al., 2019], which we exploit for contradiction detection without additional fine-tuning.

Our contribution extends this paradigm with NER-based routing: rather than applying a single retrieval strategy to all claims, we map entity types to the most appropriate external knowledge source. This improves detection rate by 8 pp over entity-agnostic keyword retrieval (C2 → C3 in our ablation).

### 2.3 Retrieval-Augmented Generation

RAG [Lewis et al., 2020] conditions generation on retrieved passages, reducing hallucination intrinsically. Subsequent variants [Izacard et al., 2021; Shi et al., 2023] improve retrieval quality and grounding. However, RAG does not eliminate hallucination: residual errors arise from retrieval failures, out-of-distribution questions, and parametric knowledge overriding retrieved context.

Our RAG detection baseline (Section 4.4) shows that using TF-IDF overlap between an answer and its retrieved context as a hallucination signal achieves DR = 98.4% but FPR = 91.4% at the F1-maximizing threshold. This confirms that simple retrieval overlap is insufficient as a detection mechanism and motivates the additional NLI and confidence signals in our pipeline.

### 2.4 Confidence Estimation in Neural Models

Token-level probability as a proxy for factual accuracy has been studied in the context of calibration [Guo et al., 2017] and selective prediction [Geifman & El-Yaniv, 2017]. Attention entropy as a confidence signal was proposed by [Voita et al., 2019] in the context of head pruning. We combine both signals via a lightweight GPT-2 scorer [Radford et al., 2019] that runs in 210 ms on CPU, making it suitable for the fast innate-stage gate.

The critical observation from our ablation (Table 3) is that the confidence signal alone achieves DR = 50.8% — capturing hallucinations that evidence-based methods miss because they have no retrievable external evidence — while the NLI signal adds 12 additional true positives. The union (C4) outperforms either signal alone.

### 2.5 Bio-Inspired Computing

Artificial immune systems (AIS) apply immunological principles to anomaly detection and pattern recognition [Forrest et al., 1994; Hofmeyr & Forrest, 2000; de Castro & Timmis, 2002]. Classic AIS models include negative selection (self/non-self discrimination) and clonal selection (adaptive amplification of high-affinity detectors). Our work adapts the conceptual framework of immune cascade escalation — rather than specific AIS algorithms — to the pipeline architecture.

The cascade structure we implement directly maps to the immune system's priority ordering: fast, non-specific innate mechanisms first; slow, specific adaptive mechanisms only when needed. This differs from prior AIS-based NLP work [Wang et al., 2007; Bezerra & Barra, 2010], which applies AIS algorithms to specific classification tasks. Our contribution is architectural: using the immune cascade as a design principle for a multi-stage NLP pipeline.

### 2.6 Blockchain in AI Governance

Blockchain-based audit logging for AI systems has been proposed as a solution to explainability and accountability requirements [Dinh et al., 2018; Salah et al., 2019; Mamoshina et al., 2018]. The immutability of blockchain records makes them attractive for recording model decisions in regulated environments. Smart contracts enable automated governance rules on top of immutable logs.

Prior work on blockchain-AI integration focuses primarily on federated learning audit [Warnat-Herresthal et al., 2021] and data provenance [Ahmed et al., 2020]. Our architecture applies this to per-decision hallucination detection audit, providing a two-layer design that separates immutable records (Layer 1) from mutable governance verdicts (Layer 2), supporting human-in-the-loop override while preserving the original detection record.

---

## 3. Biological Inspiration: The Immune Cascade Model

The vertebrate immune system is among the most sophisticated multi-stage pattern recognition systems in nature. Its relevance to hallucination detection stems not from superficial analogy but from a structural isomorphism: both problems involve distinguishing harmful from benign outputs (pathogens vs. normal cells; hallucinations vs. truthful answers) using multiple detection mechanisms of varying speed and specificity.

### 3.1 Immune System Architecture

The immune response is organized into two functionally distinct subsystems:

**Innate immunity** responds within minutes to hours using germline-encoded pattern recognition receptors (PRRs) such as Toll-like receptors. PRRs recognize pathogen-associated molecular patterns (PAMPs) — conserved structural features shared across broad categories of pathogens. The innate response is fast (sub-minute) and broad-spectrum, providing rapid threat containment while the slower adaptive system activates. Crucially, PRR activation follows a cascade: complement activation, cytokine release, and neutrophil recruitment occur in a specific temporal order, with each stage triggered by the preceding one.

**Adaptive immunity** responds over days to weeks through antigen-specific lymphocytes (T and B cells). Antigen-presenting cells (APCs) capture foreign material, process it into peptide fragments, and present them via MHC complexes to T cells. This antigen presentation step is the critical routing mechanism: it determines which lymphocyte clones are activated, analogous to determining which verification strategy to apply to a given entity type. Clonal selection and expansion then amplify the specific response. Immunological memory ensures that repeat encounters trigger faster, stronger responses.

**Regulatory mechanisms** (regulatory T cells, cytokine feedback, apoptosis) prevent overreaction, which is the immune analog of our governance layer's ability to override false-positive detections.

### 3.2 Mapping to Hallucination Detection

| Biological Component | Pipeline Component | Implementation |
|---|---|---|
| PRRs (innate) | Confidence Gate | GPT-2 log-probability + attention entropy |
| Innate activation threshold | Early-reject/accept thresholds | conf < 0.50 (flag); conf > 0.57 (accept) |
| Antigen presentation (APC) | NER router | spaCy en_core_web_sm entity extraction |
| MHC peptide selection | Verification strategy routing | Entity type → {entity\_lookup, structured\_fact, text\_search, keyword\_search} |
| Lymphocyte activation | Evidence retrieval | Wikidata REST + Wikipedia MediaWiki API |
| Effector response | NLI contradiction | RoBERTa-large-MNLI zero-shot classification |
| Regulatory T cells | Governance override | Layer-2 smart contract `overrideDecision()` |
| Immunological memory | Blockchain audit | AuditLog.sol immutable record |
| Clonal expansion cascade | Stage escalation | Resolve 80% at early stages; only 20% reach NLI |

### 3.3 The Staged Escalation Principle

The central computational insight is that the most informative (and expensive) detection mechanisms should be applied selectively. In the immune system, the metabolic cost of mounting a full adaptive response is substantial; the innate system acts as a gating mechanism, escalating only genuine threats. The cascade structure achieves this:

1. **Stage 1 (Confidence Gate, 210 ms):** The GPT-2 confidence scorer provides a fast, universal signal. Samples with confidence below 0.50 or above 0.57 are resolved immediately — 13.5% of all inputs. These are the extreme cases where the model's own internal uncertainty is definitive enough to warrant immediate action.

2. **Stage 2 (Evidence Gate, 1,532 ms average):** For the remaining 86.5%, NER routing and external retrieval are applied. Because 77% of LLaMA-2 answers about TruthfulQA topics have no external evidence retrievable from Wikidata or Wikipedia, the majority (67% of all samples) exit here. Without evidence, the confidence score alone determines the verdict.

3. **Stage 3 (NLI Gate, 4,417 ms average):** Only 19.5% of samples — those with retrievable evidence — reach full NLI contradiction detection. These are the cases where the model discusses verifiable facts about specific named entities, and external validation is possible.

The resulting computational profile mirrors immune economy: 80.5% of inputs are resolved without NLI, preserving the expensive adaptive response for the minority of cases where it adds value.

### 3.4 Cascade vs. Non-Cascade Behavior

Running all four stages on all inputs (flat C4 pipeline) costs an average of 3,822 ms per sample. The cascade reduces this to 1,916 ms — a 1.99× speedup — while achieving F1 = 39.4% vs. 39.2% (ΔF1 = +0.2%). The marginal accuracy cost is negligible because:

- Stage 1 early-reject samples (24 samples, 12.0% of inputs) are predominantly low-confidence, with 6 of 24 being true hallucinations (25% precision at Stage 1 alone — comparable to overall pipeline precision of 28.6%).
- Stage 2 no-evidence samples (134 samples, 67.0%) produce NLI scores of exactly 0.0 in the flat pipeline (no evidence → zero-shot NLI is undefined), so running NLI on them adds no information.

This is the biological parallel's deepest insight: the expensive adaptive mechanism only adds value when the innate + antigen-presentation stages have identified a genuine, specific target. When evidence is absent, the innate confidence score is the best available signal.

---

## 4. NLP Foundations

### 4.1 TruthfulQA and the Hallucination Evaluation Problem

TruthfulQA [Lin et al., 2022] contains 817 questions designed to elicit common misconceptions, spanning categories including health, history, law, science, and conspiracy theories. Questions are crafted so that "the truthful answer contradicts a false belief that many humans hold." This makes TruthfulQA particularly valuable for studying hallucination: it targets the type of confident-sounding errors that characterize LLM failures, not merely out-of-distribution uncertainty.

We use the MC2 (Multiple Choice, version 2) metric as a baseline for comparison with published results: Mistral-7B achieves MC2 = 0.503 (published range [0.42, 0.50]) and LLaMA-2-7B achieves MC2 = 0.452 (published range [0.39, 0.45]). For our evaluation, we use free-form generation rather than multiple-choice to avoid format artifacts in the judge evaluation.

**Judge selection.** We evaluate two LLM judges: gpt-oss-120b (via OpenRouter) and Mistral-7B-Instruct (via Ollama). Cohen's κ between the two judges on 100 shared samples is 0.054 — indicating near-chance agreement despite 66% raw agreement, driven by label imbalance. This low inter-rater reliability is consistent with the difficulty of factuality evaluation and has been reported in other automatic evaluation studies [Wang et al., 2023]. We select gpt-oss-120b as the reference judge, which labels 30.5% of LLaMA-2-7B main-split answers as hallucinations (61/200).

### 4.2 Named Entity Recognition and Routing

Named entity recognition (NER) with spaCy's `en_core_web_sm` model identifies entity mentions in each extracted claim. We define four verification strategies based on entity type:

| Entity Types | Strategy | Rationale |
|---|---|---|
| PERSON, ORG, GPE, LOC, FAC | `entity_lookup` | Direct Wikidata entity search provides factual biography or description |
| DATE, TIME, QUANTITY, CARDINAL | `structured_fact` | Wikidata property queries support numerical and temporal verification |
| EVENT, WORK\_OF\_ART, LAW, NORP | `text_search` | Wikipedia fulltext captures narrative and contextual information |
| *(no entities)* | `keyword_search` | Content-word fallback search when NER finds nothing |

Strategy routing is evaluated in our ablation: C3 (NER-routed) achieves DR = 26.2% vs. C2 (keyword-only) DR = 18.0%, an 8 pp improvement from better evidence targeting.

### 4.3 Evidence Retrieval and the No-Evidence Problem

Evidence retrieval follows a four-step fallback chain: (1) Wikidata entity search via the `wbsearchentities` API; (2) Wikipedia page extraction via `extracts` prop; (3) Wikipedia fulltext search; (4) no evidence. The Wikidata reliability diagnostic (Week 4) measures the success rate across 200 queries: a substantial fraction of claims generate no retrievable evidence, reflecting either the absence of the relevant entity in Wikidata or the claim's non-factual nature (speculative, subjective, or creative content).

The **no-evidence problem** is the dominant limiting factor in our pipeline. On the 200 LLaMA-2 evaluation samples, 77% have no retrieved evidence (154/200), and only 18.5% with evidence show non-zero TF-IDF overlap with the answer text. This explains why the contradiction detector alone (C3, 26.2% DR) underperforms the confidence-only approach (C1, 50.8% DR): when evidence is absent, NLI is silent. The full pipeline (C4) achieves higher DR (62.3%) by combining both signals via logical OR: flag if contradiction detected OR confidence is low.

### 4.4 Natural Language Inference for Contradiction Detection

We use `roberta-large-mnli` [Liu et al., 2019] via the HuggingFace `zero-shot-classification` pipeline. For each (evidence, claim) pair, the model classifies whether the evidence *entails*, is *neutral* toward, or *contradicts* the claim. We use the zero-shot hypothesis template: "This text _{label}s_ the statement: _{claim}_", applied with `multi_label=False` to obtain a normalized probability distribution over the three labels.

The MNLI-based approach is chosen over FEVER-fine-tuned models [Thorne et al., 2018] because: (a) it generalizes to open-domain evidence without domain-specific fine-tuning; (b) RoBERTa-large-MNLI is publicly available and runs on CPU within budget constraints; (c) zero-shot classification handles the diversity of evidence types (Wikidata descriptions, Wikipedia abstracts) without format adaptation.

The contradiction threshold (τ_c = 0.02) is calibrated on our evaluation set: the distribution of max contradiction scores is bimodal with 80% of samples at exactly 0.000 (no evidence → zero-shot returns 0 score) and the positive tail ranging to 0.357. A threshold of 0.02 captures any non-trivial NLI signal.

### 4.5 Confidence Scoring with GPT-2

We score every generated answer with GPT-2 (124M parameters) as a *proxy* model — not the generator. The scorer computes two signals:

**Token log-probability:** Mean per-token log P(t_i | t_{<i}) under GPT-2 using teacher-forced decoding. Low mean log-probability indicates the model assigns low likelihood to the text, which we interpret as a weak hallucination signal under the assumption that factually plausible text is more likely under a strong language model.

**Attention entropy:** Mean entropy of attention distributions across all 12 GPT-2 layers and 12 heads. High entropy (diffuse attention) indicates the model cannot form focused token associations, which correlates with semantic incoherence.

Combined score: conf = 0.5 × σ(mean\_log\_prob / 3.0) + 0.5 × (1 − H\_normalized)

where σ is the sigmoid function and H\_normalized = attention\_entropy / 6.0. Scores range [0, 1]; higher indicates higher confidence. On our evaluation set, scores range [0.483, 0.599] with a mean of approximately 0.531.

GPT-2 runs on CPU in approximately 210 ms per answer, making it suitable for the fast innate-stage gate. The threshold τ_c = 0.52 (default) yields C1 metrics of DR = 50.8%, FPR = 63.3%. Threshold sensitivity analysis (Section 6.2) shows that the C1 stable region (F1 within 5% of maximum) spans conf ∈ [0.546, 0.610], indicating robustness over a 64 milli-unit range.

---

## 5. System Architecture

### 5.1 Overview

The full system consists of five components arranged in the cascade described in Section 3:

```
                    [Question] → [LLM Generator] → [Answer]
                                                       │
                    ┌─────────────────────────────────┤
                    │                                 ↓
                    │          [Claim Extraction (spaCy sentence segmentation)]
                    │                                 │
                    │                    ┌────────────┴────────────┐
                    │                    ↓                         │
                    │           [Stage 1: Confidence Gate]          │
                    │        conf < 0.50 → REJECT                  │
                    │        conf > 0.57 → ACCEPT                  │
                    │        else → Stage 2 ↓                      │
                    │                    │                         │
                    │           [Stage 2: Evidence Gate]           │
                    │        [NER Router] → [Evidence Retrieval]   │
                    │        no evidence → conf < 0.52 → REJECT    │
                    │        evidence found → Stage 3 ↓            │
                    │                    │                         │
                    │           [Stage 3: NLI Gate]                │
                    │        contra ≥ 0.02 → REJECT                │
                    │        else → ACCEPT                         │
                    │                    │                         │
                    └────────────────────┴─────────────────────────┘
                                         │
                              [Pipeline Decision: flagged/clean]
                                         │
                    ┌────────────────────┴──────────────────────────┐
                    ↓                                               ↓
         [Layer 1: AuditLog.sol]                   [Layer 2: GovernanceDecision.sol]
       (immutable: question_hash,                (mutable: flagged, conf_score,
        answer_hash, pipeline_version)            contra_score, override support)
```

### 5.2 Claim Extraction

Input answers are segmented into claims using spaCy sentence segmentation (`en_core_web_sm`). LLaMA-2-7B answers average 2.41 claims per answer (Week 3 statistics), and we cap pipeline processing at 4 claims per answer for computational tractability. Each claim is processed independently through Stage 2 and Stage 3; the sample-level decision is the logical OR of all claim-level decisions (any flagged claim → flag the answer).

### 5.3 Stage 1: Confidence Gate

The confidence scorer runs once per answer (not per claim). GPT-2 is loaded once at pipeline startup and processes answers in approximately 210 ms on CPU. The two outputs — `token_log_prob` and `attention_entropy` — are combined into a single `confidence` score in [0, 1].

**Early exit conditions:**
- `confidence < CONF_EARLY_REJECT (0.50)`: answer flagged immediately; NER and NLI skipped
- `confidence > CONF_EARLY_ACCEPT (0.57)`: answer accepted immediately; NER and NLI skipped
- otherwise: answer forwarded to Stage 2

In our evaluation, 12.0% of answers trigger early rejection (24/200) and 1.5% trigger early acceptance (3/200), leaving 86.5% (173/200) to proceed.

### 5.4 Stage 2: Evidence Gate

**NER routing:** spaCy extracts entity mentions from each claim. The highest-priority entity type present determines the verification strategy per Table 2. If no entities are found, `keyword_search` is used.

**Evidence retrieval:** The retrieval chain contacts Wikidata (via `wbsearchentities` API) for entity lookup; falls back to Wikipedia page extraction (MediaWiki `extracts` API); then to Wikipedia fulltext search; then records no-evidence. Each call has a 10-second timeout with 2 retries.

**Exit condition:** If no evidence is retrieved for any claim, the answer exits at Stage 2 with the confidence-only verdict (conf < 0.52 → flag). In our evaluation, 67.0% of answers exit at Stage 2 due to no evidence.

### 5.5 Stage 3: NLI Gate

For answers with at least one claim supported by retrieved evidence, `roberta-large-MNLI` scores each (evidence, claim) pair. The maximum `contradiction_score` across all claims is compared to τ_c = 0.02. An answer is flagged if any claim's contradiction score meets or exceeds this threshold.

RoBERTa-large runs on CPU in approximately 890 ms per (evidence, claim) pair. In our evaluation, 19.5% of answers (39/200) reach Stage 3.

### 5.6 Two-Layer Blockchain Governance

Each pipeline run produces two on-chain records:

**Layer 1 — AuditLog (immutable):** Deployed as `AuditLog.sol` (Solidity 0.8.28). Stores `(runId, timestamp, sha256(question), sha256(answer), pipelineVersion)` in an append-only array. Gas cost: approximately 164,753 gas / write. Write latency: 35.7 ms mean on local py-evm; estimated ~12 s on Sepolia testnet (1-block confirmation).

**Layer 2 — GovernanceDecision (mutable):** Deployed as `GovernanceDecision.sol`. Stores the pipeline verdict with confidence and contradiction scores. Supports `overrideDecision()` for human-in-the-loop correction. Gas cost: approximately 170,823 gas / write. Write latency: 43.6 ms mean local.

**Audit completeness:** Reconstruction test across all 200 pipeline runs achieves 100% completeness — all six auditable fields (question hash, answer hash, pipeline version, flagged decision, confidence score, contradiction score) are recoverable from chain records alone, with no access to the original corpus.

**Governance value:** Threshold sensitivity analysis (Section 6.2) identifies 14 C4 configurations with FPR < 50%, with the best achieving DR = 36.1%, FPR = 28.1%, F1 = 36.1% (at conf\_t = 0.50, contra\_t = 0.20). At the default operating point (conf\_t = 0.52), FPR = 68.3% — meaning 68% of truthful answers are incorrectly flagged. The governance override mechanism enables human review of flagged answers, providing a practical path to deployment despite the high FPR.

### 5.7 Implementation Details

All pipeline components are implemented in Python 3.9. Key dependencies:

| Component | Library | Version |
|---|---|---|
| Confidence scorer | HuggingFace Transformers (GPT-2) | 4.57.6 |
| NER router | spaCy en\_core\_web\_sm | 3.8.14 |
| Evidence retrieval | requests (Wikidata/Wikipedia APIs) | 2.34.1 |
| NLI contradiction | HuggingFace Transformers (RoBERTa) | 4.57.6 |
| Blockchain compilation | py-solc-x + solc 0.8.28 | 2.0.5 |
| Blockchain runtime | web3.py + eth-tester (py-evm) | 6.20.4 |
| RAG baseline | scikit-learn TfidfVectorizer | 1.6.1 |

The generator (LLaMA-2-7B-GGUF, Q4 quantization) runs via llama-cpp-python on CPU, producing all 500 generations (250 LLaMA-2 + 250 Mistral) offline before pipeline evaluation.

---

*[End of Sections 1–5. Results (Section 6), Limitations (Section 7), Future Work (Section 8), and Conclusion (Section 9) to be written in Week 8 using the complete measured numbers.]*

---

## Appendix A: Key Measured Numbers (Weeks 2–7)

| Metric | Value | Source |
|---|---|---|
| LLaMA-2-7B MC2 (TruthfulQA) | 0.452 | Week 2 |
| Mistral-7B MC2 (TruthfulQA) | 0.503 | Week 2 |
| LLaMA-2-7B hallucination rate (gpt-oss) | 30.5% (61/200) | Week 3 |
| Mistral-7B hallucination rate (gpt-oss) | 17.0% (34/200) | Week 3 |
| Inter-judge Cohen's κ | 0.054 | Week 3 |
| C0 (no detection) DR / FPR / F1 | 0.0% / 0.0% / 0.0% | Week 5 |
| C1 (confidence only) DR / FPR / F1 | 50.8% / 63.3% / 34.4% | Week 5 |
| C2 (keyword + NLI) DR / FPR / F1 | 18.0% / 20.9% / 21.8% | Week 5 |
| C3 (NER + NLI) DR / FPR / F1 | 26.2% / 21.6% / 29.9% | Week 5 |
| C4 (full pipeline) DR / FPR / F1 | 62.3% / 68.3% / 39.2% | Week 5 |
| NER routing improvement (C2→C3 DR) | +8.2 pp | Week 5 |
| Confidence threshold range | [0.483, 0.599] | Week 5 |
| Contradiction score range | [0.000, 0.357] | Week 5 |
| No-evidence rate | 77.0% (154/200) | Week 6 |
| RAG-TF-IDF best F1 / FPR at best F1 | 48.4% / 91.4% | Week 6 |
| Blockchain L1 write time (local) | 35.7 ms mean | Week 6 |
| Blockchain L2 write time (local) | 43.6 ms mean | Week 6 |
| Blockchain total write per run (local) | 79.3 ms mean | Week 6 |
| Blockchain gas (L1 + L2) per run | ~335,651 | Week 6 |
| Testnet expected confirmation | ~12 s (1 block, Sepolia) | Week 6 |
| Audit completeness rate | 100.0% (200/200) | Week 6 |
| Threshold sensitivity: C1 stable region | conf ∈ [0.546, 0.610] | Week 7 |
| Threshold sensitivity: C3 stable region | contra ∈ [0.001, 0.224] | Week 7 |
| Production-viable C4 configs (FPR < 50%) | 14 of 49 | Week 7 |
| Best production-viable: DR / FPR / F1 | 36.1% / 28.1% / 36.1% | Week 7 |
| Cascade Stage 1 exit rate | 13.5% (27/200) | Week 7 |
| Cascade Stage 2 exit rate | 67.0% (134/200) | Week 7 |
| Cascade Stage 3 exit rate | 19.5% (39/200) | Week 7 |
| Cascade F1 vs. flat C4 F1 | 39.4% vs. 39.2% (ΔF1 = +0.2%) | Week 7 |
| Cascade speedup over flat C4 | 1.99× (49.9% latency reduction) | Week 7 |
