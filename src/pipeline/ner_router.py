"""
Week 4 — NER + Routing Layer

Extracts named entities from a claim using spaCy and decides which
verification strategy to apply:
  - PERSON, ORG, GPE, LOC, FAC    → entity_lookup  (Wikidata entity search)
  - DATE, TIME, QUANTITY, CARDINAL → structured_fact (Wikidata property search)
  - EVENT, WORK_OF_ART, LAW, NORP → text_search    (Wikipedia fulltext)
  - no entities found              → keyword_search (Wikipedia keyword fallback)

Returns a RoutingDecision with extracted entities, their types, the chosen
strategy, and keyword fallback terms.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional
import spacy

# ── spaCy entity → verification strategy mapping ────────────────────────────

_ENTITY_LOOKUP_TYPES = {"PERSON", "ORG", "GPE", "LOC", "FAC"}
_STRUCTURED_FACT_TYPES = {"DATE", "TIME", "QUANTITY", "CARDINAL", "MONEY", "PERCENT", "ORDINAL"}
_TEXT_SEARCH_TYPES = {"EVENT", "WORK_OF_ART", "LAW", "NORP", "LANGUAGE", "PRODUCT"}

_STRATEGY_PRIORITY = ["entity_lookup", "structured_fact", "text_search", "keyword_search"]


@dataclass
class Entity:
    text: str
    label: str        # spaCy entity label, e.g. "PERSON"
    start: int        # char offset in claim
    end: int


@dataclass
class RoutingDecision:
    claim: str
    entities: list[Entity]
    strategy: str                          # one of _STRATEGY_PRIORITY
    primary_query: Optional[str] = None    # best entity text to look up
    keywords: list[str] = field(default_factory=list)   # fallback keyword terms
    rationale: str = ""


# ── load spaCy once ──────────────────────────────────────────────────────────

_nlp: Optional[spacy.language.Language] = None


def _get_nlp() -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                           check=True)
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ── helpers ──────────────────────────────────────────────────────────────────

def _extract_keywords(text: str) -> list[str]:
    """Return content words from text (stopword-stripped, deduplicated)."""
    nlp = _get_nlp()
    doc = nlp(text)
    seen: set[str] = set()
    kws: list[str] = []
    for tok in doc:
        if tok.is_stop or tok.is_punct or tok.is_space:
            continue
        lemma = tok.lemma_.lower()
        if lemma not in seen and len(lemma) > 2:
            seen.add(lemma)
            kws.append(tok.text)
    return kws[:10]


def _pick_primary_entity(entities: list[Entity]) -> Optional[Entity]:
    """Pick the best entity to use as the primary Wikidata search term."""
    # prefer PERSON > GPE > ORG > LOC > others
    priority = {"PERSON": 0, "GPE": 1, "ORG": 2, "LOC": 3, "FAC": 4}
    candidates = [e for e in entities if e.label in _ENTITY_LOOKUP_TYPES]
    if not candidates:
        candidates = [e for e in entities if e.label in _STRUCTURED_FACT_TYPES]
    if not candidates:
        candidates = entities
    candidates.sort(key=lambda e: priority.get(e.label, 99))
    return candidates[0] if candidates else None


# ── public API ────────────────────────────────────────────────────────────────

def route(claim: str) -> RoutingDecision:
    """
    Run NER on *claim* and return a RoutingDecision describing which
    verification strategy and primary query to use.
    """
    nlp = _get_nlp()
    doc = nlp(claim[:512])

    entities = [
        Entity(text=ent.text, label=ent.label_, start=ent.start_char, end=ent.end_char)
        for ent in doc.ents
    ]

    # determine strategy by entity type priority
    labels = {e.label for e in entities}

    if labels & _ENTITY_LOOKUP_TYPES:
        strategy = "entity_lookup"
    elif labels & _STRUCTURED_FACT_TYPES:
        strategy = "structured_fact"
    elif labels & _TEXT_SEARCH_TYPES:
        strategy = "text_search"
    else:
        strategy = "keyword_search"

    primary_ent = _pick_primary_entity(entities)
    primary_query = primary_ent.text if primary_ent else None
    keywords = _extract_keywords(claim)

    rationale_parts = [f"entities=[{', '.join(f'{e.text}({e.label})' for e in entities)}]"]
    rationale_parts.append(f"strategy={strategy}")
    if primary_query:
        rationale_parts.append(f"primary_query={primary_query!r}")

    return RoutingDecision(
        claim=claim,
        entities=entities,
        strategy=strategy,
        primary_query=primary_query,
        keywords=keywords,
        rationale="; ".join(rationale_parts),
    )


def route_batch(claims: list[str]) -> list[RoutingDecision]:
    return [route(c) for c in claims]


# ── strategy distribution summary ────────────────────────────────────────────

def strategy_summary(decisions: list[RoutingDecision]) -> dict:
    from collections import Counter
    counts = Counter(d.strategy for d in decisions)
    total = len(decisions)
    return {s: {"count": counts[s], "pct": round(counts[s] / total * 100, 1)}
            for s in _STRATEGY_PRIORITY if s in counts}


if __name__ == "__main__":
    test_claims = [
        "Albert Einstein was born in Ulm, Germany in 1879.",
        "The Eiffel Tower is located in Paris.",
        "Poverty in Africa has declined over recent decades.",
        "The speed of light is approximately 299,792 km/s.",
        "Shakespeare wrote Hamlet.",
        "This answer is truthful and concise.",
    ]
    print("NER + Routing Layer — smoke test")
    print("=" * 55)
    for claim in test_claims:
        d = route(claim)
        ents = [(e.text, e.label) for e in d.entities]
        print(f"\n  Claim    : {claim[:60]}")
        print(f"  Entities : {ents}")
        print(f"  Strategy : {d.strategy}")
        print(f"  Query    : {d.primary_query}")
