"""
Week 4 — Evidence Retrieval with Wikidata Reliability Logging + Fallback Chain

Retrieval chain (in order):
  1. Wikidata entity search  → description + aliases via REST API
  2. Wikipedia page extract  → intro paragraph via MediaWiki API
  3. Wikipedia fulltext search → top search hit extract
  4. No evidence             → empty result, logged as "none"

Every attempt is logged to a per-session list so the diagnostic can
compute: success_rate, fallback_rate, no_evidence_rate.

Public API:
    result = retrieve(query, strategy="entity_lookup")
    batch  = retrieve_batch(queries_with_strategies)
"""

from __future__ import annotations
import time
import re
import json
import requests
from dataclasses import dataclass, field
from typing import Optional

WIKIDATA_API  = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

_REQUEST_TIMEOUT = 10   # seconds per HTTP call
_RETRY_DELAY     = 1.0  # seconds between retries
_MAX_RETRIES     = 2


# ── result dataclass ─────────────────────────────────────────────────────────

@dataclass
class EvidenceResult:
    query: str
    strategy: str                 # routing strategy that triggered this call
    source: str                   # "wikidata" | "wikipedia_page" | "wikipedia_search" | "none"
    evidence: str                 # the retrieved text (empty string if none)
    wikidata_qid: Optional[str] = None
    fallback_used: bool = False   # True if wikidata failed and we fell back
    no_evidence: bool = False     # True if all sources failed
    latency_ms: int = 0
    error: Optional[str] = None


# ── session-level reliability log ────────────────────────────────────────────

_log: list[dict] = []


def get_log() -> list[dict]:
    return list(_log)


def clear_log() -> None:
    _log.clear()


def _record(result: EvidenceResult) -> None:
    _log.append({
        "query":         result.query,
        "strategy":      result.strategy,
        "source":        result.source,
        "fallback_used": result.fallback_used,
        "no_evidence":   result.no_evidence,
        "latency_ms":    result.latency_ms,
        "evidence_len":  len(result.evidence),
        "wikidata_qid":  result.wikidata_qid,
    })


# ── HTTP helpers ─────────────────────────────────────────────────────────────

def _get(url: str, params: dict, retries: int = _MAX_RETRIES) -> Optional[dict]:
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT,
                             headers={"User-Agent": "HallucinationSuppressPipeline/1.0"})
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries:
                return None
            time.sleep(_RETRY_DELAY * (attempt + 1))
    return None


# ── Wikidata retrieval ───────────────────────────────────────────────────────

def _wikidata_search(query: str) -> Optional[tuple[str, str]]:
    """Search Wikidata for *query*. Returns (qid, description) or None."""
    data = _get(WIKIDATA_API, {
        "action":   "wbsearchentities",
        "search":   query,
        "language": "en",
        "format":   "json",
        "limit":    3,
    })
    if not data:
        return None
    results = data.get("search", [])
    if not results:
        return None
    top = results[0]
    qid  = top.get("id", "")
    desc = top.get("description", "")
    label = top.get("label", query)
    if not desc:
        # try fetching entity description directly
        desc = _wikidata_entity_description(qid) or ""
    text = f"{label}: {desc}" if desc else label
    return qid, text


def _wikidata_entity_description(qid: str) -> Optional[str]:
    data = _get(WIKIDATA_API, {
        "action":   "wbgetentities",
        "ids":      qid,
        "props":    "descriptions|labels",
        "languages": "en",
        "format":   "json",
    })
    if not data:
        return None
    entity = data.get("entities", {}).get(qid, {})
    desc = entity.get("descriptions", {}).get("en", {}).get("value", "")
    return desc or None


# ── Wikipedia retrieval ──────────────────────────────────────────────────────

def _wikipedia_page_extract(title: str) -> Optional[str]:
    """Fetch the intro paragraph of a Wikipedia page."""
    data = _get(WIKIPEDIA_API, {
        "action":      "query",
        "prop":        "extracts",
        "exintro":     True,
        "explaintext": True,
        "redirects":   True,
        "titles":      title,
        "format":      "json",
    })
    if not data:
        return None
    pages = data.get("query", {}).get("pages", {})
    for pid, page in pages.items():
        if pid == "-1":
            return None
        extract = page.get("extract", "").strip()
        if extract:
            return extract[:1000]   # cap at 1000 chars
    return None


def _wikipedia_search(query: str) -> Optional[str]:
    """Search Wikipedia and return the intro of the top hit."""
    data = _get(WIKIPEDIA_API, {
        "action":   "query",
        "list":     "search",
        "srsearch": query,
        "srlimit":  1,
        "format":   "json",
    })
    if not data:
        return None
    results = data.get("query", {}).get("search", [])
    if not results:
        return None
    title = results[0]["title"]
    return _wikipedia_page_extract(title)


# ── main retrieval function ──────────────────────────────────────────────────

def retrieve(query: str, strategy: str = "entity_lookup") -> EvidenceResult:
    """
    Retrieve evidence for *query* using the fallback chain.
    *strategy* is the routing strategy from ner_router (used for logging only).
    """
    t0 = time.time()

    # ── step 1: Wikidata ─────────────────────────────────────────────────────
    wd = _wikidata_search(query)
    if wd:
        qid, text = wd
        latency = int((time.time() - t0) * 1000)
        result = EvidenceResult(
            query=query, strategy=strategy,
            source="wikidata", evidence=text,
            wikidata_qid=qid, fallback_used=False,
            no_evidence=False, latency_ms=latency,
        )
        _record(result)
        return result

    # ── step 2: Wikipedia page (exact title match) ────────────────────────────
    extract = _wikipedia_page_extract(query)
    if extract:
        latency = int((time.time() - t0) * 1000)
        result = EvidenceResult(
            query=query, strategy=strategy,
            source="wikipedia_page", evidence=extract,
            fallback_used=True, no_evidence=False, latency_ms=latency,
        )
        _record(result)
        return result

    # ── step 3: Wikipedia fulltext search ────────────────────────────────────
    extract = _wikipedia_search(query)
    if extract:
        latency = int((time.time() - t0) * 1000)
        result = EvidenceResult(
            query=query, strategy=strategy,
            source="wikipedia_search", evidence=extract,
            fallback_used=True, no_evidence=False, latency_ms=latency,
        )
        _record(result)
        return result

    # ── step 4: no evidence ───────────────────────────────────────────────────
    latency = int((time.time() - t0) * 1000)
    result = EvidenceResult(
        query=query, strategy=strategy,
        source="none", evidence="",
        fallback_used=True, no_evidence=True, latency_ms=latency,
    )
    _record(result)
    return result


def retrieve_batch(
    queries: list[str],
    strategies: Optional[list[str]] = None,
    delay: float = 0.3,
) -> list[EvidenceResult]:
    """Retrieve evidence for multiple queries with rate limiting."""
    if strategies is None:
        strategies = ["entity_lookup"] * len(queries)
    results = []
    for q, s in zip(queries, strategies):
        results.append(retrieve(q, strategy=s))
        time.sleep(delay)
    return results


# ── reliability summary ───────────────────────────────────────────────────────

def reliability_summary(log: Optional[list[dict]] = None) -> dict:
    entries = log if log is not None else _log
    n = len(entries)
    if n == 0:
        return {"n": 0, "success_rate": 0.0, "fallback_rate": 0.0, "no_evidence_rate": 0.0}

    wikidata_hits  = sum(1 for e in entries if e["source"] == "wikidata")
    wiki_page_hits = sum(1 for e in entries if e["source"] == "wikipedia_page")
    wiki_srch_hits = sum(1 for e in entries if e["source"] == "wikipedia_search")
    no_ev          = sum(1 for e in entries if e["no_evidence"])

    fallback_total = wiki_page_hits + wiki_srch_hits
    avg_latency    = sum(e["latency_ms"] for e in entries) / n

    return {
        "n":                   n,
        "wikidata_hits":        wikidata_hits,
        "wikipedia_page_hits":  wiki_page_hits,
        "wikipedia_search_hits": wiki_srch_hits,
        "no_evidence":          no_ev,
        "success_rate":         round(wikidata_hits / n, 4),
        "fallback_rate":        round(fallback_total / n, 4),
        "no_evidence_rate":     round(no_ev / n, 4),
        "avg_latency_ms":       round(avg_latency, 1),
    }


if __name__ == "__main__":
    test_queries = [
        ("Albert Einstein",     "entity_lookup"),
        ("Eiffel Tower",        "entity_lookup"),
        ("speed of light",      "structured_fact"),
        ("Hamlet Shakespeare",  "text_search"),
        ("xyzzy_nonexistent_term_abc", "keyword_search"),
    ]
    print("Evidence Retrieval — smoke test")
    print("=" * 55)
    for q, strat in test_queries:
        r = retrieve(q, strategy=strat)
        print(f"\n  Query   : {q}")
        print(f"  Source  : {r.source}  (fallback={r.fallback_used}  no_ev={r.no_evidence})")
        print(f"  QID     : {r.wikidata_qid}")
        print(f"  Evidence: {r.evidence[:120]}...")

    print("\n\nReliability summary:", json.dumps(reliability_summary(), indent=2))
