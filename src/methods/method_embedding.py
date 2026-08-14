"""Method A — Embedding retrieval.

Encode each ORD resource by its (title + shortDescription + description),
encode the activity label, return the Top-k by cosine similarity.
"""

from __future__ import annotations

import math

from src import config
from src import llm


def _resource_text(r: dict) -> str:
    """Compose the embedding input for a resource.

    A is the free-text baseline, so anything that lives as a string in ORD
    is fair game. We append three optional fields when present:
      - partOfGroups   (x1-only enrichment from design-time)
      - processNext    (x1-only enrichment from design-time)
      - capabilities   (handauthored on x0 + derived from BPMN/CMMN labels
                        on x1 — so on enriched ORD this also carries the
                        process-grounded "verb-noun" signal)
    All three are part of the H4 effect: enriched ORD gives A textual
    signal about the business-process context that clean ORD lacks.
    """
    parts = [r["title"], r["shortDescription"], r["description"]]
    groups = r.get("partOfGroups") or []
    if groups:
        names = [g.get("groupId", "") for g in groups]
        parts.append("partOfGroups: " + ", ".join(n for n in names if n))
    nexts = r.get("processNext") or []
    if nexts:
        parts.append("processNext: " + ", ".join(nexts))
    caps = r.get("capabilities") or []
    if caps:
        parts.append("capabilities: " + ", ".join(caps))
    use_cases = r.get("useCases") or []
    if use_cases:
        parts.append("useCases: " + " | ".join(use_cases))
    return " | ".join(p for p in parts if p)


def _cosine(a: list[float], b: list[float]) -> float:
    s = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return s / (na * nb)


def retrieve(label: str, resources: list[dict], top_k: int = config.TOP_K) -> dict:
    """Return {"candidates": [{ordId, score}, ...], "trace": {...}}

    Token accounting reflects production deployment: resource embeddings are
    treated as an amortised landscape-setup cost (paid once, reused across all
    cases), so only the per-case query embedding contributes to the reported
    ``tokens`` field. The landscape cost is reported separately as
    ``landscape_setup_tokens`` for transparency.
    """
    query_vec, q_meta = llm.embed(label)

    scored: list[tuple[float, str]] = []
    landscape_tokens = 0
    total_latency = q_meta["latency"]
    for r in resources:
        vec, m = llm.embed(_resource_text(r))
        landscape_tokens += m["tokens"]
        total_latency += m["latency"]
        scored.append((_cosine(query_vec, vec), r["ordId"]))

    scored.sort(key=lambda t: -t[0])
    candidates = [{"ordId": oid, "score": round(s, 4)} for s, oid in scored[:top_k]]

    return {
        "method": "A",
        "candidates": candidates,
        "trace": {
            "embedding_model": config.EMBEDDING_MODEL,
            "resource_count": len(resources),
            # Per-case cost: only the query embedding.
            # Resource embeddings are a one-time landscape-setup cost, reported below.
            "tokens": q_meta["tokens"],
            "landscape_setup_tokens": landscape_tokens,
            "latency_s": round(total_latency, 3),
        },
    }
