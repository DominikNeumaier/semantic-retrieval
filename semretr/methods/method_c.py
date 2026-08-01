"""Method C — Multi-Hop Graph Walk over the ORD knowledge graph.

ORD encodes a typed graph between resources. This method makes the walk
explicit:

  1. ENTRY POINTS  (1 LLM call)
     LLM picks anchor concept nodes from controlled vocabularies:
       - entity types
       - business process groups
       - lines of business

  2. SEED                       (deterministic, hop 0)
     Resources directly tied to any anchor become seed nodes, each
     carrying a seed weight equal to the number of anchors they match.

  3. WALK (deterministic, up to MAX_HOPS hops)
     For each frontier resource we traverse four typed edge kinds and
     accumulate weight on every visited resource:
       co-exposes (share an entityType)         weight 1.0
       co-partOf  (share a businessGroup)       weight 2.0
       calls      (integrationDependencies)     weight 3.0
       processNext                              weight 3.0
     Each hop discounts the contributed weight by HOP_DECAY.

  4. FINAL RE-RANK              (1 LLM call)
     The walk surfaces the cluster of plausible candidates but the
     graph alone cannot tell which one the activity verb really points
     at. The top RERANK_POOL walk-results are handed to a small LLM
     re-ranker that sees the activity label and per-candidate typed
     metadata (ordId, title, type, capabilities, shortDescription) —
     no full free-text descriptions — and returns the final ranking.

Caller contract: returns {"method": "C", "candidates", "trace"}.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from src.core import config, llm


# ─── Walk parameters ────────────────────────────────────────────────────────

MAX_HOPS = 2
HOP_DECAY = 0.5
RERANK_POOL = 8   # how many walk-top resources the final LLM re-ranker sees

EDGE_WEIGHT = {
    "co_exposes":   1.0,
    "co_partOf":    2.0,
    "calls":        3.0,
    "processNext":  3.0,
}


# ─── Re-rank prompt (final stage) ────────────────────────────────────────────

_RERANK_SYS = """You re-rank candidate ORD resources for an activity. The candidates were surfaced by a knowledge-graph walk and are all topically relevant; your job is to decide which one the activity verb most directly points to.

Prefer Agents when the activity describes a worker doing something, APIs when the activity is pure CRUD on master data, dataProducts for analytical reads. Use the resource's title, type, capabilities and shortDescription as your evidence.

Respond with a JSON array of ORD IDs ordered by relevance, no other text.

If NONE of the candidates is a meaningful match for the activity (e.g. the activity asks for a capability that the graph walk surfaced only because of topical overlap, but none of the candidates actually fulfils the activity --- think translation, weather data, room booking, e-signature, sentiment analysis), return an empty array [] instead of picking a tangentially related one."""

_RERANK_SYS_NO_REFUSE = """You re-rank candidate ORD resources for an activity. The candidates were surfaced by a knowledge-graph walk; your job is to rank them by relevance.

Prefer Agents when the activity describes a worker doing something, APIs when the activity is pure CRUD on master data, dataProducts for analytical reads. Use the resource's title, type, capabilities and shortDescription as your evidence.

Always return a JSON array of ORD IDs ordered by relevance. You MUST pick at least one — never return an empty array. If none is perfect, pick the closest match."""


def _rerank_block(r: dict) -> str:
    lines = [f"- ordId: {r['ordId']}",
             f"  type: {r.get('type','')}",
             f"  title: {r.get('title','')}"]
    if r.get("capabilities"):
        lines.append(f"  capabilities: {', '.join(r['capabilities'])}")
    if r.get("useCases"):
        lines.append(f"  useCases: {' | '.join(r['useCases'])}")
    if r.get("shortDescription"):
        lines.append(f"  shortDescription: {r['shortDescription']}")
    return "\n".join(lines)


def _rerank_prompt(label: str, pool: list[dict], k: int) -> str:
    lines = ["Activity:", f"  {label}", "", f"Candidates ({len(pool)}):"]
    for r in pool:
        lines.append(_rerank_block(r))
        lines.append("")
    return "\n".join(lines) + f"Return a JSON array of up to {k} ORD IDs."


def _parse_id_array(text: str) -> list[str]:
    m = re.search(r"\[[\s\S]*?\]", text)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
        return [str(x) for x in arr if isinstance(x, str)]
    except json.JSONDecodeError:
        return []


# ─── Stage 1: anchor inference (LLM) ────────────────────────────────────────

_ANCHOR_SYS = """You map an activity to anchor concept nodes from ORD vocabularies so a graph walk can start from there.

Pick up to 4 entity types, up to 3 business process groups, and up to 2 lines of business that the activity touches. Empty list when unsure; do not guess.

Respond with one JSON object:
{
  "entity_types":      [...],
  "business_groups":   [...],
  "lines_of_business": [...],
  "reason":            "<one short sentence>"
}"""


def _anchor_prompt(label: str, ets: list[str], groups: list[str],
                   lobs: list[str]) -> str:
    lines = ["Activity:", f"  {label}", "", "Available entity types:"]
    lines += [f"  - {e}" for e in ets]
    lines += ["", "Available business process groups:"]
    lines += [f"  - {g}" for g in groups]
    lines += ["", "Available lines of business:"]
    lines += [f"  - {l}" for l in lobs]
    return "\n".join(lines) + "\n\nReturn one JSON object."


def _parse_anchors(text: str, vets: set[str], vgrps: set[str],
                   vlobs: set[str]) -> dict:
    m = re.search(r"\{[\s\S]*\}", text)
    obj = {}
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = {}

    def _filt(k: str, valid: set[str]) -> list[str]:
        return [str(x) for x in (obj.get(k) or [])
                if isinstance(x, str) and x in valid]

    return {
        "entity_types":      _filt("entity_types", vets),
        "business_groups":   _filt("business_groups", vgrps),
        "lines_of_business": _filt("lines_of_business", vlobs),
        "reason":            str(obj.get("reason", ""))[:300],
    }


# ─── Graph index (built once per landscape) ─────────────────────────────────


def _index(resources: list[dict]) -> dict:
    """Build adjacency indices once. Pure dicts of sets — minimal."""
    by_id = {r["ordId"]: r for r in resources}
    et2res: dict[str, set[str]] = defaultdict(set)
    grp2res: dict[str, set[str]] = defaultdict(set)
    lob2res: dict[str, set[str]] = defaultdict(set)
    next_out: dict[str, set[str]] = defaultdict(set)

    for r in resources:
        rid = r["ordId"]
        for et in r.get("entityTypes") or []:
            et2res[et].add(rid)
        for g in r.get("partOfGroups") or []:
            gid = g.get("groupId") if isinstance(g, dict) else None
            if gid:
                grp2res[gid].add(rid)
        for lob in r.get("lineOfBusiness") or []:
            lob2res[lob].add(rid)
        for nx in r.get("processNext") or []:
            if isinstance(nx, str):
                next_out[rid].add(nx)

    # calls edges come from integrationDependencies in the per-namespace
    # ord_enriched.json — parse once.
    calls_out: dict[str, set[str]] = defaultdict(set)
    for ns_dir in sorted(Path(config.LANDSCAPE_DIR).iterdir()):
        if ns_dir.name == "sap.odm":
            continue
        # integrationDependencies only in enriched ORD
        enriched_path = config.LANDSCAPE_ENRICHED_DIR / ns_dir.name / "ord_enriched.json"
        path = enriched_path if enriched_path.exists() else ns_dir / "ord.json"
        if not path.exists():
            continue
        try:
            doc = json.loads(path.read_text())
        except Exception:
            continue
        for dep in doc.get("integrationDependencies", []) or []:
            src = dep.get("ordId")
            if not src:
                continue
            for asp in dep.get("aspects", []) or []:
                for api in asp.get("apiResources", []) or []:
                    tgt = api.get("ordId") if isinstance(api, dict) else None
                    if tgt:
                        calls_out[src].add(tgt)

    return {
        "by_id": by_id,
        "et2res": et2res,
        "grp2res": grp2res,
        "lob2res": lob2res,
        "next_out": next_out,
        "calls_out": calls_out,
    }


def _all_anchors(idx: dict) -> tuple[list[str], list[str], list[str]]:
    return (sorted(idx["et2res"].keys()),
            sorted(idx["grp2res"].keys()),
            sorted(idx["lob2res"].keys()))


# ─── The walk itself ────────────────────────────────────────────────────────


def _walk(idx: dict, ets: set[str], grps: set[str], lobs: set[str],
          previous_resolved: set[str]) -> dict[str, float]:
    """Return {ord_id: accumulated_score}.

    Hop 0: seed from anchor matches (and from previous plan step).
    Hop 1..MAX_HOPS: expand frontier through typed edges, decaying weight.
    """
    score: dict[str, float] = defaultdict(float)

    # Hop 0 — seeds from anchors + plan context.
    seeds: set[str] = set()
    for et in ets:
        for rid in idx["et2res"].get(et, ()):
            score[rid] += 1.0       # ET hit
            seeds.add(rid)
    for g in grps:
        for rid in idx["grp2res"].get(g, ()):
            score[rid] += 2.0       # group hit
            seeds.add(rid)
    for lob in lobs:
        for rid in idx["lob2res"].get(lob, ()):
            score[rid] += 0.5       # LoB is a soft signal
            seeds.add(rid)
    # plan context: the previous step's top-1 is also a frontier seed
    for rid in previous_resolved:
        if rid in idx["by_id"]:
            seeds.add(rid)

    frontier = set(seeds)
    visited = set(seeds)
    decay = 1.0
    for _ in range(MAX_HOPS):
        decay *= HOP_DECAY
        next_frontier: set[str] = set()

        for rid in frontier:
            r = idx["by_id"].get(rid)
            if not r:
                continue

            # co-exposes: other resources that share an ET with r
            for et in r.get("entityTypes") or []:
                for nb in idx["et2res"].get(et, ()):
                    if nb == rid:
                        continue
                    score[nb] += EDGE_WEIGHT["co_exposes"] * decay
                    next_frontier.add(nb)

            # co-partOf: resources in the same business group
            for g in r.get("partOfGroups") or []:
                gid = g.get("groupId") if isinstance(g, dict) else None
                if not gid:
                    continue
                for nb in idx["grp2res"].get(gid, ()):
                    if nb == rid:
                        continue
                    score[nb] += EDGE_WEIGHT["co_partOf"] * decay
                    next_frontier.add(nb)

            # calls: direct invocation edge (Agent → API/Agent)
            for nb in idx["calls_out"].get(rid, ()):
                score[nb] += EDGE_WEIGHT["calls"] * decay
                next_frontier.add(nb)

            # processNext: sequential successor
            for nb in idx["next_out"].get(rid, ()):
                score[nb] += EDGE_WEIGHT["processNext"] * decay
                next_frontier.add(nb)

        frontier = next_frontier - visited
        visited |= next_frontier
        if not frontier:
            break

    return dict(score)


# ─── Public API ─────────────────────────────────────────────────────────────


def retrieve(label: str,
             resources: list[dict],
             top_k: int = config.TOP_K,
             restrict_to_entity_types: list[str] | None = None,
             previous_resolved_ord_ids: list[str] | None = None,
             allow_refuse: bool = True) -> dict:
    """Anchor → seed → multi-hop walk → top-k by score.

    allow_refuse=False forces the reranker to always pick at least one candidate.
    Use False for design-time evaluation where every activity has a known GT resource.
    """
    idx = _index(resources)
    all_ets, all_grps, all_lobs = _all_anchors(idx)
    if restrict_to_entity_types is not None:
        all_ets = [e for e in all_ets if e in set(restrict_to_entity_types)]

    # ─── Stage 1: anchor selection ─────────────────────────────────────────
    text, meta = llm.chat(
        _anchor_prompt(label, all_ets, all_grps, all_lobs),
        system=_ANCHOR_SYS,
    )
    anchors = _parse_anchors(text, set(all_ets), set(all_grps), set(all_lobs))

    # ─── Walk ──────────────────────────────────────────────────────────────
    scores = _walk(
        idx,
        ets=set(anchors["entity_types"]),
        grps=set(anchors["business_groups"]),
        lobs=set(anchors["lines_of_business"]),
        previous_resolved=set(previous_resolved_ord_ids or []),
    )

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    walk_top = ranked[:RERANK_POOL]

    # ─── Final re-rank: small LLM call activity-aware on the walk pool ────
    rerank_tokens = 0
    rerank_latency = 0.0
    rerank_raw = ""
    refused = False
    if walk_top:
        pool_resources = [idx["by_id"][rid] for rid, _ in walk_top
                          if rid in idx["by_id"]]
        valid_ids = {r["ordId"] for r in pool_resources}
        rr_text, rr_meta = llm.chat(
            _rerank_prompt(label, pool_resources, top_k),
            system=_RERANK_SYS if allow_refuse else _RERANK_SYS_NO_REFUSE,
        )
        rerank_raw = rr_text.strip()
        rerank_tokens = rr_meta["tokens"]
        rerank_latency = round(rr_meta["latency"], 3)
        parsed_ids = _parse_id_array(rr_text)
        ranked_ids = [x for x in parsed_ids if x in valid_ids]
        # Distinguish a deliberate refusal (LLM returned an explicit empty
        # array) from a parse failure. On parse failure we keep the walk
        # order as a defensive fallback; on refusal we honour it.
        if not ranked_ids:
            if re.search(r"\[\s*\]", rr_text):
                refused = True
            else:
                # parse failure → fall back to walk ordering
                ranked_ids = [rid for rid, _ in walk_top[:top_k]]
    else:
        ranked_ids = []

    candidates = [
        {"ordId": rid, "score": round(scores.get(rid, 0.0), 3)}
        for rid in ranked_ids[:top_k]
    ]

    result = {
        "method": "C",
        "candidates": candidates,
        "trace": {
            "anchors": anchors,
            "stage1_raw": text.strip(),
            "max_hops": MAX_HOPS,
            "hop_decay": HOP_DECAY,
            "edge_weights": EDGE_WEIGHT,
            "scored_resources": len(scores),
            "walk_top_ids": [rid for rid, _ in walk_top],
            "rerank_raw": rerank_raw,
            "refused": refused,
            "previous_resolved": list(previous_resolved_ord_ids or []),
            "tokens": meta["tokens"] + rerank_tokens,
            "latency_s": round(meta["latency"] + rerank_latency, 3),
            "llm_calls": 1 + (1 if walk_top else 0),
        },
    }
    return result
