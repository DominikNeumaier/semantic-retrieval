"""Method S — Baseline-Solver (no-retrieval baseline).

Input: activity label + flat list of 273 resources (title + shortDescription only).
The LLM picks directly — no namespace filtering, no graph walk, no entity type
reasoning, no embeddings. Weakest possible baseline.

When top_k > 1, returns a ranked list of up to top_k candidates.

Role differs by evaluation level:
  Design-Time: pure measurement baseline — has no influence on case selection.
  Run-Time:    embedded in the adversarial loop as a difficulty gate — a case is
               only accepted if Solver fails on it (top_k=1 for gate check).
"""
from __future__ import annotations

import json
import re

from src.core import config, llm

_SYSTEM_SINGLE = (
    "You are a resource selector. Given a business activity and a list of ORD resources, "
    "return the single ordId that best fulfils the activity. "
    'Respond with a JSON object: {"ordId": "<best match>"}'
)

_SYSTEM_RANKED = (
    "You are a resource ranker. Given a business activity and a list of ORD resources, "
    "return the top-{k} ordIds in order of relevance (best first). "
    'Respond with a JSON array of strings: ["<best>", "<second>", ...]'
)


def retrieve(label: str, resources: list[dict], top_k: int = config.TOP_K) -> dict:
    """Return {"candidates": [{ordId, score}], "trace": {...}}"""
    lines = [f"Activity: {label}", "", "Resources:"]
    for r in resources:
        lines.append(f"  - {r['ordId']}: {r['title']}. {r.get('shortDescription', '')}")

    if top_k == 1:
        # Adversarial gate mode — single pick
        text, meta = llm.chat("\n".join(lines), system=_SYSTEM_SINGLE)
        ordId = None
        try:
            m = re.search(r'\{[^}]*"ordId"\s*:\s*"([^"]+)"', text)
            if m:
                ordId = m.group(1).strip()
        except Exception:
            pass
        candidates = [{"ordId": ordId, "score": 1.0}] if ordId else []
    else:
        # Ranked retrieval mode — top_k candidates
        system = _SYSTEM_RANKED.format(k=top_k)
        text, meta = llm.chat("\n".join(lines), system=system)
        ord_ids: list[str] = []
        try:
            m = re.search(r'\[.*?\]', text, re.DOTALL)
            if m:
                arr = json.loads(m.group())
                ord_ids = [str(x).strip() for x in arr if isinstance(x, str)][:top_k]
        except Exception:
            pass
        # Fallback: try to find single ordId
        if not ord_ids:
            m2 = re.search(r'"([a-z]+\.[a-z]+:[a-z]+:[A-Za-z0-9_]+:v\d+)"', text)
            if m2:
                ord_ids = [m2.group(1)]
        candidates = [{"ordId": oid, "score": round(1.0 - i * 0.1, 2)}
                      for i, oid in enumerate(ord_ids)]

    return {
        "method": "S",
        "candidates": candidates,
        "trace": {
            "tokens": meta["tokens"],
            "latency_s": round(meta["latency"], 3),
            "llm_calls": 1,
        },
    }
