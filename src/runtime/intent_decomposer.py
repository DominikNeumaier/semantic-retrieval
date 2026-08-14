"""Intent decomposer — runs on every dynamic case.

Classifies whether the prompt contains a single intent or multiple independent
intents, then splits accordingly. This makes the decomposer a true orchestration
component that always executes for dynamic mode, rather than a post-hoc patch.

Single-intent:  returns [original_prompt], n_detected=1, no split overhead
Multi-intent:   returns [sub-query-1, sub-query-2, ...], n_detected=N
"""
from __future__ import annotations

import json
import re

from src import llm

SYSTEM_PROMPT = (
    "You are an enterprise query decomposer for an agent orchestration system. "
    "Your job: detect whether a user request contains one or multiple independent "
    "intents, and split it accordingly. Answer ONLY with valid JSON."
)


def decompose(user_prompt: str) -> tuple[list[str], dict]:
    """Detect intent count and split if needed.

    Returns (sub_queries, trace).
    sub_queries has 1 entry for single-intent (the original prompt, unchanged),
    or N entries for multi-intent (one self-contained sub-query per intent).
    """
    user_p = f"""Analyse this enterprise user request and decompose it into independent sub-queries.

User request: {user_prompt}

Rules:
- If the request asks for ONE capability or resource → return the original request as-is (n=1)
- If the request asks for MULTIPLE independent capabilities → split into one sub-query per intent
- Each sub-query must be self-contained and describe exactly ONE capability
- Do NOT split sequential steps of the same process (those are single-intent)
- Only split when the user clearly needs two or more unrelated resources simultaneously

Return JSON:
{{
  "n_intents": 1,
  "sub_queries": ["<original or sub-query 1>", "<sub-query 2 if n>1>", ...]
}}"""

    text, meta = llm.chat(user_p, system=SYSTEM_PROMPT)
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        data = json.loads(m.group()) if m else {}
        n = int(data.get("n_intents", 1))
        sub_queries = data.get("sub_queries", [])
        if not sub_queries or len(sub_queries) != n:
            sub_queries = [user_prompt]
            n = 1
    except Exception:
        sub_queries = [user_prompt]
        n = 1

    trace = {
        "tokens": meta["tokens"],
        "latency_s": meta.get("latency_s", 0.0),
        "llm_calls": 1,
        "n_detected": n,
        "sub_queries": sub_queries,
        "original_prompt": user_prompt,
    }
    return sub_queries, trace
