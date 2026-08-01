"""Intent Resolver (Stage 1 of the Orchestration Layer).

The first component in the planning-to-execution pipeline of Sec. 4.3.
Given a natural-language request, it inspects the skill registry and
decides whether any registered skill matches the request meaningfully.

It returns a single `skill_id` (the closest match) or `None` (no skill
matches). It does *not* judge coverage or build a plan — that is the
Planner's job (Stage 2).

The LLM sees only skill ids and short descriptions. Step lists are not
sent here, because the question at this stage is purely "is there a
skill for this intent?". The expensive step-level check happens later
and only for the one selected skill.
"""

from __future__ import annotations

import json
import re

from src.core import llm


_SYS = """You decide whether any registered skill covers a user request end-to-end.

Each skill is a named multi-step process with a short description. Your task: find the single skill whose process completely covers the user's request from start to finish — or report null if no skill applies.

Return null when:
- The request is a one-off, ad-hoc lookup (e.g. "find the status of order X", "who handles vendor invoices?")
- The request only overlaps with part of a skill but doesn't need the full process
- The request mentions a topic that a skill covers but the user's intent is clearly narrower
- You are not confident — when in doubt, return null

Only return a skill_id if the skill is a strong, complete match for the user's stated need.

Respond with one JSON object:
{"best_skill_id": "<id or null>", "reason": "<one short sentence>"}"""


def _prompt(user_request: str, skills: list[dict]) -> str:
    lines = ["User request:", f"  {user_request}", "", "Registered skills:"]
    for s in skills:
        lines.append(f"  - id: {s['skill_id']}")
        lines.append(f"    description: {s['description'][:200]}")
    lines.append("")
    lines.append("Return one JSON object with best_skill_id and reason.")
    return "\n".join(lines)


def _parse(text: str, valid_ids: set[str]) -> dict:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {"best_skill_id": None, "reason": "parser_failed"}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"best_skill_id": None, "reason": "parser_failed"}
    sid = obj.get("best_skill_id")
    if isinstance(sid, str) and sid.lower() == "null":
        sid = None
    if isinstance(sid, str) and sid not in valid_ids:
        sid = None
    if not isinstance(sid, str):
        sid = None
    return {
        "best_skill_id": sid,
        "reason": str(obj.get("reason", ""))[:300],
    }


def resolve(user_request: str, skills: list[dict]) -> dict:
    """Stage 1 — return {"skill_id": str|None, "trace": {...}}."""
    valid = {s["skill_id"] for s in skills}
    text, meta = llm.chat(_prompt(user_request, skills), system=_SYS)
    parsed = _parse(text, valid)
    return {
        "skill_id": parsed["best_skill_id"],
        "trace": {
            "stage": "intent_resolver",
            "raw": text.strip(),
            "reason": parsed["reason"],
            "tokens": meta["tokens"],
            "latency_s": round(meta["latency"], 3),
            "llm_calls": 1,
        },
    }


if __name__ == "__main__":
    from src.runtime import skill_registry
    skills = skill_registry.load_skills()
    print(f"Registry: {len(skills)} skills")
    demos = [
        "A machine on production line 4 has just failed. Handle the full breakdown resolution process.",
        "Recall batch SH-2024-118 and additionally publish a regulatory bulletin to affected partner platforms.",
        "I need to order 200 units of hydraulic filter from vendor 4400123.",
    ]
    for q in demos:
        r = resolve(q, skills)
        print(f"\n  Q: {q[:80]}")
        print(f"     -> skill_id={r['skill_id']}  reason={r['trace']['reason'][:100]}")
