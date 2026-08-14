"""Planner / Gap-Resolver (Stage 2 of the Orchestration Layer).

Runs after the Intent Resolver. Given the user request and the single skill
that the Intent Resolver picked (or `None`), it produces an executable plan
in one of three orchestration modes:

  - skill_guided   : the skill's steps cover the request end-to-end.
  - skill_adjusted : the skill covers the core, but extra activities are
                     needed. The Planner names those gap steps.
  - dynamic        : no skill applied at all — the request itself becomes
                     a single ad-hoc step.

If the Intent Resolver returned `None`, no LLM call is made here at all:
the mode is set to `dynamic` and the user request becomes the only step.
Otherwise one LLM call inspects the skill's steps and decides coverage.
"""

from __future__ import annotations

import json
import re

from src import llm


_SYS = """You evaluate whether a registered skill covers a user request end-to-end.

You are given:
- the user request,
- one specific skill with its full ordered step list.

Inspect the steps and decide:
- coverage = "full"    if the skill's steps cover the core intent of the request.
                       Vague or general phrasing in the request does NOT imply extra steps.
                       Only mark "partial" if the request EXPLICITLY names concrete activities
                       that are clearly absent from the skill's step list.
- coverage = "partial" if the request explicitly requires one or more concrete activities
                       that are NOT represented anywhere in the step list.
                       In this case list only the concretely missing activities as gap_steps.

Default to "full" when in doubt. A skill covers a request if its steps address the main
business goal — minor wording differences or implied sub-tasks do not make it "partial".

Respond with one JSON object:
{
  "coverage":  "full" | "partial",
  "gap_steps": ["<short imperative step label>", ...],
  "reason":    "<one sentence: what the skill covers and why it is full or what is concretely missing>"
}

Use an empty gap_steps list when coverage is "full"."""


def _prompt(user_request: str, skill: dict) -> str:
    lines = [
        "User request:",
        f"  {user_request}",
        "",
        f"Candidate skill: {skill['skill_id']}",
        f"  description: {skill['description'][:300]}",
        "  steps:",
    ]
    for st in skill["steps"]:
        lines.append(f"    {st['index']}. {st['name']}")
    lines.append("")
    lines.append("Decide coverage and list any missing activities as gap_steps.")
    return "\n".join(lines)


def _parse(text: str) -> dict:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {"coverage": "full", "gap_steps": [], "reason": "parser_failed"}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"coverage": "full", "gap_steps": [], "reason": "parser_failed"}
    coverage = str(obj.get("coverage", "full")).lower()
    if coverage not in ("full", "partial"):
        coverage = "full"
    gaps = obj.get("gap_steps", []) or []
    gaps = [str(g).strip() for g in gaps if isinstance(g, str) and str(g).strip()]
    if coverage == "full":
        gaps = []
    return {
        "coverage": coverage,
        "gap_steps": gaps,
        "reason": str(obj.get("reason", ""))[:300],
    }


def plan(user_request: str, skill: dict | None) -> dict:
    """Stage 2 — return {"mode", "skill_id", "gap_steps", "trace"}.

    If `skill` is None, returns dynamic mode without an LLM call.
    Otherwise runs one LLM call that decides full vs partial and surfaces
    any gap steps.
    """
    if skill is None:
        return {
            "mode": "dynamic",
            "skill_id": None,
            "gap_steps": [],
            "trace": {
                "stage": "planner",
                "skipped": True,
                "reason": "no skill matched at stage 1",
                "tokens": 0,
                "latency_s": 0.0,
                "llm_calls": 0,
            },
        }

    text, meta = llm.chat(_prompt(user_request, skill), system=_SYS)
    parsed = _parse(text)
    mode = "skill_guided" if parsed["coverage"] == "full" else "skill_adjusted"
    return {
        "mode": mode,
        "skill_id": skill["skill_id"],
        "gap_steps": parsed["gap_steps"],
        "trace": {
            "stage": "planner",
            "raw": text.strip(),
            "coverage": parsed["coverage"],
            "reason": parsed["reason"],
            "tokens": meta["tokens"],
            "latency_s": round(meta["latency"], 3),
            "llm_calls": 1,
        },
    }


if __name__ == "__main__":
    from src.runtime import skill_registry, intent_resolver
    skills = skill_registry.load_skills()
    print(f"Registry: {len(skills)} skills")
    demos = [
        "A machine on production line 4 has just failed. Handle the full breakdown resolution process.",
        "Recall batch SH-2024-118 and additionally publish a regulatory bulletin to affected partner platforms.",
        "I need to order 200 units of hydraulic filter from vendor 4400123.",
    ]
    by_id = {s["skill_id"]: s for s in skills}
    for q in demos:
        ir = intent_resolver.resolve(q, skills)
        sk = by_id.get(ir["skill_id"]) if ir["skill_id"] else None
        pl = plan(q, sk)
        print(f"\n  Q: {q[:80]}")
        print(f"     stage1 skill_id = {ir['skill_id']}")
        print(f"     stage2 mode     = {pl['mode']}  gap_steps={pl['gap_steps']}")
        print(f"     reason          = {pl['trace'].get('reason','')[:120]}")
