"""Runtime pipeline: turn a user request into a sequence of resolved resources.

Three Orchestration-Layer stages in sequence (Sec. 4.3):

    user_request
        │
        ▼
    1. Intent Resolver  ── 1 LLM call ─► skill_id  OR  None
        │
        ▼
    2. Planner          ── 1 LLM call if skill_id, else 0 ─►
                              (mode, gap_steps) where mode is
                              skill_guided | skill_adjusted | dynamic
        │
        ▼
    Build step list:
        skill_guided    → skill.steps               (no extra LLM)
        skill_adjusted  → skill.steps + gap_steps   (gap_steps from stage 2)
        dynamic         → [user_request]
        │
        ▼
    3. Retrieval        ── method.retrieve per non-pinned step
                              (A | B | C)
        │
        ▼
    Plan with per-step top-k candidates + traces from every stage.
"""

from __future__ import annotations

from src import config
from src.runtime import intent_resolver, planner, intent_decomposer
from src.methods import method_embedding, method_progressive, method_graph, method_tools, method_raw, method_baseline, method_hybrid


METHODS = {
    "A": method_embedding.retrieve,
    "B": method_progressive.retrieve,
    "C": method_graph.retrieve,
    "D": method_tools.retrieve,
    "E": method_raw.retrieve,
    "S": method_baseline.retrieve,
    "F": method_hybrid.retrieve,  # agentic meta-retrieval over A/B/C/D tools
}


def plan_and_resolve(user_request: str,
                     resources: list[dict],
                     skills: list[dict],
                     method_name: str = "A",
                     top_k: int = config.TOP_K,
                     hint_skill_id: str | None = None,
                     force_mode: str | None = None,
                     use_graph: bool = False,
                     n_intents_hint: int | None = None,
                     hint_gap_steps: list[str] | None = None) -> dict:
    """Run the full runtime pipeline.

    hint_skill_id: skip resolver and use this skill directly (SA cases).
    force_mode: skip resolver and planner — "dynamic" forces ad-hoc retrieval
      regardless of what the resolver would say. Used for fair retrieval eval.
    hint_gap_steps: inject pre-computed gap step labels, bypassing the planner
      LLM call. Used to guarantee identical gap steps across methods.
    """
    if method_name not in METHODS:
        raise ValueError(f"unknown method {method_name}")
    retrieve = METHODS[method_name]

    # ─── Stage 1: Intent Resolver ───────────────────────────────────────────
    if force_mode == "dynamic":
        # Bypass resolver and planner entirely — force ad-hoc retrieval.
        # Used to measure retrieval quality independently of routing errors.
        skill_id = None
        intent = {"skill_id": None, "trace": {"stage": "forced_dynamic", "tokens": 0, "latency_s": 0.0, "llm_calls": 0, "reason": "forced dynamic mode"}}
    elif hint_skill_id is not None:
        # Skip resolver — skill is known from case definition (SG/SA cases)
        skill_id = hint_skill_id
        intent = {"skill_id": skill_id, "trace": {"stage": "hint", "tokens": 0, "latency_s": 0.0, "llm_calls": 0, "reason": "hint provided"}}
    else:
        intent = intent_resolver.resolve(user_request, skills)
        skill_id = intent["skill_id"]
    skill = next((s for s in skills if s["skill_id"] == skill_id), None)

    total_tokens = intent["trace"]["tokens"]
    total_latency = intent["trace"]["latency_s"]
    total_llm = intent["trace"]["llm_calls"]

    # ─── Stage 2: Planner (coverage + gap inference) ─────────────────────────
    if force_mode == "dynamic":
        plan_out = {"mode": "dynamic", "gap_steps": [], "trace": {"tokens": 0, "latency_s": 0.0, "llm_calls": 0}}
    elif hint_gap_steps is not None:
        # Injected gap steps — skip planner LLM call for fair cross-method comparison
        plan_out = {"mode": "skill_adjusted", "gap_steps": hint_gap_steps,
                    "trace": {"tokens": 0, "latency_s": 0.0, "llm_calls": 0,
                              "note": "gap steps injected from reference run"}}
    else:
        plan_out = planner.plan(user_request, skill)
    mode = plan_out["mode"]
    gap_steps = plan_out["gap_steps"]

    total_tokens += plan_out["trace"]["tokens"]
    total_latency += plan_out["trace"]["latency_s"]
    total_llm += plan_out["trace"]["llm_calls"]

    coverage = (
        "none"
        if mode == "dynamic"
        else ("full" if mode == "skill_guided" else "partial")
    )

    # ─── Build step list ────────────────────────────────────────────────────
    step_list: list[dict] = []
    decomp_trace_out = None  # only set for dynamic mode

    if mode == "skill_guided" and skill:
        for st in skill["steps"]:
            step_list.append({
                "step_name": st["name"],
                "source": "skill",
                "ord_confirmed": st.get("ord_confirmed", []),
            })
    elif mode == "skill_adjusted" and skill:
        for st in skill["steps"]:
            step_list.append({
                "step_name": st["name"],
                "source": "skill",
                "ord_confirmed": st.get("ord_confirmed", []),
            })
        for g in gap_steps:
            step_list.append({
                "step_name": g,
                "source": "gap",
                "ord_confirmed": [],
            })
    else:  # dynamic
        if n_intents_hint == 1:
            # Caller certifies single-intent — skip decomposer entirely.
            # Used for forced single-intent evaluation (*f records) so the
            # decomposer can't accidentally split a single-intent prompt.
            decomp_trace_out = {"tokens": 0, "latency_s": 0.0, "llm_calls": 0,
                                 "n_detected": 1, "sub_queries": [user_request]}
            steps_to_add = [user_request]
        else:
            # Decomposer classifies single vs multi-intent.
            _, decomp_trace_out = intent_decomposer.decompose(user_request)
            total_tokens += decomp_trace_out["tokens"]
            total_latency += decomp_trace_out["latency_s"]
            total_llm += decomp_trace_out["llm_calls"]
            n_detected = decomp_trace_out["n_detected"]
            # Only use sub-queries if decomposer detected >1 intent
            use_sub_queries = n_detected >= 2
            steps_to_add = decomp_trace_out["sub_queries"] if use_sub_queries else [user_request]
        for sq in steps_to_add:
            step_list.append({
                "step_name": sq,
                "source": "adhoc",
                "ord_confirmed": [],
            })

    # ─── Stage 3: Retrieval ─────────────────────────────────────────────────
    # Skill steps with ord_confirmed bypass retrieval (Sec. 4.3 / Fig. 8:
    # Skill-Guided runs the resources already matched at design time).
    # Gap and ad-hoc steps go through the chosen retrieval method, and
    # the retriever receives the top-1 ord_id of the previous step as
    # plan context — this is the channel through which processNext
    # influences the next step's ranking (Method C consumes it; A and B
    # accept the kwarg but ignore it, keeping their isolated-step
    # semantics intact).
    resolved: list[dict] = []
    prev_top_ord_ids: list[str] = []
    for st in step_list:
        if st["ord_confirmed"]:
            resolved.append({
                "step_name": st["step_name"],
                "source": st["source"],
                "candidates": [{"ordId": oid, "score": None}
                               for oid in st["ord_confirmed"]],
                "method_trace": {
                    "via": "skill_pinned",
                    "tokens": 0, "latency_s": 0.0, "llm_calls": 0,
                },
            })
            # pinned skill step still produces context for the next step
            prev_top_ord_ids = list(st["ord_confirmed"])
            continue
        # Pass optional plan-context kwargs; methods that don't accept
        # them raise TypeError and we fall back to the basic signature.
        # D needs the skill registry too — it can inspect skills as one
        # of its tools.
        optional_kwargs: dict = {
            "previous_resolved_ord_ids": prev_top_ord_ids or None,
            "skills": skills,
            "use_graph": use_graph,
            "n_intents_hint": n_intents_hint,
        }
        try:
            result = retrieve(
                st["step_name"], resources, top_k=top_k, **optional_kwargs,
            )
        except TypeError:
            # Try with just previous_resolved_ord_ids (C accepts it)
            try:
                result = retrieve(
                    st["step_name"], resources, top_k=top_k,
                    previous_resolved_ord_ids=prev_top_ord_ids or None,
                )
            except TypeError:
                # Plain signature (A, B)
                result = retrieve(st["step_name"], resources, top_k=top_k)
        resolved.append({
            "step_name": st["step_name"],
            "source": st["source"],
            "candidates": result["candidates"],
            "method_trace": result["trace"],
        })
        total_tokens += result["trace"].get("tokens", 0)
        total_latency += result["trace"].get("latency_s", 0.0)
        total_llm += result["trace"].get("llm_calls", 0)
        # remember top-1 for the next step's context
        if result["candidates"]:
            prev_top_ord_ids = [result["candidates"][0]["ordId"]]

    return {
        "request": user_request,
        "mode": mode,
        "skill_id": skill_id if mode != "dynamic" else None,
        "coverage": coverage,
        "method": method_name,
        "steps": resolved,
        "trace": {
            "intent_trace": intent["trace"],
            "planner_trace": plan_out["trace"],
            "decomp_trace": decomp_trace_out,
            "total_tokens": total_tokens,
            "total_latency_s": round(total_latency, 3),
            "total_llm_calls": total_llm,
        },
    }


if __name__ == "__main__":
    from src import loader as ord_loader
    from src.runtime import skill_registry
    resources = ord_loader.load_landscape()
    skills = skill_registry.load_skills()

    demos = [
        ("A machine on production line 4 has just failed. Handle the full breakdown resolution process.", "A"),
        ("Recall batch SH-2024-118 and additionally publish a regulatory bulletin to affected partner platforms.", "A"),
        ("I need to order 200 units of hydraulic filter from vendor 4400123.", "A"),
    ]
    for q, m in demos:
        plan = plan_and_resolve(q, resources, skills, method_name=m)
        print(f"\n[mode={plan['mode']:<14} method={plan['method']}] skill={plan['skill_id']}")
        print(f"  Q: {q[:80]}")
        for i, s in enumerate(plan["steps"][:3], 1):
            top = s["candidates"][0]["ordId"] if s["candidates"] else "(none)"
            print(f"    {i}. [{s['source']}] {s['step_name'][:50]:<50} -> {top}")
        print(f"  intent={plan['trace']['intent_trace']['llm_calls']} planner={plan['trace']['planner_trace']['llm_calls']} "
              f"total_llm_calls={plan['trace']['total_llm_calls']} tokens={plan['trace']['total_tokens']} latency_s={plan['trace']['total_latency_s']}")
