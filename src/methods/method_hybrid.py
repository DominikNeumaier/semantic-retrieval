"""Method F — Agentic Meta-Retrieval with method tools.

A reasoning LLM is given the full ORD landscape AND the four retrieval methods
(A=embedding, B=progressive disclosure stage4, C=graph walk, D=agentic) as tools.
The agent decides per-case which tools to call, in what order, and commits to a
final ordId via pick_resource().

Tools (5 total):
  - embedding_search(query, top_k)         → A's cosine similarity
  - progressive_filter(query, top_k)        → B's namespace/modality/ET/final pipeline
  - graph_walk(anchor_ord_ids, top_k)       → C's graph walk from given anchors
  - resources_with_capability(capability)   → enriched-only structural lookup
  - resources_exposing(entity_type)         → ET-based lookup
  - get_resource(ord_id)                    → fetch full descriptor
  - pick_resource(ord_id, reason)           → commit final answer

F is a post-hoc composite constructed after A--E results were observed. It
tests whether a free agent that dispatches between embedding recall,
progressive disclosure, and typed tool calls exceeds any single strategy
without proportionally increasing token cost.
"""
from __future__ import annotations

import json
import time
from typing import Any

from src import config, llm
from src.methods import method_embedding, method_progressive, method_graph


MAX_TOOL_CALLS = 10


_SYSTEM = """You are an ORD retrieval agent with access to four retrieval methods as tools. Given a business activity, you decide which tools to call to identify the single best ORD resource. End with pick_resource(ord_id, reason).

ORD GROUNDING
Each resource has:
  - ordId: <namespace>:<type>:<localId>:<version>
  - type: apiResource | agent | dataProduct
  - entityTypes: ODM business objects (Customer, Machine, …)
  - lineOfBusiness: functional domain
  - capabilities: verb-noun tokens (only on enriched landscape, e.g. "manage-material-provisioning")

YOUR TOOLS
  embedding_search(query, top_k)  → semantic similarity over title+description (best for vocabulary-gap, returns top-k)
  progressive_filter(query, top_k) → namespace + modality + ET filter pipeline (good when domain is clear)
  graph_walk(anchor_ord_ids)       → enriched-only — walks processNext/co_exposes from given anchors
  resources_with_capability(cap)   → enriched-only — direct capability lookup (verb-noun)
  resources_exposing(entity_type)  → resources exposing this entity type
  get_resource(ord_id)             → fetch full descriptor of a resource
  pick_resource(ord_id, reason)    → COMMIT — must be your last call
  refuse(reason)                   → no resource fits

STRATEGY ADVICE
  - First try embedding_search — fastest, broad recall
  - If results look semantically close but unsure, also try progressive_filter or resources_with_capability for a different perspective
  - On enriched landscapes, always check capabilities and graph neighbours
  - get_resource a few promising candidates before committing
  - You have at most 10 tool calls. Be efficient.

End with pick_resource. No explanatory text — only tool calls."""


def _build_tools() -> list[dict]:
    return [
        {"type": "function", "function": {
            "name": "embedding_search",
            "description": "Return top_k resources by embedding cosine similarity to the query.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 10}
            }, "required": ["query"]}
        }},
        {"type": "function", "function": {
            "name": "progressive_filter",
            "description": "Run B's progressive disclosure pipeline (namespace → modality → entity-type filter → final pick) and return top_k resources.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 5}
            }, "required": ["query"]}
        }},
        {"type": "function", "function": {
            "name": "graph_walk",
            "description": "Walk the ORD graph (processNext, co_exposes, calls, co_partOf) from given anchor resources. Returns top_k graph-scored resources. Only meaningful on enriched landscape.",
            "parameters": {"type": "object", "properties": {
                "anchor_ord_ids": {"type": "array", "items": {"type": "string"}},
                "top_k": {"type": "integer", "default": 8}
            }, "required": ["anchor_ord_ids"]}
        }},
        {"type": "function", "function": {
            "name": "resources_with_capability",
            "description": "List ordIds that have the given capability label (verb-noun). Enriched-only.",
            "parameters": {"type": "object", "properties": {
                "capability": {"type": "string"}
            }, "required": ["capability"]}
        }},
        {"type": "function", "function": {
            "name": "resources_exposing",
            "description": "List ordIds exposing the given ODM entity type.",
            "parameters": {"type": "object", "properties": {
                "entity_type": {"type": "string"}
            }, "required": ["entity_type"]}
        }},
        {"type": "function", "function": {
            "name": "get_resource",
            "description": "Fetch the full descriptor (title, shortDescription, ets, capabilities, lob) of one resource.",
            "parameters": {"type": "object", "properties": {
                "ord_id": {"type": "string"}
            }, "required": ["ord_id"]}
        }},
        {"type": "function", "function": {
            "name": "pick_resource",
            "description": "Commit final answer. The given ordId is the chosen resource.",
            "parameters": {"type": "object", "properties": {
                "ord_id": {"type": "string"},
                "reason": {"type": "string"}
            }, "required": ["ord_id", "reason"]}
        }},
        {"type": "function", "function": {
            "name": "refuse",
            "description": "No resource in the landscape fulfils the activity. Use only as last resort.",
            "parameters": {"type": "object", "properties": {
                "reason": {"type": "string"}
            }, "required": ["reason"]}
        }},
    ]


def _resource_summary(r: dict) -> dict:
    return {
        "ordId": r["ordId"],
        "type": r.get("type"),
        "title": r.get("title", ""),
        "shortDescription": r.get("shortDescription", "")[:120],
        "entityTypes": (r.get("entityTypes") or [])[:4],
        "lineOfBusiness": (r.get("lineOfBusiness") or [])[:2],
        "capabilities": (r.get("capabilities") or [])[:5],
    }


def _execute_tool(name: str, args: dict, resources: list[dict],
                  res_by_id: dict, query_label: str) -> Any:
    """Execute a tool call against the resource pool."""
    if name == "embedding_search":
        q = args.get("query") or query_label
        k = args.get("top_k", 10)
        result = method_embedding.retrieve(q, resources, top_k=k)
        return [_resource_summary(res_by_id[c["ordId"]])
                for c in result["candidates"] if c["ordId"] in res_by_id]

    if name == "progressive_filter":
        q = args.get("query") or query_label
        k = args.get("top_k", 5)
        result = method_progressive.retrieve(q, resources, top_k=k, allow_refuse=False)
        return [_resource_summary(res_by_id[c["ordId"]])
                for c in result["candidates"] if c["ordId"] in res_by_id]

    if name == "graph_walk":
        anchors = args.get("anchor_ord_ids") or []
        k = args.get("top_k", 8)
        result = method_graph.retrieve(query_label, resources, top_k=k,
                                    previous_resolved_ord_ids=anchors,
                                    allow_refuse=False)
        return [_resource_summary(res_by_id[c["ordId"]])
                for c in result["candidates"] if c["ordId"] in res_by_id]

    if name == "resources_with_capability":
        cap = args.get("capability", "")
        hits = [r for r in resources if cap in (r.get("capabilities") or [])]
        return [_resource_summary(r) for r in hits[:15]]

    if name == "resources_exposing":
        et = args.get("entity_type", "")
        hits = [r for r in resources if et in (r.get("entityTypes") or [])]
        return [_resource_summary(r) for r in hits[:20]]

    if name == "get_resource":
        oid = args.get("ord_id", "")
        r = res_by_id.get(oid)
        if not r:
            return {"error": f"unknown ordId: {oid}"}
        # Full descriptor
        return {
            "ordId": r["ordId"], "type": r.get("type"),
            "title": r.get("title", ""),
            "shortDescription": r.get("shortDescription", ""),
            "description": r.get("description", "")[:300],
            "entityTypes": r.get("entityTypes") or [],
            "lineOfBusiness": r.get("lineOfBusiness") or [],
            "capabilities": r.get("capabilities") or [],
            "tags": r.get("tags") or [],
        }

    return {"error": f"unknown tool: {name}"}


def retrieve(label: str,
             resources: list[dict],
             top_k: int = config.TOP_K,
             skills: list[dict] | None = None,
             previous_resolved_ord_ids: list[str] | None = None,
             allow_refuse: bool = True,
             use_graph: bool = False,
             n_intents_hint: int | None = None) -> dict:
    """F++ — agentic meta-retrieval with method tools."""
    res_by_id = {r["ordId"]: r for r in resources}
    tools = _build_tools()

    state_note = ("Landscape: ENRICHED-ORD — capabilities and processNext/co_exposes available."
                  if use_graph
                  else "Landscape: CLEAN-ORD — capabilities and graph edges NOT available; rely on embedding/ET/namespace.")

    user_msg = f"Activity:\n  {label}\n\n{state_note}\n\nUse tools to find the best resource. End with pick_resource."

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    trace_steps: list[dict] = []
    total_tokens = 0
    total_latency = 0.0
    picked_id: str | None = None
    picked_reason = ""
    refused = False
    refuse_reason = ""

    for step in range(MAX_TOOL_CALLS + 1):
        t0 = time.time()
        def _do_call():
            client, _ = llm.get_clients()
            return client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto" if step < MAX_TOOL_CALLS else "none",
                temperature=config.LLM_TEMPERATURE,
                seed=config.LLM_SEED,
            )
        try:
            resp = llm._call_with_retry(_do_call, what=f"F++step{step}")
        except Exception as e:
            trace_steps.append({"step": step, "kind": "api_error", "error": str(e)[:200]})
            break
        total_latency += time.time() - t0
        total_tokens += (resp.usage.total_tokens if resp.usage else 0)
        msg = resp.choices[0].message

        if not msg.tool_calls:
            trace_steps.append({"step": step, "kind": "final_text",
                                "content": (msg.content or "")[:300]})
            break

        messages.append({
            "role": "assistant", "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            entry = {"step": step, "kind": "tool_call", "name": name, "args": args}

            if name == "pick_resource":
                picked_id = args.get("ord_id")
                picked_reason = args.get("reason", "")
                entry["result"] = {"committed": picked_id}
                trace_steps.append(entry)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps({"committed": picked_id})})
                break

            if name == "refuse":
                if allow_refuse:
                    refused = True
                    refuse_reason = args.get("reason", "")
                    entry["result"] = {"refused": True}
                    trace_steps.append(entry)
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": json.dumps({"refused": True})})
                    break
                else:
                    entry["result"] = {"refused_blocked": True}
                    trace_steps.append(entry)
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": json.dumps({"error": "refuse not allowed; pick a resource"})})
                    continue

            try:
                result = _execute_tool(name, args, resources, res_by_id, label)
            except Exception as e:
                result = {"error": str(e)[:200]}

            entry["result_summary"] = (
                f"{len(result)} items" if isinstance(result, list) else "object"
            )
            # Store ordIds returned by lookup tools so the viewer can show a pool
            if isinstance(result, list) and result and "ordId" in result[0]:
                entry["result_ord_ids"] = [r["ordId"] for r in result]
            trace_steps.append(entry)
            result_str = json.dumps(result)[:1500]
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": result_str})

        if picked_id or refused:
            break

    # Build candidates list — picked resource is top-1; fill remaining slots
    # with resources the agent saw (embedding_search / progressive_filter results)
    # so the viewer can show a meaningful top-5 instead of just top-1=top-5.
    seen_ids: list[str] = []
    for s in trace_steps:
        for oid in (s.get("result_ord_ids") or []):
            if oid not in seen_ids:
                seen_ids.append(oid)

    if picked_id and picked_id in res_by_id:
        pool = [picked_id] + [oid for oid in seen_ids if oid != picked_id and oid in res_by_id]
        candidates = [{"ordId": oid, "score": None} for oid in pool[:top_k]]
    elif refused:
        candidates = []
    else:
        # Agent stopped without commit — fall back to A as last resort
        result = method_embedding.retrieve(label, resources, top_k=top_k)
        candidates = result["candidates"]

    return {
        "candidates": candidates,
        "trace": {
            "tokens": total_tokens,
            "latency_s": round(total_latency, 3),
            "llm_calls": len([s for s in trace_steps if s.get("kind") == "tool_call"]),
            "use_graph": use_graph,
            "picked_ord_id": picked_id,
            "picked_reason": picked_reason,
            "refused": refused,
            "refuse_reason": refuse_reason,
            "agent_steps": trace_steps,
        },
    }
