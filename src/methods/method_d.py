"""Method D — Agentic ORD Discovery.

A reasoning LLM is given a typed tool-set over the ORD landscape and the
skill registry and decides itself which tools to call, in what order,
to identify the single ORD resource that best fulfils a business
activity. There is no fixed retrieval pipeline; the agent composes its
own strategy.

This is the fourth retrieval paradigm next to A (embedding),
B (progressive disclosure) and C (graph walk):

  - A / B / C  = structured retrieval algorithms on ORD
  - D          = LLM agent that operates ORD directly via typed tools

Tools (15 total):
  Vocabulary    list_namespaces, list_entity_types,
                list_lines_of_business, list_business_groups,
                list_capabilities
  Filters       resources_exposing, resources_in_lob,
                resources_in_group, resources_with_capability
  Graph         integration_callees, resources_next_after
  Detail        get_resource
  Skill layer   list_skills, describe_skill
  Stop          pick_resource     ← the agent must end with this

The agent has at most MAX_TOOL_CALLS tool calls before forced stop.
Reasoning + every tool call ends up in `trace["agent_steps"]` so the
empirical chapter of the thesis can quote individual decisions.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from src.core import config, llm


# ─── Configuration ──────────────────────────────────────────────────────────

MAX_TOOL_CALLS = 8

# Shared client via llm.get_clients() so token refresh propagates here.
# D bypasses the simple chat() cache because tool-use requires a
# multi-turn conversation that the cache helper doesn't model.


# ─── System prompt — three grounding blocks ─────────────────────────────────

_SYSTEM = """You are an ORD discovery agent. Given a business activity (a step in a plan, or an ad-hoc user request), you select the single ORD resource that best fulfils that activity. End every session with EITHER the pick_resource tool (a suitable resource exists in the landscape) OR the refuse tool (no resource in this landscape fulfils the activity — do not pick a tangentially related one to be polite).

ORD GROUNDING
ORD (Open Resource Discovery) is a typed metadata protocol. Each resource has:
  - ordId in the format <namespace>:<type>:<localId>:<version>
  - type ∈ {apiResource, agent, dataProduct}
  - entityTypes — ODM business objects it operates on (e.g. Customer, Machine)
  - lineOfBusiness — functional domain (e.g. Human Resources, Manufacturing)
  - capabilities — verb-noun tokens (e.g. skill-matching, fault-diagnosis)
  - partOfGroups — business processes the resource participates in (DT-enriched)
  - processNext — typical successor resources in a plan (DT-enriched)
  - integrationDependencies — which API/agent resources this resource calls

ARCHITECTURE CONTEXT
You operate as the retrieval stage of a planner. The mode (skill_guided / skill_adjusted / dynamic) and any matching skill have already been decided by an upstream router. If a skill_id is provided to you, you may inspect it with describe_skill — its pinned ord_confirmed resources are usually a strong signal.

TOOL USAGE GUIDANCE
A productive pattern:
  1. Vocabulary first: call list_entity_types or list_lines_of_business to know what you can filter on.
  2. Narrow the pool by combining filters (resources_exposing × resources_in_lob × resources_in_group). Intersect their outputs in your reasoning.
  3. Inspect typed metadata of the survivors with get_resource.
  4. Use integration_dependencies(namespace) to see cross-system calls declared by a system, and resources_next_after when the activity follows a known prior resource.
  5. When one candidate clearly dominates, call pick_resource with the best match. Before calling pick_resource, use get_resource on your top 2-3 candidates so they appear in the tool history for ranking analysis.

Prefer Agent resources when the activity describes a worker performing something (resolve, diagnose, screen, notify). Prefer apiResource for pure CRUD on master data. Prefer dataProduct for analytical reads.

REFUSAL
If you have explored the landscape and found that no resource fulfils the activity in a substantive way, call the refuse tool with a short reason. Refusing is correct when the activity asks for a capability outside this landscape (e.g. translation, weather data, room booking, payment processing). Refusing is incorrect if there is a plausible ORD resource that just isn't a perfect fit — in that case, pick the best available match.

You have at most 8 tool calls. Be efficient."""


# ─── Tool implementations ──────────────────────────────────────────────────


def _build_indices(resources: list[dict], skills: list[dict]) -> dict:
    """Pre-compute the data structures every tool needs. Pure dicts/sets,
    no side effects. Built once per retrieve() call."""
    by_id = {r["ordId"]: r for r in resources}

    by_ns: dict[str, list[str]] = defaultdict(list)
    by_et: dict[str, list[str]] = defaultdict(list)
    by_lob: dict[str, list[str]] = defaultdict(list)
    by_group: dict[str, list[str]] = defaultdict(list)
    by_cap: dict[str, list[str]] = defaultdict(list)
    next_out: dict[str, list[str]] = defaultdict(list)
    calls_out: dict[str, list[str]] = defaultdict(list)

    for r in resources:
        rid = r["ordId"]
        if r["type"] == "entityType":
            continue
        by_ns[r["namespace"]].append(rid)
        for et in r.get("entityTypes") or []:
            by_et[et].append(rid)
        for lob in r.get("lineOfBusiness") or []:
            by_lob[lob].append(rid)
        for g in r.get("partOfGroups") or []:
            gid = g.get("groupId") if isinstance(g, dict) else None
            if gid:
                by_group[gid].append(rid)
        for cap in r.get("capabilities") or []:
            by_cap[cap].append(rid)
        for nx in r.get("processNext") or []:
            if isinstance(nx, str):
                next_out[rid].append(nx)

    # integrationDependencies — ORD spec does not record an explicit caller
    # resource; the dependency is attached to the package. We therefore
    # collect the dependency targets per namespace: "system N calls these
    # external resources". The agent uses this to discover cross-system
    # links, e.g. "sap.s4 calls sap.ariba:apiResource:PurchaseOrder".
    deps_by_ns: dict[str, list[dict]] = defaultdict(list)
    for ns_dir in sorted(Path(config.LANDSCAPE_DIR).iterdir()):
        if ns_dir.name == "sap.odm":
            continue
        enriched_path = config.LANDSCAPE_ENRICHED_DIR / ns_dir.name / "ord_enriched.json"
        path = enriched_path if enriched_path.exists() else ns_dir / "ord.json"
        if not path.exists():
            continue
        try:
            doc = json.loads(path.read_text())
        except Exception:
            continue
        for dep in doc.get("integrationDependencies", []) or []:
            entry = {
                "ordId": dep.get("ordId", ""),
                "title": dep.get("title", ""),
                "description": (dep.get("description") or "")[:200],
                "targets": [],
            }
            for asp in dep.get("aspects", []) or []:
                for api in asp.get("apiResources", []) or []:
                    tgt = api.get("ordId") if isinstance(api, dict) else None
                    if tgt:
                        entry["targets"].append(tgt)
            if entry["targets"]:
                deps_by_ns[ns_dir.name].append(entry)

    return {
        "by_id": by_id, "by_ns": by_ns, "by_et": by_et, "by_lob": by_lob,
        "by_group": by_group, "by_cap": by_cap, "next_out": next_out,
        "deps_by_ns": deps_by_ns,
        "skills_by_id": {s["skill_id"]: s for s in skills},
        "all_entity_types": sorted(by_et.keys()),
        "all_lobs": sorted(by_lob.keys()),
        "all_groups": sorted(by_group.keys()),
        "all_caps": sorted(by_cap.keys()),
        "all_namespaces": sorted(by_ns.keys()),
    }


def _summary(idx: dict, ord_id: str) -> dict:
    """One-line summary of a resource — what every list_* tool returns
    per entry. Keeps tool outputs compact."""
    r = idx["by_id"].get(ord_id)
    if not r:
        return {"ordId": ord_id, "missing": True}
    return {
        "ordId": r["ordId"],
        "type": r["type"],
        "title": r["title"],
        "shortDescription": r.get("shortDescription", ""),
    }


def _full(idx: dict, ord_id: str) -> dict:
    r = idx["by_id"].get(ord_id)
    if not r:
        return {"ordId": ord_id, "missing": True}
    return {
        "ordId": r["ordId"],
        "type": r["type"],
        "namespace": r["namespace"],
        "title": r["title"],
        "shortDescription": r.get("shortDescription", ""),
        "description": r.get("description", ""),
        "entityTypes": r.get("entityTypes", []),
        "lineOfBusiness": r.get("lineOfBusiness", []),
        "capabilities": r.get("capabilities", []),
        "useCases": r.get("useCases", []),
        "tags": r.get("tags", []),
        "partOfGroups": [
            g.get("groupId") if isinstance(g, dict) else g
            for g in (r.get("partOfGroups") or [])
        ],
        "processNext": r.get("processNext", []),
        "partOfPackage": r.get("partOfPackage", ""),
    }


# Map from tool name → (impl, returns_kind). Impl signature: (idx, **args).

def _tool_impls():
    return {
        "list_namespaces":          lambda idx: idx["all_namespaces"],
        "list_entity_types":        lambda idx: idx["all_entity_types"],
        "list_lines_of_business":   lambda idx: idx["all_lobs"],
        "list_business_groups":     lambda idx: idx["all_groups"],
        "list_capabilities":        lambda idx: idx["all_caps"],
        "resources_exposing":       lambda idx, entity_type: [
            _summary(idx, oid) for oid in idx["by_et"].get(entity_type, [])
        ],
        "resources_in_lob":         lambda idx, line_of_business: [
            _summary(idx, oid) for oid in idx["by_lob"].get(line_of_business, [])
        ],
        "resources_in_group":       lambda idx, group_id: [
            _summary(idx, oid) for oid in idx["by_group"].get(group_id, [])
        ],
        "resources_with_capability": lambda idx, capability: [
            _summary(idx, oid) for oid in idx["by_cap"].get(capability, [])
        ],
        "integration_dependencies": lambda idx, namespace: idx["deps_by_ns"].get(namespace, []),
        "resources_next_after":     lambda idx, ord_id: [
            _summary(idx, oid) for oid in idx["next_out"].get(ord_id, [])
        ],
        "get_resource":             lambda idx, ord_id: _full(idx, ord_id),
        "list_skills":              lambda idx: [
            {"skill_id": s["skill_id"], "description": s["description"][:200]}
            for s in idx["skills_by_id"].values()
        ],
        "describe_skill":           lambda idx, skill_id: (
            None if skill_id not in idx["skills_by_id"]
            else {
                "skill_id": skill_id,
                "description": idx["skills_by_id"][skill_id]["description"],
                "steps": [
                    {"index": st["index"], "name": st["name"],
                     "ord_confirmed": st.get("ord_confirmed", [])}
                    for st in idx["skills_by_id"][skill_id]["steps"]
                ],
            }
        ),
    }


# ─── Tool schemas (OpenAI tool-use format) ──────────────────────────────────


def _schemas() -> list[dict]:
    p = lambda **kw: {"type": "object", "properties": kw,
                       "required": list(kw.keys()), "additionalProperties": False}
    s = lambda desc: {"type": "string", "description": desc}
    no_args = {"type": "object", "properties": {}, "additionalProperties": False}

    tools = [
        ("list_namespaces",
         "List every namespace in the ORD landscape.", no_args),
        ("list_entity_types",
         "List every ODM entity type (full ORD ID).", no_args),
        ("list_lines_of_business",
         "List every line of business token present in the landscape.", no_args),
        ("list_business_groups",
         "List every business-process group (partOfGroups). Enriched-only — "
         "empty on clean ORD.", no_args),
        ("list_capabilities",
         "List every capability token used by any resource.", no_args),
        ("resources_exposing",
         "Resources that expose the given entity type (any resource kind).",
         p(entity_type=s("Full entity-type ORD ID, e.g. sap.odm:entityType:Machine:v1"))),
        ("resources_in_lob",
         "Resources tagged with the given line of business.",
         p(line_of_business=s("e.g. 'Human Resources'"))),
        ("resources_in_group",
         "Resources that are part of the given business-process group "
         "(enriched-only).",
         p(group_id=s("Full group ORD ID, e.g. sap.s4:businessProcess:machine_breakdown_resolution:v1"))),
        ("resources_with_capability",
         "Resources tagged with the given capability token.",
         p(capability=s("e.g. 'skill-matching'"))),
        ("integration_dependencies",
         "Integration dependencies declared by the given system namespace: "
         "each entry lists a title, description, and the external ORD targets "
         "the system reads from or writes to.",
         p(namespace=s("Namespace, e.g. 'sap.s4' or 'corp.itsm'"))),
        ("resources_next_after",
         "Typical successor resources after this one, per processNext "
         "(enriched-only).",
         p(ord_id=s("Full ORD ID of the predecessor"))),
        ("get_resource",
         "Full typed metadata for the given resource: title, descriptions, "
         "entity types, line of business, capabilities, tags, partOfGroups, "
         "processNext.",
         p(ord_id=s("Full ORD ID"))),
        ("list_skills",
         "List every registered skill (process pre-bound to ORD resources).",
         no_args),
        ("describe_skill",
         "Describe one skill: ordered steps with their pinned ord_confirmed "
         "resources. Useful when an upstream router identified a candidate "
         "skill.",
         p(skill_id=s("Skill ID, e.g. 'proc_machine_breakdown'"))),
        ("pick_resource",
         "FINAL TOOL. Commit to one ORD resource as the answer and end the "
         "session.",
         p(ord_id=s("The chosen ORD ID"),
           reason=s("One short sentence explaining the pick"))),
        ("refuse",
         "FINAL TOOL. End the session with NO resource pick because the "
         "landscape does not fulfil the activity. Use when the request "
         "asks for a capability that genuinely is not in the landscape.",
         p(reason=s("One short sentence explaining why no resource fits"))),
    ]
    return [
        {"type": "function",
         "function": {"name": name, "description": desc, "parameters": params}}
        for name, desc, params in tools
    ]


# ─── Public API ─────────────────────────────────────────────────────────────


def retrieve(label: str,
             resources: list[dict],
             top_k: int = config.TOP_K,
             skills: list[dict] | None = None,
             previous_resolved_ord_ids: list[str] | None = None,
             allow_refuse: bool = True) -> dict:
    """Agentic discovery. Signature compatible with A/B/C.

    Parameters
    ----------
    label : the business activity (step name or full user request).
    resources : the loaded ORD landscape.
    top_k : ignored — D commits to exactly one resource.
    skills : optional registry; if None, list_skills/describe_skill return [].
    previous_resolved_ord_ids : optional plan context, passed to the agent
        as part of the user message so it can call resources_next_after.
    """
    skills = skills or []
    idx = _build_indices(resources, skills)
    impls = _tool_impls()

    user_msg = f"Activity:\n  {label}\n"
    if previous_resolved_ord_ids:
        user_msg += (
            f"\nPlan context: the previously resolved step is "
            f"{previous_resolved_ord_ids[0]}. Consider resources_next_after.\n"
        )
    user_msg += "\nUse the tools to identify the best ORD resource and end with pick_resource."

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    tools = _schemas()

    trace_steps: list[dict] = []
    total_tokens = 0
    total_latency = 0.0
    picked_id: str | None = None
    picked_reason: str = ""
    refused: bool = False
    refuse_reason: str = ""

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
        resp = llm._call_with_retry(_do_call, what=f"D-step{step}")
        total_latency += time.time() - t0
        total_tokens += (resp.usage.total_tokens if resp.usage else 0)
        msg = resp.choices[0].message

        if not msg.tool_calls:
            # Agent stopped without picking — record and break.
            trace_steps.append({
                "step": step, "kind": "final_text",
                "content": (msg.content or "")[:300],
            })
            break

        # OpenAI sends back the assistant message with tool_calls; append it
        # before adding tool responses.
        messages.append({
            "role": "assistant", "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            trace_entry: dict = {"step": step, "kind": "tool_call",
                                  "name": name, "args": args}

            if name == "pick_resource":
                picked_id = args.get("ord_id")
                picked_reason = args.get("reason", "")
                result_payload = {"committed": picked_id}
                trace_entry["result"] = result_payload
                trace_steps.append(trace_entry)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result_payload)})
                break

            if name == "refuse":
                if not allow_refuse:
                    # Design-time: refuse not allowed — send tool result and ask agent to pick instead
                    trace_entry["result"] = {"refused_blocked": True, "reason": args.get("reason","")}
                    trace_steps.append(trace_entry)
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": "Refusal not allowed in this context. You must call pick_resource with the best available match."})
                    messages.append({"role": "user", "content": "Please pick the closest matching resource using pick_resource."})
                    continue
                refused = True
                refuse_reason = args.get("reason", "")
                result_payload = {"refused": True, "reason": refuse_reason}
                trace_entry["result"] = result_payload
                trace_steps.append(trace_entry)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result_payload)})
                break

            impl = impls.get(name)
            if not impl:
                payload = {"error": f"unknown tool {name}"}
            else:
                try:
                    payload = impl(idx, **args)
                except TypeError as e:
                    payload = {"error": f"bad args: {e}"}

            trace_entry["result_preview"] = (
                _trunc(payload, 3) if isinstance(payload, list) else payload
            )
            trace_steps.append(trace_entry)
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(payload)[:8000]})

        if picked_id is not None or refused:
            break

    candidates = (
        [{"ordId": picked_id, "score": None}] if picked_id else []
    )
    # If top_k > 1 and we have a pick, surface additional candidates from tool history
    # by scanning get_resource / search_by_title calls for other mentioned ordIds
    if picked_id and top_k > 1:
        seen_ids: list[str] = [picked_id]
        for step in trace_steps:
            result = step.get("result_preview") or {}
            if isinstance(result, list):
                for item in result:
                    oid = item.get("ordId") if isinstance(item, dict) else None
                    if oid and oid != picked_id and oid not in seen_ids:
                        seen_ids.append(oid)
                        if len(seen_ids) >= top_k:
                            break
            if len(seen_ids) >= top_k:
                break
        candidates = [{"ordId": oid, "score": None} for oid in seen_ids]

    return {
        "method": "D",
        "candidates": candidates,
        "trace": {
            "agent_steps": trace_steps,
            "picked_ord_id": picked_id,
            "picked_reason": picked_reason,
            "refused": refused,
            "refuse_reason": refuse_reason,
            "tool_calls": sum(1 for s in trace_steps if s["kind"] == "tool_call"),
            "tokens": total_tokens,
            "latency_s": round(total_latency, 3),
            "llm_calls": sum(1 for s in trace_steps if s["kind"] == "tool_call"),
        },
    }


def _trunc(lst: list, n: int) -> list:
    """Keep traces compact: only the first n entries of a list result."""
    if not isinstance(lst, list):
        return lst
    if len(lst) <= n:
        return lst
    return lst[:n] + [f"... ({len(lst) - n} more)"]
