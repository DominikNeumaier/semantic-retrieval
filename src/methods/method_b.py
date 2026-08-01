"""Method B — Progressive Disclosure over ORD's hierarchy.

ORD IDs are hierarchical:  <namespace> : <type> : <localId> : <version>
and resources carry typed entity references.

This method descends that hierarchy one level at a time. Each LLM call
sees only the slice it needs to pick the next level, never the full
landscape. Four small calls instead of one big one:

  Stage 1  pick up to 2 namespaces      ← sees: 10 NS summaries
  Stage 2  pick modality                ← sees: operational vs analytical
                                          (Agent + apiResource are grouped
                                          as 'operational' — both call into
                                          live systems; dataProduct +
                                          eventResource as 'analytical' —
                                          they expose read-only or async
                                          signals)
  Stage 3  pick entity types            ← sees: ETs of the surviving pool
  Stage 4  pick the tool                ← sees: the final candidate slice;
                                          the agent-vs-api distinction is
                                          decided here implicitly from
                                          titles and descriptions. On the
                                          enriched ORD landscape the
                                          partOfGroups / processNext /
                                          capabilities fields are added
                                          per candidate so process-context
                                          can break ties.

This is the "progressive disclosure" paradigm: a hierarchical filter on
the typed structure of ORD. It is intentionally NOT a graph walk and
NOT free-text matching — it leans on ORD's namespace, modality, and
entity hierarchy.
"""

from __future__ import annotations

import json
import re

from src.core import config
from src.core import llm


MAX_NAMESPACES = 2

# Two-class modality. Operational covers the live invocation surface
# (Agents that act, APIs they call). Analytical covers read-only or
# asynchronous surfaces. Most activities are operational; an analytics
# or notification activity flips to analytical.
MODALITIES = {
    "operational": {"agent", "apiResource"},
    "analytical":  {"dataProduct", "eventResource"},
}


_NS_SYS = """You select up to two ORD namespaces that most likely contain the resources for an activity. Most activities sit in one namespace; some genuinely span two (e.g. HR + IT for onboarding). Respond with one JSON object:
{"namespaces": ["<ns1>", "<ns2 if needed>"], "reason": "<one short sentence>"}"""


_MOD_SYS = """You decide the modality of an activity:

  - "operational"  the activity drives live execution: an agent does work, or an API reads / writes master data. Pick this whenever the activity creates, updates, approves, screens, notifies, or otherwise acts.
  - "analytical"   the activity is purely analytical or event-driven: read-only KPIs, dashboards, telemetry, asynchronous notifications.

Most enterprise activities are operational. Return both modalities only if the activity genuinely needs an analytical lookup alongside the operational work.

Respond with one JSON object:
{"modalities": ["operational"], "reason": "<one short sentence>"}"""


_ET_SYS = """You decide which ODM entity types the activity primarily concerns. Pick up to 3 from the listed entity types — these are the business objects the activity reads or writes. Respond with one JSON object:
{"entity_types": ["sap.odm:entityType:Machine:v1", ...], "reason": "<one short sentence>"}

Use an empty list if no entity type clearly applies (the activity will then pass through this stage unfiltered)."""


_TOOL_SYS = """You select up to {k} ORD resource IDs that best match the activity. Respond with a JSON array of ORD IDs ordered by relevance, no other text.

If NONE of the candidates is a meaningful match for the activity (e.g. the activity asks for a capability that is genuinely not in the landscape such as translation, weather data, room booking, e-signature, sentiment analysis), return an empty array [] instead of picking a tangentially related resource."""


# ─── Prompts ────────────────────────────────────────────────────────────────


def _ns_prompt(label: str, model: dict) -> str:
    lines = ["Activity:", f"  {label}", "", "Namespaces:"]
    for ns, info in model.items():
        bits = [f"{info['resource_count']} resources"]
        if info["dominant_entity_types"]:
            bits.append("entities: " + ", ".join(info["dominant_entity_types"]))
        if info["lines_of_business"]:
            bits.append("LoB: " + ", ".join(info["lines_of_business"]))
        lines.append(f"  - {ns} ({'; '.join(bits)})")
    return "\n".join(lines) + f'\n\nReturn JSON with up to {MAX_NAMESPACES} namespaces.'


def _mod_prompt(label: str, mod_counts: dict[str, int]) -> str:
    lines = ["Activity:", f"  {label}", "", "Modalities available in chosen namespace(s):"]
    for m, types in MODALITIES.items():
        n = mod_counts.get(m, 0)
        if n:
            lines.append(f"  - {m} ({n} resources: {', '.join(sorted(types))})")
    return "\n".join(lines) + '\n\nReturn JSON: {"modalities": [...], "reason": "..."}'


def _et_prompt(label: str, et_counts: dict[str, int]) -> str:
    """Stage 3: pick entity types from the ETs that are actually present
    in the surviving pool. We show only what is reachable from here."""
    lines = ["Activity:", f"  {label}", "", "Entity types exposed by the surviving candidates:"]
    for et, n in sorted(et_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        local = et.split(":")[-2] if et.count(":") >= 3 else et
        lines.append(f"  - {et}  ({local}, {n} resource{'s' if n != 1 else ''})")
    return "\n".join(lines) + "\n\nReturn JSON with up to 3 entity types, or an empty list."


def _tool_prompt(label: str, resources: list[dict], k: int) -> str:
    lines = ["Activity:", f"  {label}", "", f"Candidate resources ({len(resources)}):"]
    for r in resources:
        ets = [e.split(":")[-2] for e in (r.get("entityTypes") or [])]
        et_part = f"  [entities: {', '.join(ets)}]" if ets else ""
        lines.append(f"  - {r['ordId']} | {r['title']} | {r['shortDescription']}{et_part}")
        # Enriched ORD context (partOfGroups / processNext / capabilities)
        # — written back by the design-time data flow. Surfaced here so the
        # final tie-break can use process membership, sequencing and
        # behavioural labels instead of only titles + entity types.
        # partOfGroups entries are {groupId, groupTypeId} dicts per ORD spec;
        # processNext entries are bare ord-id strings. Extract the local part
        # of each so the prompt stays compact.
        groups = r.get("partOfGroups") or []
        if groups:
            short_g = []
            for g in groups[:3]:
                gid = g.get("groupId", "") if isinstance(g, dict) else str(g)
                short_g.append(gid.split(":")[-1] if gid else "")
            short_g = [g for g in short_g if g]
            if short_g:
                lines.append(f"      processes: {', '.join(short_g)}")
        nexts = r.get("processNext") or []
        if nexts:
            short_n = []
            for n in nexts[:3]:
                nid = n if isinstance(n, str) else n.get("ordId", "")
                if nid and nid.count(":") >= 3:
                    short_n.append(nid.split(":")[-2])
                elif nid:
                    short_n.append(nid)
            if short_n:
                lines.append(f"      next: {', '.join(short_n)}")
        caps = r.get("capabilities") or []
        if caps:
            lines.append(f"      capabilities: {', '.join(caps[:5])}")
        use_cases = r.get("useCases") or []
        if use_cases:
            lines.append(f"      useCases: {use_cases[0]}")
    return "\n".join(lines) + f"\n\nReturn a JSON array of up to {k} ORD IDs."


# ─── Parsers ────────────────────────────────────────────────────────────────


def _parse_list(text: str, key: str, valid: set[str]) -> list[str]:
    """Parse {key: [...]} from a JSON object embedded in text."""
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    arr = obj.get(key) or []
    if isinstance(arr, str):
        arr = [arr]
    return [x for x in arr if isinstance(x, str) and x in valid]


def _parse_array(text: str) -> list[str]:
    m = re.search(r"\[[\s\S]*?\]", text)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
        return [str(x) for x in arr if isinstance(x, str)]
    except json.JSONDecodeError:
        return []


# ─── Landscape summary (Stage 1 input) ──────────────────────────────────────


def _ns_summary(resources: list[dict]) -> dict:
    """Per-namespace summary: count + dominant ETs + LoBs + sample titles."""
    by_ns: dict[str, list[dict]] = {}
    for r in resources:
        by_ns.setdefault(r["namespace"], []).append(r)

    out: dict[str, dict] = {}
    for ns, items in sorted(by_ns.items()):
        et_count: dict[str, int] = {}
        for it in items:
            for et in it.get("entityTypes") or []:
                et_count[et] = et_count.get(et, 0) + 1
        top_ets = [e.split(":")[-2] for e, _ in
                   sorted(et_count.items(), key=lambda kv: -kv[1])[:3]]
        lobs: list[str] = []
        for it in items:
            for lob in it.get("lineOfBusiness") or []:
                if lob not in lobs:
                    lobs.append(lob)
        out[ns] = {
            "resource_count": len(items),
            "dominant_entity_types": top_ets,
            "lines_of_business": lobs[:4],
        }
    return out


# ─── Public API ─────────────────────────────────────────────────────────────


def retrieve(label: str, resources: list[dict], top_k: int = config.TOP_K, allow_refuse: bool = True) -> dict:
    """Progressive Disclosure: 4 LLM calls, each looking at a smaller slice."""

    # ─── Stage 1: namespace ────────────────────────────────────────────────
    model = _ns_summary(resources)
    valid_ns = set(model.keys())
    ns_text, ns_meta = llm.chat(_ns_prompt(label, model), system=_NS_SYS)
    chosen_ns = _parse_list(ns_text, "namespaces", valid_ns)[:MAX_NAMESPACES]

    if not chosen_ns:
        return _empty(ns_text, ns_meta)

    pool1 = [r for r in resources if r["namespace"] in set(chosen_ns)]

    # ─── Stage 2: modality (operational vs analytical) ─────────────────────
    type_to_mod = {t: m for m, ts in MODALITIES.items() for t in ts}
    mod_counts: dict[str, int] = {}
    for r in pool1:
        m = type_to_mod.get(r["type"])
        if m:
            mod_counts[m] = mod_counts.get(m, 0) + 1
    mod_text, mod_meta = llm.chat(
        _mod_prompt(label, mod_counts), system=_MOD_SYS,
    )
    chosen_mods = _parse_list(mod_text, "modalities", set(MODALITIES.keys()))
    # Fallback: keep all modalities present in pool1
    if not chosen_mods:
        chosen_mods = list(mod_counts.keys())

    allowed_types: set[str] = set()
    for m in chosen_mods:
        allowed_types |= MODALITIES[m]
    pool2 = [r for r in pool1 if r["type"] in allowed_types]

    # ─── Stage 3: entity type ──────────────────────────────────────────────
    # Show only the ETs that the surviving pool actually exposes. The LLM
    # may return an empty list, in which case the stage is a pass-through.
    et_counts: dict[str, int] = {}
    for r in pool2:
        for et in r.get("entityTypes") or []:
            et_counts[et] = et_counts.get(et, 0) + 1
    if et_counts:
        et_text, et_meta = llm.chat(
            _et_prompt(label, et_counts), system=_ET_SYS,
        )
        chosen_ets = _parse_list(et_text, "entity_types", set(et_counts.keys()))
    else:
        et_text, et_meta = "", {"tokens": 0, "latency": 0.0}
        chosen_ets = []

    if chosen_ets:
        chosen_et_set = set(chosen_ets)
        pool3 = [r for r in pool2
                 if set(r.get("entityTypes") or []) & chosen_et_set]
        if not pool3:
            pool3 = pool2  # never filter to zero
    else:
        pool3 = pool2

    # ─── Stage 4: tool ─────────────────────────────────────────────────────
    tool_text, tool_meta = llm.chat(
        _tool_prompt(label, pool3, top_k),
        system=_TOOL_SYS.format(k=top_k),
    )
    valid_ids = {r["ordId"] for r in pool3}
    parsed_ids = _parse_array(tool_text)
    ord_ids = [oid for oid in parsed_ids if oid in valid_ids][:top_k]
    # Distinguish a deliberate refusal (parser saw a JSON array, possibly
    # empty, with no candidate matching) from a hard parse failure. The
    # heuristic: if the response contained a literal "[]" or a JSON array
    # that yielded zero valid IDs, treat as refusal; otherwise as parse
    # failure that the bench should not score as refusal.
    refused = (allow_refuse and bool(re.search(r"\[\s*\]", tool_text))) or (
        bool(parsed_ids) and not ord_ids
    )

    return {
        "method": "B",
        "candidates": [{"ordId": oid, "score": None} for oid in ord_ids],
        "trace": {
            "stage1_namespaces": chosen_ns,
            "stage1_raw": ns_text.strip(),
            "stage2_modalities": chosen_mods,
            "stage2_raw": mod_text.strip(),
            "stage2_pool_size": len(pool2),
            "stage3_entity_types": chosen_ets,
            "stage3_raw": et_text.strip() if et_text else "",
            "stage3_pool_size": len(pool3),
            "stage4_raw": tool_text.strip(),
            "stage4_valid_count": len(ord_ids),
            "refused": refused,
            "tokens": (ns_meta["tokens"] + mod_meta["tokens"]
                       + et_meta["tokens"] + tool_meta["tokens"]),
            "latency_s": round(
                ns_meta["latency"] + mod_meta["latency"]
                + et_meta["latency"] + tool_meta["latency"], 3
            ),
            "llm_calls": 4 if et_counts else 3,
        },
    }


def _empty(ns_text: str, ns_meta: dict) -> dict:
    return {
        "method": "B",
        "candidates": [],
        "trace": {
            "stage1_namespaces": [],
            "stage1_raw": ns_text,
            "stage2_skipped": True,
            "tokens": ns_meta["tokens"],
            "latency_s": round(ns_meta["latency"], 3),
            "llm_calls": 1,
        },
    }
