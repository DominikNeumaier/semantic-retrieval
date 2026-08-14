"""Method E — Plain Ralph: agentic discovery via raw file access.

Contrast to D: D gives the agent a typed ORD vocabulary (15 specialised
tools). E gives the agent only the rawest possible tools — list_dir and
read_file — and lets it navigate the benchmark/landscape/ directory tree
itself.

The question E answers is: does an agent need typed ORD tools to find
the right resource, or can a sufficiently capable reasoning model just
read the JSON files directly and figure things out?

This is the most honest "let the LLM do everything" position. It does
not assume the existence of an ORD parsing layer.

Sandboxing: read access is restricted to the benchmark/ subtree.
Any other path is refused. Read responses are capped at
MAX_READ_LINES per call — an operative survival limit so the
LLM context does not collapse on large ORD documents; the cap
carries no ORD schema knowledge and therefore preserves the
E-vs-D contrast (see thesis Sec. 4.2).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from src import config, llm


MAX_TOOL_CALLS = 10           # E needs more calls since each is a raw read
MAX_READ_LINES = 200          # line-based cap per read_file (operative survival
                              # limit; carries no ORD-schema knowledge, so the
                              # E-vs-D contrast is preserved — see Sec. 4.2)
ALLOWED_ROOTS = ("benchmark",)  # whitelist for path access

# Persistent notes: the agent may write to and read from a shared scratch
# file that survives across retrieve() calls. This is the Ralph-style
# "learning through note-taking" mechanism — what the agent figures out
# about the ORD schema in case 1 is available in case 2. The notes are
# automatically pre-loaded into the system prompt of every run so the
# agent can immediately benefit from prior insights without spending
# tool calls on a read.
NOTES_PATH = config.CACHE_DIR / "method_raw_notes.md"
NOTES_MAX_BYTES = 4096        # cap so notes can't grow unboundedly

# Shared client via llm.get_clients() so token refresh propagates here.


def reset_notes() -> None:
    """Wipe the persistent notes. Call once at the start of a benchmark
    run so cases are evaluated against a deterministic starting state."""
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTES_PATH.write_text("")


def _load_notes() -> str:
    if not NOTES_PATH.exists():
        return ""
    try:
        return NOTES_PATH.read_text()[:NOTES_MAX_BYTES]
    except Exception:
        return ""


def _save_notes(text: str) -> dict:
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    truncated = len(text) > NOTES_MAX_BYTES
    NOTES_PATH.write_text(text[:NOTES_MAX_BYTES])
    return {
        "stored_bytes": min(len(text), NOTES_MAX_BYTES),
        "truncated": truncated,
        "cap_bytes": NOTES_MAX_BYTES,
    }


_SYSTEM_BASE = """You are an ORD discovery agent operating directly on the filesystem. You have five tools:
  - list_dir(path) : list entries in a directory under the project landscape.
  - read_file(path): read the contents of an ORD JSON file (capped at 200 lines).
  - read_notes()   : read your persistent notes from previous sessions.
  - write_notes(content): overwrite your persistent notes (max 4KB).
  - pick_resource(ord_id, reason) OR refuse(reason): final action.

End EITHER with pick_resource (a resource in the landscape fulfils the activity) OR with refuse (no resource in this landscape fulfils the activity — do not pick a tangentially related one).

ORD GROUNDING
ORD (Open Resource Discovery) is a typed metadata protocol. The landscape lives under benchmark/landscape/systems/<system>/ord.json (clean state) and benchmark/landscape/systems_enriched/<system>/ord_enriched.json (enriched state, with partOfGroups, processNext, capabilities).

Each ORD document has top-level arrays: apiResources, agents, dataProducts, entityTypes, integrationDependencies, packages. A resource has an ordId in the form <namespace>:<type>:<localId>:<version> plus title, shortDescription, description, entityTypes (or exposedEntityTypes / relatedEntityTypes), lineOfBusiness, capabilities, tags, partOfGroups, processNext.

PERSISTENT NOTES
You have a notebook that survives across queries. Use it to record what you have learned about the landscape — system inventory, which entity types live where, useful navigation patterns, recurring distractors. On every new query, the notes are shown to you below; use them to skip re-discovery. After resolving a query, update the notes with anything new and useful you learned. Keep notes concise — every wasted byte is a future read cost. Maximum 4KB.

YOUR JOB
Given a business activity, navigate the landscape directory (or trust your notes when they cover the question), and either pick the single ORD resource that best fulfils the activity, or refuse if none does.

REFUSAL
If you have explored the landscape and confirmed that no resource fulfils the activity in a substantive way, call refuse(reason). Refusing is correct when the activity asks for a capability outside this landscape (e.g. translation, weather data, room booking, payment processing). Refusing is incorrect if there is a plausible ORD resource that just isn't a perfect fit — in that case, pick the best available match.

EFFICIENCY
You have at most 10 tool calls. If your notes already cover the activity, you may pick or refuse directly. Otherwise read 1-3 system documents — there are 10 systems and reading them all would exhaust your budget."""


# ─── Sandboxing ─────────────────────────────────────────────────────────────


def _safe_path(rel: str) -> Path | None:
    """Resolve `rel` under config.ROOT, refuse if it escapes the allowed
    roots."""
    if not rel:
        return None
    rel = rel.strip().lstrip("/")
    target = (config.ROOT / rel).resolve()
    try:
        target.relative_to(config.ROOT.resolve())
    except ValueError:
        return None
    # Restrict to landscape/* subtree
    parts = target.relative_to(config.ROOT.resolve()).parts
    if not parts or parts[0] not in ALLOWED_ROOTS:
        return None
    return target


# ─── Tool implementations ──────────────────────────────────────────────────


def _list_dir(rel: str) -> dict:
    p = _safe_path(rel)
    if not p:
        return {"error": f"path not allowed: {rel!r}. Allowed roots: {ALLOWED_ROOTS}"}
    if not p.exists():
        return {"error": f"not found: {rel}"}
    if not p.is_dir():
        return {"error": f"not a directory: {rel}"}
    entries = []
    for child in sorted(p.iterdir()):
        entries.append({
            "name": child.name,
            "kind": "dir" if child.is_dir() else "file",
        })
    return {"path": rel, "entries": entries}


def _read_file(rel: str) -> dict:
    p = _safe_path(rel)
    if not p:
        return {"error": f"path not allowed: {rel!r}"}
    if not p.exists() or not p.is_file():
        return {"error": f"file not found: {rel}"}
    text = p.read_text()
    lines = text.splitlines()
    truncated = len(lines) > MAX_READ_LINES
    if truncated:
        omitted = len(lines) - MAX_READ_LINES
        content = "\n".join(lines[:MAX_READ_LINES]) + (
            f"\n\n[... file truncated; {omitted} more line(s) of "
            f"{len(lines)} total. Read a different file or accept "
            f"the partial view.]"
        )
    else:
        content = text
    return {
        "path": rel,
        "content": content,
        "truncated": truncated,
        "total_lines": len(lines),
    }


# ─── Tool schemas (OpenAI tool-use format) ──────────────────────────────────


def _schemas() -> list[dict]:
    s = lambda d: {"type": "string", "description": d}
    p = lambda **kw: {"type": "object", "properties": kw,
                       "required": list(kw.keys()), "additionalProperties": False}
    no_args = {"type": "object", "properties": {}, "additionalProperties": False}
    return [
        {"type": "function", "function": {
            "name": "list_dir",
            "description": "List entries under a directory in the landscape "
                           "subtree. Start with path='landscape' to see the "
                           "systems.",
            "parameters": p(path=s("Relative path, e.g. 'landscape' or "
                                    "'benchmark/landscape/systems/sap.s4'")),
        }},
        {"type": "function", "function": {
            "name": "read_file",
            "description": "Read an ORD JSON file (or any file in the "
                           "landscape subtree). Capped at 200 lines per "
                           "call; a 'truncated' flag in the result tells "
                           "you whether content was cut off.",
            "parameters": p(path=s("Relative path, e.g. "
                                    "'benchmark/landscape/systems_enriched/sap.s4/ord_enriched.json'")),
        }},
        {"type": "function", "function": {
            "name": "read_notes",
            "description": "Read the persistent notes you have written across "
                           "previous sessions. The notes are also shown in the "
                           "system prompt automatically; this tool returns the "
                           "live disk copy.",
            "parameters": no_args,
        }},
        {"type": "function", "function": {
            "name": "write_notes",
            "description": "Overwrite your persistent notes with the given "
                           "content. Use to record landscape insights (system "
                           "inventory, distractor patterns, useful heuristics) "
                           "for future queries. Capped at 4KB.",
            "parameters": p(content=s("The full notes text to save. "
                                       "Overwrites any prior content.")),
        }},
        {"type": "function", "function": {
            "name": "pick_resource",
            "description": "FINAL TOOL. Commit to one ORD resource as the "
                           "answer and end the session.",
            "parameters": p(ord_id=s("The chosen ORD ID"),
                             reason=s("One short sentence explaining the pick")),
        }},
        {"type": "function", "function": {
            "name": "refuse",
            "description": "FINAL TOOL. End the session with NO resource pick "
                           "because the landscape does not fulfil the activity. "
                           "Use when the request asks for a capability that is "
                           "genuinely not in the landscape.",
            "parameters": p(reason=s("One short sentence explaining why no "
                                      "resource fits")),
        }},
    ]


# ─── Public API ─────────────────────────────────────────────────────────────


def retrieve(label: str,
             resources: list[dict],
             top_k: int = config.TOP_K,
             skills: list[dict] | None = None,
             previous_resolved_ord_ids: list[str] | None = None,
             allow_refuse: bool = True) -> dict:
    """Plain-Ralph agentic discovery. Signature compatible with A/B/C/D.

    Parameters
    ----------
    label : business activity.
    resources, top_k, skills, previous_resolved_ord_ids :
        Accepted for signature compatibility; D2 ignores all of them and
        navigates the filesystem itself. (The agent could read the skills
        directory too if it wanted — also a deliberate part of the test.)
    """
    # Pre-load the persistent notes into the system prompt so the agent
    # doesn't have to spend a read_notes call to see them. Saves one tool
    # call per query while keeping read_notes available for cases where
    # the agent updated notes mid-conversation (rare but possible).
    notes_now = _load_notes()
    notes_block = (
        f"\n\nPERSISTENT NOTES (from prior sessions, ≤{NOTES_MAX_BYTES} bytes):\n"
        f"{notes_now if notes_now else '(empty — no notes yet)'}\n"
    )
    system_prompt = _SYSTEM_BASE + notes_block

    user_msg = f"Activity:\n  {label}\n"
    if previous_resolved_ord_ids:
        user_msg += (
            f"\nPlan context: previously resolved step is "
            f"{previous_resolved_ord_ids[0]}.\n"
        )
    user_msg += ("\nNavigate the landscape directory (or trust your notes), "
                  "either pick the best ORD resource (pick_resource) or refuse "
                  "if no resource fulfils the activity (refuse). Update your "
                  "notes if you learned something new.")

    messages = [
        {"role": "system", "content": system_prompt},
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
        resp = llm._call_with_retry(_do_call, what=f"E-step{step}")
        total_latency += time.time() - t0
        total_tokens += (resp.usage.total_tokens if resp.usage else 0)
        msg = resp.choices[0].message

        if not msg.tool_calls:
            trace_steps.append({
                "step": step, "kind": "final_text",
                "content": (msg.content or "")[:300],
            })
            break

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
                payload = {"committed": picked_id}
            elif name == "refuse":
                if not allow_refuse:
                    # Design-time: refuse blocked — tell agent to pick instead
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": "Refusal not allowed. Pick the best available resource with pick_resource."})
                    messages.append({"role": "user", "content": "Please use pick_resource to select the closest matching resource."})
                    continue
                refused = True
                refuse_reason = args.get("reason", "")
                payload = {"refused": True, "reason": refuse_reason}
            elif name == "list_dir":
                payload = _list_dir(args.get("path", ""))
            elif name == "read_file":
                payload = _read_file(args.get("path", ""))
            elif name == "read_notes":
                payload = {"content": _load_notes() or "(empty)"}
            elif name == "write_notes":
                payload = _save_notes(args.get("content", ""))
            else:
                payload = {"error": f"unknown tool {name}"}

            trace_entry["result_preview"] = _preview(payload)
            trace_steps.append(trace_entry)
            messages.append({"role": "tool", "tool_call_id": tc.id,
                              "content": json.dumps(payload)[:8500]})

        if picked_id is not None or refused:
            break

    candidates = [{"ordId": picked_id, "score": None}] if picked_id else []
    # If top_k > 1, surface additional ordIds the agent read from the filesystem
    if picked_id and top_k > 1:
        import re as _re
        seen_ids: list[str] = [picked_id]
        for step in trace_steps:
            payload = step.get("payload") or step.get("result_preview") or {}
            text = json.dumps(payload) if not isinstance(payload, str) else payload
            for oid in _re.findall(r'[a-z]+\.[a-z]+:[a-z]+:[A-Za-z0-9_]+:v\d+', text):
                if oid != picked_id and oid not in seen_ids:
                    seen_ids.append(oid)
                if len(seen_ids) >= top_k:
                    break
            if len(seen_ids) >= top_k:
                break
        candidates = [{"ordId": oid, "score": None} for oid in seen_ids]

    return {
        "method": "E",
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


def _preview(payload: dict) -> dict:
    """Shrink large payloads for trace storage."""
    if not isinstance(payload, dict):
        return payload
    if "content" in payload and isinstance(payload["content"], str):
        return {**payload, "content": payload["content"][:200] + " …(truncated)"}
    if "entries" in payload and isinstance(payload["entries"], list):
        return {**payload, "entries_count": len(payload["entries"])}
    return payload
