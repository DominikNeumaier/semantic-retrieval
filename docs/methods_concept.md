# Retrieval Methods A–E

Five retrieval strategies evaluated against the 273-resource ORD benchmark landscape. Each method receives a `label` (activity text) and the flat resource list, and returns `{"candidates": [{ordId, score}], "trace": {...}}`.

---

## Method A — Embedding Retrieval

**Paradigm:** Dense vector similarity. No LLM reasoning, no ORD structure.

**How it works:**
1. Embed the activity label with the embedding model
2. Embed each resource: `title | shortDescription | description [| capabilities | partOfGroups | processNext]`
3. Return top-k by cosine similarity

**ORD interaction:** Text fields only. On Enriched-ORD, `capabilities`, `partOfGroups.groupId`, and `processNext` are appended to the embedding input — this is the primary enrichment signal for Method A.

**LLM calls:** 0 (embedding only)

**Strengths:** Fully deterministic given a fixed model, fast once embeddings are cached.  
**Weakness:** Cannot reason about ORD type structure (agent vs. API vs. dataProduct).

---

## Method B — Progressive Disclosure

**Paradigm:** Hierarchical funnel — 4 sequential LLM stages, each sees only the slice it needs.

**How it works:**
1. **Stage 1 — Namespace** (LLM): Pick up to 2 of 10 namespaces. Input: namespace summaries.
2. **Stage 2 — Modality** (LLM): Operational (agent + apiResource) vs. analytical (dataProduct). Input: modality counts in surviving pool.
3. **Stage 3 — Entity type** (LLM): Pick up to 3 ODM entity types. Input: ETs present in surviving pool.
4. **Stage 4 — Tool** (LLM): Pick the final resource(s). Input: final candidate slice with title, type, shortDescription, and (on Enriched-ORD) capabilities, processNext, partOfGroups.

**ORD interaction:** Exploits ORD's namespace hierarchy, resource type (modality), and entity type vocabulary. On Enriched-ORD, enrichment fields appear in Stage 4 to break ties.

**LLM calls:** 4 per query (can be fewer if early stages produce an empty pool)

**Strengths:** Mirrors how a human would browse ORD — narrows scope progressively.  
**Weakness:** Early-stage errors (wrong namespace) are not recovered.

---

## Method C — Multi-Hop Graph Walk

**Paradigm:** Knowledge graph traversal using ORD's typed edges, plus 2 LLM calls (anchor inference + rerank).

**How it works:**
1. **Anchor inference** (LLM): Map activity to anchor concept nodes from controlled vocabularies: entity types, business process groups (`partOfGroups.groupId`), lines of business.
2. **Seed** (deterministic): Resources matching any anchor become seed nodes (weight = number of anchors matched).
3. **Walk** (deterministic, up to `MAX_HOPS=2`):
   - `co_exposes`: shared entityType → weight 1.0
   - `co_partOf`: shared processGroup → weight 2.0
   - `calls`: integrationDependency → weight 3.0
   - `processNext`: sequential process link → weight 3.0
   - Each hop discounts by `HOP_DECAY=0.5`
4. **Rerank** (LLM): Top `RERANK_POOL=8` walk results passed to a re-ranker with title, type, capabilities, shortDescription.

**ORD interaction:** Directly exploits graph structure: entityTypes (co-exposes edges), partOfGroups (co-partOf edges), processNext (process edges), integrationDependencies (calls edges). On Enriched-ORD, processNext edges and partOfGroups become active → significantly more graph connectivity.

**LLM calls:** 2 per query

**Strengths:** Finds resources that are structurally connected, not just textually similar. Benefits strongly from enrichment.  
**Weakness:** Anchor inference errors starve the walk; sparse graphs (Clean-ORD) may miss relevant resources.

---

## Method D — Agentic Tool Use

**Paradigm:** LLM agent with 15 typed tools — a full ORD API surface plus Skill Registry access.

**Tool groups:**
- **Lookup tools:** `get_resource`, `search_by_title`, `search_by_capability`
- **Filter tools:** `resources_by_namespace`, `resources_by_type`, `resources_by_entity_type`, `resources_by_lob`, `resources_in_group`
- **Graph tools:** `get_process_next`, `get_integration_deps`
- **Skill tools:** `list_skills`, `describe_skill`
- **Final action:** `pick_resource(ord_id, reason)` or `refuse(reason)`

**ORD interaction:** Complete typed ORD API. The agent chooses which tools to call and in what order. On Enriched-ORD, `resources_in_group`, `get_process_next`, and `search_by_capability` become meaningful. Skills provide a second semantic layer: `describe_skill` reveals `ord_confirmed` ordIds per step.

**LLM calls:** 1–N (agent loop, typically 3–6 tool calls per query)

**Strengths:** Most flexible — can combine multiple ORD dimensions in a single query. Can use Skill Registry to short-circuit retrieval.  
**Weakness:** Expensive; non-deterministic; may waste calls on dead ends.

---

## Method E — Filesystem Agent

**Paradigm:** Minimal tool surface — the agent navigates the raw ORD JSON files directly.

**Tools:** `list_dir(path)`, `read_file(path)` (capped at 200 lines), `pick_resource(ord_id, reason)`, `refuse(reason)`, plus persistent notes (`read_notes`, `write_notes`).

**Allowed paths:** `benchmark/` subtree only.

**How it works:**
- Agent starts cold (no pre-loaded index)
- Navigates `benchmark/landscape/systems/<ns>/ord.json` or `benchmark/landscape/systems_enriched/<ns>/ord_enriched.json`
- Reads 1–3 system files guided by persistent notes from previous queries
- Maximum 10 tool calls per query

**ORD interaction:** Raw JSON — agent must parse ORD structure itself. Notes help it remember which systems contain which entity types across queries. On Enriched-ORD, the agent can read `capabilities`, `processNext`, `partOfGroups` directly from the JSON.

**LLM calls:** 1 (agent loop with embedded tool calls)

**Strengths:** Most similar to a real zero-shot agent. Tests whether raw ORD JSON is navigable without any pre-computed index.  
**Weakness:** Slow; limited by read cap; cannot see the full landscape in one pass.

---

## Evaluation Interface

All methods expose the same contract:

```python
result = retrieve(label: str, resources: list[dict], top_k: int = 5) -> dict
# Returns:
# {
#   "method": "A" | "B" | "C" | "D" | "E",
#   "candidates": [{"ordId": str, "score": float}, ...],  # len <= top_k
#   "trace": {"tokens": int, "latency_s": float, ...}
# }
```

Methods D and E additionally accept `skills: list[dict] | None` for Skill Registry access.

---

## ORD Fields Used per Method

**Clean-ORD fields** (always present):

| Field | A | B | C | D | E |
|---|---|---|---|---|---|
| title, shortDescription, description | ✓ embed | ✓ | rerank | ✓ tool | raw |
| entityTypes | — | ✓ filter | ✓ edge | ✓ tool | raw |
| lineOfBusiness | — | — | ✓ anchor | ✓ tool | raw |
| tags | — | — | — | ✓ tool | raw |
| namespace | — | ✓ filter | ✓ anchor | ✓ tool | ✓ dir |
| type (agent/API/DP) | — | ✓ filter | rerank | ✓ tool | raw |
| integrationDependencies | — | — | ✓ edge | ✓ tool | raw |

**Enriched-ORD fields** (added by design-time flow, GT resources only):

| Field | A | B | C | D | E |
|---|---|---|---|---|---|
| capabilities | ✓ embed | ✓ Stage 4 | ✓ rerank | ✓ tool | raw |
| useCases | ✓ embed | ✓ Stage 4 | ✓ rerank | ✓ tool | raw |
| processNext | ✓ embed | ✓ Stage 4 | ✓ edge | ✓ tool | raw |
| partOfGroups | ✓ embed | ✓ Stage 4 | ✓ edge/anchor | ✓ tool | raw |

`raw` = Method E reads directly from JSON file.

**Skill Registry** is a separate semantic layer outside ORD — SKILL.md files with `ord_confirmed` annotations. Available to Method D via `list_skills()` / `describe_skill()`. Not an ORD field.
