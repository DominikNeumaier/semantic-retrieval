# Implementation Reference

This document describes the complete implementation of the benchmark — retrieval methods, evaluation runners, runtime planner, and supporting infrastructure. It is the technical counterpart to `docs/BENCHMARK.md`, which describes the scientific design.

---

## Module Overview

```
src/
  core/           — Shared infrastructure (LLM client, ORD loader, config)
  methods/        — Retrieval methods A–E + S (Baseline-Solver)
  design_time/    — Design-time benchmark runner and case loader
  runtime/        — Runtime benchmark runner, planner, skill registry
  IMPLEMENTATION.md  — This file
```

---

## core/

**`config.py`** — Central configuration. Reads `.env` from project root. Key paths:
- `LANDSCAPE_DIR` → `benchmark/landscape/systems/`
- `LANDSCAPE_ENRICHED_DIR` → `benchmark/landscape/systems_enriched/`
- `DT_OUTPUT_DIR` → `benchmark/test_cases/design_time/output/`
- `RT_OUTPUT_DIR` → `benchmark/test_cases/runtime/output/`
- `RESULTS_DT` / `RESULTS_RT` → `results/design-time/` / `results/runtime/`

**`llm.py`** — LLM and embedding client with file-based cache (hash of model+prompt). Every call returns `(text, {"tokens": int, "latency_s": float, "cached": bool})`. Auto-refreshes JWT token on 401 errors.

**`ord_loader.py`** — Loads the flat resource list from the landscape. `load_landscape(state)`:
- `"clean"` → reads `systems/{ns}/ord.json`
- `"enriched"` → reads `systems_enriched/{ns}/ord_enriched.json`, falls back to clean

Returns list of dicts with `ordId`, `namespace`, `type`, `title`, `shortDescription`, `description`, `entityTypes`, `lineOfBusiness`, `tags`, `capabilities`, `useCases`, `processNext`, `partOfGroups`.

---

## design_time/

**`dt_cases.py`** — Loads and normalises design-time cases from `activity_cases.json`. Adds process title prefix to the label (e.g. `"ManufacturingChangeCompliance process: Diagnose equipment failure — ..."`). Returns dicts with `case_id`, `process`, `process_title`, `label`, `label_raw`, `expected_ordIds`, `is_gt`.

**`dt_benchmark.py`** — Design-time evaluation runner. Runs all 6 methods × 240 cases × 1 ORD state.
- `--methods A B C D E S` — which methods to run
- `--state clean|enriched` — which ORD state
- `--limit N` — smoke test (first N cases)
- `--no-wipe` — keep existing results, only update specified methods
- Outputs: `results/design-time/summary.json`, `by_process.json`, `by_modeling_notation.json`, `traces/<method>/<case_id>.json`

Note: For Design-Time, all methods use `allow_refuse=False` — the benchmark always has a known GT resource so refusing is a bug, not a feature.

---

## runtime/

**`rt_benchmark.py`** — Runtime evaluation runner. Runs all methods × modes × 2 ORD states.
- `--methods A B C D E S`
- `--states clean enriched`
- `--mode skill_guided skill_adjusted dynamic out_of_scope`
- `--limit N` / `--no-wipe`
- Normalises new case format (field mapping `case_id`, `expected_mode`, `difficulty`)

**`rt_planner.py`** — Orchestration pipeline: Intent Resolver → Planner → Retrieval.
1. **Intent Resolver** (`intent_resolver.py`): decides whether a skill matches the request (strict: only commits if skill covers request end-to-end). For SG/SA cases, `hint_skill_id` bypasses the resolver — the correct skill is known from the case definition.
2. **Planner** (`planner.py`): decides coverage (full/partial/none) and builds the step list.
3. **Retrieval**: calls the configured method per step.

**`skill_registry.py`** — Loads SKILL.md files from `benchmark/test_cases/design_time/output/skills/`. Parses frontmatter and `ord_confirmed` annotations. Used by Method D's `list_skills()`/`describe_skill()` tools and the Intent Resolver.

**`intent_resolver.py`** — LLM-based skill matcher. Uses a strict system prompt: only returns a skill_id if the skill covers the request end-to-end. Returns null for ad-hoc / single-resource requests (Dynamic mode).

**`planner.py`** — Runs after the Intent Resolver. In a single LLM call, decides whether the matched skill covers the request (`full` → Skill-Guided) or requires extra steps (`partial` → Skill-Adjusted, returning concrete gap step labels). If the Intent Resolver returned null, the mode becomes Dynamic and the request itself is the sole step. No LLM call is made in that case.

---

## Retrieval Methods A–E + S (Baseline-Solver)

Six retrieval strategies evaluated against the 273-resource ORD benchmark landscape. Each method receives a `label` (activity text) and the flat resource list, and returns `{"candidates": [{ordId, score}], "trace": {...}}`.

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

---

## Method S — Baseline-Solver (No-Retrieval Baseline)

Method S is the weakest possible retrieval strategy and serves two distinct roles depending on the evaluation phase.

**What it does:**
Method S receives the user prompt (or activity label) plus a flat, unstructured list of all 273 resources — each as `ordId: title. shortDescription`. A single LLM call selects the best-matching ordId. No namespace filtering, no entity type traversal, no graph walk, no Skill Registry. The LLM operates on plain text alone.

```
Input:  prompt + ["sap.s4:agent:X: Title. ShortDesc.", "sap.sf:apiResource:Y: ...", ...]
Output: {"ordId": "sap.s4:agent:X:v1"}
```

When `top_k > 1`, a ranked-list prompt is used instead (returns up to top_k candidates in order of relevance). When `top_k = 1` (adversarial gate), retains single-pick behaviour.

**Implementation:** `src/methods/method_s.py`

**Role 1 — Measurement baseline (Design-Time and Run-Time evaluation):**
Method S appears alongside Methods A–E in the results tables. If any method does not beat Method S, it has no retrieval value — it adds no benefit over naive keyword scanning of the resource list. This is the scientific floor: every structured retrieval method must clear it.

**Role 2 — Adversarial difficulty gate (Run-Time case construction only):**
During Run-Time case construction (Dynamic, Skill-Adjusted, and some Skill-Guided cases), Method S is run against each candidate prompt before the Judge sees it. If Method S predicts the correct ordId, the case is rejected as `TOO_EASY` — the generator must produce a harder, more implicit prompt. This proactively ensures that every accepted case in the benchmark requires structured retrieval to solve.

Empirically: 4 of 148 construction iterations were rejected as TOO_EASY (Method S solved them directly). All 4 were replaced with harder prompts in the next iteration. The final 110 accepted cases were all cases that Method S failed on.

**Why not in Design-Time construction:**
Design-Time cases are deterministically extracted from process models — the `expected_ordId` is an explicit assignment, not something to be found. Applying the gate would eliminate clear-activity cases on purpose, distorting the difficulty distribution. The GT-eligible / non-GT split provides the structural difficulty gradient instead.
