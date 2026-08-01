# Evaluation Approach per Mode

This document describes exactly how each run-time mode is evaluated — what runs, what is measured, and where results land.

---

## Skill-Guided (SG)

**Question:** Does the intent resolver find the correct skill from the registry?

**How it runs (`eval_sg_routing.py`):**
1. Load all 30 SG cases from `benchmark/test_cases/runtime/output/skill_guided.json`
2. Load all 30 skills from `benchmark/test_cases/design_time/output/skills/`
3. For each case: call `intent_resolver.resolve(user_prompt, skills)` — no hint, no ORD resources
4. Compare `skill_picked` vs `skill_expected` → `routing_ok = 1/0`
5. Run once — ORD state (clean/enriched) is irrelevant; resolver only sees text

**Primary metric:** `Routing-Acc` = fraction of cases where the correct skill was picked

**What is NOT measured:**
- Retrieval method (A–E + S) — irrelevant, all steps come from `ord_confirmed` once skill matches
- ORD state — resolver uses only prompt + skill descriptions, no landscape
- Step coverage — deterministic once skill is matched

**Process (what happens internally):**
```
1. LLM receives:
     - user_prompt
     - list of 30 skill descriptions (skill_id + description + step names)

2. LLM returns:
     { "best_skill_id": "proc_025", "reason": "proc_025 explicitly orchestrates..." }

3. Scoring:
     routing_ok = (skill_picked == skill_expected)
```
One LLM call, ~1,600 tokens, ~1.5s wall time per case.

**Scalability note:** All 30 skills are loaded into the LLM context in a single call — this works at our benchmark scale (30 skills). At production scale (hundreds of skills) this approach would hit context limits and require vector-based skill retrieval as a pre-filter.

**Example (sg-05, success):**
```
INPUT  prompt:   "We have an engineering change to process. Take it from the initial
                  request through material and compliance validation, workforce checks,
                  and getting the right vendor data locked in."
       registry: 30 skills

OUTPUT skill_expected: proc_025
       skill_picked:   proc_025  ✓
       reason: "proc_025 explicitly orchestrates the complete engineering change process
                from initial request through material validation, workforce readiness
                assessment, and supplier qualification—exactly matching the user's scope."
       tokens: 1,652 | wall_s: 1.4
```

**Example (sg-01, failure):**
```
INPUT  prompt:   "We need to handle a product engineering change end to end — from
                  assessing the production impact all the way through validating our
                  materials are compliant with the new design."

OUTPUT skill_expected: proc_022
       skill_picked:   proc_025  ✗  (both cover eng. change, resolver prefers proc_025)
       reason: "proc_025 explicitly orchestrates the end-to-end engineering change process
                including material validation and supplier qualification..."
```

**Output:**
- `results/runtime/skill_guided/routing/sg-{id}.json` — one trace per case
- `results/runtime/skill_guided/routing/summary.json` — `{cases, Routing-Acc, correct, total_tokens}`

---

## Skill-Adjusted (SA)

**Question:** Given the correct skill (provided via hint), does the orchestrator find the gap resources?

**How it runs (`rt_benchmark.py --mode skill_adjusted`):**
1. Load 20 SA cases from `skill_adjusted.json`
2. For each case × 6 methods × 2 ORD states:
   - Pass `hint_skill_id` to planner — resolver is bypassed, skill is pre-given
   - Planner infers gap steps (activities not covered by the skill)
   - Retrieval method runs on each gap step
   - Score: did the top-1 candidate for each gap step match `expected_gap_ordIds`?

**Primary metric:** `Gap-Detection` = fraction of expected gap ordIds found as top-1 of gap steps

**Process (what happens internally):**
```
Stage 1 — Resolver (bypassed for SA):
   hint_skill_id is injected directly → no LLM call

Stage 2 — Planner LLM call:
   INPUT:  user_prompt + skill description + skill steps
   OUTPUT: { "coverage": "partial",
             "gap_steps": ["Track milestone progression",
                           "Handle customer grievances", ...],
             "reason": "The skill covers vendor assessment... but misses delivery tracking" }

Stage 3 — Retrieval (per gap step, uses chosen method):
   For each gap_step text → method returns top-k candidates
   top-1 candidate ordId is compared to expected_gap_ordIds
```

**Example (sa-01, Method A, clean, success):**
```
INPUT  prompt:      "We need to establish a comprehensive procurement and delivery
                     workflow that coordinates vendor assessment..."
       hint_skill:  proc_033  (resolver skipped)
       expected_gap_ordIds: ["sap.crm:apiResource:CustomerOrderFulfillmentTracking:v1",
                              "sap.crm:apiResource:CustomerAccountDisputeManagement:v1"]

PLANNER gap_steps:
  1. "Track milestone progression and delivery status"
  2. "Handle customer grievances and complaints"
  3. "Reconcile disagreements during fulfillment"
  4. "Monitor delivery status throughout fulfillment cycle"

RETRIEVAL (Method A, embedding):
  gap_step 1 → top-1: sap.crm:apiResource:CustomerOrderFulfillmentTracking:v1  ✓
  gap_step 2 → top-1: sap.crm:apiResource:CustomerAccountDisputeManagement:v1  ✓

gap_detection: 1.0
```

**What is NOT measured:**
- `Routing-Acc` — trivially 1.0 because hint bypasses resolver

**Output:**
- `results/runtime/skill_adjusted/records.jsonl`
- `results/runtime/skill_adjusted/summary.json`

---

## Dynamic (DY)

**Question:** Can the retrieval method find the right resource(s) for an ad-hoc request with no matching skill?

**How it runs (`rt_benchmark.py --mode dynamic`):**
1. Load 40 DY cases (20 single-intent, 20 multi-intent)
2. For each case × 6 methods × 2 ORD states:
   - Resolver runs freely — no hint
   - If resolver finds no skill → retrieval method runs on the raw request
   - Score: top-1 candidate vs `expected_ordIds`

**Primary metric:** `Top-1`

**Process (what happens internally):**
```
Stage 1 — Resolver LLM call:
   INPUT:  user_prompt + 30 skill descriptions
   OUTPUT: { "best_skill_id": null, "reason": "Ad-hoc task, no skill matches" }
   → mode = dynamic

Stage 2 — Planner LLM call:
   INPUT:  user_prompt + skill=null
   OUTPUT: { "coverage": "none", "gap_steps": [] }
   → single adhoc step = full user_prompt

Stage 3 — Retrieval method:
   Method A: embed prompt → cosine vs all 273 resources → top-5
   Method D: LLM agent calls tools iteratively:
     list_entity_types() → list_lines_of_business() →
     resources_by_entity_type("CustomerOrder") →
     resources_by_type("apiResource") →
     pick_resource("sap.s4:apiResource:CustomerOrderManagementAPI:v1",
                   "Directly manages customer account and order modifications")
```

**Example (dy-09, Method D, clean, success):**
```
INPUT  prompt:   "I need to pull up a customer's recent purchases and modify one of
                  their orders before it ships out."
       expected: sap.s4:apiResource:CustomerOrderManagementAPI:v1

METHOD D agent (9 LLM calls, 21,758 tokens):
  call 1: list_entity_types()
  call 2: list_lines_of_business()
  call 3: resources_by_entity_type("CustomerOrder")
  call 4: resources_by_type("apiResource")
  call 5-9: filter + pick_resource(...)

OUTPUT top-1: sap.s4:apiResource:CustomerOrderManagementAPI:v1  ✓
       top1_acc: 1
```

**Example (dy-01, Method A, clean, failure):**
```
INPUT  prompt:   "I need to pull up a recent issue that one of our customers reported
                  last week so I can see what we promised to deliver and when."
       expected: corp.itsm:apiResource:ServiceTicketOrderManagementAPI:v1

METHOD A (0 LLM calls, pure embedding):
  → embed prompt → cosine similarity vs 273 resources
  → top-1: sap.crm:dataProduct:IncidentResolutionMetrics:v1  ✗
    (CRM analytics product instead of ITSM API — vocab mismatch)

OUTPUT top1_acc: 0
```

**Output:**
- `results/runtime/dynamic/records.jsonl`
- `results/runtime/dynamic/summary.json`

---

## Out-of-Scope (OOS)

**Question:** Does the method refuse gracefully when no resource in the landscape matches?

**How it runs (`rt_benchmark.py --mode out_of_scope`):**
1. Load 20 OOS cases
2. For each case × 6 methods × 2 ORD states:
   - Retrieval method runs — expected to return no candidates or call `refuse()`
   - Score: `correctly_refused = 1` if no candidate returned, `falsely_picked = 1` otherwise

**Note:** Methods A/B/C always return top-k by design — they cannot refuse. Only D and E can call `refuse()`. This is part of the finding: algorithmic methods cannot withhold a guess.

**Process (what happens internally):**
```
Stage 1 — Resolver:
   Returns no skill (capability absent from registry)

Stage 2 — Planner:
   mode = dynamic (no skill) → single adhoc step

Stage 3 — Retrieval method:
   Method S: single LLM call → picks closest-sounding resource  (always falsely picks)
   Method D: agent can call refuse("no matching resource") → correctly_refused = 1
   Method A: returns top-1 by cosine regardless              → always falsely picks
```

**Example (oos-01, Method S, failure):**
```
INPUT  prompt:   "We need a way to automatically flag suppliers whose credit ratings
                  or financial health metrics have deteriorated in the past 90 days..."
       (capability absent — no financial health monitoring resource in landscape)

OUTPUT falsely_picked_ord_id: sap.ariba:dataProduct:VendorPerformanceAndSegmentationAnalytics:v1
       correctly_refused: 0   ← picked closest-sounding resource anyway
       falsely_picked:    1
```

**Primary metric:** `Refusal-Rate`

**Output:**
- `results/runtime/out_of_scope/records.jsonl`
- `results/runtime/out_of_scope/summary.json`

---

## Retrieval Methods

Six methods are evaluated. Each receives a query and 273 ORD resources, returns a ranked candidate list.

| Method | Paradigm | LLM calls | Key idea |
|--------|----------|-----------|----------|
| **A** | Embedding cosine | 0 | Embed query + resource text, top-k by cosine. Benefits from enriched text fields. |
| **B** | Progressive Disclosure | 4 | 4-stage funnel: namespace → modality → entity type → final resource. |
| **C** | Multi-Hop Graph Walk | 2 | Anchor inference → seed matching → graph walk (entityType, partOfGroups, processNext) → rerank. |
| **D** | Agentic Tool Use | 1–N | LLM agent with 15 typed tools. Free tool call order. Can inspect skill registry. |
| **E** | Filesystem Agent | 1 (loop) | Agent reads raw `ord.json` files via `list_dir`/`read_file`. No pre-loaded index. |
| **S** | Baseline-Solver | 1 | Single LLM call on flat list of 273 resources. No structure. Weakest baseline. |

**ORD state:**
- Clean-ORD (0): mandatory fields only (title, shortDescription, entityTypes, lineOfBusiness, tags)
- Enriched-ORD (1): adds capabilities, useCases, processNext, partOfGroups for GT resources
- Δ = Enriched − Clean = measured value of ORD enrichment per method

---

## Status

| Mode | Cases | Methods | States | Status |
|------|-------|---------|--------|--------|
| Skill-Guided | 30 | resolver only | n/a | ✓ Done (Routing-Acc=0.700) |
| Skill-Adjusted | 20 | A B C D E S | clean enriched | ⟳ Running |
| Dynamic | 40 | A B C D E S | clean enriched | ⟳ Running |
| Out-of-Scope | 20 | A B C D E S | clean enriched | ✗ Not started |
