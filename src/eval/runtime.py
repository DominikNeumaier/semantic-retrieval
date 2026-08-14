"""Runtime benchmark runner (Sec. 5.4 of the thesis).

Runs the test cases of one expected mode (skill_guided, skill_adjusted,
or dynamic) under every condition of the 5×2 factorial design (retrieval
method × ORD state). The three modes are evaluated in separate runs with
separate output trees, so each mode's metrics and traces can be inspected
in isolation. The default is to run all three modes one after another.

Test cases live in test_cases/runtime/{skill_guided,skill_adjusted,dynamic}.json.

Metrics (per case):
  top1_acc        — first candidate of step 1 matches ground truth.
  acceptable      — first candidate in expected_ordIds OR acceptable_alternatives.
  candidate_recall — at least one expected ID in candidates considered.
  mode_routing_ok — routed mode == expected_mode.
  gap_detection   — for skill_adjusted: fraction of expected_gaps covered.
  step_coverage   — for skill_guided: fraction of expected steps that resolved.
  tokens, latency, llm_calls — operational cost.

Output (under results/runtime/<mode>/):
  records.jsonl            — all per-case records.
  summary.csv / .json      — one row per condition (Tab. VI).
  enrichment.csv           — Δ x1−x0 per method (Tab. VII).
  by_difficulty.csv        — easy/hard split (Tab. X).
  traces/<condition>/<case>.json — full forensic trace per (cond, case).

The routing-confusion and by-mode reports of the legacy layout are not
written per-mode, because they are only meaningful across the three
modes. They can be reconstructed after the runs from the three
records.jsonl files.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from src import config
from src import loader as ord_loader
from src.runtime import rt_planner
from src.runtime import skill_registry


METHODS = ["A", "B", "C", "D", "E", "S", "F"]  # F = Hybrid (A+D+C+B)
STATES = ["clean", "enriched"]
MODES = ["skill_guided", "skill_adjusted", "dynamic", "out_of_scope"]
# Skill-Guided is scored by routing accuracy alone (eval_sg_routing.py), so
# running retrieval methods over it only produces redundant per-method traces.
# It stays a valid --mode choice, but the default run skips it.
DEFAULT_MODES = ["skill_adjusted", "dynamic", "out_of_scope"]


def _load_cases(mode: str) -> list[dict]:
    """Load test cases for one expected mode."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode}")
    raw = json.loads((config.RT_OUTPUT_DIR / f"{mode}.json").read_text())
    return [_normalize_case(c) for c in raw]


def _normalize_case(c: dict) -> dict:
    """Normalize new case format to the field names expected by this runner.

    New format uses: case_id, mode, query_class, user_prompt, expected_skill_id,
                     expected_steps, expected_gap_ordIds, expected_ordIds
    Runner expects: id, expected_mode, expected_gaps, user_prompt, expected_skill_id,
                    expected_steps
    """
    out = dict(c)
    # id
    if "case_id" in c and "id" not in c:
        out["id"] = c["case_id"]
    # expected_mode
    if "expected_mode" not in c:
        out["expected_mode"] = c.get("mode", "dynamic")
    # expected_gaps (SA uses expected_gap_ordIds)
    if "expected_gaps" not in c and "expected_gap_ordIds" in c:
        out["expected_gaps"] = c["expected_gap_ordIds"]
    # expected_steps — SG has it, others may not
    if "expected_steps" not in c:
        # build from expected_ordIds if available (Dynamic/OOS)
        # Multi-intent cases have multiple ordIds — keep them in ONE step
        # so the trace correctly shows all GT and the scorer sees them together.
        oids = c.get("expected_ordIds", [])
        out["expected_steps"] = [{"expected_ordIds": oids, "acceptable_alternatives": []}] if oids else []
    else:
        # ensure each step has expected_ordIds field
        steps = []
        for st in c["expected_steps"]:
            s = dict(st)
            if "expected_ordIds" not in s:
                s["expected_ordIds"] = s.get("expected_ordId", [])
                if isinstance(s["expected_ordIds"], str):
                    s["expected_ordIds"] = [s["expected_ordIds"]]
            if "acceptable_alternatives" not in s:
                s["acceptable_alternatives"] = []
            steps.append(s)
        out["expected_steps"] = steps
    # difficulty — new cases don't have this; derive from query_class
    if "difficulty" not in c:
        qc = c.get("query_class", "")
        out["difficulty"] = "hard" if qc in ("implicit_multi_step", "conditional_multi_step") else "easy"
    return out


def _out_dir(mode: str) -> Path:
    """Per-mode output directory under results/runtime/."""
    return config.RESULTS_RT / mode


def _trace_path_for(mode: str, condition: str, case_id: str) -> Path:
    p = _out_dir(mode) / "traces" / condition / f"{case_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _condition(method: str, state: str) -> str:
    state_tag = "1" if state == "enriched" else "0"
    base = {"A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "S": "S", "F": "F"}[method]
    return f"{base}{state_tag}"


def _score(plan: dict, case: dict) -> dict:
    """Apply the runtime metric set.

    Top-1 / Acceptable / Candidate-Recall are evaluated against the
    *retrieved* step in the plan. For skill_guided and dynamic this is
    plan.steps[0]. For skill_adjusted the first plan step is a *pinned
    skill step* whose candidates come from <!-- ord_confirmed: ... -->,
    while case.expected_steps describe the GAP extensions only — so a
    naive plan.steps[0] vs expected[0] comparison is apples vs oranges.
    For skill_adjusted we therefore score the first *gap* step instead;
    if no gap step exists in the plan we fall back to plan.steps[0].
    """
    expected_mode = case["expected_mode"]
    expected_steps = case.get("expected_steps", [])

    routing_ok = int(plan["mode"] == expected_mode)

    # ─── top-1 / acceptable / candidate recall ───
    expected_ids: set[str] = set()
    acceptable_ids: set[str] = set()
    for st in expected_steps:
        expected_ids.update(st.get("expected_ordIds", []))
        acceptable_ids.update(st.get("acceptable_alternatives", []))

    # Pick which plan step Top-1 / Acceptable are measured against.
    scoring_step = None
    if plan["steps"]:
        if expected_mode == "skill_adjusted":
            gap_steps = [s for s in plan["steps"] if s["source"] == "gap"]
            scoring_step = gap_steps[0] if gap_steps else plan["steps"][0]
        else:
            scoring_step = plan["steps"][0]

    top1_acc = 0
    acceptable = 0
    candidate_recall = 0

    # Dynamic multi-intent: N decomposed steps, each scored against its own GT.
    # top1_acc = mean(per-step top-1 hit); candidate_recall = any GT in any step.
    # Use case query_class as ground truth for scoring (not decomposer n_detected,
    # which may misclassify ambiguous single-intent prompts).
    if (expected_mode == "dynamic"
            and case.get("query_class") == "multi_intent"
            and len(plan["steps"]) >= 2
            and len(expected_steps) == 1):
        # expected_steps has one entry with all GT ordIds; pair each step to one GT
        all_gts = list(expected_steps[0].get("expected_ordIds", []))
        n = min(len(plan["steps"]), len(all_gts))
        per_step_top1 = []
        all_cands = set()
        for i in range(n):
            cands = [c["ordId"] for c in plan["steps"][i]["candidates"]]
            per_step_top1.append(int(bool(cands) and cands[0] == all_gts[i]))
            all_cands.update(cands)
        top1_acc = round(sum(per_step_top1) / n, 4) if n else 0
        acceptable = top1_acc
        candidate_recall = int(bool(set(all_gts) & all_cands))
    elif scoring_step:
        first_cands = [c["ordId"] for c in scoring_step["candidates"]]
        if first_cands:
            top1_acc = int(first_cands[0] in expected_ids)
            acceptable = int(first_cands[0] in (expected_ids | acceptable_ids))
        all_cands = {c["ordId"] for step in plan["steps"]
                                 for c in step["candidates"]}
        candidate_recall = int(bool(expected_ids & all_cands))

    # ─── step coverage (skill_guided) ───
    step_coverage = None
    if expected_mode == "skill_guided" and expected_steps:
        resolved_ids = {c["ordId"] for step in plan["steps"]
                                   for c in step["candidates"][:1]}  # top-1 only
        hit = sum(
            1 for st in expected_steps
            if set(st.get("expected_ordIds", [])) & resolved_ids
        )
        step_coverage = round(hit / len(expected_steps), 4)

    # ─── gap detection (skill_adjusted) ───
    # expected_gap_ordIds: list of ordIds the orchestrator must retrieve as gap steps.
    # A gap is "detected" if its ordId appears as the top-1 candidate of any gap step.
    gap_detection = None
    if expected_mode == "skill_adjusted":
        expected_gap_ids: list[str] = case.get("expected_gap_ordIds", [])
        gap_steps = [s for s in plan["steps"] if s["source"] == "gap"]
        if expected_gap_ids:
            resolved_gap_ids = {
                s["candidates"][0]["ordId"]
                for s in gap_steps if s.get("candidates")
            }
            hits = sum(1 for eid in expected_gap_ids if eid in resolved_gap_ids)
            gap_detection = round(hits / len(expected_gap_ids), 4)
        else:
            gap_detection = 1.0 if not gap_steps else 0.0

    # ─── counterfactual refusal (out_of_scope) ───
    # H5: A method should refuse / return nothing when no resource in the
    # landscape fulfils the request. For out_of_scope cases, expected_ids
    # is empty by construction.
    correctly_refused: int | None = None
    falsely_picked: int | None = None
    falsely_picked_ord_id: str | None = None
    if expected_mode == "out_of_scope":
        # "Refused" = no candidate was committed to across ANY sub-step.
        # For D/E/F this means refuse() was called on every decomposed
        # activity (no candidate emitted). For A/B/C this only happens if
        # every step's result is empty, which by design is rare — they
        # return top-k. That is part of the H5 finding: algorithmic methods
        # cannot refuse. A single sub-step that surfaces a resource counts
        # as a false pick for the whole out-of-scope request.
        picked_steps = [s for s in plan["steps"] if s.get("candidates")]
        any_pick = bool(picked_steps)
        correctly_refused = int(not any_pick)
        falsely_picked = int(any_pick)
        if any_pick:
            first = picked_steps[0]
            falsely_picked_ord_id = (
                first["method_trace"].get("picked_ord_id")
                or first["candidates"][0]["ordId"]
            )
        # Top-1 / Acceptable / Cand-Recall are not meaningful here; zero
        # them out for consistency in the records.
        top1_acc = 0
        acceptable = 0
        candidate_recall = 0

    return {
        "top1_acc": top1_acc,
        "acceptable": acceptable,
        "candidate_recall": candidate_recall,
        "mode_routing_ok": routing_ok,
        "predicted_mode": plan["mode"],
        "step_coverage": step_coverage,
        "gap_detection": gap_detection,
        "correctly_refused": correctly_refused,
        "falsely_picked": falsely_picked,
        "falsely_picked_ord_id": falsely_picked_ord_id,
    }


def _trace_path(mode: str, condition: str, case_id: str) -> Path:
    """Mode-aware trace path under results/runtime/<mode>/traces/."""
    return _trace_path_for(mode, condition, case_id)


def run_case(case: dict, method: str, state: str, mode: str,
             resources: list[dict], skills: list[dict],
             force_mode: str | None = None,
             hint_gap_steps: list[str] | None = None) -> dict:
    t0 = time.time()
    hint_skill = case.get("expected_skill_id") or case.get("skill_id")
    use_hint = case.get("expected_mode") == "skill_adjusted" and hint_skill
    plan = rt_planner.plan_and_resolve(
        case["user_prompt"], resources, skills, method_name=method,
        hint_skill_id=hint_skill if use_hint else None,
        force_mode=force_mode,
        use_graph=(state == "enriched"),
        n_intents_hint=len(case.get("expected_ordIds") or []) or None,
        hint_gap_steps=hint_gap_steps,
    )
    wall = time.time() - t0
    metrics = _score(plan, case)
    cond = _condition(method, state) + ("f" if force_mode else "")

    # Build a step-by-step view that includes the full method trace
    # (Stage-1 anchors / tool calls / agent reasoning) AND the ground
    # truth expected per step, so a single trace file is self-contained
    # for forensic inspection.
    expected_steps = case.get("expected_steps", []) or []
    plan_steps = plan["steps"]
    detailed_steps: list[dict] = []
    for i, s in enumerate(plan_steps):
        # Align expected step at the same index when possible.
        # For skill_adjusted, the expected_steps describe gap steps only,
        # so we attach the i-th expected step to the i-th gap step.
        exp_for_this = None
        if plan["mode"] == "skill_adjusted":
            gap_idx = sum(1 for prior in plan_steps[:i] if prior["source"] == "gap")
            if s["source"] == "gap" and gap_idx < len(expected_steps):
                exp_for_this = expected_steps[gap_idx]
        elif i < len(expected_steps):
            exp_for_this = expected_steps[i]

        detailed_steps.append({
            "step_name": s["step_name"],
            "source": s["source"],
            "candidates": s["candidates"][:5],
            "method_trace": s.get("method_trace", {}),
            "expected": (
                {
                    "step_name": exp_for_this.get("step_name", ""),
                    "expected_ordIds": exp_for_this.get("expected_ordIds", []),
                    "acceptable_alternatives": exp_for_this.get(
                        "acceptable_alternatives", []
                    ),
                }
                if exp_for_this else None
            ),
        })

    rec = {
        "case_id": case['case_id'],
        "user_prompt": case["user_prompt"],
        "mode_expected": case["expected_mode"],
        "difficulty": case["difficulty"],
        "method": method,
        "state": state,
        "condition": cond,
        "skill_expected": case.get("expected_skill_id"),
        "skill_picked": plan["skill_id"],
        "expected_gaps": case.get("expected_gaps", []),
        "metrics": metrics,
        "plan": {
            "mode": plan["mode"],
            "coverage": plan["coverage"],
            "steps": detailed_steps,
        },
        # Full orchestration trace — router decision, planner reasoning, step-level retrieval
        "orchestration_trace": {
            "intent_resolver": {
                "skill_id_picked": plan["skill_id"],
                "skill_id_expected": case.get("expected_skill_id"),
                "routing_correct": plan["mode"] == case.get("expected_mode", ""),
                **plan["trace"].get("intent_trace", {}),
            },
            "planner_trace": plan["trace"].get("planner_trace", {}),
        },
        "trace": {
            "tokens": plan["trace"]["total_tokens"],
            "latency_s": plan["trace"]["total_latency_s"],
            "llm_calls": plan["trace"]["total_llm_calls"],
        },
        "wall_s": round(wall, 3),
    }
    _trace_path(mode, cond, case['case_id']).write_text(json.dumps(rec, indent=2))
    return rec


def _mean(items: list[dict], key: str, allow_none: bool = False) -> float | None:
    vals = [i["metrics"][key] for i in items if i["metrics"][key] is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def aggregate(records: list[dict]) -> list[dict]:
    """One row per condition (Tab. VI)."""
    by_cond: dict[str, list[dict]] = {}
    for r in records:
        by_cond.setdefault(r["condition"], []).append(r)

    summary: list[dict] = []
    for cond, items in by_cond.items():
        n = len(items)
        method = items[0]["method"]
        state = items[0]["state"]
        tokens = sum(i["trace"]["tokens"] for i in items)
        latency = sum(i["trace"]["latency_s"] for i in items)
        llm_calls = sum(i["trace"]["llm_calls"] for i in items)
        summary.append({
            "condition": cond,
            "method": method,
            "state": state,
            "cases": n,
            "Top-1": _mean(items, "top1_acc"),
            "Acceptable": _mean(items, "acceptable"),
            "Cand-Recall": _mean(items, "candidate_recall"),
            "Routing-Acc": _mean(items, "mode_routing_ok"),
            "Step-Coverage": _mean(items, "step_coverage"),
            "Gap-Detection": _mean(items, "gap_detection"),
            "Refusal-Rate": _mean(items, "correctly_refused"),
            "False-Pick-Rate": _mean(items, "falsely_picked"),
            "total_tokens": tokens,
            "total_latency_s": round(latency, 2),
            "total_llm_calls": llm_calls,
            "avg_latency_per_case_s": round(latency / n, 3),
        })
    summary.sort(key=lambda r: r["condition"])
    return summary


def aggregate_enrichment(summary: list[dict]) -> list[dict]:
    """Tab. VII — Δ enriched − clean per method."""
    by_method: dict[str, dict[str, dict]] = {}
    for row in summary:
        by_method.setdefault(row["method"], {})[row["state"]] = row

    out: list[dict] = []
    for method, states in by_method.items():
        c = states.get("clean")
        e = states.get("enriched")
        if not c or not e:
            continue
        out.append({
            "method": method,
            "ΔTop-1": round((e["Top-1"] or 0) - (c["Top-1"] or 0), 4),
            "ΔAcceptable": round((e["Acceptable"] or 0) - (c["Acceptable"] or 0), 4),
            "ΔCand-Recall": round((e["Cand-Recall"] or 0) - (c["Cand-Recall"] or 0), 4),
        })
    out.sort(key=lambda r: r["method"])
    return out


def aggregate_per_case(records: list[dict]) -> list[dict]:
    """One row per case_id with all 8 conditions side-by-side.

    Lets the reader scan: did this individual case work under every method?
    Useful for picking Worked Examples and for inspecting which cases are
    consistently hard.
    """
    cases_seen: dict[str, dict] = {}
    for r in records:
        cid = r["case_id"]
        row = cases_seen.setdefault(cid, {
            "case_id": cid,
            "expected_mode": r["mode_expected"],
            "expected_skill": r["skill_expected"],
            "difficulty": r["difficulty"],
        })
        cond = r["condition"]
        m = r["metrics"]
        row[f"{cond}_routing"] = m["mode_routing_ok"]
        row[f"{cond}_top1"]    = m["top1_acc"]
        row[f"{cond}_accept"]  = m["acceptable"]
        row[f"{cond}_skill"]   = (
            int(r["skill_picked"] == r["skill_expected"])
            if r["skill_expected"]
            else int(r["skill_picked"] is None) if r["mode_expected"] == "dynamic"
            else None
        )
    rows = list(cases_seen.values())
    rows.sort(key=lambda r: r["case_id"])
    return rows


def aggregate_by_difficulty(records: list[dict]) -> list[dict]:
    """Tab. X — Top-1 per condition, split by difficulty."""
    rows: list[dict] = []
    by_cond: dict[str, list[dict]] = {}
    for r in records:
        by_cond.setdefault(r["condition"], []).append(r)
    for cond in sorted(by_cond):
        items = by_cond[cond]
        easy = [i for i in items if i["difficulty"] == "easy"]
        hard = [i for i in items if i["difficulty"] == "hard"]
        rows.append({
            "condition": cond,
            "Top-1_easy": _mean(easy, "top1_acc") if easy else None,
            "Top-1_hard": _mean(hard, "top1_acc") if hard else None,
            "n_easy": len(easy),
            "n_hard": len(hard),
        })
    return rows


def _write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_outputs(records: list[dict], mode: str) -> None:
    """Write all per-mode outputs under results/runtime/<mode>/."""
    out = _out_dir(mode)
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records)
    )

    summary = aggregate(records)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if summary:
        _write_csv(out / "summary.csv", summary, list(summary[0].keys()))

    enrichment = aggregate_enrichment(summary)
    if enrichment:
        _write_csv(out / "enrichment.csv", enrichment,
                   ["method", "ΔTop-1", "ΔAcceptable", "ΔCand-Recall"])

    by_diff = aggregate_by_difficulty(records)
    if by_diff:
        _write_csv(out / "by_difficulty.csv", by_diff,
                   ["condition", "Top-1_easy", "Top-1_hard", "n_easy", "n_hard"])

    # Per-case wide view (one row per case, all conditions side-by-side).
    per_case = aggregate_per_case(records)
    if per_case:
        col_set: dict[str, None] = {}
        for r in per_case:
            for k in r.keys():
                col_set.setdefault(k, None)
        cols = list(col_set.keys())
        _write_csv(out / "by_case.csv", per_case, cols)


def run_mode(mode: str, methods: list[str], states: list[str],
             skills: list[dict], limit: int | None,
             case_ids: set[str] | None, no_wipe: bool = False,
             force_mode: str | None = None) -> None:
    """Run the full benchmark for one expected mode and write its outputs."""
    cases = _load_cases(mode)
    if case_ids:
        cases = [c for c in cases if c["id"] in case_ids]
    if limit:
        cases = cases[:limit]
    if not cases:
        print(f"\n[{mode}] no cases — skipping")
        return

    print(f"\n=== mode={mode}  cases={len(cases)}  methods={methods}  states={states} ===")

    # Wipe ONLY this mode's output directory. Other modes and the
    # design-time results are left untouched — each mode owns its own
    # subtree under results/runtime/<mode>/.
    import shutil
    out = _out_dir(mode)
    if out.exists() and not no_wipe:
        shutil.rmtree(out)
        print(f"  cleared {out.relative_to(config.ROOT)}/")
    out.mkdir(parents=True, exist_ok=True)

    if "E" in methods:
        from src.methods import method_raw
        method_raw.reset_notes()
        print("  reset E's persistent notes")

    records: list[dict] = []
    out = _out_dir(mode)
    records_path = out / "records.jsonl"

    # Load already-completed records when resuming (--no-wipe)
    done: set[tuple] = set()
    if no_wipe and records_path.exists():
        for line in records_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                records.append(r)
                # key includes force suffix: forced records have condition ending in 'f'
                is_forced = r.get("condition", "").endswith("f")
                done.add((r["case_id"], r["method"], r["state"], "dynamic" if is_forced else ""))
            except Exception:
                pass
        print(f"  resumed: {len(records)} existing records, {len(done)} combos done")
    elif not no_wipe:
        records_path.write_text("")

    # Build gap-steps cache from A0 records for fair cross-method comparison.
    # Only used when running F on skill_adjusted: inject the exact same gap step
    # labels that the planner produced for A0, bypassing a fresh LLM call.
    gap_steps_cache: dict[str, list[str]] = {}
    if "F" in methods and mode == "skill_adjusted" and not force_mode:
        if records_path.exists():
            for line in records_path.read_text().splitlines():
                try:
                    r = json.loads(line)
                    if r.get("condition") == "A0":
                        steps = [s["step_name"] for s in r.get("plan", {}).get("steps", [])
                                 if s.get("source") == "gap"]
                        if steps:
                            gap_steps_cache[r["case_id"]] = steps
                except Exception:
                    pass
        if gap_steps_cache:
            print(f"  gap_steps_cache: {len(gap_steps_cache)} cases loaded from A0")
        else:
            print("  WARNING: gap_steps_cache empty — A0 records not found, planner will run normally")

    for state in states:
        resources = ord_loader.load_landscape(state=state)
        print(f"  state={state}  resources={len(resources)}")
        for method in methods:
            for case in cases:
                done_key = (case["case_id"], method, state, force_mode or "")
                if done_key in done:
                    print(f"    [{_condition(method, state)}{'f' if force_mode else ''}] {case['case_id']} skip")
                    continue
                cond_label = _condition(method, state) + ("f" if force_mode else "")
                print(f"    [{cond_label}] {case['case_id']}")
                hint_gaps = (gap_steps_cache.get(case["case_id"])
                             if method == "F" and mode == "skill_adjusted" and not force_mode
                             else None)
                try:
                    rec = run_case(case, method, state, mode, resources, skills,
                                   force_mode=force_mode, hint_gap_steps=hint_gaps)
                    records.append(rec)
                    with records_path.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                except Exception as e:
                    print(f"      ERROR: {type(e).__name__}: {e}")

    write_outputs(records, mode)

    summary = aggregate(records)
    print(f"\n=== Summary [{mode}] ===")
    if mode == "out_of_scope":
        # H5 metrics: refusal rate & false-pick rate replace Top-1 / Step-Cov
        print(f"{'cond':<6} {'Refuse':>7} {'FalsePk':>8} {'Route':>6} {'tok':>9} {'lat_s':>8}")
        for s in summary:
            rr = "  --  " if s["Refusal-Rate"] is None else f"{s['Refusal-Rate']:>7.3f}"
            fp = "  --   " if s["False-Pick-Rate"] is None else f"{s['False-Pick-Rate']:>8.3f}"
            print(f"{s['condition']:<6} {rr} {fp} {s['Routing-Acc']:>6.3f} "
                  f"{s['total_tokens']:>9d} {s['total_latency_s']:>8.2f}")
    else:
        print(f"{'cond':<6} {'Top-1':>6} {'Accept':>7} {'CandR':>6} {'Route':>6} {'StpCov':>7} {'GapDet':>7} {'tok':>9} {'lat_s':>8}")
        for s in summary:
            sc = "  --  " if s["Step-Coverage"] is None else f"{s['Step-Coverage']:>7.3f}"
            gd = "  --  " if s["Gap-Detection"] is None else f"{s['Gap-Detection']:>7.3f}"
            t1 = "  -- " if s["Top-1"] is None else f"{s['Top-1']:>6.3f}"
            ac = "   -- " if s["Acceptable"] is None else f"{s['Acceptable']:>7.3f}"
            cr = "  -- " if s["Cand-Recall"] is None else f"{s['Cand-Recall']:>6.3f}"
            print(f"{s['condition']:<6} {t1} {ac} {cr} "
                  f"{s['Routing-Acc']:>6.3f} {sc} {gd} {s['total_tokens']:>9d} {s['total_latency_s']:>8.2f}")
    print(f"  → {_out_dir(mode)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--states", nargs="+", default=STATES, choices=STATES)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mode", nargs="+", default=DEFAULT_MODES, choices=MODES,
                    help="which expected mode(s) to run; defaults to SA/dynamic/OOS "
                         "(skill_guided is scored separately by eval_sg_routing.py)")
    ap.add_argument("--case-ids", nargs="+", default=None,
                    help="run only the cases with these ids (filtered after mode)")
    ap.add_argument("--no-wipe", action="store_true",
                    help="skip clearing the output directory (safe for partial reruns)")
    ap.add_argument("--force-mode", default=None, choices=["dynamic"],
                    help="bypass router/planner and force this mode (e.g. dynamic for fair retrieval eval)")
    args = ap.parse_args()

    skills = skill_registry.load_skills()
    case_ids = set(args.case_ids) if args.case_ids else None
    print(f"Skills: {len(skills)}")

    for mode in args.mode:
        run_mode(mode, args.methods, args.states, skills, args.limit, case_ids,
                 no_wipe=args.no_wipe, force_mode=args.force_mode)


if __name__ == "__main__":
    main()
