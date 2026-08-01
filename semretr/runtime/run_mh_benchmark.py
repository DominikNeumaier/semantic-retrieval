"""Multi-Hint (mh) run for Dynamic multi-intent cases.

Runs methods A-S on pre-decomposed sub-queries (one retrieval call per
sub-query) so that retrieval quality is measured independently of router
and decomposer decisions. Mirrors how Method F handles multi-intent.

Each multi-intent case (dy-21..dy-40) produces N records (one per sub-query),
stored in records with condition A0mh / A1mh / ... / S1mh.
Scoring: top1_acc per sub-query vs its own GT ordId.
Summary: mean top1_acc over all sub-query records per condition.

Output:
  results/runtime/dynamic/records_mh.jsonl
  results/runtime/dynamic/summary_mh.json
  results/runtime/dynamic/traces/A0mh/dy-21_sub1.json ...
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.core import config, ord_loader
from src.runtime import rt_planner, skill_registry

SUBQUERIES_PATH = ROOT / "benchmark" / "test_cases" / "runtime" / "output" / "dynamic_mh_subqueries.json"
OUT_RECORDS = ROOT / "results" / "runtime" / "dynamic" / "records_mh.jsonl"
OUT_SUMMARY = ROOT / "results" / "runtime" / "dynamic" / "summary_mh.json"
TRACES_BASE = ROOT / "results" / "runtime" / "dynamic" / "traces"

METHODS = ["A", "B", "C", "D", "E", "S", "F"]
STATES = [("0", "clean"), ("1", "enriched")]


def run():
    subqueries = json.loads(SUBQUERIES_PATH.read_text())
    skills = skill_registry.load_skills()

    records = []
    done = set()

    # Resume: load existing records
    if OUT_RECORDS.exists():
        for line in OUT_RECORDS.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["case_id"], r["sub_idx"], r["condition"]))
                records.append(r)
        print(f"Resumed: {len(records)} existing records")

    for state_num, state in STATES:
        resources = ord_loader.load_landscape(state)
        print(f"\nstate={state}  resources={len(resources)}")

        for method in METHODS:
            cond = method + state_num + "mh"
            for case_id, case_data in subqueries.items():
                expected_ids = case_data["expected_ordIds"]
                sub_queries = case_data["sub_queries"]
                n = len(sub_queries)

                for i, sq in enumerate(sub_queries):
                    gt_id = expected_ids[i] if i < len(expected_ids) else None
                    key = (case_id, i, cond)
                    if key in done:
                        print(f"  [{cond}] {case_id}_sub{i+1} skip")
                        continue

                    print(f"  [{cond}] {case_id}_sub{i+1}")
                    t0 = time.time()
                    try:
                        # Call the retrieval method DIRECTLY — bypass rt_planner and decomposer.
                        # Sub-queries are pre-decomposed; each is atomic. No further splitting.
                        from src.runtime.rt_planner import METHODS as METHOD_MAP
                        retrieve_fn = METHOD_MAP[method]
                        extra = {"skills": skills, "use_graph": (state == "enriched"),
                                 "n_intents_hint": 1}
                        try:
                            result = retrieve_fn(sq, resources, top_k=5, **extra)
                        except TypeError:
                            try:
                                result = retrieve_fn(sq, resources, top_k=5,
                                                     skills=skills,
                                                     use_graph=(state == "enriched"))
                            except TypeError:
                                try:
                                    result = retrieve_fn(sq, resources, top_k=5,
                                                         previous_resolved_ord_ids=None)
                                except TypeError:
                                    result = retrieve_fn(sq, resources, top_k=5)
                        candidates = [c["ordId"] for c in result.get("candidates", [])[:5]]
                        method_trace = result.get("trace", {})
                        total_tokens = method_trace.get("tokens", 0)
                        plan = {
                            "request": sq, "mode": "dynamic", "skill_id": None,
                            "coverage": "none", "method": method,
                            "steps": [{
                                "step_name": sq, "source": "adhoc",
                                "candidates": [{"ordId": oid} for oid in candidates],
                                "method_trace": method_trace,
                            }],
                            "trace": {"total_tokens": total_tokens, "total_latency_s": 0,
                                      "total_llm_calls": method_trace.get("llm_calls", 0)},
                        }
                    except Exception as e:
                        print(f"    ERROR: {e}")
                        import traceback; traceback.print_exc()
                        continue
                    wall = time.time() - t0

                    # Detect refusal
                    refused = (not candidates and
                               (method_trace.get("refused") or
                                method_trace.get("via") in ("refused", "empty")))

                    top1 = int(bool(candidates) and candidates[0] == gt_id) if gt_id else 0
                    top5 = int(gt_id in candidates) if gt_id else 0

                    rec = {
                        "case_id": case_id,
                        "sub_idx": i,
                        "sub_query": sq,
                        "gt_id": gt_id,
                        "method": method,
                        "state": state,
                        "condition": cond,
                        "top1_acc": top1,
                        "candidate_recall": top5,
                        "top5_candidates": candidates,
                        "refused": refused,
                        "tokens": total_tokens,
                        "wall_s": round(wall, 3),
                    }
                    records.append(rec)
                    done.add(key)

                    # Save trace — include full plan with method_trace
                    trace_dir = TRACES_BASE / cond
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    trace_file = trace_dir / f"{case_id}_sub{i+1}.json"
                    trace_file.write_text(json.dumps({
                        **rec,
                        "plan": plan,
                    }, indent=2))

                    # Append to records file
                    with OUT_RECORDS.open("a") as f:
                        f.write(json.dumps(rec) + "\n")

    # Build summary: mean top1 per condition
    from collections import defaultdict
    by_cond: dict[str, list] = defaultdict(list)
    for r in records:
        by_cond[r["condition"]].append(r)

    summary_rows = []
    print("\n=== Multi-Hint Summary ===")
    print(f"{'cond':<8} {'Top-1':>6} {'Top-5':>6} {'n':>4}")
    for cond, recs in sorted(by_cond.items()):
        n = len(recs)
        top1 = sum(r["top1_acc"] for r in recs) / n
        top5 = sum(r["candidate_recall"] for r in recs) / n
        summary_rows.append({"condition": cond, "n": n,
                              "top1_acc": round(top1, 4),
                              "candidate_recall": round(top5, 4)})
        print(f"{cond:<8} {top1:>6.3f} {top5:>6.3f} {n:>4}")

    OUT_SUMMARY.write_text(json.dumps({
        "_note": "Multi-hinted run: A-S each called once per sub-query with GT-count hint. "
                 "Measures pure retrieval quality independent of decomposition.",
        "conditions": summary_rows
    }, indent=2))
    print(f"\nWrote {OUT_SUMMARY}")


if __name__ == "__main__":
    run()
