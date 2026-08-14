"""Design-Time benchmark runner.

For every (case, method):
  - call method.retrieve(label, resources)
  - score against ground truth (case["expected_ordIds"])
  - write per-case trace to results/design-time/traces/<method>/<case>.json

Aggregate per method into results/design-time/summary.json and a CSV row
table for the thesis (results/design-time/summary.csv).

Metrics:
  P@1      — Top-1 candidate is in expected_ordIds.
  R@5      — at least one expected_ordIds is among top-5.
  MRR      — 1/rank of the FIRST hit in the candidate list (0 if none).
  tokens   — sum of LLM + embedding tokens per case.
  latency  — sum of cold-call seconds per case.
  llm_calls — total LLM chat calls per case (0 for A; 1-2 for C; 3-4 for B).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from src import config
from src.eval import dt_cases
from src import loader as ord_loader
from src.methods import method_embedding, method_progressive, method_graph, method_tools, method_raw, method_baseline, method_hybrid

# GT-eligible ordIds for P@1_gt / R@5_gt split
def _load_gt_eligible() -> set[str]:
    try:
        import json
        p = config.ROOT / "benchmark" / "ambiguity" / "landscape_ambiguity_report.json"
        data = json.loads(p.read_text())
        return {r["ordId"] for r in data.get("resources", []) if r.get("ground_truth_eligible")}
    except Exception:
        return set()

_GT_ELIGIBLE: set[str] = _load_gt_eligible()


METHODS = {
    "A": method_embedding.retrieve,
    "B": lambda label, resources, **kw: method_progressive.retrieve(label, resources, allow_refuse=False, **kw),
    "C": lambda label, resources, **kw: method_graph.retrieve(label, resources, allow_refuse=False, **kw),
    "D": lambda label, resources, **kw: method_tools.retrieve(label, resources, allow_refuse=False, **kw),
    "E": lambda label, resources, **kw: method_raw.retrieve(label, resources, allow_refuse=False, **kw),
    "S": method_baseline.retrieve,
    "F": lambda label, resources, **kw: method_hybrid.retrieve(label, resources, allow_refuse=False, **kw),
}


def _namespace(ord_id: str) -> str:
    """Extract the namespace prefix from an ORD ID (everything before the first ':')."""
    return ord_id.split(":", 1)[0] if ":" in ord_id else ord_id


def _score(candidates: list[dict], expected: list[str], trace: dict | None = None) -> dict:
    """Compute per-case metrics.

    Two regimes:
      - In-scope:  expected_ordIds non-empty → P@1, R@5, MRR (plus B's
                   stage accuracies). correctly_refused / falsely_picked
                   are None.
      - Out-of-scope: expected_ordIds empty → correctly_refused (1 if no
                   candidate was returned, else 0) and falsely_picked
                   (the complement). P@1 / R@5 / MRR are reported as None
                   so they do not bias the in-scope means.
    """
    expected_set = set(expected)
    expected_namespaces = {_namespace(eid) for eid in expected}

    if not expected_set:
        # Out-of-scope case: ground truth = "no resource fits".
        # Top-1/R@5/MRR are undefined; only refusal metrics make sense.
        refused = not bool(candidates)
        out = {
            "p_at_1": None,
            "r_at_5": None,
            "mrr": None,
            "first_hit_rank": None,
            "correctly_refused": int(refused),
            "falsely_picked": int(not refused),
        }
    elif not candidates:
        out = {
            "p_at_1": 0, "r_at_5": 0, "mrr": 0.0, "first_hit_rank": None,
            "correctly_refused": None, "falsely_picked": None,
        }
    else:
        ranks = [i + 1 for i, c in enumerate(candidates) if c["ordId"] in expected_set]
        first_hit = ranks[0] if ranks else None
        out = {
            "p_at_1": int(candidates[0]["ordId"] in expected_set),
            "r_at_5": int(any(c["ordId"] in expected_set for c in candidates[:5])),
            "mrr": round((1.0 / first_hit) if first_hit else 0.0, 4),
            "first_hit_rank": first_hit,
            "correctly_refused": None,
            "falsely_picked": None,
        }

    # Method-B-specific: stage1 namespace correctness, stage2 conditional on stage1
    # B.1 (multi-namespace) returns a list under "stage1_namespaces"; the old
    # single-namespace shape used "stage1_namespace". Accept both.
    if trace and ("stage1_namespaces" in trace or "stage1_namespace" in trace):
        chosen_ns_list: list[str] = []
        if "stage1_namespaces" in trace:
            chosen_ns_list = list(trace.get("stage1_namespaces") or [])
        elif "stage1_namespace" in trace:
            single = trace.get("stage1_namespace")
            chosen_ns_list = [single] if single else []
        # Stage 1 is "correct" if ANY of the chosen namespaces is in the
        # expected namespace set — i.e. the union covered at least one
        # expected target.
        out["b_ns_correct"] = int(
            any(ns in expected_namespaces for ns in chosen_ns_list)
        ) if chosen_ns_list else 0
        # stage2 accuracy = p_at_1 given that stage1 was correct; otherwise None
        out["b_tool_given_ns"] = out["p_at_1"] if out["b_ns_correct"] else None
    return out


def _trace_path(method: str, case_id: str) -> Path:
    safe_method = method.replace("->", "_to_")
    safe_case = case_id.replace("/", "__")
    p = config.RESULTS_DT / "traces" / safe_method / f"{safe_case}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def run_one(method_name: str, case: dict, resources: list[dict]) -> dict:
    retrieve = METHODS[method_name]
    t0 = time.time()
    result = retrieve(case["label"], resources)
    wall = time.time() - t0
    metrics = _score(result["candidates"], case["expected_ordIds"], result.get("trace"))
    record = {
        "case_id": case["case_id"],
        "process": case["process"],
        "label": case["label"],
        "method": method_name,
        "expected_ordIds": case["expected_ordIds"],
        "candidates": result["candidates"],
        "metrics": metrics,
        "trace": result["trace"],
        "wall_s": round(wall, 3),
    }
    _trace_path(method_name, case["case_id"]).write_text(json.dumps(record, indent=2))
    return record


def _mean_metric(items: list[dict], key: str) -> float:
    """Mean of `metrics[key]` across items, ignoring None entries.

    Returns 0.0 if no item has a defined value for this key (so the bench
    can keep emitting a numeric column when out-of-scope cases dominate
    a per-process bucket).
    """
    vals = [i["metrics"][key] for i in items if i["metrics"].get(key) is not None]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def aggregate(records: list[dict]) -> list[dict]:
    """Group by method, compute means + Method-B stage accuracies.

    Headline P@1/R@5/MRR are computed over in-scope items only (i.e.
    items whose ground truth has expected_ordIds). Refusal metrics
    (correctly_refused, falsely_picked) are computed over out-of-scope
    items only. Both regimes contribute to the same per-method row.
    """
    by_method: dict[str, list[dict]] = {}
    for r in records:
        by_method.setdefault(r["method"], []).append(r)

    summary: list[dict] = []
    for method, items in by_method.items():
        n = len(items)
        in_scope = [i for i in items if i["expected_ordIds"]]
        out_scope = [i for i in items if not i["expected_ordIds"]]
        tokens = sum(i["trace"].get("tokens", 0) for i in items)
        # Method A reports the query-embedding cost per case in `tokens` and the
        # one-off landscape embedding cost in `landscape_setup_tokens`. Amortise
        # the setup cost across the run so avg_tokens_per_case reflects what a
        # production deployment would pay.
        landscape_setup_once = max(
            (i["trace"].get("landscape_setup_tokens", 0) or 0 for i in items),
            default=0,
        )
        latency = sum(i["trace"].get("latency_s", 0) for i in items)
        llm_calls = sum(i["trace"].get("llm_calls", 0) for i in items)

        row = {
            "method": method,
            "cases": n,
            "in_scope": len(in_scope),
            "oos": len(out_scope),
            "P@1": round(_mean_metric(in_scope, "p_at_1"), 4) if in_scope else None,
            "R@5": round(_mean_metric(in_scope, "r_at_5"), 4) if in_scope else None,
            "MRR": round(_mean_metric(in_scope, "mrr"), 4) if in_scope else None,
            "Refusal-Rate": round(_mean_metric(out_scope, "correctly_refused"), 4) if out_scope else None,
            "False-Pick-Rate": round(_mean_metric(out_scope, "falsely_picked"), 4) if out_scope else None,
            "B_ns_acc": "",
            "B_tool_given_ns": "",
            "total_tokens": tokens,
            "landscape_setup_tokens": landscape_setup_once,
            "total_latency_s": round(latency, 2),
            "total_llm_calls": llm_calls,
            "avg_tokens_per_case": round((tokens + landscape_setup_once) / n, 1),
            "avg_latency_per_case_s": round(latency / n, 3),
        }
        # GT-eligible vs non-GT split
        gt_items  = [i for i in in_scope if any(o in _GT_ELIGIBLE for o in i.get("expected_ordIds", []))]
        ngt_items = [i for i in in_scope if not any(o in _GT_ELIGIBLE for o in i.get("expected_ordIds", []))]
        row["P@1_gt"]  = round(_mean_metric(gt_items,  "p_at_1"), 4) if gt_items  else None
        row["P@1_ngt"] = round(_mean_metric(ngt_items, "p_at_1"), 4) if ngt_items else None
        row["R@5_gt"]  = round(_mean_metric(gt_items,  "r_at_5"), 4) if gt_items  else None
        row["R@5_ngt"] = round(_mean_metric(ngt_items, "r_at_5"), 4) if ngt_items else None
        # Method B stage accuracies (in-scope only — namespace expectations
        # are not defined for OOS cases).
        if method == "B":
            ns_correct = [i["metrics"].get("b_ns_correct", 0) for i in in_scope]
            row["B_ns_acc"] = round(sum(ns_correct) / len(in_scope), 4) if in_scope else 0.0
            tool_hits = [i["metrics"]["b_tool_given_ns"] for i in in_scope
                         if i["metrics"].get("b_tool_given_ns") is not None]
            row["B_tool_given_ns"] = round(sum(tool_hits) / len(tool_hits), 4) if tool_hits else 0.0
        summary.append(row)
    summary.sort(key=lambda x: x["method"])
    return summary


def aggregate_by_process(records: list[dict]) -> list[dict]:
    """One row per (process, method).

    Ranking metrics (P@1/R@5/MRR) are emitted when at least one in-scope
    case exists; refusal metrics are emitted when at least one OOS case
    exists. For an in-scope process all OOS columns are blank; for an OOS
    process all ranking columns are blank.
    """
    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        grouped.setdefault((r["process"], r["method"]), []).append(r)

    rows: list[dict] = []
    for (process, method), items in grouped.items():
        in_scope = [i for i in items if i["expected_ordIds"]]
        out_scope = [i for i in items if not i["expected_ordIds"]]
        rows.append({
            "process": process,
            "method": method,
            "cases": len(items),
            "P@1": round(_mean_metric(in_scope, "p_at_1"), 4) if in_scope else "",
            "R@5": round(_mean_metric(in_scope, "r_at_5"), 4) if in_scope else "",
            "MRR": round(_mean_metric(in_scope, "mrr"), 4) if in_scope else "",
            "Refusal-Rate": round(_mean_metric(out_scope, "correctly_refused"), 4) if out_scope else "",
            "False-Pick-Rate": round(_mean_metric(out_scope, "falsely_picked"), 4) if out_scope else "",
        })
    rows.sort(key=lambda r: (r["process"], r["method"]))
    return rows


def _notation(process_id: str) -> str:
    """Derive process modeling notation from the XML content of the process file.

    process_id is e.g. "proc_001" — we look up the XML in DT_OUTPUT_DIR/processes/.
    CMMN: file has type="cmmn" attribute, <CasePlanModel>, or cmmn: namespace prefix.
    Everything else is treated as BPMN.
    """
    xml_path = config.DT_OUTPUT_DIR / "processes" / f"{process_id}.xml"
    if not xml_path.exists():
        return "unknown"
    try:
        content = xml_path.read_text(encoding="utf-8", errors="ignore")
        if ('type="cmmn"' in content or "CasePlanModel" in content
                or "cmmn:" in content or "<case " in content):
            return "CMMN"
        return "BPMN"
    except Exception:
        return "unknown"


def aggregate_by_notation(records: list[dict]) -> list[dict]:
    """One row per (notation, method): P@1, R@5, MRR.

    Splits the DT cases by modeling notation (BPMN vs. CMMN) to expose
    how each retrieval method behaves on structured vs. knowledge-
    intensive process descriptions. This is the direct empirical answer
    to the 'different types of process' aspect of the research question
    in Sec. 1.2 — the same five methods are evaluated on both ends of
    the structuredness spectrum.
    """
    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        notation = _notation(r["process"])
        if notation == "unknown":
            continue
        grouped.setdefault((notation, r["method"]), []).append(r)

    rows: list[dict] = []
    for (notation, method), items in sorted(grouped.items()):
        in_scope = [i for i in items if i["expected_ordIds"]]
        out_scope = [i for i in items if not i["expected_ordIds"]]
        rows.append({
            "notation": notation,
            "method": method,
            "cases": len(items),
            "in_scope": len(in_scope),
            "oos": len(out_scope),
            "P@1": round(_mean_metric(in_scope, "p_at_1"), 4) if in_scope else "",
            "R@5": round(_mean_metric(in_scope, "r_at_5"), 4) if in_scope else "",
            "MRR": round(_mean_metric(in_scope, "mrr"), 4) if in_scope else "",
            "Refusal-Rate": round(_mean_metric(out_scope, "correctly_refused"), 4) if out_scope else "",
            "False-Pick-Rate": round(_mean_metric(out_scope, "falsely_picked"), 4) if out_scope else "",
        })
    rows.sort(key=lambda r: (r["notation"], r["method"]))
    return rows


def write_summary(summary: list[dict],
                  by_process: list[dict],
                  by_notation: list[dict]) -> None:
    config.RESULTS_DT.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DT / "summary.json").write_text(json.dumps(summary, indent=2))

    cols = ["method", "cases", "in_scope", "oos",
            "P@1", "P@1_gt", "P@1_ngt", "R@5", "R@5_gt", "R@5_ngt", "MRR",
            "Refusal-Rate", "False-Pick-Rate",
            "B_ns_acc", "B_tool_given_ns",
            "total_tokens", "total_latency_s", "total_llm_calls",
            "avg_tokens_per_case", "avg_latency_per_case_s"]
    with (config.RESULTS_DT / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in summary:
            w.writerow(row)

    # per-process breakdown
    proc_cols = ["process", "method", "cases",
                 "P@1", "R@5", "MRR",
                 "Refusal-Rate", "False-Pick-Rate"]
    with (config.RESULTS_DT / "by_process.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=proc_cols)
        w.writeheader()
        for row in by_process:
            w.writerow(row)
    (config.RESULTS_DT / "by_process.json").write_text(json.dumps(by_process, indent=2))

    # per-notation breakdown (BPMN vs CMMN — addresses RQ "different types
    # of process"; see aggregate_by_notation docstring)
    notation_cols = ["notation", "method", "cases", "in_scope", "oos",
                     "P@1", "R@5", "MRR",
                     "Refusal-Rate", "False-Pick-Rate"]
    with (config.RESULTS_DT / "by_modeling_notation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=notation_cols)
        w.writeheader()
        for row in by_notation:
            w.writerow(row)
    (config.RESULTS_DT / "by_modeling_notation.json").write_text(
        json.dumps(by_notation, indent=2)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(METHODS.keys()),
                    choices=list(METHODS.keys()))
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N cases (smoke test)")
    ap.add_argument("--state", default="clean", choices=["clean"],
                    help="DT evaluation always uses Clean-ORD (enrichment is this flow's output)")
    ap.add_argument("--processes", nargs="+", default=None,
                    help="filter cases to these BPMN/CMMN file names "
                         "(with extension, e.g. machine_breakdown.bpmn)")
    ap.add_argument("--no-wipe", action="store_true",
                    help="skip clearing results dir — safe for partial reruns")
    args = ap.parse_args()

    resources = ord_loader.load_landscape(state=args.state)
    cases = dt_cases.build_cases()
    if args.processes:
        wanted = set(args.processes)
        cases = [c for c in cases if c["process"] in wanted]
    if args.limit:
        cases = cases[:args.limit]

    print(f"Resources: {len(resources)}  Cases: {len(cases)}  Methods: {args.methods}")
    print(f"ORD state: {args.state or config.ORD_STATE}")

    # Wipe ONLY this benchmark's output directory so a re-run produces an
    # authoritative state. results/runtime/ is left untouched — the two
    # benches own disjoint subtrees of results/.
    import shutil
    if config.RESULTS_DT.exists() and not args.no_wipe:
        shutil.rmtree(config.RESULTS_DT)
        print(f"Cleared {config.RESULTS_DT.relative_to(config.ROOT)}/")
    config.RESULTS_DT.mkdir(parents=True, exist_ok=True)

    # E uses persistent notes that survive across calls — wipe them at
    # the start of every benchmark run so the order in which cases are
    # processed produces a reproducible result.
    if "E" in args.methods:
        method_raw.reset_notes()
        print("Reset E's persistent notes.")

    records: list[dict] = []
    for case in cases:
        for method in args.methods:
            print(f"  [{method}] {case['case_id']}")
            try:
                rec = run_one(method, case, resources)
                records.append(rec)
            except Exception as e:
                print(f"    ERROR: {type(e).__name__}: {e}")

    # also dump the flat record list for forensics
    (config.RESULTS_DT / "records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records)
    )

    summary = aggregate(records)
    by_process = aggregate_by_process(records)
    by_notation = aggregate_by_notation(records)
    write_summary(summary, by_process, by_notation)

    def _fmt(v, w=6):
        if v is None or v == "":
            return f"{'  --  ':>{w}}"
        return f"{v:>{w}.3f}"

    print("\n=== Summary (in-scope ranking + OOS refusal) ===")
    print(f"{'method':<8} {'P@1':>6} {'R@5':>6} {'MRR':>6} {'Refuse':>7} {'FalsePk':>8} {'B_ns':>6} {'B_t|ns':>7} {'tokens':>9} {'lat_s':>8} {'calls':>6}")
    for s in summary:
        ns = f"{s['B_ns_acc']:.3f}" if s['B_ns_acc'] != "" else "  -- "
        tn = f"{s['B_tool_given_ns']:.3f}" if s['B_tool_given_ns'] != "" else "  -- "
        print(f"{s['method']:<8} {_fmt(s['P@1'])} {_fmt(s['R@5'])} {_fmt(s['MRR'])} "
              f"{_fmt(s['Refusal-Rate'], 7)} {_fmt(s['False-Pick-Rate'], 8)} "
              f"{ns:>6} {tn:>7} {s['total_tokens']:>9d} {s['total_latency_s']:>8.2f} {s['total_llm_calls']:>6d}")

    print("\n=== BPMN vs CMMN (split into in-scope + OOS) ===")
    print(f"{'notation':<10} {'method':<8} {'n':>4} {'isc':>4} {'oos':>4} {'P@1':>6} {'R@5':>6} {'MRR':>6} {'Refuse':>7} {'FalsePk':>8}")
    for r in by_notation:
        print(f"{r['notation']:<10} {r['method']:<8} {r['cases']:>4d} {r['in_scope']:>4d} {r['oos']:>4d} "
              f"{_fmt(r['P@1'])} {_fmt(r['R@5'])} {_fmt(r['MRR'])} "
              f"{_fmt(r['Refusal-Rate'], 7)} {_fmt(r['False-Pick-Rate'], 8)}")
    print(f"\nWritten to: {config.RESULTS_DT}")


if __name__ == "__main__":
    main()
