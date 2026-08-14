"""Generate human-readable log files from Method F trace JSONs.

Produces results/runtime/dynamic/traces/F{state}/logs/{case_id}.log
Format: identical to the console trace shown during development.
Run after the F benchmark completes.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACES_BASE = ROOT / "results" / "runtime" / "dynamic" / "traces"


def fmt_trace(rec: dict) -> str:
    """Format a single F-method trace record as human-readable log."""
    lines = []
    t = rec.get("trace", {})  # orchestration-level trace in record
    plan = rec.get("plan", {})
    steps = plan.get("steps", [])

    # Get method trace from first step
    method_trace = steps[0].get("method_trace", {}) if steps else {}
    f_tokens = method_trace.get("tokens", t.get("tokens", 0))
    f_llm = method_trace.get("llm_calls", 0)

    top5 = [s["candidates"][0]["ordId"] if s.get("candidates") else "(none)"
            for s in steps[:5]]
    gt = rec.get("plan", {}).get("steps", [{}])[0].get("expected", {})
    gt_ids = set(gt.get("expected_ordIds", [])) if gt else set()
    top1 = int(bool(top5) and top5[0] in gt_ids)
    top5h = int(any(oid in gt_ids for oid in top5))

    state = "enriched" if rec.get("state") == "enriched" else "clean"
    lines.append(f"┌─ {rec['case_id']} [{rec.get('difficulty','?')}] [{state}]"
                 f"  top1={top1} top5={top5h}"
                 f"  tokens={f_tokens}  llm_calls={f_llm}")
    lines.append(f"│  Prompt:  {rec['user_prompt']}")
    lines.append(f"│  GT:      {list(gt_ids)}")
    lines.append(f"│")

    # Decomposer info from method_trace
    dc = method_trace.get("decomp_trace", {})
    n_det = dc.get("n_detected", 1)
    n_used = method_trace.get("n_intents", 1)
    sub_queries = dc.get("sub_queries", [rec["user_prompt"]])
    lines.append(f"│  [Decomposer] n_detected={n_det}  n_used={n_used}")
    for j, sq in enumerate(sub_queries, 1):
        lines.append(f"│    sub_{j}: {sq}")
    lines.append(f"│")

    def fmt_sub(pt: dict, pool_size: int, prefix: str = "│  ") -> None:
        h = pt.get("phase0_intent", {}).get("hints", {})
        lines.append(f"{prefix}[Phase 0 · Intent Analysis]")
        lines.append(f"{prefix}  namespace_hint={h.get('likely_namespace')}"
                     f"  type={h.get('resource_type_hint')}"
                     f"  verb={h.get('action_verb')}")
        lines.append(f"{prefix}  entity_types={h.get('entity_types', [])}")

        a_info = pt.get("phase1a_embedding", {})
        lines.append(f"{prefix}[Phase 1a · Embedding A]  candidates={a_info.get('n_candidates', 0)}")

        d_info = pt.get("phase1b_guided_d", {})
        agent_steps = d_info.get("agent_steps", [])
        lines.append(f"{prefix}[Phase 1b · Guided D]"
                     f"  tool_calls={len(agent_steps)}"
                     f"  candidates={d_info.get('n_candidates', 0)}")
        for s in agent_steps:
            vals = list(s.get("input", {}).values())[:1]
            n_res = s.get("n_results", "")
            val_str = str(vals[0]) if vals else ""
            lines.append(f"{prefix}  → {s['tool']}({val_str})"
                         f"{'  → ' + str(n_res) + ' results' if n_res else ''}")

        pg = pt.get("phase2_graph", {})
        if pg:
            lines.append(f"{prefix}[Phase 2 · Graph C]"
                         f"  new_from_graph={pg.get('n_new_from_graph', 0)}")

        lines.append(f"{prefix}[Pool after merge]  size={pool_size}")

        pr = pt.get("phase3_rerank", {})
        lines.append(f"{prefix}[Phase 3 · Re-Rank B]"
                     f"  pick={pr.get('picked', '?')}"
                     f"  conf={pr.get('confidence', 0):.2f}")

        pv = pt.get("phase4_verify")
        if pv:
            lines.append(f"{prefix}[Phase 4 · Verify D]  confirmed={pv.get('confirmed')}")

    sub_traces = method_trace.get("sub_traces", [])
    if sub_traces:
        for i, st in enumerate(sub_traces):
            sq = st.get("sub_query", "")
            st_trace = st.get("trace", {})
            pt = st_trace.get("phase_traces", {})
            lines.append(f"│  ── Sub-query {i+1}: \"{sq[:80]}\"")
            fmt_sub(pt, st_trace.get("pool_size", 0), "│    ")
            lines.append(f"│")
        ur = method_trace.get("unified_rerank", {})
        lines.append(f"│  [Unified Re-Rank B · {n_used} intents]")
        lines.append(f"│    picks={ur.get('picks')}  conf={ur.get('confidence', 0):.2f}")
        reason = ur.get("reason", "")
        if reason:
            lines.append(f"│    reason={reason}")
    else:
        pt = method_trace.get("phase_traces", {})
        fmt_sub(pt, method_trace.get("pool_size", 0))

    lines.append(f"│")
    lines.append(f"│  Top-5 result:")

    # Get candidates from all steps
    all_cands = []
    for s in steps:
        for c in s.get("candidates", []):
            if c["ordId"] not in {x["ordId"] for x in all_cands}:
                all_cands.append(c)
    for i, cand in enumerate(all_cands[:5], 1):
        oid = cand["ordId"]
        lines.append(f"│    {i}. {oid}" + (" ← GT ✓" if oid in gt_ids else ""))
    lines.append(f"└─")
    return "\n".join(lines)


def main() -> None:
    for cond in ["F0f", "F1f"]:
        cond_dir = TRACES_BASE / cond
        if not cond_dir.exists():
            continue
        log_dir = cond_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        count = 0
        for trace_file in sorted(cond_dir.glob("dy-*.json")):
            rec = json.loads(trace_file.read_text())
            log_text = fmt_trace(rec)
            (log_dir / f"{trace_file.stem}.log").write_text(log_text)
            count += 1
        print(f"{cond}: wrote {count} log files → {log_dir}")


if __name__ == "__main__":
    main()
