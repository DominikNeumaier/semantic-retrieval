"""Build the consolidated dynamic evaluation from the curated traces.

Single source of truth: results/runtime/dynamic/traces/{cond}/ where
cond in {A,B,C,D,E,F,S}{0,1}. Each condition folder holds:
  dy-01.json .. dy-20.json          single-intent cases (forced run, router
                                     bypassed) — metrics live in trace["metrics"]
  dy-21_sub1.json / dy-21_sub2.json  multi-intent cases (decomposed into
  .. dy-40_sub1/sub2.json            sub-queries) — top1_acc / candidate_recall
                                     are top-level fields on each sub trace

Outputs (into results/runtime/dynamic/):
  summary.json    per-condition single / multi / combined R@1 and R@5,
                  same schema the web app already renders (replaces the old
                  summary_combined.json + summary_mh.json pair).

With --merge it also folds records_mh.jsonl into records.jsonl (idempotent:
mh lines already present are not appended again), so the dynamic mode keeps
one record file instead of two.

Run:  python3 analysis/build_dynamic_summary.py [--merge]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DY = ROOT / "results" / "runtime" / "dynamic"
TRACES = DY / "traces"

METHODS = ["A", "B", "C", "D", "E", "S", "F"]
STATES = ["0", "1"]


def _single_metric(trace: dict, key: str) -> int:
    """Pull a 0/1 metric from a single-intent trace (nested under metrics)."""
    m = trace.get("metrics") or {}
    return m.get(key, 0) or 0


def build_condition(cond: str) -> dict | None:
    """Compute one condition row from its curated trace folder."""
    d = TRACES / cond
    if not d.is_dir():
        return None

    # ── Single-intent: dy-01..20, files without a _sub suffix ──────────────
    single_files = sorted(
        p for p in d.glob("dy-*.json") if "_sub" not in p.name
    )
    single_top1_vals, single_top5_vals = [], []
    for p in single_files:
        t = json.loads(p.read_text())
        single_top1_vals.append(_single_metric(t, "top1_acc"))
        single_top5_vals.append(_single_metric(t, "candidate_recall"))

    # ── Multi-intent: dy-21..40, grouped by case across its sub traces ─────
    by_case: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(d.glob("dy-*_sub*.json")):
        case_id = p.name.split("_sub")[0]  # dy-21_sub1.json -> dy-21
        by_case[case_id].append(json.loads(p.read_text()))

    per_case_top1, per_case_top5, strict_top1 = [], [], []
    for _case, subs in by_case.items():
        n = len(subs)
        per_case_top1.append(sum(s.get("top1_acc", 0) or 0 for s in subs) / n)
        per_case_top5.append(sum(s.get("candidate_recall", 0) or 0 for s in subs) / n)
        strict_top1.append(int(all((s.get("top1_acc", 0) or 0) == 1 for s in subs)))

    n_single = len(single_files)
    n_multi = len(by_case)

    single_top1 = sum(single_top1_vals) / n_single if n_single else None
    single_top5 = sum(single_top5_vals) / n_single if n_single else None
    mh_top1 = sum(per_case_top1) / n_multi if n_multi else None
    mh_top5 = sum(per_case_top5) / n_multi if n_multi else None
    mh_strict = sum(strict_top1) / n_multi if n_multi else None

    if single_top1 is not None and mh_top1 is not None:
        n_total = n_single + n_multi
        combined_top1 = (single_top1 * n_single + mh_top1 * n_multi) / n_total
        combined_top5 = (single_top5 * n_single + mh_top5 * n_multi) / n_total
    else:
        n_total = n_single + n_multi
        combined_top1 = combined_top5 = None

    r = lambda x: round(x, 4) if x is not None else None
    return {
        "condition": cond,
        "n_total": n_total,
        "n_single": n_single,
        "n_multi": n_multi,
        "single_top1": r(single_top1),
        "single_top5": r(single_top5),
        "mh_top1_partial": r(mh_top1),
        "mh_top5_partial": r(mh_top5),
        "mh_top1_strict": r(mh_strict),
        "combined_top1": r(combined_top1),
        "combined_top5": r(combined_top5),
    }


def build_summary() -> None:
    rows = []
    for method in METHODS:
        for state in STATES:
            row = build_condition(method + state)
            if row is None:
                continue
            rows.append(row)
            if row["combined_top1"] is not None:
                print(f"{row['condition']:<4}  single={row['single_top1']:.3f}  "
                      f"mh_partial={row['mh_top1_partial']:.3f}  "
                      f"mh_strict={row['mh_top1_strict']:.3f}  "
                      f"combined={row['combined_top1']:.3f}")

    out = {
        "_note": (
            "Consolidated dynamic metric built from dynamic/traces/. "
            "single-intent = cases dy-01..20 (forced run, router bypassed); "
            "multi-intent = cases dy-21..40 (decomposed sub-queries, partial "
            "credit per case). combined = (single x n_single + mh_partial x "
            "n_multi) / n_total."
        ),
        "conditions": rows,
    }
    (DY / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {DY / 'summary.json'} ({len(rows)} conditions)")


def merge_records() -> None:
    """Fold records_mh.jsonl into records.jsonl (idempotent)."""
    base = DY / "records.jsonl"
    mh = DY / "records_mh.jsonl"
    if not mh.exists():
        print(f"no {mh.name} to merge (already consolidated)")
        return
    base_lines = base.read_text().splitlines() if base.exists() else []
    existing = set(base_lines)
    mh_lines = [l for l in mh.read_text().splitlines() if l.strip()]
    added = [l for l in mh_lines if l not in existing]
    with base.open("a") as f:
        for l in added:
            f.write(l + "\n")
    print(f"merged {len(added)} mh lines into {base.name} "
          f"(now {len(base_lines) + len(added)} total; {len(mh_lines) - len(added)} already present)")


if __name__ == "__main__":
    if "--merge" in sys.argv:
        merge_records()
    build_summary()
