"""Compute post-hoc metrics that aren't in records.jsonl directly:

  - recall_all (DY): all expected_ordIds in top-5 (vs candidate_recall = ANY)
  - bootstrap-CI for Top-1 per condition (DY + SA)
  - termination class for D/E (picked / refused / budget_exhausted)

Output: results/runtime/post_hoc_metrics.json — consumed by the web app.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_records(mode: str) -> list[dict]:
    p = ROOT / "results" / "runtime" / mode / "records.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def bootstrap_ci(values: list[int], n_iter: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    """95% CI for the mean of binary values via percentile bootstrap."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(42)
    n = len(values)
    means = []
    for _ in range(n_iter):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(n_iter * alpha / 2)]
    hi = means[int(n_iter * (1 - alpha / 2))]
    return (round(lo, 4), round(hi, 4))


# ─── DY: UNIFIED — single from *f, multi from *mh (partial credit) ─────────
# This is the single source of truth for ALL DY metrics.
# single-intent (dy-01..dy-20): always *f records (router bypassed)
# multi-intent  (dy-21..dy-40): always *mh records (per-sub-query partial credit)
dy_cases = {c["case_id"]: c for c in json.loads(
    (ROOT / "benchmark/test_cases/runtime/output/dynamic.json").read_text()
)}
single_ids = {cid for cid, c in dy_cases.items() if c.get("query_class", "single_intent") == "single_intent"}
multi_ids  = {cid for cid, c in dy_cases.items() if c.get("query_class") == "multi_intent"}

dy_records = load_records("dynamic")
mh_records = [json.loads(l) for l in (ROOT / "results/runtime/dynamic/records_mh.jsonl").read_text().splitlines() if l.strip()]
dy_metrics = {}

for method in ["A", "B", "C", "D", "E", "S", "F"]:
    for state_num, state in [("0", "clean"), ("1", "enriched")]:
        cond = method + state_num

        # ── Single-intent: *f records ────────────────────────────────────────
        single_recs = [r for r in dy_records
                       if r["condition"] == cond + "f" and r["case_id"] in single_ids]
        if not single_recs:
            continue
        single_top1 = [r["metrics"]["top1_acc"] or 0 for r in single_recs]
        single_top5 = [r["metrics"]["candidate_recall"] or 0 for r in single_recs]

        # ── Multi-intent: *mh records (per-case partial credit) ──────────────
        mh_cond = cond + "mh"
        mh_by_case: dict[str, list] = defaultdict(list)
        for r in mh_records:
            if r["condition"] == mh_cond and r["case_id"] in multi_ids:
                mh_by_case[r["case_id"]].append(r)

        multi_top1_partial: list[float] = []
        multi_top5_partial: list[float] = []
        recall_all_multi: list[int] = []
        for cid, subs in mh_by_case.items():
            t1 = sum(s["top1_acc"] or 0 for s in subs) / len(subs)
            t5 = sum(s["candidate_recall"] or 0 for s in subs) / len(subs)
            multi_top1_partial.append(t1)
            multi_top5_partial.append(t5)
            # recall_all: all sub-GTs in top-5
            all_in = all(s["candidate_recall"] == 1 for s in subs)
            recall_all_multi.append(int(all_in))

        # ── Combined n=40 ────────────────────────────────────────────────────
        combined_top1 = single_top1 + multi_top1_partial
        combined_top5 = single_top5 + multi_top5_partial
        n = len(combined_top1)
        if n == 0:
            continue

        top1_ci = bootstrap_ci([round(v) for v in combined_top1])
        top5_ci = bootstrap_ci([round(v) for v in combined_top5])

        # recall_all over all 40: single uses any-GT-in-step-candidates
        recall_all_single: list[int] = []
        for r in single_recs:
            cid = r["case_id"]
            expected = set(dy_cases[cid].get("expected_ordIds", []))
            top5_set: set[str] = set()
            for step in r.get("plan", {}).get("steps", []):
                for c in (step.get("candidates") or [])[:5]:
                    top5_set.add(c["ordId"])
            recall_all_single.append(int(bool(expected) and expected.issubset(top5_set)))

        recall_all_all = recall_all_single + recall_all_multi
        top1_mean = round(sum(combined_top1) / n, 4)
        top5_mean = round(sum(combined_top5) / n, 4)

        dy_metrics[cond] = {
            "n": n,
            "top1_mean": top1_mean,
            "top1_ci": top1_ci,
            "top5_mean": top5_mean,
            "top5_ci": top5_ci,
            "recall_all_all": round(sum(recall_all_all) / len(recall_all_all), 4) if recall_all_all else None,
            "recall_all_multi": round(sum(recall_all_multi) / len(recall_all_multi), 4) if recall_all_multi else None,
            "n_multi": len(recall_all_multi),
            "top1_single": round(sum(single_top1) / len(single_top1), 4) if single_top1 else None,
            "top1_multi": round(sum(multi_top1_partial) / len(multi_top1_partial), 4) if multi_top1_partial else None,
        }


# ─── SA: merge gf (gap-forced) records, prefer gf if exists ───────────────
sa_records = load_records("skill_adjusted")
sa_metrics = {}

for method in ["A", "B", "C", "D", "E", "S"]:
    for state_num, state in [("0", "clean"), ("1", "enriched")]:
        cond = method + state_num
        regular = {r["case_id"]: r for r in sa_records if r["condition"] == cond}
        forced = {r["case_id"]: r for r in sa_records if r["condition"] == cond + "gf"}
        merged_dict = {**regular, **forced}  # gf wins
        merged = list(merged_dict.values())
        if not merged:
            continue

        # Per-case partial Top-1 (#hits / #expected)
        top1_per_case = []
        top5_per_case = []
        for r in merged:
            expected = set(r.get("expected_gaps", []))
            if not expected:
                continue
            gap_steps = [s for s in r.get("plan", {}).get("steps", [])
                         if s.get("source") in ("gap", "gap_forced")]
            t1 = sum(
                1 for e in expected
                if any(((s.get("candidates") or [{}])[0].get("ordId") == e)
                       for s in gap_steps if s.get("candidates"))
            ) / len(expected)
            t5 = sum(
                1 for e in expected
                if any(any(c.get("ordId") == e
                           for c in (s.get("candidates") or [])[:5])
                       for s in gap_steps)
            ) / len(expected)
            top1_per_case.append(t1)
            top5_per_case.append(t5)

        # bootstrap CI on continuous-valued per-case Top-1
        rng = random.Random(42)
        n = len(top1_per_case)
        means = []
        for _ in range(2000):
            sample = [top1_per_case[rng.randrange(n)] for _ in range(n)]
            means.append(sum(sample) / n)
        means.sort()
        top1_ci = (round(means[50], 4), round(means[1949], 4))

        sa_metrics[cond] = {
            "n": n,
            "top1_mean": round(sum(top1_per_case) / n, 4),
            "top1_ci": top1_ci,
            "top5_mean": round(sum(top5_per_case) / n, 4),
        }


# ─── D/E: termination classification ──────────────────────────────────────
termination = defaultdict(lambda: {"picked": 0, "refused": 0, "budget_exhausted": 0})
for mode in ["dynamic", "skill_adjusted", "out_of_scope"]:
    for r in load_records(mode):
        if r["method"] not in ("D", "E"):
            continue
        for step in r.get("plan", {}).get("steps", []):
            mt = step.get("method_trace", {})
            if mt.get("refused"):
                termination[(r["method"], r["state"])]["refused"] += 1
            elif mt.get("picked_ord_id"):
                termination[(r["method"], r["state"])]["picked"] += 1
            else:
                # No pick, no refuse → budget exhausted
                termination[(r["method"], r["state"])]["budget_exhausted"] += 1

termination_out = {f"{m}_{s}": v for (m, s), v in termination.items()}


# ─── Output ───────────────────────────────────────────────────────────────
out = {
    "dy_metrics": dy_metrics,
    "sa_metrics": sa_metrics,
    "termination": termination_out,
    "_meta": {
        "bootstrap_iter": 2000,
        "alpha": 0.05,
        "note": "post-hoc metrics computed from existing records.jsonl files",
    },
}

out_path = ROOT / "results/runtime/post_hoc_metrics.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"Wrote {out_path}")
print(f"  DY: {len(dy_metrics)} conditions")
print(f"  SA: {len(sa_metrics)} conditions")
print(f"  Termination: {len(termination_out)} (method, state) groups")
