"""Compute combined DY summary: single_forced + multi_mh (partial credit).

For each condition (A0, A1, ... F1):
  single_forced: top1 from *f records, only single_intent cases (n=20)
  mh_partial:    per-case mean top1 from *mh records (n=20 cases, 2 sub-queries each)
  combined_top1: (single_forced × 20 + mh_partial × 20) / 40
  combined_top5: same logic for candidate_recall
  mh_strict:     fraction of cases where ALL sub-queries top1 correct

Output: results/runtime/dynamic/summary_combined.json
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DY = ROOT / "results" / "runtime" / "dynamic"

def load_records(path: Path) -> list[dict]:
    if not path.exists(): return []
    out = []
    for l in path.read_text().splitlines():
        if not l.strip(): continue
        try: out.append(json.loads(l))
        except: pass
    return out

def load_case_classes() -> dict[str, str]:
    cases = json.loads((ROOT / "benchmark" / "test_cases" / "runtime" / "output" / "dynamic.json").read_text())
    return {c["case_id"]: c.get("query_class", "single_intent") for c in cases}

def main():
    forced_records = load_records(DY / "records.jsonl")
    mh_records = load_records(DY / "records_mh.jsonl")
    case_classes = load_case_classes()

    METHODS = ["A", "B", "C", "D", "E", "S", "F"]
    STATES = ["0", "1"]
    results = []

    for method in METHODS:
        for state in STATES:
            cond_base = method + state
            cond_f = cond_base + "f"
            cond_mh = cond_base + "mh"

            # ── Single-intent from forced records ──────────────────────────
            f_single = [r for r in forced_records
                        if r.get("condition") == cond_f
                        and case_classes.get(r.get("case_id", "")) == "single_intent"]

            if f_single:
                single_top1 = sum(r["metrics"].get("top1_acc", 0) or 0 for r in f_single) / len(f_single)
                single_top5 = sum(r["metrics"].get("candidate_recall", 0) or 0 for r in f_single) / len(f_single)
                n_single = len(f_single)
            else:
                single_top1 = single_top5 = None
                n_single = 0

            # ── Multi-intent from mh records (partial credit per case) ─────
            mh = [r for r in mh_records if r.get("condition") == cond_mh]
            if mh:
                # Group by case_id → partial credit
                by_case: dict[str, list] = defaultdict(list)
                for r in mh:
                    by_case[r["case_id"]].append(r)

                per_case_top1 = []
                per_case_top5 = []
                strict_top1 = []
                for case_id, recs in by_case.items():
                    n = len(recs)
                    t1 = sum(r["top1_acc"] for r in recs) / n
                    t5 = sum(r["candidate_recall"] for r in recs) / n
                    per_case_top1.append(t1)
                    per_case_top5.append(t5)
                    strict_top1.append(int(all(r["top1_acc"] == 1 for r in recs)))

                mh_top1 = sum(per_case_top1) / len(per_case_top1)
                mh_top5 = sum(per_case_top5) / len(per_case_top5)
                mh_strict = sum(strict_top1) / len(strict_top1)
                n_multi = len(by_case)
            else:
                mh_top1 = mh_top5 = mh_strict = None
                n_multi = 0

            # ── Combined ───────────────────────────────────────────────────
            if single_top1 is not None and mh_top1 is not None:
                n_total = n_single + n_multi
                combined_top1 = (single_top1 * n_single + mh_top1 * n_multi) / n_total
                combined_top5 = (single_top5 * n_single + mh_top5 * n_multi) / n_total
            else:
                combined_top1 = combined_top5 = None
                n_total = n_single + n_multi

            row = {
                "condition": cond_base,
                "n_total": n_total,
                "n_single": n_single,
                "n_multi": n_multi,
                "single_top1": round(single_top1, 4) if single_top1 is not None else None,
                "single_top5": round(single_top5, 4) if single_top5 is not None else None,
                "mh_top1_partial": round(mh_top1, 4) if mh_top1 is not None else None,
                "mh_top5_partial": round(mh_top5, 4) if mh_top5 is not None else None,
                "mh_top1_strict": round(mh_strict, 4) if mh_strict is not None else None,
                "combined_top1": round(combined_top1, 4) if combined_top1 is not None else None,
                "combined_top5": round(combined_top5, 4) if combined_top5 is not None else None,
            }
            results.append(row)

            if combined_top1 is not None:
                print(f"{cond_base:<4}  single={single_top1:.3f}  mh_partial={mh_top1:.3f}  "
                      f"mh_strict={mh_strict:.3f}  combined={combined_top1:.3f}")
            else:
                print(f"{cond_base:<4}  single={'—' if single_top1 is None else f'{single_top1:.3f}':>5}  "
                      f"mh={'—' if mh_top1 is None else f'{mh_top1:.3f}':>5}  combined=—")

    out = {
        "_note": (
            "Combined DY metric: single-intent uses forced (*f) records (router bypassed), "
            "multi-intent uses mh records (per-sub-query retrieval, partial credit). "
            "combined_top1 = (single_forced × n_single + mh_partial × n_multi) / n_total"
        ),
        "conditions": results,
    }
    (DY / "summary_combined.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {DY / 'summary_combined.json'}")

if __name__ == "__main__":
    main()
