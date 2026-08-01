"""Reproduce the Statistical-reliability paragraph of Paper II from raw traces.

This is the reproducibility record for the numbers quoted in Sec. Results
"Statistical reliability" (bootstrap confidence intervals + paired McNemar
p-values) on the Dynamic mode, enriched state.

WHY A SEPARATE SCRIPT
  The Dynamic column of the paper is scored the way the author scored it by
  hand, which the generic numbers.py does NOT reproduce: Dynamic combines the
  20 single-intent cases (dy-01..dy-20) with the 20 multi-intent cases
  (dy-21..dy-40). The two case families are stored differently on disk:
    - single-intent: results/runtime/dynamic/traces/<M>1f/  (one file per case,
      metrics in a "metrics" wrapper, 0/1 outcome per case)
    - multi-intent:  results/runtime/dynamic/traces/<M>1mh/ (one file per
      SUB-QUERY, flat top1_acc/candidate_recall fields; a case has several)
  A multi-intent case is scored as the MEAN over its sub-queries (fraction of
  sub-queries answered correctly), so each of the 40 Dynamic cases contributes
  exactly one unit. This reproduces the paper's table values exactly, e.g.
  Embedding R@1=28.7 / R@5=70.0 and Agentic Hybrid R@1=45.0.

USAGE
  python3 experiments/paper2/statistics.py
  (run from the repo root; deterministic, seed=42)
"""
from __future__ import annotations

import glob
import json
import os
from math import comb
import random

random.seed(42)  # fixed seed -> identical bootstrap CIs on every run

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RT = os.path.join(ROOT, "results", "runtime", "dynamic", "traces")
BOOT = 10_000

NAMES = {"S": "Baseline", "A": "Embedding", "B": "Progressive", "C": "Graph",
         "D": "Agentic Tools", "E": "Agentic Raw", "F": "Agentic Hybrid"}
ORDER = ["S", "A", "B", "C", "D", "E", "F"]


def dynamic_percase(m: str, state: int, metric: str) -> dict:
    """case_id -> score for all 40 Dynamic cases (reported combined scoring).

    single-intent (dy-01..20): 0/1 from the <M><state>f folder.
    multi-intent  (dy-21..40): mean over sub-queries from <M><state>mh.
    """
    out: dict[str, float] = {}
    # single-intent forced runs: metrics wrapper, keep only dy-01..dy-20
    for fp in glob.glob(os.path.join(RT, f"{m}{state}f", "*.json")):
        t = json.load(open(fp))
        cid = t["case_id"]
        if int(cid.split("-")[1]) <= 20:
            out[cid] = t["metrics"].get(metric) or 0
    # multi-intent: one row per sub-query, average per case
    agg: dict[str, list] = {}
    for fp in glob.glob(os.path.join(RT, f"{m}{state}mh", "*.json")):
        t = json.load(open(fp))
        agg.setdefault(t["case_id"], []).append(t.get(metric) or 0)
    for cid, vals in agg.items():
        out[cid] = sum(vals) / len(vals)
    return out


def pct(vals) -> float:
    return round(sum(vals) / len(vals) * 100, 1) if vals else 0.0


def boot_ci(vals) -> tuple:
    """95% bootstrap CI on the mean of a numeric list, in percent."""
    n = len(vals)
    if n == 0:
        return (0.0, 0.0, 0.0)
    ms = sorted(sum(vals[random.randrange(n)] for _ in range(n)) / n * 100
                for _ in range(BOOT))
    return pct(vals), round(ms[int(0.025 * BOOT)], 1), round(ms[int(0.975 * BOOT)], 1)


def mcnemar(a: dict, b: dict) -> tuple:
    """Exact two-sided McNemar on paired case scores (thresholded at >=0.5).

    b_only = cases method a solves but b does not; c_only = the reverse.
    """
    cases = sorted(set(a) & set(b))
    A = {c: 1 if a[c] >= 0.5 else 0 for c in cases}
    B = {c: 1 if b[c] >= 0.5 else 0 for c in cases}
    bb = sum(1 for c in cases if A[c] == 1 and B[c] == 0)
    cc = sum(1 for c in cases if A[c] == 0 and B[c] == 1)
    n = bb + cc
    if n == 0:
        return bb, cc, 1.0
    k = min(bb, cc)
    p = min(1.0, sum(comb(n, i) for i in range(k + 1)) / (2 ** n) * 2)
    return bb, cc, p


def main() -> None:
    print("Statistical reliability (Paper II) — Dynamic, enriched, n=40")
    print(f"Repo root: {ROOT}\nSeed: 42, bootstrap resamples: {BOOT}\n")

    print("R@1 and R@5 with 95% bootstrap confidence intervals:")
    print(f"  {'Method':14} {'R@1 [CI]':>22} {'R@5 [CI]':>22}")
    p1_scores = {}
    for m in ORDER:
        t1 = dynamic_percase(m, 1, "top1_acc")
        r5 = dynamic_percase(m, 1, "candidate_recall")
        p1_scores[m] = t1
        p1, lo1, hi1 = boot_ci(list(t1.values()))
        r5v, lo5, hi5 = boot_ci(list(r5.values()))
        print(f"  {NAMES[m]:14} {f'{p1} [{lo1}, {hi1}]':>22} {f'{r5v} [{lo5}, {hi5}]':>22}")

    print("\nPaired McNemar p-values on R@1 (structured methods vs the free-text Baseline):")
    for m in ("F", "D", "A"):
        bb, cc, p = mcnemar(p1_scores[m], p1_scores["S"])
        sig = "  SIG (p<0.05)" if p < 0.05 else ""
        print(f"  {NAMES[m]:14} vs Baseline   b/c={bb}/{cc}  p={p:.4f}{sig}")

    print("\nPaired McNemar p-values among the leading structured methods:")
    for a, b in (("F", "D"), ("F", "A"), ("D", "A")):
        bb, cc, p = mcnemar(p1_scores[a], p1_scores[b])
        sig = "  SIG (p<0.05)" if p < 0.05 else "  (not significant)"
        print(f"  {NAMES[a]:14} vs {NAMES[b]:14} b/c={bb}/{cc}  p={p:.4f}{sig}")

    print("\nNote: Embedding leads on R@5, Agentic Hybrid on R@1 — no single method")
    print("wins every metric. See the paper's Statistical-reliability paragraph.")


if __name__ == "__main__":
    main()
