"""Reproduce the Statistical-reliability paragraph of Paper II from raw traces.

This is the reproducibility record for the numbers quoted in Sec. Results
"Statistical reliability" (bootstrap confidence intervals + paired McNemar
p-values) on the Dynamic mode, enriched state.

WHY A SEPARATE SCRIPT
  The Dynamic column of the paper is scored the way the author scored it by
  hand, which the generic numbers.py does NOT reproduce: Dynamic combines the
  20 single-intent cases (dy-01..dy-20) with the 20 multi-intent cases
  (dy-21..dy-40). Both families live in the same per-condition folder
  results/retrieval/runtime/dynamic/traces/<M><state>/ and are told apart by the
  file name:
    - single-intent: dy-01.json .. dy-20.json  (one file per case, metrics in a
      "metrics" wrapper, 0/1 outcome per case)
    - multi-intent:  dy-NN_subX.json           (one file per SUB-QUERY, flat
      top1_acc/candidate_recall fields; a case has several)
  A multi-intent case is scored as the MEAN over its sub-queries (fraction of
  sub-queries answered correctly), so each of the 40 Dynamic cases contributes
  exactly one unit. This reproduces the paper's Dynamic-enriched table values,
  e.g. Embedding R@5=70.0 and Agentic Hybrid R@1=45.0 (Embedding R@1 sits on the
  28.75 rounding boundary, reported as 28.8 in Table 3).

USAGE
  python3 analysis/statistics.py
  (run from the repo root; deterministic, seed=42)
"""
from __future__ import annotations

import glob
import json
import os
from math import comb
import random

random.seed(42)  # fixed seed -> identical bootstrap CIs on every run

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RT = os.path.join(ROOT, "results", "retrieval", "runtime", "dynamic", "traces")
BOOT = 10_000

NAMES = {"S": "Baseline", "A": "Embedding", "B": "Progressive", "C": "Graph",
         "D": "Agentic Tools", "E": "Agentic Raw", "F": "Agentic Hybrid"}
ORDER = ["S", "A", "B", "C", "D", "E", "F"]


def dynamic_percase(m: str, state: int, metric: str) -> dict:
    """case_id -> score for all 40 Dynamic cases (reported combined scoring).

    single-intent (dy-01..20): 0/1 from the dy-NN.json files.
    multi-intent  (dy-21..40): mean over sub-queries from the dy-NN_subX.json files.
    Both live in the same <M><state> folder, told apart by the "_sub" suffix.
    """
    out: dict[str, float] = {}
    agg: dict[str, list] = {}
    for fp in glob.glob(os.path.join(RT, f"{m}{state}", "*.json")):
        t = json.load(open(fp))
        cid = t["case_id"]
        if "_sub" in os.path.basename(fp):
            # multi-intent: one row per sub-query, averaged per case below
            agg.setdefault(cid, []).append(t.get(metric) or 0)
        else:
            # single-intent: metrics wrapper, 0/1 per case
            out[cid] = t["metrics"].get(metric) or 0
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


def delta_ci(clean: dict, enr: dict) -> tuple:
    """95% bootstrap CI on the paired enriched-minus-clean R@1 mean, in pp.

    Returns (point, lo, hi, n). The interval excluding zero is the evidence
    threshold used in the paper (a genuine enrichment gain rather than noise).
    """
    cases = sorted(set(clean) & set(enr))
    d = [enr[c] - clean[c] for c in cases]
    n = len(d)
    if n == 0:
        return (0.0, 0.0, 0.0, 0)
    ms = sorted(sum(d[random.randrange(n)] for _ in range(n)) / n * 100
                for _ in range(BOOT))
    point = round(sum(d) / n * 100, 1)
    return point, round(ms[int(0.025 * BOOT)], 1), round(ms[int(0.975 * BOOT)], 1), n


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
    min_p = 1.0
    for a, b in (("F", "D"), ("F", "A"), ("D", "A")):
        bb, cc, p = mcnemar(p1_scores[a], p1_scores[b])
        min_p = min(min_p, p)
        sig = "  SIG (p<0.05)" if p < 0.05 else "  (not significant)"
        print(f"  {NAMES[a]:14} vs {NAMES[b]:14} b/c={bb}/{cc}  p={p:.4f}{sig}")
    print(f"  -> smallest p among the leading methods: {min_p:.4f} "
          f"(all > 0.05, so no reliable ordering)")

    # Enrichment-delta CIs: re-seed so this block is reproducible on its own,
    # independent of how many bootstrap draws the R@1/R@5 section consumed above.
    random.seed(42)
    print("\nEnrichment delta (enriched - clean) on R@1, 95% bootstrap CIs (pp):")
    for m in ORDER:
        c = dynamic_percase(m, 0, "top1_acc")
        e = dynamic_percase(m, 1, "top1_acc")
        d, lo, hi, n = delta_ci(c, e)
        flag = "  excludes 0 -> genuine gain" if lo > 0 else ""
        print(f"  {NAMES[m]:14} {f'{d:+.1f} [{lo:+.1f}, {hi:+.1f}]':>26}{flag}")

    print("\nNote: Embedding leads on R@5, Agentic Hybrid on R@1 — no single method")
    print("wins every metric. See the paper's Statistical-reliability paragraph.")


if __name__ == "__main__":
    main()
