"""Reproduce the Fig. 7 (cost--precision) and Fig. 8 (token efficiency) numbers.

Run this and it prints exactly the coordinates plotted in Paper II's
Fig.~\\ref{fig:pareto} and Fig.~\\ref{fig:token-eff}: per method, on Dynamic in
the enriched state, the average tokens per case, the R@1, and R@1 per 1k tokens.

TOKEN DEFINITION (matches the "Tokens = average per case" column of
tab:runtime-results)
  Tokens per case = the retrieval method's own token spend on that case:
    - single-intent case (dy-01..20):  trace.tokens of the one retrieval.
    - multi-intent case (dy-21..40):   SUM of trace tokens over its sub-queries
      (you pay for every sub-query the decomposer issues).
  Averaged over the 40 Dynamic cases.

  EMBEDDING is the one exception. Its retrieval issues no per-query LLM call
  (llm_calls == 0); the ~30k it records under trace.tokens is the ONE-TIME cost
  of embedding the candidate corpus, which is a reusable index, not a per-query
  cost. Charging that amortised index build to every query would misrepresent
  its runtime cost, so Embedding is reported at its per-query cost of ~1k
  tokens, consistent with the table and Fig. 7. This constant is the only value
  not read straight from the traces; everything else is computed.

SCORING
  Dynamic combined = 20 single-intent cases (0/1 from dy-NN.json "metrics") plus
  20 multi-intent cases (mean over sub-queries from dy-NN_subX.json), each case
  contributing one unit, so R@1 here equals the Dynamic-enriched R@1 of Table 3.

USAGE
  python3 analysis/figures.py     (run from the repo root; deterministic)
"""
from __future__ import annotations

import glob
import json
import math
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RT = os.path.join(ROOT, "results", "retrieval", "runtime", "dynamic", "traces")

NAMES = {"S": "Baseline", "A": "Embedding", "B": "Progressive", "C": "Graph",
         "D": "Agentic Tools", "E": "Agentic Raw", "F": "Agentic Hybrid"}
ORDER = ["S", "A", "B", "C", "D", "E", "F"]

# Embedding per-query cost (see TOKEN DEFINITION above). The traces store the
# one-time corpus-embedding cost, not the per-query cost, so this is pinned.
EMB_QUERY_TOKENS = 1000


def hround(x: float, nd: int = 1) -> float:
    """Round half up (the rounding used to produce Table 3). Values are >= 0.

    The 1e-6 nudge counters binary float error at exact .x5 boundaries, e.g.
    Embedding's 23/80 = 0.2875 stores as 0.28749999.. and must round to 28.8.
    """
    f = 10 ** nd
    return math.floor(x * f + 1e-6 + 0.5) / f


def load_cond(m: str, state: int) -> tuple[dict, dict]:
    """Return (r1_percase, tokens_percase) over the 40 Dynamic cases.

    r1_percase[cid]     -> 0/1 (single) or mean-over-subs (multi) top1_acc.
    tokens_percase[cid] -> trace.tokens (single) or SUM of sub tokens (multi).
    """
    r1: dict[str, float] = {}
    tok: dict[str, float] = {}
    sub_r1: dict[str, list] = {}
    for fp in glob.glob(os.path.join(RT, f"{m}{state}", "*.json")):
        t = json.load(open(fp))
        cid = t["case_id"]
        if "_sub" in os.path.basename(fp):
            sub_r1.setdefault(cid, []).append(t.get("top1_acc") or 0)
            tok[cid] = tok.get(cid, 0) + (t.get("tokens") or 0)
        else:
            r1[cid] = t["metrics"].get("top1_acc") or 0
            tok[cid] = (t.get("trace") or {}).get("tokens") or 0
    for cid, vals in sub_r1.items():
        r1[cid] = sum(vals) / len(vals)
    return r1, tok


def main() -> None:
    rows = []
    for m in ORDER:
        r1, tok = load_cond(m, 1)  # state 1 = enriched
        n = len(r1)
        p1 = hround(sum(r1.values()) / n * 100, 1) if n else 0.0
        mtok = EMB_QUERY_TOKENS if m == "A" else round(sum(tok.values()) / n) if n else 0
        eff = hround(p1 / (mtok / 1000), 1) if mtok else 0.0
        rows.append((m, mtok, p1, eff))

    print("Fig. 7 (fig:pareto) — Dynamic, enriched: tokens per case vs R@1")
    print(f"  {'Method':14}{'Tokens/case':>13}{'R@1 (%)':>10}")
    for m, mtok, p1, _ in rows:
        note = "  (per-query; corpus index amortised)" if m == "A" else ""
        print(f"  {NAMES[m]:14}{mtok:>13}{p1:>10}{note}")

    print("\nFig. 8 (fig:token-eff) — R@1 per 1k tokens (Dynamic, enriched)")
    print(f"  {'Method':14}{'R@1/1k':>10}")
    for m, _, _, eff in sorted(rows, key=lambda r: r[3]):
        print(f"  {NAMES[m]:14}{eff:>10}")

    best = max(rows, key=lambda r: r[3])[3]
    second = sorted((r[3] for r in rows), reverse=True)[1]
    print(f"\n  Embedding lead over the next method: {best/second:.1f}x "
          f"({best} vs {second} R@1/1k)")


if __name__ == "__main__":
    main()
