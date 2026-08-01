"""Reproduce every quantitative result reported in Paper II (Semantic Retrieval).

This script recomputes, directly from the raw per-case trace JSONs under
`results/runtime/` and `results/design-time/`, every number that appears in
Paper II's Results/Discussion sections, its four result figures, and its
appendix result tables. It is the reproducibility record for that paper: run it
and you get back exactly the values printed in the .tex.

WHAT IT REPRODUCES
  1. Runtime results table (tab:runtime-results): R@1, R@5 per method, Skill-
     Adjusted + Dynamic, clean vs enriched, plus the enrichment delta.
  2. Out-of-scope table (tab:oos-results): refusal + false-pick rate per method.
  3. Funnel figure (fig:funnel): per-mode routing accuracy + best-method
     resolution rate, clean and enriched.
  4. gap / pareto / token-efficiency figures (fig:gap, fig:pareto,
     fig:token-eff): Dynamic-enriched R@1, R@5, tokens, R@1-per-1k-tokens.
  5. Failure-mode table (tab:failure-modes): per-method breakdown into
     correct / rank-error / retrieval-miss / empty-return on Dynamic enriched.
  6. Statistical-reliability paragraph: 95% bootstrap CIs (paired, on the
     enrichment delta and on absolute metrics) and paired McNemar p-values for
     the method comparisons quoted in the text.
  7. Design-time appendix table (tab:full-results-dt) + its bootstrap CIs.

SCORING CONVENTIONS (must match the paper — see appendix "Condition labels")
  - Dynamic combined = single-intent forced runs (`<M><state>f`) over the 20
    single-intent cases merged with multi-intent multi-hint runs
    (`<M><state>mh`) over the 20 multi-intent cases. Here we report the
    `f` variant where a single Dynamic number is needed, matching the tables.
  - Skill-Adjusted = gap-forced runs (`<M><state>gf`) where present.
  - Out-of-scope = plain `<M><state>`.
  - Design-time = plain `<M>` (clean only; enrichment is its OUTPUT).
  Method letters map to names: S=Baseline, A=Embedding, B=Progressive,
  C=Graph, D=Agentic Tools, E=Agentic Raw, F=Agentic Hybrid.

USAGE
  python3 experiments/paper2/numbers.py
  (run from the repo root; no arguments; deterministic, seed=42)
"""
from __future__ import annotations

import glob
import json
import os
from math import comb
import random

random.seed(42)  # fixed seed -> identical bootstrap CIs on every run

# Repo root is two levels up from this file (experiments/paper2/).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RT = os.path.join(ROOT, "results", "runtime")
DT = os.path.join(ROOT, "results", "design-time")

NAMES = {"S": "Baseline", "A": "Embedding", "B": "Progressive",
         "C": "Graph", "D": "Agentic Tools", "E": "Agentic Raw",
         "F": "Agentic Hybrid"}
ORDER = ["S", "A", "B", "C", "D", "E", "F"]
BOOT = 10_000


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load(mode: str, cond: str, metric: str) -> dict:
    """case_id -> 0/1 metric value for every trace in results/runtime/<mode>/traces/<cond>/."""
    d = os.path.join(RT, mode, "traces", cond)
    if not os.path.isdir(d):
        return {}
    out = {}
    for fp in glob.glob(os.path.join(d, "*.json")):
        t = json.load(open(fp))
        out[t["case_id"]] = t["metrics"].get(metric, 0) or 0
    return out


def _variant(mode: str, m: str, state: int) -> str:
    """Reported condition per mode (see scoring conventions)."""
    base = os.path.join(RT, mode)
    if mode == "dynamic":
        cand = f"{m}{state}f"
    elif mode == "skill_adjusted":
        cand = f"{m}{state}gf"
    else:
        cand = f"{m}{state}"
    return cand if os.path.isdir(os.path.join(base, "traces", cand)) else f"{m}{state}"


def _pct(vals) -> float:
    return round(sum(vals) / len(vals) * 100, 1) if vals else 0.0


def _boot_ci(vals) -> tuple:
    """95% bootstrap CI on the mean of a 0/1 (or numeric) list, in percent."""
    n = len(vals)
    if n == 0:
        return (0.0, 0.0, 0.0)
    ms = sorted(sum(vals[random.randrange(n)] for _ in range(n)) / n * 100
                for _ in range(BOOT))
    return _pct(vals), round(ms[int(0.025 * BOOT)], 1), round(ms[int(0.975 * BOOT)], 1)


def _paired(mode: str, m: str, metric: str = "top1_acc") -> list:
    """List of (clean, enriched) pairs on the same cases (reported variant)."""
    cl = dict(_load(mode, _variant(mode, m, 0), metric))
    en = _load(mode, _variant(mode, m, 1), metric)
    cases = sorted(set(cl) & set(en))
    return [(cl[c], en[c]) for c in cases]


def _boot_delta_ci(pairs) -> tuple:
    n = len(pairs)
    if n == 0:
        return (0.0, 0.0, 0.0)
    base = [en - cl for cl, en in pairs]
    ds = sorted(sum(base[random.randrange(n)] for _ in range(n)) / n * 100
                for _ in range(BOOT))
    return round(sum(base) / n * 100, 1), round(ds[int(0.025 * BOOT)], 1), round(ds[int(0.975 * BOOT)], 1)


def _mcnemar(a: dict, b: dict) -> tuple:
    """Exact two-sided McNemar on paired 0/1 dicts. Returns (b_only, c_only, p)."""
    cases = sorted(set(a) & set(b))
    bb = sum(1 for x in cases if a[x] == 1 and b[x] == 0)
    cc = sum(1 for x in cases if a[x] == 0 and b[x] == 1)
    n = bb + cc
    if n == 0:
        return bb, cc, 1.0
    k = min(bb, cc)
    p = min(1.0, sum(comb(n, i) for i in range(k + 1)) / (2 ** n) * 2)
    return bb, cc, p


# ---------------------------------------------------------------------------
# 1. Runtime results table  (tab:runtime-results)
# ---------------------------------------------------------------------------
def runtime_results():
    print("=" * 74)
    print("1. RUNTIME RESULTS (tab:runtime-results) — R@1 / R@5, clean->enriched, delta")
    print("=" * 74)
    for mode in ("skill_adjusted", "dynamic"):
        print(f"\n[{mode}]  n = {20 if mode == 'skill_adjusted' else 40}")
        print(f"  {'Method':14} {'R@1 c->e':>12} {'R@5 c->e':>12} {'dR@1':>6}")
        for m in ORDER:
            p0 = _pct(list(_load(mode, _variant(mode, m, 0), "top1_acc").values()))
            p1 = _pct(list(_load(mode, _variant(mode, m, 1), "top1_acc").values()))
            r0 = _pct(list(_load(mode, _variant(mode, m, 0), "candidate_recall").values()))
            r1 = _pct(list(_load(mode, _variant(mode, m, 1), "candidate_recall").values()))
            print(f"  {NAMES[m]:14} {f'{p0}->{p1}':>12} {f'{r0}->{r1}':>12} {p1-p0:+6.1f}")


# ---------------------------------------------------------------------------
# 2. Out-of-scope table  (tab:oos-results)
# ---------------------------------------------------------------------------
def oos_results():
    print("\n" + "=" * 74)
    print("2. OUT-OF-SCOPE (tab:oos-results) — refusal / false-pick, clean->enriched")
    print("=" * 74)
    print(f"  {'Method':14} {'refuse c->e':>14} {'false-pick c->e':>16}")
    for m in ORDER:
        ref0 = _pct([1 if v else 0 for v in _load('out_of_scope', f'{m}0', 'correctly_refused').values()])
        ref1 = _pct([1 if v else 0 for v in _load('out_of_scope', f'{m}1', 'correctly_refused').values()])
        fp0 = _pct([1 if v else 0 for v in _load('out_of_scope', f'{m}0', 'falsely_picked').values()])
        fp1 = _pct([1 if v else 0 for v in _load('out_of_scope', f'{m}1', 'falsely_picked').values()])
        print(f"  {NAMES[m]:14} {f'{ref0}->{ref1}':>14} {f'{fp0}->{fp1}':>16}")


# ---------------------------------------------------------------------------
# 3. Funnel figure  (fig:funnel)
# ---------------------------------------------------------------------------
def funnel():
    print("\n" + "=" * 74)
    print("3. FUNNEL (fig:funnel) — routing accuracy + best-method resolution")
    print("=" * 74)
    # routing accuracy per mode (enrichment-independent)
    def routing(mode):
        if mode == "skill_guided":
            files = glob.glob(os.path.join(RT, "skill_guided", "routing", "sg-*.json"))
            acc = [1 if json.load(open(f)).get("routing_ok") else 0
                   for f in files if json.load(open(f)).get("routing_ok") is not None]
            return _pct(acc), len(acc)
        dd = os.path.join(RT, mode, "traces", f"A1")
        if not os.path.isdir(dd):
            dd = os.path.join(RT, mode, "traces", "A0")
        vals = [1 if json.load(open(f))["metrics"].get("mode_routing_ok") else 0
                for f in glob.glob(os.path.join(dd, "*.json"))
                if json.load(open(f))["metrics"].get("mode_routing_ok") is not None]
        return _pct(vals), len(vals)
    for mode in ("skill_guided", "skill_adjusted", "dynamic", "out_of_scope"):
        acc, n = routing(mode)
        print(f"  routing  {mode:16} = {acc:5.1f}%  (n={n})")
    # best-method resolution used in the funnel white/grey bars
    print("  resolution (best method per mode): SA=Embedding R@1, DY=Hybrid R@1, OOS=Raw refusal")
    for mode, m, metric in (("skill_adjusted", "A", "top1_acc"),
                            ("dynamic", "F", "top1_acc"),
                            ("out_of_scope", "E", "correctly_refused")):
        c0 = _pct([1 if v else 0 for v in _load(mode, _variant(mode, m, 0), metric).values()])
        c1 = _pct([1 if v else 0 for v in _load(mode, _variant(mode, m, 1), metric).values()])
        print(f"    {mode:16} {NAMES[m]:14} clean={c0:5.1f}  enriched={c1:5.1f}")


# ---------------------------------------------------------------------------
# 4. gap / pareto / token-efficiency  (Dynamic enriched)
# ---------------------------------------------------------------------------
def cost_figures():
    print("\n" + "=" * 74)
    print("4. gap / pareto / token-eff (Dynamic enriched) — R@1, R@5, tokens, R@1/1k")
    print("=" * 74)
    print(f"  {'Method':14} {'R@1':>6} {'R@5':>6} {'tokens':>8} {'R@1/1k':>7}")
    for m in ORDER:
        cond = _variant("dynamic", m, 1)
        p1 = _pct(list(_load("dynamic", cond, "top1_acc").values()))
        r5 = _pct(list(_load("dynamic", cond, "candidate_recall").values()))
        # tokens: mean per-case from the trace 'trace.tokens'
        toks = []
        for fp in glob.glob(os.path.join(RT, "dynamic", "traces", cond, "*.json")):
            tr = json.load(open(fp)).get("trace", {})
            if isinstance(tr, dict) and tr.get("tokens"):
                toks.append(tr["tokens"])
        mtok = round(sum(toks) / len(toks)) if toks else 0
        eff = round(p1 / (mtok / 1000), 1) if mtok else 0.0
        print(f"  {NAMES[m]:14} {p1:6.1f} {r5:6.1f} {mtok:8} {eff:7.1f}")


# ---------------------------------------------------------------------------
# 5. Failure-mode breakdown  (tab:failure-modes)
# ---------------------------------------------------------------------------
def failure_modes():
    print("\n" + "=" * 74)
    print("5. FAILURE MODES (tab:failure-modes) — Dynamic enriched, % of cases")
    print("=" * 74)
    print(f"  {'Method':14} {'correct':>7} {'rank_err':>8} {'retr_miss':>9} {'empty':>6}")
    for m in ORDER:
        cond = _variant("dynamic", m, 1)
        n = cor = re = rm = em = 0
        for fp in glob.glob(os.path.join(RT, "dynamic", "traces", cond, "*.json")):
            t = json.load(open(fp)); me = t["metrics"]; n += 1
            r, t1 = me.get("candidate_recall"), me.get("top1_acc")
            if t1 == 1:
                cor += 1
            elif r == 1:
                re += 1
            else:
                nc = sum(len(s.get("candidates", [])) for s in t.get("plan", {}).get("steps", []))
                em += 1 if nc == 0 else 0
                rm += 0 if nc == 0 else 1
        f = lambda x: round(x / n * 100) if n else 0
        print(f"  {NAMES[m]:14} {f(cor):7} {f(re):8} {f(rm):9} {f(em):6}")


# ---------------------------------------------------------------------------
# 6. Statistical reliability  (bootstrap CIs + McNemar)
# ---------------------------------------------------------------------------
def statistics():
    print("\n" + "=" * 74)
    print("6. STATISTICAL RELIABILITY (Dynamic enriched, n=40)")
    print("=" * 74)
    print("  Absolute R@1 with 95% bootstrap CI:")
    for m in ORDER:
        cond = _variant("dynamic", m, 1)
        p1, lo, hi = _boot_ci(list(_load("dynamic", cond, "top1_acc").values()))
        print(f"    {NAMES[m]:14} R@1={p1:5.1f}  [{lo:5.1f}, {hi:5.1f}]")
    print("  Enrichment delta with 95% paired bootstrap CI (Dynamic):")
    for m in ORDER:
        d, lo, hi = _boot_delta_ci(_paired("dynamic", m))
        star = "  *CI excludes 0" if not (lo <= 0 <= hi) else ""
        print(f"    {NAMES[m]:14} dR@1={d:+5.1f}  [{lo:+5.1f}, {hi:+5.1f}]{star}")
    print("  Paired McNemar p-values (Dynamic enriched, top-1):")
    L = lambda m: _load("dynamic", _variant("dynamic", m, 1), "top1_acc")
    for a, b in (("F", "S"), ("D", "S"), ("A", "S"),
                 ("F", "E"), ("D", "E"), ("F", "D"), ("F", "A"), ("D", "A")):
        bb, cc, p = _mcnemar(L(a), L(b))
        sig = "  SIG" if p < 0.05 else ""
        print(f"    {NAMES[a]:14} vs {NAMES[b]:14} b/c={bb}/{cc}  p={p:.4f}{sig}")


# ---------------------------------------------------------------------------
# 7. Design-time appendix table  (tab:full-results-dt)
# ---------------------------------------------------------------------------
def design_time():
    print("\n" + "=" * 74)
    print("7. DESIGN-TIME (tab:full-results-dt, n=240) — R@1 / R@5 with 95% CI")
    print("=" * 74)
    print(f"  {'Method':14} {'R@1 [CI]':>22} {'R@5 [CI]':>22}")
    for m in ORDER:
        d = os.path.join(DT, "traces", m)
        if not os.path.isdir(d):
            continue
        p, r = [], []
        for fp in glob.glob(os.path.join(d, "*.json")):
            me = json.load(open(fp))["metrics"]
            p.append(1 if me.get("p_at_1") else 0)
            r.append(1 if me.get("r_at_5") else 0)
        p1, pl, ph = _boot_ci(p)
        r5, rl, rh = _boot_ci(r)
        print(f"  {NAMES[m]:14} {f'{p1} [{pl},{ph}]':>22} {f'{r5} [{rl},{rh}]':>22}")


if __name__ == "__main__":
    print("Reproducing Paper II (Semantic Retrieval) numbers from raw traces.")
    print(f"Repo root: {ROOT}\nSeed: 42, bootstrap resamples: {BOOT}\n")
    runtime_results()
    oos_results()
    funnel()
    cost_figures()
    failure_modes()
    statistics()
    design_time()
    print("\nDone. Compare against the values in paper/conference/orchestration/.")
