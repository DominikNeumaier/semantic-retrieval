"""Ambiguity profile of retrieval candidates (Paper II, Discussion).

Beyond R@1/R@5 (which say how often a method is right) and the failure-mode
split (which only redistributes the R@5-minus-R@1 gap), this script asks a
sharper question tied to the benchmark's core construct: when a retrieval
method places a resource in its Top-5, how ambiguous is that resource relative
to the ground-truth target?

Two analyses over the retrieval-driven modes (Skill-Adjusted + Dynamic):

  A. Distractor ambiguity profile.
     For every WRONG Top-5 candidate (candidate != target), classify its
     structural ambiguity tier to the target (HIGH / MEDIUM / LOW / NONE) and
     report the per-method distribution, separately for clean and enriched ORD.
     Reveals whether a method's mistakes are genuine HIGH-tier confusables
     (hard, legitimately similar) or unrelated NONE noise, and whether
     enrichment shifts that profile.

  B. False attractors.
     Across ALL traces, which non-target resources appear in the Top-5 most
     often? These are the landscape's chronic distractors. Reported with each
     attractor's own landscape ambiguity_score, to check whether the metric
     anticipated them.

Tiers use the benchmark's canonical structural metric (clean, six-field) from
ord-bench's landscape_ambiguity_report.json, so clean and enriched candidates
are graded on ONE fixed difficulty scale. Difficulty is a fixed property of the
landscape; the question is which tier of distractor each method pulls in.

USAGE
  python3 analysis/ambiguity_profile.py
  (run from the repo root; no API calls; reads traces + the ambiguity report)

The ambiguity report is located via, in order:
  1. $ORD_BENCH_DIR/data/ambiguity/landscape_ambiguity_report.json
  2. ../ord-bench/data/ambiguity/landscape_ambiguity_report.json  (sibling repo)
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RT = os.path.join(ROOT, "results", "runtime")

NAMES = {"S": "Baseline", "A": "Embedding", "B": "Progressive",
         "C": "Graph", "D": "Agentic Tools", "E": "Agentic Raw",
         "F": "Agentic Hybrid"}
METHOD_ORDER = ["S", "A", "B", "C", "D", "E", "F"]
TIERS = ["HIGH", "MEDIUM", "LOW", "NONE"]


# ---------------------------------------------------------------------------
# ambiguity report: pairwise structural tier lookup
# ---------------------------------------------------------------------------
def _find_report() -> str:
    cands = []
    env = os.environ.get("ORD_BENCH_DIR")
    if env:
        cands.append(os.path.join(env, "data", "ambiguity",
                                  "landscape_ambiguity_report.json"))
    cands.append(os.path.join(ROOT, "..", "ord-bench", "data", "ambiguity",
                              "landscape_ambiguity_report.json"))
    for c in cands:
        if os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "landscape_ambiguity_report.json not found. Set ORD_BENCH_DIR or place "
        "the ord-bench repo as a sibling of this one.")


def _load_tiers() -> tuple[dict, dict, dict]:
    """Return (pair_sim, score_by_id, thresholds).

    pair_sim[a][b] = structural similarity of the pair (symmetric, only pairs
    with sim >= low_lo are listed; anything absent is NONE).
    """
    rep = json.load(open(_find_report()))
    th = rep["thresholds"]
    pair_sim: dict = defaultdict(dict)
    score_by_id: dict = {}
    for res in rep["resources"]:
        rid = res["ordId"]
        score_by_id[rid] = res.get("ambiguity_score")
        for nb in res.get("all_neighbors", []):
            pair_sim[rid][nb["ordId"]] = nb["sim"]
    return pair_sim, score_by_id, th


def _tier(sim: float | None, th: dict) -> str:
    if sim is None or sim < th["low_lo"]:
        return "NONE"
    if sim < th["medium_lo"]:
        return "LOW"
    if sim < th["high_lo"]:
        return "MEDIUM"
    return "HIGH"


# ---------------------------------------------------------------------------
# trace normalisation: (method, state, target, top5) records
# ---------------------------------------------------------------------------
def _iter_dy_single(m: str, state_digit: str):
    """Dynamic single-intent: <M><state>f, target in step.expected."""
    for fp in glob.glob(os.path.join(RT, "dynamic", "traces", f"{m}{state_digit}f", "*.json")):
        t = json.load(open(fp))
        for st in t.get("plan", {}).get("steps", []):
            exp = (st.get("expected") or {}).get("expected_ordIds") or []
            cands = [c["ordId"] for c in st.get("candidates", [])]
            if exp and cands:
                yield exp[0], cands


def _iter_dy_multi(m: str, state_digit: str):
    """Dynamic multi-intent: <M><state>mh, one file per sub-query, gt_id +
    top5_candidates (flat list of ordId strings)."""
    for fp in glob.glob(os.path.join(RT, "dynamic", "traces", f"{m}{state_digit}mh", "*.json")):
        t = json.load(open(fp))
        gt = t.get("gt_id")
        cands = t.get("top5_candidates") or []
        if gt and cands:
            yield gt, cands


def _iter_sa(m: str, state_digit: str):
    """Skill-Adjusted gap-forced: <M><state>gf (or ...f for Hybrid). Targets are
    the top-level expected_gaps, aligned by step index to plan.steps."""
    base = os.path.join(RT, "skill_adjusted", "traces")
    for suffix in ("gf", "f"):
        d = os.path.join(base, f"{m}{state_digit}{suffix}")
        if not os.path.isdir(d):
            continue
        for fp in glob.glob(os.path.join(d, "*.json")):
            t = json.load(open(fp))
            gaps = t.get("expected_gaps") or []
            steps = t.get("plan", {}).get("steps", [])
            for i, st in enumerate(steps):
                if i >= len(gaps):
                    break
                cands = [c["ordId"] for c in st.get("candidates", [])]
                if gaps[i] and cands:
                    yield gaps[i], cands
        break  # only one of gf/f exists per method


def iter_records(m: str, state_digit: str):
    """All (target, top5) retrieval records for a method+state across SA+DY."""
    yield from _iter_dy_single(m, state_digit)
    yield from _iter_dy_multi(m, state_digit)
    yield from _iter_sa(m, state_digit)


# ---------------------------------------------------------------------------
# A. distractor ambiguity profile (wrong Top-5)
# ---------------------------------------------------------------------------
REL_TIERS = ["HIGH", "MEDIUM", "LOW"]  # NONE excluded from the split, see below


def analysis_a(pair_sim: dict, th: dict) -> None:
    """For every WRONG Top-5 candidate (candidate != target), measure how
    structurally similar it is to the true target, per method, clean vs enriched.

    Two readings of the same wrong-Top-5 set:
      * abar  -- mean similarity of ALL wrong picks to the target (NONE=0). One
                 number for how target-like a method's mistakes are.
      * High/Med/Low -- of the wrong picks that are genuinely RELATED (sim >=
                 low_lo), how the related mass splits over the three tiers.
                 Unrelated NONE picks are excluded here and the mass
                 renormalised, so the split asks "WHEN a pick is confusable,
                 how confusable" rather than being dominated by random noise.
    n is the count of related (non-NONE) distractors behind the split.
    """
    print("=" * 74)
    print("A. DISTRACTOR AMBIGUITY PROFILE (SA + DY) — wrong Top-5 picks, how")
    print("   similar they are to the target (abar over all; H/M/L over related)")
    print("=" * 74)
    dist: dict = {"clean": {}, "enriched": {}}
    n_rel: dict = {"clean": {}, "enriched": {}}
    abar: dict = {"clean": {}, "enriched": {}}  # mean sim over ALL wrong (NONE=0)
    for state_digit, state in (("0", "clean"), ("1", "enriched")):
        for m in METHOD_ORDER:
            c: Counter = Counter()
            all_sims: list = []
            for target, top5 in iter_records(m, state_digit):
                for cand in top5:
                    if cand == target:
                        continue  # count only distractors
                    sim = pair_sim.get(target, {}).get(cand)
                    all_sims.append(sim if sim is not None else 0.0)
                    tier = _tier(sim, th)
                    if tier != "NONE":
                        c[tier] += 1  # related distractors only
            dist[state][m] = c
            n_rel[state][m] = sum(c.values())
            abar[state][m] = (sum(all_sims) / len(all_sims)) if all_sims else 0.0

    for state in ("clean", "enriched"):
        print(f"\n[{state} ORD]")
        print(f"  {'Method':16} {'abar':>7} {'HIGH':>6} {'MED':>6} {'LOW':>6}   n(related)")
        for m in METHOD_ORDER:
            c = dist[state][m]
            tot = n_rel[state][m] or 1
            print(f"  {NAMES[m]:16} {abar[state][m]:7.3f} "
                  f"{100*c['HIGH']/tot:6.1f} {100*c['MEDIUM']/tot:6.1f} "
                  f"{100*c['LOW']/tot:6.1f}   {n_rel[state][m]}")

    _latex_a(dist, n_rel, abar)


def _latex_a(dist: dict, n_rel: dict, abar: dict) -> None:
    print("\n--- LaTeX (Table 6: distractor ambiguity profile, wrong Top-5) ---")
    print(r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}l r rrr r@{}}")
    print(r"\toprule")
    print(r"\textbf{Method} & $\bar{a}$ & High & Med & Low & $n$ \\")
    print(r"\midrule")
    for state in ("clean", "enriched"):
        label = "Clean ORD" if state == "clean" else "Enriched ORD"
        print(r"\addlinespace[1pt]")
        print(r"\multicolumn{6}{@{}l}{\emph{" + label + r"}} \\")
        for m in METHOD_ORDER:
            c = dist[state][m]
            tot = n_rel[state][m] or 1
            trip = " & ".join(f"{100*c[t]/tot:.1f}" for t in REL_TIERS)
            print(f"\\quad {NAMES[m]} & {abar[state][m]:.3f} & {trip} & {n_rel[state][m]} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular*}")


# ---------------------------------------------------------------------------
# B. false attractors (paired clean vs enriched)
# ---------------------------------------------------------------------------
def _short(rid: str) -> str:
    """Human-readable local name for a figure label (last path segment, no :vN)."""
    parts = rid.split(":")
    return parts[-2] if len(parts) >= 2 else rid


def analysis_b(score_by_id: dict, top_n: int = 10) -> None:
    print("\n" + "=" * 74)
    print(f"B. FALSE ATTRACTORS (SA + DY) — non-target resources most often placed")
    print(f"   in Top-5, clean vs enriched ORD side by side")
    print("=" * 74)
    appear = {"clean": Counter(), "enriched": Counter()}
    for state_digit, state in (("0", "clean"), ("1", "enriched")):
        for m in METHOD_ORDER:
            for target, top5 in iter_records(m, state_digit):
                for cand in top5:
                    if cand != target:
                        appear[state][cand] += 1

    combined = appear["clean"] + appear["enriched"]
    top = [rid for rid, _ in combined.most_common(top_n)]

    print(f"\n  {'#':>3} {'clean':>6} {'enr':>6} {'delta':>6} {'score':>6}  ordId")
    for i, rid in enumerate(top, 1):
        cl, en = appear["clean"][rid], appear["enriched"][rid]
        sc = score_by_id.get(rid)
        sc_s = f"{sc:.3f}" if sc is not None else " n/a"
        print(f"  {i:>3} {cl:>6} {en:>6} {en - cl:>+6} {sc_s:>6}  {rid}")

    ac, ae = sum(appear["clean"].values()), sum(appear["enriched"].values())
    print(f"\n  total distractor appearances: clean={ac}  enriched={ae}  "
          f"({100 * (ae - ac) / ac:+.1f}%)")
    tc = {r for r, _ in appear["clean"].most_common(top_n)}
    te = {r for r, _ in appear["enriched"].most_common(top_n)}
    print(f"  Top-{top_n} membership overlap clean vs enriched: {len(tc & te)}/{top_n}")

    _latex_b(appear, score_by_id, top)


def _latex_b(appear: dict, score_by_id: dict, top: list) -> None:
    print("\n--- pgfplots (Figure B: false attractors, clean vs enriched) ---")
    # symbolic y coords, most frequent at top
    labels = [_short(r) for r in top]
    coords_cl = " ".join(f"({appear['clean'][r]},{lab})" for r, lab in zip(top, labels))
    coords_en = " ".join(f"({appear['enriched'][r]},{lab})" for r, lab in zip(top, labels))
    symbolic = ",".join(reversed(labels))  # reversed -> rank 1 on top
    print(r"\begin{tikzpicture}")
    print(r"  \begin{axis}[")
    print(r"    xbar, y=15pt, bar width=5pt, bar shift=0pt,")
    print(r"    enlarge y limits=0.06, xmin=0,")
    print(r"    width=\linewidth, height=0.42\textheight,")
    print(f"    symbolic y coords={{{symbolic}}},")
    print(r"    ytick=data, yticklabel style={font=\scriptsize},")
    print(r"    xlabel={Top-5 appearances as a non-target (SA + DY)},")
    print(r"    xlabel style={font=\scriptsize}, xticklabel style={font=\scriptsize},")
    print(r"    legend style={at={(0.98,0.04)}, anchor=south east, font=\scriptsize, draw=none},")
    print(r"    nodes near coords, nodes near coords style={font=\tiny},")
    print(r"  ]")
    print(r"    \addplot+[fill=black!25, draw=black!55, bar shift=-3pt] coordinates {" + coords_cl + "};")
    print(r"    \addplot+[fill=white, draw=black!55, bar shift=3pt] coordinates {" + coords_en + "};")
    print(r"    \legend{Clean ORD, Enriched ORD}")
    print(r"  \end{axis}")
    print(r"\end{tikzpicture}")


# ---------------------------------------------------------------------------
# C. method complementarity (per-target solve, union, unique solves)
# ---------------------------------------------------------------------------
def _solved_targets(state_digit: str, mode: str = "r5") -> tuple:
    """solved[m] = set of targets the method retrieves, for the given ORD state.

    mode='r5': target appears anywhere in the method's Top-5 (recall).
    mode='r1': target is ranked first.
    A target counts as solved if the method succeeds in at least one of its
    records for that target.
    """
    solved: dict = {m: set() for m in METHOD_ORDER}
    targets: set = set()
    for m in METHOD_ORDER:
        for target, top5 in iter_records(m, state_digit):
            targets.add(target)
            hit = (top5 and top5[0] == target) if mode == "r1" else (target in top5)
            if hit:
                solved[m].add(target)
    return solved, targets


def analysis_c(mode: str = "r5") -> None:
    """Do the methods retrieve the SAME targets, or complementary ones? Per ORD
    state, compares each method's solve set (Top-5 recall) to the union over all
    methods. 'headroom' is union minus best single method, the ceiling a perfect
    per-query router would add. 'unique' counts targets only ONE method retrieves.
    'nobody' is the residue no method reaches, a landscape-inherent hard floor.
    """
    metric = "R@1" if mode == "r1" else "R@5"
    print("\n" + "=" * 74)
    print(f"C. METHOD COMPLEMENTARITY (SA + DY) — {metric} solve sets, clean vs enriched")
    print("=" * 74)
    rows = {}
    for state_digit, state in (("0", "clean"), ("1", "enriched")):
        solved, targets = _solved_targets(state_digit, mode)
        T = len(targets)
        union = set().union(*solved.values())
        cnt = Counter()
        for m in METHOD_ORDER:
            for t in solved[m]:
                cnt[t] += 1
        uniques = {t for t, k in cnt.items() if k == 1}
        best = max(METHOD_ORDER, key=lambda m: len(solved[m]))
        rows[state] = {
            "T": T, "best_m": NAMES[best], "best": len(solved[best]),
            "union": len(union), "headroom": len(union) - len(solved[best]),
            "one": len(uniques), "nobody": T - len(union),
        }
        print(f"\n[{state} ORD]  cases (distinct targets) = {T}")
        for m in METHOD_ORDER:
            print(f"  {NAMES[m]:16} solved={len(solved[m]):3d}  "
                  f"unique={len(solved[m] & uniques)}")
        r = rows[state]
        print(f"  best single = {r['best_m']} {r['best']}/{T}   "
              f"union = {r['union']}/{T}   headroom = +{r['headroom']}")
        print(f"  solved by exactly one = {r['one']}   solved by nobody = {r['nobody']}")

    # LaTeX: quantities as rows, clean/enriched as columns
    print(f"\n--- LaTeX (Table: method complementarity, {metric}) ---")
    print(r"\begin{tabular}{@{}l cc@{}}")
    print(r"\toprule")
    print(r"& Clean ORD & Enriched ORD \\")
    print(r"\midrule")
    c, e = rows["clean"], rows["enriched"]
    print(f"Cases (distinct targets)          & {c['T']} & {e['T']} \\\\")
    print(f"Best single method                & {c['best']} & {e['best']} \\\\")
    print(f"Union of all seven                & {c['union']} & {e['union']} \\\\")
    print(f"Ensemble headroom                 & $+{c['headroom']}$ & $+{e['headroom']}$ \\\\")
    print(f"Solved by exactly one method      & {c['one']} & {e['one']} \\\\")
    print(f"Solved by no method               & {c['nobody']} & {e['nobody']} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def main() -> None:
    print("Ambiguity profile of retrieval candidates (Paper II).")
    print(f"Repo root: {ROOT}")
    print(f"Ambiguity report: {_find_report()}\n")
    pair_sim, score_by_id, th = _load_tiers()
    print(f"Tier bands: HIGH [{th['high_lo']},{th['high_hi']}]  "
          f"MEDIUM [{th['medium_lo']},{th['high_lo']})  "
          f"LOW [{th['low_lo']},{th['medium_lo']})  NONE <{th['low_lo']}\n")
    analysis_a(pair_sim, th)
    analysis_b(score_by_id)
    analysis_c("r5")


if __name__ == "__main__":
    main()
