"""Reproduce the headline Paper II tables from committed result traces.

This script is read-only and uses the published, paper-referenced result tree:
``results/retrieval`` and ``results/orchestration``.  It fails loudly when a
required condition is missing instead of silently printing zeros.

Use ``analysis/statistics.py`` for bootstrap confidence intervals and McNemar
p-values, and ``analysis/figures.py`` for the cost/efficiency figure values.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL = ROOT / "results" / "retrieval"
ORCHESTRATION = ROOT / "results" / "orchestration"
RT = RETRIEVAL / "runtime"
DT = RETRIEVAL / "design-time"

NAMES = {
    "S": "Baseline",
    "A": "Embedding",
    "B": "Progressive",
    "C": "Graph",
    "D": "Agentic Tools",
    "E": "Agentic Raw",
    "F": "Agentic Hybrid",
}
ORDER = ["S", "A", "B", "C", "D", "E", "F"]


def _hround(value: float, digits: int = 1) -> float:
    factor = 10 ** digits
    return math.floor(value * factor + 0.5 + 1e-9) / factor


def _pct(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute a percentage from an empty result set")
    return _hround(sum(values) / len(values) * 100, 1)


def _load_dir(path: Path) -> list[dict]:
    if not path.is_dir():
        raise FileNotFoundError(f"missing published trace directory: {path}")
    rows = []
    for file in sorted(path.glob("*.json")):
        try:
            rows.append(json.loads(file.read_text()))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON trace: {file}") from exc
    if not rows:
        raise RuntimeError(f"no JSON traces found in {path}")
    return rows


def _condition_dir(mode: str, method: str, state: int) -> Path:
    return RT / mode / "traces" / f"{method}{state}"


def dynamic_per_case(method: str, state: int, metric: str) -> dict[str, float]:
    """Return one score per Dynamic case, averaging multi-intent sub-queries."""
    singles: dict[str, float] = {}
    subs: dict[str, list[float]] = defaultdict(list)
    for trace in _load_dir(_condition_dir("dynamic", method, state)):
        case_id = trace["case_id"]
        if "sub_idx" in trace:
            subs[case_id].append(float(trace.get(metric) or 0))
        else:
            singles[case_id] = float(trace.get("metrics", {}).get(metric) or 0)
    out = dict(singles)
    out.update({case_id: sum(values) / len(values) for case_id, values in subs.items()})
    if len(out) != 40:
        raise AssertionError(f"Dynamic {method}{state}: expected 40 cases, got {len(out)}")
    return out


def skill_adjusted_per_case(method: str, state: int, top_k: int) -> dict[str, float]:
    """Score the fraction of expected gap resources retrieved per SA case."""
    out: dict[str, float] = {}
    for trace in _load_dir(_condition_dir("skill_adjusted", method, state)):
        expected = set(trace.get("expected_gaps", []))
        if not expected:
            raise ValueError(f"{trace.get('case_id')}: missing expected_gaps")
        gap_steps = [
            step for step in trace.get("plan", {}).get("steps", [])
            if step.get("source") in {"gap", "gap_forced"}
        ]
        hits = 0
        for expected_id in expected:
            if any(
                any(c.get("ordId") == expected_id for c in step.get("candidates", [])[:top_k])
                for step in gap_steps
            ):
                hits += 1
        out[trace["case_id"]] = hits / len(expected)
    if len(out) != 20:
        raise AssertionError(
            f"Skill-Adjusted {method}{state}: expected 20 cases, got {len(out)}"
        )
    return out


def oos_per_case(method: str, state: int, metric: str) -> dict[str, float]:
    out = {
        trace["case_id"]: float(trace.get("metrics", {}).get(metric) or 0)
        for trace in _load_dir(_condition_dir("out_of_scope", method, state))
    }
    if len(out) != 20:
        raise AssertionError(f"OOS {method}{state}: expected 20 cases, got {len(out)}")
    return out


def _orchestration_dir(mode: str, condition: str) -> Path:
    direct = ORCHESTRATION / mode / condition
    nested = ORCHESTRATION / mode / "traces" / condition
    return direct if direct.is_dir() else nested


def routing_accuracy(mode: str) -> tuple[float, int]:
    if mode == "skill_guided":
        summary_path = ORCHESTRATION / mode / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"missing Skill-Guided summary: {summary_path}")
        rows = json.loads(summary_path.read_text())
        if not rows:
            raise RuntimeError(f"empty Skill-Guided summary: {summary_path}")
        return float(rows[0]["Routing-Acc"]) * 100, int(rows[0]["cases"])
    rows = _load_dir(_orchestration_dir(mode, "A0"))
    values = [float(row.get("metrics", {}).get("mode_routing_ok") or 0) for row in rows]
    expected = 20 if mode == "skill_adjusted" else 40
    if len(values) != expected:
        raise AssertionError(f"{mode}: expected {expected} routing cases, got {len(values)}")
    return _pct(values), len(values)


def runtime_results() -> None:
    print("=" * 78)
    print("RUNTIME RETRIEVAL (Table 3): R@1 / R@5, Clean-ORD -> Enriched-ORD")
    print("=" * 78)
    for method in ORDER:
        sa1 = [_pct(list(skill_adjusted_per_case(method, state, 1).values())) for state in (0, 1)]
        sa5 = [_pct(list(skill_adjusted_per_case(method, state, 5).values())) for state in (0, 1)]
        dy1 = [_pct(list(dynamic_per_case(method, state, "top1_acc").values())) for state in (0, 1)]
        dy5 = [_pct(list(dynamic_per_case(method, state, "candidate_recall").values())) for state in (0, 1)]
        print(
            f"{NAMES[method]:14}  SA R@1 {sa1[0]:4.1f}->{sa1[1]:4.1f} "
            f"R@5 {sa5[0]:4.1f}->{sa5[1]:4.1f} | "
            f"DY R@1 {dy1[0]:4.1f}->{dy1[1]:4.1f} "
            f"R@5 {dy5[0]:4.1f}->{dy5[1]:4.1f}"
        )


def oos_results() -> None:
    print("\n" + "=" * 78)
    print("OUT-OF-SCOPE (Table 4): refusal / false-pick")
    print("=" * 78)
    for method in ORDER:
        refusal = [_pct(list(oos_per_case(method, state, "correctly_refused").values())) for state in (0, 1)]
        false_pick = [_pct(list(oos_per_case(method, state, "falsely_picked").values())) for state in (0, 1)]
        print(
            f"{NAMES[method]:14} refusal {refusal[0]:4.1f}->{refusal[1]:4.1f} "
            f"false-pick {false_pick[0]:4.1f}->{false_pick[1]:4.1f}"
        )


def routing_results() -> None:
    print("\n" + "=" * 78)
    print("ORCHESTRATION ROUTING (Figure 4)")
    print("=" * 78)
    totals = []
    for mode in ("skill_guided", "skill_adjusted", "dynamic"):
        accuracy, count = routing_accuracy(mode)
        totals.append((accuracy, count))
        print(f"{mode:16} {accuracy:5.1f}% (n={count})")
    weighted = sum(acc * count for acc, count in totals) / sum(count for _, count in totals)
    print(f"case-weighted     {_hround(weighted, 1):5.1f}% (n={sum(c for _, c in totals)})")
    skill_summary = json.loads((RT / "skill_guided" / "traces" / "summary.json").read_text())
    print(f"skill selection   {float(skill_summary['Routing-Acc']) * 100:5.1f}% (n={skill_summary['cases']})")


def design_time_results() -> None:
    summary_path = DT / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing design-time summary: {summary_path}")
    rows = json.loads(summary_path.read_text())
    by_method = {row["method"]: row for row in rows}
    missing = set(ORDER) - set(by_method)
    if missing:
        raise AssertionError(f"design-time summary missing methods: {sorted(missing)}")
    print("\n" + "=" * 78)
    print("DESIGN-TIME (Appendix Table 10)")
    print("=" * 78)
    for method in ORDER:
        row = by_method[method]
        if int(row["cases"]) != 240:
            raise AssertionError(f"Design-Time {method}: expected 240 cases")
        print(f"{NAMES[method]:14} R@1={row['P@1'] * 100:5.1f} R@5={row['R@5'] * 100:5.1f}")


def main() -> None:
    print("Reproducing Paper II headline values from committed traces.\n")
    runtime_results()
    oos_results()
    routing_results()
    design_time_results()
    print("\nFor confidence intervals and p-values: python3 analysis/statistics.py")
    print("For cost and token-efficiency values: python3 analysis/figures.py")


if __name__ == "__main__":
    main()
