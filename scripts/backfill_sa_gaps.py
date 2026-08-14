"""One-off backfill: fill the 17 deficient Skill-Adjusted gap-retrieval cases so
every retrieval method has 20 gap-detection+retrieval cases, independent of routing.

Groups:
  G1 native "gap": F0 cases where A0 has genuine planner gap steps -> reuse A0
      labels, source stays "gap", condition "F0" (matches existing F0 siblings).
  G2 "gap_forced": cases where A0 lacks genuine gaps (sa-07 full, sa-15 forced) ->
      inject canonical forced labels, relabel source -> "gap_forced",
      condition base+"gf" (matches all other methods' *gf files).

Writes ONLY the 17 trace files under results/retrieval/runtime/skill_adjusted/traces.
Does not touch records.jsonl. Cleans up the stray results/runtime side-effect dir.
"""
import json, os, shutil
from pathlib import Path
from src import config
from src import loader as ord_loader
from src.runtime import skill_registry
from src.eval.runtime import _load_cases, run_case

RET = config.ROOT / "results" / "retrieval" / "runtime" / "skill_adjusted" / "traces"
A0DIR = RET / "A0"

FORCED_LABELS = {
    "sa-07": ["Hazardous Substance Compliance API", "Payroll Processing Data Management"],
    "sa-15": ["Operational Capability Assessment for Product Transitions",
              "Organizational Readiness Assessment for Product Transitions"],
}

def a0_gap_labels(cid):
    j = json.load(open(A0DIR / f"{cid}.json"))
    return [s["step_name"] for s in (j.get("plan") or {}).get("steps", [])
            if s.get("source") == "gap"]

# (method, state, case_id, group) — group: "gap" (G1) or "gap_forced" (G2)
G1_F0 = ["sa-01","sa-02","sa-03","sa-04","sa-05","sa-06","sa-08","sa-09","sa-10","sa-11","sa-12","sa-13"]
TARGETS = [("F","clean",c,"gap") for c in G1_F0] + [
    ("A","clean","sa-07","gap_forced"),
    ("F","clean","sa-07","gap_forced"),
    ("F","enriched","sa-07","gap_forced"),
    ("F","clean","sa-15","gap_forced"),
    ("F","enriched","sa-15","gap_forced"),
]

def main():
    cases = {c["case_id"]: c for c in _load_cases("skill_adjusted")}
    skills = skill_registry.load_skills()
    res = {st: ord_loader.load_landscape(state=st) for st in ("clean","enriched")}
    print(f"cases={len(cases)} skills={len(skills)} "
          f"resources clean={len(res['clean'])} enriched={len(res['enriched'])}")
    print(f"backfilling {len(TARGETS)} cases\n")

    for method, state, cid, group in TARGETS:
        case = cases[cid]
        labels = FORCED_LABELS[cid] if group == "gap_forced" else a0_gap_labels(cid)
        base = f"{method}{'1' if state=='enriched' else '0'}"
        print(f"[{base}{'gf' if group=='gap_forced' else ''}] {cid}  "
              f"labels={labels}")
        rec = run_case(case, method, state, "skill_adjusted", res[state], skills,
                       hint_gap_steps=labels)
        if group == "gap_forced":
            rec["condition"] = base + "gf"
            for s in rec["plan"]["steps"]:
                if s.get("source") == "gap":
                    s["source"] = "gap_forced"
        out = RET / base / f"{cid}.json"
        out.write_text(json.dumps(rec, indent=2))
        gap_n = sum(1 for s in rec["plan"]["steps"]
                    if s.get("source") in ("gap","gap_forced"))
        print(f"    -> wrote {out.relative_to(config.ROOT)}  "
              f"cond={rec['condition']} gap_steps={gap_n} "
              f"top1={rec['metrics']['top1_acc']} gap_det={rec['metrics']['gap_detection']}\n")

    stray = config.RESULTS_RT / "skill_adjusted"
    if stray.exists():
        shutil.rmtree(stray)
        print(f"cleaned stray {stray.relative_to(config.ROOT)}")

if __name__ == "__main__":
    main()
