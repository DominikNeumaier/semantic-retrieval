"""Skill-Guided routing evaluation — method- and ORD-state-independent.

For SG the only question is: does the intent resolver pick the correct skill?
The resolver only sees user prompt + skill descriptions — no ORD resources,
no retrieval method. Running once is sufficient; ORD state is irrelevant.

Results go to results/runtime/skill_guided/routing/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from src import config
from src.runtime import intent_resolver, skill_registry

CASES_FILE = config.RT_OUTPUT_DIR / "skill_guided.json"
OUT_DIR    = config.RESULTS_RT / "skill_guided" / "routing"


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    skills = skill_registry.load_skills()
    cases  = json.loads(CASES_FILE.read_text())
    print(f"SG routing eval — {len(cases)} cases · {len(skills)} skills in registry")
    print("(ORD state irrelevant — resolver uses only prompt + skill descriptions)")

    records: list[dict] = []

    for case in cases:
        cid            = case["case_id"]
        expected_skill = case.get("expected_skill_id") or case.get("skill_id")
        prompt         = case["user_prompt"]

        t0     = time.time()
        result = intent_resolver.resolve(prompt, skills)
        wall   = round(time.time() - t0, 3)

        picked     = result["skill_id"]
        routing_ok = int(picked == expected_skill)
        reason     = result["trace"].get("reason", "")
        tokens     = result["trace"].get("tokens", 0)

        rec = {
            "case_id":        cid,
            "user_prompt":    prompt,
            "skill_expected": expected_skill,
            "skill_picked":   picked,
            "routing_ok":     routing_ok,
            "reason":         reason,
            "tokens":         tokens,
            "wall_s":         wall,
        }
        records.append(rec)

        icon = "✓" if routing_ok else "✗"
        print(f"  {icon} {cid}  expected={expected_skill}  picked={picked}")
        if not routing_ok:
            print(f"    → {reason[:120]}")

        (OUT_DIR / f"{cid}.json").write_text(json.dumps(rec, indent=2))

    (OUT_DIR / "records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records)
    )

    acc     = sum(r["routing_ok"] for r in records) / len(records) if records else 0
    correct = sum(r["routing_ok"] for r in records)
    tokens  = sum(r["tokens"] for r in records)
    print(f"\nRouting-Acc: {acc:.3f}  ({correct}/{len(records)})  total_tokens={tokens:,}")

    summary = {
        "cases":        len(records),
        "Routing-Acc":  round(acc, 4),
        "correct":      correct,
        "total_tokens": tokens,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"→ {OUT_DIR}")


if __name__ == "__main__":
    run()
