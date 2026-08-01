"""Parse SKILL.md files (Agent Skills Standard) into a small in-memory
registry used by the runtime planner.

A Skill is:
    {
      "skill_id":   "proc_machine_breakdown",
      "name":       "proc-machine-breakdown",
      "description":"...",
      "process":    "machine_breakdown",
      "process_type": "bpmn" | "cmmn",
      "steps": [
        {"index": 1, "name": "Equipment Failure Diagnosis",
         "ord_confirmed": ["my.mes:agent:EquipmentDiagnostic:v1", ...]},
        ...
      ]
    }
"""

from __future__ import annotations

import re
from pathlib import Path

from src.core import config


SKILLS_DIR = config.ROOT / "benchmark" / "test_cases" / "design_time" / "output" / "skills"

_HEADER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_STEP_RE = re.compile(r"^###\s+\d+\.\s+(.+?)\s*$", re.MULTILINE)
_ORD_CONFIRMED_RE = re.compile(r"<!--\s*ord_confirmed:\s*([^>]+?)-->", re.DOTALL)


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Tiny YAML reader for skill frontmatter.

    Top-level keys (e.g. `name:`, `description:`, `metadata:`) are read
    as strings. Indented children of a top-level `metadata:` mapping are
    promoted to top-level keys themselves (e.g. `process-type` becomes
    directly available under `fm["process-type"]`), so callers can read
    them without re-parsing the nested block. Indented continuations of
    other top-level keys (e.g. a multi-line `description: >` block) are
    concatenated into the parent value.
    """
    m = _HEADER_RE.match(text)
    if not m:
        return {}
    fm: dict[str, str] = {}
    block = m.group(1)
    current_key: str | None = None
    in_metadata = False
    for line in block.splitlines():
        if re.match(r"^[a-zA-Z][\w-]*:", line):
            k, _, v = line.partition(":")
            key = k.strip()
            fm[key] = v.strip().rstrip(">").strip()
            current_key = key
            in_metadata = (key == "metadata" and not fm[key])
        elif in_metadata and re.match(r"^\s{2}[a-zA-Z][\w-]*:", line):
            # nested metadata key — promote to top level
            k, _, v = line.strip().partition(":")
            fm[k.strip()] = v.strip()
        elif current_key and line.startswith("  "):
            fm[current_key] = (fm[current_key] + " " + line.strip()).strip()
    return fm


def _parse_steps(body: str) -> list[dict]:
    """Split body at '### N. Name', then pick the ord_confirmed block per step."""
    # find all step headers with their start position
    headers = [(m.start(), m.group(1).strip()) for m in _STEP_RE.finditer(body)]
    if not headers:
        return []
    headers.append((len(body), ""))   # sentinel
    steps: list[dict] = []
    for i in range(len(headers) - 1):
        start, name = headers[i]
        end, _ = headers[i + 1]
        block = body[start:end]
        confirmed: list[str] = []
        for cm in _ORD_CONFIRMED_RE.finditer(block):
            for ord_id in cm.group(1).split(","):
                oid = ord_id.strip()
                if oid:
                    confirmed.append(oid)
        steps.append({"index": i + 1, "name": name, "ord_confirmed": confirmed})
    return steps


def parse_skill(path: Path) -> dict:
    text = path.read_text()
    fm = _parse_frontmatter(text)
    body_start = _HEADER_RE.match(text).end() if _HEADER_RE.match(text) else 0
    body = text[body_start:]
    return {
        "skill_id": fm.get("process-id", path.stem),
        "name": fm.get("name", path.stem),
        "description": fm.get("description", ""),
        "process": fm.get("source-file", "").replace(".bpmn", "").replace(".cmmn", ""),
        "process_type": fm.get("process-type", "bpmn"),
        "steps": _parse_steps(body),
    }


def load_skills() -> list[dict]:
    return [parse_skill(p) for p in sorted(SKILLS_DIR.glob("*.md"))]


if __name__ == "__main__":
    skills = load_skills()
    print(f"Loaded {len(skills)} skills")
    for s in skills:
        n_steps = len(s["steps"])
        n_confirmed = sum(1 for st in s["steps"] if st["ord_confirmed"])
        print(f"  {s['skill_id']:<28} steps={n_steps:>2} confirmed={n_confirmed:>2}  ({s['process_type']})")
