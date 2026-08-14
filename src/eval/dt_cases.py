"""Build design-time test cases from activity_cases.json.

A DT case is:
    {
      "case_id":        "proc_001_s01",
      "process":        "proc_001",
      "process_title":  "Manufacturing Change Compliance",
      "step_id":        "s1",
      "step_index":     1,
      "label":          "Manufacturing Change Compliance process: Request Production Schedule Adjustment — Initiate...",
      "expected_ordIds": ["sap.crm:apiResource:MachineProductionOrderIntegration:v1"],
      "is_gt":          True,
    }

The label includes the process title as context — in practice an orchestrator always
knows which process it is executing, so this is realistic additional context.
"""
from __future__ import annotations

import json
import re

from src import config


def _load_process_titles() -> dict[str, str]:
    """Load human-readable process titles from XML id attributes."""
    proc_dir = config.DT_OUTPUT_DIR / "processes"
    titles: dict[str, str] = {}
    if not proc_dir.exists():
        return titles
    for xml_path in sorted(proc_dir.glob("proc_*.xml")):
        proc_id = xml_path.stem
        try:
            xml = xml_path.read_text()
            m = re.search(r'<process[^>]+id="([^"]+)"', xml)
            if m:
                raw = m.group(1)
                title = raw.replace("proc_", "").replace("_v1", "").replace("_", " ").strip()
                titles[proc_id] = title
        except Exception:
            pass
    return titles


def build_cases(gt_only: bool = False) -> list[dict]:
    """Load cases from activity_cases.json.

    gt_only: if True, return only GT-eligible steps (is_gt=True).
             For evaluation we typically want all 240 cases; the GT flag
             is used for stratified analysis.

    The label is enriched with the process title as context prefix.
    """
    path = config.DT_OUTPUT_DIR / "activity_cases.json"
    if not path.exists():
        raise FileNotFoundError(f"activity_cases.json not found at {path}")

    proc_titles = _load_process_titles()
    raw: list[dict] = json.loads(path.read_text())
    cases = []
    for c in raw:
        if gt_only and not c.get("is_gt", False):
            continue
        proc_id = c["process_id"]
        proc_title = proc_titles.get(proc_id, "")
        # Prepend process title so methods have domain context
        activity = c["input"]
        label = f"{proc_title} process: {activity}" if proc_title else activity
        cases.append({
            "case_id":         c["case_id"],
            "process":         proc_id,
            "process_title":   proc_title,
            "step_id":         c.get("step_id", ""),
            "step_index":      c.get("step_index", 0),
            "label":           label,
            "label_raw":       activity,   # original without process context
            "expected_ordIds": [c["expected_ordId"]] if c.get("expected_ordId") else [],
            "is_gt":           c.get("is_gt", False),
        })
    return cases
