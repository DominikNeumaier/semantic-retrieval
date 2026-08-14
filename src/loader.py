"""Landscape loader — reads ORD JSON files from the ord-bench data directory."""

from __future__ import annotations

import json
from pathlib import Path

from src import config

_ORD_FILES = {"clean": "ord.json", "enriched": "ord_enriched.json"}
_RESOURCE_KEYS = ("apiResources", "dataProducts", "agents")


def load_landscape(state: str = "clean") -> list[dict]:
    """Return a flat list of all ORD resources for the given description state.

    Each resource dict is the raw ORD JSON object with an added ``_system``
    field containing the system directory name for traceability.
    """
    filename = _ORD_FILES.get(state)
    if filename is None:
        raise ValueError(f"Unknown state {state!r}; expected 'clean' or 'enriched'")

    base = config.LANDSCAPE_ENRICHED_DIR if state == "enriched" else config.LANDSCAPE_DIR
    resources: list[dict] = []
    for system_dir in sorted(base.iterdir()):
        ord_file = system_dir / filename
        if not ord_file.exists():
            continue
        data = json.loads(ord_file.read_text())
        for key in _RESOURCE_KEYS:
            for resource in data.get(key, []):
                resource["_system"] = system_dir.name
                resources.append(resource)
    return resources
