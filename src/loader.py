"""Load and normalize ORD-Bench resources for retrieval methods.

The benchmark stores different resource types in separate ORD arrays and uses
resource-specific entity-type fields.  Retrieval methods consume one stable,
flat representation; this adapter is the explicit data contract between the
separate ORD-Bench and Semantic Retrieval repositories.
"""
from __future__ import annotations

import json
from typing import Any

from src import config

RESOURCE_KEYS = ("apiResources", "agents", "dataProducts")
TYPE_FROM_KEY = {
    "apiResources": "apiResource",
    "agents": "agent",
    "dataProducts": "dataProduct",
}


def _extract_entity_types(item: dict[str, Any]) -> list[str]:
    out: list[str] = []

    def add(value: str) -> None:
        if value and value not in out:
            out.append(value)

    for mapping in item.get("entityTypeMappings", []) or []:
        for target in mapping.get("entityTypeTargets", []) or []:
            if isinstance(target, dict) and isinstance(target.get("ordId"), str):
                add(target["ordId"])

    for ref in item.get("exposedEntityTypes", []) or []:
        if isinstance(ref, dict) and isinstance(ref.get("ordId"), str):
            add(ref["ordId"])

    for ref in item.get("relatedEntityTypes", []) or []:
        if isinstance(ref, dict) and isinstance(ref.get("ordId"), str):
            add(ref["ordId"])
        elif isinstance(ref, str):
            add(ref)

    for value in item.get("entityTypes", []) or []:
        if isinstance(value, str):
            add(value)

    return out


def _package_index(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for package in document.get("packages", []) or []:
        ord_id = package.get("ordId")
        if ord_id:
            index[ord_id] = package
    return index


def _flatten(document: dict[str, Any], namespace: str) -> list[dict[str, Any]]:
    packages = _package_index(document)
    resources: list[dict[str, Any]] = []
    for collection in RESOURCE_KEYS:
        for item in document.get(collection, []) or []:
            package_id = item.get("partOfPackage", "")
            package = packages.get(package_id, {})
            resources.append({
                "ordId": item.get("ordId", ""),
                "namespace": namespace,
                "_system": namespace,
                # Keep the retrieval modality separate from raw type/category
                # fields such as dataProduct type=primary|derived.
                "type": TYPE_FROM_KEY[collection],
                "rawType": item.get("type", ""),
                "title": item.get("title", ""),
                "shortDescription": item.get("shortDescription", ""),
                "description": item.get("description", ""),
                "entityTypes": _extract_entity_types(item),
                "partOfPackage": package_id,
                "packageTitle": package.get("title", ""),
                "lineOfBusiness": item.get("lineOfBusiness")
                                  or package.get("lineOfBusiness", []) or [],
                "partOfProducts": item.get("partOfProducts")
                                  or package.get("partOfProducts", []) or [],
                "tags": item.get("tags", []) or [],
                "capabilities": item.get("capabilities", []) or [],
                "useCases": item.get("useCases", []) or [],
                "partOfGroups": item.get("partOfGroups", []) or [],
                "processNext": item.get("processNext", []) or [],
                "apiProtocol": item.get("apiProtocol", ""),
                "resourceDefinitions": item.get("resourceDefinitions", []) or [],
                "apiResourceLinks": item.get("apiResourceLinks", []) or [],
                "responsible": item.get("responsible", ""),
                # Preserve the source object for methods that need additional
                # ORD fields without weakening the normalized contract.
                "raw": item,
            })
    return resources


def load_landscape(state: str = "clean") -> list[dict[str, Any]]:
    """Return the 273 normalized retrieval resources for one ORD state."""
    if state not in {"clean", "enriched"}:
        raise ValueError(f"Unknown state {state!r}; expected 'clean' or 'enriched'")

    if not config.LANDSCAPE_DIR.exists():
        raise FileNotFoundError(
            f"ORD-Bench landscape not found at {config.LANDSCAPE_DIR}. "
            "Set ORD_BENCH_DIR to the ord-bench repository root."
        )

    resources: list[dict[str, Any]] = []
    for system_dir in sorted(config.LANDSCAPE_DIR.iterdir()):
        if not system_dir.is_dir() or system_dir.name == "sap.odm":
            continue
        if state == "enriched":
            candidate = config.LANDSCAPE_ENRICHED_DIR / system_dir.name / "ord_enriched.json"
            path = candidate if candidate.exists() else system_dir / "ord.json"
        else:
            path = system_dir / "ord.json"
        if not path.exists():
            continue
        resources.extend(_flatten(json.loads(path.read_text()), system_dir.name))

    if not resources:
        raise RuntimeError(f"No ORD resources loaded from {config.LANDSCAPE_DIR}")
    return resources
