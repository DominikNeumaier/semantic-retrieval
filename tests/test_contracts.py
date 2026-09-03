from __future__ import annotations

import unittest

from src import config
from src.loader import load_landscape
from src.methods.method_raw import _safe_path
from src.runtime.skill_registry import load_skills


class SemanticRetrievalContractTest(unittest.TestCase):
    def test_ord_bench_dependency_is_available(self) -> None:
        self.assertTrue(config.ORD_BENCH_ROOT.is_dir())
        self.assertTrue((config.BENCHMARK_DIR / "landscape").is_dir())

    def test_landscape_is_normalized(self) -> None:
        clean = load_landscape("clean")
        enriched = load_landscape("enriched")
        self.assertEqual(273, len(clean))
        self.assertEqual({r["ordId"] for r in clean}, {r["ordId"] for r in enriched})
        for resource in clean + enriched:
            for key in ("ordId", "namespace", "type", "title", "entityTypes"):
                self.assertIn(key, resource)
            self.assertIn(resource["type"], {"apiResource", "agent", "dataProduct"})

    def test_skills_match_current_ord_bench_format(self) -> None:
        skills = load_skills()
        self.assertEqual(30, len(skills))
        self.assertEqual(240, sum(len(skill["steps"]) for skill in skills))
        self.assertEqual(
            120,
            sum(len(step["ord_confirmed"]) for skill in skills for step in skill["steps"]),
        )
        self.assertEqual(30, len({skill["process"] for skill in skills}))

    def test_agentic_raw_virtual_path_is_sandboxed(self) -> None:
        mapped = _safe_path("benchmark/landscape/systems/sap.s4/ord.json")
        self.assertIsNotNone(mapped)
        self.assertTrue(mapped.is_file())
        self.assertIsNone(_safe_path("../ord-bench/data/landscape/systems/sap.s4/ord.json"))

    def test_published_result_paths_exist(self) -> None:
        root = config.ROOT / "results"
        self.assertTrue((root / "retrieval" / "design-time").is_dir())
        self.assertTrue((root / "retrieval" / "runtime").is_dir())
        self.assertTrue((root / "orchestration").is_dir())

    def test_published_headline_metrics(self) -> None:
        from analysis.numbers import _pct, dynamic_per_case, routing_accuracy

        self.assertEqual(45.0, _pct(list(dynamic_per_case("F", 1, "top1_acc").values())))
        self.assertEqual(70.0, _pct(list(dynamic_per_case("A", 1, "candidate_recall").values())))
        self.assertEqual((60.0, 30), routing_accuracy("skill_guided"))


if __name__ == "__main__":
    unittest.main()
