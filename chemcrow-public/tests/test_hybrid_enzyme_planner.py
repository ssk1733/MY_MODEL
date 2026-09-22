from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from chemcrow.planner import EnzRetroClient, EnzymeReactionDatabase, HybridPlannerEngine
from chemcrow.tools import EnzymeRetroPathFinder, HybridEnzymeRetroPlanner

from planner_test_utils import create_test_database, small_config
from test_candidate_fusion import FakePredictor


class HybridPlannerAndToolTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        path = create_test_database(Path(self.tempdir.name) / "planner.db")
        self.database = EnzymeReactionDatabase(path)
        self.predictor = FakePredictor()
        self.client = EnzRetroClient(predictor=self.predictor)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_hybrid_route_preserves_both_provenance(self):
        config = small_config(use_model=True)
        output = HybridPlannerEngine(
            database=self.database, model_client=self.client, config=config
        ).plan("Target", ["Start"])
        self.assertGreaterEqual(output["route_count"], 1)
        provenances = [
            step["provenance"] for step in output["routes"][0]["steps"]
        ]
        self.assertIn("BOTH", provenances)
        self.assertIn("DATABASE", provenances)

    def test_old_langchain_run_accepts_legacy_string(self):
        tool = EnzymeRetroPathFinder(
            database_service=self.database,
            config={"max_expansions": 50, "top_k_routes": 2},
        )
        result = json.loads(
            tool.run(
                "target=Target;building_blocks=Start;"
                "forward_anchor_depth=1;top_k=2"
            )
        )
        self.assertTrue(result["ok"])
        self.assertGreaterEqual(result["route_count"], 1)
        self.assertNotIn("_parse_input", EnzymeRetroPathFinder.__dict__)

    def test_hybrid_tool_can_expose_model_only_mode(self):
        tool = HybridEnzymeRetroPlanner(
            database_service=self.database,
            model_client=self.client,
            config={
                "use_database": False,
                "use_model": True,
                "forward_anchor_depth": 0,
                "max_expansions": 10,
            },
        )
        result = json.loads(
            tool.run("target=Target;building_blocks=Intermediate A;top_k=2")
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "model-only")
        self.assertGreaterEqual(result["route_count"], 1)

    def test_hybrid_tool_is_one_call_planner(self):
        tool = HybridEnzymeRetroPlanner(
            database_service=self.database,
            model_client=self.client,
            config={"max_expansions": 50, "top_k_routes": 2},
        )
        result = json.loads(
            tool.run(
                json.dumps(
                    {
                        "target": "Target",
                        "building_blocks": ["Start"],
                        "config": {"forward_anchor_depth": 1},
                    }
                )
            )
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["planner"], "HybridEnzymeRetroPlanner")
        self.assertEqual(self.predictor.calls, 1)


if __name__ == "__main__":
    unittest.main()
