from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from chemcrow.planner import EnzymeReactionDatabase, HybridPlannerEngine

from planner_test_utils import create_test_database, small_config


class DatabaseOnlyPathFinderTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        db_path = create_test_database(Path(self.tempdir.name) / "planner.db")
        self.database = EnzymeReactionDatabase(db_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_direction_filter_and_normalization(self):
        target = self.database.normalize_molecule("Target")
        candidates = self.database.backward_candidates(target, limit=20)
        ids = {value for item in candidates for value in item.database_reaction_ids}
        self.assertIn("R10002", ids)
        self.assertIn("R10004", ids)
        self.assertNotIn("R10007", ids)
        self.assertEqual(target.key, self.database.normalize_molecule("CC=O").key)

    def test_forward_anchor_and_top_k_routes(self):
        engine = HybridPlannerEngine(database=self.database, config=small_config())
        output = engine.plan("Target", ["Start"])
        self.assertTrue(output["ok"])
        self.assertGreaterEqual(output["route_count"], 2)
        self.assertTrue(all(route["complete"] for route in output["routes"]))
        self.assertTrue(
            all(
                step["provenance"] == "DATABASE"
                for route in output["routes"]
                for step in route["steps"]
            )
        )
        self.assertGreater(output["stats"]["forward_anchor_count"], 1)

    def test_and_reaction_requires_every_noncurrency_leaf(self):
        config = small_config(forward_anchor_depth=0, top_k_routes=2)
        engine = HybridPlannerEngine(database=self.database, config=config)
        missing = engine.plan("AND target", ["Start"])
        self.assertEqual(missing["route_count"], 0)
        solved = engine.plan("AND target", ["Start", "Second substrate"])
        self.assertEqual(solved["route_count"], 1)
        roles = {item["role"] for item in solved["routes"][0]["steps"][0]["precursors"]}
        self.assertIn("main", roles)

    def test_target_equal_to_building_block_is_zero_step_route(self):
        engine = HybridPlannerEngine(database=self.database, config=small_config())
        output = engine.plan("Start", ["CC"])
        self.assertEqual(output["route_count"], 1)
        self.assertEqual(output["routes"][0]["step_count"], 0)
        self.assertEqual(output["stats"]["termination_reason"], "TARGET_IS_BUILDING_BLOCK")

    def test_cycle_is_filtered(self):
        config = small_config(forward_anchor_depth=0)
        output = HybridPlannerEngine(
            database=self.database, config=config
        ).plan("Target", ["Start"])
        self.assertGreaterEqual(output["route_count"], 1)
        self.assertGreater(output["stats"]["filter_reasons"].get("route_cycle", 0), 0)


if __name__ == "__main__":
    unittest.main()
