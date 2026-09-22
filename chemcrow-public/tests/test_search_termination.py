from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from chemcrow.planner import EnzymeReactionDatabase, HybridPlannerEngine

from planner_test_utils import create_test_database, small_config


class SearchTerminationTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        path = create_test_database(Path(self.tempdir.name) / "planner.db")
        self.database = EnzymeReactionDatabase(path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_depth_limit_prevents_incomplete_route(self):
        config = small_config(max_depth=1, forward_anchor_depth=0)
        output = HybridPlannerEngine(
            database=self.database, config=config
        ).plan("Target", ["Start"])
        self.assertEqual(output["route_count"], 0)

    def test_expansion_budget_is_honored(self):
        config = small_config(max_expansions=1, forward_anchor_depth=0)
        output = HybridPlannerEngine(
            database=self.database, config=config
        ).plan("Target", ["Start"])
        self.assertLessEqual(output["stats"]["states_expanded"], 1)
        self.assertEqual(output["route_count"], 0)

    def test_invalid_input_is_structured(self):
        output = HybridPlannerEngine(
            database=self.database, config=small_config()
        ).plan("not a molecule", ["Start"])
        self.assertFalse(output["ok"])
        self.assertEqual(output["error"]["type"], "invalid_input")


if __name__ == "__main__":
    unittest.main()
