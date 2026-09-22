from __future__ import annotations

import json
import os
import unittest

from chemcrow.tools import HybridEnzymeRetroPlanner


GLUCOSE = "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"


@unittest.skipUnless(
    os.getenv("RUN_MOORETRO_E2E") == "1",
    "set RUN_MOORETRO_E2E=1 to run the real SQLite + real MooRetro acceptance test",
)
class RealHybridPlannerE2ETest(unittest.TestCase):
    def test_glucose_to_ethanol_acceptance(self):
        result = json.loads(
            HybridEnzymeRetroPlanner().run(
                "target=CCO;"
                f"building_blocks={GLUCOSE};"
                "top_k=10;"
                "max_depth=10"
            )
        )
        self.assertTrue(result["ok"])
        self.assertGreaterEqual(result["route_count"], 2)
        for name in (
            "states_expanded",
            "states_generated",
            "backward_queries",
            "model_calls",
            "model_candidates",
        ):
            self.assertGreater(result["stats"][name], 0, name)
        self.assertGreater(result["stats"]["forward_direct_solutions"], 0)
        self.assertFalse(
            any(
                route["step_count"] == 1
                and route["steps"][0]["database_reaction_ids"] == ["R04410"]
                for route in result["routes"]
            )
        )
        self.assertTrue(
            any(
                {"C00022", "C00084", "C00469"}.issubset(
                    {
                        step["target"]["id"]
                        for step in route["steps"]
                        if step["target"]["id"]
                    }
                )
                for route in result["routes"]
            )
        )
        if result["stats"]["consensus_candidates"] > 0:
            self.assertTrue(
                any(
                    step["provenance"] == "BOTH"
                    for route in result["routes"]
                    for step in route["steps"]
                )
            )


if __name__ == "__main__":
    unittest.main()
