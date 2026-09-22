from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from chemcrow.planner import EnzRetroClient, EnzymeReactionDatabase, HybridPlannerEngine

from planner_test_utils import create_test_database, small_config
from test_candidate_fusion import FakePredictor


class HybridOptimizationRegressionTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        path = create_test_database(Path(self.tempdir.name) / "planner.db")
        self.database = EnzymeReactionDatabase(path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_forward_direct_solution_does_not_skip_hybrid_backward_search(self):
        predictor = FakePredictor()
        output = HybridPlannerEngine(
            database=self.database,
            model_client=EnzRetroClient(predictor=predictor),
            config=small_config(
                use_model=True,
                forward_anchor_depth=2,
                top_k_routes=10,
                include_search_trace=True,
            ),
        ).plan("Target", ["Start"])

        stats = output["stats"]
        self.assertGreaterEqual(stats["forward_direct_solutions"], 1)
        self.assertGreater(stats["states_expanded"], 0)
        self.assertGreater(stats["states_generated"], 0)
        self.assertGreater(stats["backward_queries"], 0)
        self.assertGreater(stats["model_calls"], 0)
        self.assertGreater(stats["model_candidates"], 0)
        self.assertGreaterEqual(output["route_count"], 2)
        self.assertEqual(
            stats["search_phases"],
            [
                "PHASE_A_FORWARD_ANCHOR",
                "PHASE_B_BACKWARD_HYBRID_SEARCH",
                "PHASE_C_SOLUTION_MERGE",
                "PHASE_D_ROUTE_SCORING",
                "PHASE_E_DIVERSITY_RERANK",
            ],
        )
        self.assertTrue(
            any(
                step["provenance"] == "BOTH"
                for route in output["routes"]
                for step in route["steps"]
            )
        )

    def test_forward_anchor_cache_reused_across_engine_instances(self):
        config = small_config(forward_anchor_depth=2)
        first = HybridPlannerEngine(
            database=self.database, config=config
        ).plan("Target", ["Start"])
        second = HybridPlannerEngine(
            database=self.database, config=config
        ).plan("Target", ["Start"])
        self.assertGreaterEqual(first["stats"]["forward_anchor_cache_misses"], 1)
        self.assertGreaterEqual(second["stats"]["forward_anchor_cache_hits"], 1)
        self.assertEqual(first["route_count"], second["route_count"])

    def test_beam_pruning_and_best_first_trace_are_exercised(self):
        weight = 0.35
        output = HybridPlannerEngine(
            database=self.database,
            config=small_config(
                forward_anchor_depth=0,
                global_beam_width=1,
                heuristic_weight=weight,
                include_search_trace=True,
            ),
        ).plan("Target", ["Start"])
        self.assertGreater(output["stats"]["states_generated"], 1)
        self.assertGreater(output["stats"]["beam_states_pruned"], 0)
        expanded = [
            event
            for event in output["search_trace"]
            if event["event"] == "state_expanded"
        ]
        self.assertTrue(expanded)
        first = expanded[0]
        self.assertAlmostEqual(
            first["f_cost"], first["g_cost"] + weight * first["h_cost"], places=7
        )
        self.assertIn("expanded_molecule", first)

    def test_model_failure_retries_obey_call_budget(self):
        def failing_predictor(**kwargs):
            raise RuntimeError("synthetic model failure")

        output = HybridPlannerEngine(
            database=self.database,
            model_client=EnzRetroClient(predictor=failing_predictor),
            config=small_config(
                use_model=True,
                forward_anchor_depth=0,
                max_model_calls=2,
            ),
        ).plan("Target", ["Start"])
        self.assertEqual(output["stats"]["model_failures"], 2)
        self.assertEqual(output["stats"]["model_requests"], 2)
        self.assertGreaterEqual(output["route_count"], 1)

    def test_model_only_mode_and_ablation_controls(self):
        output = HybridPlannerEngine(
            database=self.database,
            model_client=EnzRetroClient(predictor=FakePredictor()),
            config=small_config(
                use_database=False,
                use_model=True,
                forward_anchor_depth=0,
                heuristic_weight=0.0,
                diversity_penalty=0.0,
            ),
        ).plan("Target", ["Intermediate A"])
        self.assertEqual(output["mode"], "model-only")
        self.assertEqual(output["config"]["heuristic_weight"], 0.0)
        self.assertEqual(output["config"]["diversity_penalty"], 0.0)
        self.assertGreaterEqual(output["route_count"], 1)
        self.assertTrue(
            all(
                step["provenance"] == "MODEL"
                for route in output["routes"]
                for step in route["steps"]
            )
        )


if __name__ == "__main__":
    unittest.main()
