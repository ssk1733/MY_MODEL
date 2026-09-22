from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from chemcrow.planner import EnzRetroClient, EnzymeReactionDatabase
from chemcrow.tools import (
    EnzymeMultiStepEvaluator,
    EnzymeRetrosynthesisSearch,
    EnzymeSingleStepEvaluator,
)

from planner_test_utils import create_test_database
from test_candidate_fusion import FakePredictor


class LayeredRetrosynthesisToolsTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        path = create_test_database(Path(self.tempdir.name) / "planner.db")
        self.database = EnzymeReactionDatabase(path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_single_step_evaluator_ranks_candidates(self):
        tool = EnzymeSingleStepEvaluator(database_service=self.database)
        result = json.loads(
            tool.run(
                json.dumps(
                    {
                        "target": "Target",
                        "building_blocks": ["Start"],
                        "candidates": [
                            {
                                "candidate_id": "low",
                                "precursors": ["Intermediate B"],
                                "source": "MODEL",
                                "model_score": 0.10,
                                "feasibility": 0.20,
                                "ec": "2.2.2.2",
                            },
                            {
                                "candidate_id": "high",
                                "precursors": ["Intermediate A"],
                                "source": "MODEL",
                                "model_score": 0.95,
                                "feasibility": 0.90,
                                "ec": "1.1.1.1",
                            },
                        ],
                    }
                )
            )
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["ranked_candidates"][0]["candidate_id"], "high")
        metrics = result["ranked_candidates"][0]["raw_metrics"]
        self.assertIsNotNone(metrics["structural_similarity"])
        self.assertIsNotNone(metrics["approx_structural_overlap"])

    def test_tree_search_exports_and_or_solution_graph(self):
        tool = EnzymeRetrosynthesisSearch(
            database_service=self.database,
            model_client=EnzRetroClient(predictor=FakePredictor()),
            config={
                "use_database": True,
                "use_model": True,
                "max_depth": 6,
                "max_expansions": 100,
                "forward_anchor_depth": 1,
                "forward_anchor_max_nodes": 30,
                "top_k_routes": 3,
                "global_beam_width": 30,
            },
        )
        result = json.loads(
            tool.run(
                json.dumps(
                    {
                        "target": "Target",
                        "building_blocks": ["Start"],
                        "top_k": 2,
                    }
                )
            )
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["planner"], "EnzymeRetrosynthesisSearch")
        self.assertLessEqual(result["route_count"], 2)
        self.assertGreaterEqual(result["route_count"], 1)
        self.assertEqual(
            result["and_or_tree"]["representation"], "AND_OR_HYPERGRAPH"
        )
        self.assertTrue(
            all(
                node["node_type"] == "OR"
                for node in result["and_or_tree"]["molecule_nodes"]
            )
        )
        self.assertTrue(
            all(
                node["node_type"] == "AND"
                for node in result["and_or_tree"]["reaction_nodes"]
            )
        )
        self.assertTrue(
            all(route["complete"] for route in result["routes"])
        )
        self.assertTrue(
            all(route["route_validation"]["valid"] for route in result["routes"])
        )
        self.assertEqual(result["endpoint_task"]["endpoint_mode"], "any_stock")
        self.assertFalse(result["endpoint_task"]["terminal_set_must_equal_building_blocks"])


    def test_stock_mode_allows_subset_but_exact_mode_requires_every_endpoint(self):
        tool = EnzymeRetrosynthesisSearch(
            database_service=self.database,
            config={
                "use_database": True,
                "use_model": False,
                "max_depth": 6,
                "max_expansions": 40,
                "forward_anchor_depth": 0,
                "top_k_routes": 2,
            },
        )

        def run(endpoint_mode):
            return json.loads(
                tool.run(
                    json.dumps(
                        {
                            "target": "Target",
                            "building_blocks": ["Start", "Second substrate"],
                            "endpoint_mode": endpoint_mode,
                            "top_k": 2,
                        }
                    )
                )
            )

        stock = run("any_stock")
        exact = run("exact_set")
        self.assertTrue(stock["ok"])
        self.assertGreaterEqual(stock["route_count"], 1)
        self.assertFalse(stock["config"]["require_all_building_blocks"])
        self.assertEqual(exact["route_count"], 0)
        self.assertTrue(exact["config"]["require_all_building_blocks"])
        self.assertGreater(exact["stats"]["endpoint_routes_rejected"], 0)

    def test_multiple_targets_return_ranked_complete_route_bundles(self):
        tool = EnzymeRetrosynthesisSearch(
            database_service=self.database,
            config={
                "use_database": True,
                "use_model": False,
                "max_depth": 6,
                "max_expansions": 50,
                "forward_anchor_depth": 0,
                "top_k_routes": 3,
            },
        )
        result = json.loads(
            tool.run(
                json.dumps(
                    {
                        "targets": ["Target", "AND target"],
                        "building_blocks": ["Start", "Second substrate"],
                        "endpoint_mode": "any_stock",
                        "top_k": 3,
                    }
                )
            )
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "multi-target")
        self.assertEqual(result["target_count"], 2)
        self.assertTrue(result["all_targets_solved"])
        self.assertGreaterEqual(result["route_count"], 1)
        bundle = result["route_bundles"][0]
        self.assertTrue(bundle["complete"])
        self.assertEqual(len(bundle["target_routes"]), 2)
        self.assertTrue(
            all(
                item["route"]["route_validation"]["valid"]
                for item in bundle["target_routes"]
            )
        )

    def test_local_text_building_block_library_is_loaded(self):
        library = Path(self.tempdir.name) / "stock.txt"
        library.write_text("Start\nSecond substrate\n", encoding="utf-8")
        tool = EnzymeRetrosynthesisSearch(
            database_service=self.database,
            config={
                "use_database": True,
                "use_model": False,
                "max_depth": 4,
                "max_expansions": 30,
                "forward_anchor_depth": 0,
            },
        )
        result = json.loads(
            tool.run(
                json.dumps(
                    {
                        "target": "AND target",
                        "building_block_library": str(library),
                        "endpoint_mode": "any_stock",
                    }
                )
            )
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["route_count"], 1)
        self.assertEqual(result["input"]["library_sources"][0]["entry_count"], 2)


    def test_and_or_mcts_runs_selection_rollout_and_backpropagation(self):
        tool = EnzymeRetrosynthesisSearch(
            database_service=self.database,
            config={
                "use_database": True,
                "use_model": False,
                "max_depth": 6,
                "max_expansions": 12,
                "forward_anchor_depth": 0,
                "top_k_routes": 5,
                "solution_pool_multiplier": 2,
                "mcts_rollout_depth": 2,
                "mcts_random_seed": 3,
            },
        )
        result = json.loads(
            tool.run(
                json.dumps(
                    {
                        "target": "Target",
                        "building_blocks": ["Start"],
                        "top_k": 5,
                    }
                )
            )
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["search_algorithm"]["name"], "and_or_mcts")
        self.assertEqual(
            result["search_algorithm"]["graph"], "AND_OR_HYPERGRAPH"
        )
        stats = result["stats"]
        self.assertGreater(stats["mcts_iterations"], 0)
        self.assertGreater(stats["mcts_selection_steps"], 0)
        self.assertGreater(stats["mcts_expansion_nodes"], 0)
        self.assertGreater(stats["mcts_rollouts"], 0)
        self.assertGreater(stats["mcts_rollout_steps"], 0)
        self.assertEqual(
            stats["mcts_backpropagations"], stats["mcts_iterations"]
        )
        self.assertEqual(stats["mcts_root_visits"], stats["mcts_iterations"])
        self.assertGreaterEqual(result["route_count"], 1)
        route = result["routes"][0]
        annotated = route["enzyme_annotated_retrosynthesis"]
        self.assertIn(chr(0x2190), annotated)
        self.assertIn("EC ", annotated)
        self.assertTrue(route["retrosynthesis_reactions"])
        first_reaction = route["retrosynthesis_reactions"][0]
        self.assertIn("enzyme", first_reaction)
        self.assertTrue(first_reaction["enzyme"]["ec_numbers"])
        self.assertIn("provenance", first_reaction["enzyme"])

    def test_and_reaction_requires_every_noncurrency_precursor(self):
        def search(blocks):
            tool = EnzymeRetrosynthesisSearch(
                database_service=self.database,
                config={
                    "use_database": True,
                    "use_model": False,
                    "max_depth": 4,
                    "max_expansions": 20,
                    "forward_anchor_depth": 0,
                    "top_k_routes": 1,
                    "solution_pool_multiplier": 1,
                    "mcts_rollout_depth": 2,
                    "mcts_random_seed": 5,
                },
            )
            return json.loads(
                tool.run(
                    json.dumps(
                        {
                            "target": "AND target",
                            "building_blocks": blocks,
                            "top_k": 1,
                        }
                    )
                )
            )

        missing_one = search(["Start"])
        both_present = search(["Start", "Second substrate"])

        self.assertEqual(missing_one["route_count"], 0)
        self.assertEqual(both_present["route_count"], 1)
        self.assertTrue(both_present["routes"][0]["route_validation"]["valid"])
        reaction = both_present["and_or_tree"]["reaction_nodes"][0]
        self.assertEqual(reaction["node_type"], "AND")
        self.assertEqual(len(reaction["precursor_keys"]), 2)
        route_reaction = both_present["routes"][0]["retrosynthesis_reactions"][0]
        self.assertEqual(len(route_reaction["noncurrency_precursors"]), 2)
        self.assertIn(" + ", route_reaction["equation"])
        self.assertIn(chr(0x2190), route_reaction["equation"])

    def test_missing_ec_is_explicit_and_never_invented(self):
        from chemcrow.planner.models import MoleculeNode, ReactionCandidate
        from chemcrow.planner.search import HybridPlannerEngine

        target = MoleculeNode(key="target", query="Target", name="Target")
        start = MoleculeNode(key="start", query="Start", name="Start")
        candidate = ReactionCandidate(
            candidate_id="unknown-ec",
            target=target,
            precursors=(start,),
            source="MODEL",
        )

        self.assertEqual(
            HybridPlannerEngine._enzyme_label(candidate),
            "EC unknown; MODEL",
        )

    def test_uniform_cost_search_switch_disables_heuristic(self):
        tool = EnzymeRetrosynthesisSearch(
            database_service=self.database,
            config={
                "use_database": True,
                "use_model": False,
                "max_expansions": 30,
                "forward_anchor_depth": 0,
            },
        )
        result = json.loads(
            tool.run(
                json.dumps(
                    {
                        "target": "Target",
                        "building_blocks": ["Start"],
                        "search_algorithm": "uniform_cost",
                    }
                )
            )
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["search_algorithm"]["name"], "uniform_cost")
        self.assertEqual(result["config"]["heuristic_weight"], 0.0)

    def test_multi_step_evaluator_returns_requested_top_k(self):
        route_short = {
            "rank": 2,
            "complete": True,
            "all_noncurrency_leaves_solved": True,
            "step_count": 1,
            "target_carbon_efficiency": 0.9,
            "major_organic_byproduct_penalty": 0.0,
            "currency_cofactor_count": 0,
            "steps": [{"candidate_id": "short", "score": 0.9}],
        }
        route_long = {
            "rank": 1,
            "complete": True,
            "all_noncurrency_leaves_solved": True,
            "step_count": 3,
            "target_carbon_efficiency": 0.5,
            "major_organic_byproduct_penalty": 0.4,
            "currency_cofactor_count": 2,
            "steps": [
                {"candidate_id": "long-1", "score": 0.7},
                {"candidate_id": "long-2", "score": 0.7},
                {"candidate_id": "long-3", "score": 0.7},
            ],
        }
        result = json.loads(
            EnzymeMultiStepEvaluator().run(
                json.dumps(
                    {
                        "routes": [route_long, route_short],
                        "top_k": 1,
                        "diversity_penalty": 0.0,
                    }
                )
            )
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["route_count"], 1)
        self.assertEqual(
            result["top_k_routes"][0]["steps"][0]["candidate_id"], "short"
        )
        self.assertEqual(result["top_k_routes"][0]["evaluation_rank"], 1)


if __name__ == "__main__":
    unittest.main()
