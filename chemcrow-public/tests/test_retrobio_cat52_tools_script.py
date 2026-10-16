from __future__ import annotations

import importlib.util
import json
import unittest
from datetime import datetime, timezone
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_retrobio_cat52_tools.py"
)
SPEC = importlib.util.spec_from_file_location("evaluate_retrobio_cat52_tools", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeTool:
    def __init__(self, result):
        self.result = result
        self.inputs = []

    def run(self, value):
        self.inputs.append(json.loads(value))
        return json.dumps(self.result)


class DirectToolsBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.case = {
            "benchmark_id": "RBC52-TEST",
            "input": {
                "target_smiles": "CCN",
                "substrate_smiles": ["CC=O", "N"],
            },
            "expected": {"enzyme_types": ["TA"]},
        }
        self.normalizer = FakeTool(
            {
                "ok": True,
                "normalized_input": {
                    "targets": ["CCN"],
                    "building_blocks": ["CC=O", "N"],
                },
            }
        )
        self.search = FakeTool(
            {
                "ok": True,
                "routes": [
                    {
                        "rank": 1,
                        "complete": True,
                        "enzymes": ["TA"],
                        "steps": [
                            {
                                "target": {"canonical_smiles": "CCN"},
                                "precursors": [
                                    {"canonical_smiles": "CC=O"},
                                    {"canonical_smiles": "N"},
                                ],
                            }
                        ],
                    }
                ],
                "stats": {},
                "candidate_generation": {"model_backend": "neural_model"},
            }
        )

    def test_timestamp_directory_has_requested_hierarchy(self):
        value = datetime(2026, 9, 22, 7, 5, tzinfo=timezone.utc)
        output = MODULE.timestamp_output_dir(Path("/results"), value)
        self.assertEqual(output, Path("/results/2026/09/22/07/05"))

    def test_direct_calls_simulate_normalization_then_exact_hybrid_search(self):
        result = MODULE.run_tool_case(
            self.normalizer,
            self.search,
            self.case,
            top_k=10,
            max_depth=8,
            max_expansions=50,
            max_model_calls=12,
        )
        self.assertEqual(
            [call["tool"] for call in result["tool_calls"]],
            ["MoleculeInputNormalizer", "EnzymeRetrosynthesisSearch"],
        )
        self.assertNotIn("expected", self.normalizer.inputs[0])
        payload = self.search.inputs[0]
        self.assertEqual(payload["targets"], ["CCN"])
        self.assertEqual(set(payload["building_blocks"]), {"CC=O", "N"})
        self.assertEqual(payload["endpoint_mode"], "exact_set")
        self.assertEqual(payload["search_algorithm"], "and_or_mcts")
        self.assertTrue(payload["config"]["use_database"])
        self.assertTrue(payload["config"]["use_model"])
        evaluated = MODULE.evaluate_case_result(self.case, result["prediction"])
        self.assertEqual(evaluated["first_hit_rank"]["complete_route_found"], 1)
        self.assertEqual(evaluated["first_hit_rank"]["enzyme_all_present"], 1)

    def test_database_lookup_model_backend_is_rejected(self):
        self.search.result["candidate_generation"]["model_backend"] = "database_lookup"
        with self.assertRaisesRegex(RuntimeError, "neural_model"):
            MODULE.run_tool_case(
                self.normalizer,
                self.search,
                self.case,
                top_k=10,
                max_depth=8,
                max_expansions=50,
                max_model_calls=12,
            )


if __name__ == "__main__":
    unittest.main()
