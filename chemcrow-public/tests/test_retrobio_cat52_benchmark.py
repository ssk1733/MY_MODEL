from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from chemcrow.benchmarks.retrobiocat52 import (
    build_benchmark,
    evaluate_case_result,
    evaluate_route,
    load_jsonl,
    summarize_case_results,
    validate_route_graph,
)


def connected_route(rank=1, substrates=None, enzymes=None, complete=True):
    substrates = substrates or ["CC=O", "N"]
    return {
        "rank": rank,
        "complete": complete,
        "enzymes": enzymes or ["ADH", "TA"],
        "steps": [
            {
                "target": {"key": "target", "canonical_smiles": "CCN"},
                "precursors": [
                    {"key": f"s{index}", "canonical_smiles": smiles}
                    for index, smiles in enumerate(substrates)
                ],
            }
        ],
    }


class RetroBioCat52AgentBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.case = {
            "benchmark_id": "RBC52-TEST",
            "input": {
                "target_smiles": "CCN",
                "substrate_smiles": ["CC=O", "N"],
            },
            "expected": {"enzyme_types": ["TA", "ADH"]},
        }

    def test_complete_route_connects_exact_substrate_set_order_independently(self):
        result = evaluate_route(self.case, connected_route(substrates=["N", "CC=O"]))
        self.assertTrue(result["metrics"]["complete_route_found"])
        self.assertTrue(result["metrics"]["enzyme_all_present"])

    def test_extra_or_missing_substrate_fails_pathway_and_enzyme_metrics(self):
        extra = connected_route(substrates=["N", "CC=O", "O"])
        missing = connected_route(substrates=["CC=O"])
        for route in (extra, missing):
            result = evaluate_route(self.case, route)
            self.assertFalse(result["metrics"]["complete_route_found"])
            self.assertFalse(result["metrics"]["enzyme_all_present"])

    def test_disconnected_step_is_not_counted_as_part_of_target_route(self):
        route = connected_route()
        route["steps"].append(
            {
                "target": {"key": "orphan", "canonical_smiles": "CCC"},
                "precursors": [
                    {"key": "orphan", "canonical_smiles": "CCC"},
                ],
            }
        )
        result = evaluate_route(self.case, route)
        self.assertFalse(result["metrics"]["complete_route_found"])
        self.assertIn("disconnected_steps", result["route_validation"]["reasons"])

    def test_cycle_and_disconnected_gold_leaves_cannot_fake_completion(self):
        route = {
            "rank": 1,
            "complete": True,
            "enzymes": ["ADH", "TA"],
            "steps": [
                {
                    "target": {"key": "target", "canonical_smiles": "CCN"},
                    "precursors": [{"key": "mid", "canonical_smiles": "CCO"}],
                },
                {
                    "target": {"key": "mid", "canonical_smiles": "CCO"},
                    "precursors": [{"key": "target", "canonical_smiles": "CCN"}],
                },
                {
                    "target": {"key": "orphan", "canonical_smiles": "CCC"},
                    "precursors": [
                        {"key": "a", "canonical_smiles": "CC=O"},
                        {"key": "b", "canonical_smiles": "N"},
                    ],
                },
            ],
        }
        validation = validate_route_graph(route, "CCN", ["CC=O", "N"])
        self.assertFalse(validation["valid"])
        self.assertFalse(validation["acyclic"])
        self.assertFalse(validation["connected"])
        result = evaluate_route(self.case, route)
        self.assertFalse(result["metrics"]["complete_route_found"])
        self.assertFalse(result["metrics"]["enzyme_all_present"])

    def test_empty_or_incomplete_route_cannot_score(self):
        empty = {"rank": 1, "complete": True, "steps": [], "enzymes": ["TA", "ADH"]}
        incomplete = connected_route(complete=False)
        for route in (empty, incomplete):
            result = evaluate_route(self.case, route)
            self.assertFalse(result["metrics"]["complete_route_found"])
            self.assertFalse(result["metrics"]["enzyme_all_present"])

    def test_enzyme_metric_requires_all_gold_enzymes_on_same_complete_route(self):
        covered = connected_route(enzymes=["P450", "TA", "ADH"])
        missing = connected_route(enzymes=["TA"])
        self.assertTrue(evaluate_route(self.case, covered)["metrics"]["enzyme_all_present"])
        self.assertFalse(evaluate_route(self.case, missing)["metrics"]["enzyme_all_present"])

    def test_top_k_uses_first_matching_route(self):
        wrong = connected_route(rank=1, substrates=["CCO"], enzymes=["TA"])
        correct = connected_route(rank=2)
        result = evaluate_case_result(
            self.case,
            {"ok": True, "routes": [wrong, correct]},
        )
        self.assertEqual(result["first_hit_rank"]["complete_route_found"], 2)
        self.assertEqual(result["first_hit_rank"]["enzyme_all_present"], 2)
        result["wall_seconds"] = 1.0
        summary = summarize_case_results([result], [1, 3, 10])
        self.assertEqual(summary["pathway_accuracy_at_k"]["1"], 0.0)
        self.assertEqual(summary["pathway_accuracy_at_k"]["3"], 1.0)
        self.assertEqual(summary["enzyme_accuracy_at_k"]["10"], 1.0)
        self.assertEqual(summary["complete_pathway_success"]["fraction"], "1/1")
        self.assertEqual(summary["complete_pathway_success"]["success_rate"], 1.0)
        self.assertEqual(
            summary["enzyme_complete_pathway_success"]["fraction"], "1/1"
        )
        self.assertEqual(summary["enzyme_top_k"]["1"]["fraction"], "0/1")
        self.assertEqual(summary["enzyme_top_k"]["3"]["fraction"], "1/1")
        self.assertEqual(summary["enzyme_top_k"]["10"]["fraction"], "1/1")

    def test_summary_reports_counts_over_52(self):
        results = []
        for index in range(52):
            enzyme_rank = 1 if index < 2 else 2 if index == 2 else 5 if index == 3 else None
            results.append(
                {
                    "ok": True,
                    "first_hit_rank": {
                        "complete_route_found": 1 if index < 10 else None,
                        "enzyme_all_present": enzyme_rank,
                    },
                }
            )

        summary = summarize_case_results(results, [1, 3, 10])

        self.assertEqual(summary["complete_pathway_success"]["fraction"], "10/52")
        self.assertEqual(summary["enzyme_complete_pathway_success"]["fraction"], "4/52")
        self.assertEqual(summary["enzyme_top_k"]["1"]["fraction"], "2/52")
        self.assertEqual(summary["enzyme_top_k"]["3"]["fraction"], "3/52")
        self.assertEqual(summary["enzyme_top_k"]["10"]["fraction"], "4/52")
        self.assertEqual(
            summary["complete_pathway_success"]["success_rate"],
            round(10 / 52, 6),
        )

    def test_source_workbook_builds_target_plus_substrates_cases(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "dataset"
            / "RetroBioCat 52-pathway test set.xlsx"
        )
        if not source.exists():
            self.skipTest("source RetroBioCat workbook is outside this checkout")
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_benchmark(source, directory)
            cases = load_jsonl(Path(directory) / "benchmark.jsonl")
        self.assertEqual(manifest["case_count"], 52)
        self.assertEqual(
            manifest["schema_version"],
            "retrobiocat52.agent_target_substrates.v5",
        )
        self.assertEqual(manifest["files"], {"benchmark_jsonl": "benchmark.jsonl"})
        self.assertEqual(len(cases), 52)
        self.assertTrue(all(case["input"]["target_smiles"] for case in cases))
        self.assertTrue(all(case["input"]["substrate_smiles"] for case in cases))
        self.assertTrue(all(set(case["expected"]) == {"enzyme_types"} for case in cases))


if __name__ == "__main__":
    unittest.main()
