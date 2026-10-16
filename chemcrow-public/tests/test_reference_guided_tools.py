from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from chemcrow.planner.config import PlannerConfig
from chemcrow.planner.model_client import EnzRetroClient
from chemcrow.planner.models import MoleculeNode, ReactionCandidate
from chemcrow.planner.scoring import fuse_candidates_by_complete_precursors

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_reference_guided_tools.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_reference_guided_tools", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def mol(smiles: str) -> MoleculeNode:
    return MODULE.node_for_smiles(smiles)


def candidate(
    target: str,
    precursors: list[str],
    candidate_id: str,
    source: str = "MODEL",
    coefficients: tuple[str, ...] | None = None,
) -> ReactionCandidate:
    precursor_nodes = tuple(mol(value) for value in precursors)
    return ReactionCandidate(
        candidate_id=candidate_id,
        target=mol(target),
        precursors=precursor_nodes,
        source=source,
        main_precursor_keys=tuple(item.key for item in precursor_nodes),
        coefficients=coefficients or tuple("1" for _ in precursor_nodes),
        model_score=0.8 if source == "MODEL" else None,
        database_support=0.8 if source == "DATABASE" else None,
    )


def reference_tree():
    return {
        "type": "mol",
        "smiles": "CCOC",
        "in_stock": False,
        "children": [{
            "type": "reaction",
            "smiles": "",
            "metadata": {
                "smiles": "[CH3:1][CH2:2]O.C>>[CH3:1][CH2:2]OC",
                "ID": "r1",
            },
            "children": [
                {
                    "type": "mol",
                    "smiles": "CCO",
                    "in_stock": False,
                    "children": [{
                        "type": "reaction",
                        "smiles": "",
                        "metadata": {"ID": "r2"},
                        "children": [
                            {
                                "type": "mol",
                                "smiles": "CC",
                                "in_stock": True,
                                "children": [],
                            },
                            {
                                "type": "mol",
                                "smiles": "O",
                                "in_stock": True,
                                "children": [],
                            },
                        ],
                    }],
                },
                {
                    "type": "mol",
                    "smiles": "C",
                    "in_stock": True,
                    "children": [],
                },
            ],
        }],
    }


class ReferenceGuidedToolsTest(unittest.TestCase):
    def test_canonicalization_clears_atom_maps_but_keeps_stereo(self):
        self.assertEqual(MODULE.canonical("[CH3:1][CH2:2]O"), "CCO")
        left = MODULE.canonical("N[C@@H](C)C(=O)O")
        right = MODULE.canonical("N[C@H](C)C(=O)O")
        self.assertNotEqual(left, right)

    def test_extracts_stable_dfs_steps_and_route_leaves(self):
        result = MODULE.extract_reference_route(
            reference_tree(),
            split="n1",
            source_file="n1-routes.json",
            source_route_index=0,
        )
        self.assertEqual(result["route_id"], "n1-000001")
        self.assertEqual(result["total_steps"], 2)
        self.assertEqual(
            [step["product"] for step in result["reference_steps"]],
            ["CCOC", "CCO"],
        )
        self.assertEqual(
            result["reference_steps"][0]["downstream_step_ids"],
            ["n1-000001:step:0002"],
        )
        self.assertEqual(
            [item["smiles"] for item in result["starting_materials"]],
            ["CC", "O", "C"],
        )

    def test_signature_is_exact_multiset_and_expands_integer_coefficients(self):
        collapsed = candidate(
            "CCOC", ["CCO", "O"], "db", "DATABASE", ("1", "2")
        )
        expanded = candidate("CCOC", ["CCO", "O", "O"], "model")
        missing = candidate("CCOC", ["CCO", "O"], "missing")
        self.assertEqual(
            MODULE.candidate_signature(collapsed),
            MODULE.candidate_signature(expanded),
        )
        self.assertNotEqual(
            MODULE.candidate_signature(collapsed),
            MODULE.candidate_signature(missing),
        )

    def test_complete_fusion_does_not_consume_same_main_different_auxiliary(self):
        config = PlannerConfig.from_mapping(
            {"use_database": True, "use_model": True}
        )
        database = candidate("CCOC", ["CCO", "O"], "db", "DATABASE")
        exact_model = candidate("CCOC", ["O", "CCO"], "model-exact")
        different_model = candidate("CCOC", ["CCO", "N"], "model-other")
        fused = fuse_candidates_by_complete_precursors(
            [database], [different_model, exact_model], config
        )
        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0].source, "BOTH")
        signatures = {MODULE.candidate_signature(item) for item in fused}
        self.assertIn(MODULE.candidate_signature(different_model), signatures)

    def test_model_audit_keeps_invalid_raw_prediction(self):
        def predictor(**kwargs):
            return [
                {"all_reactants": "CC.O", "scores": 0.9},
                {"all_reactants": "not-a-smiles", "scores": 0.1},
            ]

        client = EnzRetroClient(predictor=predictor)
        values, audit = client.predict_candidates_with_audit(
            mol("CCO"),
            [mol("CC")],
            top_k=5,
            normalizer=MODULE.node_for_smiles,
        )
        self.assertEqual(len(values), 1)
        self.assertEqual(audit["raw_returned"], 2)
        self.assertEqual(audit["valid_returned"], 1)
        self.assertFalse(audit["records"][1]["valid"])
        self.assertIn("invalid_precursor", audit["records"][1]["invalid_reason"])

    def test_first_miss_stops_route_and_top_k_uses_attempted_steps(self):
        reference = MODULE.extract_reference_route(
            reference_tree(),
            split="n1",
            source_file="n1-routes.json",
            source_route_index=0,
        )
        calls = []
        correct = candidate("CCOC", ["CCO", "C"], "correct", "DATABASE")
        wrong_first = candidate("CCOC", ["N"], "wrong-first")
        wrong_second = candidate("CCO", ["N"], "wrong-second")

        def generated(**kwargs):
            calls.append(kwargs["step_id"])
            if len(calls) == 1:
                db_values, model_values = (correct,), (wrong_first,)
                ranked = (wrong_first, correct)
            else:
                db_values, model_values = (), (wrong_second,)
                ranked = (wrong_second,)
            return {
                "database_candidates": db_values,
                "model_candidates": model_values,
                "ranked_candidates": ranked,
                "candidate_records": [],
                "tool_calls": [],
                "errors": {},
                "counts": {
                    "database": {
                        "requested": 2,
                        "raw_returned": len(db_values),
                        "valid_returned": len(db_values),
                    },
                    "model": {
                        "requested": 2,
                        "raw_returned": len(model_values),
                        "valid_returned": len(model_values),
                    },
                    "merged_count": len(ranked),
                    "ranked_count": len(ranked),
                },
            }

        with mock.patch.object(
            MODULE, "generate_step_candidates", side_effect=generated
        ):
            result = MODULE.evaluate_route(
                reference,
                database=object(),
                model_client=object(),
                split_stock=(),
                building_block_strategy="route_leaves",
                config=PlannerConfig.from_mapping(
                    {"use_database": True, "use_model": True}
                ),
                n_db=2,
                n_model=2,
            )
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["route"]["status"], "miss")
        self.assertEqual(result["route"]["coverage_text"], "1/2")
        self.assertEqual(result["route"]["attempted_steps"], 2)
        self.assertFalse(result["steps"][0]["top1_hit"])
        self.assertTrue(result["steps"][0]["top2_hit"])
        self.assertEqual(result["steps"][1]["status"], "miss")

    def test_validation_uses_raw_stock_leaf_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "n1-routes.json").write_text(
                json.dumps([reference_tree()]), encoding="utf-8"
            )
            (root / "n1-targets.txt").write_text("CCOC\n", encoding="utf-8")
            (root / "n1-stock.txt").write_text("CC\nO\nC\n", encoding="utf-8")
            result = MODULE.validate_split(root, "n1")
        self.assertEqual(result["route_count"], 1)
        self.assertEqual(result["reaction_node_count"], 2)
        self.assertTrue(result["stock_equals_reference_leaf_union"])

    def test_summary_excludes_error_routes_from_main_denominators(self):
        routes = [
            {
                "route_id": "n1-000001",
                "status": "miss",
                "found_steps": 1,
                "total_steps": 2,
                "coverage_ratio": 0.5,
            },
            {
                "route_id": "n1-000002",
                "status": "error",
                "found_steps": 1,
                "total_steps": 3,
                "coverage_ratio": 1 / 3,
            },
        ]
        steps = [
            {
                "route_id": "n1-000001",
                "status": "hit",
                "top1_hit": True,
                "top2_hit": True,
                "top3_hit": True,
                "database_hit": True,
                "model_hit": False,
                "both_hit": False,
                "hybrid_hit": True,
            },
            {
                "route_id": "n1-000001",
                "status": "miss",
                "top1_hit": False,
                "top2_hit": False,
                "top3_hit": False,
                "database_hit": False,
                "model_hit": False,
                "both_hit": False,
                "hybrid_hit": False,
            },
            {
                "route_id": "n1-000002",
                "status": "hit",
                "top1_hit": True,
                "database_hit": True,
                "model_hit": True,
                "both_hit": True,
                "hybrid_hit": True,
            },
        ]
        summary = MODULE.summarize(routes, steps)
        self.assertEqual(summary["route_outcomes"]["scored_routes"], 1)
        self.assertEqual(
            summary["top_k_accuracy_on_attempted_steps"]["1"]["denominator"], 2
        )
        self.assertEqual(
            summary["aggregate_step_coverage"]["numerator"], 1
        )


if __name__ == "__main__":
    unittest.main()
