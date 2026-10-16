from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from chemcrow.planner.models import MoleculeNode, ReactionCandidate

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_paroutes_tools.py"
SPEC = importlib.util.spec_from_file_location("evaluate_paroutes_tools", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeNormalizer:
    def run(self, value):
        payload = json.loads(value)
        targets = payload["targets"]
        return json.dumps({
            "ok": True,
            "normalized_input": {"targets": targets, "target": targets[0], "building_blocks": []},
        })


class FakeModelClient:
    def cache_info(self):
        return {"hits": 0, "misses": 0, "entries": 0, "model": "biocyc",
                "backend_mode": "neural_model", "runtime_device": "cuda:0"}


def mol(smiles):
    smiles = MODULE.canonical(smiles)
    return MoleculeNode(key="smi:" + smiles, query=smiles, smiles=smiles,
                        canonical_smiles=smiles)


def candidate(target, precursors, candidate_id="candidate", source="MODEL", score=0.8):
    target_node = mol(target)
    precursor_nodes = tuple(mol(value) for value in precursors)
    return ReactionCandidate(
        candidate_id=candidate_id,
        target=target_node,
        precursors=precursor_nodes,
        source=source,
        main_precursor_keys=tuple(item.key for item in precursor_nodes),
        model_score=score,
        step_score=score,
        step_cost=1.0 - score,
    )


def reference_tree():
    return {
        "type": "mol", "smiles": "CCOC", "in_stock": False,
        "children": [{
            "type": "reaction", "metadata": {"reaction_hash": "r1", "ID": "p1"},
            "children": [
                {"type": "mol", "smiles": "CCO", "in_stock": False,
                 "children": [{
                     "type": "reaction", "metadata": {"reaction_hash": "r2", "ID": "p2"},
                     "children": [
                         {"type": "mol", "smiles": "CC", "in_stock": True, "children": []},
                         {"type": "mol", "smiles": "O", "in_stock": True, "children": []},
                     ],
                 }]},
                {"type": "mol", "smiles": "C", "in_stock": True, "children": []},
            ],
        }],
    }


class PaRoutesTeacherForcedTest(unittest.TestCase):
    def test_timestamp_hierarchy(self):
        value = datetime(2026, 9, 24, 7, 5, tzinfo=timezone.utc)
        self.assertEqual(MODULE.timestamp_output_dir(Path("/results"), value),
                         Path("/results/2026/09/24/07/05"))

    def test_streams_json_array_across_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routes.json"
            path.write_text(json.dumps([{"i": 1}, {"i": 2}, {"i": 3}]))
            values = list(MODULE.iter_json_array(path, chunk_size=4))
        self.assertEqual([item["i"] for item in values], [1, 2, 3])

    def test_extracts_preorder_steps_and_route_leaves(self):
        reference = MODULE.extract_reference(reference_tree())
        self.assertEqual(reference["target_smiles"], "CCOC")
        self.assertEqual(reference["step_count"], 2)
        self.assertEqual(set(reference["substrate_smiles"]), {"C", "CC", "O"})
        self.assertEqual(reference["steps"][0]["target_smiles"], "CCOC")
        self.assertEqual(reference["steps"][1]["target_smiles"], "CCO")

    def test_candidate_hit_requires_all_reference_precursors_but_allows_extras(self):
        value = candidate("CCOC", ["CCO", "C", "O"])
        self.assertTrue(MODULE.candidate_contains_reference(value, ["CCO", "C"]))
        self.assertFalse(MODULE.candidate_contains_reference(value, ["CCO", "N"]))

    def test_hit_at_any_rank_continues_then_first_miss_stops(self):
        reference = {
            "target_smiles": "CCOC",
            "steps": [
                {"step_index": 1, **MODULE.signature_record("CCOC", ["CCO", "C"])},
                {"step_index": 2, **MODULE.signature_record("CCO", ["CC", "O"])},
                {"step_index": 3, **MODULE.signature_record("CC", ["C", "C"])},
            ],
            "step_count": 3,
        }
        calls = []

        def generated(target, *args, **kwargs):
            calls.append(target)
            if len(calls) == 1:
                ranked = (
                    candidate(target, ["N"], "wrong-1"),
                    candidate(target, ["CCO", "C"], "correct-rank-2"),
                )
            else:
                ranked = (candidate(target, ["N"], "wrong-2"),)
            return {
                "target": mol(target).to_dict(),
                "ranked_candidates": ranked,
                "database_candidate_count": 0,
                "model_candidate_count": len(ranked),
                "fused_candidate_count": len(ranked),
                "returned_candidate_count": len(ranked),
                "filtered_candidate_count": 0,
                "filter_reasons": {},
                "warnings": [],
                "model_cache_delta": {"hits": 0, "misses": 1,
                                      "backend_mode": "neural_model",
                                      "runtime_device": "cuda:0"},
            }

        with mock.patch.object(MODULE, "generate_ranked_candidates", side_effect=generated):
            result = MODULE.run_teacher_forced_case(
                FakeNormalizer(), object(), FakeModelClient(), (mol("C"),),
                reference, object(), report_top_ks=[1, 3, 10],
                candidate_limit=100, saved_candidate_count=20,
            )
        self.assertEqual(calls, ["CCOC", "CCO"])
        self.assertEqual(result["completed_single_steps"], 1)
        self.assertEqual(result["single_step_fraction"], "1/3")
        self.assertEqual(result["stopped_at_step"], 2)
        self.assertEqual(result["step_results"][0]["hit_rank"], 2)
        self.assertFalse(result["step_results"][0]["top_k_hits"]["1"])
        self.assertTrue(result["step_results"][0]["top_k_hits"]["3"])
        self.assertFalse(result["step_results"][2]["attempted"])

    def test_summary_top_k_denominator_is_every_reference_step(self):
        summary = MODULE.summarize_records([
            {"split": "n1", "ok": True, "complete_multistep_prediction": False,
             "completed_single_steps": 1, "total_single_steps": 3,
             "attempted_step_count": 2,
             "top_k_hit_counts": {"1": 0, "3": 1, "10": 1}, "wall_seconds": 2.0},
            {"split": "n1", "ok": True, "complete_multistep_prediction": True,
             "completed_single_steps": 2, "total_single_steps": 2,
             "attempted_step_count": 2,
             "top_k_hit_counts": {"1": 1, "3": 2, "10": 2}, "wall_seconds": 4.0},
        ], [1, 3, 10])
        n1 = summary["splits"]["n1"]
        self.assertEqual(n1["complete_multistep_paths"]["fraction"], "1/2")
        self.assertEqual(n1["single_steps_successful_at_any_returned_rank"]["fraction"], "3/5")
        self.assertEqual(n1["single_step_top_k_accuracy"]["1"]["fraction"], "1/5")
        self.assertEqual(n1["single_step_top_k_accuracy"]["3"]["fraction"], "3/5")
        self.assertEqual(n1["unattempted_after_miss"], 1)


if __name__ == "__main__":
    unittest.main()
