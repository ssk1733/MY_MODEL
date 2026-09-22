from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from chemcrow.planner import EnzRetroClient, EnzymeReactionDatabase
from chemcrow.planner.scoring import fuse_candidates, score_and_rank_candidates

from planner_test_utils import create_test_database, small_config


class FakePredictor:
    def __init__(self):
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        return [
            {
                "reactants": "CCO",
                "scores": 0.8,
                "templates": "test-template[EC]1.1.1.1",
                "ec": "1.1.1.1",
                "similarity": 0.7,
                "feasibility": 0.9,
                "yield": 0.75,
                "toxicity": 0.1,
                "topk": "Top-1",
            }
        ]


class CandidateFusionTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        path = create_test_database(Path(self.tempdir.name) / "planner.db")
        self.database = EnzymeReactionDatabase(path)
        self.predictor = FakePredictor()
        self.client = EnzRetroClient(predictor=self.predictor)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_model_cache_and_database_model_fusion(self):
        target = self.database.normalize_molecule("Target")
        blocks = (self.database.normalize_molecule("Start"),)
        model_first = self.client.predict_candidates(
            target, blocks, normalizer=self.database.normalize_molecule
        )
        model_second = self.client.predict_candidates(
            target, blocks, normalizer=self.database.normalize_molecule
        )
        self.assertEqual(self.predictor.calls, 1)
        self.assertEqual(model_first, model_second)
        database = self.database.backward_candidates(target)
        fused = fuse_candidates(database, model_first, small_config())
        both = [candidate for candidate in fused if candidate.source == "BOTH"]
        self.assertEqual(len(both), 1)
        self.assertEqual(both[0].database_reaction_ids, ("R10002",))
        self.assertEqual(both[0].model_score, 0.8)

    def test_masked_scoring_and_consensus_bonus(self):
        target = self.database.normalize_molecule("Target")
        blocks = (self.database.normalize_molecule("Start"),)
        database = self.database.backward_candidates(target)
        model = self.client.predict_candidates(
            target, blocks, normalizer=self.database.normalize_molecule
        )
        fused = fuse_candidates(database, model, small_config())
        scored = score_and_rank_candidates(fused, blocks, small_config())
        both = next(candidate for candidate in scored if candidate.source == "BOTH")
        database_only = next(
            candidate for candidate in scored if candidate.source == "DATABASE"
        )
        self.assertGreater(both.step_score, database_only.step_score)
        self.assertIsNone(both.atom_conservation)
        self.assertIn("approx_structural_overlap", both.score_components)
        self.assertNotIn("inverse_toxicity", both.score_components)
        self.assertNotIn("yield_score", both.score_components)
        self.assertEqual(
            both.metadata["score_provenance"]["aggregation"],
            "MASKED_WEIGHTED_AVERAGE",
        )


if __name__ == "__main__":
    unittest.main()
