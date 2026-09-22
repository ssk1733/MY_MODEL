from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from chemcrow.planner import EnzRetroClient, EnzymeReactionDatabase
from chemcrow.planner.scoring import score_candidate

from planner_test_utils import create_test_database, small_config


class PlaceholderMetricPredictor:
    def __init__(self, yield_value: float, toxicity_value: float):
        self.yield_value = yield_value
        self.toxicity_value = toxicity_value
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        return [
            {
                "reactants": "CCO",
                "scores": 0.8,
                "templates": "template[EC]1.1.1.1",
                "ec": "1.1.1.1",
                "similarity": 0.7,
                "feasibility": 0.9,
                "yield": self.yield_value,
                "toxicity": self.toxicity_value,
                "topk": "Top-1",
            }
        ]


class MetricDeterminismTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        path = create_test_database(Path(self.tempdir.name) / "planner.db")
        self.database = EnzymeReactionDatabase(path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_random_placeholder_metrics_do_not_change_default_score(self):
        target = self.database.normalize_molecule("Target")
        blocks = (self.database.normalize_molecule("Start"),)
        low = PlaceholderMetricPredictor(0.01, 0.99)
        high = PlaceholderMetricPredictor(0.99, 0.01)
        low_candidate = EnzRetroClient(predictor=low).predict_candidates(
            target, blocks, normalizer=self.database.normalize_molecule
        )[0]
        high_candidate = EnzRetroClient(predictor=high).predict_candidates(
            target, blocks, normalizer=self.database.normalize_molecule
        )[0]

        low_scored = score_candidate(low_candidate, blocks, small_config())
        high_scored = score_candidate(high_candidate, blocks, small_config())
        self.assertNotEqual(low_scored.yield_score, high_scored.yield_score)
        self.assertNotEqual(low_scored.toxicity, high_scored.toxicity)
        self.assertAlmostEqual(low_scored.step_score, high_scored.step_score, places=12)
        self.assertNotIn("yield_score", low_scored.score_components)
        self.assertNotIn("inverse_toxicity", low_scored.score_components)
        self.assertEqual(
            low_scored.metadata["metric_provenance"]["yield_score"],
            "MOORETRO_RANDOM_PLACEHOLDER_DISABLED_BY_DEFAULT",
        )


if __name__ == "__main__":
    unittest.main()
