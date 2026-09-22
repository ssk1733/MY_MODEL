from __future__ import annotations

import unittest

from chemcrow.planner import EnzymeReactionDatabase, HybridPlannerEngine, PlannerConfig


GLUCOSE = "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"


class RealDatabaseR04410RegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.database = EnzymeReactionDatabase()

    def test_real_sqlite_lookup_uses_participant_indexes(self):
        diagnostics = self.database.sqlite_index_diagnostics()
        indexes = diagnostics["reaction_participant_indexes"]
        self.assertIn("idx_participants_metabolite", indexes)
        self.assertIn("idx_participants_reaction", indexes)
        plan = " ".join(
            str(row[-1]) for row in diagnostics["metabolite_lookup_plan"]
        )
        self.assertIn("idx_participants_metabolite", plan)

    def test_r04410_is_not_direct_glucose_to_ethanol_route(self):
        target = self.database.normalize_molecule("CCO")
        glucose = self.database.normalize_molecule(GLUCOSE)
        candidates = self.database.backward_candidates(target, limit=200)
        shortcut = next(
            candidate
            for candidate in candidates
            if "R04410" in candidate.database_reaction_ids
        )

        required = {item.key for item in shortcut.noncurrency_precursors}
        self.assertIn(glucose.key, required)
        self.assertEqual(len(required), 2)
        self.assertNotEqual(required, {glucose.key})

        output = HybridPlannerEngine(
            database=self.database,
            config=PlannerConfig.from_mapping(
                {
                    "use_database": True,
                    "use_model": False,
                    "forward_anchor_depth": 0,
                    "max_depth": 1,
                    "max_expansions": 5,
                    "database_top_k": 200,
                    "candidate_beam_width": 200,
                    "global_beam_width": 200,
                    "top_k_routes": 10,
                }
            ),
        ).plan("CCO", [GLUCOSE])
        self.assertFalse(
            any(
                route["step_count"] == 1
                and route["steps"][0]["database_reaction_ids"] == ["R04410"]
                for route in output["routes"]
            )
        )


if __name__ == "__main__":
    unittest.main()
