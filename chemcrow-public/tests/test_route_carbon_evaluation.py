from __future__ import annotations

import unittest

from chemcrow.planner.models import MoleculeNode, ReactionCandidate
from chemcrow.planner.route_evaluation import evaluate_route_carbon


def molecule(key: str, name: str, smiles: str) -> MoleculeNode:
    return MoleculeNode(
        key=key,
        query=name,
        name=name,
        smiles=smiles,
        canonical_smiles=smiles,
    )


class RouteCarbonEvaluationTest(unittest.TestCase):
    def test_carbon_efficiency_and_major_organic_byproduct(self):
        glucose = molecule("glucose", "Glucose", "OCC1OC(O)C(O)C(O)C1O")
        pyruvate = molecule("pyruvate", "Pyruvate", "CC(=O)C(=O)O")
        glyceraldehyde = molecule("glyceraldehyde", "Glyceraldehyde", "O=CC(O)CO")
        acetaldehyde = molecule("acetaldehyde", "Acetaldehyde", "CC=O")
        ethanol = molecule("ethanol", "Ethanol", "CCO")

        steps = (
            ReactionCandidate(
                candidate_id="split",
                target=pyruvate,
                precursors=(glucose,),
                source="DATABASE",
                main_precursor_keys=(glucose.key,),
                all_products=(pyruvate, glyceraldehyde),
                coefficients=("1",),
                product_coefficients=("1", "1"),
            ),
            ReactionCandidate(
                candidate_id="decarboxylate",
                target=acetaldehyde,
                precursors=(pyruvate,),
                source="DATABASE",
                main_precursor_keys=(pyruvate.key,),
                all_products=(acetaldehyde,),
                coefficients=("1",),
                product_coefficients=("1",),
            ),
            ReactionCandidate(
                candidate_id="reduce",
                target=ethanol,
                precursors=(acetaldehyde,),
                source="DATABASE",
                main_precursor_keys=(acetaldehyde.key,),
                all_products=(ethanol,),
                coefficients=("1",),
                product_coefficients=("1",),
            ),
        )
        result = evaluate_route_carbon(ethanol, (glucose,), steps, ())
        self.assertAlmostEqual(result["target_carbon_efficiency"], 2 / 6)
        self.assertAlmostEqual(result["main_branch_carbon_retention"], 2 / 6)
        self.assertAlmostEqual(result["major_organic_byproduct_penalty"], 3 / 6)
        self.assertEqual(
            result["major_organic_byproducts"][0]["name"], "Glyceraldehyde"
        )
        self.assertFalse(result["atom_mapping_used"])
        self.assertIn("APPROXIMATION", result["carbon_evaluation_method"])


if __name__ == "__main__":
    unittest.main()
