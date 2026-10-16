from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from chemcrow.planner import EnzymeReactionDatabase
from chemcrow.tools import MoleculeInputNormalizer

from planner_test_utils import create_test_database


class MoleculeInputNormalizerTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        path = create_test_database(Path(self.tempdir.name) / "normalizer.db")
        self.database = EnzymeReactionDatabase(path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_canonicalizes_smiles(self):
        result = json.loads(
            MoleculeInputNormalizer(database_service=self.database).run("OCC")
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["canonical_smiles"], "CCO")
        self.assertEqual(
            result["normalized_molecules"][0]["matched_by"], "smiles"
        )

    def test_valid_smiles_wins_over_ambiguous_exact_database_name(self):
        connection = sqlite3.connect(self.database.db_path)
        connection.execute(
            "INSERT INTO metabolites VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "C19999", "CO", "co", "Cytosol", "CO", 0.0,
                "[C-]#[O+]", "[C-]#[O+]", None, 0.2,
            ),
        )
        connection.commit()
        connection.close()

        database = EnzymeReactionDatabase(self.database.db_path)
        result = json.loads(
            MoleculeInputNormalizer(database_service=database).run("CO")
        )
        molecule = result["normalized_molecules"][0]
        self.assertTrue(result["ok"])
        self.assertEqual(molecule["canonical_smiles"], "CO")
        self.assertEqual(molecule["database_id"], "C10005")
        self.assertEqual(molecule["matched_by"], "smiles")

    def test_resolves_exact_local_name(self):
        result = json.loads(
            MoleculeInputNormalizer(database_service=self.database).run("Target")
        )
        self.assertTrue(result["ok"])
        molecule = result["normalized_molecules"][0]
        self.assertEqual(molecule["database_id"], "C10003")
        self.assertEqual(molecule["canonical_smiles"], "CC=O")
        self.assertEqual(molecule["matched_by"], "local_exact_name")

    def test_normalizes_target_and_building_blocks_together(self):
        result = json.loads(
            MoleculeInputNormalizer(database_service=self.database).run(
                json.dumps(
                    {
                        "target": "Target",
                        "building_blocks": ["Start", "OCC"],
                    }
                )
            )
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["normalized_input"]["target"], "CC=O")
        self.assertEqual(
            result["normalized_input"]["building_blocks"], ["CC", "CCO"]
        )

    def test_external_name_fallback_is_canonicalized(self):
        tool = MoleculeInputNormalizer(
            database_service=self.database,
            name_resolver=lambda _: "OCC",
        )
        result = json.loads(tool.run("external compound"))
        self.assertTrue(result["ok"])
        self.assertEqual(result["canonical_smiles"], "CCO")
        self.assertEqual(
            result["normalized_molecules"][0]["matched_by"], "external_name"
        )

    def test_failure_blocks_downstream_calls(self):
        tool = MoleculeInputNormalizer(
            database_service=self.database,
            name_resolver=lambda _: "not a smiles",
        )
        result = json.loads(tool.run("unknown compound"))
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["type"], "normalization_failed")
        self.assertEqual(result["normalized_molecules"], [])


    def test_multiple_targets_are_normalized_without_dropping_any(self):
        result = json.loads(
            MoleculeInputNormalizer(database_service=self.database).run(
                json.dumps(
                    {
                        "targets": ["CCO", "CCN"],
                        "building_blocks": ["O"],
                        "allow_external_lookup": False,
                    }
                )
            )
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["normalized_input"]["targets"], ["CCO", "CCN"])
        self.assertIsNone(result["normalized_input"]["target"])
        self.assertEqual(result["normalized_input"]["building_blocks"], ["O"])


if __name__ == "__main__":
    unittest.main()
