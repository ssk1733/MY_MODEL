"""Synthetic SQLite fixture for deterministic planner tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from chemcrow.planner import PlannerConfig


METABOLITES = (
    ("C10001", "Start", "CC"),
    ("C10002", "Intermediate A", "CCO"),
    ("C10003", "Target", "CC=O"),
    ("C10004", "Intermediate B", "CCN"),
    ("C10005", "Second substrate", "CO"),
    ("C10006", "AND target", "CCOC"),
    ("C10007", "Wrong-direction product", "CCC"),
    ("C00001", "H2O", "O"),
)

REACTIONS = (
    ("R10001", "start to A", "C10001 => C10002", "forward", "EC 1.1.1.1"),
    ("R10002", "A to target", "C10002 => C10003", "forward", "EC 1.1.1.1"),
    ("R10003", "start to B", "C10001 => C10004", "forward", "EC 2.2.2.2"),
    ("R10004", "B to target", "C10004 => C10003", "forward", "EC 2.2.2.2"),
    ("R10005", "cycle edge", "C10003 => C10002", "forward", "EC 3.3.3.3"),
    ("R10006", "AND reaction", "C10001 + C10005 => C10006", "forward", "EC 4.4.4.4"),
    ("R10007", "wrong direction", "C10003 => C10007", "forward", "EC 5.5.5.5"),
)

PARTICIPANTS = (
    ("R10001", "left", 1, "1", "C10001"),
    ("R10001", "right", 1, "1", "C10002"),
    ("R10002", "left", 1, "1", "C10002"),
    ("R10002", "left", 2, "1", "C00001"),
    ("R10002", "right", 1, "1", "C10003"),
    ("R10003", "left", 1, "1", "C10001"),
    ("R10003", "right", 1, "1", "C10004"),
    ("R10004", "left", 1, "1", "C10004"),
    ("R10004", "right", 1, "1", "C10003"),
    ("R10005", "left", 1, "1", "C10003"),
    ("R10005", "right", 1, "1", "C10002"),
    ("R10006", "left", 1, "1", "C10001"),
    ("R10006", "left", 2, "1", "C10005"),
    ("R10006", "right", 1, "1", "C10006"),
    ("R10007", "left", 1, "1", "C10003"),
    ("R10007", "right", 1, "1", "C10007"),
)


def create_test_database(path: Path) -> str:
    connection = sqlite3.connect(str(path))
    connection.executescript(
        """
        CREATE TABLE metabolites (
            id TEXT PRIMARY KEY, name TEXT, name_norm TEXT, compartment TEXT,
            formula TEXT, charge REAL, smiles TEXT, canonical_smiles TEXT,
            charge_parent_smiles TEXT, toxicity REAL
        );
        CREATE TABLE reactions (
            id TEXT PRIMARY KEY, name TEXT, equation TEXT NOT NULL,
            lower_bound REAL, upper_bound REAL, direction TEXT NOT NULL,
            gene_rule TEXT, subsystem TEXT
        );
        CREATE TABLE reaction_participants (
            reaction_id TEXT NOT NULL, side TEXT NOT NULL, position INTEGER NOT NULL,
            coefficient TEXT NOT NULL, metabolite_id TEXT NOT NULL, annotation TEXT,
            PRIMARY KEY (reaction_id, side, position)
        );
        CREATE INDEX idx_metabolites_canonical_smiles ON metabolites(canonical_smiles);
        CREATE INDEX idx_metabolites_name_norm ON metabolites(name_norm);
        CREATE INDEX idx_participants_metabolite ON reaction_participants(metabolite_id);
        CREATE INDEX idx_participants_reaction ON reaction_participants(reaction_id);
        """
    )
    from rdkit import Chem

    for identifier, name, smiles in METABOLITES:
        canonical = Chem.MolToSmiles(
            Chem.MolFromSmiles(smiles), canonical=True, isomericSmiles=True
        )
        connection.execute(
            "INSERT INTO metabolites VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                identifier,
                name,
                name.casefold(),
                "Cytosol",
                None,
                None,
                smiles,
                canonical,
                None,
                0.2,
            ),
        )
    for identifier, name, equation, direction, ec in REACTIONS:
        bounds = (0.0, 1000.0) if direction == "forward" else (-1000.0, 0.0)
        connection.execute(
            "INSERT INTO reactions VALUES (?,?,?,?,?,?,?,?)",
            (identifier, name, equation, bounds[0], bounds[1], direction, ec, "test"),
        )
    for reaction_id, side, position, coefficient, metabolite_id in PARTICIPANTS:
        connection.execute(
            "INSERT INTO reaction_participants VALUES (?,?,?,?,?,NULL)",
            (reaction_id, side, position, coefficient, metabolite_id),
        )
    connection.commit()
    connection.close()
    return str(path)


def small_config(**overrides):
    values = {
        "use_database": True,
        "use_model": False,
        "max_depth": 6,
        "max_expansions": 100,
        "forward_anchor_depth": 1,
        "forward_anchor_max_nodes": 30,
        "database_top_k": 20,
        "candidate_beam_width": 20,
        "global_beam_width": 30,
        "top_k_routes": 5,
        "solution_pool_multiplier": 2,
        "reject_generic_structures": True,
    }
    values.update(overrides)
    return PlannerConfig.from_mapping(values)
