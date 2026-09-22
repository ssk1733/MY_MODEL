"""Read-only SQLite service shared by the multi-step planner.

This module deliberately contains no LangChain dependency.  It follows the same
matching and reaction-direction semantics as ``EnzymeRetroDatabase`` while
returning structured objects suitable for graph search.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .adjacency import SQLiteAdjacencyIndex
from .chem_utils import (
    candidate_structure_metrics,
    canonicalize_smiles,
    classify_precursors,
    extract_ec_numbers,
    is_currency,
    is_generic_structure,
    molecule_key,
    normalized_toxicity,
)
from .config import DEFAULT_CURRENCY_IDS
from .models import MoleculeNode, ReactionCandidate

_METABOLITE_ID_RE = re.compile(r"^C\d{5}$", re.IGNORECASE)


class EnzymeReactionDatabase:
    """Structured, cached, read-only access to ``enzyme_retro.db``."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        *,
        use_memory_index: bool = True,
    ):
        self.db_path = str(
            Path(
                db_path
                or Path(__file__).resolve().parents[1] / "data" / "enzyme_retro.db"
            ).expanduser().resolve()
        )
        self._normalization_cache: Dict[str, MoleculeNode] = {}
        self._backward_cache: Dict[Tuple[object, ...], Tuple[ReactionCandidate, ...]] = {}
        self._forward_cache: Dict[Tuple[object, ...], Tuple[ReactionCandidate, ...]] = {}
        self._lock = threading.RLock()
        self._adjacency = (
            SQLiteAdjacencyIndex(self.db_path) if use_memory_index else None
        )
        self.cache_hits = 0
        self.cache_misses = 0

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            "file:" + self.db_path + "?mode=ro", uri=True, timeout=30.0
        )
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _preferred_row(rows: Sequence[sqlite3.Row]) -> sqlite3.Row:
        return sorted(
            rows,
            key=lambda row: (
                int(str(row["id"])[1:]) >= 30000,
                row["name"] is None,
                row["smiles"] is None,
                row["id"],
            ),
        )[0]

    @staticmethod
    def _row_to_molecule(row: sqlite3.Row, query: Optional[str] = None) -> MoleculeNode:
        canonical = row["canonical_smiles"] or canonicalize_smiles(row["smiles"])
        identifier = row["id"]
        return MoleculeNode(
            key=molecule_key(canonical or row["smiles"], identifier),
            query=query or identifier,
            smiles=row["smiles"],
            canonical_smiles=canonical,
            metabolite_id=identifier,
            name=row["name"],
            formula=row["formula"],
            charge=row["charge"],
            toxicity=normalized_toxicity(row["toxicity"]),
            generic=is_generic_structure(row["smiles"], row["formula"]),
        )

    def normalize_molecule(self, query: str) -> MoleculeNode:
        """Resolve exact ID/name/canonical SMILES or retain a novel valid SMILES."""
        text = str(query or "").strip()
        if not text:
            raise ValueError("Molecule input cannot be empty.")
        cache_key = text.casefold()
        with self._lock:
            cached = self._normalization_cache.get(cache_key)
            if cached is not None:
                self.cache_hits += 1
                return cached
        with self._connect() as connection:
            rows: Sequence[sqlite3.Row]
            if _METABOLITE_ID_RE.match(text):
                rows = connection.execute(
                    "SELECT * FROM metabolites WHERE id = ?", (text.upper(),)
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM metabolites WHERE name_norm = ? LIMIT 20",
                    (text.casefold(),),
                ).fetchall()
                # Some fused-database internal aliases (for example plain
                # "glucose") have no structure. Prefer a structured D/alpha/beta
                # source record when it exists, while retaining exact-name matching.
                if rows and all(not row["smiles"] for row in rows):
                    aliases = (
                        "d-" + text.casefold(),
                        "alpha-d-" + text.casefold(),
                        "beta-d-" + text.casefold(),
                    )
                    alias_rows = connection.execute(
                        "SELECT * FROM metabolites WHERE name_norm IN (?,?,?) LIMIT 20",
                        aliases,
                    ).fetchall()
                    if alias_rows:
                        rows = tuple(rows) + tuple(alias_rows)
                if not rows:
                    canonical = canonicalize_smiles(text)
                    if canonical:
                        rows = connection.execute(
                            "SELECT * FROM metabolites WHERE canonical_smiles = ? LIMIT 20",
                            (canonical,),
                        ).fetchall()
            if rows:
                molecule = self._row_to_molecule(self._preferred_row(rows), text)
            else:
                canonical = canonicalize_smiles(text)
                if canonical is None:
                    raise ValueError(
                        f"{text!r} is neither an exact local metabolite ID/name "
                        "nor a valid SMILES string."
                    )
                molecule = MoleculeNode(
                    key=molecule_key(canonical, None),
                    query=text,
                    smiles=text,
                    canonical_smiles=canonical,
                    generic=is_generic_structure(text),
                )
        with self._lock:
            self.cache_misses += 1
            self._normalization_cache[cache_key] = molecule
            self._normalization_cache[molecule.key] = molecule
        return molecule

    def _matching_ids(
        self, connection: sqlite3.Connection, molecule: MoleculeNode
    ) -> Tuple[str, ...]:
        if self._adjacency is not None:
            return self._adjacency.matching_ids(
                molecule.canonical_smiles, molecule.metabolite_id
            )
        identifiers: List[str] = []
        if molecule.metabolite_id:
            identifiers.append(molecule.metabolite_id)
        if molecule.canonical_smiles:
            rows = connection.execute(
                "SELECT id FROM metabolites WHERE canonical_smiles = ?",
                (molecule.canonical_smiles,),
            ).fetchall()
            identifiers.extend(row["id"] for row in rows)
        return tuple(dict.fromkeys(identifiers))

    def _participants(
        self, connection: sqlite3.Connection, reaction_id: str, side: str
    ) -> Tuple[Tuple[MoleculeNode, str, Optional[str]], ...]:
        rows = (
            self._adjacency.participants(reaction_id, side)
            if self._adjacency is not None
            else connection.execute(
                """
                SELECT rp.coefficient, rp.annotation, m.*
                FROM reaction_participants rp
                JOIN metabolites m ON m.id = rp.metabolite_id
                WHERE rp.reaction_id = ? AND rp.side = ?
                ORDER BY rp.position
                """,
                (reaction_id, side),
            ).fetchall()
        )
        return tuple(
            (self._row_to_molecule(row), str(row["coefficient"]), row["annotation"])
            for row in rows
        )

    @staticmethod
    def _legal_side_pairs(
        direction: str,
        respect_direction: bool,
        allow_unknown_direction: bool,
    ) -> Tuple[Tuple[str, str, bool], ...]:
        if not respect_direction:
            return (("left", "right", False), ("right", "left", False))
        if direction == "forward":
            return (("left", "right", False),)
        if direction == "reverse":
            return (("right", "left", False),)
        if direction == "reversible":
            return (("left", "right", False), ("right", "left", False))
        if direction == "unknown" and allow_unknown_direction:
            return (("left", "right", True), ("right", "left", True))
        return ()

    @staticmethod
    def _database_support(row: sqlite3.Row, uncertain: bool) -> float:
        value = 0.62 if uncertain else 0.82
        if row["gene_rule"]:
            value += 0.10
        if row["subsystem"]:
            value += 0.05
        return min(1.0, value)

    @staticmethod
    def _id_origin(identifier: str) -> str:
        try:
            return "internal" if int(identifier[1:]) >= 30000 else "source"
        except (TypeError, ValueError):
            return "unknown"

    def _build_candidate(
        self,
        row: sqlite3.Row,
        target: MoleculeNode,
        precursor_records: Sequence[Tuple[MoleculeNode, str, Optional[str]]],
        product_records: Sequence[Tuple[MoleculeNode, str, Optional[str]]],
        precursor_side: str,
        product_side: str,
        uncertain: bool,
        currency_ids: Iterable[str],
    ) -> ReactionCandidate:
        precursors = tuple(record[0] for record in precursor_records)
        main, auxiliary, currency = classify_precursors(
            target, precursors, currency_ids
        )
        noncurrency = [item for item in precursors if item.key not in set(currency)]
        structural, conservation = candidate_structure_metrics(target, noncurrency)
        toxicity_values = [
            item.toxicity for item in noncurrency if item.toxicity is not None
        ]
        warnings: List[str] = []
        if uncertain:
            warnings.append("reaction direction is unknown in the source database")
        if not noncurrency:
            warnings.append("reaction has no non-currency precursor")
        if any(item.generic for item in noncurrency):
            warnings.append("one or more precursors has a generic or missing structure")
        annotations = [
            record[2] for record in precursor_records + tuple(product_records) if record[2]
        ]
        identifier = row["id"]
        return ReactionCandidate(
            candidate_id=(
                f"db:{identifier}:{product_side}>{precursor_side}:{target.key}"
            ),
            target=target,
            precursors=precursors,
            source="DATABASE",
            main_precursor_keys=main,
            auxiliary_precursor_keys=auxiliary,
            currency_precursor_keys=currency,
            coefficients=tuple(record[1] for record in precursor_records),
            all_products=tuple(record[0] for record in product_records),
            product_coefficients=tuple(record[1] for record in product_records),
            database_reaction_ids=(identifier,),
            reaction_name=row["name"],
            equation=row["equation"],
            direction=row["direction"],
            direction_uncertain=uncertain,
            ec_numbers=extract_ec_numbers(row["gene_rule"]),
            subsystem=row["subsystem"],
            database_support=self._database_support(row, uncertain),
            structural_similarity=structural,
            atom_conservation=None,
            approx_structural_overlap=conservation,
            toxicity=max(toxicity_values) if toxicity_values else None,
            ec_consistency=1.0 if extract_ec_numbers(row["gene_rule"]) else None,
            warnings=tuple(warnings),
            metadata={
                "precursor_side": precursor_side,
                "product_side": product_side,
                "stored_forward_direction": (
                    "left_to_right" if precursor_side == "left" else "right_to_left"
                ),
                "retrosynthesis_direction": (
                    "right_to_left" if product_side == "right" else "left_to_right"
                ),
                "atom_mapping_source": "NONE",
                "lower_bound": row["lower_bound"],
                "upper_bound": row["upper_bound"],
                "gene_rule": row["gene_rule"],
                "id_origin": self._id_origin(identifier),
                "participant_annotations": annotations,
                "metric_provenance": {
                    "database_support": "LOCAL_SQLITE_REACTION_ANNOTATION",
                    "structural_similarity": "RDKIT_MORGAN_TANIMOTO",
                    "approx_structural_overlap": "RDKIT_MCS_TARGET_FRACTION",
                    "atom_conservation": "UNAVAILABLE_NO_REACTION_ATOM_MAPPING",
                    "toxicity": "LOCAL_DATABASE_UNCALIBRATED_SCALAR",
                },
            },
        )

    def backward_candidates(
        self,
        target: MoleculeNode,
        *,
        limit: int = 20,
        respect_direction: bool = True,
        allow_unknown_direction: bool = True,
        currency_ids: Iterable[str] = DEFAULT_CURRENCY_IDS,
    ) -> Tuple[ReactionCandidate, ...]:
        currency_tuple = tuple(sorted(set(currency_ids)))
        key = (
            target.key,
            limit,
            respect_direction,
            allow_unknown_direction,
            currency_tuple,
        )
        with self._lock:
            cached = self._backward_cache.get(key)
            if cached is not None:
                self.cache_hits += 1
                return cached
        candidates: List[ReactionCandidate] = []
        seen = set()
        with self._connect() as connection:
            identifiers = self._matching_ids(connection, target)
            if identifiers:
                placeholders = ",".join("?" for _ in identifiers)
                rows = (
                    self._adjacency.reaction_occurrences(identifiers)
                    if self._adjacency is not None
                    else connection.execute(
                        f"""
                        SELECT DISTINCT r.*, rp.side AS occurrence_side
                        FROM reaction_participants rp
                        JOIN reactions r ON r.id = rp.reaction_id
                        WHERE rp.metabolite_id IN ({placeholders})
                        """,
                        identifiers,
                    ).fetchall()
                )
                for row in rows:
                    for precursor_side, product_side, uncertain in self._legal_side_pairs(
                        row["direction"],
                        respect_direction,
                        allow_unknown_direction,
                    ):
                        if row["occurrence_side"] != product_side:
                            continue
                        dedupe = (row["id"], precursor_side, product_side, target.key)
                        if dedupe in seen:
                            continue
                        seen.add(dedupe)
                        precursor_records = self._participants(
                            connection, row["id"], precursor_side
                        )
                        product_records = self._participants(
                            connection, row["id"], product_side
                        )
                        candidates.append(
                            self._build_candidate(
                                row,
                                target,
                                precursor_records,
                                product_records,
                                precursor_side,
                                product_side,
                                uncertain,
                                currency_tuple,
                            )
                        )
        candidates.sort(
            key=lambda item: (
                item.direction_uncertain,
                not bool(item.ec_numbers),
                any(value.generic for value in item.noncurrency_precursors),
                -(item.atom_conservation or item.approx_structural_overlap or 0.0),
                item.database_reaction_ids,
            )
        )
        result = tuple(candidates[:limit])
        with self._lock:
            self.cache_misses += 1
            self._backward_cache[key] = result
        return result

    def forward_candidates(
        self,
        source: MoleculeNode,
        *,
        limit: int = 20,
        respect_direction: bool = True,
        allow_unknown_direction: bool = True,
        currency_ids: Iterable[str] = DEFAULT_CURRENCY_IDS,
    ) -> Tuple[ReactionCandidate, ...]:
        """Return legal products for reactions in which ``source`` is a main input."""
        currency_tuple = tuple(sorted(set(currency_ids)))
        key = (
            source.key,
            limit,
            respect_direction,
            allow_unknown_direction,
            currency_tuple,
        )
        with self._lock:
            cached = self._forward_cache.get(key)
            if cached is not None:
                self.cache_hits += 1
                return cached
        candidates: List[ReactionCandidate] = []
        seen = set()
        with self._connect() as connection:
            identifiers = self._matching_ids(connection, source)
            if identifiers:
                placeholders = ",".join("?" for _ in identifiers)
                rows = (
                    self._adjacency.reaction_occurrences(identifiers)
                    if self._adjacency is not None
                    else connection.execute(
                        f"""
                        SELECT DISTINCT r.*, rp.side AS occurrence_side
                        FROM reaction_participants rp
                        JOIN reactions r ON r.id = rp.reaction_id
                        WHERE rp.metabolite_id IN ({placeholders})
                        """,
                        identifiers,
                    ).fetchall()
                )
                for row in rows:
                    for precursor_side, product_side, uncertain in self._legal_side_pairs(
                        row["direction"],
                        respect_direction,
                        allow_unknown_direction,
                    ):
                        if row["occurrence_side"] != precursor_side:
                            continue
                        precursor_records = self._participants(
                            connection, row["id"], precursor_side
                        )
                        product_records = self._participants(
                            connection, row["id"], product_side
                        )
                        for product, _, _ in product_records:
                            if is_currency(product, currency_tuple):
                                continue
                            dedupe = (row["id"], precursor_side, product_side, product.key)
                            if dedupe in seen:
                                continue
                            seen.add(dedupe)
                            candidate = self._build_candidate(
                                row,
                                product,
                                precursor_records,
                                product_records,
                                precursor_side,
                                product_side,
                                uncertain,
                                currency_tuple,
                            )
                            if source.key not in set(candidate.main_precursor_keys):
                                continue
                            candidates.append(candidate)
        candidates.sort(
            key=lambda item: (
                item.direction_uncertain,
                not bool(item.ec_numbers),
                any(value.generic for value in item.noncurrency_precursors),
                -(item.atom_conservation or item.approx_structural_overlap or 0.0),
                item.database_reaction_ids,
                item.target.key,
            )
        )
        result = tuple(candidates[:limit])
        with self._lock:
            self.cache_misses += 1
            self._forward_cache[key] = result
        return result

    def cache_info(self) -> Dict[str, int]:
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "normalizations": len(self._normalization_cache),
            "backward_queries": len(self._backward_cache),
            "forward_queries": len(self._forward_cache),
            "adjacency_index": (
                self._adjacency.info()
                if self._adjacency is not None
                else {"enabled": False, "loaded": False}
            ),
        }

    def sqlite_index_diagnostics(self) -> Dict[str, object]:
        """Return index names and representative SQLite query plans."""
        with self._connect() as connection:
            indexes = [
                row[1]
                for row in connection.execute(
                    "PRAGMA index_list('reaction_participants')"
                ).fetchall()
            ]
            plans = connection.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT r.id
                FROM reaction_participants rp
                JOIN reactions r ON r.id = rp.reaction_id
                WHERE rp.metabolite_id = ?
                """,
                ("C00469",),
            ).fetchall()
        return {
            "reaction_participant_indexes": indexes,
            "metabolite_lookup_plan": [tuple(row) for row in plans],
        }
