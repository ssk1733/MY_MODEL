"""Shared, LangChain-independent planner data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple


def _rounded(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 6)


@dataclass(frozen=True)
class MoleculeNode:
    key: str
    query: str
    smiles: Optional[str] = None
    canonical_smiles: Optional[str] = None
    metabolite_id: Optional[str] = None
    name: Optional[str] = None
    formula: Optional[str] = None
    charge: Optional[float] = None
    toxicity: Optional[float] = None
    generic: bool = False

    @property
    def label(self) -> str:
        return self.name or self.metabolite_id or self.canonical_smiles or self.query

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "id": self.metabolite_id,
            "name": self.name,
            "smiles": self.smiles,
            "canonical_smiles": self.canonical_smiles,
            "formula": self.formula,
            "charge": self.charge,
            "toxicity": _rounded(self.toxicity),
            "generic": self.generic,
        }


@dataclass(frozen=True)
class ReactionCandidate:
    candidate_id: str
    target: MoleculeNode
    precursors: Tuple[MoleculeNode, ...]
    source: str
    main_precursor_keys: Tuple[str, ...] = ()
    auxiliary_precursor_keys: Tuple[str, ...] = ()
    currency_precursor_keys: Tuple[str, ...] = ()
    coefficients: Tuple[str, ...] = ()
    all_products: Tuple[MoleculeNode, ...] = ()
    product_coefficients: Tuple[str, ...] = ()
    database_reaction_ids: Tuple[str, ...] = ()
    reaction_name: Optional[str] = None
    equation: Optional[str] = None
    direction: Optional[str] = None
    direction_uncertain: bool = False
    ec_numbers: Tuple[str, ...] = ()
    subsystem: Optional[str] = None
    template: Optional[str] = None
    rank: Optional[str] = None
    database_support: Optional[float] = None
    model_score: Optional[float] = None
    structural_similarity: Optional[float] = None
    # Only populate atom_conservation when reaction atom mapping is available.
    atom_conservation: Optional[float] = None
    approx_structural_overlap: Optional[float] = None
    feasibility: Optional[float] = None
    yield_score: Optional[float] = None
    toxicity: Optional[float] = None
    ec_consistency: Optional[float] = None
    building_block_progress: Optional[float] = None
    step_score: Optional[float] = None
    step_cost: Optional[float] = None
    score_components: Mapping[str, float] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def noncurrency_precursors(self) -> Tuple[MoleculeNode, ...]:
        currency = set(self.currency_precursor_keys)
        return tuple(item for item in self.precursors if item.key not in currency)

    @property
    def main_precursors(self) -> Tuple[MoleculeNode, ...]:
        main = set(self.main_precursor_keys)
        return tuple(item for item in self.precursors if item.key in main)

    @property
    def fingerprint(self) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
        core = self.main_precursor_keys or tuple(
            sorted(item.key for item in self.noncurrency_precursors)
        )
        return self.target.key, tuple(sorted(core)), tuple(sorted(self.ec_numbers))

    def to_dict(self) -> Dict[str, Any]:
        main = set(self.main_precursor_keys)
        currency = set(self.currency_precursor_keys)
        precursor_values = []
        for index, molecule in enumerate(self.precursors):
            if molecule.key in currency:
                role = "currency"
            elif molecule.key in main:
                role = "main"
            else:
                role = "auxiliary"
            value = molecule.to_dict()
            value.update(
                {
                    "role": role,
                    "coefficient": (
                        self.coefficients[index]
                        if index < len(self.coefficients)
                        else "1"
                    ),
                }
            )
            precursor_values.append(value)
        return {
            "candidate_id": self.candidate_id,
            "provenance": self.source,
            "database_reaction_ids": list(self.database_reaction_ids),
            "reaction_name": self.reaction_name,
            "stored_equation": self.equation,
            "stored_direction": self.direction,
            "direction_uncertain": self.direction_uncertain,
            "target": self.target.to_dict(),
            "precursors": precursor_values,
            "other_products": [
                {
                    **item.to_dict(),
                    "coefficient": (
                        self.product_coefficients[index]
                        if index < len(self.product_coefficients)
                        else "1"
                    ),
                }
                for index, item in enumerate(self.all_products)
                if item.key != self.target.key
            ],
            "ec_numbers": list(self.ec_numbers),
            "subsystem": self.subsystem,
            "template": self.template,
            "model_rank": self.rank,
            "score": _rounded(self.step_score),
            "cost": _rounded(self.step_cost),
            "score_components": {
                key: _rounded(value) for key, value in self.score_components.items()
            },
            "raw_metrics": {
                "database_support": _rounded(self.database_support),
                "model_score": _rounded(self.model_score),
                "structural_similarity": _rounded(self.structural_similarity),
                "atom_conservation": _rounded(self.atom_conservation),
                "atom_mapping_source": self.metadata.get("atom_mapping_source", "NONE"),
                "approx_structural_overlap": _rounded(
                    self.approx_structural_overlap
                ),
                "feasibility": _rounded(self.feasibility),
                "yield": _rounded(self.yield_score),
                "toxicity": _rounded(self.toxicity),
                "ec_consistency": _rounded(self.ec_consistency),
                "building_block_similarity": _rounded(
                    self.building_block_progress
                ),
                "building_block_similarity_semantics": (
                    "1.0=closest to an input building block; 0.0=farther away"
                ),
            },
            "warnings": list(self.warnings),
            "direction_semantics": {
                "stored_reaction_direction": self.direction,
                "stored_forward_direction": self.metadata.get(
                    "stored_forward_direction"
                ),
                "retrosynthesis_direction": self.metadata.get(
                    "retrosynthesis_direction"
                ),
                "precursor_side": self.metadata.get("precursor_side"),
                "target_side": self.metadata.get("product_side"),
            },
            "metric_provenance": dict(self.metadata.get("metric_provenance", {})),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ForwardAnchor:
    molecule: MoleculeNode
    steps: Tuple[ReactionCandidate, ...]
    cost: float


@dataclass
class RouteState:
    unresolved: Tuple[str, ...]
    molecules: Dict[str, MoleculeNode]
    steps: Tuple[ReactionCandidate, ...]
    cost: float
    backward_depth: int
    model_only_steps: int
    consecutive_model_only_steps: int = 0

    @property
    def signature(self) -> Tuple[str, ...]:
        return tuple(sorted(self.unresolved))
