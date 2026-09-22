"""Configuration objects for the enzymatic retrosynthesis planner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Mapping, Tuple


@dataclass(frozen=True)
class ScoringWeights:
    """Weights used by the masked candidate-score average."""

    database_support: float = 0.25
    model_score: float = 0.15
    feasibility: float = 0.15
    # The local MooRetro implementation currently fills yield/toxicity with
    # random placeholders. They remain available but are excluded by default.
    yield_score: float = 0.00
    inverse_toxicity: float = 0.00
    ec_consistency: float = 0.10
    atom_conservation: float = 0.10
    approx_structural_overlap: float = 0.10
    building_block_progress: float = 0.05
    structural_similarity: float = 0.10

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "ScoringWeights":
        if not values:
            return cls()
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError("Unknown scoring weight(s): " + ", ".join(unknown))
        parsed = {key: float(value) for key, value in values.items()}
        if any(value < 0 for value in parsed.values()):
            raise ValueError("Scoring weights must be non-negative.")
        return cls(**parsed)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


DEFAULT_CURRENCY_IDS: Tuple[str, ...] = (
    "C00001",  # water
    "C00002",  # ATP
    "C00003",  # NAD+
    "C00004",  # NADH
    "C00005",  # NADPH
    "C00006",  # NADP+
    "C00007",  # oxygen
    "C00008",  # ADP
    "C00009",  # orthophosphate
    "C00010",  # CoA
    "C00011",  # carbon dioxide
    "C00013",  # diphosphate
    "C00014",  # ammonia
    "C00015",  # UDP
    "C00016",  # FAD
    "C00019",  # S-adenosyl-L-methionine
    "C00020",  # AMP
    "C00021",  # S-adenosyl-L-homocysteine
    "C00027",  # hydrogen peroxide
    "C00044",  # GTP
    "C00046",  # dTTP
    "C00055",  # CMP
    "C00063",  # CTP
    "C00075",  # UTP
    "C00080",  # proton
    "C00081",  # ITP
    "C00112",  # CDP
    "C00126",  # glutathione disulfide
    "C00138",  # reduced ferredoxin
    "C00139",  # oxidized ferredoxin
    "C00282",  # hydrogen
    "C00288",  # bicarbonate
    "C00342",  # thioredoxin
    "C00343",  # reduced thioredoxin
    "C01352",  # FADH2
)


@dataclass(frozen=True)
class PlannerConfig:
    """All search, filtering and scoring controls for one planning run."""

    use_database: bool = True
    use_model: bool = False
    respect_reaction_direction: bool = True
    allow_unknown_direction: bool = True
    # When enabled, a complete route must terminate at exactly the supplied
    # non-currency building-block set. Agent-facing endpoint-constrained
    # searches enable this; direct inventory-style planner callers may leave it
    # disabled so that any stocked subset can close a route.
    require_all_building_blocks: bool = False

    max_depth: int = 10
    max_expansions: int = 5000
    forward_anchor_depth: int = 6
    forward_anchor_max_nodes: int = 5000
    database_top_k: int = 20
    model_top_k: int = 10
    candidate_beam_width: int = 20
    global_beam_width: int = 50
    top_k_routes: int = 10
    solution_pool_multiplier: int = 3
    max_states_per_signature: int = 3
    max_forward_routes_per_node: int = 10
    max_precursors_per_step: int = 8
    max_model_only_steps: int = 5
    max_consecutive_model_only_steps: int = 2
    max_model_calls: int = 6

    # Search policy. The agent-facing EnzymeRetrosynthesisSearch tool defaults
    # to ``and_or_mcts`` while the engine keeps best-first as its compatibility
    # default for direct PlannerConfig callers.
    search_algorithm: str = "best_first_beam"
    mcts_exploration_constant: float = 1.4
    mcts_rollout_depth: int = 3
    mcts_random_seed: int = 100
    mcts_epsilon: float = 0.10
    mcts_progressive_widening: bool = True
    mcts_progressive_widening_alpha: float = 0.50
    mcts_progressive_widening_constant: float = 2.0
    mcts_completion_bonus: float = 1.0

    min_candidate_score: float = 0.02
    min_model_score: float = 0.0
    min_feasibility: float = 0.0
    min_atom_conservation: float = 0.0
    min_approx_structural_overlap: float = 0.0
    max_toxicity: float = 1.0
    require_ec: bool = False
    reject_generic_structures: bool = True

    heuristic_weight: float = 0.35
    consensus_bonus: float = 0.10
    database_preference_bonus: float = 0.03
    model_only_penalty: float = 0.25
    generic_structure_penalty: float = 0.40
    unknown_direction_penalty: float = 0.20
    auxiliary_precursor_penalty: float = 0.04
    route_length_penalty: float = 0.03
    diversity_penalty: float = 0.15
    fusion_overlap_threshold: float = 0.50
    score_epsilon: float = 1.0e-8

    route_reaction_quality_weight: float = 0.65
    route_carbon_efficiency_weight: float = 0.25
    route_byproduct_weight: float = 0.10
    route_cofactor_weight: float = 0.00

    enable_forward_anchor_cache: bool = True
    forward_anchor_cache_size: int = 16
    currency_metabolite_ids: Tuple[str, ...] = DEFAULT_CURRENCY_IDS
    scoring_weights: ScoringWeights = ScoringWeights()
    include_search_trace: bool = False

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None,
        *,
        use_database: bool | None = None,
        use_model: bool | None = None,
    ) -> "PlannerConfig":
        values = dict(values or {})
        weight_values = values.pop("scoring_weights", None)
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError("Unknown planner option(s): " + ", ".join(unknown))
        if "currency_metabolite_ids" in values:
            values["currency_metabolite_ids"] = tuple(
                str(item).upper() for item in values["currency_metabolite_ids"]
            )
        if weight_values is not None:
            values["scoring_weights"] = ScoringWeights.from_mapping(weight_values)
        if use_database is not None:
            values["use_database"] = bool(use_database)
        if use_model is not None:
            values["use_model"] = bool(use_model)
        config = cls(**values)
        config._validate()
        return config

    def with_overrides(self, values: Mapping[str, Any] | None) -> "PlannerConfig":
        if not values:
            return self
        merged = self.to_dict()
        merged.update(dict(values))
        if "scoring_weights" not in values:
            merged["scoring_weights"] = self.scoring_weights.to_dict()
        return self.from_mapping(merged)

    def _validate(self) -> None:
        positive_ints = (
            "max_depth",
            "max_expansions",
            "forward_anchor_max_nodes",
            "database_top_k",
            "model_top_k",
            "candidate_beam_width",
            "global_beam_width",
            "top_k_routes",
            "solution_pool_multiplier",
            "max_states_per_signature",
            "max_forward_routes_per_node",
            "max_precursors_per_step",
            "forward_anchor_cache_size",
            "mcts_rollout_depth",
        )
        for name in positive_ints:
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1.")
        if (
            self.forward_anchor_depth < 0
            or self.max_model_only_steps < 0
            or self.max_consecutive_model_only_steps < 0
            or self.max_model_calls < 0
        ):
            raise ValueError("Depth, model-call and model-only limits cannot be negative.")
        unit_fields = (
            "min_candidate_score",
            "min_model_score",
            "min_feasibility",
            "min_atom_conservation",
            "min_approx_structural_overlap",
            "max_toxicity",
            "fusion_overlap_threshold",
        )
        for name in unit_fields:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.search_algorithm not in {
            "best_first_beam",
            "uniform_cost",
            "and_or_mcts",
        }:
            raise ValueError(
                "search_algorithm must be best_first_beam, uniform_cost, "
                "or and_or_mcts."
            )
        if self.mcts_exploration_constant < 0:
            raise ValueError("mcts_exploration_constant must be non-negative.")
        if not 0.0 <= self.mcts_epsilon <= 1.0:
            raise ValueError("mcts_epsilon must be between 0 and 1.")
        if not 0.0 < self.mcts_progressive_widening_alpha <= 1.0:
            raise ValueError(
                "mcts_progressive_widening_alpha must be in the interval (0, 1]."
            )
        if self.mcts_progressive_widening_constant <= 0:
            raise ValueError(
                "mcts_progressive_widening_constant must be positive."
            )
        if self.mcts_completion_bonus < 0:
            raise ValueError("mcts_completion_bonus must be non-negative.")
        route_weights = (
            self.route_reaction_quality_weight,
            self.route_carbon_efficiency_weight,
            self.route_byproduct_weight,
            self.route_cofactor_weight,
        )
        if any(value < 0 for value in route_weights) or sum(route_weights) <= 0:
            raise ValueError("Route scoring weights must be non-negative with a positive sum.")
        if not self.use_database and not self.use_model:
            raise ValueError("At least one candidate source must be enabled.")

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["currency_metabolite_ids"] = list(self.currency_metabolite_ids)
        return result
