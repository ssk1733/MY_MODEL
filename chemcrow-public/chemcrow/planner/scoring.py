"""Candidate fusion and configurable, masked reaction scoring."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, Iterable, List, Sequence, Tuple

from .chem_utils import (
    candidate_structure_metrics,
    ec_compatibility,
    normalized_toxicity,
    structural_similarity,
)
from .config import PlannerConfig
from .models import MoleculeNode, ReactionCandidate


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    set_left, set_right = set(left), set(right)
    if not set_left and not set_right:
        return 1.0
    if not set_left or not set_right:
        return 0.0
    return len(set_left & set_right) / len(set_left | set_right)


def _main_signature(candidate: ReactionCandidate) -> Tuple[str, ...]:
    values = candidate.main_precursor_keys or tuple(
        item.key for item in candidate.noncurrency_precursors
    )
    return tuple(sorted(values))


def complete_precursor_signature(
    candidate: ReactionCandidate,
) -> Tuple[str, Tuple[str, ...]]:
    """Return the target plus the complete precursor multiset.

    This includes currency and auxiliary molecules and preserves repeated
    precursors, so ``{R, O, O}`` does not collapse to ``{R, O}``.
    """
    return candidate.target.key, tuple(
        sorted(item.key for item in candidate.precursors)
    )


def _fusion_quality(
    database: ReactionCandidate, model: ReactionCandidate
) -> Tuple[float, float, bool]:
    left = _main_signature(database)
    right = _main_signature(model)
    overlap = _jaccard(left, right)
    exact = bool(left and right and left == right)
    ec_score = ec_compatibility(database.ec_numbers, model.ec_numbers)
    return overlap, ec_score if ec_score is not None else 0.0, exact


def _combine_evidence(
    database: ReactionCandidate,
    model: ReactionCandidate,
    overlap: float,
    ec_score: float,
) -> ReactionCandidate:
    ec_numbers = tuple(dict.fromkeys(database.ec_numbers + model.ec_numbers))
    consistency = ec_compatibility(database.ec_numbers, model.ec_numbers)
    if consistency is None:
        consistency = 0.75 if ec_numbers else None
    metadata = dict(database.metadata)
    metadata.update(
        {
            "database_candidate_id": database.candidate_id,
            "model_candidate_id": model.candidate_id,
            "fusion_main_precursor_overlap": overlap,
            "fusion_ec_compatibility": ec_score,
            "fusion_match_score": 0.70 * overlap + 0.30 * ec_score,
            "fusion_direction_compatible": True,
            "model_metadata": dict(model.metadata),
        }
    )
    return replace(
        database,
        candidate_id="both:" + database.candidate_id + ":" + model.candidate_id,
        source="BOTH",
        ec_numbers=ec_numbers,
        template=model.template,
        rank=model.rank,
        model_score=model.model_score,
        structural_similarity=(
            max(
                value
                for value in (
                    database.structural_similarity,
                    model.structural_similarity,
                )
                if value is not None
            )
            if any(
                value is not None
                for value in (
                    database.structural_similarity,
                    model.structural_similarity,
                )
            )
            else None
        ),
        atom_conservation=(
            max(
                value
                for value in (
                    database.atom_conservation,
                    model.atom_conservation,
                )
                if value is not None
            )
            if any(
                value is not None
                for value in (
                    database.atom_conservation,
                    model.atom_conservation,
                )
            )
            else None
        ),
        approx_structural_overlap=(
            max(
                value
                for value in (
                    database.approx_structural_overlap,
                    model.approx_structural_overlap,
                )
                if value is not None
            )
            if any(
                value is not None
                for value in (
                    database.approx_structural_overlap,
                    model.approx_structural_overlap,
                )
            )
            else None
        ),
        feasibility=model.feasibility,
        yield_score=model.yield_score,
        toxicity=(
            model.toxicity
            if model.toxicity is not None
            else database.toxicity
        ),
        ec_consistency=consistency,
        warnings=tuple(dict.fromkeys(database.warnings + model.warnings)),
        metadata=metadata,
    )


def fuse_candidates(
    database_candidates: Sequence[ReactionCandidate],
    model_candidates: Sequence[ReactionCandidate],
    config: PlannerConfig,
) -> Tuple[ReactionCandidate, ...]:
    """Union both sources and mark matching evidence as ``BOTH``."""
    remaining_models = list(model_candidates)
    fused: List[ReactionCandidate] = []
    for database in database_candidates:
        best_index = None
        best_quality = (-1.0, -1.0, False)
        for index, model in enumerate(remaining_models):
            if database.target.key != model.target.key:
                continue
            quality = _fusion_quality(database, model)
            ec_compatible = (
                not database.ec_numbers
                or not model.ec_numbers
                or quality[1] >= 0.50
            )
            # Do not label partial precursor overlap as BOTH. A consensus needs
            # identical main precursor identities plus compatible EC evidence when
            # both sources provide EC annotations.
            eligible = (
                quality[2]
                and quality[0] >= config.fusion_overlap_threshold
                and ec_compatible
            )
            if eligible and quality > best_quality:
                best_index = index
                best_quality = quality
        if best_index is None:
            fused.append(database)
        else:
            model = remaining_models.pop(best_index)
            fused.append(
                _combine_evidence(database, model, best_quality[0], best_quality[1])
            )
    fused.extend(remaining_models)

    # Remove exact duplicate precursor/EC/source candidates while preserving the
    # best-supported first occurrence from each generator.
    result: List[ReactionCandidate] = []
    seen = set()
    for candidate in fused:
        signature = (
            candidate.target.key,
            tuple(sorted(item.key for item in candidate.noncurrency_precursors)),
            tuple(sorted(candidate.ec_numbers)),
            candidate.source,
        )
        if signature not in seen:
            seen.add(signature)
            result.append(candidate)
    return tuple(result)


def fuse_candidates_by_complete_precursors(
    database_candidates: Sequence[ReactionCandidate],
    model_candidates: Sequence[ReactionCandidate],
    config: PlannerConfig,
) -> Tuple[ReactionCandidate, ...]:
    """Fuse exact complete precursor multisets without consuming near matches.

    Unmatched and EC-conflicting versions remain in the union. Therefore every
    complete combination returned by either generator stays represented in the
    merged pool.
    """
    remaining_models = list(model_candidates)
    fused: List[ReactionCandidate] = []
    for database in database_candidates:
        match_index = None
        match_quality = (1.0, 0.0)
        database_signature = complete_precursor_signature(database)
        for index, model in enumerate(remaining_models):
            if complete_precursor_signature(model) != database_signature:
                continue
            compatibility = ec_compatibility(
                database.ec_numbers, model.ec_numbers
            )
            if (
                database.ec_numbers
                and model.ec_numbers
                and (compatibility is None or compatibility < 0.50)
            ):
                continue
            match_index = index
            match_quality = (1.0, compatibility or 0.0)
            break
        if match_index is None:
            fused.append(database)
            continue
        model = remaining_models.pop(match_index)
        combined = _combine_evidence(
            database, model, match_quality[0], match_quality[1]
        )
        metadata = dict(combined.metadata)
        metadata.update(
            {
                "fusion_rule": "EXACT_COMPLETE_PRECURSOR_MULTISET",
                "database_source_candidate_ids": [database.candidate_id],
                "model_source_candidate_ids": [model.candidate_id],
            }
        )
        fused.append(replace(combined, metadata=metadata))
    fused.extend(remaining_models)
    return tuple(fused)


def _building_block_progress(
    candidate: ReactionCandidate, building_blocks: Sequence[MoleculeNode]
) -> float | None:
    precursors = candidate.main_precursors or candidate.noncurrency_precursors
    if not precursors or not building_blocks:
        return None
    values = []
    block_keys = {item.key for item in building_blocks}
    for precursor in precursors:
        if precursor.key in block_keys:
            values.append(1.0)
            continue
        precursor_smiles = precursor.canonical_smiles or precursor.smiles
        for block in building_blocks:
            similarity = structural_similarity(
                precursor_smiles, block.canonical_smiles or block.smiles
            )
            if similarity is not None:
                values.append(similarity)
    return max(values) if values else None


def score_candidate(
    candidate: ReactionCandidate,
    building_blocks: Sequence[MoleculeNode],
    config: PlannerConfig,
) -> ReactionCandidate:
    structural = candidate.structural_similarity
    conservation = candidate.atom_conservation
    approximate_overlap = candidate.approx_structural_overlap
    if structural is None or approximate_overlap is None:
        computed_structure, computed_overlap = candidate_structure_metrics(
            candidate.target,
            candidate.main_precursors or candidate.noncurrency_precursors,
        )
        structural = structural if structural is not None else computed_structure
        approximate_overlap = (
            approximate_overlap if approximate_overlap is not None else computed_overlap
        )
    progress = _building_block_progress(candidate, building_blocks)
    toxicity = normalized_toxicity(candidate.toxicity)
    values = {
        "database_support": candidate.database_support,
        "model_score": candidate.model_score,
        "feasibility": candidate.feasibility,
        "yield_score": candidate.yield_score,
        "inverse_toxicity": None if toxicity is None else 1.0 - toxicity,
        "ec_consistency": candidate.ec_consistency,
        "atom_conservation": conservation,
        "approx_structural_overlap": approximate_overlap,
        "building_block_progress": progress,
        "structural_similarity": structural,
    }
    weights = config.scoring_weights.to_dict()
    numerator = 0.0
    denominator = 0.0
    used: Dict[str, float] = {}
    for name, value in values.items():
        weight = float(weights.get(name, 0.0))
        if value is None or weight <= 0.0:
            continue
        clamped = max(0.0, min(1.0, float(value)))
        numerator += weight * clamped
        denominator += weight
        used[name] = clamped
    score = numerator / denominator if denominator else config.score_epsilon
    if candidate.source == "BOTH":
        score += config.consensus_bonus
    elif candidate.source == "DATABASE":
        score += config.database_preference_bonus
    score = max(config.score_epsilon, min(0.999999, score))

    cost = -math.log(score + config.score_epsilon)
    if candidate.source == "MODEL":
        cost += config.model_only_penalty
    if candidate.direction_uncertain:
        cost += config.unknown_direction_penalty
    if any(item.generic for item in candidate.noncurrency_precursors):
        cost += config.generic_structure_penalty
    cost += config.auxiliary_precursor_penalty * len(
        candidate.auxiliary_precursor_keys
    )
    metadata = dict(candidate.metadata)
    metadata["score_provenance"] = {
        "aggregation": "MASKED_WEIGHTED_AVERAGE",
        "active_weights": {
            name: weights[name] for name in used if weights.get(name, 0.0) > 0.0
        },
        "consensus_bonus": config.consensus_bonus if candidate.source == "BOTH" else 0.0,
        "database_preference_bonus": (
            config.database_preference_bonus if candidate.source == "DATABASE" else 0.0
        ),
    }
    return replace(
        candidate,
        structural_similarity=structural,
        atom_conservation=conservation,
        approx_structural_overlap=approximate_overlap,
        toxicity=toxicity,
        building_block_progress=progress,
        step_score=score,
        step_cost=cost,
        score_components=used,
        metadata=metadata,
    )


def score_and_rank_candidates(
    candidates: Sequence[ReactionCandidate],
    building_blocks: Sequence[MoleculeNode],
    config: PlannerConfig,
) -> Tuple[ReactionCandidate, ...]:
    scored = [score_candidate(candidate, building_blocks, config) for candidate in candidates]
    scored.sort(
        key=lambda item: (
            item.step_cost if item.step_cost is not None else float("inf"),
            item.candidate_id,
        )
    )
    return tuple(scored)
