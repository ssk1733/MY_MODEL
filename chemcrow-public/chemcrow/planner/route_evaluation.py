"""Route-level production and approximate carbon-flow evaluation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from .chem_utils import carbon_count, is_currency
from .models import MoleculeNode, ReactionCandidate


def _coefficient(value: str | None) -> float:
    try:
        parsed = float(str(value or "1").strip())
        return parsed if parsed > 0 else 1.0
    except (TypeError, ValueError):
        return 1.0


def _main_chain_steps(
    target_key: str, steps: Sequence[ReactionCandidate]
) -> Tuple[ReactionCandidate, ...]:
    producers = {step.target.key: step for step in steps}
    selected = []
    seen = set()
    key = target_key
    while key in producers and key not in seen:
        seen.add(key)
        step = producers[key]
        selected.append(step)
        main = step.main_precursors or step.noncurrency_precursors
        if not main:
            break
        precursor = max(main, key=lambda item: (carbon_count(item), item.key))
        key = precursor.key
    selected.reverse()
    return tuple(selected)


def evaluate_route_carbon(
    target: MoleculeNode,
    building_blocks: Sequence[MoleculeNode],
    steps: Sequence[ReactionCandidate],
    currency_ids: Iterable[str],
) -> Dict[str, Any]:
    """Estimate main-branch carbon retention without claiming atom mapping.

    This is explicitly a stoichiometry-aware carbon-count approximation.  It is
    not a substitute for reaction atom mapping and is labelled as such in output.
    """
    main_chain = _main_chain_steps(target.key, steps)
    step_metrics: Dict[str, Dict[str, Any]] = {}
    retained_product = 1.0
    major_penalty_sum = 0.0
    organic_byproducts = []

    for step in main_chain:
        main = step.main_precursors or step.noncurrency_precursors
        main_precursor = max(main, key=lambda item: (carbon_count(item), item.key))
        precursor_index = next(
            (
                index
                for index, item in enumerate(step.precursors)
                if item.key == main_precursor.key
            ),
            0,
        )
        precursor_coefficient = (
            _coefficient(step.coefficients[precursor_index])
            if precursor_index < len(step.coefficients)
            else 1.0
        )
        precursor_carbon = carbon_count(main_precursor) * precursor_coefficient

        product_index = next(
            (
                index
                for index, item in enumerate(step.all_products)
                if item.key == step.target.key
            ),
            0,
        )
        product_coefficient = (
            _coefficient(step.product_coefficients[product_index])
            if product_index < len(step.product_coefficients)
            else 1.0
        )
        target_carbon = carbon_count(step.target) * product_coefficient
        retained_fraction = (
            min(1.0, target_carbon / precursor_carbon)
            if precursor_carbon > 0
            else None
        )
        if retained_fraction is not None:
            retained_product *= retained_fraction

        byproducts = []
        byproduct_carbon = 0.0
        for index, product in enumerate(step.all_products):
            if product.key == step.target.key or is_currency(product, currency_ids):
                continue
            carbon = carbon_count(product)
            if carbon < 2:
                continue
            coefficient = (
                _coefficient(step.product_coefficients[index])
                if index < len(step.product_coefficients)
                else 1.0
            )
            carbon_amount = carbon * coefficient
            byproduct_carbon += carbon_amount
            record = {
                "id": product.metabolite_id,
                "name": product.name,
                "smiles": product.canonical_smiles or product.smiles,
                "carbon_atoms": carbon,
                "coefficient": coefficient,
                "carbon_amount": carbon_amount,
            }
            byproducts.append(record)
            organic_byproducts.append({"reaction": step.candidate_id, **record})
        byproduct_penalty = (
            min(1.0, byproduct_carbon / precursor_carbon)
            if precursor_carbon > 0
            else 0.0
        )
        major_penalty_sum += byproduct_penalty
        step_metrics[step.candidate_id] = {
            "main_precursor_carbon": precursor_carbon,
            "target_product_carbon": target_carbon,
            "main_carbon_retained_fraction": retained_fraction,
            "major_organic_byproduct_carbon": byproduct_carbon,
            "major_organic_byproduct_penalty": byproduct_penalty,
            "major_organic_byproducts": byproducts,
            "carbon_flow_method": "STOICHIOMETRIC_CARBON_COUNT_APPROXIMATION",
        }

    producers = {step.target.key for step in steps}
    leaves = {
        precursor.key
        for step in steps
        for precursor in step.noncurrency_precursors
        if precursor.key not in producers
    }
    used_blocks = [block for block in building_blocks if block.key in leaves]
    if not used_blocks and not steps and target.key in {b.key for b in building_blocks}:
        used_blocks = [target]
    start_carbon = sum(carbon_count(block) for block in used_blocks)
    target_carbon = carbon_count(target)
    direct_ratio = (
        min(1.0, target_carbon / start_carbon) if start_carbon > 0 else None
    )
    efficiency_values = [value for value in (direct_ratio, retained_product) if value is not None]
    efficiency = min(efficiency_values) if efficiency_values else None
    major_penalty = min(1.0, major_penalty_sum)
    currency_count = sum(len(step.currency_precursor_keys) for step in steps)
    return {
        "target_carbon_efficiency": efficiency,
        "main_branch_carbon_retention": retained_product if main_chain else None,
        "starting_substrate_carbon": start_carbon or None,
        "target_product_carbon": target_carbon or None,
        "major_organic_byproduct_penalty": major_penalty,
        "major_organic_byproducts": organic_byproducts,
        "currency_cofactor_count": currency_count,
        "carbon_evaluation_method": "MAIN_BRANCH_STOICHIOMETRIC_CARBON_COUNT_APPROXIMATION",
        "atom_mapping_used": False,
        "step_metrics": step_metrics,
    }


def production_quality_score(
    reaction_quality: float,
    evaluation: Mapping[str, Any],
    *,
    reaction_weight: float,
    carbon_weight: float,
    byproduct_weight: float,
    cofactor_weight: float,
) -> Tuple[float, Dict[str, float]]:
    components = {"reaction_quality": max(0.0, min(1.0, reaction_quality))}
    weights = {"reaction_quality": max(0.0, reaction_weight)}
    carbon = evaluation.get("target_carbon_efficiency")
    if carbon is not None:
        components["target_carbon_efficiency"] = max(0.0, min(1.0, float(carbon)))
        weights["target_carbon_efficiency"] = max(0.0, carbon_weight)
    components["inverse_major_byproduct_penalty"] = 1.0 - max(
        0.0, min(1.0, float(evaluation.get("major_organic_byproduct_penalty", 0.0)))
    )
    weights["inverse_major_byproduct_penalty"] = max(0.0, byproduct_weight)
    cofactor_count = max(0.0, float(evaluation.get("currency_cofactor_count", 0.0)))
    components["inverse_cofactor_burden"] = 1.0 / (1.0 + cofactor_count)
    weights["inverse_cofactor_burden"] = max(0.0, cofactor_weight)
    denominator = sum(weights[name] for name in components if weights.get(name, 0.0) > 0)
    if denominator <= 0:
        return components["reaction_quality"], components
    score = sum(components[name] * weights.get(name, 0.0) for name in components) / denominator
    return max(0.0, min(1.0, score)), components
