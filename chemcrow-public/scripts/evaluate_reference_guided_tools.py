#!/usr/bin/env python
"""Reference-guided PaRoutes evaluation using local DB and model tools only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chemcrow.planner import EnzRetroClient, EnzymeReactionDatabase, PlannerConfig
from chemcrow.planner.chem_utils import (
    canonicalize_smiles_without_atom_maps,
    is_generic_structure,
    molecule_key,
)
from chemcrow.planner.models import MoleculeNode, ReactionCandidate
from chemcrow.planner.scoring import (
    fuse_candidates_by_complete_precursors,
    score_and_rank_candidates,
)

SPLITS = ("n1", "n5")
PROTOCOL = "paroutes-reference-guided-v1"
ROUTE_FIELDS = (
    "dataset_split", "source_file", "source_route_index", "route_id", "target",
    "starting_materials", "status", "found_steps", "total_steps",
    "attempted_steps", "started_steps", "not_evaluated_steps",
    "coverage_ratio", "coverage_text", "top1_hits", "top2_hits", "top3_hits",
    "stop_step_id", "stop_reason", "endpoint_match", "elapsed_seconds",
)


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, tuple, Path)):
        return list(value) if not isinstance(value, Path) else str(value)
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                   default=json_default) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(
                record, ensure_ascii=False, sort_keys=True, default=json_default
            ) + "\n")
        handle.flush()


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("{}:{} is not a JSON object".format(path, line_number))
            yield value


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Dict[str, Any]]:
    """Stream a top-level JSON array without loading a split into memory."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer, position, started, eof = "", 0, False, False
        while True:
            if position > chunk_size:
                buffer, position = buffer[position:], 0
            while position >= len(buffer) and not eof:
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer) and eof:
                    raise ValueError("Empty JSON file: {}".format(path))
                if position >= len(buffer):
                    continue
                if buffer[position] != "[":
                    raise ValueError("Expected a JSON array in {}".format(path))
                started, position = True, position + 1
                continue
            while True:
                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position < len(buffer):
                    break
                if eof:
                    raise ValueError("Unterminated JSON array in {}".format(path))
                buffer += handle.read(chunk_size)
            if buffer[position] == "]":
                return
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
                continue
            if not isinstance(value, dict):
                raise ValueError("Route array item must be an object")
            position = end
            yield value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    result = canonicalize_smiles_without_atom_maps(str(value or ""))
    if result is None:
        raise ValueError("Invalid SMILES: {!r}".format(value))
    return result


def node_for_smiles(value: str) -> MoleculeNode:
    normalized = canonical(value)
    return MoleculeNode(
        key=molecule_key(normalized, None),
        query=str(value),
        smiles=str(value),
        canonical_smiles=normalized,
        generic=is_generic_structure(normalized),
    )


def ratio(numerator: int | float, denominator: int) -> Dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": (float(numerator) / denominator) if denominator else None,
    }


def route_id_for(split: str, index: int) -> str:
    return "{}-{:06d}".format(split, index + 1)


def split_paths(dataset_root: Path, split: str) -> Dict[str, Path]:
    return {
        "routes": dataset_root / "{}-routes.json".format(split),
        "targets": dataset_root / "{}-targets.txt".format(split),
        "stock": dataset_root / "{}-stock.txt".format(split),
    }


def metadata_tree_consistency(
    product: str,
    precursors: Sequence[str],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    reaction = str(metadata.get("smiles") or "")
    policy = "tree_parent_and_direct_children_are_authoritative"
    if not reaction:
        return {"status": "unavailable", "policy": policy}
    pieces = reaction.split(">>")
    if len(pieces) != 2:
        return {
            "status": "unverifiable",
            "policy": policy,
            "reason": "metadata.smiles is not reactants>>products",
        }
    try:
        metadata_precursors = [
            canonical(item) for item in pieces[0].split(".") if item
        ]
        metadata_products = [
            canonical(item) for item in pieces[1].split(".") if item
        ]
        expected_product = canonical(product)
        expected_precursors = [canonical(item) for item in precursors]
    except ValueError as exc:
        return {
            "status": "unverifiable",
            "policy": policy,
            "reason": str(exc),
        }
    matches = (
        Counter(metadata_precursors) == Counter(expected_precursors)
        and Counter(metadata_products) == Counter([expected_product])
    )
    return {
        "status": "match" if matches else "conflict",
        "policy": policy,
        "metadata_reactants": metadata_precursors,
        "metadata_products": metadata_products,
    }


def extract_reference_route(
    tree: Mapping[str, Any],
    *,
    split: str,
    source_file: str,
    source_route_index: int,
) -> Dict[str, Any]:
    """Adapt one PaRoutes tree to stable route, step and molecule records."""
    route_id = route_id_for(split, source_route_index)
    if tree.get("type") != "mol":
        raise ValueError("root must have type=mol")

    steps: List[Dict[str, Any]] = []
    leaves: List[Dict[str, Any]] = []

    def visit(node: Mapping[str, Any], node_path: str) -> None:
        if node.get("type") != "mol":
            raise ValueError("{} must have type=mol".format(node_path))
        raw_product = str(node.get("smiles") or "")
        normalized_product = canonical(raw_product)
        children = list(node.get("children") or ())
        if not children:
            leaves.append({
                "node_id": node_path,
                "smiles": raw_product,
                "canonical_smiles": normalized_product,
                "declared_in_stock": node.get("in_stock"),
            })
            return
        if len(children) != 1 or children[0].get("type") != "reaction":
            raise ValueError(
                "{} must have exactly one reaction child".format(node_path)
            )
        reaction = children[0]
        reaction_path = "{}.children[0]".format(node_path)
        precursor_nodes = list(reaction.get("children") or ())
        if not precursor_nodes or any(
            item.get("type") != "mol" for item in precursor_nodes
        ):
            raise ValueError(
                "{} must contain molecule children".format(reaction_path)
            )
        raw_precursors = [str(item.get("smiles") or "") for item in precursor_nodes]
        normalized_precursors = [canonical(item) for item in raw_precursors]
        position = len(steps) + 1
        step_id = "{}:step:{:04d}".format(route_id, position)
        precursor_node_ids = [
            "{}.children[0].children[{}]".format(node_path, index)
            for index in range(len(precursor_nodes))
        ]
        steps.append({
            "dataset_split": split,
            "route_id": route_id,
            "step_id": step_id,
            "traversal_index": position,
            "reaction_node_id": reaction_path,
            "product_node_id": node_path,
            "product": raw_product,
            "normalized_product": normalized_product,
            "precursor_node_ids": precursor_node_ids,
            "precursors": raw_precursors,
            "normalized_precursors": normalized_precursors,
            "source_metadata": dict(reaction.get("metadata") or {}),
            "metadata_tree_consistency": metadata_tree_consistency(
                raw_product,
                raw_precursors,
                dict(reaction.get("metadata") or {}),
            ),
            "downstream_step_ids": [],
        })
        for index, precursor in enumerate(precursor_nodes):
            visit(precursor, precursor_node_ids[index])

    visit(tree, "$")
    if not steps:
        raise ValueError("route contains no reaction nodes")
    step_by_product_node = {
        step["product_node_id"]: step["step_id"] for step in steps
    }
    for step in steps:
        step["downstream_step_ids"] = [
            step_by_product_node[node_id]
            for node_id in step["precursor_node_ids"]
            if node_id in step_by_product_node
        ]
    return {
        "dataset_split": split,
        "source_file": source_file,
        "source_route_index": source_route_index,
        "route_id": route_id,
        "target": str(tree.get("smiles") or ""),
        "normalized_target": canonical(tree.get("smiles")),
        "starting_materials": leaves,
        "reference_steps": steps,
        "total_steps": len(steps),
        "traversal": "depth_first_preorder_children_array_order",
        "node_id_rule": "JSON path from root '$'; arrays use zero-based indices",
        "step_id_rule": "route_id:step:one-based-four-digit-DFS-position",
    }


def validate_split(dataset_root: Path, split: str) -> Dict[str, Any]:
    paths = split_paths(dataset_root, split)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing dataset files: " + ", ".join(missing))
    targets = paths["targets"].read_text(encoding="utf-8").splitlines()
    stock = [
        line for line in paths["stock"].read_text(encoding="utf-8").splitlines()
        if line
    ]
    errors: List[Dict[str, Any]] = []
    leaf_strings: set[str] = set()
    reaction_count = 0
    step_counts: List[int] = []
    branched_routes = 0
    repeated_precursor_steps = 0
    metadata_tree_conflicts = 0
    invalid_indices: List[int] = []
    route_count = 0

    for index, tree in enumerate(iter_json_array(paths["routes"])):
        route_count += 1
        route_id = route_id_for(split, index)
        try:
            reference = extract_reference_route(
                tree, split=split, source_file=paths["routes"].name,
                source_route_index=index,
            )
            if index >= len(targets):
                raise ValueError("route has no corresponding target line")
            if reference["target"] != targets[index]:
                raise ValueError("tree root and target line differ as raw SMILES")
            route_has_branch = False
            for leaf in reference["starting_materials"]:
                leaf_strings.add(leaf["smiles"])
                if leaf["declared_in_stock"] is not True:
                    errors.append({
                        "route_id": route_id,
                        "node_id": leaf["node_id"],
                        "error": "leaf_not_declared_in_stock",
                    })
            nonleaf_ids = {
                step["product_node_id"] for step in reference["reference_steps"]
            }
            for step in reference["reference_steps"]:
                if Counter(step["precursors"]).most_common(1)[0][1] > 1:
                    repeated_precursor_steps += 1
                if step["metadata_tree_consistency"]["status"] == "conflict":
                    metadata_tree_conflicts += 1
                    errors.append({
                        "route_id": route_id,
                        "step_id": step["step_id"],
                        "reaction_node_id": step["reaction_node_id"],
                        "error": "metadata_tree_conflict",
                        "policy": (
                            "tree_parent_and_direct_children_are_authoritative"
                        ),
                    })
                continuing = sum(
                    node_id in nonleaf_ids for node_id in step["precursor_node_ids"]
                )
                route_has_branch = route_has_branch or continuing > 1
            if route_has_branch:
                branched_routes += 1
            reaction_count += reference["total_steps"]
            step_counts.append(reference["total_steps"])
        except Exception as exc:
            invalid_indices.append(index)
            errors.append({
                "route_id": route_id,
                "source_route_index": index,
                "error": "invalid_reference",
                "type": type(exc).__name__,
                "message": str(exc),
            })

    if route_count != len(targets):
        errors.append({
            "error": "route_target_count_mismatch",
            "route_count": route_count,
            "target_count": len(targets),
        })
    stock_set = set(stock)
    if leaf_strings != stock_set:
        errors.append({
            "error": "stock_leaf_union_mismatch",
            "missing_from_stock_count": len(leaf_strings - stock_set),
            "extra_in_stock_count": len(stock_set - leaf_strings),
            "missing_examples": sorted(leaf_strings - stock_set)[:20],
            "extra_examples": sorted(stock_set - leaf_strings)[:20],
        })
    return {
        "dataset_split": split,
        "files": {
            name: {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for name, path in paths.items()
        },
        "route_count": route_count,
        "target_count": len(targets),
        "unique_target_string_count": len(set(targets)),
        "stock_line_count": len(stock),
        "unique_stock_string_count": len(stock_set),
        "reaction_node_count": reaction_count,
        "reaction_count_min": min(step_counts) if step_counts else None,
        "reaction_count_max": max(step_counts) if step_counts else None,
        "reaction_count_mean": (
            sum(step_counts) / len(step_counts) if step_counts else None
        ),
        "routes_with_multiple_continuing_precursors": branched_routes,
        "reaction_nodes_with_repeated_precursor_strings": repeated_precursor_steps,
        "metadata_tree_conflict_count": metadata_tree_conflicts,
        "metadata_conflict_policy": (
            "record conflict; use tree parent and direct molecule children"
        ),
        "stock_equals_reference_leaf_union": leaf_strings == stock_set,
        "valid_reference_routes": route_count - len(invalid_indices),
        "invalid_reference_routes": len(invalid_indices),
        "invalid_route_indices": invalid_indices,
        "errors": errors,
        "normalization_rule": (
            "RDKit canonical isomeric SMILES after clearing atom-map numbers; "
            "stereochemistry, charge and isotopes retained"
        ),
    }


def selected_indices(
    count: int, start_index: int, limit: Optional[int]
) -> List[int]:
    values = list(range(start_index, count))
    return values[:limit] if limit is not None else values


def iter_selected_trees(
    path: Path, selected: set[int]
) -> Iterator[Tuple[int, Mapping[str, Any]]]:
    for index, tree in enumerate(iter_json_array(path)):
        if index in selected:
            yield index, tree


def coefficient_count(value: Any) -> Tuple[Optional[int], Optional[str]]:
    text = str(value if value is not None else "1").strip()
    if not text:
        text = "1"
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None, "ambiguous_stoichiometric_coefficient:{}".format(text)
    if number <= 0 or number != number.to_integral_value():
        return None, "non_positive_or_non_integer_coefficient:{}".format(text)
    return int(number), None


def expand_candidate_stoichiometry(
    candidate: ReactionCandidate,
) -> Tuple[ReactionCandidate, Optional[str], List[Dict[str, Any]]]:
    expanded: List[MoleculeNode] = []
    rules: List[Dict[str, Any]] = []
    for index, precursor in enumerate(candidate.precursors):
        coefficient = (
            candidate.coefficients[index]
            if index < len(candidate.coefficients) else "1"
        )
        count, error = coefficient_count(coefficient)
        if error is not None:
            return candidate, error, rules
        expanded.extend([precursor] * int(count))
        if int(count) != 1:
            rules.append({
                "precursor_index": index,
                "coefficient": str(coefficient),
                "expanded_occurrences": int(count),
            })
    if tuple(expanded) == candidate.precursors:
        return candidate, None, rules
    return replace(
        candidate,
        precursors=tuple(expanded),
        coefficients=tuple("1" for _ in expanded),
    ), None, rules


def candidate_precursors(candidate: ReactionCandidate) -> Dict[str, Any]:
    prepared, error, rules = expand_candidate_stoichiometry(candidate)
    raw = [
        item.smiles or item.canonical_smiles or item.query
        for item in prepared.precursors
    ]
    normalized: List[str] = []
    if error is None:
        try:
            normalized = [
                canonical(item.canonical_smiles or item.smiles)
                for item in prepared.precursors
            ]
        except ValueError as exc:
            error = "invalid_candidate:{}".format(exc)
    return {
        "raw": raw,
        "normalized": normalized,
        "multiplicity": dict(sorted(Counter(normalized).items())),
        "coefficient_expansions": rules,
        "comparison_error": error,
    }


def complete_signature(
    product: str, precursors: Sequence[str]
) -> Tuple[str, Tuple[str, ...]]:
    return canonical(product), tuple(sorted(canonical(item) for item in precursors))


def candidate_signature(
    candidate: ReactionCandidate,
) -> Optional[Tuple[str, Tuple[str, ...]]]:
    values = candidate_precursors(candidate)
    if values["comparison_error"] is not None:
        return None
    product = candidate.target.canonical_smiles or candidate.target.smiles
    return canonical(product), tuple(sorted(values["normalized"]))


def candidate_allowed(
    candidate: ReactionCandidate, config: PlannerConfig
) -> Tuple[bool, Optional[str]]:
    precursors = candidate.noncurrency_precursors
    reason = None
    if not precursors:
        reason = "no_noncurrency_precursor"
    elif len(precursors) > config.max_precursors_per_step:
        reason = "too_many_precursors"
    elif candidate.target.key in {item.key for item in precursors}:
        reason = "self_cycle"
    elif config.require_ec and not candidate.ec_numbers:
        reason = "missing_ec"
    elif config.reject_generic_structures and any(item.generic for item in precursors):
        reason = "generic_structure"
    elif (
        candidate.source in {"MODEL", "BOTH"}
        and candidate.model_score is not None
        and candidate.model_score < config.min_model_score
    ):
        reason = "low_model_score"
    elif (
        candidate.feasibility is not None
        and candidate.feasibility < config.min_feasibility
    ):
        reason = "low_feasibility"
    elif (
        candidate.atom_conservation is not None
        and candidate.atom_conservation < config.min_atom_conservation
    ):
        reason = "low_atom_conservation"
    elif (
        candidate.approx_structural_overlap is not None
        and candidate.approx_structural_overlap
        < config.min_approx_structural_overlap
    ):
        reason = "low_approx_structural_overlap"
    elif candidate.toxicity is not None and candidate.toxicity > config.max_toxicity:
        reason = "toxicity_threshold"
    elif (
        candidate.step_score is not None
        and candidate.step_score < config.min_candidate_score
    ):
        reason = "low_candidate_score"
    return reason is None, reason


def candidate_output(
    candidate: ReactionCandidate,
    *,
    split: str,
    route_id: str,
    step_id: str,
    stage: str,
    rank: Optional[int],
    config: PlannerConfig,
    raw_content: Any = None,
    invalid_reason: Optional[str] = None,
) -> Dict[str, Any]:
    precursor_info = candidate_precursors(candidate)
    allowed, filter_reason = candidate_allowed(candidate, config)
    metadata = dict(candidate.metadata)
    return {
        "dataset_split": split,
        "route_id": route_id,
        "step_id": step_id,
        "stage": stage,
        "rank": rank,
        "candidate_id": candidate.candidate_id,
        "source": candidate.source,
        "raw_content": raw_content,
        "raw_precursors": precursor_info["raw"],
        "normalized_precursors": precursor_info["normalized"],
        "precursor_multiplicity": precursor_info["multiplicity"],
        "coefficient_expansions": precursor_info["coefficient_expansions"],
        "comparison_error": precursor_info["comparison_error"],
        "valid": invalid_reason is None and precursor_info["comparison_error"] is None,
        "invalid_reason": invalid_reason or precursor_info["comparison_error"],
        "database_reaction_ids": list(candidate.database_reaction_ids),
        "model_score": candidate.model_score,
        "step_score": candidate.step_score,
        "step_cost": candidate.step_cost,
        "would_be_filtered": not allowed,
        "filter_reason": filter_reason,
        "fusion_mapping": {
            "database_candidate_ids": metadata.get(
                "database_source_candidate_ids",
                [metadata["database_candidate_id"]]
                if metadata.get("database_candidate_id") else [],
            ),
            "model_candidate_ids": metadata.get(
                "model_source_candidate_ids",
                [metadata["model_candidate_id"]]
                if metadata.get("model_candidate_id") else [],
            ),
        },
        "ec_numbers": list(candidate.ec_numbers),
        "template": candidate.template,
        "metadata": metadata,
    }


def invalid_model_output(
    audit: Mapping[str, Any],
    *,
    split: str,
    route_id: str,
    step_id: str,
) -> Dict[str, Any]:
    return {
        "dataset_split": split,
        "route_id": route_id,
        "step_id": step_id,
        "stage": "MODEL_RAW",
        "rank": audit.get("raw_rank"),
        "candidate_id": None,
        "source": "MODEL",
        "raw_content": audit.get("raw_prediction"),
        "raw_precursors": None,
        "normalized_precursors": None,
        "precursor_multiplicity": None,
        "coefficient_expansions": None,
        "comparison_error": None,
        "valid": False,
        "invalid_reason": audit.get("invalid_reason"),
        "database_reaction_ids": [],
        "model_score": None,
        "step_score": None,
        "step_cost": None,
        "would_be_filtered": None,
        "filter_reason": None,
        "fusion_mapping": None,
        "ec_numbers": [],
        "template": None,
        "metadata": {},
    }


def cache_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, int]:
    return {
        "hits": max(0, int(after.get("hits", 0)) - int(before.get("hits", 0))),
        "misses": max(
            0, int(after.get("misses", 0)) - int(before.get("misses", 0))
        ),
    }


def generate_step_candidates(
    *,
    database: EnzymeReactionDatabase,
    model_client: EnzRetroClient,
    product: str,
    building_blocks: Sequence[MoleculeNode],
    config: PlannerConfig,
    n_db: int,
    n_model: int,
    split: str,
    route_id: str,
    step_id: str,
) -> Dict[str, Any]:
    """Call both generators, retain source snapshots, then freeze full-pool rank."""
    target = node_for_smiles(product)
    source_errors: Dict[str, Dict[str, str]] = {}
    tool_calls: List[Dict[str, Any]] = []
    candidate_records: List[Dict[str, Any]] = []

    db_before = database.cache_info()
    db_started = time.monotonic()
    database_candidates: Tuple[ReactionCandidate, ...] = ()
    try:
        database_candidates = database.backward_candidates(
            target,
            limit=n_db,
            respect_direction=config.respect_reaction_direction,
            allow_unknown_direction=config.allow_unknown_direction,
            currency_ids=config.currency_metabolite_ids,
        )
    except Exception as exc:
        source_errors["DATABASE"] = {
            "type": type(exc).__name__, "message": str(exc)
        }
    db_elapsed = time.monotonic() - db_started
    db_after = database.cache_info()
    db_cache = cache_delta(db_before, db_after)
    database_valid_count = sum(
        candidate_precursors(candidate)["comparison_error"] is None
        for candidate in database_candidates
    )
    tool_calls.append({
        "dataset_split": split,
        "route_id": route_id,
        "step_id": step_id,
        "source": "DATABASE",
        "product": product,
        "parameters": {
            "limit": n_db,
            "respect_direction": config.respect_reaction_direction,
            "allow_unknown_direction": config.allow_unknown_direction,
        },
        "requested": n_db,
        "raw_returned": len(database_candidates),
        "valid_returned": database_valid_count,
        "cache_hit": db_cache["hits"] > 0 and db_cache["misses"] == 0,
        "actual_execution": db_cache["misses"] > 0,
        "elapsed_seconds": round(db_elapsed, 6),
        "error": source_errors.get("DATABASE"),
        "llm_api_calls": 0,
    })

    model_before = model_client.cache_info()
    model_started = time.monotonic()
    model_candidates: Tuple[ReactionCandidate, ...] = ()
    model_audit: Dict[str, Any] = {
        "requested": n_model, "raw_returned": 0,
        "valid_returned": 0, "records": [],
    }
    try:
        audit_method = getattr(model_client, "predict_candidates_with_audit", None)
        if audit_method is not None:
            model_candidates, model_audit = audit_method(
                target,
                building_blocks,
                top_k=n_model,
                currency_ids=config.currency_metabolite_ids,
                normalizer=node_for_smiles,
            )
        else:
            model_candidates = model_client.predict_candidates(
                target,
                building_blocks,
                top_k=n_model,
                currency_ids=config.currency_metabolite_ids,
                normalizer=node_for_smiles,
            )
            model_audit = {
                "requested": n_model,
                "raw_returned": len(model_candidates),
                "valid_returned": len(model_candidates),
                "records": [],
            }
    except Exception as exc:
        source_errors["MODEL"] = {
            "type": type(exc).__name__, "message": str(exc)
        }
    model_elapsed = time.monotonic() - model_started
    model_after = model_client.cache_info()
    model_cache = cache_delta(model_before, model_after)
    tool_calls.append({
        "dataset_split": split,
        "route_id": route_id,
        "step_id": step_id,
        "source": "MODEL",
        "product": product,
        "parameters": {
            "top_k": n_model,
            "backend": model_after.get("model"),
            "runtime_device": model_after.get("runtime_device"),
            "building_block_count": len(building_blocks),
        },
        "requested": n_model,
        "raw_returned": model_audit.get("raw_returned", 0),
        "valid_returned": len(model_candidates),
        "cache_hit": model_cache["hits"] > 0 and model_cache["misses"] == 0,
        "actual_execution": model_cache["misses"] > 0,
        "elapsed_seconds": round(model_elapsed, 6),
        "error": source_errors.get("MODEL"),
        "llm_api_calls": 0,
    })

    for rank, candidate in enumerate(database_candidates, 1):
        candidate_records.append(candidate_output(
            candidate, split=split, route_id=route_id, step_id=step_id,
            stage="DATABASE_RAW", rank=rank, config=config,
            raw_content=candidate.to_dict(),
        ))
    raw_by_id = {
        record.get("candidate_id"): record.get("raw_prediction")
        for record in model_audit.get("records", [])
        if record.get("candidate_id")
    }
    for rank, candidate in enumerate(model_candidates, 1):
        candidate_records.append(candidate_output(
            candidate, split=split, route_id=route_id, step_id=step_id,
            stage="MODEL_RAW", rank=rank, config=config,
            raw_content=raw_by_id.get(candidate.candidate_id),
        ))
    for audit in model_audit.get("records", []):
        if not audit.get("valid"):
            candidate_records.append(invalid_model_output(
                audit, split=split, route_id=route_id, step_id=step_id
            ))

    result: Dict[str, Any] = {
        "database_candidates": database_candidates,
        "model_candidates": model_candidates,
        "ranked_candidates": (),
        "candidate_records": candidate_records,
        "tool_calls": tool_calls,
        "errors": source_errors,
        "counts": {
            "database": {
                "requested": n_db,
                "raw_returned": len(database_candidates),
                "valid_returned": database_valid_count,
            },
            "model": {
                "requested": n_model,
                "raw_returned": int(model_audit.get("raw_returned", 0)),
                "valid_returned": len(model_candidates),
            },
            "merged_count": None,
            "ranked_count": None,
        },
    }
    if source_errors:
        return result

    prepared_db = tuple(
        expand_candidate_stoichiometry(candidate)[0]
        for candidate in database_candidates
        if candidate_signature(candidate) is not None
    )
    prepared_model = tuple(
        expand_candidate_stoichiometry(candidate)[0]
        for candidate in model_candidates
        if candidate_signature(candidate) is not None
    )
    try:
        merged = fuse_candidates_by_complete_precursors(
            prepared_db, prepared_model, config
        )
        ranked = score_and_rank_candidates(merged, building_blocks, config)
    except Exception as exc:
        result["errors"]["SCORING"] = {
            "type": type(exc).__name__, "message": str(exc)
        }
        return result

    result["ranked_candidates"] = ranked
    result["counts"]["merged_count"] = len(merged)
    result["counts"]["ranked_count"] = len(ranked)
    for rank, candidate in enumerate(ranked, 1):
        result["candidate_records"].append(candidate_output(
            candidate, split=split, route_id=route_id, step_id=step_id,
            stage="MERGED_RANKED", rank=rank, config=config,
        ))
    return result


def first_matching_rank(
    candidates: Sequence[ReactionCandidate],
    expected: Tuple[str, Tuple[str, ...]],
) -> Optional[int]:
    for rank, candidate in enumerate(candidates, 1):
        if candidate_signature(candidate) == expected:
            return rank
    return None


def stable_precursor_mapping(
    candidate: ReactionCandidate, step: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    info = candidate_precursors(candidate)
    unmatched = list(range(len(step["normalized_precursors"])))
    mapping: List[Dict[str, Any]] = []
    for candidate_index, value in enumerate(info["normalized"]):
        reference_index = next(
            index for index in unmatched
            if step["normalized_precursors"][index] == value
        )
        unmatched.remove(reference_index)
        mapping.append({
            "candidate_precursor_index": candidate_index,
            "reference_precursor_index": reference_index,
            "reference_node_id": step["precursor_node_ids"][reference_index],
            "canonical_smiles": value,
        })
    if unmatched:
        raise RuntimeError("matched signature left unmapped reference nodes")
    return mapping


def base_step_record(step: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        **dict(step),
        "status": "not_evaluated",
        "database_counts": None,
        "model_counts": None,
        "merged_count": None,
        "ranked_count": None,
        "database_hit": None,
        "model_hit": None,
        "both_hit": None,
        "hybrid_hit": None,
        "database_raw_rank": None,
        "model_raw_rank": None,
        "hybrid_rank": None,
        "top1_hit": None,
        "top2_hit": None,
        "top3_hit": None,
        "matched_candidate_id": None,
        "selected_precursor_mapping": None,
        "elapsed_seconds": None,
        "error": None,
    }


def evaluate_route(
    reference: Mapping[str, Any],
    *,
    database: EnzymeReactionDatabase,
    model_client: EnzRetroClient,
    split_stock: Sequence[MoleculeNode],
    building_block_strategy: str,
    config: PlannerConfig,
    n_db: int,
    n_model: int,
) -> Dict[str, Any]:
    route_started = time.monotonic()
    route_id = str(reference["route_id"])
    split = str(reference["dataset_split"])
    leaf_nodes = tuple(
        node_for_smiles(item["smiles"])
        for item in reference["starting_materials"]
    )
    if building_block_strategy == "split_stock":
        building_blocks = tuple(split_stock)
    else:
        unique: Dict[str, MoleculeNode] = {}
        for node in leaf_nodes:
            unique.setdefault(node.key, node)
        building_blocks = tuple(unique.values())

    step_records = [
        base_step_record(step) for step in reference["reference_steps"]
    ]
    candidates: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []
    found_steps = attempted_steps = started_steps = 0
    status = "complete"
    stop_step_id = stop_reason = None
    endpoint_match: Optional[bool] = None
    terminal_node_ids = {
        item["node_id"] for item in reference["starting_materials"]
    }
    matched_terminal_smiles: List[str] = []

    for index, step in enumerate(reference["reference_steps"]):
        started_steps += 1
        step_started = time.monotonic()
        generated = generate_step_candidates(
            database=database,
            model_client=model_client,
            product=step["product"],
            building_blocks=building_blocks,
            config=config,
            n_db=n_db,
            n_model=n_model,
            split=split,
            route_id=route_id,
            step_id=step["step_id"],
        )
        candidates.extend(generated["candidate_records"])
        tool_calls.extend(generated["tool_calls"])
        record = step_records[index]
        record.update({
            "database_counts": generated["counts"]["database"],
            "model_counts": generated["counts"]["model"],
            "merged_count": generated["counts"]["merged_count"],
            "ranked_count": generated["counts"]["ranked_count"],
            "elapsed_seconds": round(time.monotonic() - step_started, 6),
        })
        if generated["errors"]:
            status = "error"
            record["status"] = "error"
            record["error"] = generated["errors"]
            stop_step_id = step["step_id"]
            stop_reason = "candidate_generation_or_scoring_error"
            break

        expected = complete_signature(step["product"], step["precursors"])
        db_rank = first_matching_rank(
            generated["database_candidates"], expected
        )
        model_rank = first_matching_rank(
            generated["model_candidates"], expected
        )
        hybrid_rank = first_matching_rank(
            generated["ranked_candidates"], expected
        )
        database_hit = db_rank is not None
        model_hit = model_rank is not None
        hybrid_hit = hybrid_rank is not None
        if hybrid_hit != (database_hit or model_hit):
            status = "error"
            record["status"] = "error"
            record["error"] = {
                "INVARIANT": {
                    "type": "FusionInvariantError",
                    "message": (
                        "hybrid_hit must equal database_hit OR model_hit"
                    ),
                }
            }
            stop_step_id = step["step_id"]
            stop_reason = "fusion_invariant_violation"
            break

        attempted_steps += 1
        record.update({
            "database_hit": database_hit,
            "model_hit": model_hit,
            "both_hit": database_hit and model_hit,
            "hybrid_hit": hybrid_hit,
            "database_raw_rank": db_rank,
            "model_raw_rank": model_rank,
            "hybrid_rank": hybrid_rank,
            "top1_hit": bool(hybrid_rank is not None and hybrid_rank <= 1),
            "top2_hit": bool(hybrid_rank is not None and hybrid_rank <= 2),
            "top3_hit": bool(hybrid_rank is not None and hybrid_rank <= 3),
        })
        if not hybrid_hit:
            status = "miss"
            record["status"] = "miss"
            stop_step_id = step["step_id"]
            stop_reason = "reference_complete_precursor_multiset_not_returned"
            break

        found_steps += 1
        selected = generated["ranked_candidates"][int(hybrid_rank) - 1]
        selected_mapping = stable_precursor_mapping(selected, step)
        matched_terminal_smiles.extend(
            item["canonical_smiles"] for item in selected_mapping
            if item["reference_node_id"] in terminal_node_ids
        )
        record.update({
            "status": "hit",
            "matched_candidate_id": selected.candidate_id,
            "selected_precursor_mapping": selected_mapping,
        })

    if found_steps == reference["total_steps"]:
        status = "complete"
        endpoint_match = Counter(matched_terminal_smiles) == Counter(
            item["canonical_smiles"]
            for item in reference["starting_materials"]
        )
        if not endpoint_match:
            status = "error"
            stop_reason = "terminal_leaf_multiset_mismatch"
    elif status in {"miss", "error"}:
        endpoint_match = False

    total_steps = int(reference["total_steps"])
    route = {
        "dataset_split": split,
        "source_file": reference["source_file"],
        "source_route_index": reference["source_route_index"],
        "route_id": route_id,
        "target": reference["target"],
        "starting_materials": json.dumps(
            [item["smiles"] for item in reference["starting_materials"]],
            ensure_ascii=False,
        ),
        "status": status,
        "found_steps": found_steps,
        "total_steps": total_steps,
        "attempted_steps": attempted_steps,
        "started_steps": started_steps,
        "not_evaluated_steps": total_steps - (
            started_steps if status in {"error", "budget_exceeded"}
            else attempted_steps
        ),
        "coverage_ratio": found_steps / total_steps if total_steps else None,
        "coverage_text": "{}/{}".format(found_steps, total_steps),
        "top1_hits": sum(item["top1_hit"] is True for item in step_records),
        "top2_hits": sum(item["top2_hit"] is True for item in step_records),
        "top3_hits": sum(item["top3_hit"] is True for item in step_records),
        "stop_step_id": stop_step_id,
        "stop_reason": stop_reason,
        "endpoint_match": endpoint_match,
        "elapsed_seconds": round(time.monotonic() - route_started, 6),
    }
    return {
        "route": route,
        "steps": step_records,
        "candidates": candidates,
        "tool_calls": tool_calls,
    }


def invalid_route_record(
    split: str, source_file: str, index: int, target: str, exc: Exception
) -> Dict[str, Any]:
    return {
        "dataset_split": split,
        "source_file": source_file,
        "source_route_index": index,
        "route_id": route_id_for(split, index),
        "target": target,
        "starting_materials": "[]",
        "status": "invalid_reference",
        "found_steps": 0,
        "total_steps": 0,
        "attempted_steps": 0,
        "started_steps": 0,
        "not_evaluated_steps": 0,
        "coverage_ratio": None,
        "coverage_text": "0/0",
        "top1_hits": 0,
        "top2_hits": 0,
        "top3_hits": 0,
        "stop_step_id": None,
        "stop_reason": "{}: {}".format(type(exc).__name__, exc),
        "endpoint_match": None,
        "elapsed_seconds": 0.0,
    }


def append_route_csv(path: Path, route: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROUTE_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: route.get(field) for field in ROUTE_FIELDS})


def read_routes_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            record: Dict[str, Any] = dict(raw)
            for field in (
                "source_route_index", "found_steps", "total_steps",
                "attempted_steps", "started_steps", "not_evaluated_steps",
                "top1_hits", "top2_hits", "top3_hits",
            ):
                record[field] = int(record[field]) if record.get(field) else 0
            for field in ("coverage_ratio", "elapsed_seconds"):
                record[field] = (
                    float(record[field]) if record.get(field) else None
                )
            records.append(record)
    return records


def summarize(
    routes: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    scored = [route for route in routes if route.get("status") in {"complete", "miss"}]
    scored_ids = {route["route_id"] for route in scored}
    scored_steps = [
        step for step in steps
        if step.get("route_id") in scored_ids
        and step.get("status") in {"hit", "miss"}
    ]
    complete = sum(route.get("status") == "complete" for route in scored)
    found = sum(int(route.get("found_steps", 0)) for route in scored)
    total = sum(int(route.get("total_steps", 0)) for route in scored)
    attempted = len(scored_steps)
    top_counts = {
        str(k): sum(step.get("top{}_hit".format(k)) is True for step in scored_steps)
        for k in (1, 2, 3)
    }
    db_hits = sum(step.get("database_hit") is True for step in scored_steps)
    model_hits = sum(step.get("model_hit") is True for step in scored_steps)
    both = sum(step.get("both_hit") is True for step in scored_steps)
    hybrid = sum(step.get("hybrid_hit") is True for step in scored_steps)
    statuses = Counter(str(route.get("status")) for route in routes)
    mean_coverage_sum = sum(
        float(route.get("coverage_ratio") or 0.0) for route in scored
    )
    return {
        "protocol": PROTOCOL,
        "result_interpretation": (
            "reference-guided local tool-chain coverage and candidate ranking; "
            "not autonomous LLM Agent or unconstrained MCTS success"
        ),
        "route_outcomes": {
            "total_selected_routes": len(routes),
            "valid_reference_routes": len(routes) - statuses["invalid_reference"],
            "scored_routes": len(scored),
            "complete_routes": statuses["complete"],
            "miss_routes": statuses["miss"],
            "invalid_reference_routes": statuses["invalid_reference"],
            "error_routes": statuses["error"],
            "budget_exceeded_routes": statuses["budget_exceeded"],
        },
        "reference_steps_in_scored_routes": total,
        "attempted_steps_in_scored_routes": attempted,
        "found_steps_in_scored_routes": found,
        "route_coverage_mean": ratio(mean_coverage_sum, len(scored)),
        "aggregate_step_coverage": ratio(found, total),
        "guided_completion_rate": ratio(complete, len(scored)),
        "top_k_accuracy_on_attempted_steps": {
            key: ratio(value, attempted) for key, value in top_counts.items()
        },
        "source_attribution_on_attempted_steps": {
            "database_hit": ratio(db_hits, attempted),
            "model_hit": ratio(model_hits, attempted),
            "both_hit": ratio(both, attempted),
            "database_only": ratio(db_hits - both, attempted),
            "model_only": ratio(model_hits - both, attempted),
            "neither_hit": ratio(attempted - hybrid, attempted),
            "hybrid_hit": ratio(hybrid, attempted),
        },
        "excluded_from_main_metrics": {
            "statuses": ["invalid_reference", "error", "budget_exceeded"],
            "route_count": len(routes) - len(scored),
        },
        "llm_api_calls": 0,
    }


def compatible_config(value: Mapping[str, Any]) -> Dict[str, Any]:
    ignored = {"started_at", "completed_at", "resume"}
    result = {
        key: item for key, item in value.items() if key not in ignored
    }
    model = dict(result.get("model") or {})
    model.pop("runtime_device", None)
    result["model"] = model
    return result


def load_model_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("--model-config must contain a JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=PROJECT_ROOT.parent / "dataset" / "PaRoutes",
    )
    parser.add_argument(
        "--split", action="append", choices=("n1", "n5", "all"),
        help="n1, n5 or all; may be repeated (default: all)",
    )
    parser.add_argument("--n-model", type=int, default=10)
    parser.add_argument("--n-db", type=int, default=20)
    parser.add_argument(
        "--database-path", "--db-path", dest="database_path", type=Path
    )
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--moo-root", type=Path)
    parser.add_argument(
        "--single-fn-type",
        choices=("biocyc", "ecreact", "model_xlsx", "xlsx"),
    )
    parser.add_argument("--ec-level", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--candidate-beam-width", type=int, default=20)
    parser.add_argument(
        "--building-block-strategy",
        choices=("route_leaves", "split_stock"),
        default="route_leaves",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--start-index", "--offset", dest="start_index", type=int, default=0
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=False
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if min(args.n_model, args.n_db, args.candidate_beam_width) <= 0:
        parser.error("candidate budgets and beam width must be positive")
    requested_splits = args.split or ["all"]
    splits = SPLITS if "all" in requested_splits else tuple(
        dict.fromkeys(requested_splits)
    )
    if not args.dataset_root.is_dir():
        parser.error("dataset root does not exist: {}".format(args.dataset_root))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    validation: Dict[str, Any] = {
        "protocol": PROTOCOL,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "splits": {},
    }
    for split in splits:
        print("[validation] {}".format(split), flush=True)
        validation["splits"][split] = validate_split(args.dataset_root, split)
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            split_dir / "validation.json", validation["splits"][split]
        )
    write_json(output_dir / "validation.json", validation)

    model_values = load_model_config(args.model_config)
    moo_root = Path(
        args.moo_root
        or model_values.get("moo_root")
        or (PROJECT_ROOT.parent / "MooRetro-master")
    ).resolve()
    single_fn_type = (
        args.single_fn_type
        or model_values.get("single_fn_type")
        or "biocyc"
    )
    ec_level = int(args.ec_level or model_values.get("ec_level") or 3)
    model_xlsx_value = model_values.get("model_xlsx")
    model_xlsx = (
        Path(model_xlsx_value).expanduser().resolve()
        if model_xlsx_value
        else (moo_root / "model_output.xlsx")
    )
    if single_fn_type in {"model_xlsx", "xlsx"} and model_xlsx.is_file():
        os.environ.setdefault("MOO_RETRO_MODEL_XLSX", str(model_xlsx))
    if single_fn_type == "biocyc":
        model_artifact = (
            moo_root / "enzretro" / "model" / "biocyc_backward"
            / "EC{}".format(ec_level) / "model" / "pytorch_model.pth"
        )
    elif single_fn_type == "ecreact":
        model_artifact = (
            moo_root / "enzretro" / "model" / "ecreact"
            / "backward_ecs{}".format(ec_level) / "model" / "pytorch_model.pth"
        )
    else:
        model_artifact = model_xlsx
    database_path = Path(
        args.database_path
        or PROJECT_ROOT / "chemcrow" / "data" / "enzyme_retro.db"
    ).resolve()
    selection = {
        split: selected_indices(
            validation["splits"][split]["target_count"],
            args.start_index,
            args.limit,
        )
        for split in splits
    }
    run_config = {
        "protocol": PROTOCOL,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "execution_mode": "deterministic_python_controller_no_remote_llm",
        "llm_api_calls": 0,
        "dataset_root": str(args.dataset_root.resolve()),
        "splits": list(splits),
        "selection": {
            "start_index_zero_based_per_split": args.start_index,
            "limit_per_split": args.limit,
            "selected_counts": {
                split: len(indices) for split, indices in selection.items()
            },
        },
        "candidate_budgets": {
            "n_model": args.n_model,
            "n_db": args.n_db,
            "candidate_beam_width_for_filter_annotation_only": (
                args.candidate_beam_width
            ),
            "ranking_pool_is_not_beam_truncated": True,
        },
        "model": {
            "moo_root": str(moo_root),
            "single_fn_type": single_fn_type,
            "backend_mode": (
                "database_lookup"
                if single_fn_type in {"model_xlsx", "xlsx"}
                else "neural_model"
            ),
            "ec_level": ec_level,
            "client_top_k_semantics": (
                "single_fn has no top_k argument; client slices actual decoder "
                "output to the fixed requested budget"
            ),
            "entry_point": str(moo_root / "utils" / "single_retro_fn.py"),
            "model_xlsx": (
                str(model_xlsx)
                if single_fn_type in {"model_xlsx", "xlsx"} else None
            ),
            "artifact": {
                "path": str(model_artifact),
                "exists": model_artifact.is_file(),
                "size_bytes": (
                    model_artifact.stat().st_size
                    if model_artifact.is_file() else None
                ),
                "sha256": (
                    file_sha256(model_artifact)
                    if model_artifact.is_file() else None
                ),
            },
            "runtime_device": None,
        },
        "database": {
            "path": str(database_path),
            "read_only": True,
            "sha256": (
                file_sha256(database_path) if database_path.is_file() else None
            ),
        },
        "seed": args.seed,
        "normalization": (
            "RDKit canonical isomeric SMILES after atom-map removal; retain "
            "stereochemistry, charge, isotopes and multiplicity"
        ),
        "fusion": "exact_current_product_plus_complete_precursor_multiset",
        "ranking": "score_and_rank_candidates: step_cost asc, candidate_id asc",
        "traversal": "DFS preorder; precursor children in source array order",
        "building_block_strategy": args.building_block_strategy,
        "validate_only": args.validate_only,
        "resume": args.resume,
        "source_hashes": {
            "evaluation_script": file_sha256(Path(__file__).resolve()),
            "model_client": file_sha256(
                PROJECT_ROOT / "chemcrow" / "planner" / "model_client.py"
            ),
            "scoring": file_sha256(
                PROJECT_ROOT / "chemcrow" / "planner" / "scoring.py"
            ),
        },
    }
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        if not args.resume:
            parser.error(
                "output directory already contains a run; use --resume or another directory"
            )
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if compatible_config(previous) != compatible_config(run_config):
            parser.error("resume configuration does not match existing run")
        run_config = previous
    else:
        write_json(config_path, run_config)

    if args.validate_only:
        print("Validation complete: {}".format(output_dir))
        return
    if not moo_root.is_dir():
        parser.error("MooRetro root does not exist: {}".format(moo_root))
    if not database_path.is_file():
        parser.error("database does not exist: {}".format(database_path))

    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass

    planner_config = PlannerConfig.from_mapping({
        "use_database": True,
        "use_model": True,
        "database_top_k": args.n_db,
        "model_top_k": args.n_model,
        "candidate_beam_width": args.candidate_beam_width,
    })
    database = EnzymeReactionDatabase(str(database_path))
    model_client = EnzRetroClient(
        moo_root=str(moo_root),
        single_fn_type=single_fn_type,
        ec_level=ec_level,
    )

    for split in splits:
        paths = split_paths(args.dataset_root, split)
        split_dir = output_dir / split
        routes_path = split_dir / "routes.csv"
        steps_path = split_dir / "steps.jsonl"
        candidates_path = split_dir / "candidates.jsonl"
        calls_path = split_dir / "tool_calls.jsonl"
        completed = {
            route["route_id"] for route in read_routes_csv(routes_path)
        } if args.resume else set()
        selected = set(selection[split])
        stock_lines = [
            line for line in paths["stock"].read_text(
                encoding="utf-8"
            ).splitlines() if line
        ]
        stock_raw = set(stock_lines)
        split_stock = tuple(
            node_for_smiles(line) for line in stock_lines
        )
        targets = paths["targets"].read_text(encoding="utf-8").splitlines()
        pending = sum(
            route_id_for(split, index) not in completed for index in selected
        )
        progress = 0
        for index, tree in iter_selected_trees(paths["routes"], selected):
            route_id = route_id_for(split, index)
            if route_id in completed:
                continue
            progress += 1
            try:
                reference = extract_reference_route(
                    tree,
                    split=split,
                    source_file=paths["routes"].name,
                    source_route_index=index,
                )
                if index >= len(targets) or reference["target"] != targets[index]:
                    raise ValueError("tree root does not match target line")
                leaf_raw = {
                    item["smiles"] for item in reference["starting_materials"]
                }
                if not leaf_raw.issubset(stock_raw):
                    raise ValueError("route leaves are missing from split stock")
                evaluated = evaluate_route(
                    reference,
                    database=database,
                    model_client=model_client,
                    split_stock=split_stock,
                    building_block_strategy=args.building_block_strategy,
                    config=planner_config,
                    n_db=args.n_db,
                    n_model=args.n_model,
                )
                route = evaluated["route"]
                append_jsonl(steps_path, evaluated["steps"])
                append_jsonl(candidates_path, evaluated["candidates"])
                append_jsonl(calls_path, evaluated["tool_calls"])
            except Exception as exc:
                route = invalid_route_record(
                    split, paths["routes"].name, index,
                    targets[index] if index < len(targets) else "", exc,
                )
            append_route_csv(routes_path, route)
            completed.add(route_id)
            print(
                "[{} {}/{}] {} status={} coverage={}".format(
                    split, progress, pending, route_id,
                    route["status"], route["coverage_text"],
                ),
                flush=True,
            )

        selected_ids = {route_id_for(split, index) for index in selected}
        route_records = [
            route for route in read_routes_csv(routes_path)
            if route["route_id"] in selected_ids
        ]
        step_records = [
            step for step in iter_jsonl(steps_path)
            if step.get("route_id") in selected_ids
        ]
        split_summary = summarize(route_records, step_records)
        split_summary["dataset_split"] = split
        split_summary["tool_call_totals"] = {
            "requests": sum(
                1 for call in iter_jsonl(calls_path)
                if call.get("route_id") in selected_ids
            ),
            "actual_executions": sum(
                call.get("actual_execution") is True
                for call in iter_jsonl(calls_path)
                if call.get("route_id") in selected_ids
            ),
            "llm_api_calls": 0,
        }
        write_json(split_dir / "summary.json", split_summary)

    all_routes: List[Dict[str, Any]] = []
    all_steps: List[Dict[str, Any]] = []
    split_summaries: Dict[str, Any] = {}
    for split in splits:
        split_dir = output_dir / split
        selected_ids = {
            route_id_for(split, index) for index in selection[split]
        }
        all_routes.extend(
            route for route in read_routes_csv(split_dir / "routes.csv")
            if route["route_id"] in selected_ids
        )
        all_steps.extend(
            step for step in iter_jsonl(split_dir / "steps.jsonl")
            if step.get("route_id") in selected_ids
        )
        split_summaries[split] = json.loads(
            (split_dir / "summary.json").read_text(encoding="utf-8")
        )
    run_config["model"]["runtime_device"] = (
        model_client.cache_info().get("runtime_device")
    )
    run_config["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(config_path, run_config)

    overall = summarize(all_routes, all_steps)
    overall.update({
        "splits": split_summaries,
        "combined_record_semantics": (
            "n1 and n5 are reported separately; overall combines selected route "
            "records and does not claim unique targets across splits"
        ),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    write_json(output_dir / "summary.json", overall)
    print("Reference-guided evaluation complete: {}".format(output_dir))


if __name__ == "__main__":
    main()
