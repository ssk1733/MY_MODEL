#!/usr/bin/env python
"""Teacher-forced PaRoutes n1/n5 one-step ranking without an LLM API.

For each hidden reference route, start at its target and repeatedly generate the
same database + MooRetro one-step candidates used by ChemCrow. A step succeeds
when one ranked candidate contains every canonical reference precursor. Record
its rank and continue regardless of rank; stop the case at the first miss.
Prediction input is only targets.txt plus the complete split stock.txt. The
routes JSON is hidden ground truth used only for teacher forcing and scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chemcrow.planner import EnzRetroClient, EnzymeReactionDatabase, PlannerConfig  # noqa: E402
from chemcrow.planner.chem_utils import canonicalize_smiles, molecule_key  # noqa: E402
from chemcrow.planner.models import MoleculeNode, ReactionCandidate  # noqa: E402
from chemcrow.planner.scoring import fuse_candidates, score_and_rank_candidates  # noqa: E402
from chemcrow.tools.normalization import MoleculeInputNormalizer  # noqa: E402

DEFAULT_SPLITS = ("n1", "n5")
PROTOCOL = "paroutes-tools-v2-teacher-forced-one-step-ranking"


def parse_top_ks(value: str) -> List[int]:
    try:
        values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("top-k must be comma-separated integers") from exc
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("top-k values must be positive")
    return values


def timestamp_output_dir(root: Path, started_at: Optional[datetime] = None) -> Path:
    value = started_at or datetime.now().astimezone()
    return root.joinpath(*(value.strftime(part) for part in ("%Y", "%m", "%d", "%H", "%M")))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield value


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Dict[str, Any]]:
    """Stream a top-level JSON array without retaining all route trees in RAM."""
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
                    raise ValueError(f"Empty JSON file: {path}")
                if position >= len(buffer):
                    continue
                if buffer[position] != "[":
                    raise ValueError(f"Expected a JSON array in {path}")
                started, position = True, position + 1
                continue
            while True:
                while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
                    position += 1
                if position < len(buffer):
                    break
                if eof:
                    raise ValueError(f"Unterminated JSON array in {path}")
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
                raise ValueError(f"PaRoutes array item must be an object in {path}")
            position = end
            yield value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    result = canonicalize_smiles(str(value or ""))
    if result is None:
        raise ValueError(f"Invalid PaRoutes SMILES: {value!r}")
    return result


def signature_record(target: str, precursors: Sequence[str]) -> Dict[str, Any]:
    target = canonical(target)
    precursors = sorted(canonical(item) for item in precursors)
    return {
        "target_smiles": target,
        "precursor_smiles": precursors,
        "retrosynthesis": f"{target}>>{'.'.join(precursors)}",
    }


def extract_reference(tree: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract hidden reference reactions in deterministic preorder."""
    if tree.get("type") != "mol":
        raise ValueError("PaRoutes root must be a molecule node")
    root, leaves, steps = canonical(tree.get("smiles")), [], []

    def visit(node: Mapping[str, Any]) -> None:
        if node.get("type") != "mol":
            raise ValueError("Expected a molecule node in reference tree")
        product = canonical(node.get("smiles"))
        children = list(node.get("children") or ())
        if not children:
            leaves.append(product)
            return
        if len(children) != 1 or children[0].get("type") != "reaction":
            raise ValueError("Reference molecule must have exactly one reaction child")
        reaction = children[0]
        precursor_nodes = list(reaction.get("children") or ())
        if not precursor_nodes or any(item.get("type") != "mol" for item in precursor_nodes):
            raise ValueError("Reference reaction must contain molecule precursors")
        precursors = [canonical(item.get("smiles")) for item in precursor_nodes]
        metadata = dict(reaction.get("metadata") or {})
        steps.append({
            "step_index": len(steps) + 1,
            **signature_record(product, precursors),
            "reaction_hash": metadata.get("reaction_hash"),
            "reaction_id": metadata.get("ID"),
            "reaction_smiles": metadata.get("smiles"),
        })
        for precursor in precursor_nodes:
            visit(precursor)

    visit(tree)
    if not steps or not leaves:
        raise ValueError("Reference must contain reactions and stock leaves")
    return {
        "target_smiles": root,
        "substrate_smiles": sorted(set(leaves)),
        "steps": steps,
        "step_count": len(steps),
    }


def stock_nodes(values: Sequence[str]) -> Tuple[MoleculeNode, ...]:
    nodes, seen = [], set()
    for raw in values:
        smiles = canonical(raw)
        if smiles in seen:
            continue
        seen.add(smiles)
        nodes.append(MoleculeNode(
            key=molecule_key(smiles, None), query=raw, smiles=smiles,
            canonical_smiles=smiles,
        ))
    return tuple(nodes)


def candidate_allowed(candidate: ReactionCandidate, config: PlannerConfig) -> Tuple[bool, Optional[str]]:
    """Apply HybridPlannerEngine's state-independent candidate filters."""
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
    elif candidate.source in {"MODEL", "BOTH"} and candidate.model_score is not None and candidate.model_score < config.min_model_score:
        reason = "low_model_score"
    elif candidate.feasibility is not None and candidate.feasibility < config.min_feasibility:
        reason = "low_feasibility"
    elif candidate.atom_conservation is not None and candidate.atom_conservation < config.min_atom_conservation:
        reason = "low_atom_conservation"
    elif candidate.approx_structural_overlap is not None and candidate.approx_structural_overlap < config.min_approx_structural_overlap:
        reason = "low_approx_structural_overlap"
    elif candidate.toxicity is not None and candidate.toxicity > config.max_toxicity:
        reason = "toxicity_threshold"
    elif candidate.step_score is not None and candidate.step_score < config.min_candidate_score:
        reason = "low_candidate_score"
    return reason is None, reason


def candidate_record(candidate: ReactionCandidate, rank: int) -> Dict[str, Any]:
    return {
        "rank": rank,
        "candidate_id": candidate.candidate_id,
        "source": candidate.source,
        "target_smiles": candidate.target.canonical_smiles or candidate.target.smiles,
        "precursor_smiles": sorted(canonical(item.canonical_smiles or item.smiles) for item in candidate.precursors),
        "noncurrency_precursor_smiles": sorted(canonical(item.canonical_smiles or item.smiles) for item in candidate.noncurrency_precursors),
        "ec_numbers": list(candidate.ec_numbers),
        "reaction_name": candidate.reaction_name,
        "database_reaction_ids": list(candidate.database_reaction_ids),
        "template": candidate.template,
        "step_score": candidate.step_score,
        "step_cost": candidate.step_cost,
        "model_score": candidate.model_score,
        "structural_similarity": candidate.structural_similarity,
        "building_block_progress": candidate.building_block_progress,
    }


def generate_ranked_candidates(
    target_smiles: str,
    database: EnzymeReactionDatabase,
    model_client: EnzRetroClient,
    stock: Sequence[MoleculeNode],
    config: PlannerConfig,
    candidate_limit: int,
 ) -> Dict[str, Any]:
    """Use current DB/model generation, fusion, scoring and filtering logic."""
    target = database.normalize_molecule(target_smiles)
    database_candidates = database.backward_candidates(
        target, limit=config.database_top_k,
        respect_direction=config.respect_reaction_direction,
        allow_unknown_direction=config.allow_unknown_direction,
        currency_ids=config.currency_metabolite_ids,
    )
    model_candidates: Tuple[ReactionCandidate, ...] = ()
    warnings = []
    before = model_client.cache_info()
    try:
        model_candidates = model_client.predict_candidates(
            target, stock, top_k=config.model_top_k,
            currency_ids=config.currency_metabolite_ids,
            normalizer=database.normalize_molecule,
        )
    except Exception as exc:
        warnings.append(str(exc))
    after = model_client.cache_info()
    fused = fuse_candidates(database_candidates, model_candidates, config)
    scored = score_and_rank_candidates(fused, stock, config)
    accepted, reasons = [], {}
    for candidate in scored:
        allowed, reason = candidate_allowed(candidate, config)
        if allowed:
            accepted.append(candidate)
        else:
            reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    ranked = tuple(accepted[:candidate_limit])
    return {
        "target": target.to_dict(),
        "ranked_candidates": ranked,
        "database_candidate_count": len(database_candidates),
        "model_candidate_count": len(model_candidates),
        "fused_candidate_count": len(fused),
        "returned_candidate_count": len(ranked),
        "filtered_candidate_count": len(scored) - len(accepted),
        "filter_reasons": reasons,
        "warnings": warnings,
        "model_cache_delta": {
            "hits": max(0, after["hits"] - before["hits"]),
            "misses": max(0, after["misses"] - before["misses"]),
            "backend_mode": after["backend_mode"],
            "runtime_device": after["runtime_device"],
        },
    }


def candidate_contains_reference(candidate: ReactionCandidate, expected: Sequence[str]) -> bool:
    expected_set = {canonical(value) for value in expected}
    predicted_set = {canonical(item.canonical_smiles or item.smiles) for item in candidate.precursors}
    return expected_set.issubset(predicted_set)


def parse_tool_json(raw: Any, tool_name: str) -> Dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    try:
        value = json.loads(str(raw))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{tool_name} returned invalid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{tool_name} must return a JSON object")
    return dict(value)


def run_teacher_forced_case(
    normalizer: Any,
    database: EnzymeReactionDatabase,
    model_client: EnzRetroClient,
    stock: Sequence[MoleculeNode],
    reference: Mapping[str, Any],
    config: PlannerConfig,
    *,
    report_top_ks: Sequence[int],
    candidate_limit: int,
    saved_candidate_count: int,
 ) -> Dict[str, Any]:
    """Predict the reference prefix and stop immediately after the first miss."""
    normalized = parse_tool_json(normalizer.run(json.dumps({
        "targets": [reference["target_smiles"]], "allow_external_lookup": False,
    })), "MoleculeInputNormalizer")
    if not normalized.get("ok"):
        raise ValueError(f"target normalization failed: {normalized.get('error')}")
    if (normalized.get("normalized_input") or {}).get("targets") != [reference["target_smiles"]]:
        raise ValueError("normalization changed the canonical target")

    results, successful, stopped_at = [], 0, None
    source_counts = {"DATABASE": 0, "MODEL": 0, "BOTH": 0}
    generator_counts = {"DATABASE": 0, "MODEL": 0, "FUSED": 0}
    warnings = []
    steps = list(reference["steps"])
    for position, step in enumerate(steps, 1):
        if stopped_at is not None:
            results.append({
                "step_index": position, "reference": step, "attempted": False,
                "success": False, "hit_rank": None,
                "top_k_hits": {str(k): False for k in report_top_ks},
                "status": "not_attempted_after_previous_miss",
            })
            continue
        generated = generate_ranked_candidates(
            step["target_smiles"], database, model_client, stock, config,
            candidate_limit,
        )
        ranked = generated.pop("ranked_candidates")
        for candidate in ranked:
            source_counts[candidate.source] = source_counts.get(candidate.source, 0) + 1
        generator_counts["DATABASE"] += generated["database_candidate_count"]
        generator_counts["MODEL"] += generated["model_candidate_count"]
        generator_counts["FUSED"] += generated["fused_candidate_count"]
        warnings.extend(generated["warnings"])
        hit_rank = next((rank for rank, candidate in enumerate(ranked, 1)
                         if candidate_contains_reference(candidate, step["precursor_smiles"])), None)
        success = hit_rank is not None
        if success:
            successful += 1
        else:
            stopped_at = position
        results.append({
            "step_index": position,
            "reference": step,
            "attempted": True,
            "success": success,
            "hit_rank": hit_rank,
            "top_k_hits": {str(k): hit_rank is not None and hit_rank <= k for k in report_top_ks},
            "status": "matched_and_continue" if success else "missed_and_stop",
            "matched_candidate": candidate_record(ranked[hit_rank - 1], hit_rank) if hit_rank else None,
            "candidate_generation": generated,
            "ranked_candidates": [candidate_record(candidate, rank) for rank, candidate in enumerate(ranked[:saved_candidate_count], 1)],
        })
    total = len(steps)
    return {
        "initial_normalization": normalized,
        "completed_single_steps": successful,
        "total_single_steps": total,
        "single_step_fraction": f"{successful}/{total}",
        "single_step_success_rate": round(successful / total, 6),
        "complete_multistep_prediction": successful == total,
        "stopped_at_step": stopped_at,
        "attempted_step_count": sum(bool(step["attempted"]) for step in results),
        "top_k_hit_counts": {str(k): sum(bool(step["top_k_hits"][str(k)]) for step in results) for k in report_top_ks},
        "step_results": results,
        "candidate_source_counts": source_counts,
        "generator_candidate_counts": generator_counts,
        "model_warnings": list(dict.fromkeys(warnings)),
        "model_cache": model_client.cache_info(),
    }


def selected_indices(split: str, count: int, offset: int, limit: Optional[int], case_ids: Sequence[str]) -> List[int]:
    indices = list(range(offset, count))
    if limit is not None:
        indices = indices[:limit]
    if case_ids:
        requested = set(case_ids)
        indices = [index for index in indices if f"{split}-{index + 1:05d}" in requested]
    return indices


def iter_split_references(dataset_root: Path, split: str, selected: set[int]) -> Iterator[Tuple[int, Dict[str, Any]]]:
    targets = (dataset_root / f"{split}-targets.txt").read_text(encoding="utf-8").splitlines()
    seen = 0
    for index, tree in enumerate(iter_json_array(dataset_root / f"{split}-routes.json")):
        seen += 1
        if index >= len(targets):
            raise ValueError(f"{split}: more routes than targets")
        if index not in selected:
            continue
        reference = extract_reference(tree)
        if reference["target_smiles"] != canonical(targets[index]):
            raise ValueError(f"{split}-{index + 1:05d}: tree root/target mismatch")
        yield index, reference
    if seen != len(targets):
        raise ValueError(f"{split}: route count {seen} != target count {len(targets)}")


def outcome(success: int, total: int) -> Dict[str, Any]:
    return {"success_count": success, "total_count": total, "fraction": f"{success}/{total}",
            "success_rate": round(success / total, 6) if total else 0.0}


def summarize_records(records: Iterable[Mapping[str, Any]], top_ks: Sequence[int]) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = {"overall": []}
    for record in records:
        groups.setdefault(str(record["split"]), []).append(record)
        groups["overall"].append(record)
    output: Dict[str, Any] = {"splits": {}}
    for name, values in groups.items():
        cases = len(values)
        total_steps = sum(int(item.get("total_single_steps", 0)) for item in values)
        successful_steps = sum(int(item.get("completed_single_steps", 0)) for item in values)
        attempted = sum(int(item.get("attempted_step_count", 0)) for item in values)
        runtimes = [float(item["wall_seconds"]) for item in values
                    if item.get("wall_seconds") is not None and math.isfinite(float(item["wall_seconds"]))]
        output["splits"][name] = {
            "case_count": cases,
            "successful_cases": sum(bool(item.get("ok")) for item in values),
            "error_count": sum(not bool(item.get("ok")) for item in values),
            "complete_multistep_paths": outcome(sum(bool(item.get("complete_multistep_prediction")) for item in values), cases),
            "single_steps_successful_at_any_returned_rank": outcome(successful_steps, total_steps),
            "attempted_single_steps": attempted,
            "unattempted_after_miss": max(0, total_steps - attempted),
            "single_step_top_k_accuracy": {
                str(k): outcome(sum(int((item.get("top_k_hit_counts") or {}).get(str(k), 0)) for item in values), total_steps)
                for k in top_ks
            },
            "runtime": {
                "total_seconds": round(sum(runtimes), 6),
                "mean_seconds": round(statistics.fmean(runtimes), 6) if runtimes else None,
                "median_seconds": round(statistics.median(runtimes), 6) if runtimes else None,
            },
        }
    output["overall"] = output["splits"].pop("overall")
    return output


def build_report(summary: Mapping[str, Any], top_ks: Sequence[int]) -> str:
    lines = [
        "PaRoutes 逐步工具直调评测（无 LLM/API）",
        "输入：targets.txt 目标物 + 对应 split 的完整 stock.txt",
        "流程：沿隐藏参考路线逐步预测；命中则继续，首次未命中立即停止",
        "单步命中：同一候选反应包含全部参考前体；排名为融合候选反应排名",
    ]
    groups = dict(summary.get("splits") or {})
    groups["overall"] = summary["overall"]
    for name, result in groups.items():
        complete = result["complete_multistep_paths"]
        any_rank = result["single_steps_successful_at_any_returned_rank"]
        lines += ["", name,
                  f"  完整多步路径: {complete['fraction']} ({complete['success_rate'] * 100:.2f}%)",
                  f"  成功单步（任意返回排名）: {any_rank['fraction']} ({any_rank['success_rate'] * 100:.2f}%)",
                  f"  实际尝试单步数: {result['attempted_single_steps']}",
                  f"  前序失败后未再尝试: {result['unattempted_after_miss']}",
                  "  单步 Top-K 准确率:"]
        for k in top_ks:
            metric = result["single_step_top_k_accuracy"][str(k)]
            lines.append(f"    Top {k}: {metric['fraction']} ({metric['success_rate'] * 100:.2f}%)")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT.parent / "dataset" / "PaRoutes")
    parser.add_argument("--split", action="append", choices=DEFAULT_SPLITS,
                        help="repeat to select splits; default: n1 and n5")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "PaRoutes_results")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--db-path", type=Path)
    parser.add_argument("--moo-root", type=Path, default=PROJECT_ROOT.parent / "MooRetro-master")
    parser.add_argument("--single-fn-type", choices=("biocyc", "ecreact"), default="biocyc")
    parser.add_argument("--ec-level", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument("--report-top-k", type=parse_top_ks, default=parse_top_ks("1,3,10"))
    parser.add_argument("--database-top-k", type=int, default=100)
    parser.add_argument("--model-top-k", type=int, default=50)
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--saved-candidate-count", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0, help="applied per split")
    parser.add_argument("--limit", type=int, help="maximum cases per split")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    parser, args = build_parser(), None
    args = parser.parse_args()
    splits = tuple(dict.fromkeys(args.split or DEFAULT_SPLITS))
    if not args.dataset_root.is_dir():
        parser.error(f"dataset root does not exist: {args.dataset_root}")
    if not args.moo_root.is_dir():
        parser.error(f"MooRetro root does not exist: {args.moo_root}")
    if min(args.database_top_k, args.model_top_k, args.candidate_limit, args.saved_candidate_count) <= 0:
        parser.error("candidate limits must be positive")
    if args.candidate_limit < max(args.report_top_k):
        parser.error("candidate-limit must be at least the largest report-top-k")
    if args.offset < 0 or (args.limit is not None and args.limit <= 0):
        parser.error("offset must be non-negative and limit must be positive")

    dataset_files, target_counts, selection = {}, {}, {}
    for split in splits:
        paths = {name: args.dataset_root / f"{split}-{suffix}" for name, suffix in
                 (("routes", "routes.json"), ("targets", "targets.txt"), ("stock", "stock.txt"))}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            parser.error("missing PaRoutes files: " + ", ".join(missing))
        dataset_files[split] = paths
        count = len(paths["targets"].read_text(encoding="utf-8").splitlines())
        target_counts[split] = count
        selection[split] = selected_indices(split, count, args.offset, args.limit, args.case_id)
    selected_ids = {f"{split}-{index + 1:05d}" for split, indices in selection.items() for index in indices}
    unknown = set(args.case_id) - selected_ids
    if unknown:
        parser.error("unknown or excluded case IDs: " + ", ".join(sorted(unknown)))
    if not selected_ids:
        parser.error("no PaRoutes cases selected")

    started_at = datetime.now().astimezone()
    output_dir = args.output_dir.resolve() if args.output_dir else timestamp_output_dir(args.output_root.resolve(), started_at)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path, results_path = output_dir / "run_config.json", output_dir / "case_results.jsonl"
    summary_path, text_path = output_dir / "summary.json", output_dir / "summary.txt"
    planner_config = PlannerConfig.from_mapping({
        "use_database": True, "use_model": True,
        "database_top_k": args.database_top_k, "model_top_k": args.model_top_k,
        "candidate_beam_width": args.candidate_limit,
    })
    run_config = {
        "protocol": PROTOCOL,
        "execution_mode": "manual_agent_simulation_without_llm_or_api",
        "started_at": started_at.isoformat(), "output_directory": str(output_dir),
        "dataset_root": str(args.dataset_root.resolve()), "splits": list(splits),
        "dataset_files": {split: {name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
                                  for name, path in paths.items()} for split, paths in dataset_files.items()},
        "target_counts": target_counts,
        "selected_counts": {split: len(indices) for split, indices in selection.items()},
        "selection": {"offset_per_split": args.offset, "limit_per_split": args.limit,
                      "case_ids": list(args.case_id)},
        "prediction_input": "target from targets.txt plus complete split stock.txt",
        "ground_truth_not_sent_to_predictor": "routes.json reaction tree",
        "evaluation": {
            "mode": "teacher_forced_reference_step_prefix",
            "step_success": "one fused candidate contains every canonical reference precursor; extra precursors allowed",
            "continuation": "continue after hit regardless of rank; stop case at first miss",
            "report_top_k": args.report_top_k,
            "top_k_denominator": "all reference steps in all selected cases",
        },
        "candidate_generation": {
            "sources": ["MODEL", "DATABASE"], "fusion": "fuse_candidates",
            "ranking": "score_and_rank_candidates", "filtering": "planner state-independent filters",
            "database_top_k": args.database_top_k, "model_top_k": args.model_top_k,
            "candidate_limit": args.candidate_limit, "saved_candidate_count": args.saved_candidate_count,
        },
        "moo_root": str(args.moo_root.resolve()), "single_fn_type": args.single_fn_type,
        "ec_level": args.ec_level, "db_path": str(args.db_path.resolve()) if args.db_path else None,
    }
    if config_path.exists():
        if not args.resume:
            parser.error(f"output directory already contains a run: {output_dir}")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        ignored = {"started_at", "output_directory"}
        if {k: v for k, v in previous.items() if k not in ignored} != {k: v for k, v in run_config.items() if k not in ignored}:
            parser.error("output directory has an incompatible run; choose another --output-dir")
        run_config = previous
    else:
        write_json(config_path, run_config)

    completed_ids = ({str(record["case_id"]) for record in iter_jsonl(results_path)} if args.resume else set())
    database = EnzymeReactionDatabase(str(args.db_path) if args.db_path else None)
    model_client = EnzRetroClient(moo_root=str(args.moo_root), single_fn_type=args.single_fn_type,
                                  ec_level=args.ec_level)
    normalizer = MoleculeInputNormalizer(database_service=database, allow_external_lookup=False)

    for split in splits:
        selected = set(selection[split])
        if not selected:
            continue
        raw_stock = [line for line in dataset_files[split]["stock"].read_text(encoding="utf-8").splitlines() if line.strip()]
        split_stock = stock_nodes(raw_stock)
        stock_keys = {item.canonical_smiles for item in split_stock}
        pending_total = sum(f"{split}-{index + 1:05d}" not in completed_ids for index in selected)
        pending_index = 0
        for index, reference in iter_split_references(args.dataset_root, split, selected):
            case_id = f"{split}-{index + 1:05d}"
            if case_id in completed_ids:
                continue
            pending_index += 1
            started = time.monotonic()
            try:
                missing = sorted(set(reference["substrate_smiles"]) - stock_keys)
                if missing:
                    raise ValueError("hidden reference leaves missing from split stock: " + ", ".join(missing))
                result = run_teacher_forced_case(
                    normalizer, database, model_client, split_stock, reference,
                    planner_config, report_top_ks=args.report_top_k,
                    candidate_limit=args.candidate_limit,
                    saved_candidate_count=args.saved_candidate_count,
                )
                record = {
                    "protocol": PROTOCOL, "case_id": case_id, "split": split,
                    "split_index": index + 1, "ok": True, "error": None,
                    "input": {"target_smiles": reference["target_smiles"],
                              "stock_file": str(dataset_files[split]["stock"].resolve()),
                              "stock_size": len(split_stock)},
                    "hidden_reference": {"substrate_smiles": reference["substrate_smiles"],
                                         "steps": reference["steps"]},
                    **result,
                }
            except Exception as exc:
                total = int(reference["step_count"])
                record = {
                    "protocol": PROTOCOL, "case_id": case_id, "split": split,
                    "split_index": index + 1, "ok": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "input": {"target_smiles": reference["target_smiles"],
                              "stock_file": str(dataset_files[split]["stock"].resolve()),
                              "stock_size": len(split_stock)},
                    "hidden_reference": {"substrate_smiles": reference["substrate_smiles"],
                                         "steps": reference["steps"]},
                    "completed_single_steps": 0, "total_single_steps": total,
                    "single_step_fraction": f"0/{total}", "single_step_success_rate": 0.0,
                    "complete_multistep_prediction": False, "stopped_at_step": 1,
                    "attempted_step_count": 0,
                    "top_k_hit_counts": {str(k): 0 for k in args.report_top_k},
                    "step_results": [],
                }
            record["wall_seconds"] = round(time.monotonic() - started, 6)
            append_jsonl(results_path, record)
            completed_ids.add(case_id)
            print(f"[{split} {pending_index}/{pending_total}] {case_id}: "
                  f"steps={record['single_step_fraction']} "
                  f"complete={record['complete_multistep_prediction']} "
                  f"stop={record['stopped_at_step']} time={record['wall_seconds']:.1f}s", flush=True)

    records = [record for record in iter_jsonl(results_path) if record.get("case_id") in selected_ids]
    summary = summarize_records(records, args.report_top_k)
    summary.update({"protocol": PROTOCOL, "run_config": run_config,
                    "output_directory": str(output_dir)})
    write_json(summary_path, summary)
    report = build_report(summary, args.report_top_k)
    text_path.write_text(report, encoding="utf-8")
    print("\n" + report, end="")
    print(f"结果目录: {output_dir}")
    print(f"逐组逐步结果: {results_path}")
    print(f"JSON 汇总: {summary_path}")
    print(f"文本汇总: {text_path}")


if __name__ == "__main__":
    main()
