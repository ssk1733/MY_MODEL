#!/usr/bin/env python
"""Evaluate MooRetro MCTS directly on the RetroBioCat52 v5 benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import re
import resource
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# psutil is used only by MooRetro's optional progress-memory reporting. Some
# released environments omit it, so supply the tiny API surface required by
# mcts_searcher.py rather than blocking model inference.
try:
    import psutil  # noqa: F401
except ModuleNotFoundError:
    class _MemoryInfo:
        @property
        def rss(self) -> int:
            # Linux reports ru_maxrss in KiB.
            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)

    class _Process:
        def memory_info(self) -> _MemoryInfo:
            return _MemoryInfo()

    shim = types.ModuleType("psutil")
    shim.Process = _Process  # type: ignore[attr-defined]
    sys.modules["psutil"] = shim

# MooRetro's active single-objective search path does not call pymoo, although
# legacy classes import its symbols at module load time.
try:
    from pymoo.indicators.hv import HV as _HV  # noqa: F401
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting as _NDS  # noqa: F401
except ModuleNotFoundError:
    class _MissingPymoo:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "pymoo is unavailable and a legacy multi-objective method was called"
            )

    module_names = (
        "pymoo",
        "pymoo.util",
        "pymoo.util.nds",
        "pymoo.util.nds.non_dominated_sorting",
        "pymoo.indicators",
        "pymoo.indicators.hv",
    )
    for name in module_names:
        sys.modules[name] = types.ModuleType(name)
    sys.modules[module_names[3]].NonDominatedSorting = _MissingPymoo
    sys.modules[module_names[5]].HV = _MissingPymoo

from config.config import MCTSConfig, SelectionConfig  # noqa: E402
from searcher.mcts_searcher import MCTSSearcher  # noqa: E402
from src.evaluator import PathEvaluator  # noqa: E402
from src.selector import Selector  # noqa: E402
from src.synthesis_tree import SynthesisPath, SynthesisTree  # noqa: E402
from utils.single_retro_fn import single_fn  # noqa: E402

SCHEMA_VERSION = "retrobiocat52.agent_target_substrates.v5"
DEFAULT_DATASET = ROOT.parent / "dataset" / "retrobiocat52" / "benchmark.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "retrobiocat52_mooretro_v5"
EC_RE = re.compile(r"(?:\[EC\]|\bEC\s*[:=]?\s*)(\d+(?:\.(?:\d+|-)){0,3})", re.I)


class BenchmarkMCTSSearcher(MCTSSearcher):
    """MooRetro MCTS without its heavyweight graph/pickle finalizer."""

    def _finalize_search_results(self) -> None:
        return None


def canonicalize_smiles(value: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(str(value or "").strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonical_components(value: str) -> List[str]:
    output = []
    for component in str(value or "").split("."):
        canonical = canonicalize_smiles(component)
        if canonical is not None:
            output.append(canonical)
    return output


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line {line_number}: {exc}") from exc
    return records


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def read_existing(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    return {item["benchmark_id"]: item for item in load_jsonl(path)}


def normal_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def extract_ec_numbers(*values: Any) -> List[str]:
    output: List[str] = []
    for value in values:
        text = str(value or "")
        for match in EC_RE.finditer(text):
            ec = match.group(1).rstrip(".")
            if ec not in output:
                output.append(ec)
        if text and re.fullmatch(r"\d+(?:\.(?:\d+|-)){0,3}", text.strip()):
            ec = text.strip()
            if ec not in output:
                output.append(ec)
    return output


def enzyme_evidence(
    enzyme_class: str,
    annotation: Mapping[str, Any],
    rules: Mapping[str, Any],
) -> Optional[Dict[str, str]]:
    rule = rules.get(enzyme_class)
    if not isinstance(rule, Mapping):
        return None
    text = " ".join(
        normal_text(annotation.get(field))
        for field in ("enzyme_class", "reaction_name", "template")
        if annotation.get(field)
    )
    for alias in rule.get("aliases", ()):
        normalized = normal_text(alias)
        if normalized and re.search(
            rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])", text
        ):
            return {"method": "text_alias", "value": str(alias)}
    for ec_number in annotation.get("ec_numbers", ()):
        normalized_ec = str(ec_number).replace("EC", "").strip()
        for prefix in rule.get("ec_prefixes", ()):
            prefix = str(prefix)
            if normalized_ec == prefix or normalized_ec.startswith(prefix + "."):
                return {"method": "ec_prefix", "value": str(ec_number)}
    return None


def compare_enzymes(
    gold: Sequence[str],
    annotations: Sequence[Mapping[str, Any]],
    rules: Mapping[str, Any],
) -> Dict[str, Any]:
    adjacency: List[List[Tuple[int, Dict[str, str]]]] = []
    for enzyme_class in gold:
        matches = []
        for step_index, annotation in enumerate(annotations):
            evidence = enzyme_evidence(enzyme_class, annotation, rules)
            if evidence is not None:
                matches.append((step_index, evidence))
        adjacency.append(matches)

    assigned_step: Dict[int, int] = {}

    def assign(gold_index: int, visited: set[int]) -> bool:
        for step_index, _ in adjacency[gold_index]:
            if step_index in visited:
                continue
            visited.add(step_index)
            previous = assigned_step.get(step_index)
            if previous is None or assign(previous, visited):
                assigned_step[step_index] = gold_index
                return True
        return False

    for gold_index in range(len(gold)):
        assign(gold_index, set())
    assigned_gold = {gold_index: step for step, gold_index in assigned_step.items()}
    matches = []
    for gold_index, enzyme_class in enumerate(gold):
        step_index = assigned_gold.get(gold_index)
        evidence = None
        if step_index is not None:
            evidence = next(
                item for index, item in adjacency[gold_index] if index == step_index
            )
        matches.append(
            {
                "gold_index": gold_index,
                "gold_enzyme_class": enzyme_class,
                "matched_step_index": step_index,
                "evidence": evidence,
            }
        )
    return {
        "gold_sequence": list(gold),
        "predicted_step_count": len(annotations),
        "matched_count": len(assigned_gold),
        "all_present": len(assigned_gold) == len(gold),
        "matches": matches,
    }


def orient_path(
    path: SynthesisPath, target_smiles: str
) -> Tuple[List[str], List[Any]]:
    molecules = list(path.molecules)
    reactions = list(path.reactions)
    target = canonicalize_smiles(target_smiles)
    first = canonicalize_smiles(molecules[0]) if molecules else None
    last = canonicalize_smiles(molecules[-1]) if molecules else None
    if target is not None and last == target and first != target:
        molecules.reverse()
        reactions.reverse()
    return molecules, reactions


def path_score(
    evaluator: PathEvaluator,
    molecules: Sequence[str],
    reactions: Sequence[Any],
) -> float:
    try:
        return float(evaluator.evaluate_path(list(molecules), list(reactions), True))
    except Exception:
        scores = [float(getattr(item, "scores", 0.0) or 0.0) for item in reactions]
        return (sum(scores) / len(scores) if scores else 0.0) - 0.01 * len(reactions)


def serialize_path(
    path: SynthesisPath,
    target_smiles: str,
    expected_substrates: Sequence[str],
    gold_enzymes: Sequence[str],
    enzyme_rules: Mapping[str, Any],
    evaluator: PathEvaluator,
    discovery_index: int,
) -> Dict[str, Any]:
    molecules, reactions = orient_path(path, target_smiles)
    canonical_molecules = [canonicalize_smiles(value) or str(value) for value in molecules]
    terminal = molecules[-1] if molecules else ""
    predicted_substrates = sorted(set(canonical_components(terminal)))
    expected = sorted(
        {canonical for value in expected_substrates if (canonical := canonicalize_smiles(value))}
    )
    annotations = []
    steps = []
    for index, reaction in enumerate(reactions):
        template = str(getattr(reaction, "templates", "") or "")
        ec = getattr(reaction, "ec", "")
        ec_numbers = extract_ec_numbers(ec, template)
        annotation = {
            "reaction_name": None,
            "template": template,
            "ec_numbers": ec_numbers,
        }
        annotations.append(annotation)
        raw_reactants = list(getattr(reaction, "reactants", ()) or ())
        precursor_smiles: List[str] = []
        for value in raw_reactants:
            precursor_smiles.extend(canonical_components(str(value)))
        steps.append(
            {
                "step": index + 1,
                "target_smiles": (
                    canonical_molecules[index]
                    if index < len(canonical_molecules)
                    else None
                ),
                "precursor_smiles": precursor_smiles,
                "ec_numbers": ec_numbers,
                "template": template,
                "model_score": float(getattr(reaction, "scores", 0.0) or 0.0),
                "single_step_rank": str(getattr(reaction, "top_k", "") or ""),
            }
        )
    model_complete = bool(getattr(path, "is_complete", False))
    endpoint_match = predicted_substrates == expected
    complete_route_found = model_complete and bool(reactions) and endpoint_match
    enzyme = compare_enzymes(gold_enzymes, annotations, enzyme_rules)
    return {
        "discovery_index": discovery_index,
        "model_complete": model_complete,
        "endpoint_match": endpoint_match,
        "complete_route_found": complete_route_found,
        "enzyme_all_present": complete_route_found and enzyme["all_present"],
        "expected_substrates": expected,
        "predicted_substrates": predicted_substrates,
        "molecules": canonical_molecules,
        "steps": steps,
        "enzyme_comparison": enzyme,
        "route_score": path_score(evaluator, molecules, reactions),
    }


def rank_paths(
    paths: Sequence[SynthesisPath],
    case: Mapping[str, Any],
    enzyme_rules: Mapping[str, Any],
    evaluator: PathEvaluator,
    top_k: int,
) -> List[Dict[str, Any]]:
    serialized = [
        serialize_path(
            path,
            str(case["input"]["target_smiles"]),
            list(case["input"]["substrate_smiles"]),
            list(case["expected"]["enzyme_types"]),
            enzyme_rules,
            evaluator,
            index,
        )
        for index, path in enumerate(paths, 1)
    ]
    serialized.sort(
        key=lambda item: (
            -float(item["route_score"]),
            len(item["steps"]),
            item["discovery_index"],
        )
    )
    unique = []
    seen = set()
    for route in serialized:
        signature = (
            tuple(route["molecules"]),
            tuple(
                (tuple(step["ec_numbers"]), step["template"])
                for step in route["steps"]
            ),
        )
        if signature in seen:
            continue
        seen.add(signature)
        ranked = dict(route)
        ranked["rank"] = len(unique) + 1
        unique.append(ranked)
        if len(unique) >= top_k:
            break
    return unique


def quiet_logger(case_id: str) -> logging.Logger:
    logger = logging.getLogger(f"retrobiocat52.mooretro.{case_id}")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)
    return logger


def run_case(
    case: Mapping[str, Any],
    enzyme_rules: Mapping[str, Any],
    args: argparse.Namespace,
    case_index: int,
    case_output: Path,
) -> Dict[str, Any]:
    target = str(case["input"]["target_smiles"])
    substrates = [str(value) for value in case["input"]["substrate_smiles"]]
    seed = args.seed + case_index
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    config = MCTSConfig(
        project_name=f"retrobiocat52_{case['benchmark_id']}",
        target_molecule=target,
        commercial_molecules=substrates,
        random_seed=seed,
        c_param=args.exploration_constant,
        max_iterations=args.max_iterations,
        max_simulation_depth=args.max_depth,
        real_time_feedback_interval=args.max_iterations + 1,
        epsilon_greedy=args.epsilon_greedy,
        single_fn_type=args.single_fn_type,
        result_path=str(case_output),
    )
    tree = SynthesisTree(target, substrates, logger=quiet_logger(case["benchmark_id"]))
    evaluator = PathEvaluator(substrates)
    selector = Selector(substrates, evaluator, SelectionConfig(), single_objective=True)
    searcher = BenchmarkMCTSSearcher(
        tree=tree,
        single_fn_function=single_fn,
        evaluator=evaluator,
        selector=selector,
        mcts_config=config,
        analyzer=None,
        logger=quiet_logger(case["benchmark_id"]),
    )
    started = time.monotonic()
    searcher.search(max_iterations=args.max_iterations)
    elapsed = time.monotonic() - started
    routes = rank_paths(tree.complete_paths, case, enzyme_rules, evaluator, args.top_k)
    path_ranks = [route["rank"] for route in routes if route["complete_route_found"]]
    enzyme_ranks = [route["rank"] for route in routes if route["enzyme_all_present"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": case["benchmark_id"],
        "ok": True,
        "error": None,
        "input": {"target_smiles": target, "substrate_smiles": substrates},
        "expected_enzyme_types": list(case["expected"]["enzyme_types"]),
        "route_count": len(routes),
        "raw_complete_path_count": len(tree.complete_paths),
        "first_hit_rank": {
            "complete_route_found": min(path_ranks) if path_ranks else None,
            "enzyme_all_present": min(enzyme_ranks) if enzyme_ranks else None,
        },
        "complete_pathway_success": bool(path_ranks),
        "enzyme_complete_pathway_success": bool(enzyme_ranks),
        "routes": routes,
        "runtime": {
            "wall_seconds": round(elapsed, 6),
            "mcts_iterations": searcher.iteration_count,
            "tree_nodes": tree.get_statistics().total_nodes,
            "cuda_peak_allocated_mb": (
                round(torch.cuda.max_memory_allocated() / 1024**2, 2)
                if torch.cuda.is_available()
                else 0.0
            ),
        },
    }


def outcome(success_count: int, total: int) -> Dict[str, Any]:
    return {
        "success_count": success_count,
        "total_count": total,
        "fraction": f"{success_count}/{total}",
        "success_rate": round(success_count / total, 6) if total else 0.0,
    }


def summarize(results: Sequence[Mapping[str, Any]], top_ks: Sequence[int]) -> Dict[str, Any]:
    total = len(results)
    path_ranks = [item.get("first_hit_rank", {}).get("complete_route_found") for item in results]
    enzyme_ranks = [item.get("first_hit_rank", {}).get("enzyme_all_present") for item in results]
    top_ks = sorted(set(top_ks))
    path_hits = sum(rank is not None for rank in path_ranks)
    enzyme_hits = sum(rank is not None for rank in enzyme_ranks)
    elapsed = [float(item.get("runtime", {}).get("wall_seconds", 0.0)) for item in results]
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluator": "MooRetro direct MCTS (no Agent, no ChemCrow tools)",
        "case_count": total,
        "error_count": sum(not item.get("ok") for item in results),
        "complete_pathway_success": outcome(path_hits, total),
        "enzyme_complete_pathway_success": outcome(enzyme_hits, total),
        "enzyme_top_k": {
            str(k): outcome(sum(rank is not None and rank <= k for rank in enzyme_ranks), total)
            for k in top_ks
        },
        "pathway_top_k": {
            str(k): outcome(sum(rank is not None and rank <= k for rank in path_ranks), total)
            for k in top_ks
        },
        "runtime": {
            "total_seconds": round(sum(elapsed), 6),
            "mean_seconds": round(sum(elapsed) / len(elapsed), 6) if elapsed else None,
        },
    }


def format_report(summary: Mapping[str, Any], top_ks: Sequence[int], top_k: int) -> str:
    pathway = summary["complete_pathway_success"]
    enzyme = summary["enzyme_complete_pathway_success"]
    lines = [
        "RetroBioCat52 MooRetro 多步模型评测（不使用 Agent/工具）",
        f"找到完整逆合成路径（Top {top_k} 内）: {pathway['fraction']} ({pathway['success_rate'] * 100:.2f}%)",
        f"包含全部测试集酶的完整路径: {enzyme['fraction']} ({enzyme['success_rate'] * 100:.2f}%)",
        "完整路径准确率:",
    ]
    for k in top_ks:
        value = summary["pathway_top_k"][str(k)]
        lines.append(f"  Top {k}: {value['fraction']} ({value['success_rate'] * 100:.2f}%)")
    lines.append(
        "酶预测准确率:",
    )
    for k in top_ks:
        value = summary["enzyme_top_k"][str(k)]
        lines.append(f"  Top {k}: {value['fraction']} ({value['success_rate'] * 100:.2f}%)")
    return "\n".join(lines) + "\n"


def validate_dataset(cases: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> None:
    if len(cases) != 52:
        raise ValueError(f"Expected 52 benchmark cases, found {len(cases)}")
    expected_ids = [f"RBC52-{index:03d}" for index in range(1, 53)]
    if [case.get("benchmark_id") for case in cases] != expected_ids:
        raise ValueError("Benchmark IDs must be ordered RBC52-001 through RBC52-052")
    for case in cases:
        if case.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{case.get('benchmark_id')}: expected schema {SCHEMA_VERSION}")
        if not case.get("input", {}).get("target_smiles"):
            raise ValueError(f"{case.get('benchmark_id')}: missing target_smiles")
        if not case.get("input", {}).get("substrate_smiles"):
            raise ValueError(f"{case.get('benchmark_id')}: missing substrate_smiles")
        if not case.get("expected", {}).get("enzyme_types"):
            raise ValueError(f"{case.get('benchmark_id')}: missing enzyme_types")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Manifest schema does not match the benchmark")
    if not manifest.get("enzyme_class_rules"):
        raise ValueError("Manifest has no enzyme_class_rules")


def parse_top_ks(value: str) -> List[int]:
    try:
        result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("top-k must be comma-separated integers") from exc
    if not result or result[0] <= 0:
        raise argparse.ArgumentTypeError("top-k values must be positive")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--single-fn-type", choices=("biocyc", "ecreact", "model"), default="biocyc")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--report-top-k", type=parse_top_ks, default=parse_top_ks("1,3,10"))
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--exploration-constant", type=float, default=1.4)
    parser.add_argument("--epsilon-greedy", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.single_fn_type == "model":
        args.single_fn_type = "biocyc"
    if args.top_k < max(args.report_top_k):
        raise SystemExit("--top-k must be at least the largest --report-top-k")
    if args.max_iterations <= 0 or args.max_depth <= 0:
        raise SystemExit("--max-iterations and --max-depth must be positive")
    dataset = args.dataset.resolve()
    manifest_path = dataset.with_name("manifest.json")
    if not dataset.is_file() or not manifest_path.is_file():
        raise SystemExit(f"Dataset or manifest not found: {dataset}, {manifest_path}")
    cases = load_jsonl(dataset)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_dataset(cases, manifest)
    if args.validate_only:
        print(json.dumps({"ok": True, "case_count": len(cases), "schema_version": SCHEMA_VERSION, "dataset": str(dataset)}, ensure_ascii=False, indent=2))
        return
    if not args.allow_cpu and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; use --allow-cpu only if CPU inference is intentional")
    if args.case_id:
        requested = set(args.case_id)
        cases = [case for case in cases if case["benchmark_id"] in requested]
        missing = requested - {case["benchmark_id"] for case in cases}
        if missing:
            raise SystemExit(f"Unknown case IDs: {sorted(missing)}")
    cases = cases[args.offset:]
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        raise SystemExit("No benchmark cases selected")

    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    result_path = output_dir / "case_results.jsonl"
    summary_path = output_dir / "summary.json"
    summary_text_path = output_dir / "summary.txt"
    config_path = output_dir / "run_config.json"
    run_config = {
        "protocol": "retrobiocat52-mooretro-direct-v1",
        "dataset": str(dataset),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "selected_case_ids": [case["benchmark_id"] for case in cases],
        "single_fn_type": args.single_fn_type,
        "backend_mode": "neural_model",
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "top_k": args.top_k,
        "report_top_k": args.report_top_k,
        "max_iterations": args.max_iterations,
        "max_depth": args.max_depth,
        "exploration_constant": args.exploration_constant,
        "epsilon_greedy": args.epsilon_greedy,
        "seed": args.seed,
    }
    if args.overwrite:
        for path in (result_path, summary_path, summary_text_path, config_path):
            if path.exists():
                path.unlink()
    elif config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != run_config:
            raise SystemExit("Output directory has another configuration; use --overwrite")
    config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    existing = read_existing(result_path) if args.resume else {}
    pending = [case for case in cases if case["benchmark_id"] not in existing]

    for position, case in enumerate(pending, 1):
        case_id = case["benchmark_id"]
        case_output = raw_dir / case_id
        case_output.mkdir(exist_ok=True)
        try:
            result = run_case(case, manifest["enzyme_class_rules"], args, int(case_id[-3:]), case_output)
        except Exception as exc:
            result = {
                "schema_version": SCHEMA_VERSION,
                "benchmark_id": case_id,
                "ok": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "route_count": 0,
                "first_hit_rank": {"complete_route_found": None, "enzyme_all_present": None},
                "complete_pathway_success": False,
                "enzyme_complete_pathway_success": False,
                "routes": [],
                "runtime": {},
            }
        (case_output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        append_jsonl(result_path, result)
        existing[case_id] = result
        print(f"[{position}/{len(pending)}] {case_id}: routes={result['route_count']} path_success={result['complete_pathway_success']} enzyme_success={result['enzyme_complete_pathway_success']}", flush=True)

    selected = [existing[case["benchmark_id"]] for case in cases if case["benchmark_id"] in existing]
    summary = summarize(selected, args.report_top_k)
    summary["run_config"] = run_config
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = format_report(summary, args.report_top_k, args.top_k)
    summary_text_path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Case results: {result_path}")
    print(f"Summary: {summary_path}")
    print(f"Text summary: {summary_text_path}")


if __name__ == "__main__":
    main()
