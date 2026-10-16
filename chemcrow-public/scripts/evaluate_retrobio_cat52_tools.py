#!/usr/bin/env python
"""Evaluate RetroBioCat52 by directly calling the tools an Agent would use.

This benchmark deliberately does not construct ChemCrow or call an LLM API.  It
simulates the deterministic part of the Agent protocol for every case:

1. MoleculeInputNormalizer(target + known substrates)
2. EnzymeRetrosynthesisSearch(exact_set, database + neural model, AND-OR MCTS)

The default output directory is benchmark_results/YYYY/MM/DD/HH/MM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chemcrow.benchmarks.retrobiocat52 import (  # noqa: E402
    SCHEMA_VERSION,
    evaluate_case_result,
    load_jsonl,
    summarize_case_results,
)
from chemcrow.planner import EnzRetroClient, EnzymeReactionDatabase  # noqa: E402
from chemcrow.tools.normalization import MoleculeInputNormalizer  # noqa: E402
from chemcrow.tools.retrosynthesis_workflow import (  # noqa: E402
    EnzymeRetrosynthesisSearch,
)


def parse_top_ks(value: str) -> List[int]:
    try:
        values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("top-k must be comma-separated integers") from exc
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("top-k values must be positive")
    return values


def timestamp_output_dir(root: Path, started_at: Optional[datetime] = None) -> Path:
    """Return ROOT/YYYY/MM/DD/HH/MM using the host's local timezone."""
    value = started_at or datetime.now().astimezone()
    return root.joinpath(
        value.strftime("%Y"),
        value.strftime("%m"),
        value.strftime("%d"),
        value.strftime("%H"),
        value.strftime("%M"),
    )


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def read_existing(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    return {item["benchmark_id"]: item for item in load_jsonl(path)}


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


def run_tool_case(
    normalizer: Any,
    search_tool: Any,
    case: Mapping[str, Any],
    *,
    top_k: int,
    max_depth: int,
    max_expansions: int,
    max_model_calls: int,
) -> Dict[str, Any]:
    """Simulate the Agent's normalization and exact endpoint search calls."""
    target = str(case["input"]["target_smiles"])
    substrates = [str(value) for value in case["input"]["substrate_smiles"]]
    normalization_input = {
        "targets": [target],
        "building_blocks": substrates,
        "allow_external_lookup": False,
    }
    normalization_result = parse_tool_json(
        normalizer.run(json.dumps(normalization_input, ensure_ascii=False)),
        "MoleculeInputNormalizer",
    )
    if not normalization_result.get("ok"):
        raise ValueError(
            "MoleculeInputNormalizer failed: "
            + json.dumps(normalization_result.get("error"), ensure_ascii=False)
        )
    normalized = normalization_result.get("normalized_input") or {}
    targets = list(normalized.get("targets") or ())
    building_blocks = list(normalized.get("building_blocks") or ())
    if len(targets) != 1 or len(building_blocks) != len(substrates):
        raise ValueError("normalization did not preserve one target and all substrates")

    search_input = {
        "targets": targets,
        "building_blocks": building_blocks,
        "endpoint_mode": "exact_set",
        "top_k": top_k,
        "search_algorithm": "and_or_mcts",
        "config": {
            "use_database": True,
            "use_model": True,
            "require_all_building_blocks": True,
            "max_depth": max_depth,
            "max_expansions": max_expansions,
            "max_model_calls": max_model_calls,
            "top_k_routes": top_k,
        },
    }
    search_result = parse_tool_json(
        search_tool.run(json.dumps(search_input, ensure_ascii=False)),
        "EnzymeRetrosynthesisSearch",
    )
    if not search_result.get("ok"):
        raise ValueError(
            "EnzymeRetrosynthesisSearch failed: "
            + json.dumps(search_result.get("error"), ensure_ascii=False)
        )
    backend = (search_result.get("candidate_generation") or {}).get("model_backend")
    if backend != "neural_model":
        raise RuntimeError(
            f"expected MooRetro neural_model backend, received {backend!r}"
        )

    # The benchmark evaluator consumes only the retained Top-K routes, but the
    # complete tool response is kept in raw/ for later diagnosis.
    prediction = {
        "ok": True,
        "route_count": len(search_result.get("routes") or ()),
        "routes": list(search_result.get("routes") or ())[:top_k],
        "stats": dict(search_result.get("stats") or {}),
        "endpoint_task": dict(search_result.get("endpoint_task") or {}),
        "candidate_generation": dict(
            search_result.get("candidate_generation") or {}
        ),
        "search_algorithm": dict(search_result.get("search_algorithm") or {}),
    }
    return {
        "execution_mode": "manual_agent_simulation_without_llm",
        "tool_calls": [
            {
                "tool": "MoleculeInputNormalizer",
                "tool_input": normalization_input,
                "observation": normalization_result,
            },
            {
                "tool": "EnzymeRetrosynthesisSearch",
                "tool_input": search_input,
                "observation": search_result,
            },
        ],
        "normalization_result": normalization_result,
        "search_result": search_result,
        "prediction": prediction,
    }


def selected_cases(cases: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    output = list(cases)
    if args.case_id:
        requested = set(args.case_id)
        output = [case for case in output if case["benchmark_id"] in requested]
        missing = requested - {case["benchmark_id"] for case in output}
        if missing:
            raise ValueError(f"unknown case IDs: {sorted(missing)}")
    output = output[args.offset :]
    if args.limit is not None:
        output = output[: args.limit]
    return output


def build_report(summary: Mapping[str, Any], top_ks: Sequence[int]) -> str:
    lines = [
        "RetroBioCat52 工具直调评测（无 LLM/API）",
        "输入：目标物 SMILES + 已知底物 SMILES 集合",
        "搜索：本地数据库 + MooRetro 神经模型 + 分源 AND-OR MCTS",
        "判定：完整路线的非辅因子叶节点集合必须与输入底物集合完全相等",
        "",
        f"评测条数: {summary['case_count']}",
        f"工具调用成功: {summary['successful_tool_calls']}/{summary['case_count']}",
        "完整路径成功率:",
    ]
    for top_k in top_ks:
        value = summary["pathway_top_k"][str(top_k)]
        lines.append(
            f"  Top {top_k}: {value['fraction']} "
            f"({value['success_rate'] * 100:.2f}%)"
        )
    lines.append("酶预测成功率（同一条完整路线包含全部参考酶类别）:")
    for top_k in top_ks:
        value = summary["enzyme_top_k"][str(top_k)]
        lines.append(
            f"  Top {top_k}: {value['fraction']} "
            f"({value['success_rate'] * 100:.2f}%)"
        )
    runtime = summary.get("runtime") or {}
    lines.extend(
        [
            "",
            f"总耗时: {runtime.get('total_seconds')} s",
            f"平均每条: {runtime.get('mean_seconds')} s",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT.parent / "dataset" / "retrobiocat52" / "benchmark.jsonl",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "benchmark_results",
        help="timestamp hierarchy is created below this directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="explicit run directory; overrides --output-root/YYYY/MM/DD/HH/MM",
    )
    parser.add_argument("--db-path", type=Path)
    parser.add_argument(
        "--moo-root",
        type=Path,
        default=PROJECT_ROOT.parent / "MooRetro-master",
    )
    parser.add_argument(
        "--single-fn-type",
        choices=("biocyc", "ecreact"),
        default="biocyc",
        help="neural checkpoint family; XLSX lookup modes are intentionally disabled",
    )
    parser.add_argument("--ec-level", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--report-top-k", type=parse_top_ks, default=parse_top_ks("1,3,10"))
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--max-expansions", type=int, default=5000)
    parser.add_argument("--max-model-calls", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume compatible results if the minute directory already exists",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.dataset.is_file():
        parser.error(f"dataset does not exist: {args.dataset}")
    if not args.moo_root.is_dir():
        parser.error(f"MooRetro directory does not exist: {args.moo_root}")
    if args.top_k < max(args.report_top_k):
        parser.error("--top-k must be at least the largest --report-top-k value")
    if min(args.top_k, args.max_depth, args.max_expansions, args.max_model_calls) <= 0:
        parser.error("search budgets and top-k must be positive")
    if args.offset < 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--offset must be non-negative and --limit must be positive")

    cases = load_jsonl(args.dataset)
    if any(
        case.get("schema_version") != SCHEMA_VERSION
        or "target_smiles" not in case.get("input", {})
        or "substrate_smiles" not in case.get("input", {})
        or "enzyme_types" not in case.get("expected", {})
        for case in cases
    ):
        parser.error("dataset is not the RetroBioCat52 target-plus-substrates format")
    try:
        cases = selected_cases(cases, args)
    except ValueError as exc:
        parser.error(str(exc))
    if not cases:
        parser.error("no benchmark cases selected")

    started_at = datetime.now().astimezone()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else timestamp_output_dir(args.output_root.resolve(), started_at)
    )
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    result_path = output_dir / "case_results.jsonl"
    summary_path = output_dir / "summary.json"
    summary_text_path = output_dir / "summary.txt"
    config_path = output_dir / "run_config.json"
    routes_path = output_dir / "top_k_routes.jsonl"

    run_config = {
        "protocol": "retrobiocat52-tools-v1-source-partitioned-exact-set",
        "execution_mode": "manual_agent_simulation_without_llm_or_api",
        "started_at": started_at.isoformat(),
        "output_directory": str(output_dir),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "selected_case_ids": [case["benchmark_id"] for case in cases],
        "tool_sequence": ["MoleculeInputNormalizer", "EnzymeRetrosynthesisSearch"],
        "search_input": {
            "endpoint_mode": "exact_set",
            "search_algorithm": "and_or_mcts",
            "use_database": True,
            "use_model": True,
            "model_backend_required": "neural_model",
        },
        "moo_root": str(args.moo_root.resolve()),
        "single_fn_type": args.single_fn_type,
        "ec_level": args.ec_level,
        "db_path": str(args.db_path.resolve()) if args.db_path else None,
        "top_k": args.top_k,
        "report_top_k": args.report_top_k,
        "max_depth": args.max_depth,
        "max_expansions": args.max_expansions,
        "max_model_calls": args.max_model_calls,
        "gold_fields_not_sent_to_tools": ["expected.enzyme_types", "reference.doi"],
        "route_validity": "rooted_connected_acyclic_exact_terminal_set",
    }
    if config_path.exists():
        if not args.resume:
            parser.error(f"output directory already contains a run: {output_dir}")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        comparable_keys = (
            "protocol", "dataset_sha256", "selected_case_ids", "single_fn_type",
            "ec_level", "db_path", "top_k", "report_top_k", "max_depth",
            "max_expansions", "max_model_calls",
        )
        if any(previous.get(key) != run_config.get(key) for key in comparable_keys):
            parser.error(
                "timestamp directory has an incompatible run; use --output-dir "
                "to choose a different directory"
            )
        run_config = previous
    else:
        write_json(config_path, run_config)

    database = EnzymeReactionDatabase(str(args.db_path) if args.db_path else None)
    model_client = EnzRetroClient(
        moo_root=str(args.moo_root),
        single_fn_type=args.single_fn_type,
        ec_level=args.ec_level,
    )
    normalizer = MoleculeInputNormalizer(
        database_service=database,
        allow_external_lookup=False,
    )
    search_tool = EnzymeRetrosynthesisSearch(
        database_service=database,
        model_client=model_client,
        config={
            "use_database": True,
            "use_model": True,
            "search_algorithm": "and_or_mcts",
            "require_all_building_blocks": True,
            "top_k_routes": args.top_k,
            "max_depth": args.max_depth,
            "max_expansions": args.max_expansions,
            "max_model_calls": args.max_model_calls,
        },
    )

    existing = read_existing(result_path) if args.resume else {}
    pending = [case for case in cases if case["benchmark_id"] not in existing]
    for index, case in enumerate(pending, start=1):
        case_id = case["benchmark_id"]
        started = time.monotonic()
        try:
            raw = run_tool_case(
                normalizer,
                search_tool,
                case,
                top_k=args.top_k,
                max_depth=args.max_depth,
                max_expansions=args.max_expansions,
                max_model_calls=args.max_model_calls,
            )
            prediction = raw["prediction"]
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            prediction = {
                "ok": False,
                "route_count": 0,
                "routes": [],
                "error": error,
            }
            raw = {
                "execution_mode": "manual_agent_simulation_without_llm",
                "error": error,
                "prediction": prediction,
            }
        wall_seconds = time.monotonic() - started
        write_json(raw_dir / f"{case_id}.json", raw)
        evaluated = evaluate_case_result(case, prediction)
        evaluated["wall_seconds"] = round(wall_seconds, 6)
        evaluated["dataset_input"] = dict(case["input"])
        evaluated["tool_input"] = {
            "targets": [case["input"]["target_smiles"]],
            "building_blocks": list(case["input"]["substrate_smiles"]),
            "endpoint_mode": "exact_set",
        }
        append_jsonl(result_path, evaluated)
        existing[case_id] = evaluated
        top_limit = max(args.report_top_k)
        path_rank = evaluated["first_hit_rank"]["complete_route_found"]
        enzyme_rank = evaluated["first_hit_rank"]["enzyme_all_present"]
        print(
            f"[{index}/{len(pending)}] {case_id}: "
            f"routes={evaluated['route_count']} "
            f"path_top{top_limit}={path_rank is not None and path_rank <= top_limit} "
            f"enzyme_top{top_limit}={enzyme_rank is not None and enzyme_rank <= top_limit} "
            f"time={wall_seconds:.1f}s",
            flush=True,
        )

    results = [existing[case["benchmark_id"]] for case in cases]
    with routes_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            case_id = case["benchmark_id"]
            raw_path = raw_dir / f"{case_id}.json"
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            prediction = raw.get("prediction") or {}
            record = {
                "benchmark_id": case_id,
                "input": dict(case["input"]),
                "route_count": len(prediction.get("routes") or ()),
                "routes": list(prediction.get("routes") or ()),
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    summary = summarize_case_results(results, args.report_top_k)
    summary["run_config"] = run_config
    summary["output_directory"] = str(output_dir)
    write_json(summary_path, summary)
    report = build_report(summary, args.report_top_k)
    summary_text_path.write_text(report, encoding="utf-8")
    print("\n" + report, end="")
    print(f"结果目录: {output_dir}")
    print(f"逐条结果: {result_path}")
    print(f"汇总结果: {summary_path}")
    print(f"完整路线: {routes_path}")


if __name__ == "__main__":
    main()
