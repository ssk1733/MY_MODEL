#!/usr/bin/env python
"""Evaluate ChemCrow Agent routes from target and substrate SMILES."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chemcrow.agents import ChemCrow  # noqa: E402
from chemcrow.benchmarks.retrobiocat52 import (  # noqa: E402
    SCHEMA_VERSION,
    evaluate_case_result,
    load_jsonl,
    summarize_case_results,
)
from chemcrow.planner.chem_utils import canonicalize_smiles  # noqa: E402


def _top_ks(value: str) -> List[int]:
    try:
        values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("top-k must be comma-separated integers") from exc
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("top-k values must be positive")
    return values


def _read_existing(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    return {item["benchmark_id"]: item for item in load_jsonl(path)}


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def build_agent_prompt(
    target_smiles: str,
    substrate_smiles: List[str],
    top_k: int,
    max_depth: int,
    max_expansions: int = 5000,
    max_model_calls: int = 100,
) -> str:
    substrate_json = json.dumps(substrate_smiles, ensure_ascii=False)
    normalization_json = json.dumps(
        {"targets": [target_smiles], "building_blocks": substrate_smiles},
        ensure_ascii=False,
    )
    action_json = json.dumps(
        {
            "targets": [target_smiles],
            "building_blocks": substrate_smiles,
            "endpoint_mode": "exact_set",
            "top_k": top_k,
            "search_algorithm": "and_or_mcts",
            "config": {
                "use_database": True,
                "use_model": True,
                "max_depth": max_depth,
                "max_expansions": max_expansions,
                "max_model_calls": max_model_calls,
            },
        },
        ensure_ascii=False,
    )
    return f"""
这是一次端点受约束的酶催化多步逆合成路线测试。
目标物集合 A（本条数据为单元素集合）：["{target_smiles}"]
已知底物集合 D：{substrate_json}

任务不是自由预测底物，而是寻找从 A 逆合成到整个 D 的完整路线。
请由 Agent 调用项目工具完成以下任务：
1. 先调用 MoleculeInputNormalizer 规范化目标物和 D 中的每一个底物，
   调用 JSON 为：{normalization_json}
2. 规范化成功后只调用一次 EnzymeRetrosynthesisSearch 完成图搜索；必须把规范化后的目标集合
   原样作为 targets，并把完整的 D 原样作为 building_blocks。
3. 调用 JSON 应与下列结构一致：{action_json}
4. 必须设置 endpoint_mode=exact_set，并在每个扩展节点同时使用本地数据库
   和 MooRetro 神经模型，先建立 MODEL/DATABASE 来源分支，再分别扩展其反应 AND
   节点并执行 MCTS。有效路线必须以 A 为根、
   连通、无环，而且所有非辅因子叶节点集合必须与 D 完全相等；不得遗漏、替换或增加底物。
5. 只接受 route_validation.valid=true 的路线，并保留工具返回的完整 Top {top_k}
   routes、steps、所有 AND 分支、反应方程与酶信息，不得猜测未知酶。

最终答案逐条输出全部 Top-K 路线。每条路线先输出
`enzyme_annotated_retrosynthesis`，再输出其全部
`retrosynthesis_reactions[].equation`，并附上完整 JSON。
""".strip()


def _tool_call_record(step: Any) -> Dict[str, Any]:
    action, observation = step
    return {
        "tool": str(getattr(action, "tool", "")),
        "tool_input": getattr(action, "tool_input", None),
        "observation": observation,
    }


def _tool_input_object(raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw, Mapping):
        return dict(raw)
    text = str(raw or "").strip()
    if not text.startswith("{"):
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _canonical_set(values: Any) -> set[str]:
    if isinstance(values, str):
        values = [values]
    output = set()
    for value in values or ():
        canonical = canonicalize_smiles(str(value))
        if canonical is None:
            raise ValueError(f"Invalid molecule in Agent search input: {value}")
        output.add(canonical)
    return output


def _molecule_smiles(value: Any) -> Optional[str]:
    if isinstance(value, Mapping):
        value = value.get("canonical_smiles") or value.get("smiles") or value.get("key")
    return canonicalize_smiles(str(value or ""))


def _routes_from_latest_result(
    value: Mapping[str, Any], expected_target: str
) -> List[Mapping[str, Any]]:
    """Extract one target's routes from current single- or multi-target output."""
    raw_routes = value.get("routes")
    if isinstance(raw_routes, list) and (
        not raw_routes
        or all(isinstance(route, Mapping) and "steps" in route for route in raw_routes)
    ):
        return raw_routes

    target_results = value.get("target_results")
    if isinstance(target_results, list):
        for target_result in target_results:
            if not isinstance(target_result, Mapping):
                continue
            if _molecule_smiles(target_result.get("target")) != expected_target:
                continue
            routes = target_result.get("routes")
            if isinstance(routes, list):
                return routes

    bundles = value.get("route_bundles", raw_routes)
    if not isinstance(bundles, list):
        raise ValueError(
            "EnzymeRetrosynthesisSearch result must contain routes, target_results, "
            "or route_bundles"
        )
    extracted: List[Mapping[str, Any]] = []
    for bundle_rank, bundle in enumerate(bundles, start=1):
        if not isinstance(bundle, Mapping):
            continue
        for target_route in bundle.get("target_routes") or ():
            if not isinstance(target_route, Mapping):
                continue
            if _molecule_smiles(target_route.get("target")) != expected_target:
                continue
            route = target_route.get("route")
            if isinstance(route, Mapping):
                normalized = dict(route)
                normalized["rank"] = int(bundle.get("rank") or bundle_rank)
                extracted.append(normalized)
                break
    return extracted


def _normalize_prediction(
    value: Mapping[str, Any], top_k: int, expected_target: str
) -> Dict[str, Any]:
    raw_routes = _routes_from_latest_result(value, expected_target)
    routes = []
    for rank, raw_route in enumerate(raw_routes[:top_k], start=1):
        if not isinstance(raw_route, Mapping):
            raise ValueError(f"Route {rank} must be a JSON object")
        route = dict(raw_route)
        route["rank"] = rank
        routes.append(route)
    return {
        "ok": bool(value.get("ok")),
        "mode": value.get("mode", "single-target"),
        "route_count": len(routes),
        "routes": routes,
        "stats": value.get("stats") or {},
        "endpoint_task": value.get("endpoint_task") or {},
        "candidate_generation": value.get("candidate_generation") or {},
        "search_algorithm": value.get("search_algorithm") or {},
    }


class AgentCaseError(ValueError):
    """Agent protocol failure that preserves every attempted trace."""

    def __init__(self, message: str, raw: Mapping[str, Any]):
        super().__init__(message)
        self.raw = dict(raw)


def _strict_endpoint_requested(payload: Mapping[str, Any]) -> bool:
    config = payload.get("config")
    nested = config if isinstance(config, Mapping) else {}
    endpoint_mode = str(payload.get("endpoint_mode") or "").strip().casefold()
    return (
        endpoint_mode in {"exact_set", "exact", "all", "required"}
        or payload.get("require_all_building_blocks") is True
        or nested.get("require_all_building_blocks") is True
    )


def _hybrid_sources_requested(payload: Mapping[str, Any]) -> bool:
    config = payload.get("config")
    nested = config if isinstance(config, Mapping) else {}
    database = payload.get("use_database", nested.get("use_database"))
    model = payload.get("use_model", nested.get("use_model"))
    return database is True and model is True


def _canonical_targets(payload: Mapping[str, Any]) -> List[str]:
    values = payload.get("targets")
    if values is None:
        values = payload.get("target") or payload.get("target_smiles")
    if isinstance(values, (str, Mapping)):
        values = [values]
    output = []
    for value in values or ():
        if isinstance(value, Mapping):
            value = value.get("canonical_smiles") or value.get("smiles") or value.get("name")
        canonical = canonicalize_smiles(str(value or ""))
        if canonical is None:
            raise ValueError(f"Invalid target in Agent search input: {value}")
        if canonical not in output:
            output.append(canonical)
    return output


def _normalization_matches(
    calls: List[Mapping[str, Any]], expected_target: str, expected_blocks: set[str]
) -> bool:
    for call in calls:
        if call.get("tool") != "MoleculeInputNormalizer":
            continue
        observation = call.get("observation")
        try:
            result = (
                dict(observation)
                if isinstance(observation, Mapping)
                else json.loads(str(observation))
            )
            normalized_input = result.get("normalized_input") or {}
            targets = _canonical_targets(normalized_input)
            blocks = _canonical_set(normalized_input.get("building_blocks", []))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if result.get("ok") and targets == [expected_target] and blocks == expected_blocks:
            return True
    return False


def _latest_agent_input(case: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "targets": [str(case["input"]["target_smiles"])],
        "building_blocks": list(case["input"]["substrate_smiles"]),
        "endpoint_mode": "exact_set",
    }


def run_agent_case(
    agent: ChemCrow,
    case: Mapping[str, Any],
    *,
    top_k: int,
    max_depth: int,
    max_expansions: int = 5000,
    max_model_calls: int = 100,
    agent_attempts: int = 2,
) -> Dict[str, Any]:
    target = str(case["input"]["target_smiles"])
    substrates = list(case["input"]["substrate_smiles"])
    base_prompt = build_agent_prompt(
        target, substrates, top_k, max_depth, max_expansions, max_model_calls
    )
    expected_target = canonicalize_smiles(target)
    expected_substrates = _canonical_set(substrates)
    attempts: List[Dict[str, Any]] = []
    all_tool_calls: List[Dict[str, Any]] = []
    last_problem = "Agent did not produce a valid endpoint-constrained search call"

    for attempt_index in range(1, agent_attempts + 1):
        prompt = base_prompt
        if attempt_index > 1:
            prompt += f"""

上一次尝试不符合测试协议：{last_problem}
请纠正后重新调用工具。必须使用完整且精确的 targets、building_blocks，
并明确设置 endpoint_mode=exact_set、use_database=true、use_model=true；
不要只在文字中描述路线。
"""
        outputs = agent.agent_executor({"input": prompt})
        tool_calls = [
            _tool_call_record(step)
            for step in outputs.get("intermediate_steps", ())
        ]
        all_tool_calls.extend(tool_calls)
        attempt_record = {
            "attempt": attempt_index,
            "prompt": prompt,
            "final_answer": outputs.get("output", ""),
            "tool_calls": tool_calls,
        }
        attempts.append(attempt_record)

        valid_search_calls = []
        valid_normalization_calls = []
        for call_index, call in enumerate(tool_calls):
            if call.get("tool") != "EnzymeRetrosynthesisSearch":
                continue
            payload = _tool_input_object(call.get("tool_input"))
            if payload is None:
                continue
            try:
                supplied_targets = _canonical_targets(payload)
                supplied_blocks = _canonical_set(
                    payload.get("building_blocks", payload.get("substrates", []))
                )
            except ValueError:
                continue
            normalization_ok = _normalization_matches(
                tool_calls[:call_index], expected_target, expected_substrates
            )
            if (
                normalization_ok
                and supplied_targets == [expected_target]
                and supplied_blocks == expected_substrates
                and _strict_endpoint_requested(payload)
                and _hybrid_sources_requested(payload)
            ):
                valid_search_calls.append(call)
                valid_normalization_calls.extend(
                    previous
                    for previous in tool_calls[:call_index]
                    if previous.get("tool") == "MoleculeInputNormalizer"
                )
        if not valid_search_calls:
            last_problem = (
                "Agent did not normalize the exact targets/building_blocks before "
                "calling EnzymeRetrosynthesisSearch with endpoint_mode=exact_set "
                "and both use_database=true and use_model=true"
            )
            continue

        observation = valid_search_calls[-1].get("observation")
        try:
            tool_result = (
                dict(observation)
                if isinstance(observation, Mapping)
                else json.loads(str(observation))
            )
        except (TypeError, json.JSONDecodeError) as exc:
            last_problem = f"EnzymeRetrosynthesisSearch returned invalid JSON: {exc}"
            continue
        if not tool_result.get("ok"):
            last_problem = (
                f"EnzymeRetrosynthesisSearch failed: {tool_result.get('error')}"
            )
            continue

        prediction = _normalize_prediction(tool_result, top_k, expected_target)
        prediction["stats"] = {
            **dict(prediction.get("stats") or {}),
            "agent_attempt_count": attempt_index,
            "agent_tool_call_count": len(all_tool_calls),
            "agent_tool_names": [item["tool"] for item in all_tool_calls],
            "valid_normalization_call_count": len(valid_normalization_calls),
            "valid_search_call_count": len(valid_search_calls),
        }
        return {
            "prompt": base_prompt,
            "attempts": attempts,
            "final_answer": outputs.get("output", ""),
            "tool_calls": all_tool_calls,
            "search_result": tool_result,
            "prediction": prediction,
        }

    failure_raw = {
        "prompt": base_prompt,
        "attempts": attempts,
        "tool_calls": all_tool_calls,
    }
    raise AgentCaseError(last_problem, failure_raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT.parent / "dataset" / "retrobiocat52" / "benchmark.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "benchmark_results" / "retrobiocat52_agent_v8",
    )
    parser.add_argument("--model", default=os.getenv("CHEMCROW_MODEL", "deepseek-flash"))
    parser.add_argument("--tools-model", default=None)
    parser.add_argument(
        "--api-base",
        default=None,
        help=(
            "explicit LLM API base; otherwise use DEEPSEEK_API_BASE for DeepSeek "
            "or OPENAI_API_BASE for OpenAI"
        ),
    )
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--report-top-k", type=_top_ks, default=_top_ks("1,3,10"))
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--max-expansions", type=int, default=5000)
    parser.add_argument("--max-model-calls", type=int, default=100)
    parser.add_argument("--max-agent-iterations", type=int, default=80)
    parser.add_argument("--agent-attempts", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--verbose-agent", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.top_k < max(args.report_top_k):
        parser.error("--top-k must be at least the largest --report-top-k value")
    if (
        args.max_depth <= 0
        or args.max_expansions <= 0
        or args.max_model_calls <= 0
        or args.max_agent_iterations <= 0
        or args.agent_attempts <= 0
    ):
        parser.error(
            "max-depth, max-expansions, max-model-calls, max-agent-iterations, "
            "and agent-attempts must be positive"
        )
    if not args.dataset.is_file():
        parser.error(f"dataset does not exist: {args.dataset}")

    cases = load_jsonl(args.dataset)
    if any(
        case.get("schema_version") != SCHEMA_VERSION
        or "target_smiles" not in case.get("input", {})
        or "substrate_smiles" not in case.get("input", {})
        or "enzyme_types" not in case.get("expected", {})
        for case in cases
    ):
        parser.error("dataset is not the target-plus-substrates v5 format; rebuild it first")
    if args.case_id:
        requested = set(args.case_id)
        cases = [case for case in cases if case["benchmark_id"] in requested]
        missing = requested - {case["benchmark_id"] for case in cases}
        if missing:
            parser.error(f"unknown case IDs: {sorted(missing)}")
    cases = cases[args.offset:]
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        parser.error("no benchmark cases selected")

    key_env = args.api_key_env or (
        "DEEPSEEK_API_KEY" if args.model.startswith("deepseek-") else "OPENAI_API_KEY"
    )
    api_key = os.getenv(key_env)
    if not api_key:
        parser.error(f"missing API key environment variable: {key_env}")
    if args.api_base:
        args.api_base = args.api_base.rstrip("/")
    elif args.model.startswith("deepseek-"):
        args.api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com").rstrip("/")
    else:
        openai_base = os.getenv("OPENAI_API_BASE")
        args.api_base = openai_base.rstrip("/") if openai_base else None

    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    result_path = output_dir / "case_results.jsonl"
    summary_path = output_dir / "summary.json"
    summary_text_path = output_dir / "summary.txt"
    config_path = output_dir / "run_config.json"
    top_k_routes_path = output_dir / "top_k_routes.jsonl"

    run_config = {
        "protocol": "targets-plus-building-blocks-agent-v8-source-partitioned-exact-set",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "selected_case_ids": [case["benchmark_id"] for case in cases],
        "dataset_input_fields": [
            "input.target_smiles",
            "input.substrate_smiles",
        ],
        "agent_tool_input": {
            "targets": "[input.target_smiles]",
            "building_blocks": "input.substrate_smiles",
            "endpoint_mode": "exact_set",
            "search_algorithm": "and_or_mcts",
            "use_database": True,
            "use_model": True,
        },
        "gold_fields_never_sent_to_agent": ["expected.enzyme_types"],
        "model": args.model,
        "tools_model": args.tools_model or args.model,
        "api_base": args.api_base,
        "top_k": args.top_k,
        "report_top_k": args.report_top_k,
        "max_depth": args.max_depth,
        "max_expansions": args.max_expansions,
        "max_model_calls": args.max_model_calls,
        "max_agent_iterations": args.max_agent_iterations,
        "agent_attempts": args.agent_attempts,
        "route_validity": "rooted_connected_acyclic_exact_terminal_set",
    }
    if args.overwrite:
        for path in (
            result_path, summary_path, summary_text_path, config_path,
            top_k_routes_path,
        ):
            if path.exists():
                path.unlink()
    elif config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != run_config:
            parser.error(
                "output-dir has a different run config; use a new directory or --overwrite"
            )
    elif result_path.exists() and not args.resume:
        parser.error("case_results.jsonl exists; use --resume or --overwrite")

    config_path.write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    existing = _read_existing(result_path) if args.resume else {}
    pending = [case for case in cases if case["benchmark_id"] not in existing]

    agent: Optional[ChemCrow] = None
    if pending:
        agent = ChemCrow(
            model=args.model,
            tools_model=args.tools_model or args.model,
            temp=0.0,
            max_iterations=args.max_agent_iterations,
            verbose=args.verbose_agent,
            streaming=False,
            openai_api_key=api_key,
            api_base=args.api_base,
            save_transcripts=False,
            enzyme_planner_config={
                "max_depth": args.max_depth,
                "top_k_routes": args.top_k,
                "use_database": True,
                "use_model": True,
                "search_algorithm": "and_or_mcts",
                "require_all_building_blocks": True,
                "max_expansions": args.max_expansions,
                "max_model_calls": args.max_model_calls,
            },
        )

    for index, case in enumerate(pending, start=1):
        case_id = case["benchmark_id"]
        started = time.monotonic()
        try:
            raw = run_agent_case(
                agent,
                case,
                top_k=args.top_k,
                max_depth=args.max_depth,
                max_expansions=args.max_expansions,
                max_model_calls=args.max_model_calls,
                agent_attempts=args.agent_attempts,  # type: ignore[arg-type]
            )
            prediction = raw["prediction"]
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            prediction = {"ok": False, "routes": [], "route_count": 0, "error": error}
            raw = dict(getattr(exc, "raw", {}) or {})
            raw.update({"error": error, "prediction": prediction})
        wall_seconds = time.monotonic() - started
        (raw_dir / f"{case_id}.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        evaluated = evaluate_case_result(case, prediction)
        evaluated["wall_seconds"] = round(wall_seconds, 6)
        evaluated["dataset_input"] = dict(case["input"])
        evaluated["agent_input"] = _latest_agent_input(case)
        evaluated["complete_pathway_success"] = (
            evaluated["first_hit_rank"]["complete_route_found"] is not None
        )
        evaluated["enzyme_complete_pathway_success"] = (
            evaluated["first_hit_rank"]["enzyme_all_present"] is not None
        )
        _append_jsonl(result_path, evaluated)
        existing[case_id] = evaluated
        valid_route_count = sum(
            item["metrics"]["complete_route_found"]
            for item in evaluated["route_evaluations"]
        )
        print(
            f"[{index}/{len(pending)}] {case_id}: candidates={evaluated['route_count']} "
            f"valid_routes={valid_route_count} "
            f"path_success={evaluated['complete_pathway_success']} " 
            f"enzyme_success={evaluated['enzyme_complete_pathway_success']} " 
            f"time={wall_seconds:.1f}s",
            flush=True,
        )

    selected_results = [
        existing[case["benchmark_id"]]
        for case in cases
        if case["benchmark_id"] in existing
    ]
    with top_k_routes_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            case_id = case["benchmark_id"]
            raw_path = raw_dir / f"{case_id}.json"
            raw = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else {}
            prediction = raw.get("prediction") or {}
            value = {
                "benchmark_id": case_id,
                "dataset_input": dict(case["input"]),
                "agent_input": _latest_agent_input(case),
                "mode": prediction.get("mode"),
                "endpoint_task": prediction.get("endpoint_task") or {},
                "candidate_generation": prediction.get("candidate_generation") or {},
                "search_algorithm": prediction.get("search_algorithm") or {},
                "route_count": len(prediction.get("routes") or ()),
                "routes": list(prediction.get("routes") or ()),
            }
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")

    summary = summarize_case_results(selected_results, args.report_top_k)
    summary["evaluated_route_limit"] = args.top_k
    summary["run_config"] = run_config
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pathway = summary["complete_pathway_success"]
    enzyme = summary["enzyme_complete_pathway_success"]
    report_lines = [
        "RetroBioCat52 Agent v8 评测（targets + exact_set + DB/MooRetro AND-OR MCTS）",
        (
            f"找到完整逆合成路径（Top {args.top_k} 内）: "
            f"{pathway['fraction']} ({pathway['success_rate'] * 100:.2f}%)"
        ),
        (
            "包含全部测试集酶的完整路径: "
            f"{enzyme['fraction']} ({enzyme['success_rate'] * 100:.2f}%)"
        ),
        "完整路径准确率:",
    ]
    for top_k in args.report_top_k:
        value = summary["pathway_top_k"][str(top_k)]
        report_lines.append(
            f"  Top {top_k}: {value['fraction']} "
            f"({value['success_rate'] * 100:.2f}%)"
        )
    report_lines.append("酶预测准确率:")
    for top_k in args.report_top_k:
        value = summary["enzyme_top_k"][str(top_k)]
        report_lines.append(
            f"  Top {top_k}: {value['fraction']} "
            f"({value['success_rate'] * 100:.2f}%)"
        )
    report_text = "\n".join(report_lines) + "\n"
    summary_text_path.write_text(report_text, encoding="utf-8")
    print(report_text, end="")
    print(f"Case results: {result_path}")
    print(f"Summary: {summary_path}")
    print(f"Text summary: {summary_text_path}")
    print(f"Complete Top-K routes: {top_k_routes_path}")


if __name__ == "__main__":
    main()
