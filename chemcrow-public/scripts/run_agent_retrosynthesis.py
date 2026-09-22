#!/usr/bin/env python
"""Run ChemCrow Agent retrosynthesis for multiple targets and a local stock."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chemcrow.agents import ChemCrow  # noqa: E402

DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"
DEFAULT_DEEPSEEK_API_BASE = "https://api.deepseek.com"


def resolve_api_base(model: str, api_base: Optional[str]) -> Optional[str]:
    if api_base:
        return api_base.rstrip("/")
    return DEFAULT_DEEPSEEK_API_BASE if model.startswith("deepseek-") else None


def build_prompt(
    targets: List[str],
    building_blocks: List[str],
    building_block_library: Optional[str],
    endpoint_mode: str,
    top_k: int,
    max_depth: int,
) -> str:
    search_input: Dict[str, Any] = {
        "targets": targets,
        "building_blocks": building_blocks,
        "endpoint_mode": endpoint_mode,
        "top_k": top_k,
        "search_algorithm": "and_or_mcts",
        "config": {
            "use_database": True,
            "use_model": True,
            "max_depth": max_depth,
        },
    }
    if building_block_library:
        search_input["building_block_library"] = str(
            Path(building_block_library).expanduser().resolve()
        )
    return f"""
执行酶促多步逆合成路径预测。

目标分子集合：{json.dumps(targets, ensure_ascii=False)}
直接给出的底物/库存分子：{json.dumps(building_blocks, ensure_ascii=False)}
底物库文件：{building_block_library or "无"}
终止模式：{endpoint_mode}

请先规范化直接给出的分子；不要把整个底物库文件复制进上下文。随后只调用一次
EnzymeRetrosynthesisSearch，让该工具在每个被扩展分子上同时调用本地数据库和
MooRetro 神经单步模型，将两类候选加入同一 AND-OR 图并执行 MCTS。

搜索输入必须保持以下结构和全部目标/底物：
{json.dumps(search_input, ensure_ascii=False)}

输出完整 Top {top_k} 路线。若有多个目标，输出完整 Top {top_k} 路线组合；每个
目标都要给出 enzyme_annotated_retrosynthesis 及全部 AND 反应方程、EC、反应 ID、
模型模板和来源。不得自行补写工具没有返回的酶信息。
""".strip()


def _tool_call_record(step: Any) -> Dict[str, Any]:
    action, observation = step
    return {
        "tool": str(getattr(action, "tool", "")),
        "tool_input": getattr(action, "tool_input", None),
        "observation": observation,
    }


def _observation_object(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return None
    return dict(parsed) if isinstance(parsed, Mapping) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", required=True)
    parser.add_argument("--building-block", action="append", default=[])
    parser.add_argument("--building-block-library", type=Path)
    parser.add_argument(
        "--endpoint-mode", choices=("any_stock", "exact_set"), default="any_stock"
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--max-expansions", type=int, default=5000)
    parser.add_argument("--max-model-calls", type=int, default=100)
    parser.add_argument("--max-agent-iterations", type=int, default=80)
    parser.add_argument(
        "--model", default=os.getenv("CHEMCROW_MODEL", DEFAULT_DEEPSEEK_MODEL)
    )
    parser.add_argument("--tools-model", default=None)
    parser.add_argument(
        "--api-base",
        default=os.getenv("OPENAI_API_BASE") or os.getenv("DEEPSEEK_API_BASE"),
    )
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "result" / "agent_retrosynthesis.json",
    )
    parser.add_argument("--verbose-agent", action="store_true")
    args = parser.parse_args()

    if not args.building_block and args.building_block_library is None:
        parser.error("provide --building-block or --building-block-library")
    if args.building_block_library and not args.building_block_library.is_file():
        parser.error(f"library does not exist: {args.building_block_library}")
    if args.top_k <= 0 or args.max_depth <= 0 or args.max_expansions <= 0:
        parser.error("top-k, max-depth, and max-expansions must be positive")

    key_env = args.api_key_env or (
        "DEEPSEEK_API_KEY" if args.model.startswith("deepseek-") else "OPENAI_API_KEY"
    )
    api_key = os.getenv(key_env)
    if not api_key:
        parser.error(f"missing API key environment variable: {key_env}")
    args.api_base = resolve_api_base(args.model, args.api_base)

    prompt = build_prompt(
        args.target,
        args.building_block,
        str(args.building_block_library) if args.building_block_library else None,
        args.endpoint_mode,
        args.top_k,
        args.max_depth,
    )
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
            "use_database": True,
            "use_model": True,
            "search_algorithm": "and_or_mcts",
            "max_depth": args.max_depth,
            "max_expansions": args.max_expansions,
            "max_model_calls": args.max_model_calls,
            "top_k_routes": args.top_k,
        },
    )
    outputs = agent.agent_executor({"input": prompt})
    tool_calls = [
        _tool_call_record(step) for step in outputs.get("intermediate_steps", ())
    ]
    search_results = [
        parsed
        for call in tool_calls
        if call["tool"] == "EnzymeRetrosynthesisSearch"
        if (parsed := _observation_object(call.get("observation"))) is not None
    ]
    result = {
        "input": {
            "targets": args.target,
            "building_blocks": args.building_block,
            "building_block_library": (
                str(args.building_block_library.resolve())
                if args.building_block_library
                else None
            ),
            "endpoint_mode": args.endpoint_mode,
            "top_k": args.top_k,
        },
        "prompt": prompt,
        "final_answer": outputs.get("output", ""),
        "tool_calls": tool_calls,
        "search_result": search_results[-1] if search_results else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(outputs.get("output", ""))
    print(f"\nStructured result: {args.output.resolve()}")
    if not search_results:
        print("WARNING: Agent did not return a parseable EnzymeRetrosynthesisSearch result.")


if __name__ == "__main__":
    main()
