"""One-call LangChain tools for deterministic multi-step enzyme planning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from langchain.pydantic_v1 import PrivateAttr
from langchain.tools import BaseTool

from chemcrow.planner import (
    EnzRetroClient,
    EnzymeReactionDatabase,
    HybridPlannerEngine,
    PlannerConfig,
)

__all__ = ["EnzymeRetroPathFinder", "HybridEnzymeRetroPlanner"]

_DEFAULT_DB_PATH = str(
    Path(__file__).resolve().parents[1] / "data" / "enzyme_retro.db"
)


def _coerce_scalar(value: str) -> Any:
    text = value.strip()
    if text.casefold() in {"true", "false"}:
        return text.casefold() == "true"
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _parse_planner_input(raw_input: str) -> Tuple[str, List[str], Dict[str, Any]]:
    """Parse JSON first, with a small legacy ``key=value;`` compatibility mode."""
    text = str(raw_input or "").strip()
    if not text:
        raise ValueError("Planner input cannot be empty.")
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid planner JSON: {exc.msg}.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Planner JSON input must be an object.")
    else:
        payload: Dict[str, Any] = {}
        for part in text.split(";"):
            if not part.strip():
                continue
            key, separator, value = part.partition("=")
            if not separator:
                if "target" not in payload:
                    payload["target"] = part.strip()
                    continue
                raise ValueError(
                    "Legacy planner input must use key=value pairs separated by ';'."
                )
            payload[key.strip()] = _coerce_scalar(value)

    target = str(
        payload.get("target")
        or payload.get("target_smiles")
        or payload.get("product")
        or payload.get("product_smiles")
        or ""
    ).strip()
    blocks_value = payload.get("building_blocks", payload.get("substrates", []))
    if isinstance(blocks_value, str):
        delimiter = "|" if "|" in blocks_value else ","
        blocks = [item.strip() for item in blocks_value.split(delimiter) if item.strip()]
    elif isinstance(blocks_value, (list, tuple)):
        blocks = [str(item).strip() for item in blocks_value if str(item).strip()]
    else:
        raise ValueError("building_blocks must be a JSON list or a '|' separated string.")
    if not target:
        raise ValueError("A target molecule is required.")
    if not blocks:
        raise ValueError("At least one building block is required.")

    options = payload.get("config", {})
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("config must be a JSON object.")
    options = dict(options)
    ignored = {
        "target",
        "target_smiles",
        "product",
        "product_smiles",
        "building_blocks",
        "substrates",
        "config",
    }
    aliases = {"top_k": "top_k_routes", "beam_width": "global_beam_width"}
    for key, value in payload.items():
        if key in ignored:
            continue
        options[aliases.get(key, key)] = value
    return target, blocks, options


class _BaseEnzymePlannerTool(BaseTool):
    db_path: str = _DEFAULT_DB_PATH

    _base_config: PlannerConfig = PrivateAttr()
    _database: EnzymeReactionDatabase = PrivateAttr()
    _model_client: Optional[EnzRetroClient] = PrivateAttr(default=None)
    _hybrid: bool = PrivateAttr(default=False)

    def __init__(
        self,
        db_path: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
        database_service: Optional[EnzymeReactionDatabase] = None,
        model_client: Optional[EnzRetroClient] = None,
        hybrid: bool = False,
        **kwargs: Any,
    ):
        database = database_service or EnzymeReactionDatabase(db_path or _DEFAULT_DB_PATH)
        if hybrid:
            config_values = dict(config or {})
            config_values.setdefault("use_database", True)
            config_values.setdefault("use_model", True)
            base = PlannerConfig.from_mapping(config_values)
        else:
            base = PlannerConfig.from_mapping(
                config, use_database=True, use_model=False
            )
        super().__init__(db_path=database.db_path, **kwargs)
        self._database = database
        self._model_client = model_client
        self._base_config = base
        self._hybrid = hybrid

    @staticmethod
    def _error(error_type: str, message: str) -> str:
        return json.dumps(
            {"ok": False, "error": {"type": error_type, "message": message}},
            ensure_ascii=False,
        )

    def _run(self, input_str: str) -> str:
        try:
            target, building_blocks, overrides = _parse_planner_input(input_str)
            merged = self._base_config.to_dict()
            merged.update(overrides)
            config = (
                PlannerConfig.from_mapping(merged)
                if self._hybrid
                else PlannerConfig.from_mapping(
                    merged, use_database=True, use_model=False
                )
            )
        except (TypeError, ValueError) as exc:
            return self._error("invalid_input", str(exc))
        try:
            engine = HybridPlannerEngine(
                database=self._database,
                model_client=self._model_client,
                config=config,
            )
            result = engine.plan(target, building_blocks)
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return self._error("planning_failed", str(exc))

    async def _arun(self, input_str: str) -> str:
        raise NotImplementedError("Asynchronous enzyme path planning is not implemented.")


class EnzymeRetroPathFinder(_BaseEnzymePlannerTool):
    """Database-only multi-step planner."""

    name: str = "EnzymeRetroPathFinder"
    description: str = (
        "Find complete multi-step enzymatic retrosynthesis routes using the local "
        "SQLite reaction database only. Call this tool ONCE with JSON: "
        '{"target":"product SMILES/name/Cxxxxx","building_blocks":["SMILES or ID"],'
        '"config":{"top_k_routes":10,"max_depth":10}}. The Python engine performs '
        "direction-aware AND-OR best-first/beam graph search, forward anchoring, "
        "cycle checks and Top-K ranking internally. Do not call this tool repeatedly "
        "to traverse intermediates."
    )

    def __init__(self, **kwargs: Any):
        kwargs["hybrid"] = False
        super().__init__(**kwargs)


class HybridEnzymeRetroPlanner(_BaseEnzymePlannerTool):
    """Database + local MooRetro multi-step planner."""

    name: str = "HybridEnzymeRetroPlanner"
    description: str = (
        "Find complete multi-step enzymatic retrosynthesis routes by fusing the "
        "local SQLite database with the local MooRetro single-step model. Call this "
        'tool ONCE with JSON: {"target":"product SMILES","building_blocks":['
        '"starting SMILES"],"config":{"top_k_routes":10,"max_depth":10}}. '
        "All graph traversal, DATABASE/MODEL/BOTH provenance, scoring, feedback "
        "and adjustment happen inside the deterministic Python planner; the agent "
        "must not iterate over intermediates itself."
    )

    def __init__(
        self,
        moo_root: Optional[str] = None,
        single_fn_type: str = "biocyc",
        ec_level: int = 3,
        **kwargs: Any,
    ):
        kwargs["hybrid"] = True
        if kwargs.get("model_client") is None:
            kwargs["model_client"] = EnzRetroClient(
                moo_root=moo_root,
                single_fn_type=single_fn_type,
                ec_level=ec_level,
            )
        super().__init__(**kwargs)
