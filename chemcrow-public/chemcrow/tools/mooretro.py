"""LangChain tool for the local MooRetro enzymatic retrosynthesis model."""

from __future__ import annotations

import importlib
import json
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from langchain.pydantic_v1 import PrivateAttr
from langchain.tools import BaseTool

from chemcrow.planner.chem_utils import canonicalize_smiles

__all__ = ["MooRetroSingleStepPredict", "MooRetroSingleStepLocal"]

_DEFAULT_MOO_ROOT = str(Path(__file__).resolve().parents[3] / "MooRetro-master")
_MOO_CALL_LOCK = threading.RLock()


@contextmanager
def _working_directory(path: str):
    """Enter MooRetro's root because its model paths are relative."""
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _json_default(value: Any):
    """Convert numpy/torch scalar-like values returned by MooRetro to JSON."""
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


class MooRetroSingleStepPredict(BaseTool):
    """Predict enzymatic precursors for one product with local MooRetro."""

    name: str = "MooRetroSingleStepPredict"
    description: str = (
        "Predict ONE enzymatic retrosynthesis step with the local MooRetro model. "
        "Use it for novel enzymatic/biocatalytic precursor predictions. Input can "
        "be the canonical product SMILES returned by MoleculeInputNormalizer, or JSON such as "
        "{\"target_smiles\":\"...\",\"building_blocks\":[\"...\"]}. "
        "All input and predicted precursor SMILES are RDKit-canonicalized. "
        "building_blocks is optional and is used to calculate precursor similarity. "
        "Returns JSON with ranked precursor SMILES, EC number, model score, reaction "
        "template, similarity, and feasibility."
    )

    moo_root: str = _DEFAULT_MOO_ROOT
    single_fn_type: str = "biocyc"
    ec_level: int = 3
    max_results: int = 10

    _single_fn: Optional[Callable[..., Any]] = PrivateAttr(default=None)
    _load_error: Optional[str] = PrivateAttr(default=None)

    def __init__(
        self,
        moo_root: Optional[str] = None,
        single_fn_type: str = "biocyc",
        ec_level: int = 3,
        max_results: int = 10,
        **kwargs,
    ):
        # Keep ``model`` as a user-friendly alias for neural inference. It
        # previously selected the XLSX lookup backend inside MooRetro.
        if single_fn_type == "model":
            single_fn_type = "biocyc"
        root = str(
            Path(moo_root or os.getenv("MOO_RETRO_PATH") or _DEFAULT_MOO_ROOT)
            .expanduser()
            .resolve()
        )
        if single_fn_type not in {"biocyc", "ecreact", "model_xlsx", "xlsx", "model"}:
            raise ValueError("single_fn_type must be biocyc, ecreact, or model_xlsx")
        if not 1 <= ec_level <= 4:
            raise ValueError("ec_level must be between 1 and 4")
        if max_results < 1:
            raise ValueError("max_results must be at least 1")
        super().__init__(
            moo_root=root,
            single_fn_type=single_fn_type,
            ec_level=ec_level,
            max_results=max_results,
            **kwargs,
        )

    @staticmethod
    def _error(error_type: str, message: str) -> str:
        return json.dumps(
            {"ok": False, "error": {"type": error_type, "message": message}},
            ensure_ascii=False,
        )

    @staticmethod
    def _molecule_value(value: Any) -> str:
        if isinstance(value, Mapping):
            value = (
                value.get("canonical_smiles")
                or value.get("smiles")
                or value.get("value")
            )
        text = str(value or "").strip()
        if not text:
            raise ValueError("A molecule must contain canonical_smiles or SMILES.")
        return text

    @staticmethod
    def _parse_tool_input(raw_input: str) -> Tuple[str, List[str]]:
        text = (raw_input or "").strip()
        if not text:
            raise ValueError("A target product SMILES is required.")
        if text.startswith("{"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON input: {exc.msg}.") from exc
            if not isinstance(payload, dict):
                raise ValueError("JSON input must be an object.")
            target_value = (
                payload.get("target_smiles")
                or payload.get("product_smiles")
                or payload.get("target")
                or ""
            )
            target = MooRetroSingleStepPredict._molecule_value(target_value)
            blocks_value = payload.get("building_blocks", [])
            if blocks_value is None:
                blocks: List[str] = []
            elif isinstance(blocks_value, list):
                blocks = [
                    MooRetroSingleStepPredict._molecule_value(item)
                    for item in blocks_value
                ]
            else:
                raise ValueError("building_blocks must be a JSON list of molecules.")
            return target, blocks

        # Compatibility with the draft integration's PRODUCT||BB1,BB2 format.
        target, separator, block_text = text.partition("||")
        blocks = (
            [item.strip() for item in block_text.split(",") if item.strip()]
            if separator
            else []
        )
        return target.strip(), blocks

    @staticmethod
    def _normalize_inputs(
        target: str, building_blocks: List[str]
    ) -> Tuple[str, List[str]]:
        canonical_target = canonicalize_smiles(target)
        if canonical_target is None:
            raise ValueError(f"Invalid target SMILES: {target}")
        canonical_blocks = []
        invalid_blocks = []
        for block in building_blocks:
            canonical = canonicalize_smiles(block)
            if canonical is None:
                invalid_blocks.append(block)
            else:
                canonical_blocks.append(canonical)
        if invalid_blocks:
            raise ValueError(
                "Invalid building-block SMILES: " + ", ".join(invalid_blocks)
            )
        return canonical_target, canonical_blocks

    def _load_single_fn(self) -> Callable[..., Any]:
        if self._single_fn is not None:
            return self._single_fn
        if self._load_error is not None:
            raise RuntimeError(self._load_error)

        root = Path(self.moo_root)
        expected_module = root / "utils" / "single_retro_fn.py"
        if not expected_module.is_file():
            self._load_error = f"MooRetro entry point not found: {expected_module}"
            raise RuntimeError(self._load_error)
        try:
            root_text = str(root)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)
            with _MOO_CALL_LOCK, _working_directory(root_text):
                module = importlib.import_module("utils.single_retro_fn")
            module_path = Path(module.__file__).resolve()
            if module_path != expected_module.resolve():
                raise ImportError(
                    "Python resolved utils.single_retro_fn from "
                    f"{module_path}, expected {expected_module.resolve()}"
                )
            self._single_fn = module.single_fn
            return self._single_fn
        except Exception as exc:
            self._load_error = f"Unable to import the local MooRetro model: {exc}"
            raise RuntimeError(self._load_error) from exc

    def _runtime_device(self) -> Optional[str]:
        if self.single_fn_type in {"model_xlsx", "xlsx"}:
            return "cpu"
        module = sys.modules.get("utils.single_retro_fn")
        cache = getattr(module, "_MODEL_CACHE", {}) if module else {}
        for key, wrapper in cache.items():
            if len(key) < 2 or key[0] != self.single_fn_type or key[1] != self.ec_level:
                continue
            predictor = getattr(wrapper, "predictor", None)
            model = getattr(predictor, "model", None)
            if model is not None:
                try:
                    return str(next(model.parameters()).device)
                except (StopIteration, AttributeError):
                    pass
            device = getattr(predictor, "device", None)
            if device is not None:
                return str(device)
        return None

    def _run(self, input_str: str) -> str:
        try:
            target, building_blocks = self._parse_tool_input(input_str)
        except ValueError as exc:
            return self._error("invalid_input", str(exc))
        try:
            target, building_blocks = self._normalize_inputs(target, building_blocks)
        except ValueError as exc:
            return self._error("invalid_smiles", str(exc))

        try:
            single_fn = self._load_single_fn()
            # Mirrors test.py::test_for_single_fn. product2order.py uses relative
            # checkpoint/tokenizer paths, so the call must run from MooRetro root.
            with _MOO_CALL_LOCK, _working_directory(self.moo_root):
                results = single_fn(
                    target_smile=target,
                    building_blocks=building_blocks,
                    ec_level=self.ec_level,
                    single_fn_type=self.single_fn_type,
                )
        except Exception as exc:
            return self._error("prediction_failed", str(exc))

        predictions: List[Dict[str, Any]] = []
        rejected_predictions: List[Dict[str, Any]] = []
        for index, result in enumerate(results[: self.max_results]):
            value = result.to_dict if hasattr(result, "to_dict") else result
            if callable(value):
                value = value()
            if not isinstance(value, Mapping):
                rejected_predictions.append(
                    {"index": index, "reason": "prediction is not a mapping"}
                )
                continue
            value = dict(value)
            raw_reactants = value.get("reactants")
            raw_all_reactants = value.get("all_reactants") or raw_reactants
            if isinstance(raw_all_reactants, (list, tuple)):
                reactants = [str(item) for item in raw_all_reactants if item]
            else:
                reactants = [
                    item for item in str(raw_all_reactants or "").split(".") if item
                ]
            canonical_reactants = []
            invalid_reactants = []
            for reactant in reactants:
                canonical = canonicalize_smiles(reactant)
                if canonical is None:
                    invalid_reactants.append(reactant)
                else:
                    canonical_reactants.append(canonical)
            if invalid_reactants or not canonical_reactants:
                rejected_predictions.append(
                    {
                        "index": index,
                        "reason": "invalid predicted reactant SMILES",
                        "invalid_reactants": invalid_reactants,
                    }
                )
                continue
            canonical_reactants.sort()
            value["raw_reactants"] = raw_reactants
            value["raw_all_reactants"] = raw_all_reactants
            value["reactant_smiles"] = canonical_reactants
            value["reactants"] = ".".join(canonical_reactants)
            value["normalization"] = "RDKIT_CANONICAL_ISOMERIC_SMILES"
            predictions.append(value)
        return json.dumps(
            {
                "ok": True,
                "target_smiles": target,
                "input_normalization": "RDKIT_CANONICAL_ISOMERIC_SMILES",
                "model": self.single_fn_type,
                "runtime_device": self._runtime_device(),
                "backend_mode": (
                    "database_lookup"
                    if self.single_fn_type in {"model_xlsx", "xlsx"}
                    else "neural_model"
                ),
                "ec_level": self.ec_level,
                "prediction_count": len(predictions),
                "rejected_prediction_count": len(rejected_predictions),
                "rejected_predictions": rejected_predictions,
                "predictions": predictions,
            },
            ensure_ascii=False,
            default=_json_default,
        )

    async def _arun(self, input_str: str) -> str:
        raise NotImplementedError("MooRetro asynchronous prediction is not implemented.")


# Backward-compatible name for code using the earlier draft integration.
MooRetroSingleStepLocal = MooRetroSingleStepPredict
