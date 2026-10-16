"""Lazy, cached MooRetro client without a LangChain dependency."""

from __future__ import annotations

import importlib
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .chem_utils import (
    candidate_structure_metrics,
    classify_precursors,
    extract_ec_numbers,
    is_generic_structure,
    molecule_key,
)
from .config import DEFAULT_CURRENCY_IDS
from .models import MoleculeNode, ReactionCandidate

_DEFAULT_MOO_ROOT = str(Path(__file__).resolve().parents[3] / "MooRetro-master")
_MODEL_CALL_LOCK = threading.RLock()


@contextmanager
def _working_directory(path: str):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class EnzRetroClient:
    """Call MooRetro once per unique target and convert predictions to candidates."""

    def __init__(
        self,
        moo_root: Optional[str] = None,
        single_fn_type: str = "biocyc",
        ec_level: int = 3,
        predictor: Optional[Callable[..., Any]] = None,
    ):
        # ``model`` is an alias for the neural BioCyc checkpoint, never XLSX.
        if single_fn_type == "model":
            single_fn_type = "biocyc"
        if single_fn_type not in {"biocyc", "ecreact", "model_xlsx", "xlsx", "model"}:
            raise ValueError("single_fn_type must be biocyc, ecreact, or model_xlsx")
        if not 1 <= int(ec_level) <= 4:
            raise ValueError("ec_level must be between 1 and 4")
        self.moo_root = str(
            Path(moo_root or os.getenv("MOO_RETRO_PATH") or _DEFAULT_MOO_ROOT)
            .expanduser()
            .resolve()
        )
        self.single_fn_type = single_fn_type
        self.ec_level = int(ec_level)
        self._predictor = predictor
        self._load_error: Optional[str] = None
        self._cache: Dict[Tuple[object, ...], Tuple[Mapping[str, Any], ...]] = {}
        self._lock = threading.RLock()
        self.cache_hits = 0
        self.cache_misses = 0

    def _load_predictor(self) -> Callable[..., Any]:
        if self._predictor is not None:
            return self._predictor
        if self._load_error is not None:
            raise RuntimeError(self._load_error)
        expected = Path(self.moo_root) / "utils" / "single_retro_fn.py"
        if not expected.is_file():
            self._load_error = f"MooRetro entry point not found: {expected}"
            raise RuntimeError(self._load_error)
        try:
            root_text = self.moo_root
            if root_text not in sys.path:
                sys.path.insert(0, root_text)
            with _MODEL_CALL_LOCK, _working_directory(root_text):
                module = importlib.import_module("utils.single_retro_fn")
            module_path = Path(module.__file__).resolve()
            if module_path != expected.resolve():
                raise ImportError(
                    f"Resolved utils.single_retro_fn from {module_path}, "
                    f"expected {expected.resolve()}"
                )
            self._predictor = module.single_fn
            return self._predictor
        except Exception as exc:
            self._load_error = f"Unable to import the local MooRetro model: {exc}"
            raise RuntimeError(self._load_error) from exc

    @staticmethod
    def _serialize_result(result: Any) -> Mapping[str, Any]:
        value = result.to_dict if hasattr(result, "to_dict") else result
        if callable(value):
            value = value()
        if not isinstance(value, Mapping):
            raise TypeError("MooRetro prediction must serialize to a mapping.")
        return dict(value)

    def _raw_predictions(
        self,
        target_smiles: str,
        building_blocks: Sequence[str],
        top_k: int,
    ) -> Tuple[Mapping[str, Any], ...]:
        key = (
            target_smiles,
            tuple(sorted(building_blocks)),
            self.single_fn_type,
            self.ec_level,
            int(top_k),
        )
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self.cache_hits += 1
                return cached
        predictor = self._load_predictor()
        with _MODEL_CALL_LOCK, _working_directory(self.moo_root):
            results = predictor(
                target_smile=target_smiles,
                building_blocks=list(building_blocks),
                ec_level=self.ec_level,
                single_fn_type=self.single_fn_type,
            )
        serialized = tuple(self._serialize_result(value) for value in results[:top_k])
        with self._lock:
            self.cache_misses += 1
            self._cache[key] = serialized
        return serialized

    def predict_candidates(
        self,
        target: MoleculeNode,
        building_blocks: Sequence[MoleculeNode],
        *,
        top_k: int = 10,
        currency_ids: Iterable[str] = DEFAULT_CURRENCY_IDS,
        normalizer: Optional[Callable[[str], MoleculeNode]] = None,
    ) -> Tuple[ReactionCandidate, ...]:
        target_smiles = target.canonical_smiles or target.smiles
        if not target_smiles:
            return ()
        block_smiles = tuple(
            item.canonical_smiles or item.smiles
            for item in building_blocks
            if item.canonical_smiles or item.smiles
        )
        raw_predictions = self._raw_predictions(target_smiles, block_smiles, top_k)
        candidates: List[ReactionCandidate] = []
        for index, raw in enumerate(raw_predictions):
            reactants_value = raw.get("all_reactants") or raw.get("reactants")
            if isinstance(reactants_value, (list, tuple)):
                reactant_smiles = [str(item) for item in reactants_value if item]
            else:
                reactant_smiles = [
                    item for item in str(reactants_value or "").split(".") if item
                ]
            precursors: List[MoleculeNode] = []
            invalid = False
            for smiles in reactant_smiles:
                try:
                    if normalizer is not None:
                        precursor = normalizer(smiles)
                    else:
                        key = molecule_key(smiles, None)
                        precursor = MoleculeNode(
                            key=key,
                            query=smiles,
                            smiles=smiles,
                            canonical_smiles=key[4:],
                            generic=is_generic_structure(smiles),
                        )
                    precursors.append(precursor)
                except ValueError:
                    invalid = True
                    break
            if invalid or not precursors:
                continue
            main, auxiliary, currency = classify_precursors(
                target, precursors, currency_ids
            )
            noncurrency = [
                item for item in precursors if item.key not in set(currency)
            ]
            structural, conservation = candidate_structure_metrics(
                target, noncurrency
            )
            ec = extract_ec_numbers(
                " ".join(
                    str(value)
                    for value in (raw.get("ec"), raw.get("templates"))
                    if value
                )
            )
            warnings = []
            if any(item.generic for item in noncurrency):
                warnings.append("model returned a generic precursor structure")
            if raw.get("yield") is not None or raw.get("toxicity") is not None:
                warnings.append(
                    "MooRetro yield/toxicity fields are uncalibrated estimates"
                )
            candidates.append(
                ReactionCandidate(
                    candidate_id=f"model:{target.key}:{index + 1}",
                    target=target,
                    precursors=tuple(precursors),
                    source="MODEL",
                    main_precursor_keys=main,
                    auxiliary_precursor_keys=auxiliary,
                    currency_precursor_keys=currency,
                    coefficients=tuple("1" for _ in precursors),
                    all_products=(target,),
                    product_coefficients=("1",),
                    ec_numbers=ec,
                    template=str(raw.get("templates") or "") or None,
                    rank=str(raw.get("topk") or raw.get("top_k") or index + 1),
                    model_score=_as_float(
                        raw.get("scores")
                        if raw.get("scores") is not None
                        else raw.get("score")
                    ),
                    structural_similarity=(
                        _as_float(raw.get("similarity"))
                        if raw.get("similarity") is not None
                        else structural
                    ),
                    atom_conservation=None,
                    approx_structural_overlap=conservation,
                    feasibility=_as_float(raw.get("feasibility")),
                    yield_score=_as_float(
                        raw.get("yield")
                        if raw.get("yield") is not None
                        else raw.get("yield_estimate")
                    ),
                    toxicity=_as_float(raw.get("toxicity")),
                    ec_consistency=0.60 if ec else None,
                    warnings=tuple(warnings),
                    metadata={
                        "model": self.single_fn_type,
                        "ec_level": self.ec_level,
                        "raw_prediction_index": index,
                        "atom_mapping_source": "NONE",
                        "retrosynthesis_direction": "target_to_predicted_precursors",
                        "metric_provenance": {
                            "model_score": "MOORETRO_BEAM_SCORE",
                            "feasibility": "MOORETRO_DELTA_SASCORE_HEURISTIC",
                            "structural_similarity": "MOORETRO_BUILDING_BLOCK_SIMILARITY",
                            "approx_structural_overlap": "RDKIT_MCS_TARGET_FRACTION",
                            "atom_conservation": "UNAVAILABLE_NO_REACTION_ATOM_MAPPING",
                            "yield_score": "MOORETRO_RANDOM_PLACEHOLDER_DISABLED_BY_DEFAULT",
                            "toxicity": "MOORETRO_RANDOM_PLACEHOLDER_DISABLED_BY_DEFAULT",
                        },
                    },
                )
            )
        return tuple(candidates)

    def predict_candidates_with_audit(
        self,
        target: MoleculeNode,
        building_blocks: Sequence[MoleculeNode],
        *,
        top_k: int = 10,
        currency_ids: Iterable[str] = DEFAULT_CURRENCY_IDS,
        normalizer: Optional[Callable[[str], MoleculeNode]] = None,
    ) -> Tuple[Tuple[ReactionCandidate, ...], Dict[str, Any]]:
        """Return converted candidates together with every raw model result.

        The normal planner API intentionally returns only valid structured
        candidates. Benchmark callers need the pre-drop snapshot as well, so
        this companion method reports one audit entry per raw prediction and a
        rejection reason for entries that could not be converted.
        """
        candidates = self.predict_candidates(
            target,
            building_blocks,
            top_k=top_k,
            currency_ids=currency_ids,
            normalizer=normalizer,
        )
        target_smiles = target.canonical_smiles or target.smiles
        block_smiles = tuple(
            item.canonical_smiles or item.smiles
            for item in building_blocks
            if item.canonical_smiles or item.smiles
        )
        raw_predictions = (
            self._raw_predictions(target_smiles, block_smiles, top_k)
            if target_smiles
            else ()
        )
        valid_indices = {
            int(candidate.metadata["raw_prediction_index"]): candidate.candidate_id
            for candidate in candidates
        }
        records: List[Dict[str, Any]] = []
        for index, raw in enumerate(raw_predictions):
            reason = None
            if index not in valid_indices:
                value = raw.get("all_reactants") or raw.get("reactants")
                if isinstance(value, (list, tuple)):
                    reactants = [str(item) for item in value if item]
                else:
                    reactants = [item for item in str(value or "").split(".") if item]
                if not reactants:
                    reason = "missing_precursors"
                else:
                    try:
                        for smiles in reactants:
                            if normalizer is not None:
                                normalizer(smiles)
                            else:
                                molecule_key(smiles, None)
                    except (TypeError, ValueError) as exc:
                        reason = "invalid_precursor: " + str(exc)
                    if reason is None:
                        reason = "candidate_conversion_failed"
            records.append(
                {
                    "raw_rank": index + 1,
                    "raw_prediction": dict(raw),
                    "candidate_id": valid_indices.get(index),
                    "valid": index in valid_indices,
                    "invalid_reason": reason,
                }
            )
        return candidates, {
            "requested": int(top_k),
            "raw_returned": len(raw_predictions),
            "valid_returned": len(candidates),
            "records": records,
        }


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

    def cache_info(self) -> Dict[str, Any]:
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "entries": len(self._cache),
            "model": self.single_fn_type,
            "backend_mode": (
                "database_lookup"
                if self.single_fn_type in {"model_xlsx", "xlsx"}
                else "neural_model"
            ),
            "runtime_device": self._runtime_device(),
        }
