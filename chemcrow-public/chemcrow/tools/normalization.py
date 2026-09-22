"""Normalize molecule names, database identifiers, and SMILES for agent tools."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from langchain.pydantic_v1 import PrivateAttr
from langchain.tools import BaseTool

from chemcrow.planner.chem_utils import canonicalize_smiles, molecule_key
from chemcrow.planner.database_service import EnzymeReactionDatabase
from chemcrow.utils import pubchem_query2smiles

__all__ = ["MoleculeInputNormalizer"]

_DEFAULT_DB_PATH = str(
    Path(__file__).resolve().parents[1] / "data" / "enzyme_retro.db"
)
_DATABASE_ID_RE = re.compile(r"^[CR]\d{5}$", re.IGNORECASE)
_PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/{}"


def _input_value(value: Any) -> str:
    if isinstance(value, Mapping):
        value = (
            value.get("value")
            or value.get("input")
            or value.get("canonical_smiles")
            or value.get("smiles")
            or value.get("id")
            or value.get("name")
        )
    text = str(value or "").strip()
    if not text:
        raise ValueError("Molecule input cannot be empty.")
    return text


def _parse_inputs(raw_input: str) -> Tuple[List[Tuple[str, str]], bool]:
    text = str(raw_input or "").strip()
    if not text:
        raise ValueError("Molecule input cannot be empty.")
    if not text.startswith("{"):
        return [("molecule", text)], True

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid normalization JSON: {exc.msg}.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Normalization JSON must be an object.")

    values: List[Tuple[str, str]] = []
    if "molecules" in payload:
        molecules = payload["molecules"]
        if not isinstance(molecules, list) or not molecules:
            raise ValueError("molecules must be a non-empty JSON list.")
        for item in molecules:
            role = (
                str(item.get("role") or "molecule")
                if isinstance(item, Mapping)
                else "molecule"
            )
            values.append((role, _input_value(item)))
    else:
        target_values = payload.get("targets")
        if target_values is None:
            target = (
                payload.get("target")
                or payload.get("target_smiles")
                or payload.get("product")
                or payload.get("product_smiles")
            )
            target_values = [] if target is None else [target]
        elif not isinstance(target_values, list) or not target_values:
            raise ValueError("targets must be a non-empty JSON list.")
        values.extend(("target", _input_value(item)) for item in target_values)
        blocks = payload.get("building_blocks", payload.get("substrates", []))
        if blocks is None:
            blocks = []
        if not isinstance(blocks, list):
            raise ValueError("building_blocks/substrates must be a JSON list.")
        values.extend(("building_block", _input_value(item)) for item in blocks)
        if not values:
            query = payload.get("query") or payload.get("input")
            if query is None:
                raise ValueError(
                    "Provide query/input, target plus building_blocks, or molecules."
                )
            values.append(("molecule", _input_value(query)))
    return values, bool(payload.get("allow_external_lookup", True))


class MoleculeInputNormalizer(BaseTool):
    """Resolve molecule input once and return a canonical representation."""

    name: str = "MoleculeInputNormalizer"
    description: str = (
        "ALWAYS call this once before using molecular database, safety, prediction, "
        "evaluation, or retrosynthesis tools. It accepts one plain molecule name, "
        "Cxxxxx ID, or SMILES; or JSON with target/targets, building_blocks, or molecules. It "
        "resolves local exact names and IDs, optionally falls back to PubChem for "
        "other names, and returns RDKit canonical isomeric SMILES plus provenance. "
        "Use the returned canonical_smiles in every later tool call."
    )

    db_path: str = _DEFAULT_DB_PATH
    allow_external_lookup: bool = True
    _database: EnzymeReactionDatabase = PrivateAttr()
    _name_resolver: Callable[[str], str] = PrivateAttr()

    def __init__(
        self,
        db_path: Optional[str] = None,
        database_service: Optional[EnzymeReactionDatabase] = None,
        name_resolver: Optional[Callable[[str], str]] = None,
        allow_external_lookup: bool = True,
        **kwargs: Any,
    ):
        database = database_service or EnzymeReactionDatabase(db_path or _DEFAULT_DB_PATH)
        super().__init__(
            db_path=database.db_path,
            allow_external_lookup=allow_external_lookup,
            **kwargs,
        )
        self._database = database
        self._name_resolver = name_resolver or (
            lambda query: pubchem_query2smiles(query, _PUBCHEM_URL)
        )

    def _normalize_one(
        self, raw_value: str, role: str, allow_external_lookup: bool
    ) -> Dict[str, Any]:
        direct_smiles = canonicalize_smiles(raw_value)
        try:
            molecule = self._database.normalize_molecule(raw_value)
        except ValueError as local_error:
            if direct_smiles is not None:
                raise
            if not allow_external_lookup or not self.allow_external_lookup:
                raise ValueError(
                    f"{raw_value!r} was not found by exact local name/ID lookup."
                ) from local_error
            try:
                resolved = self._name_resolver(raw_value)
            except Exception as exc:
                raise ValueError(
                    f"Unable to resolve molecule name {raw_value!r}: {exc}"
                ) from exc
            canonical = canonicalize_smiles(resolved)
            if canonical is None:
                raise ValueError(
                    f"External name lookup did not return a valid SMILES for "
                    f"{raw_value!r}."
                )
            return {
                "role": role,
                "original_input": raw_value,
                "matched_by": "external_name",
                "database_id": None,
                "name": raw_value,
                "canonical_smiles": canonical,
                "key": molecule_key(canonical, None),
                "formula": None,
                "charge": None,
                "valid": True,
            }

        if direct_smiles is not None:
            matched_by = "smiles"
        elif _DATABASE_ID_RE.match(raw_value):
            matched_by = "database_id"
        else:
            matched_by = "local_exact_name"
        canonical = molecule.canonical_smiles or canonicalize_smiles(molecule.smiles)
        if canonical is None:
            raise ValueError(
                f"{raw_value!r} resolved locally but has no valid SMILES structure."
            )
        return {
            "role": role,
            "original_input": raw_value,
            "matched_by": matched_by,
            "database_id": molecule.metabolite_id,
            "name": molecule.name,
            "canonical_smiles": canonical,
            "key": molecule.key,
            "formula": molecule.formula,
            "charge": molecule.charge,
            "valid": True,
        }

    def _run(self, input_str: str) -> str:
        try:
            values, request_external = _parse_inputs(input_str)
        except ValueError as exc:
            return json.dumps(
                {"ok": False, "error": {"type": "invalid_input", "message": str(exc)}},
                ensure_ascii=False,
            )

        normalized = []
        errors = []
        for index, (role, raw_value) in enumerate(values):
            try:
                normalized.append(
                    self._normalize_one(raw_value, role, request_external)
                )
            except ValueError as exc:
                errors.append(
                    {
                        "index": index,
                        "role": role,
                        "original_input": raw_value,
                        "message": str(exc),
                    }
                )
        ok = not errors
        output: Dict[str, Any] = {
            "ok": ok,
            "normalization": "RDKIT_CANONICAL_ISOMERIC_SMILES",
            "normalized_molecules": normalized,
            "errors": errors,
        }
        targets = [
            item["canonical_smiles"]
            for item in normalized
            if item["role"] == "target"
        ]
        blocks = [
            item["canonical_smiles"]
            for item in normalized
            if item["role"] == "building_block"
        ]
        if targets or blocks:
            output["normalized_input"] = {
                "target": targets[0] if len(targets) == 1 else None,
                "targets": targets,
                "building_blocks": blocks,
            }
        if len(normalized) == 1 and not errors:
            output["canonical_smiles"] = normalized[0]["canonical_smiles"]
        if errors:
            output["error"] = {
                "type": "normalization_failed",
                "message": (
                    "One or more molecules could not be normalized; do not call "
                    "downstream chemistry tools until all errors are resolved."
                ),
            }
        return json.dumps(output, ensure_ascii=False)

    async def _arun(self, input_str: str) -> str:
        raise NotImplementedError("Asynchronous molecule normalization is not implemented.")
