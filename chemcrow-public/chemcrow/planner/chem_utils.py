"""Small RDKit helpers used by the deterministic planner."""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdFMCS

from .models import MoleculeNode

RDLogger.DisableLog("rdApp.*")

_EC_RE = re.compile(
    r"(?:\[?EC\]?\s*)?(\d+|-)\.(\d+|-)\.(\d+|-)(?:\.(\d+|-))?",
    re.I,
)
_CURRENCY_NAMES = {
    "h2o",
    "water",
    "h+",
    "proton",
    "oxygen",
    "dioxygen",
    "hydrogen",
    "carbon dioxide",
    "bicarbonate",
    "orthophosphate",
    "phosphate",
    "diphosphate",
    "pyrophosphate",
    "ammonia",
    "atp",
    "adp",
    "amp",
    "gtp",
    "gdp",
    "ctp",
    "cdp",
    "utp",
    "udp",
    "nad+",
    "nadh",
    "nadp+",
    "nadph",
    "fad",
    "fadh2",
    "coenzyme a",
    "coa",
}


@lru_cache(maxsize=100000)
def canonicalize_smiles(smiles: str | None) -> str | None:
    if not smiles:
        return None
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


@lru_cache(maxsize=100000)
def _mol(smiles: str | None):
    return Chem.MolFromSmiles(smiles) if smiles else None


def molecule_key(smiles: str | None, metabolite_id: str | None) -> str:
    canonical = canonicalize_smiles(smiles)
    if canonical:
        return "smi:" + canonical
    if metabolite_id:
        return "id:" + metabolite_id.upper()
    raise ValueError("A molecule needs a valid SMILES or database identifier.")


def heavy_atom_count(molecule: MoleculeNode) -> int:
    mol = _mol(molecule.canonical_smiles or molecule.smiles)
    return int(mol.GetNumHeavyAtoms()) if mol is not None else 0


def carbon_count(molecule: MoleculeNode) -> int:
    mol = _mol(molecule.canonical_smiles or molecule.smiles)
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)


@lru_cache(maxsize=100000)
def structural_similarity(smiles_a: str | None, smiles_b: str | None) -> float | None:
    mol_a = _mol(smiles_a)
    mol_b = _mol(smiles_b)
    if mol_a is None or mol_b is None:
        return None
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


@lru_cache(maxsize=100000)
def conserved_atom_fraction(
    target_smiles: str | None, precursor_smiles: str | None
) -> float | None:
    """Approximate target-atom conservation with an RDKit MCS."""
    target = _mol(target_smiles)
    precursor = _mol(precursor_smiles)
    if target is None or precursor is None or target.GetNumHeavyAtoms() == 0:
        return None
    try:
        result = rdFMCS.FindMCS(
            [target, precursor],
            timeout=1,
            ringMatchesRingOnly=True,
            completeRingsOnly=False,
        )
        return min(1.0, float(result.numAtoms) / float(target.GetNumHeavyAtoms()))
    except Exception:
        return None


def extract_ec_numbers(value: str | None) -> Tuple[str, ...]:
    if not value:
        return ()
    found = []
    for match in _EC_RE.finditer(str(value)):
        ec = ".".join(part for part in match.groups() if part is not None)
        if ec not in found:
            found.append(ec)
    return tuple(found)


def ec_compatibility(left: Sequence[str], right: Sequence[str]) -> float | None:
    if not left or not right:
        return None
    best = 0.0
    for item_a in left:
        parts_a = item_a.replace("EC ", "").split(".")
        for item_b in right:
            parts_b = item_b.replace("EC ", "").split(".")
            matched = 0
            compared = 0
            for a, b in zip(parts_a, parts_b):
                if a == "-" or b == "-":
                    break
                compared += 1
                if a != b:
                    break
                matched += 1
            if compared:
                best = max(best, matched / 4.0)
    return best


def is_generic_structure(smiles: str | None, formula: str | None = None) -> bool:
    text = str(smiles or "")
    if not text:
        return True
    if "*" in text or "[R]" in text or "[X]" in text:
        return True
    if formula and re.search(r"(?:^|\d)[RX](?:\d|$)", str(formula)):
        return True
    return canonicalize_smiles(text) is None


def is_currency(molecule: MoleculeNode, currency_ids: Iterable[str]) -> bool:
    if molecule.metabolite_id and molecule.metabolite_id.upper() in set(currency_ids):
        return True
    return (molecule.name or "").strip().casefold() in _CURRENCY_NAMES


def normalized_toxicity(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if 0.0 <= value <= 1.0:
        return value
    # The local database stores an uncalibrated logit-like score.
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def classify_precursors(
    target: MoleculeNode,
    precursors: Sequence[MoleculeNode],
    currency_ids: Iterable[str],
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    currency = tuple(
        item.key for item in precursors if is_currency(item, currency_ids)
    )
    eligible = [item for item in precursors if item.key not in set(currency)]
    if not eligible:
        return (), (), currency

    ranked: List[Tuple[float, float, int, str]] = []
    target_smiles = target.canonical_smiles or target.smiles
    for item in eligible:
        item_smiles = item.canonical_smiles or item.smiles
        conserved = conserved_atom_fraction(target_smiles, item_smiles)
        similarity = structural_similarity(target_smiles, item_smiles)
        ranked.append(
            (
                conserved if conserved is not None else -1.0,
                similarity if similarity is not None else -1.0,
                carbon_count(item),
                item.key,
            )
        )
    ranked.sort(reverse=True)
    best = ranked[0]
    # Keep genuine multi-substrate skeleton reactions as AND-main branches when
    # their overlap is close to the best match. All other non-currency inputs are
    # still mandatory, but are labelled auxiliary.
    main = []
    for conserved, similarity, carbons, key in ranked:
        close = conserved >= max(0.35, best[0] - 0.12)
        if close and carbons >= 2:
            main.append(key)
    if not main:
        main = [best[3]]
    auxiliary = tuple(item.key for item in eligible if item.key not in set(main))
    return tuple(sorted(main)), tuple(sorted(auxiliary)), tuple(sorted(currency))


def candidate_structure_metrics(
    target: MoleculeNode, precursors: Sequence[MoleculeNode]
) -> Tuple[float | None, float | None]:
    target_smiles = target.canonical_smiles or target.smiles
    similarities = []
    conservations = []
    for item in precursors:
        item_smiles = item.canonical_smiles or item.smiles
        similarity = structural_similarity(target_smiles, item_smiles)
        conservation = conserved_atom_fraction(target_smiles, item_smiles)
        if similarity is not None:
            similarities.append(similarity)
        if conservation is not None:
            conservations.append(conservation)
    return (
        max(similarities) if similarities else None,
        max(conservations) if conservations else None,
    )
