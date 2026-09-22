
# 该model_xlsx模式会用自定义的 xlsx 反应库；如果有 SMILES 映射，会把反应物 ID 转成 SMILES 返回给 MCTS。


import os
import re
import zipfile
import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

from rdkit import Chem

from config.config import SingleStepResult
from utils.chem import canonicalize_mol_smi, get_delta_sascore_for_feasibility, similarity_calculator


DEFAULT_MODEL_XLSX = "E:\\ChxW\\\u9006\u5408\u6210\\\u7c7b\u7403\u7ea2\u7ec6\u83cc\\model_output.xlsx"
LOCAL_MODEL_XLSX = "model_output_inspect.xlsx"
XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PKG_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
TOKEN_RE = re.compile(
    r"(?:(\d+(?:\.\d+)?)\s+)?"
    r"([A-Za-z][A-Za-z0-9_.:-]*)"
    r"(?:\[[^\]]+\])?"
)
SMILES_COLUMN_CANDIDATES = (
    "smiles",
    "smile",
    "canonical_smiles",
    "canonical smiles",
    "isomeric_smiles",
    "isomeric smiles",
    "mol smiles",
    "metabolite smiles",
)
MAPPING_KEY_CANDIDATES = ("id", "compound_id", "compound id", "metabolite_id", "metabolite id", "name")


@dataclass(frozen=True)
class ModelReaction:
    reaction_id: str
    name: str
    equation: str
    reactants: Tuple[str, ...]
    products: Tuple[str, ...]
    reversible: bool
    direction: str = "forward"

    @property
    def route_id(self) -> str:
        if self.direction == "forward":
            return self.reaction_id
        return f"{self.reaction_id}:reverse"


@dataclass(frozen=True)
class ModelData:
    producers: Dict[str, List[ModelReaction]]
    name_map: Dict[str, str]
    id_to_smiles: Dict[str, str]
    smiles_to_id: Dict[str, str]


_MODEL_XLSX_CACHE: Dict[Tuple[str, Optional[str]], ModelData] = {}


def normalize_compound_id(token: str) -> str:
    token = str(token).strip()
    if token.endswith("_c") or token.endswith("_b") or token.endswith("_e"):
        return token[:-2]
    return token


def _canonical_smiles_key(value: str) -> str:
    value = str(value).strip()
    if not value:
        return ""
    try:
        return canonicalize_mol_smi(value)
    except Exception:
        return value


def _find_column(headers: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    normalized = {str(header).strip().lower(): header for header in headers}
    for candidate in candidates:
        match = normalized.get(candidate)
        if match is not None:
            return match
    return None


def _column_index(cell_ref: str) -> int:
    letters = re.sub(r"\d+", "", cell_ref)
    index = 0
    for char in letters:
        index = index * 26 + (ord(char.upper()) - ord("A") + 1)
    return index - 1


def _read_xml_from_xlsx(xlsx: zipfile.ZipFile, name: str) -> ET.Element:
    with xlsx.open(name) as fileobj:
        return ET.fromstring(fileobj.read())


def _read_shared_strings(xlsx: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in xlsx.namelist():
        return []
    root = _read_xml_from_xlsx(xlsx, "xl/sharedStrings.xml")
    values: List[str] = []
    for item in root.findall(f"{XLSX_NS}si"):
        values.append("".join(node.text or "" for node in item.iter(f"{XLSX_NS}t")))
    return values


def _workbook_sheets(xlsx: zipfile.ZipFile) -> Dict[str, str]:
    workbook = _read_xml_from_xlsx(xlsx, "xl/workbook.xml")
    rels = _read_xml_from_xlsx(xlsx, "xl/_rels/workbook.xml.rels")
    rel_map = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall(f"{PKG_REL_NS}Relationship")
    }

    sheets: Dict[str, str] = {}
    for sheet in workbook.findall(f".//{XLSX_NS}sheet"):
        name = sheet.attrib["name"]
        rid = sheet.attrib[f"{REL_NS}id"]
        target = rel_map[rid].lstrip("/")
        if not target.startswith("xl/"):
            target = f"xl/{target}"
        sheets[name] = target.replace("\\", "/")
    return sheets


def _read_cell_value(cell: ET.Element, shared_strings: Sequence[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(f"{XLSX_NS}t")).strip()

    value_node = cell.find(f"{XLSX_NS}v")
    if value_node is None or value_node.text is None:
        return ""

    value = value_node.text.strip()
    if cell_type == "s":
        try:
            return shared_strings[int(value)].strip()
        except (IndexError, ValueError):
            return ""
    return value


def _read_sheet_rows(xlsx_path: str, sheet_name: str) -> List[Dict[str, str]]:
    with zipfile.ZipFile(xlsx_path) as xlsx:
        sheets = _workbook_sheets(xlsx)
        if sheet_name not in sheets:
            available = ", ".join(sorted(sheets))
            raise ValueError(f"Sheet {sheet_name!r} not found. Available sheets: {available}")

        shared_strings = _read_shared_strings(xlsx)
        root = _read_xml_from_xlsx(xlsx, sheets[sheet_name])
        rows: List[List[str]] = []
        for row in root.findall(f".//{XLSX_NS}row"):
            values: List[str] = []
            for cell in row.findall(f"{XLSX_NS}c"):
                idx = _column_index(cell.attrib.get("r", "A1"))
                while len(values) <= idx:
                    values.append("")
                values[idx] = _read_cell_value(cell, shared_strings)
            rows.append(values)

    if not rows:
        return []
    headers = [value.strip() for value in rows[0]]
    records = []
    for row in rows[1:]:
        records.append({
            header: row[idx].strip() if idx < len(row) else ""
            for idx, header in enumerate(headers)
        })
    return records


def _parse_side(side: str) -> Tuple[str, ...]:
    compounds: List[str] = []
    for part in side.split("+"):
        match = TOKEN_RE.search(part.strip())
        if match:
            compounds.append(normalize_compound_id(match.group(2)))
    return tuple(compounds)


def _parse_equation(equation: str) -> Tuple[Tuple[str, ...], Tuple[str, ...], bool]:
    equation = str(equation).replace("&lt;=&gt;", "<=>").replace("&lt;--&gt;", "<=>")
    if "<=>" in equation:
        left, right = equation.split("<=>", 1)
        reversible = True
    elif "-->" in equation:
        left, right = equation.split("-->", 1)
        reversible = False
    elif "=>" in equation:
        left, right = equation.split("=>", 1)
        reversible = False
    else:
        return tuple(), tuple(), False
    return _parse_side(left), _parse_side(right), reversible


def _read_mapping_file(mapping_file: Optional[str]) -> Dict[str, str]:
    if not mapping_file:
        return {}
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"mapping file not found: {mapping_file}")

    delimiter = "\t" if mapping_file.lower().endswith((".tsv", ".txt")) else ","
    with open(mapping_file, "r", encoding="utf-8-sig", newline="") as fileobj:
        sample = fileobj.read(4096)
        fileobj.seek(0)
        has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
        reader = csv.reader(fileobj, delimiter=delimiter)
        rows = list(reader)

    if not rows:
        return {}

    if has_header:
        headers = [value.strip() for value in rows[0]]
        key_header = _find_column(headers, MAPPING_KEY_CANDIDATES)
        smiles_header = _find_column(headers, SMILES_COLUMN_CANDIDATES)
        if key_header is None or smiles_header is None:
            raise ValueError("Mapping file needs an ID/Name column and a SMILES column.")
        key_idx = headers.index(key_header)
        smiles_idx = headers.index(smiles_header)
        data_rows = rows[1:]
    else:
        key_idx, smiles_idx = 0, 1
        data_rows = rows

    mapping: Dict[str, str] = {}
    for row in data_rows:
        if len(row) <= max(key_idx, smiles_idx):
            continue
        key = normalize_compound_id(row[key_idx])
        smiles = row[smiles_idx].strip()
        if key and smiles:
            mapping[key] = smiles
    return mapping


def _build_id_to_smiles(metabolite_rows: List[Dict[str, str]],
                        mapping_file: Optional[str]) -> Dict[str, str]:
    id_to_smiles = _read_mapping_file(mapping_file)
    if id_to_smiles:
        return id_to_smiles

    if not metabolite_rows:
        return {}

    headers = metabolite_rows[0].keys()
    smiles_header = _find_column(headers, SMILES_COLUMN_CANDIDATES)
    if smiles_header is None:
        return {}

    result: Dict[str, str] = {}
    for row in metabolite_rows:
        compound_id = normalize_compound_id(row.get("ID", ""))
        smiles = row.get(smiles_header, "").strip()
        if compound_id and smiles:
            result[compound_id] = smiles
    return result


def _resolve_model_path(model_xlsx: Optional[str] = None) -> str:
    candidates = [
        model_xlsx,
        os.environ.get("MOO_RETRO_MODEL_XLSX"),
        DEFAULT_MODEL_XLSX,
        LOCAL_MODEL_XLSX,
    ]
    for candidate in candidates:
        if not candidate or not os.path.exists(candidate):
            continue
        try:
            with zipfile.ZipFile(candidate):
                pass
            return candidate
        except (OSError, zipfile.BadZipFile):
            continue
    raise FileNotFoundError(f"model_output.xlsx not found. Checked: {candidates}")


def _resolve_mapping_path(mapping_file: Optional[str] = None) -> Optional[str]:
    candidate = mapping_file or os.environ.get("MOO_RETRO_MODEL_MAPPING")
    return os.path.abspath(candidate) if candidate else None


def _load_model(model_xlsx: Optional[str] = None,
                mapping_file: Optional[str] = None) -> ModelData:
    path = os.path.abspath(_resolve_model_path(model_xlsx))
    mapping_path = _resolve_mapping_path(mapping_file)
    cache_key = (path, mapping_path)
    if cache_key in _MODEL_XLSX_CACHE:
        return _MODEL_XLSX_CACHE[cache_key]

    reaction_rows = _read_sheet_rows(path, "Reactions")
    metabolite_rows = _read_sheet_rows(path, "Metabolites")

    reactions: List[ModelReaction] = []
    for row in reaction_rows:
        equation = row.get("Equation", "").strip()
        reactants, products, reversible = _parse_equation(equation)
        if not reactants and not products:
            continue
        base = ModelReaction(
            reaction_id=row.get("ID", "").strip(),
            name=row.get("Name", "").strip(),
            equation=equation,
            reactants=reactants,
            products=products,
            reversible=reversible,
        )
        reactions.append(base)
        if reversible:
            reactions.append(ModelReaction(
                reaction_id=base.reaction_id,
                name=base.name,
                equation=base.equation,
                reactants=base.products,
                products=base.reactants,
                reversible=True,
                direction="reverse",
            ))

    producers: Dict[str, List[ModelReaction]] = defaultdict(list)
    for reaction in reactions:
        for product in reaction.products:
            producers[product].append(reaction)

    name_map = {
        normalize_compound_id(row.get("ID", "")): row.get("Name", "").strip()
        for row in metabolite_rows
        if row.get("ID", "").strip()
    }
    id_to_smiles = _build_id_to_smiles(metabolite_rows, mapping_path)
    smiles_to_id = {
        _canonical_smiles_key(smiles): compound_id
        for compound_id, smiles in id_to_smiles.items()
        if smiles
    }
    model_data = ModelData(
        producers=producers,
        name_map=name_map,
        id_to_smiles=id_to_smiles,
        smiles_to_id=smiles_to_id,
    )
    _MODEL_XLSX_CACHE[cache_key] = model_data
    return model_data


def _resolve_target(target: str, model_data: ModelData) -> str:
    target_id = normalize_compound_id(target)
    if target_id in model_data.name_map:
        return target_id

    smiles_key = _canonical_smiles_key(target)
    if smiles_key in model_data.smiles_to_id:
        return model_data.smiles_to_id[smiles_key]

    target_lower = str(target).strip().lower()
    for compound_id, name in model_data.name_map.items():
        if name.lower() == target_lower:
            return compound_id
    return target_id


def single_fn_model_xlsx(
    target_smile: str,
    building_blocks: List[str],
    model_xlsx: Optional[str] = None,
    mapping_file: Optional[str] = None,
) -> List[SingleStepResult]:
    model_data = _load_model(model_xlsx, mapping_file=mapping_file)
    target_id = _resolve_target(target_smile, model_data)
    reactions = model_data.producers.get(target_id, [])
    results: List[SingleStepResult] = []
    target_for_similarity = model_data.id_to_smiles.get(target_id, target_smile)
    target_is_smiles = Chem.MolFromSmiles(target_for_similarity) is not None

    if not reactions and not model_data.id_to_smiles:
        raise ValueError(
            "model_xlsx mode received SMILES-like targets/stock, but model_output.xlsx "
            "contains only metabolite IDs and has no SMILES column. Add a SMILES column "
            "to the Metabolites sheet or pass --model-mapping-file with ID,SMILES columns."
        )

    for idx, reaction in enumerate(reactions):
        reactant_ids = [item for item in reaction.reactants if item and item != target_id]
        reactants = [
            model_data.id_to_smiles.get(item, item)
            for item in reactant_ids
        ]
        if not reactants:
            continue
        if any(item not in model_data.id_to_smiles for item in reactant_ids) and model_data.id_to_smiles:
            continue
        # Some merged KEGG/MetaCyc rows contain generic structures (for
        # example [R1]) or database identifiers without a usable SMILES
        # mapping. Such values cannot become molecule nodes because the
        # evaluator and similarity functions require RDKit-valid SMILES.
        reactant_mols = [Chem.MolFromSmiles(reactant) for reactant in reactants]
        if any(mol is None for mol in reactant_mols):
            continue

        # The current MCTS implementation follows the first reaction child.
        # The primary substrate should preserve the target scaffold; choosing
        # by stock similarity can incorrectly prioritize large cofactors such
        # as NAD(P)H. Heavy-atom count keeps water/protons behind an organic
        # substrate when structural similarities tie.
        target_similarities = [
            max(
                similarity_calculator(reactant, [target_for_similarity]),
                default=0.0,
            ) if target_is_smiles else 0.0
            for reactant in reactants
        ]
        primary_idx = max(
            range(len(reactants)),
            key=lambda item: (
                target_similarities[item],
                reactant_mols[item].GetNumHeavyAtoms(),
            ),
        )
        if primary_idx:
            reactants.insert(0, reactants.pop(primary_idx))
        similarity = max(
            similarity_calculator(reactants[0], building_blocks),
            default=0.0,
        )
        score = round(1.0 / (idx + 1), 3)
        results.append(SingleStepResult(
            reactants=reactants,
            scores=score,
            templates=f"{reaction.route_id}: {reaction.equation}",
            ec=reaction.route_id,
            similarity=round(similarity, 3),
            feasibility=round(get_delta_sascore_for_feasibility(target_smile, reactants[0]), 3),
            yield_estimate=score,
            toxicity=0.0,
            top_k=f"Model-{idx + 1}",
        ))

    return results
