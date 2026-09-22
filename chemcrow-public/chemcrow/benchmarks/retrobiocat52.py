"""Agent benchmark utilities for the RetroBioCat 52 pathways.

Every case supplies a target SMILES and substrate SMILES to a tool-using Agent.
Gold enzyme classes remain hidden and are used only for evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import posixpath
import re
import statistics
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zipfile import ZipFile

from chemcrow.planner.chem_utils import canonicalize_smiles


SCHEMA_VERSION = "retrobiocat52.agent_target_substrates.v5"

# Conservative enzyme-class rules used only for comparing ChemCrow annotations
# (reaction name/template plus EC number) with the class labels in the source
# spreadsheet.  Ambiguous classes intentionally have no broad EC fallback.
ENZYME_CLASS_RULES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "AAD": {
        "aliases": ("AAD", "amino acid deaminase"),
        "ec_prefixes": ("1.4.3",),
    },
    "AADH": {
        "aliases": ("AADH", "amino acid dehydrogenase"),
        "ec_prefixes": ("1.4.1",),
    },
    "ADH": {
        "aliases": ("ADH", "alcohol dehydrogenase"),
        "ec_prefixes": ("1.1.1",),
    },
    "AlDH": {
        "aliases": ("AlDH", "aldehyde dehydrogenase"),
        "ec_prefixes": ("1.2.1",),
    },
    "AlOx": {
        "aliases": ("AlOx", "alcohol oxidase"),
        "ec_prefixes": ("1.1.3",),
    },
    "Aldolase": {
        "aliases": ("Aldolase",),
        "ec_prefixes": ("4.1.2",),
    },
    "AmDH": {
        "aliases": ("AmDH", "amine dehydrogenase"),
        "ec_prefixes": ("1.4.1",),
    },
    "AmOx": {
        "aliases": ("AmOx", "amine oxidase"),
        "ec_prefixes": ("1.4.3",),
    },
    "BVMO": {
        "aliases": ("BVMO", "Baeyer Villiger monooxygenase"),
        "ec_prefixes": ("1.14.13", "1.14.14"),
    },
    "CAR": {
        "aliases": ("CAR", "carboxylic acid reductase"),
        "ec_prefixes": ("1.2.1.30",),
    },
    "CMT": {
        "aliases": ("CMT", "carbon methyltransferase", "C methyltransferase"),
        "ec_prefixes": ("2.1.1",),
    },
    "DC": {
        "aliases": ("DC", "decarboxylase"),
        "ec_prefixes": ("4.1.1",),
    },
    "DERA Aldolase": {
        "aliases": ("DERA", "DERA Aldolase", "deoxyribose phosphate aldolase"),
        "ec_prefixes": ("4.1.2.4",),
    },
    "EH": {
        "aliases": ("EH", "epoxide hydrolase"),
        "ec_prefixes": ("3.3.2",),
    },
    "ERED": {
        "aliases": ("ERED", "ene reductase", "ene-reductase"),
        "ec_prefixes": ("1.3.1", "1.6.99"),
    },
    "Hydratase": {
        "aliases": ("Hydratase",),
        "ec_prefixes": ("4.2.1",),
    },
    "IRED": {
        "aliases": ("IRED", "imine reductase"),
        "ec_prefixes": ("1.5.1",),
    },
    "Kinase": {
        "aliases": ("Kinase",),
        "ec_prefixes": ("2.7",),
    },
    "Lipase": {
        "aliases": ("Lipase",),
        "ec_prefixes": ("3.1.1.3",),
    },
    "NIT": {
        "aliases": ("NIT", "nitrilase"),
        "ec_prefixes": ("3.5.5",),
    },
    "P450": {
        "aliases": ("P450", "cytochrome P450"),
        "ec_prefixes": ("1.14",),
    },
    "PAL": {
        "aliases": ("PAL", "phenylalanine ammonia lyase"),
        "ec_prefixes": ("4.3.1.24", "4.3.1.25"),
    },
    "PNP": {
        "aliases": ("PNP", "purine nucleoside phosphorylase"),
        "ec_prefixes": ("2.4.2.1",),
    },
    "PPM": {
        "aliases": ("PPM", "phosphopentomutase"),
        "ec_prefixes": ("5.4.2.7",),
    },
    "PSase": {
        "aliases": ("PSase", "Pictet Spenglerase", "Pictet-Spenglerase"),
        "ec_prefixes": (),
    },
    "SMO": {
        "aliases": ("SMO", "styrene monooxygenase"),
        "ec_prefixes": ("1.14.14.11",),
    },
    "TA": {
        "aliases": ("TA", "transaminase", "aminotransferase"),
        "ec_prefixes": ("2.6.1",),
    },
    "TAL": {
        "aliases": ("TAL", "tyrosine ammonia lyase"),
        "ec_prefixes": ("4.3.1.23",),
    },
    "TAM": {
        "aliases": ("TAM", "tyrosine aminomutase"),
        "ec_prefixes": ("5.4.3",),
    },
    "TDL": {
        "aliases": (
            "TDL",
            "ThDL",
            "thiamine dependent lyase",
            "thiamine-dependent lyase",
        ),
        "ec_prefixes": (),
    },
    "TPL": {
        "aliases": ("TPL", "tyrosine phenol lyase"),
        "ec_prefixes": ("4.1.99.2",),
    },
    "Threonine Aldolase": {
        "aliases": ("Threonine Aldolase",),
        "ec_prefixes": ("4.1.2.5", "4.1.2.48"),
    },
    "TrpS": {
        "aliases": ("TrpS", "tryptophan synthase"),
        "ec_prefixes": ("4.2.1.20",),
    },
    "XOR": {
        "aliases": ("XOR", "xanthine oxidoreductase"),
        "ec_prefixes": ("1.17.1.4", "1.17.3.2"),
    },
}


METRIC_NAMES = ("complete_route_found", "enzyme_all_present")


def _column_index(reference: str) -> int:
    value = 0
    for char in reference:
        if not char.isalpha():
            break
        value = value * 26 + ord(char.upper()) - 64
    return value - 1


def _workbook_rows(
    path: Path, sheet_name: Optional[str] = None
) -> List[Dict[str, str]]:
    """Read the simple tabular XLSX without adding an openpyxl dependency."""
    main = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    package_rel = {
        "r": "http://schemas.openxmlformats.org/package/2006/relationships"
    }
    document_rel = (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    )
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", main):
                shared_strings.append(
                    "".join(node.text or "" for node in item.iterfind(".//m:t", main))
                )

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
        relation_targets = {
            item.attrib["Id"]: item.attrib["Target"]
            for item in relationships.findall("r:Relationship", package_rel)
        }
        worksheets = list(workbook.find("m:sheets", main) or ())
        selected = next(
            (
                item
                for item in worksheets
                if sheet_name is None or item.attrib.get("name") == sheet_name
            ),
            None,
        )
        if selected is None:
            raise ValueError(f"Worksheet not found: {sheet_name!r}")
        relationship_id = selected.attrib[f"{{{document_rel}}}id"]
        target = relation_targets[relationship_id].lstrip("/")
        worksheet_path = (
            target
            if target.startswith("xl/")
            else posixpath.normpath(posixpath.join("xl", target))
        )
        worksheet = ET.fromstring(archive.read(worksheet_path))

        matrix: List[List[str]] = []
        maximum_column = 0
        sparse_rows: List[Dict[int, str]] = []
        for row in worksheet.findall(".//m:sheetData/m:row", main):
            values: Dict[int, str] = {}
            for cell in row.findall("m:c", main):
                index = _column_index(cell.attrib.get("r", "A1"))
                cell_type = cell.attrib.get("t")
                value = cell.findtext("m:v", default="", namespaces=main)
                if cell_type == "s" and value:
                    value = shared_strings[int(value)]
                elif cell_type == "inlineStr":
                    value = "".join(
                        node.text or ""
                        for node in cell.iterfind(".//m:t", main)
                    )
                values[index] = str(value)
                maximum_column = max(maximum_column, index + 1)
            if values:
                sparse_rows.append(values)
        for values in sparse_rows:
            matrix.append([values.get(index, "") for index in range(maximum_column)])
    if not matrix:
        return []
    headers = [value.strip() for value in matrix[0]]
    return [
        {headers[index]: row[index] for index in range(len(headers))}
        for row in matrix[1:]
        if any(value.strip() for value in row)
    ]


def _split_list(value: str) -> List[str]:
    return [
        item.strip()
        for item in re.split(r",\s+", str(value).strip())
        if item.strip()
    ]


def _integer_or_none(value: Any) -> Optional[int]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def load_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
    return records


def build_benchmark(
    input_path: Path | str,
    output_dir: Path | str,
    sheet_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert the source workbook into canonical JSONL and a manifest."""
    source = Path(input_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = _workbook_rows(source, sheet_name=sheet_name)
    required = {"Num", "Target Product", "Substrates", "Enzymes", "DOI"}
    if not rows or not required.issubset(rows[0]):
        missing = sorted(required - set(rows[0] if rows else ()))
        raise ValueError(f"Missing required workbook columns: {missing}")

    cases: List[Dict[str, Any]] = []
    all_enzyme_labels: Counter[str] = Counter()
    substrate_counts: Counter[int] = Counter()
    step_counts: Counter[int] = Counter()
    for row in rows:
        number = _integer_or_none(row["Num"])
        if number is None:
            raise ValueError(f"Invalid Num value: {row['Num']!r}")
        target_original = row["Target Product"].strip()
        target_canonical = canonicalize_smiles(target_original)
        if target_canonical is None:
            raise ValueError(f"Row {number}: invalid target SMILES {target_original!r}")
        substrate_originals = _split_list(row["Substrates"])
        building_blocks = []
        for value in substrate_originals:
            canonical = canonicalize_smiles(value)
            if canonical is None:
                raise ValueError(f"Row {number}: invalid substrate SMILES {value!r}")
            building_blocks.append(
                {"original_smiles": value, "canonical_smiles": canonical}
            )
        enzymes = _split_list(row["Enzymes"])
        unknown = [name for name in enzymes if name not in ENZYME_CLASS_RULES]
        if unknown:
            raise ValueError(f"Row {number}: enzyme class rules missing for {unknown}")
        case = {
            "schema_version": SCHEMA_VERSION,
            "benchmark_id": f"RBC52-{number:03d}",
            "source_row": number + 1,
            "input": {
                "target_smiles": target_canonical,
                "substrate_smiles": sorted(
                    {item["canonical_smiles"] for item in building_blocks}
                ),
            },
            "expected": {"enzyme_types": enzymes},
            "reference": {"doi": row["DOI"].strip()},
            "metadata": {
                "target_original_smiles": target_original,
                "substrate_original_smiles": substrate_originals,
            },
        }
        cases.append(case)
        all_enzyme_labels.update(enzymes)
        substrate_counts[len(building_blocks)] += 1
        step_counts[len(enzymes)] += 1

    cases.sort(key=lambda item: item["benchmark_id"])
    jsonl_path = output / "benchmark.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")


    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "name": "RetroBioCat 52-pathway test set adapted for ChemCrow",
        "case_count": len(cases),
        "source": {
            "file_name": source.name,
            "sha256": source_hash,
            "sheet": sheet_name or "first worksheet",
            "publication": "https://doi.org/10.1038/s41929-020-00556-z",
            "upstream_evaluator": (
                "https://github.com/willfinnigan/RetroBioCat/"
                "blob/master/scripts/pathway_testing/pathway_eval.py"
            ),
        },
        "files": {"benchmark_jsonl": jsonl_path.name},
        "statistics": {
            "unique_target_count": len(
                {case["input"]["target_smiles"] for case in cases}
            ),
            "substrate_count_distribution": {
                str(key): value for key, value in sorted(substrate_counts.items())
            },
            "step_count_distribution": {
                str(key): value for key, value in sorted(step_counts.items())
            },
            "enzyme_class_counts": dict(sorted(all_enzyme_labels.items())),
        },
        "ground_truth_scope": {
            "agent_input": [
                "target canonical SMILES",
                "literature starting-material canonical SMILES set",
            ],
            "evaluation_only": [
                "source-listed enzyme classes",
                "literature DOI",
            ],
            "unavailable": [
                "intermediate structures",
                "reaction atom mappings",
                "step-level EC numbers",
                "reaction conditions",
            ],
        },
        "metric_protocol": {
            "case_execution_order": "benchmark_id ascending (RBC52-001 through RBC52-052)",
            "agent_input": "target SMILES plus all substrate SMILES",
            "complete_route_found": (
                "a non-empty complete route must connect the target to exactly all "
                "supplied non-currency substrate leaves; missing or extra leaves fail"
            ),
            "enzyme_all_present": (
                "the same complete endpoint-correct route must contain every gold "
                "enzyme class; order is ignored and repeated classes require "
                "distinct predicted reaction steps"
            ),
            "top_k": "a case is correct at K when its first matching route rank <= K",
            "aggregate_complete_pathways": (
                "report X/N and X/N*100 percent for a complete endpoint-correct "
                "route found anywhere in all retained predictions"
            ),
            "aggregate_enzyme_complete_pathways": (
                "report X/N and X/N*100 percent where the same endpoint-correct "
                "complete route contains every source-listed enzyme class"
            ),
            "enzyme_top_k_reporting": (
                "report enzyme hit count X/N and accuracy at Top 1, Top 3, and Top 10"
            ),
            "enzyme_matching": (
                "conservative reaction-name/template aliases plus documented EC "
                "prefix rules; these are proxy class annotations, not gold EC labels"
            ),
            "intermediate_and_step_count_scoring": "not otherwise evaluated",
        },
        "enzyme_class_rules": ENZYME_CLASS_RULES,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _normal_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _molecule_smiles(value: Mapping[str, Any]) -> Optional[str]:
    alias = value.get("input_building_block")
    candidates: Iterable[Any]
    if isinstance(alias, Mapping):
        candidates = (
            alias.get("canonical_smiles"),
            alias.get("smiles"),
            value.get("canonical_smiles"),
            value.get("smiles"),
        )
    else:
        candidates = (value.get("canonical_smiles"), value.get("smiles"))
    for candidate in candidates:
        canonical = canonicalize_smiles(str(candidate)) if candidate else None
        if canonical:
            return canonical
    return None


def route_structure(route: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract canonical non-currency leaves from a complete route step graph."""
    steps = list(route.get("steps") or ())
    produced_keys = {
        str(step.get("target", {}).get("key"))
        for step in steps
        if step.get("target", {}).get("key")
    }
    leaves = set()
    for step in steps:
        for precursor in step.get("precursors", ()):
            if precursor.get("role") == "currency":
                continue
            smiles = _molecule_smiles(precursor)
            if smiles is not None and str(precursor.get("key")) not in produced_keys:
                leaves.add(smiles)
    return {"leaves": sorted(leaves), "step_count": len(steps)}




def _molecule_identity(value: Mapping[str, Any]) -> Optional[str]:
    """Return a stable graph identity, preferring canonical chemistry to DB keys."""
    smiles = _molecule_smiles(value)
    if smiles is not None:
        return f"smiles:{smiles}"
    key = value.get("key") or value.get("id") or value.get("name")
    return f"key:{key}" if key is not None and str(key).strip() else None


def validate_route_graph(
    route: Mapping[str, Any],
    target_smiles: str,
    expected_substrates: Sequence[str],
) -> Dict[str, Any]:
    """Independently validate a rooted AND route and its exact terminal set.

    This evaluator intentionally does not trust the planner's ``complete`` flag or
    its embedded ``route_validation`` record.  Only steps reachable from the
    supplied target are considered, and every supplied non-currency substrate
    must occur as a terminal leaf with no extra terminal molecules.
    """
    steps = list(route.get("steps") or ())
    target_canonical = canonicalize_smiles(str(target_smiles))
    expected = {
        canonical
        for value in expected_substrates
        if (canonical := canonicalize_smiles(str(value))) is not None
    }
    target_identity = (
        f"smiles:{target_canonical}" if target_canonical is not None else None
    )
    required_identities = {f"smiles:{value}" for value in expected}

    producers: Dict[str, List[Tuple[int, Mapping[str, Any]]]] = {}
    for index, step in enumerate(steps):
        identity = _molecule_identity(step.get("target") or {})
        if identity is not None:
            producers.setdefault(identity, []).append((index, step))
    duplicate_targets = sorted(
        identity for identity, values in producers.items() if len(values) != 1
    )

    reachable_steps: set[int] = set()
    terminal_identities: set[str] = set()
    cycle_paths: List[List[str]] = []

    def visit(identity: Optional[str], stack: Tuple[str, ...]) -> None:
        if identity is None:
            terminal_identities.add("invalid:missing_molecule_identity")
            return
        if identity in stack:
            start = stack.index(identity)
            cycle_paths.append(list(stack[start:] + (identity,)))
            return
        if identity in required_identities:
            terminal_identities.add(identity)
            return
        values = producers.get(identity, ())
        if not values:
            terminal_identities.add(identity)
            return
        if len(values) != 1:
            return
        index, step = values[0]
        if index in reachable_steps:
            return
        reachable_steps.add(index)
        next_stack = stack + (identity,)
        for precursor in step.get("precursors", ()):
            if precursor.get("role") == "currency":
                continue
            visit(_molecule_identity(precursor), next_stack)

    visit(target_identity, ())
    disconnected_indices = [
        index for index in range(len(steps)) if index not in reachable_steps
    ]
    root_resolved = bool(
        target_identity is not None
        and (
            target_identity in required_identities
            or len(producers.get(target_identity, ())) == 1
        )
    )
    connected = not disconnected_indices
    acyclic = not cycle_paths
    endpoint_match = terminal_identities == required_identities
    marked_complete = route.get("complete") is True
    valid = bool(
        marked_complete
        and steps
        and root_resolved
        and not duplicate_targets
        and connected
        and acyclic
        and endpoint_match
    )

    reasons = []
    if not marked_complete:
        reasons.append("route_not_marked_complete")
    if not steps:
        reasons.append("empty_route")
    if not root_resolved:
        reasons.append("target_not_resolved")
    if duplicate_targets:
        reasons.append("multiple_producers_for_molecule")
    if not connected:
        reasons.append("disconnected_steps")
    if not acyclic:
        reasons.append("route_cycle")
    if not endpoint_match:
        reasons.append("terminal_substrate_mismatch")

    predicted = sorted(
        identity[len("smiles:") :]
        for identity in terminal_identities
        if identity.startswith("smiles:")
    )
    return {
        "valid": valid,
        "marked_complete": marked_complete,
        "root_resolved": root_resolved,
        "connected": connected,
        "acyclic": acyclic,
        "endpoint_match": endpoint_match,
        "target": target_canonical,
        "expected_substrates": sorted(expected),
        "predicted_substrates": predicted,
        "missing_substrates": sorted(
            value for value in expected if f"smiles:{value}" not in terminal_identities
        ),
        "extra_terminal_identities": sorted(
            terminal_identities - required_identities
        ),
        "reachable_step_count": len(reachable_steps),
        "step_count": len(steps),
        "duplicate_target_identities": duplicate_targets,
        "disconnected_step_indices": disconnected_indices,
        "cycle_paths": cycle_paths,
        "reasons": reasons,
    }


def _route_enzyme_steps(route: Mapping[str, Any]) -> List[Dict[str, Any]]:
    direct = route.get("enzymes", route.get("enzyme_classes"))
    if direct is not None:
        values = [direct] if isinstance(direct, (str, Mapping)) else list(direct)
        return [
            dict(value)
            if isinstance(value, Mapping)
            else {"reaction_name": str(value), "template": str(value)}
            for value in values
        ]
    records = list(route.get("retrosynthesis_reactions") or ())
    if records:
        return [dict(record.get("enzyme") or {}) for record in records]
    # Older route results contain forward/topological steps only. Reversing is a
    # reasonable fallback for linear routes; new results use the branch-safe field.
    output = []
    for step in reversed(list(route.get("steps") or ())):
        output.append(
            {
                "ec_numbers": list(step.get("ec_numbers") or ()),
                "reaction_name": step.get("reaction_name"),
                "template": step.get("template"),
            }
        )
    return output


def _enzyme_evidence(
    enzyme_class: str, annotation: Mapping[str, Any]
) -> Optional[Dict[str, str]]:
    rule = ENZYME_CLASS_RULES.get(enzyme_class)
    if rule is None:
        return None
    text = " ".join(
        _normal_text(annotation.get(field))
        for field in ("enzyme_class", "reaction_name", "template")
        if annotation.get(field)
    )
    for alias in rule["aliases"]:
        normalized_alias = _normal_text(alias)
        if normalized_alias and re.search(
            rf"(?<![a-z0-9]){re.escape(normalized_alias)}(?![a-z0-9])", text
        ):
            return {"method": "text_alias", "value": alias}
    for ec_number in annotation.get("ec_numbers") or ():
        normalized_ec = str(ec_number).replace("EC", "").strip()
        for prefix in rule["ec_prefixes"]:
            if normalized_ec == prefix or normalized_ec.startswith(prefix + "."):
                return {"method": "ec_prefix", "value": str(ec_number)}
    return None


def enzyme_class_comparison(
    gold_sequence: Sequence[str], route: Mapping[str, Any]
) -> Dict[str, Any]:
    annotations = _route_enzyme_steps(route)
    adjacency: List[List[Tuple[int, Dict[str, str]]]] = []
    for enzyme_class in gold_sequence:
        matches = []
        for step_index, annotation in enumerate(annotations):
            evidence = _enzyme_evidence(enzyme_class, annotation)
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

    for gold_index in range(len(gold_sequence)):
        assign(gold_index, set())
    assigned_gold = {gold: step for step, gold in assigned_step.items()}
    evidence_records = []
    for gold_index, enzyme_class in enumerate(gold_sequence):
        step_index = assigned_gold.get(gold_index)
        evidence = None
        if step_index is not None:
            evidence = next(
                item for index, item in adjacency[gold_index] if index == step_index
            )
        evidence_records.append(
            {
                "gold_index": gold_index,
                "gold_enzyme_class": enzyme_class,
                "matched_step_index": step_index,
                "evidence": evidence,
            }
        )
    matched_count = len(assigned_gold)
    return {
        "gold_sequence": list(gold_sequence),
        "predicted_step_count": len(annotations),
        "matched_count": matched_count,
        "recall": (
            round(matched_count / len(gold_sequence), 6) if gold_sequence else 1.0
        ),
        "set_match": matched_count == len(gold_sequence),
        "matches": evidence_records,
    }


def evaluate_route(case: Mapping[str, Any], route: Mapping[str, Any]) -> Dict[str, Any]:
    expected_values = (
        case.get("input", {}).get("substrate_smiles")
        or case.get("expected", {}).get("substrate_smiles")
        or [
            item["canonical_smiles"]
            for item in case.get("building_blocks", ())
        ]
    )
    target_smiles = (
        case.get("input", {}).get("target_smiles")
        or case.get("target", {}).get("canonical_smiles")
        or case.get("target_smiles")
        or ""
    )
    graph_validation = validate_route_graph(
        route, str(target_smiles), [str(value) for value in expected_values]
    )
    expected = set(graph_validation["expected_substrates"])
    predicted_leaves = set(graph_validation["predicted_substrates"])
    gold_enzymes = list(
        case.get("expected", {}).get("enzyme_types")
        or case.get("gold", {}).get("retrosynthesis_enzyme_types", ())
    )
    enzyme = enzyme_class_comparison(gold_enzymes, route)
    complete_route_found = bool(graph_validation["valid"])
    metrics = {
        "complete_route_found": complete_route_found,
        "enzyme_all_present": complete_route_found and bool(enzyme["set_match"]),
    }
    return {
        "rank": int(route.get("rank", 0) or 0),
        "complete": route.get("complete") is True,
        "metrics": metrics,
        "expected_substrates": sorted(expected),
        "predicted_substrates": sorted(predicted_leaves),
        "step_count": graph_validation["step_count"],
        "route_validation": graph_validation,
        "enzyme_comparison": enzyme,
        "enzyme_annotated_retrosynthesis": route.get(
            "enzyme_annotated_retrosynthesis"
        ),
        "route_score": route.get("route_score"),
    }


def evaluate_case_result(
    case: Mapping[str, Any], agent_result: Mapping[str, Any]
) -> Dict[str, Any]:
    routes = list(agent_result.get("routes", ()))
    route_evaluations = []
    for index, route in enumerate(routes, start=1):
        normalized_route = dict(route)
        normalized_route.setdefault("rank", index)
        route_evaluations.append(evaluate_route(case, normalized_route))
    route_evaluations.sort(key=lambda item: item["rank"])
    first_hit_rank: Dict[str, Optional[int]] = {}
    for metric in METRIC_NAMES:
        ranks = [
            item["rank"]
            for item in route_evaluations
            if item["metrics"].get(metric)
        ]
        first_hit_rank[metric] = min(ranks) if ranks else None
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": case["benchmark_id"],
        "ok": bool(agent_result.get("ok")),
        "error": agent_result.get("error"),
        "route_count": len(routes),
        "first_hit_rank": first_hit_rank,
        "route_evaluations": route_evaluations,
        "agent_stats": dict(agent_result.get("stats") or {}),
    }


def _accuracy(first_ranks: Sequence[Optional[int]], top_k: int) -> float:
    if not first_ranks:
        return 0.0
    return round(
        sum(rank is not None and rank <= top_k for rank in first_ranks)
        / len(first_ranks),
        6,
    )


def summarize_case_results(
    results: Sequence[Mapping[str, Any]], top_ks: Sequence[int]
) -> Dict[str, Any]:
    top_ks = sorted({int(value) for value in top_ks if int(value) > 0})
    summary: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "case_count": len(results),
        "successful_tool_calls": sum(bool(item.get("ok")) for item in results),
        "error_count": sum(not bool(item.get("ok")) for item in results),
        "top_k_values": top_ks,
        "accuracy_at_k": {},
        "mean_reciprocal_rank": {},
        "hit_count_at_k": {},
    }
    for metric in METRIC_NAMES:
        ranks = [item.get("first_hit_rank", {}).get(metric) for item in results]
        summary["accuracy_at_k"][metric] = {
            str(top_k): _accuracy(ranks, top_k) for top_k in top_ks
        }
        reciprocal = [1.0 / rank if rank else 0.0 for rank in ranks]
        summary["hit_count_at_k"][metric] = {
            str(top_k): sum(
                rank is not None and rank <= top_k for rank in ranks
            )
            for top_k in top_ks
        }
        summary["mean_reciprocal_rank"][metric] = round(
            statistics.fmean(reciprocal) if reciprocal else 0.0, 6
        )

    elapsed = [
        float(item.get("wall_seconds"))
        for item in results
        if item.get("wall_seconds") is not None
        and math.isfinite(float(item.get("wall_seconds")))
    ]
    summary["runtime"] = {
        "total_seconds": round(sum(elapsed), 6),
        "mean_seconds": round(statistics.fmean(elapsed), 6) if elapsed else None,
        "median_seconds": round(statistics.median(elapsed), 6) if elapsed else None,
    }

    summary["pathway_accuracy_at_k"] = summary["accuracy_at_k"][
        "complete_route_found"
    ]
    summary["enzyme_accuracy_at_k"] = summary["accuracy_at_k"][
        "enzyme_all_present"
    ]

    total = len(results)
    pathway_hits = sum(
        item.get("first_hit_rank", {}).get("complete_route_found") is not None
        for item in results
    )
    enzyme_hits = sum(
        item.get("first_hit_rank", {}).get("enzyme_all_present") is not None
        for item in results
    )

    def outcome(success_count: int) -> Dict[str, Any]:
        return {
            "success_count": success_count,
            "total_count": total,
            "fraction": f"{success_count}/{total}",
            "success_rate": round(success_count / total, 6) if total else 0.0,
        }

    # Overall means a hit anywhere in all routes retained by the evaluator
    # (normally the returned Top 10). Top-K fields use the first matching rank.
    summary["complete_pathway_success"] = outcome(pathway_hits)
    summary["enzyme_complete_pathway_success"] = outcome(enzyme_hits)
    summary["pathway_top_k"] = {
        str(top_k): outcome(
            summary["hit_count_at_k"]["complete_route_found"][str(top_k)]
        )
        for top_k in top_ks
    }
    summary["enzyme_top_k"] = {
        str(top_k): outcome(
            summary["hit_count_at_k"]["enzyme_all_present"][str(top_k)]
        )
        for top_k in top_ks
    }
    return summary

