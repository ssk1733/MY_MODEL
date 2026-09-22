"""Local enzymatic reaction database tool for ChemCrow.

The tool is intentionally read-only.  It searches a SQLite database built from
Reactions/Metabolites worksheets and returns known enzymatic reactions that can
produce a requested target metabolite.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Dict, List, Optional, Tuple

from pathlib import Path
from langchain.tools import BaseTool
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

__all__ = ["EnzymeRetroDatabase"]

_METABOLITE_ID_RE = re.compile(r"^C\d{5}$", re.IGNORECASE)
_REACTION_ID_RE = re.compile(r"^R\d{5}$", re.IGNORECASE)


class EnzymeRetroDatabase(BaseTool):
    """Search a local database of known enzymatic reactions for retrosynthesis."""

    name = "EnzymeRetroDatabase"
    description = (
        "Search the local known enzymatic reaction database for retrosynthesis. "
        "Use this tool when the task asks for an enzymatic/biocatalytic synthesis "
        "or retrosynthetic pathway and you want known database reactions before or "
        "alongside a predictive retrosynthesis model. Input ONE target metabolite as "
        "a Cxxxxx database ID, an exact metabolite name, or a SMILES string. You may "
        "also input an Rxxxxx reaction ID to inspect that reaction. Returns known "
        "reaction IDs, equations, EC/Gene Rule, subsystem/pathway, reaction direction, "
        "and the precursor metabolites that can form the target. This tool searches "
        "the database only; it does not invent new reactions."
    )

    db_path: str = ""
    max_results: int = 8
    max_target_matches: int = 5

    def __init__(
        self,
        db_path: Optional[str] = None,
        max_results: int = 8,
        max_target_matches: int = 5,
        **kwargs,
    ):
        if db_path is None:
            db_path = str(Path(__file__).resolve().parents[1] / "data" / "enzyme_retro.db")
        super().__init__(
            db_path=db_path,
            max_results=max_results,
            max_target_matches=max_target_matches,
            **kwargs,
        )

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _canonical_smiles(smiles: str) -> Optional[str]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    @staticmethod
    def _id_origin(identifier: str) -> str:
        """Mark IDs from the reserved internal range used by this fused dataset."""
        m = re.match(r"^[CR](\d{5})$", identifier or "")
        if m and int(m.group(1)) >= 30000:
            return "internal unified ID (not an official KEGG accession)"
        return "KEGG-style/source ID"

    def _parse_query_input(self, raw: str) -> Tuple[str, int]:
        """Accept a plain query or a small JSON object from the agent."""
        text = (raw or "").strip()
        limit = self.max_results
        if text.startswith("{"):
            try:
                obj = json.loads(text)
                text = str(obj.get("query") or obj.get("target") or "").strip()
                requested = int(obj.get("max_results", limit))
                limit = max(1, min(requested, 20))
            except Exception:
                pass
        return text, limit

    def _resolve_metabolites(self, conn, query: str) -> Tuple[str, List[sqlite3.Row]]:
        # 1. Exact database ID.
        if _METABOLITE_ID_RE.match(query):
            rows = conn.execute(
                "SELECT * FROM metabolites WHERE id = ?", (query.upper(),)
            ).fetchall()
            return "exact ID", rows

        # 2. Exact name (case-insensitive). Do this before trying SMILES so ordinary
        # names do not generate RDKit parse warnings.
        rows = conn.execute(
            "SELECT * FROM metabolites WHERE name_norm = ? LIMIT ?",
            (query.casefold(), self.max_target_matches),
        ).fetchall()
        if rows:
            return "exact name", rows

        # 3. Exact canonical SMILES.
        canonical = self._canonical_smiles(query)
        if canonical:
            rows = conn.execute(
                "SELECT * FROM metabolites WHERE canonical_smiles = ? LIMIT ?",
                (canonical, self.max_target_matches),
            ).fetchall()
            if rows:
                return "canonical SMILES", rows

        return "not found", []

    def _name_suggestions(self, conn, query: str) -> List[sqlite3.Row]:
        if not query:
            return []
        escaped = query.replace("%", "\\%").replace("_", "\\_")
        return conn.execute(
            """
            SELECT id,name,formula,smiles
            FROM metabolites
            WHERE name IS NOT NULL AND name LIKE ? ESCAPE '\\'
            ORDER BY LENGTH(name), name
            LIMIT 8
            """,
            ("%" + escaped + "%",),
        ).fetchall()

    def _precursors(self, conn, reaction_id: str, side: str) -> List[sqlite3.Row]:
        return conn.execute(
            """
            SELECT rp.coefficient, rp.metabolite_id, rp.annotation,
                   m.name, m.formula, m.smiles
            FROM reaction_participants rp
            JOIN metabolites m ON m.id = rp.metabolite_id
            WHERE rp.reaction_id = ? AND rp.side = ?
            ORDER BY rp.position
            """,
            (reaction_id, side),
        ).fetchall()

    @staticmethod
    def _retrosynthetic_precursor_side(direction: str, target_side: str):
        if direction == "reversible":
            return ("left" if target_side == "right" else "right"), False
        if direction == "forward" and target_side == "right":
            return "left", False
        if direction == "reverse" and target_side == "left":
            return "right", False
        if direction == "unknown":
            return ("left" if target_side == "right" else "right"), True
        return None, False

    def _reaction_hits(self, conn, target_id: str, limit: int) -> List[Dict]:
        occurrences = conn.execute(
            """
            SELECT r.id, r.name, r.equation, r.lower_bound, r.upper_bound,
                   r.direction, r.gene_rule, r.subsystem, rp.side AS target_side
            FROM reaction_participants rp
            JOIN reactions r ON r.id = rp.reaction_id
            WHERE rp.metabolite_id = ?
            """,
            (target_id,),
        ).fetchall()

        hits = []
        seen = set()
        for r in occurrences:
            precursor_side, uncertain = self._retrosynthetic_precursor_side(
                r["direction"], r["target_side"]
            )
            if precursor_side is None:
                continue
            key = (r["id"], precursor_side)
            if key in seen:
                continue
            seen.add(key)
            precursors = self._precursors(conn, r["id"], precursor_side)
            hits.append(
                {
                    "row": r,
                    "precursor_side": precursor_side,
                    "precursors": precursors,
                    "uncertain": uncertain,
                }
            )

        # Prefer direction-resolved, enzyme-annotated, pathway-annotated reactions.
        hits.sort(
            key=lambda h: (
                h["uncertain"],
                0 if h["row"]["gene_rule"] else 1,
                0 if h["row"]["subsystem"] else 1,
                len(h["precursors"]),
                h["row"]["id"],
            )
        )
        return hits[:limit]

    def _format_metabolite(self, row: sqlite3.Row) -> str:
        parts = [row["id"]]
        if row["name"]:
            parts.append(str(row["name"]))
        if row["formula"]:
            parts.append("formula=" + str(row["formula"]))
        if row["smiles"]:
            parts.append("SMILES=" + str(row["smiles"]))
        return " | ".join(parts)

    def _format_precursor(self, row: sqlite3.Row) -> str:
        coeff = row["coefficient"]
        prefix = "" if coeff in (None, "", "1") else str(coeff) + " "
        label = prefix + row["metabolite_id"]
        if row["annotation"]:
            label += "(" + str(row["annotation"]) + ")"
        if row["name"]:
            label += " [" + str(row["name"]) + "]"
        return label

    def _inspect_reaction(self, conn, reaction_id: str) -> str:
        r = conn.execute(
            "SELECT * FROM reactions WHERE id = ?", (reaction_id.upper(),)
        ).fetchone()
        if r is None:
            return f"Reaction {reaction_id} was not found in the local enzymatic database."
        lines = [
            "Local enzymatic database reaction",
            f"ID: {r['id']} ({self._id_origin(r['id'])})",
            f"Name: {r['name'] or 'unknown'}",
            f"Equation: {r['equation']}",
            f"Direction from bounds: {r['direction']} "
            f"(LB={r['lower_bound']}, UB={r['upper_bound']})",
            f"EC/Gene Rule: {r['gene_rule'] or 'unknown'}",
            f"Subsystem/Pathway: {r['subsystem'] or 'unknown'}",
        ]
        return "\n".join(lines)

    def _run(self, query: str) -> str:
        query, limit = self._parse_query_input(query)
        if not query:
            return (
                "Please provide one target metabolite as Cxxxxx ID, exact name, "
                "or SMILES; or provide an Rxxxxx reaction ID."
            )

        conn = self._connect()
        try:
            if _REACTION_ID_RE.match(query):
                return self._inspect_reaction(conn, query)

            match_type, targets = self._resolve_metabolites(conn, query)
            if not targets:
                suggestions = self._name_suggestions(conn, query)
                if suggestions:
                    text = [
                        f"No exact target match for {query!r}. Possible metabolite names:"
                    ]
                    for row in suggestions:
                        text.append(
                            f"- {row['id']}: {row['name']}"
                            + (f"; formula={row['formula']}" if row['formula'] else "")
                        )
                    text.append(
                        "Call the tool again with one exact ID/name, or use a valid SMILES."
                    )
                    return "\n".join(text)
                return f"No metabolite matching {query!r} was found in the local enzymatic database."

            lines = [
                f"Local enzymatic retrosynthesis database search (match: {match_type}).",
                "Known database reactions only; these are not newly predicted reactions.",
            ]
            total_hits = 0
            remaining = limit
            for target in targets:
                if remaining <= 0:
                    break
                lines.append("")
                lines.append("Target: " + self._format_metabolite(target))
                lines.append("ID status: " + self._id_origin(target["id"]))
                hits = self._reaction_hits(conn, target["id"], remaining)
                if not hits:
                    lines.append("No direction-compatible producing reaction was found.")
                    continue
                for i, hit in enumerate(hits, start=1):
                    r = hit["row"]
                    precursors = ", ".join(
                        self._format_precursor(p) for p in hit["precursors"]
                    ) or "(empty side in source equation)"
                    lines.extend(
                        [
                            f"Reaction {total_hits + 1}: {r['id']} ({self._id_origin(r['id'])})",
                            f"  Name: {r['name'] or 'unknown'}",
                            f"  Stored equation: {r['equation']}",
                            f"  Direction: {r['direction']} (LB={r['lower_bound']}, UB={r['upper_bound']})"
                            + (" [source direction unknown; candidate only]" if hit["uncertain"] else ""),
                            f"  EC/Gene Rule: {r['gene_rule'] or 'unknown'}",
                            f"  Subsystem/Pathway: {r['subsystem'] or 'unknown'}",
                            f"  Retrosynthetic precursors for {target['id']}: {precursors}",
                        ]
                    )
                    total_hits += 1
                    remaining -= 1
                    if remaining <= 0:
                        break

            if total_hits == 0:
                lines.append("\nNo known producing reaction was found for the matched target(s).")
            return "\n".join(lines)
        except sqlite3.Error as exc:
            return f"Local enzymatic database error: {exc}"
        finally:
            conn.close()

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async search is not implemented for EnzymeRetroDatabase.")
