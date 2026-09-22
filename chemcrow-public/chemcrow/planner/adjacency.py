"""Lazy in-memory adjacency index for the local enzyme SQLite database."""

from __future__ import annotations

import sqlite3
import threading
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


class SQLiteAdjacencyIndex:
    """Load metabolite/reaction adjacency once, while keeping SQLite read-only."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._loaded = False
        self._metabolites: Dict[str, Mapping[str, Any]] = {}
        self._canonical_ids: Dict[str, Tuple[str, ...]] = {}
        self._reactions: Dict[str, Mapping[str, Any]] = {}
        self._participants: Dict[
            Tuple[str, str], Tuple[Mapping[str, Any], ...]
        ] = {}
        self._occurrences: Dict[str, Tuple[Tuple[str, str], ...]] = {}
        self.load_seconds = 0.0
        self.hits = 0

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            "file:" + self.db_path + "?mode=ro", uri=True, timeout=30.0
        )
        connection.row_factory = sqlite3.Row
        return connection

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            started = time.monotonic()
            with self._connect() as connection:
                metabolites = [
                    dict(row) for row in connection.execute("SELECT * FROM metabolites")
                ]
                reactions = [
                    dict(row) for row in connection.execute("SELECT * FROM reactions")
                ]
                participants = [
                    dict(row)
                    for row in connection.execute(
                        """
                        SELECT reaction_id, side, position, coefficient,
                               metabolite_id, annotation
                        FROM reaction_participants
                        ORDER BY reaction_id, side, position
                        """
                    )
                ]
            self._metabolites = {row["id"]: row for row in metabolites}
            canonical_ids: Dict[str, List[str]] = defaultdict(list)
            for row in metabolites:
                if row.get("canonical_smiles"):
                    canonical_ids[row["canonical_smiles"]].append(row["id"])
            self._canonical_ids = {
                key: tuple(values) for key, values in canonical_ids.items()
            }
            self._reactions = {row["id"]: row for row in reactions}
            grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
            occurrences: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
            for row in participants:
                grouped[(row["reaction_id"], row["side"])].append(row)
                occurrences[row["metabolite_id"]].append(
                    (row["reaction_id"], row["side"])
                )
            self._participants = {
                key: tuple(sorted(values, key=lambda item: item["position"]))
                for key, values in grouped.items()
            }
            self._occurrences = {
                key: tuple(dict.fromkeys(values)) for key, values in occurrences.items()
            }
            self.load_seconds = time.monotonic() - started
            self._loaded = True

    def matching_ids(
        self, canonical_smiles: str | None, metabolite_id: str | None
    ) -> Tuple[str, ...]:
        self.ensure_loaded()
        self.hits += 1
        identifiers = []
        if metabolite_id:
            identifiers.append(metabolite_id)
        if canonical_smiles:
            identifiers.extend(self._canonical_ids.get(canonical_smiles, ()))
        return tuple(dict.fromkeys(identifiers))

    def participants(
        self, reaction_id: str, side: str
    ) -> Tuple[Mapping[str, Any], ...]:
        self.ensure_loaded()
        self.hits += 1
        result = []
        for participant in self._participants.get((reaction_id, side), ()):
            metabolite = self._metabolites.get(participant["metabolite_id"])
            if metabolite is None:
                continue
            merged = dict(metabolite)
            merged["coefficient"] = participant["coefficient"]
            merged["annotation"] = participant["annotation"]
            result.append(merged)
        return tuple(result)

    def reaction_occurrences(
        self, metabolite_ids: Sequence[str]
    ) -> Tuple[Mapping[str, Any], ...]:
        self.ensure_loaded()
        self.hits += 1
        result = []
        seen = set()
        for metabolite_id in metabolite_ids:
            for reaction_id, side in self._occurrences.get(metabolite_id, ()):
                key = (reaction_id, side)
                if key in seen:
                    continue
                seen.add(key)
                reaction = self._reactions.get(reaction_id)
                if reaction is None:
                    continue
                value = dict(reaction)
                value["occurrence_side"] = side
                result.append(value)
        return tuple(result)

    def info(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "loaded": self._loaded,
            "metabolites": len(self._metabolites),
            "reactions": len(self._reactions),
            "reaction_sides": len(self._participants),
            "indexed_metabolites": len(self._occurrences),
            "hits": self.hits,
            "load_seconds": round(self.load_seconds, 6),
        }
