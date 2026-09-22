"""Persist one text transcript for every top-level ChemCrow execution."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler


class TranscriptCallbackHandler(BaseCallbackHandler):
    """Save questions, ReAct steps, observations, answers, and errors as text."""

    def __init__(
        self,
        result_dir: Optional[str] = None,
        now_factory: Optional[Callable[[], datetime]] = None,
    ):
        project_root = Path(__file__).resolve().parents[2]
        self.result_dir = Path(result_dir or project_root / "result").resolve()
        self._now = now_factory or datetime.now
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._write_lock = threading.Lock()
        self.last_path: Optional[str] = None

    @staticmethod
    def _run_key(run_id: Optional[UUID]) -> str:
        return str(run_id) if run_id is not None else "unknown"

    @staticmethod
    def _as_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except (TypeError, ValueError):
            return str(value)

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Any,
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Remember only a top-level AgentExecutor request."""
        if parent_run_id is not None:
            return
        if not isinstance(inputs, dict) or "input" not in inputs:
            return
        started_at = self._now()
        self._runs[self._run_key(run_id)] = {
            "started_at": started_at,
            "input": inputs.get("input", ""),
        }

    def _render_success(
        self,
        run_id: str,
        run: Dict[str, Any],
        outputs: Dict[str, Any],
        finished_at: datetime,
    ) -> str:
        lines = [
            "=" * 80,
            f"ChemCrow run ID: {run_id}",
            f"Started: {run['started_at'].isoformat(timespec='seconds')}",
            f"Finished: {finished_at.isoformat(timespec='seconds')}",
            "",
            "========== User input ==========",
            self._as_text(run.get("input", "")),
            "",
            "========== Intermediate steps ==========",
        ]

        steps = outputs.get("intermediate_steps", [])
        if not steps:
            lines.append("(no intermediate tool calls)")
        for index, step in enumerate(steps, start=1):
            try:
                action, observation = step
                lines.extend(
                    [
                        "",
                        f"--- Step {index} ---",
                        f"Tool: {getattr(action, 'tool', '')}",
                        "Action input:",
                        self._as_text(getattr(action, "tool_input", "")),
                        "LLM action log:",
                        self._as_text(getattr(action, "log", "")),
                        "Observation:",
                        self._as_text(observation),
                    ]
                )
            except (TypeError, ValueError):
                lines.extend(["", f"--- Step {index} ---", self._as_text(step)])

        lines.extend(
            [
                "",
                "========== Final answer ==========",
                self._as_text(outputs.get("output", outputs)),
                "",
            ]
        )
        return "\n".join(lines)

    def _render_error(
        self,
        run_id: str,
        run: Dict[str, Any],
        error: BaseException,
        finished_at: datetime,
    ) -> str:
        return "\n".join(
            [
                "=" * 80,
                f"ChemCrow run ID: {run_id}",
                f"Started: {run['started_at'].isoformat(timespec='seconds')}",
                f"Finished: {finished_at.isoformat(timespec='seconds')}",
                "",
                "========== User input ==========",
                self._as_text(run.get("input", "")),
                "",
                "========== Execution error ==========",
                f"{type(error).__name__}: {error}",
                "",
            ]
        )

    def _write(self, content: str, timestamp: datetime) -> None:
        self.result_dir.mkdir(parents=True, exist_ok=True)
        path = self.result_dir / f"{timestamp.strftime('%Y%m%d%H%M')}.txt"
        with self._write_lock:
            needs_separator = path.exists() and path.stat().st_size > 0
            with path.open("a", encoding="utf-8") as handle:
                if needs_separator:
                    handle.write("\n")
                handle.write(content)
        self.last_path = str(path)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is not None:
            return
        key = self._run_key(run_id)
        run = self._runs.pop(key, None)
        if run is None:
            return
        finished_at = self._now()
        self._write(self._render_success(key, run, outputs, finished_at), finished_at)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is not None:
            return
        key = self._run_key(run_id)
        run = self._runs.pop(key, None)
        if run is None:
            return
        finished_at = self._now()
        self._write(self._render_error(key, run, error, finished_at), finished_at)
