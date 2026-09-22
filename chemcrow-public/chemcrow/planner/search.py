"""AND-OR search policies for multi-step enzyme retrosynthesis."""

from __future__ import annotations

import heapq
import itertools
import logging
import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .chem_utils import heavy_atom_count, structural_similarity
from .config import PlannerConfig
from .database_service import EnzymeReactionDatabase
from .model_client import EnzRetroClient
from .models import ForwardAnchor, MoleculeNode, ReactionCandidate, RouteState
from .route_evaluation import evaluate_route_carbon, production_quality_score
from .scoring import fuse_candidates, score_and_rank_candidates

LOGGER = logging.getLogger(__name__)


_FORWARD_ANCHOR_CACHE: Dict[
    Tuple[Any, ...],
    Tuple[Dict[str, ForwardAnchor], Dict[str, Tuple[ForwardAnchor, ...]]],
] = {}
_FORWARD_ANCHOR_CACHE_LOCK = threading.RLock()


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)



@dataclass
class _MCTSNode:
    """One MCTS state over an AND-OR retrosynthesis hypergraph."""

    state: RouteState
    parent: Optional["_MCTSNode"] = None
    prior: float = 0.0
    incoming_candidate_id: Optional[str] = None
    children: List["_MCTSNode"] = field(default_factory=list)
    pending_states: List[Tuple[float, str, RouteState]] = field(default_factory=list)
    prepared: bool = False
    terminal: bool = False
    dead_end: bool = False
    visits: int = 0
    value_sum: float = 0.0
    best_value: float = 0.0
    tree_depth: int = 0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

class HybridPlannerEngine:
    """Run all multi-step graph traversal inside deterministic Python code."""

    def __init__(
        self,
        database: Optional[EnzymeReactionDatabase] = None,
        model_client: Optional[EnzRetroClient] = None,
        config: Optional[PlannerConfig] = None,
    ):
        self.config = config or PlannerConfig()
        self.database = database or EnzymeReactionDatabase()
        self.model_client = model_client
        if self.config.use_model and self.model_client is None:
            self.model_client = EnzRetroClient()
        self._trace: List[Dict[str, Any]] = []
        self._stats: Dict[str, Any] = {}
        self._forward_alternatives: Dict[str, Tuple[ForwardAnchor, ...]] = {}

    def _event(self, event: str, **values: Any) -> None:
        record = {"event": event, **values}
        if self.config.include_search_trace:
            self._trace.append(record)
        LOGGER.debug("enzyme planner event=%s values=%s", event, values)

    @staticmethod
    def _merge_steps(
        groups: Sequence[Sequence[ReactionCandidate]],
    ) -> Tuple[ReactionCandidate, ...]:
        merged: List[ReactionCandidate] = []
        seen = set()
        for group in groups:
            for step in group:
                if step.candidate_id not in seen:
                    seen.add(step.candidate_id)
                    merged.append(step)
        return tuple(merged)

    def _filter_candidates(
        self,
        candidates: Sequence[ReactionCandidate],
        *,
        state: Optional[RouteState] = None,
    ) -> Tuple[ReactionCandidate, ...]:
        accepted = []
        expanded_products = (
            {step.target.key for step in state.steps} if state is not None else set()
        )
        used_reactions = (
            {
                reaction_id
                for step in state.steps
                for reaction_id in step.database_reaction_ids
            }
            if state is not None
            else set()
        )
        for candidate in candidates:
            noncurrency = candidate.noncurrency_precursors
            reason = None
            if not noncurrency:
                reason = "no_noncurrency_precursor"
            elif len(noncurrency) > self.config.max_precursors_per_step:
                reason = "too_many_precursors"
            elif candidate.target.key in {item.key for item in noncurrency}:
                reason = "self_cycle"
            elif self.config.require_ec and not candidate.ec_numbers:
                reason = "missing_ec"
            elif (
                self.config.reject_generic_structures
                and any(item.generic for item in noncurrency)
            ):
                reason = "generic_structure"
            elif (
                candidate.source in {"MODEL", "BOTH"}
                and candidate.model_score is not None
                and candidate.model_score < self.config.min_model_score
            ):
                reason = "low_model_score"
            elif (
                candidate.source in {"MODEL", "BOTH"}
                and candidate.feasibility is not None
                and candidate.feasibility < self.config.min_feasibility
            ):
                reason = "low_feasibility"
            elif (
                candidate.atom_conservation is not None
                and candidate.atom_conservation < self.config.min_atom_conservation
            ):
                reason = "low_atom_conservation"
            elif (
                candidate.approx_structural_overlap is not None
                and candidate.approx_structural_overlap
                < self.config.min_approx_structural_overlap
            ):
                reason = "low_approx_structural_overlap"
            elif (
                candidate.toxicity is not None
                and candidate.toxicity > self.config.max_toxicity
            ):
                reason = "toxicity_threshold"
            elif (
                candidate.step_score is not None
                and candidate.step_score < self.config.min_candidate_score
            ):
                reason = "low_candidate_score"
            elif state is not None and any(
                item.key in expanded_products for item in noncurrency
            ):
                reason = "route_cycle"
            elif state is not None and used_reactions.intersection(
                candidate.database_reaction_ids
            ):
                reason = "repeated_reaction"
            if reason is None:
                accepted.append(candidate)
            else:
                self._stats["candidates_filtered"] += 1
                self._stats["filter_reasons"][reason] = (
                    self._stats["filter_reasons"].get(reason, 0) + 1
                )
        return tuple(accepted)

    def _candidate_batch(
        self,
        target: MoleculeNode,
        building_blocks: Sequence[MoleculeNode],
        *,
        state: Optional[RouteState] = None,
    ) -> Tuple[ReactionCandidate, ...]:
        database_candidates: Tuple[ReactionCandidate, ...] = ()
        model_candidates: Tuple[ReactionCandidate, ...] = ()
        if self.config.use_database:
            self._stats["backward_queries"] += 1
            database_candidates = self.database.backward_candidates(
                target,
                limit=self.config.database_top_k,
                respect_direction=self.config.respect_reaction_direction,
                allow_unknown_direction=self.config.allow_unknown_direction,
                currency_ids=self.config.currency_metabolite_ids,
            )
            self._stats["database_candidates"] += len(database_candidates)
        if (
            self.config.use_model
            and self.model_client is not None
            and (
                self._stats["model_calls"] + self._stats["model_failures"]
                < self.config.max_model_calls
            )
        ):
            try:
                cache_before = self.model_client.cache_info()
                self._stats["model_requests"] += 1
                model_candidates = self.model_client.predict_candidates(
                    target,
                    building_blocks,
                    top_k=self.config.model_top_k,
                    currency_ids=self.config.currency_metabolite_ids,
                    normalizer=self.database.normalize_molecule,
                )
                self._stats["model_candidates"] += len(model_candidates)
                cache_after = self.model_client.cache_info()
                self._stats["model_calls"] += max(
                    0, cache_after["misses"] - cache_before["misses"]
                )
                self._stats["model_cache_hits"] += max(
                    0, cache_after["hits"] - cache_before["hits"]
                )
            except Exception as exc:
                self._stats["model_failures"] += 1
                self._stats["warnings"].append(str(exc))
                self._event("model_failure", target=target.key, error=str(exc))
        combined = fuse_candidates(database_candidates, model_candidates, self.config)
        self._stats["fusion_candidates"] += len(combined)
        self._stats["consensus_candidates"] += sum(
            candidate.source == "BOTH" for candidate in combined
        )
        scored = score_and_rank_candidates(combined, building_blocks, self.config)
        scored = self._filter_candidates(scored, state=state)
        return scored[: self.config.candidate_beam_width]

    def _forward_anchor_cache_key(
        self,
        building_blocks: Sequence[MoleculeNode],
        target_key: str,
    ) -> Tuple[Any, ...]:
        weights = self.config.scoring_weights.to_dict()
        return (
            self.database.db_path,
            target_key,
            tuple(sorted(item.key for item in building_blocks)),
            self.config.forward_anchor_depth,
            self.config.forward_anchor_max_nodes,
            self.config.database_top_k,
            self.config.candidate_beam_width,
            self.config.global_beam_width,
            self.config.max_forward_routes_per_node,
            self.config.respect_reaction_direction,
            self.config.allow_unknown_direction,
            self.config.min_candidate_score,
            self.config.route_length_penalty,
            tuple(sorted(self.config.currency_metabolite_ids)),
            tuple(sorted(weights.items())),
        )

    def _store_forward_anchor_cache(
        self,
        cache_key: Tuple[Any, ...],
        anchors: Mapping[str, ForwardAnchor],
    ) -> None:
        if not self.config.enable_forward_anchor_cache:
            return
        with _FORWARD_ANCHOR_CACHE_LOCK:
            _FORWARD_ANCHOR_CACHE[cache_key] = (
                dict(anchors),
                dict(self._forward_alternatives),
            )
            while len(_FORWARD_ANCHOR_CACHE) > self.config.forward_anchor_cache_size:
                _FORWARD_ANCHOR_CACHE.pop(next(iter(_FORWARD_ANCHOR_CACHE)))

    def _build_forward_anchors(
        self,
        building_blocks: Sequence[MoleculeNode],
        target_key: str,
    ) -> Dict[str, ForwardAnchor]:
        """Run and cache a bounded forward hypergraph closure of database edges."""
        cache_key = self._forward_anchor_cache_key(building_blocks, target_key)
        if self.config.enable_forward_anchor_cache:
            with _FORWARD_ANCHOR_CACHE_LOCK:
                cached = _FORWARD_ANCHOR_CACHE.get(cache_key)
            if cached is not None:
                anchors, alternatives = cached
                self._forward_alternatives = dict(alternatives)
                self._stats["forward_anchor_cache_hits"] += 1
                self._stats["forward_anchor_count"] = len(anchors)
                self._stats["forward_alternative_count"] = sum(
                    len(values) for values in alternatives.values()
                )
                return dict(anchors)
        self._stats["forward_anchor_cache_misses"] += int(
            self.config.enable_forward_anchor_cache
        )

        anchors: Dict[str, ForwardAnchor] = {
            item.key: ForwardAnchor(item, (), 0.0) for item in building_blocks
        }
        alternatives: Dict[str, List[ForwardAnchor]] = {
            item.key: [ForwardAnchor(item, (), 0.0)] for item in building_blocks
        }
        self._stats["forward_anchor_count"] = len(anchors)
        if not self.config.use_database or self.config.forward_anchor_depth == 0:
            self._forward_alternatives = {
                key: tuple(values) for key, values in alternatives.items()
            }
            self._stats["forward_alternative_count"] = len(building_blocks)
            self._store_forward_anchor_cache(cache_key, anchors)
            return anchors

        queue: List[Tuple[float, int, str]] = []
        counter = itertools.count()
        for item in building_blocks:
            heapq.heappush(queue, (0.0, next(counter), item.key))
        pending: Dict[str, ReactionCandidate] = {}
        known_candidates: Dict[str, ReactionCandidate] = {}
        updates = 0

        def trim_alternatives(key: str) -> None:
            bucket = alternatives[key]
            ordered = sorted(
                bucket,
                key=lambda value: (
                    value.cost,
                    self._route_fingerprint(value.steps),
                ),
            )
            chain_best: Dict[Tuple[str, ...], ForwardAnchor] = {}
            for value in ordered:
                chain = tuple(
                    step.target.key
                    for step in self._ordered_steps(value.molecule.key, value.steps)
                )
                chain_best.setdefault(chain, value)
            selected = sorted(
                chain_best.values(),
                key=lambda value: (
                    value.cost,
                    self._route_fingerprint(value.steps),
                ),
            )[: self.config.max_forward_routes_per_node]
            selected_fingerprints = {
                self._route_fingerprint(value.steps) for value in selected
            }
            if len(selected) < self.config.max_forward_routes_per_node:
                for value in ordered:
                    fingerprint = self._route_fingerprint(value.steps)
                    if fingerprint in selected_fingerprints:
                        continue
                    selected.append(value)
                    selected_fingerprints.add(fingerprint)
                    if len(selected) >= self.config.max_forward_routes_per_node:
                        break
            bucket[:] = selected

        def record_alternative(candidate: ReactionCandidate, steps, cost: float) -> None:
            proposed = ForwardAnchor(candidate.target, steps, cost)
            bucket = alternatives.setdefault(candidate.target.key, [])
            fingerprint = self._route_fingerprint(steps)
            if all(self._route_fingerprint(item.steps) != fingerprint for item in bucket):
                bucket.append(proposed)
                trim_alternatives(candidate.target.key)

        def try_candidate(candidate: ReactionCandidate) -> bool:
            noncurrency = candidate.noncurrency_precursors
            if not noncurrency or any(item.key not in anchors for item in noncurrency):
                return False
            precursor_anchors = [anchors[item.key] for item in noncurrency]
            steps = self._merge_steps(
                [anchor.steps for anchor in precursor_anchors] + [(candidate,)]
            )
            if len(steps) > self.config.forward_anchor_depth:
                return False
            cost = sum(float(step.step_cost or 0.0) for step in steps)
            cost += self.config.route_length_penalty * len(steps)
            record_alternative(candidate, steps, cost)
            current = anchors.get(candidate.target.key)
            if current is not None and current.cost <= cost + 1.0e-12:
                return False
            anchors[candidate.target.key] = ForwardAnchor(
                candidate.target, steps, cost
            )
            heapq.heappush(queue, (cost, next(counter), candidate.target.key))
            self._event(
                "forward_anchor",
                molecule=candidate.target.key,
                depth=len(steps),
                reaction=candidate.candidate_id,
            )
            return True

        while queue and updates < self.config.forward_anchor_max_nodes:
            cost, _, key = heapq.heappop(queue)
            anchor = anchors.get(key)
            if anchor is None or cost > anchor.cost + 1.0e-12:
                continue
            if len(anchor.steps) >= self.config.forward_anchor_depth:
                continue
            source = anchor.molecule
            self._stats["forward_queries"] += 1
            raw = self.database.forward_candidates(
                source,
                limit=self.config.database_top_k,
                respect_direction=self.config.respect_reaction_direction,
                allow_unknown_direction=self.config.allow_unknown_direction,
                currency_ids=self.config.currency_metabolite_ids,
            )
            scored = score_and_rank_candidates(raw, building_blocks, self.config)
            scored = self._filter_candidates(scored)
            for candidate in scored[: self.config.candidate_beam_width]:
                pending[candidate.candidate_id] = candidate
                known_candidates[candidate.candidate_id] = candidate
            changed = True
            while changed and updates < self.config.forward_anchor_max_nodes:
                changed = False
                for candidate_id, candidate in list(pending.items()):
                    if try_candidate(candidate):
                        updates += 1
                        changed = True
                    if candidate.target.key in anchors and all(
                        item.key in anchors for item in candidate.noncurrency_precursors
                    ):
                        pending.pop(candidate_id, None)
            if len(anchors) > self.config.global_beam_width * (
                self.config.forward_anchor_depth + 1
            ):
                # Keep the closure bounded while preserving supplied leaves and
                # the requested target if it has already been reached.
                retained = sorted(
                    anchors.values(), key=lambda value: (value.cost, value.molecule.key)
                )[: self.config.global_beam_width * (self.config.forward_anchor_depth + 1)]
                block_keys = {item.key for item in building_blocks}
                if target_key in anchors:
                    retained.append(anchors[target_key])
                anchors = {
                    value.molecule.key: value
                    for value in retained
                    if value.molecule.key in block_keys or value.steps
                }
                for item in building_blocks:
                    anchors[item.key] = ForwardAnchor(item, (), 0.0)

        # The best-anchor closure above discovers reachable reaction edges.
        # Recombine bounded alternative precursor routes over that fixed graph so
        # later-arriving paths (for example pyruvate -> acetaldehyde) are not lost
        # merely because another route reached the same metabolite first.
        combination_limit = max(2, self.config.max_forward_routes_per_node * 2)
        for _ in range(self.config.forward_anchor_depth + 1):
            alternatives_changed = False
            for candidate in known_candidates.values():
                noncurrency = candidate.noncurrency_precursors
                if not noncurrency or any(
                    item.key not in alternatives for item in noncurrency
                ):
                    continue
                precursor_buckets = [
                    alternatives[item.key] for item in noncurrency
                ]
                combinations = itertools.islice(
                    itertools.product(*precursor_buckets), combination_limit
                )
                for combination in combinations:
                    steps = self._merge_steps(
                        [anchor.steps for anchor in combination] + [(candidate,)]
                    )
                    if len(steps) > self.config.forward_anchor_depth:
                        continue
                    cost = sum(float(step.step_cost or 0.0) for step in steps)
                    cost += self.config.route_length_penalty * len(steps)
                    bucket = alternatives.setdefault(candidate.target.key, [])
                    fingerprint = self._route_fingerprint(steps)
                    if any(
                        self._route_fingerprint(item.steps) == fingerprint
                        for item in bucket
                    ):
                        continue
                    bucket.append(ForwardAnchor(candidate.target, steps, cost))
                    trim_alternatives(candidate.target.key)
                    if any(
                        self._route_fingerprint(item.steps) == fingerprint
                        for item in bucket
                    ):
                        alternatives_changed = True
            if not alternatives_changed:
                break

        for key in tuple(anchors):
            if alternatives.get(key):
                anchors[key] = alternatives[key][0]
        if target_key in alternatives:
            anchors[target_key] = alternatives[target_key][0]
        self._stats["forward_anchor_updates"] = updates
        self._stats["forward_anchor_count"] = len(anchors)
        self._forward_alternatives = {
            key: tuple(values) for key, values in alternatives.items()
        }
        self._stats["forward_alternative_count"] = sum(
            len(values) for values in self._forward_alternatives.values()
        )
        self._store_forward_anchor_cache(cache_key, anchors)
        return anchors

    def _close_solved_leaves(
        self,
        state: RouteState,
        building_block_keys: set[str],
        anchors: Mapping[str, ForwardAnchor],
    ) -> Optional[RouteState]:
        unresolved = list(state.unresolved)
        steps = list(state.steps)
        molecules = dict(state.molecules)
        cost = state.cost
        changed = True
        while changed:
            changed = False
            for key in list(unresolved):
                if key in building_block_keys:
                    unresolved.remove(key)
                    changed = True
                    continue
                anchor = anchors.get(key)
                if anchor is None:
                    continue
                existing_ids = {step.candidate_id for step in steps}
                existing_targets = {step.target.key: step for step in steps}
                conflict = False
                for step in anchor.steps:
                    other = existing_targets.get(step.target.key)
                    if other is not None and other.candidate_id != step.candidate_id:
                        conflict = True
                        break
                if conflict:
                    return None
                for step in anchor.steps:
                    if step.candidate_id in existing_ids:
                        continue
                    steps.append(step)
                    existing_ids.add(step.candidate_id)
                    cost += float(step.step_cost or 0.0)
                    cost += self.config.route_length_penalty
                    molecules[step.target.key] = step.target
                    for precursor in step.precursors:
                        molecules[precursor.key] = precursor
                unresolved.remove(key)
                changed = True
        if len(steps) > self.config.max_depth:
            return None
        return RouteState(
            unresolved=tuple(dict.fromkeys(unresolved)),
            molecules=molecules,
            steps=tuple(steps),
            cost=cost,
            backward_depth=state.backward_depth,
            model_only_steps=state.model_only_steps,
            consecutive_model_only_steps=state.consecutive_model_only_steps,
        )

    def _distance_to_building_blocks(
        self, molecule: MoleculeNode, building_blocks: Sequence[MoleculeNode]
    ) -> float:
        if molecule.key in {item.key for item in building_blocks}:
            return 0.0
        values = []
        for block in building_blocks:
            value = structural_similarity(
                molecule.canonical_smiles or molecule.smiles,
                block.canonical_smiles or block.smiles,
            )
            if value is not None:
                values.append(value)
        return 1.0 - max(values) if values else 1.0

    def _heuristic(
        self, state: RouteState, building_blocks: Sequence[MoleculeNode]
    ) -> float:
        return sum(
            self._distance_to_building_blocks(state.molecules[key], building_blocks)
            for key in state.unresolved
        )

    def _choose_unresolved(
        self, state: RouteState, building_blocks: Sequence[MoleculeNode]
    ) -> str:
        return max(
            state.unresolved,
            key=lambda key: (
                self._distance_to_building_blocks(
                    state.molecules[key], building_blocks
                ),
                heavy_atom_count(state.molecules[key]),
                key,
            ),
        )

    def _expand_state(
        self,
        state: RouteState,
        target_key: str,
        candidates: Sequence[ReactionCandidate],
    ) -> Iterable[RouteState]:
        remaining = [key for key in state.unresolved if key != target_key]
        for candidate in candidates:
            model_only = state.model_only_steps + int(candidate.source == "MODEL")
            consecutive_model_only = (
                state.consecutive_model_only_steps + 1
                if candidate.source == "MODEL"
                else 0
            )
            if model_only > self.config.max_model_only_steps:
                self._stats["candidates_filtered"] += 1
                self._stats["filter_reasons"]["model_only_limit"] = (
                    self._stats["filter_reasons"].get("model_only_limit", 0) + 1
                )
                continue
            if consecutive_model_only > self.config.max_consecutive_model_only_steps:
                self._stats["candidates_filtered"] += 1
                self._stats["filter_reasons"]["consecutive_model_only_limit"] = (
                    self._stats["filter_reasons"].get(
                        "consecutive_model_only_limit", 0
                    )
                    + 1
                )
                continue
            molecules = dict(state.molecules)
            new_unresolved = list(remaining)
            for precursor in candidate.noncurrency_precursors:
                molecules[precursor.key] = precursor
                if precursor.key not in new_unresolved:
                    new_unresolved.append(precursor.key)
            yield RouteState(
                unresolved=tuple(new_unresolved),
                molecules=molecules,
                steps=state.steps + (candidate,),
                cost=(
                    state.cost
                    + float(candidate.step_cost or 0.0)
                    + self.config.route_length_penalty
                ),
                backward_depth=state.backward_depth + 1,
                model_only_steps=model_only,
                consecutive_model_only_steps=consecutive_model_only,
            )


    def _mcts_reward(
        self,
        state: RouteState,
        target: MoleculeNode,
        building_blocks: Sequence[MoleculeNode],
        *,
        complete: bool,
        dead_end: bool = False,
    ) -> float:
        """Return a bounded heuristic value, plus a bonus for complete routes."""
        step_count = max(1, len(state.steps))
        reaction_quality = math.exp(-state.cost / step_count)
        if complete:
            ordered = self._ordered_steps(target.key, state.steps)
            evaluation = evaluate_route_carbon(
                target,
                building_blocks,
                ordered,
                self.config.currency_metabolite_ids,
            )
            route_score, _ = production_quality_score(
                reaction_quality,
                evaluation,
                reaction_weight=self.config.route_reaction_quality_weight,
                carbon_weight=self.config.route_carbon_efficiency_weight,
                byproduct_weight=self.config.route_byproduct_weight,
                cofactor_weight=self.config.route_cofactor_weight,
            )
            return self.config.mcts_completion_bonus + route_score
        if dead_end:
            return 0.0
        average_distance = self._heuristic(state, building_blocks) / max(
            1, len(state.unresolved)
        )
        stock_progress = 1.0 / (1.0 + average_distance)
        unresolved_progress = 1.0 / (1.0 + 0.25 * len(state.unresolved))
        depth_discount = 1.0 / (1.0 + 0.05 * len(state.steps))
        return max(
            0.0,
            min(
                0.999999,
                (
                    0.55 * reaction_quality
                    + 0.35 * stock_progress
                    + 0.10 * unresolved_progress
                )
                * depth_discount,
            ),
        )

    def _mcts_select_child(
        self, node: _MCTSNode, rng: random.Random
    ) -> _MCTSNode:
        if rng.random() < self.config.mcts_epsilon:
            return rng.choice(node.children)

        parent_visits = max(1, node.visits)

        def ucb(child: _MCTSNode) -> Tuple[float, float, str]:
            if child.visits == 0:
                score = float("inf")
            else:
                exploration = self.config.mcts_exploration_constant * math.sqrt(
                    math.log(parent_visits + 1.0) / child.visits
                )
                # The fused single-step prior includes model/database evidence,
                # structural overlap and progress toward the supplied stock.
                prior_bonus = child.prior / (1.0 + child.visits)
                score = child.mean_value + exploration + prior_bonus
            return (
                score,
                child.prior,
                child.incoming_candidate_id or "",
            )

        return max(node.children, key=ucb)

    def _mcts_allowed_children(self, node: _MCTSNode) -> int:
        total = len(node.children) + len(node.pending_states)
        if not self.config.mcts_progressive_widening:
            return total
        allowed = math.ceil(
            self.config.mcts_progressive_widening_constant
            * max(1, node.visits + 1)
            ** self.config.mcts_progressive_widening_alpha
        )
        return min(total, max(1, allowed))

    def _mcts_prepare_node(
        self,
        node: _MCTSNode,
        building_blocks: Sequence[MoleculeNode],
        building_block_keys: set[str],
        anchors: Mapping[str, ForwardAnchor],
        signature_costs: Dict[Tuple[str, ...], List[float]],
    ) -> None:
        """Expand one OR molecule into reaction-AND alternatives."""
        if node.prepared or node.terminal or node.dead_end:
            return
        node.prepared = True
        state = node.state
        if not state.unresolved:
            node.terminal = True
            return
        if len(state.steps) >= self.config.max_depth:
            node.dead_end = True
            self._stats["dead_ends"] += 1
            return

        self._stats["states_expanded"] += 1
        selected_key = self._choose_unresolved(state, building_blocks)
        selected = state.molecules[selected_key]
        self._event(
            "mcts_expansion",
            iteration=self._stats["mcts_iterations"],
            tree_depth=node.tree_depth,
            expanded_molecule=selected_key,
            visits=node.visits,
            unresolved_count=len(state.unresolved),
        )
        candidates = self._candidate_batch(
            selected, building_blocks, state=state
        )
        if not candidates:
            node.dead_end = True
            self._stats["dead_ends"] += 1
            self._event("dead_end", molecule=selected_key)
            return

        pending: List[Tuple[float, str, RouteState]] = []
        for generated in self._expand_state(state, selected_key, candidates):
            closed = self._close_solved_leaves(
                generated, building_block_keys, anchors
            )
            if closed is None:
                continue
            self._stats["states_generated"] += 1
            if closed.unresolved:
                costs = signature_costs.setdefault(closed.signature, [])
                if (
                    len(costs) >= self.config.max_states_per_signature
                    and closed.cost >= max(costs) - 1.0e-12
                ):
                    self._stats["duplicate_states_pruned"] += 1
                    continue
                costs.append(closed.cost)
                costs.sort()
                del costs[self.config.max_states_per_signature :]

            action_index = len(state.steps)
            action = (
                generated.steps[action_index]
                if action_index < len(generated.steps)
                else generated.steps[-1]
            )
            prior = float(action.step_score or math.exp(-float(action.step_cost or 0.0)))
            pending.append((prior, action.candidate_id, closed))
            self._event(
                "mcts_reaction_child",
                parent_molecule=selected_key,
                reaction=action.candidate_id,
                and_precursor_count=len(action.noncurrency_precursors),
                prior=round(prior, 8),
            )

        pending.sort(key=lambda item: (-item[0], item[1], item[2].signature))
        node.pending_states = pending
        if not pending:
            node.dead_end = True
            self._stats["dead_ends"] += 1

    def _mcts_rollout(
        self,
        start: RouteState,
        target: MoleculeNode,
        building_blocks: Sequence[MoleculeNode],
        building_block_keys: set[str],
        anchors: Mapping[str, ForwardAnchor],
        rng: random.Random,
    ) -> Tuple[RouteState, float, bool]:
        """Run a virtual policy rollout without adding nodes to the main tree."""
        state = start
        if not state.unresolved:
            return state, self._mcts_reward(
                state, target, building_blocks, complete=True
            ), True

        for _ in range(self.config.mcts_rollout_depth):
            if len(state.steps) >= self.config.max_depth:
                break
            selected_key = self._choose_unresolved(state, building_blocks)
            selected = state.molecules[selected_key]
            candidates = self._candidate_batch(
                selected, building_blocks, state=state
            )
            options: List[RouteState] = []
            for generated in self._expand_state(state, selected_key, candidates):
                closed = self._close_solved_leaves(
                    generated, building_block_keys, anchors
                )
                if closed is not None:
                    options.append(closed)
            if not options:
                return state, self._mcts_reward(
                    state,
                    target,
                    building_blocks,
                    complete=False,
                    dead_end=True,
                ), False

            self._stats["mcts_rollout_steps"] += 1
            if rng.random() < self.config.mcts_epsilon:
                state = rng.choice(options)
            else:
                state = max(
                    options,
                    key=lambda value: (
                        not value.unresolved,
                        self._mcts_reward(
                            value,
                            target,
                            building_blocks,
                            complete=not value.unresolved,
                        ),
                        -value.cost,
                        tuple(step.candidate_id for step in value.steps),
                    ),
                )
            if not state.unresolved:
                return state, self._mcts_reward(
                    state, target, building_blocks, complete=True
                ), True

        return state, self._mcts_reward(
            state, target, building_blocks, complete=False
        ), False

    @staticmethod
    def _mcts_has_frontier(node: _MCTSNode) -> bool:
        if node.terminal:
            return False
        if not node.prepared:
            return True
        if node.pending_states:
            return True
        return any(HybridPlannerEngine._mcts_has_frontier(child) for child in node.children)

    def _run_and_or_mcts(
        self,
        initial: RouteState,
        target: MoleculeNode,
        building_blocks: Sequence[MoleculeNode],
        building_block_keys: set[str],
        anchors: Mapping[str, ForwardAnchor],
        complete: List[RouteState],
        complete_fingerprints: set[Tuple[str, ...]],
        pool_target: int,
    ) -> str:
        """Run UCB-guided MCTS over reaction-AND/molecule-OR route states."""
        rng = random.Random(self.config.mcts_random_seed)
        root = _MCTSNode(state=initial)
        signature_costs: Dict[Tuple[str, ...], List[float]] = {}
        tree_nodes = 1
        max_tree_depth = 0
        best_reward = 0.0
        termination = "MCTS_ITERATION_BUDGET_REACHED"

        def record_solution(route: RouteState) -> bool:
            validation = self._validate_endpoint_route(
                route, target, building_blocks
            )
            if not validation["valid"]:
                self._record_endpoint_rejection(validation)
                return False
            fingerprint = self._route_fingerprint(route.steps)
            if fingerprint in complete_fingerprints:
                return True
            complete_fingerprints.add(fingerprint)
            complete.append(route)
            self._stats["hybrid_backward_solutions"] += 1
            if len(route.steps) > route.backward_depth:
                self._stats["forward_anchor_stitched_solutions"] += 1
            self._event(
                "route_complete",
                cost=route.cost,
                steps=len(route.steps),
                algorithm="and_or_mcts",
            )
            return True

        for iteration in range(1, self.config.max_expansions + 1):
            self._stats["mcts_iterations"] = iteration
            node = root
            path = [root]

            while True:
                if node.terminal or node.dead_end:
                    break
                if not node.prepared:
                    if self._stats["states_expanded"] >= self.config.max_expansions:
                        break
                    self._mcts_prepare_node(
                        node,
                        building_blocks,
                        building_block_keys,
                        anchors,
                        signature_costs,
                    )
                if node.terminal or node.dead_end:
                    break

                allowed = self._mcts_allowed_children(node)
                if node.pending_states and len(node.children) < allowed:
                    if rng.random() < self.config.mcts_epsilon:
                        pending_index = rng.randrange(len(node.pending_states))
                    else:
                        pending_index = 0
                    prior, candidate_id, child_state = node.pending_states.pop(
                        pending_index
                    )
                    child = _MCTSNode(
                        state=child_state,
                        parent=node,
                        prior=prior,
                        incoming_candidate_id=candidate_id,
                        terminal=not child_state.unresolved,
                        tree_depth=node.tree_depth + 1,
                    )
                    node.children.append(child)
                    node = child
                    path.append(child)
                    tree_nodes += 1
                    max_tree_depth = max(max_tree_depth, child.tree_depth)
                    self._stats["mcts_expansion_nodes"] += 1
                    break

                if node.children:
                    node = self._mcts_select_child(node, rng)
                    path.append(node)
                    self._stats["mcts_selection_steps"] += 1
                    continue

                node.dead_end = True
                break

            self._stats["mcts_rollouts"] += 1
            rollout_state, reward, is_complete = self._mcts_rollout(
                node.state,
                target,
                building_blocks,
                building_block_keys,
                anchors,
                rng,
            )
            if is_complete:
                is_complete = record_solution(rollout_state)
                if not is_complete:
                    reward = self._mcts_reward(
                        rollout_state, target, building_blocks, complete=False
                    )

            best_reward = max(best_reward, reward)
            for visited in reversed(path):
                visited.visits += 1
                visited.value_sum += reward
                visited.best_value = max(visited.best_value, reward)
            self._stats["mcts_backpropagations"] += 1
            self._event(
                "mcts_backpropagation",
                iteration=iteration,
                reward=round(reward, 8),
                path_length=len(path),
                complete=is_complete,
            )

            if len(complete) >= pool_target:
                termination = "TOP_K_FOUND"
                break
            if not self._mcts_has_frontier(root):
                termination = (
                    "MCTS_TREE_EXHAUSTED_WITH_SOLUTIONS"
                    if complete
                    else "MCTS_TREE_EXHAUSTED_NO_SOLUTION"
                )
                break

        self._stats["mcts_tree_nodes"] = tree_nodes
        self._stats["mcts_max_tree_depth"] = max_tree_depth
        self._stats["mcts_root_visits"] = root.visits
        self._stats["mcts_root_mean_value"] = round(root.mean_value, 8)
        self._stats["mcts_best_reward"] = round(best_reward, 8)
        return termination

    @staticmethod
    def _route_fingerprint(steps: Sequence[ReactionCandidate]) -> Tuple[str, ...]:
        return tuple(sorted(step.candidate_id for step in steps))

    def _validate_endpoint_route(
        self,
        route: RouteState,
        target: MoleculeNode,
        building_blocks: Sequence[MoleculeNode],
    ) -> Dict[str, Any]:
        """Validate a rooted, acyclic AND route against its endpoint constraints."""
        required = {item.key for item in building_blocks}
        producers: Dict[str, List[Tuple[int, ReactionCandidate]]] = {}
        for index, step in enumerate(route.steps):
            producers.setdefault(step.target.key, []).append((index, step))

        duplicate_targets = sorted(
            key for key, values in producers.items() if len(values) != 1
        )
        reachable_steps = set()
        terminal_keys = set()
        cycle_paths: List[List[str]] = []

        def visit(key: str, stack: Tuple[str, ...]) -> None:
            if key in required:
                terminal_keys.add(key)
                return
            if key in stack:
                start = stack.index(key)
                cycle_paths.append(list(stack[start:] + (key,)))
                return
            values = producers.get(key, ())
            if not values:
                terminal_keys.add(key)
                return
            if len(values) != 1:
                return
            index, step = values[0]
            if index in reachable_steps:
                return
            reachable_steps.add(index)
            next_stack = stack + (key,)
            for precursor in step.noncurrency_precursors:
                visit(precursor.key, next_stack)

        visit(target.key, ())
        disconnected = [
            step.candidate_id
            for index, step in enumerate(route.steps)
            if index not in reachable_steps
        ]
        root_resolved = target.key in required or target.key in producers
        connected = not disconnected
        acyclic = not cycle_paths
        if self.config.require_all_building_blocks:
            endpoint_match = terminal_keys == required
        else:
            endpoint_match = bool(terminal_keys) and terminal_keys.issubset(required)
        valid = (
            root_resolved
            and not duplicate_targets
            and connected
            and acyclic
            and endpoint_match
        )
        reasons = []
        if not root_resolved:
            reasons.append("target_not_resolved")
        if duplicate_targets:
            reasons.append("multiple_producers_for_molecule")
        if not connected:
            reasons.append("disconnected_steps")
        if not acyclic:
            reasons.append("route_cycle")
        if not endpoint_match:
            reasons.append("terminal_building_block_mismatch")
        return {
            "valid": valid,
            "connected": connected,
            "acyclic": acyclic,
            "endpoint_match": endpoint_match,
            "require_all_building_blocks": self.config.require_all_building_blocks,
            "target_key": target.key,
            "required_building_block_keys": sorted(required),
            "terminal_building_block_keys": sorted(terminal_keys),
            "missing_building_block_keys": sorted(required - terminal_keys),
            "extra_terminal_keys": sorted(terminal_keys - required),
            "reachable_step_count": len(reachable_steps),
            "step_count": len(route.steps),
            "duplicate_target_keys": duplicate_targets,
            "disconnected_candidate_ids": disconnected,
            "cycle_paths": cycle_paths,
            "reasons": reasons,
        }

    def _record_endpoint_rejection(self, validation: Mapping[str, Any]) -> None:
        self._stats["endpoint_routes_rejected"] += 1
        counts = self._stats["endpoint_route_rejection_reasons"]
        for reason in validation.get("reasons", ()):
            counts[reason] = counts.get(reason, 0) + 1

    def _route_similarity(
        self,
        left: Sequence[ReactionCandidate],
        right: Sequence[ReactionCandidate],
    ) -> float:
        reaction_similarity = _jaccard(
            self._route_fingerprint(left), self._route_fingerprint(right)
        )
        intermediate_similarity = _jaccard(
            (step.target.key for step in left), (step.target.key for step in right)
        )
        ec_similarity = _jaccard(
            (ec for step in left for ec in step.ec_numbers),
            (ec for step in right for ec in step.ec_numbers),
        )
        return (reaction_similarity + intermediate_similarity + ec_similarity) / 3.0

    def _diverse_top_k(
        self,
        routes: Sequence[RouteState],
        target: MoleculeNode,
        building_blocks: Sequence[MoleculeNode],
    ) -> List[Tuple[RouteState, float, float, Dict[str, Any]]]:
        remaining = []
        for route in routes:
            ordered = self._ordered_steps(target.key, route.steps)
            evaluation = evaluate_route_carbon(
                target,
                building_blocks,
                ordered,
                self.config.currency_metabolite_ids,
            )
            reaction_quality = math.exp(-route.cost / max(1, len(route.steps)))
            base, components = production_quality_score(
                reaction_quality,
                evaluation,
                reaction_weight=self.config.route_reaction_quality_weight,
                carbon_weight=self.config.route_carbon_efficiency_weight,
                byproduct_weight=self.config.route_byproduct_weight,
                cofactor_weight=self.config.route_cofactor_weight,
            )
            evaluation["route_score_components"] = components
            remaining.append((route, base, evaluation))
        selected: List[Tuple[RouteState, float, float, Dict[str, Any]]] = []
        while remaining and len(selected) < self.config.top_k_routes:
            best_index = 0
            best_values = (-float("inf"), -float("inf"))
            for index, (route, base, _) in enumerate(remaining):
                max_similarity = max(
                    (
                        self._route_similarity(route.steps, chosen[0].steps)
                        for chosen in selected
                    ),
                    default=0.0,
                )
                adjusted = base - self.config.diversity_penalty * max_similarity
                values = (adjusted, base)
                if values > best_values:
                    best_values = values
                    best_index = index
            route, base, evaluation = remaining.pop(best_index)
            selected.append((route, base, best_values[0], evaluation))
        return selected

    @staticmethod
    def _ordered_steps(
        target_key: str, steps: Sequence[ReactionCandidate]
    ) -> Tuple[ReactionCandidate, ...]:
        producers = {step.target.key: step for step in steps}
        ordered: List[ReactionCandidate] = []
        visiting = set()
        visited = set()

        def visit(key: str) -> None:
            step = producers.get(key)
            if step is None or step.candidate_id in visited or key in visiting:
                return
            visiting.add(key)
            for precursor in step.noncurrency_precursors:
                visit(precursor.key)
            visiting.remove(key)
            visited.add(step.candidate_id)
            ordered.append(step)

        visit(target_key)
        for step in steps:
            if step.candidate_id not in visited:
                visit(step.target.key)
        return tuple(ordered)

    @staticmethod
    def _display_molecule(
        molecule: MoleculeNode,
        building_block_aliases: Mapping[str, MoleculeNode],
    ) -> str:
        input_block = building_block_aliases.get(molecule.key)
        if input_block is not None:
            primary = input_block.label
            if input_block.metabolite_id:
                primary += f" ({input_block.metabolite_id})"
            if (
                molecule.metabolite_id
                and molecule.metabolite_id != input_block.metabolite_id
            ):
                primary += (
                    f" [matched alias: {molecule.label} / "
                    f"{molecule.metabolite_id}]"
                )
            return primary
        return molecule.label + (
            f" ({molecule.metabolite_id})" if molecule.metabolite_id else ""
        )

    @staticmethod
    def _enzyme_information(step: ReactionCandidate) -> Dict[str, Any]:
        ec_numbers = list(step.ec_numbers)
        return {
            "ec_numbers": ec_numbers,
            "ec_display": (
                ", ".join(f"EC {value}" for value in ec_numbers)
                if ec_numbers
                else "EC unknown"
            ),
            "reaction_name": step.reaction_name,
            "database_reaction_ids": list(step.database_reaction_ids),
            "provenance": step.source,
            "template": step.template,
        }

    @classmethod
    def _enzyme_label(cls, step: ReactionCandidate) -> str:
        enzyme = cls._enzyme_information(step)
        values = [enzyme["ec_display"]]
        if enzyme["reaction_name"]:
            values.append(str(enzyme["reaction_name"]))
        if enzyme["database_reaction_ids"]:
            values.append(
                "reaction " + ",".join(enzyme["database_reaction_ids"])
            )
        values.append(str(enzyme["provenance"]))
        return "; ".join(values)

    @staticmethod
    def _retrosynthesis_order(
        target_key: str, steps: Sequence[ReactionCandidate]
    ) -> Tuple[ReactionCandidate, ...]:
        """Return steps target-first while preserving every AND branch."""
        producers = {step.target.key: step for step in steps}
        ordered: List[ReactionCandidate] = []
        visiting = set()
        visited = set()

        def visit(key: str) -> None:
            step = producers.get(key)
            if step is None or step.candidate_id in visited or key in visiting:
                return
            visiting.add(key)
            visited.add(step.candidate_id)
            ordered.append(step)
            for precursor in sorted(
                step.noncurrency_precursors, key=lambda item: item.key
            ):
                visit(precursor.key)
            visiting.remove(key)

        visit(target_key)
        for step in steps:
            if step.candidate_id not in visited:
                visit(step.target.key)
        return tuple(ordered)

    @classmethod
    def _main_chain(
        cls,
        target: MoleculeNode,
        steps: Sequence[ReactionCandidate],
        building_block_aliases: Mapping[str, MoleculeNode],
    ) -> List[str]:
        producers = {step.target.key: step for step in steps}
        labels = [cls._display_molecule(target, building_block_aliases)]
        key = target.key
        seen = set()
        while key in producers and key not in seen:
            seen.add(key)
            step = producers[key]
            main = step.main_precursors or step.noncurrency_precursors
            if not main:
                break
            precursor = sorted(
                main, key=lambda item: (heavy_atom_count(item), item.key), reverse=True
            )[0]
            labels.append(cls._display_molecule(precursor, building_block_aliases))
            key = precursor.key
        return labels

    @classmethod
    def _enzyme_annotated_main_chain(
        cls,
        target: MoleculeNode,
        steps: Sequence[ReactionCandidate],
        building_block_aliases: Mapping[str, MoleculeNode],
    ) -> str:
        producers = {step.target.key: step for step in steps}
        values = [cls._display_molecule(target, building_block_aliases)]
        key = target.key
        seen = set()
        while key in producers and key not in seen:
            seen.add(key)
            step = producers[key]
            main = step.main_precursors or step.noncurrency_precursors
            if not main:
                break
            precursor = sorted(
                main, key=lambda item: (heavy_atom_count(item), item.key), reverse=True
            )[0]
            values.append(f"{chr(0x2190)}[{cls._enzyme_label(step)}]")
            values.append(cls._display_molecule(precursor, building_block_aliases))
            key = precursor.key
        return " ".join(values)

    @classmethod
    def _retrosynthesis_reaction_records(
        cls,
        target: MoleculeNode,
        steps: Sequence[ReactionCandidate],
        building_block_aliases: Mapping[str, MoleculeNode],
    ) -> List[Dict[str, Any]]:
        records = []
        for index, step in enumerate(
            cls._retrosynthesis_order(target.key, steps), start=1
        ):
            noncurrency = list(step.noncurrency_precursors)
            currency_keys = set(step.currency_precursor_keys)
            currency = [
                item for item in step.precursors if item.key in currency_keys
            ]
            target_label = cls._display_molecule(
                step.target, building_block_aliases
            )
            precursor_labels = [
                cls._display_molecule(item, building_block_aliases)
                for item in noncurrency
            ]
            equation = (
                f"{target_label} {chr(0x2190)}[{cls._enzyme_label(step)}] "
                + " + ".join(precursor_labels)
            )
            records.append(
                {
                    "retrosynthesis_step": index,
                    "equation": equation,
                    "target": step.target.to_dict(),
                    "noncurrency_precursors": [
                        item.to_dict() for item in noncurrency
                    ],
                    "currency_precursors": [item.to_dict() for item in currency],
                    "enzyme": cls._enzyme_information(step),
                    "candidate_id": step.candidate_id,
                }
            )
        return records

    def _serialize_route(
        self,
        rank: int,
        route: RouteState,
        base_score: float,
        diversity_score: float,
        target: MoleculeNode,
        building_blocks: Sequence[MoleculeNode],
        evaluation: Mapping[str, Any],
    ) -> Dict[str, Any]:
        ordered = self._ordered_steps(target.key, route.steps)
        aliases = {item.key: item for item in building_blocks}
        chain = self._main_chain(target, route.steps, aliases)
        enzyme_annotated_chain = self._enzyme_annotated_main_chain(
            target, route.steps, aliases
        )
        retrosynthesis_reactions = self._retrosynthesis_reaction_records(
            target, route.steps, aliases
        )
        serialized_steps = []
        step_metrics = evaluation.get("step_metrics", {})
        for index, step in enumerate(ordered, start=1):
            value = {"step_number": index, **step.to_dict()}
            value["carbon_branch"] = step_metrics.get(step.candidate_id)
            for molecule_value in [value["target"], *value["precursors"]]:
                input_block = aliases.get(molecule_value.get("key"))
                if input_block is None:
                    continue
                if molecule_value.get("id") != input_block.metabolite_id:
                    molecule_value["input_building_block"] = input_block.to_dict()
                    molecule_value["matched_database_alias"] = {
                        "id": molecule_value.get("id"),
                        "name": molecule_value.get("name"),
                    }
            serialized_steps.append(value)
        counts = {"DATABASE": 0, "MODEL": 0, "BOTH": 0}
        for step in ordered:
            counts[step.source] = counts.get(step.source, 0) + 1
        route_validation = self._validate_endpoint_route(
            route, target, building_blocks
        )
        return {
            "rank": rank,
            "complete": True,
            "all_noncurrency_leaves_solved": True,
            "route_validation": route_validation,
            "route_score": round(base_score, 6),
            "diversity_adjusted_score": round(diversity_score, 6),
            "route_cost": round(route.cost, 6),
            "route_score_components": dict(
                evaluation.get("route_score_components", {})
            ),
            "target_carbon_efficiency": evaluation.get(
                "target_carbon_efficiency"
            ),
            "main_branch_carbon_retention": evaluation.get(
                "main_branch_carbon_retention"
            ),
            "major_organic_byproduct_penalty": evaluation.get(
                "major_organic_byproduct_penalty"
            ),
            "major_organic_byproducts": list(
                evaluation.get("major_organic_byproducts", [])
            ),
            "currency_cofactor_count": evaluation.get("currency_cofactor_count"),
            "carbon_evaluation_method": evaluation.get(
                "carbon_evaluation_method"
            ),
            "atom_mapping_used": evaluation.get("atom_mapping_used", False),
            "step_count": len(ordered),
            "backward_expansion_depth": route.backward_depth,
            "provenance_counts": counts,
            "retrosynthesis": " <- ".join(chain),
            "enzyme_annotated_retrosynthesis": enzyme_annotated_chain,
            "retrosynthesis_reactions": retrosynthesis_reactions,
            "forward_path": " -> ".join(reversed(chain)),
            "steps": serialized_steps,
        }

    def plan(
        self,
        target_query: str,
        building_block_queries: Sequence[str],
    ) -> Dict[str, Any]:
        started = time.monotonic()
        self._trace = []
        self._stats = {
            "states_expanded": 0,
            "states_generated": 0,
            "duplicate_states_pruned": 0,
            "beam_states_pruned": 0,
            "dead_ends": 0,
            "database_candidates": 0,
            "forward_queries": 0,
            "backward_queries": 0,
            "model_candidates": 0,
            "model_requests": 0,
            "model_calls": 0,
            "model_cache_hits": 0,
            "model_failures": 0,
            "fusion_candidates": 0,
            "consensus_candidates": 0,
            "candidates_filtered": 0,
            "filter_reasons": {},
            "forward_anchor_count": 0,
            "forward_anchor_updates": 0,
            "forward_anchor_cache_hits": 0,
            "forward_anchor_cache_misses": 0,
            "forward_alternative_count": 0,
            "forward_direct_solutions": 0,
            "forward_anchor_stitched_solutions": 0,
            "hybrid_backward_solutions": 0,
            "search_phases": [],
            "requested_top_k": self.config.top_k_routes,
            "warnings": [],
            "search_algorithm": self.config.search_algorithm,
            "mcts_iterations": 0,
            "mcts_selection_steps": 0,
            "mcts_expansion_nodes": 0,
            "mcts_rollouts": 0,
            "mcts_rollout_steps": 0,
            "mcts_backpropagations": 0,
            "mcts_tree_nodes": 0,
            "mcts_max_tree_depth": 0,
            "mcts_root_visits": 0,
            "mcts_root_mean_value": 0.0,
            "mcts_best_reward": 0.0,
            "endpoint_routes_rejected": 0,
            "endpoint_route_rejection_reasons": {},
        }
        try:
            target = self.database.normalize_molecule(target_query)
            building_blocks = tuple(
                self.database.normalize_molecule(value)
                for value in building_block_queries
                if str(value).strip()
            )
        except ValueError as exc:
            return {
                "ok": False,
                "error": {"type": "invalid_input", "message": str(exc)},
            }
        if not building_blocks:
            return {
                "ok": False,
                "error": {
                    "type": "invalid_input",
                    "message": "At least one building block is required.",
                },
            }

        self._stats["search_phases"].append("PHASE_A_FORWARD_ANCHOR")
        anchors = self._build_forward_anchors(building_blocks, target.key)
        building_block_keys = {item.key for item in building_blocks}
        initial = RouteState(
            unresolved=(target.key,),
            molecules={
                item.key: item for item in (target,) + tuple(building_blocks)
            },
            steps=(),
            cost=0.0,
            backward_depth=0,
            model_only_steps=0,
        )
        # A forward route is a high-confidence baseline solution, not a reason to
        # skip Phase B.  In particular, Hybrid mode must still query DB + model for
        # the target even when the forward closure already reaches it.
        complete: List[RouteState] = []
        direct_anchors = list(self._forward_alternatives.get(target.key, ()))
        if not direct_anchors:
            direct_anchor = anchors.get(target.key)
            if direct_anchor is not None:
                direct_anchors = [direct_anchor]
        for direct_anchor in direct_anchors:
            if not direct_anchor.steps:
                continue
            direct_route = RouteState(
                unresolved=(),
                molecules=dict(initial.molecules),
                steps=direct_anchor.steps,
                cost=direct_anchor.cost,
                backward_depth=0,
                model_only_steps=0,
            )
            validation = self._validate_endpoint_route(
                direct_route, target, building_blocks
            )
            if not validation["valid"]:
                self._record_endpoint_rejection(validation)
                continue
            complete.append(direct_route)
            self._stats["forward_direct_solutions"] += 1
            self._event(
                "forward_direct_solution",
                cost=direct_anchor.cost,
                steps=len(direct_anchor.steps),
            )

        counter = itertools.count()
        if target.key in building_block_keys:
            if not complete:
                zero_step = RouteState(
                    unresolved=(),
                    molecules=dict(initial.molecules),
                    steps=(),
                    cost=0.0,
                    backward_depth=0,
                    model_only_steps=0,
                )
                validation = self._validate_endpoint_route(
                    zero_step, target, building_blocks
                )
                if validation["valid"]:
                    complete.append(zero_step)
                else:
                    self._record_endpoint_rejection(validation)
            heap: List[Tuple[float, int, RouteState]] = []
        else:
            first_priority = initial.cost + self.config.heuristic_weight * self._heuristic(
                initial, building_blocks
            )
            heap = [(first_priority, next(counter), initial)]

        self._stats["search_phases"].append("PHASE_B_BACKWARD_HYBRID_SEARCH")
        signature_costs: Dict[Tuple[str, ...], List[float]] = {}
        complete_fingerprints = {
            self._route_fingerprint(route.steps) for route in complete
        }
        pool_target = self.config.top_k_routes * self.config.solution_pool_multiplier
        termination = "search_exhausted"

        if (
            self.config.search_algorithm == "and_or_mcts"
            and target.key not in building_block_keys
        ):
            termination = self._run_and_or_mcts(
                initial,
                target,
                building_blocks,
                building_block_keys,
                anchors,
                complete,
                complete_fingerprints,
                pool_target,
            )
            heap = []

        while (
            self.config.search_algorithm != "and_or_mcts"
            and heap
            and self._stats["states_expanded"] < self.config.max_expansions
        ):
            priority, _, state = heapq.heappop(heap)
            self._stats["states_expanded"] += 1
            if len(state.steps) >= self.config.max_depth:
                self._stats["dead_ends"] += 1
                continue
            selected_key = self._choose_unresolved(state, building_blocks)
            selected = state.molecules[selected_key]
            heuristic_cost = self._heuristic(state, building_blocks)
            self._event(
                "state_expanded",
                state_id=self._stats["states_expanded"],
                g_cost=round(state.cost, 8),
                h_cost=round(heuristic_cost, 8),
                f_cost=round(priority, 8),
                expanded_molecule=selected_key,
            )
            candidates = self._candidate_batch(
                selected, building_blocks, state=state
            )
            if not candidates:
                self._stats["dead_ends"] += 1
                self._event("dead_end", molecule=selected_key)
                continue
            for generated in self._expand_state(state, selected_key, candidates):
                closed = self._close_solved_leaves(
                    generated, building_block_keys, anchors
                )
                if closed is None:
                    continue
                self._stats["states_generated"] += 1
                if not closed.unresolved:
                    validation = self._validate_endpoint_route(
                        closed, target, building_blocks
                    )
                    if not validation["valid"]:
                        self._record_endpoint_rejection(validation)
                        continue
                    fingerprint = self._route_fingerprint(closed.steps)
                    if fingerprint not in complete_fingerprints:
                        complete_fingerprints.add(fingerprint)
                        complete.append(closed)
                        self._stats["hybrid_backward_solutions"] += 1
                        if len(closed.steps) > closed.backward_depth:
                            self._stats["forward_anchor_stitched_solutions"] += 1
                        self._event(
                            "route_complete",
                            cost=closed.cost,
                            steps=len(closed.steps),
                        )
                    if len(complete) >= pool_target:
                        termination = "solution_pool_reached"
                        heap = []
                        break
                    continue
                costs = signature_costs.setdefault(closed.signature, [])
                if (
                    len(costs) >= self.config.max_states_per_signature
                    and closed.cost >= max(costs) - 1.0e-12
                ):
                    self._stats["duplicate_states_pruned"] += 1
                    continue
                costs.append(closed.cost)
                costs.sort()
                del costs[self.config.max_states_per_signature :]
                priority = closed.cost + self.config.heuristic_weight * self._heuristic(
                    closed, building_blocks
                )
                heapq.heappush(heap, (priority, next(counter), closed))
            if len(heap) > self.config.global_beam_width:
                removed = len(heap) - self.config.global_beam_width
                heap = heapq.nsmallest(self.config.global_beam_width, heap)
                heapq.heapify(heap)
                self._stats["beam_states_pruned"] += removed

        if target.key in building_block_keys:
            termination = "TARGET_IS_BUILDING_BLOCK"
        elif self.config.search_algorithm == "and_or_mcts":
            pass
        elif termination == "solution_pool_reached":
            termination = "TOP_K_FOUND"
        elif self._stats["states_expanded"] >= self.config.max_expansions and heap:
            termination = "MAX_EXPANSIONS_REACHED"
        elif complete:
            termination = "SEARCH_EXHAUSTED_WITH_SOLUTIONS"
        else:
            termination = "SEARCH_EXHAUSTED_NO_SOLUTION"

        self._stats["search_phases"].append("PHASE_C_SOLUTION_MERGE")
        diverse = self._diverse_top_k(complete, target, building_blocks)
        routes = [
            self._serialize_route(
                index,
                route,
                base,
                adjusted,
                target,
                building_blocks,
                evaluation,
            )
            for index, (route, base, adjusted, evaluation) in enumerate(
                diverse, start=1
            )
        ]
        self._stats["search_phases"].extend(
            ["PHASE_D_ROUTE_SCORING", "PHASE_E_DIVERSITY_RERANK"]
        )
        self._stats["complete_routes_found"] = len(complete)
        self._stats["solution_pool_size_before_rerank"] = len(complete)
        self._stats["returned_route_count"] = len(routes)
        self._stats["top_k_shortfall"] = max(
            0, self.config.top_k_routes - len(routes)
        )
        self._stats["top_k_shortfall_reason"] = (
            "SEARCH_EXHAUSTED_BEFORE_ENOUGH_DISTINCT_COMPLETE_ROUTES"
            if len(routes) < self.config.top_k_routes
            else None
        )
        self._stats["database_only_routes"] = sum(
            all(step.source == "DATABASE" for step in route.steps) for route in complete
        )
        self._stats["model_assisted_routes"] = sum(
            any(step.source in {"MODEL", "BOTH"} for step in route.steps)
            for route in complete
        )
        self._stats["consensus_routes"] = sum(
            any(step.source == "BOTH" for step in route.steps) for route in complete
        )
        self._stats["termination_reason"] = termination
        self._stats["elapsed_seconds"] = round(time.monotonic() - started, 6)
        self._stats["database_cache"] = self.database.cache_info()
        if self.model_client is not None:
            self._stats["model_cache"] = self.model_client.cache_info()
        output: Dict[str, Any] = {
            "ok": True,
            "planner": (
                "HybridEnzymeRetroPlanner"
                if self.config.use_model
                else "EnzymeRetroPathFinder"
            ),
            "mode": (
                "database+model"
                if self.config.use_database and self.config.use_model
                else "model-only"
                if self.config.use_model
                else "database-only"
            ),
            "target": target.to_dict(),
            "building_blocks": [item.to_dict() for item in building_blocks],
            "endpoint_constraints": {
                "target_key": target.key,
                "required_building_block_keys": sorted(building_block_keys),
                "require_all_building_blocks": (
                    self.config.require_all_building_blocks
                ),
            },
            "route_count": len(routes),
            "routes": routes,
            "stats": self._stats,
            "config": self.config.to_dict(),
        }
        if self.config.include_search_trace:
            output["search_trace"] = self._trace
        return output
