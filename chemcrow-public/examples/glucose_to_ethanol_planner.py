"""Run glucose -> ethanol planning through the AND-OR search tool.

Examples
--------
Database-only AND-OR MCTS::

    python examples/glucose_to_ethanol_planner.py

Database + local MooRetro with AND-OR MCTS::

    python examples/glucose_to_ethanol_planner.py --hybrid

Compatibility search policies::

    python examples/glucose_to_ethanol_planner.py --algorithm best_first_beam
"""

from __future__ import annotations

import argparse
import json

from chemcrow.tools import EnzymeRetrosynthesisSearch


GLUCOSE = "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"
ETHANOL = "CCO"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="also load and query the local MooRetro model",
    )
    parser.add_argument(
        "--algorithm",
        choices=("and_or_mcts", "best_first_beam", "uniform_cost"),
        default="and_or_mcts",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-expansions", type=int, default=500)
    parser.add_argument("--mcts-rollout-depth", type=int, default=3)
    parser.add_argument("--mcts-seed", type=int, default=100)
    args = parser.parse_args()

    tool = EnzymeRetrosynthesisSearch(
        config={
            "use_database": True,
            "use_model": args.hybrid,
            "search_algorithm": args.algorithm,
            "top_k_routes": args.top_k,
            "solution_pool_multiplier": 1,
            "max_depth": 12,
            "max_expansions": args.max_expansions,
            "forward_anchor_depth": 6,
            "forward_anchor_max_nodes": 600,
            "database_top_k": 25,
            "model_top_k": 10,
            "candidate_beam_width": 20,
            "global_beam_width": 60,
            "mcts_rollout_depth": args.mcts_rollout_depth,
            "mcts_random_seed": args.mcts_seed,
        }
    )
    request = {
        "target": ETHANOL,
        "building_blocks": [GLUCOSE],
        "top_k": args.top_k,
        "search_algorithm": args.algorithm,
    }
    result = json.loads(tool.run(json.dumps(request)))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
