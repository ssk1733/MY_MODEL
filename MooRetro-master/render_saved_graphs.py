#!/usr/bin/env python3
"""Render saved MooRetro search results to Graphviz PDF/SVG/PNG files."""

import argparse
import glob
import json
import os
import re
import shutil
from typing import Dict, Iterable, Optional

from src.analyzer import Analyzer


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fileobj:
        return json.load(fileobj)


def render_graph(data: Dict,
                 save_path: str,
                 graph_name: str,
                 fmt: str,
                 include_node_png: bool) -> Optional[str]:
    molecule_nodes = data.get("molecule_nodes") or data.get("molecules")
    reaction_nodes = data.get("reaction_nodes") or data.get("reactions")
    connection = data.get("connection")

    if molecule_nodes is None or reaction_nodes is None or connection is None:
        print(f"[skip] {graph_name}: JSON does not contain graph node/edge data")
        return None

    os.makedirs(save_path, exist_ok=True)
    analyzer = Analyzer(save_path)
    output_base = analyzer.plot_search_tree_graphviz(
        molecule_info=molecule_nodes,
        reaction_info=reaction_nodes,
        connection_info=connection,
        save_path=save_path,
        format=fmt,
        include_node_png=include_node_png,
        graph_name=graph_name,
    )
    output_file = f"{output_base}.{fmt}"
    print(f"[ok] {output_file}")
    return output_file


def iter_path_jsons(pruned_dir: str) -> Iterable[str]:
    patterns = [
        os.path.join(pruned_dir, "front*_complete_paths_*.json"),
        os.path.join(pruned_dir, "pathwayEvaluation_front*.json"),
        os.path.join(pruned_dir, "complete_path_*.json"),
    ]
    seen = set()
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if path not in seen:
                seen.add(path)
                yield path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render PDFs from saved MooRetro JSON outputs without rerunning search."
    )
    parser.add_argument(
        "--result-dir",
        required=True,
        help="Single target result directory containing synthesis_tree_all_data.json and/or pruned/.",
    )
    parser.add_argument(
        "--which",
        choices=["all", "pruned", "paths", "all-and-pruned"],
        default="all-and-pruned",
        help="Which saved graph(s) to render.",
    )
    parser.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Graphviz output format.",
    )
    parser.add_argument(
        "--include-node-png",
        action="store_true",
        help="Embed molecule PNG images in graph nodes. Slower and heavier for large trees.",
    )
    args = parser.parse_args()

    dot_path = shutil.which("dot")
    if not dot_path:
        raise SystemExit("Graphviz 'dot' was not found in PATH. Install graphviz or activate the right env.")
    print(f"[info] dot: {dot_path}")

    result_dir = os.path.abspath(args.result_dir)
    pruned_dir = os.path.join(result_dir, "pruned")

    if args.which in {"all", "all-and-pruned"}:
        all_json = os.path.join(result_dir, "synthesis_tree_all_data.json")
        if os.path.exists(all_json):
            render_graph(load_json(all_json), result_dir, "synthesis_tree_all", args.format, args.include_node_png)
        else:
            print(f"[skip] missing {all_json}")

    if args.which in {"pruned", "all-and-pruned"}:
        pruned_json = os.path.join(pruned_dir, "synthesis_tree_pruned_data.json")
        if os.path.exists(pruned_json):
            render_graph(load_json(pruned_json), pruned_dir, "synthesis_tree_pruned", args.format, args.include_node_png)
        else:
            print(f"[skip] missing {pruned_json}")

    if args.which == "paths":
        if not os.path.isdir(pruned_dir):
            print(f"[skip] missing {pruned_dir}")
            return
        rendered = 0
        for path_json in iter_path_jsons(pruned_dir):
            graph_name = os.path.splitext(os.path.basename(path_json))[0]
            data = load_json(path_json)
            match = re.fullmatch(r"pathwayEvaluation_front(\d+)_(\d+)", graph_name)
            if match:
                graph_name = f"front{match.group(1)}_complete_paths_{match.group(2)}"
            if render_graph(data, pruned_dir, graph_name, args.format, args.include_node_png):
                rendered += 1
        print(f"[info] rendered {rendered} path graph(s)")


if __name__ == "__main__":
    main()
