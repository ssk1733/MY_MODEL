'''

Author: Caoyh
Date: 2025-07-19 09:23:26
LastEditors: BellaCaoyh caoyh_cyh@163.com
LastEditTime: 2025-07-19 09:24:33
'''
'''
import argparse
import csv
import gzip
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from src.analyzer import Analyzer
from src.selector import Selector
from src.evaluator import PathEvaluator
from config.config import MCTSConfig, SelectionConfig
from searcher.mcts_searcher import MCTSSearcher
from src.synthesis_tree import SynthesisTree, SynthesisPath
from utils.logger import setup_logger
from utils.utils import dict2json, json2data, read_txt


def sanitize_filename(name: str, max_length: int = 80) -> str:
    cleaned = re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip('._')
    if not cleaned:
        cleaned = 'target'
    return cleaned[:max_length]


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def build_paroutes_tree(path: SynthesisPath) -> Dict[str, Any]:
    if not path.molecule_nodes:
        return {}

    ordered_molecules = list(reversed(path.molecule_nodes))
    ordered_reactions = list(reversed(path.reaction_nodes))

    def build_molecule_node(idx: int) -> Dict[str, Any]:
        mol_node = ordered_molecules[idx]
        node = {
            'type': 'mol',
            'smiles': mol_node.molecule,
            'in_stock': bool(mol_node.is_commercial),
        }
        if idx < len(ordered_reactions):
            reaction_node = ordered_reactions[idx]
            child_mol = build_molecule_node(idx + 1)
            node['children'] = [{
                'type': 'reaction',
                'smiles': reaction_node.reaction_info.templates,
                'metadata': {
                    'ec': reaction_node.reaction_info.ec,
                    'score': reaction_node.reaction_info.scores,
                    'top_k': reaction_node.reaction_info.top_k,
                    'similarity': reaction_node.reaction_info.similarity,
                    'feasibility': reaction_node.reaction_info.feasibility,
                    'yield': reaction_node.reaction_info.yield_estimate,
                    'toxicity': reaction_node.reaction_info.toxicity,
                },
                'children': [child_mol],
            }]
        return node

    return build_molecule_node(0)


def export_paroutes_routes(complete_paths: List[SynthesisPath], output_path: str) -> List[Dict[str, Any]]:
    route_groups = [build_paroutes_tree(path) for path in complete_paths]
    if output_path.endswith('.gz'):
        with gzip.open(output_path, 'wt', encoding='utf-8') as fileobj:
            json.dump(route_groups, fileobj, indent=2)
    else:
        with open(output_path, 'w', encoding='utf-8') as fileobj:
            json.dump(route_groups, fileobj, indent=2)
    return route_groups


def evaluate_complete_paths(tree: SynthesisTree, evaluator: PathEvaluator) -> List[Dict[str, Any]]:
    evaluated_paths = []
    for path in tree.complete_paths:
        path_metrics = evaluator._evaluate_complete_path(path.molecules, path.reactions)
        objective_vector = evaluator._generate_objective_vector(path_metrics).tolist()
        path._metrics = path_metrics
        path.objective_values = objective_vector
        evaluated_paths.append({
            'path': path,
            'metrics': path_metrics.to_dict,
            'objective_values': objective_vector,
            'path_length': path.length,
        })
    return evaluated_paths


def summarize_run(config: MCTSConfig,
                  tree: SynthesisTree,
                  evaluator: PathEvaluator,
                  elapsed_time: float,
                  sample_id: Optional[str] = None,
                  target_rank: Optional[int] = None) -> Dict[str, Any]:
    evaluated_paths = evaluate_complete_paths(tree, evaluator)
    solved = len(evaluated_paths) > 0
    best_path = None
    if evaluated_paths:
        best_path = max(
            evaluated_paths,
            key=lambda item: sum(item['objective_values'])
        )

    summary = {
        'sample_id': sample_id or sanitize_filename(config.project_name),
        'target_rank': target_rank,
        'project_name': config.project_name,
        'target_molecule': config.target_molecule,
        'num_commercial_molecules': len(config.commercial_molecules),
        'solved': solved,
        'num_complete_paths': len(evaluated_paths),
        'elapsed_time_sec': round(elapsed_time, 3),
        'tree_total_nodes': tree.get_statistics().total_nodes,
        'tree_molecule_nodes': tree.get_statistics().molecule_nodes,
        'tree_reaction_nodes': tree.get_statistics().reaction_nodes,
        'pareto_front_size': len(tree.complete_paths),
        'result_path': config.result_path,
        'max_iterations': config.max_iterations,
        'max_simulation_depth': config.max_simulation_depth,
        'single_fn_type': config.single_fn_type,
    }

    if best_path:
        summary.update({
            'best_path_length': best_path['path_length'],
            'best_overall_retro_score': best_path['metrics']['overall_retro_score'],
            'best_overall_feasibility': best_path['metrics']['overall_feasibility'],
            'best_overall_similarity': best_path['metrics']['overall_similarity'],
            'best_overall_yield': best_path['metrics']['overall_theoretical_yield'],
            'best_objective_sum': round(sum(best_path['objective_values']), 6),
        })
    else:
        summary.update({
            'best_path_length': None,
            'best_overall_retro_score': None,
            'best_overall_feasibility': None,
            'best_overall_similarity': None,
            'best_overall_yield': None,
            'best_objective_sum': None,
        })

    return summary


def run_search_with_config(mcts_config: MCTSConfig,
                           enable_graphs: bool = True,
                           sample_id: Optional[str] = None,
                           target_rank: Optional[int] = None) -> Dict[str, Any]:
    from utils.single_retro_fn import single_fn
    ensure_dir(mcts_config.result_path)
    logger = setup_logger(mcts_config.result_path)
    target_molecule = mcts_config.target_molecule
    commercial_molecules = mcts_config.commercial_molecules

    tree = SynthesisTree(target_molecule, commercial_molecules, logger=logger)
    evaluator = PathEvaluator(commercial_molecules)
    selector = Selector(commercial_molecules, evaluator, SelectionConfig())
    analyzer = Analyzer(mcts_config.result_path) if enable_graphs else None

    searcher = MCTSSearcher(
        tree=tree,
        single_fn_function=single_fn,
        evaluator=evaluator,
        selector=selector,
        mcts_config=mcts_config,
        analyzer=analyzer,
        logger=logger
    )

    start_time = time.time()
    searcher.search()
    elapsed_time = time.time() - start_time

    summary = summarize_run(
        config=mcts_config,
        tree=tree,
        evaluator=evaluator,
        elapsed_time=elapsed_time,
        sample_id=sample_id,
        target_rank=target_rank,
    )

    summary_path = os.path.join(mcts_config.result_path, 'run_summary.json')
    dict2json(summary, summary_path)

    paroutes_output_path = os.path.join(mcts_config.result_path, 'paroutes_routes.json')
    paroutes_routes = export_paroutes_routes(tree.complete_paths, paroutes_output_path)
    summary['paroutes_routes_file'] = paroutes_output_path
    summary['num_exported_route_groups'] = len(paroutes_routes)
    dict2json(summary, summary_path)

    return summary


def create_dataset_config(base_config: MCTSConfig,
                          dataset_name: str,
                          target_smiles: str,
                          commercial_molecules: List[str],
                          output_root: str,
                          sample_id: str) -> MCTSConfig:
    result_path = os.path.join(output_root, dataset_name, sample_id)
    return MCTSConfig(
        project_name=f'{dataset_name}_{sample_id}',
        target_molecule=target_smiles,
        commercial_molecules=commercial_molecules,
        random_seed=base_config.random_seed,
        c_param=base_config.c_param,
        max_iterations=base_config.max_iterations,
        max_simulation_depth=base_config.max_simulation_depth,
        early_stopping_threshold=base_config.early_stopping_threshold,
        real_time_feedback_interval=base_config.real_time_feedback_interval,
        enable_progressive_widening=base_config.enable_progressive_widening,
        progressive_widening_alpha=base_config.progressive_widening_alpha,
        epsilon_greedy=base_config.epsilon_greedy,
        heuristic_similarity_param=base_config.heuristic_similarity_param,
        single_fn_type=base_config.single_fn_type,
        hypervolume_improvement_threshold=base_config.hypervolume_improvement_threshold,
        pareto_front_update_frequency=base_config.pareto_front_update_frequency,
        min_iterations_before_early_stop=base_config.min_iterations_before_early_stop,
        max_iterations_without_improvement=base_config.max_iterations_without_improvement,
        result_path=result_path,
    )


def write_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(output_path, 'w', encoding='utf-8', newline='') as fileobj:
        writer = csv.DictWriter(fileobj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_table1_like_summary(dataset_name: str,
                              run_rows: List[Dict[str, Any]],
                              stock_size: int,
                              references_file: Optional[str]) -> Dict[str, Any]:
    total_targets = len(run_rows)
    solved_targets = sum(1 for row in run_rows if row.get('solved'))
    resolution_rate = solved_targets / total_targets if total_targets else 0.0
    solved_rows = [row for row in run_rows if row.get('solved')]
    elapsed_values = [row['elapsed_time_sec'] for row in run_rows if row.get('elapsed_time_sec') is not None]
    complete_path_counts = [row.get('num_complete_paths', 0) or 0 for row in run_rows]
    solved_path_lengths = [row['best_path_length'] for row in solved_rows if row.get('best_path_length') is not None]

    return {
        'dataset_name': dataset_name,
        'search_frame': 'MooRetro-ParetoMCTS',
        'configuration': {
            'single_step_policy': run_rows[0]['single_fn_type'] if run_rows else None,
            'guided_function': 'Pareto multi-objective evaluation',
            'search_iteration': run_rows[0]['max_iterations'] if run_rows else None,
            'buyable_building_block_count': stock_size,
        },
        'total_targets': total_targets,
        'solved_targets': solved_targets,
        'resolution_rate': round(resolution_rate * 100, 2),
        'mean_elapsed_time_sec': round(
            sum(elapsed_values) / len(elapsed_values), 3
        ) if elapsed_values else None,
        'median_complete_paths': sorted(complete_path_counts)[total_targets // 2] if total_targets else 0,
        'mean_best_path_length': round(
            sum(solved_path_lengths) / len(solved_path_lengths), 3
        ) if solved_path_lengths else None,
        'references_file': references_file,
    }


def run_dataset_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if not os.path.exists(args.targets_file):
        raise FileNotFoundError(f'targets file not found: {args.targets_file}')
    if not os.path.exists(args.stock_file):
        raise FileNotFoundError(f'stock file not found: {args.stock_file}')

    targets = read_txt(args.targets_file)
    stock = read_txt(args.stock_file)

    if args.offset:
        targets = targets[args.offset:]
    if args.limit is not None:
        targets = targets[:args.limit]

    output_root = ensure_dir(args.output_root)
    dataset_root = ensure_dir(os.path.join(output_root, args.dataset_name))

    base_config = MCTSConfig(
        project_name=args.project_name or args.dataset_name,
        max_iterations=args.max_iterations,
        max_simulation_depth=args.max_simulation_depth,
        single_fn_type=args.single_fn_type,
        result_path=os.path.join(dataset_root, 'base_config_placeholder')
    )

    run_rows: List[Dict[str, Any]] = []
    all_route_groups: List[List[Dict[str, Any]]] = []

    for idx, target_smiles in enumerate(targets, start=1):
        sample_id = f'{idx:05d}_{sanitize_filename(target_smiles)}'
        config = create_dataset_config(
            base_config=base_config,
            dataset_name=args.dataset_name,
            target_smiles=target_smiles,
            commercial_molecules=stock,
            output_root=output_root,
            sample_id=sample_id,
        )
        try:
            summary = run_search_with_config(
                config,
                enable_graphs=not args.disable_graphs,
                sample_id=sample_id,
                target_rank=idx,
            )
            paroutes_routes = json2data(summary['paroutes_routes_file']) if os.path.exists(summary['paroutes_routes_file']) else []
        except Exception as exc:
            summary = {
                'sample_id': sample_id,
                'target_rank': idx,
                'project_name': config.project_name,
                'target_molecule': target_smiles,
                'num_commercial_molecules': len(stock),
                'solved': False,
                'num_complete_paths': 0,
                'elapsed_time_sec': None,
                'tree_total_nodes': None,
                'tree_molecule_nodes': None,
                'tree_reaction_nodes': None,
                'pareto_front_size': None,
                'result_path': config.result_path,
                'max_iterations': config.max_iterations,
                'max_simulation_depth': config.max_simulation_depth,
                'single_fn_type': config.single_fn_type,
                'best_path_length': None,
                'best_overall_retro_score': None,
                'best_overall_feasibility': None,
                'best_overall_similarity': None,
                'best_overall_yield': None,
                'best_objective_sum': None,
                'error': str(exc),
            }
            ensure_dir(config.result_path)
            dict2json(summary, os.path.join(config.result_path, 'run_summary.json'))
            paroutes_routes = []
        run_rows.append(summary)
        all_route_groups.append(paroutes_routes)

    summary_csv = os.path.join(dataset_root, 'dataset_summary.csv')
    write_csv(run_rows, summary_csv)

    summary_json = os.path.join(dataset_root, 'dataset_summary.json')
    dict2json(run_rows, summary_json)

    paroutes_json = os.path.join(dataset_root, 'paroutes_routes.json.gz')
    with gzip.open(paroutes_json, 'wt', encoding='utf-8') as fileobj:
        json.dump(all_route_groups, fileobj, indent=2)

    table1_like = build_table1_like_summary(
        dataset_name=args.dataset_name,
        run_rows=run_rows,
        stock_size=len(stock),
        references_file=args.references_file,
    )
    table1_like['paroutes_routes_file'] = paroutes_json
    table1_like['dataset_summary_csv'] = summary_csv
    table1_like['dataset_summary_json'] = summary_json
    table1_path = os.path.join(dataset_root, 'table1_like_summary.json')
    dict2json(table1_like, table1_path)

    run_metadata = {
        'dataset_name': args.dataset_name,
        'targets_file': args.targets_file,
        'stock_file': args.stock_file,
        'references_file': args.references_file,
        'output_root': output_root,
        'limit': args.limit,
        'offset': args.offset,
        'max_iterations': args.max_iterations,
        'max_simulation_depth': args.max_simulation_depth,
        'single_fn_type': args.single_fn_type,
        'disable_graphs': args.disable_graphs,
        'notes': [
            'solved 定义为至少找到一条完整闭合路径',
            'PaRoutes 兼容导出基于当前 MooRetro 线性路径生成递归 route tree',
            '当参考 routes 文件存在时，可继续使用 PaRoutes analysis/route_quality.py 做进一步分析',
        ],
    }
    dict2json(run_metadata, os.path.join(dataset_root, 'run_metadata.json'))

    return table1_like


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run MooRetro single-target or PaRoutes-style dataset experiments.')
    parser.add_argument('--mode', choices=['single', 'dataset'], default='single')
    parser.add_argument('--project-name', default=None)
    parser.add_argument('--target', default=None)
    parser.add_argument('--targets-file', default='/root/PaRoutes/data/n1-targets.txt')
    parser.add_argument('--stock-file', default='/root/PaRoutes/data/n1-stock.txt')
    parser.add_argument('--references-file', default=None)
    parser.add_argument('--dataset-name', default='n1')
    parser.add_argument('--output-root', default='results/paroutes_runs')
    parser.add_argument('--single-fn-type', default='biocyc')
    parser.add_argument('--max-iterations', type=int, default=10)
    parser.add_argument('--max-simulation-depth', type=int, default=10)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--disable-graphs', action='store_true')
    return parser


def test_for_single_fn():
    from utils.single_retro_fn import single_fn

    target_molecule: str = '[NH3+]CCCC[NH3+]'
    commercial_molecules = [
        'NC(=[NH2+])NCCC[C@H]([NH3+])C(=O)[O-]',
    ]
    res = single_fn(target_smile=target_molecule, building_blocks=commercial_molecules)
    print(res)


def test_for_mcts(config: Optional[MCTSConfig] = None, enable_graphs: bool = True):
    mcts_config = config or MCTSConfig()
    return run_search_with_config(mcts_config, enable_graphs=enable_graphs)


if __name__ == '__main__':
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == 'dataset':
        result = run_dataset_experiment(args)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        config = MCTSConfig(
            project_name=args.project_name or 'PWY-43',
            target_molecule=args.target or '[NH3+]CCCC[NH3+]',
            commercial_molecules=MCTSConfig().commercial_molecules,
            max_iterations=args.max_iterations,
            max_simulation_depth=args.max_simulation_depth,
            single_fn_type=args.single_fn_type,
        )
        result = test_for_mcts(config=config, enable_graphs=not args.disable_graphs)
        print(json.dumps(result, indent=2, ensure_ascii=False))
'''
#去掉帕累托

import argparse
import csv
import gzip
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from src.analyzer import Analyzer
from src.selector import Selector
from src.evaluator import PathEvaluator
from config.config import MCTSConfig, SelectionConfig
from searcher.mcts_searcher import MCTSSearcher
from src.synthesis_tree import SynthesisTree, SynthesisPath
from utils.logger import setup_logger
from utils.utils import dict2json, json2data, read_txt


def sanitize_filename(name: str, max_length: int = 80) -> str:
    cleaned = re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip('._')
    if not cleaned:
        cleaned = 'target'
    return cleaned[:max_length]


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def build_paroutes_tree(path: SynthesisPath) -> Dict[str, Any]:
    if not path.molecule_nodes:
        return {}

    ordered_molecules = list(reversed(path.molecule_nodes))
    ordered_reactions = list(reversed(path.reaction_nodes))

    def build_molecule_node(idx: int) -> Dict[str, Any]:
        mol_node = ordered_molecules[idx]
        node = {
            'type': 'mol',
            'smiles': mol_node.molecule,
            'in_stock': bool(mol_node.is_commercial),
        }
        if idx < len(ordered_reactions):
            reaction_node = ordered_reactions[idx]
            child_mol = build_molecule_node(idx + 1)
            node['children'] = [{
                'type': 'reaction',
                'smiles': reaction_node.reaction_info.templates,
                'metadata': {
                    'ec': reaction_node.reaction_info.ec,
                    'score': reaction_node.reaction_info.scores,
                    'top_k': reaction_node.reaction_info.top_k,
                    'similarity': reaction_node.reaction_info.similarity,
                    'feasibility': reaction_node.reaction_info.feasibility,
                    'yield': reaction_node.reaction_info.yield_estimate,
                    'toxicity': reaction_node.reaction_info.toxicity,
                },
                'children': [child_mol],
            }]
        return node

    return build_molecule_node(0)


def export_paroutes_routes(complete_paths: List[SynthesisPath], output_path: str) -> List[Dict[str, Any]]:
    route_groups = [build_paroutes_tree(path) for path in complete_paths]
    if output_path.endswith('.gz'):
        with gzip.open(output_path, 'wt', encoding='utf-8') as fileobj:
            json.dump(route_groups, fileobj, indent=2)
    else:
        with open(output_path, 'w', encoding='utf-8') as fileobj:
            json.dump(route_groups, fileobj, indent=2)
    return route_groups


def evaluate_complete_paths(tree: SynthesisTree, evaluator: PathEvaluator) -> List[Dict[str, Any]]:
    evaluated_paths = []
    for path in tree.complete_paths:
        path_metrics = evaluator._evaluate_complete_path(path.molecules, path.reactions)
        # 单目标：直接获取标量得分
        objective_value = evaluator._generate_objective_value(path_metrics)
        path._metrics = path_metrics
        path.objective_values = objective_value   # 存储标量
        evaluated_paths.append({
            'path': path,
            'metrics': path_metrics.to_dict,
            'objective_value': objective_value,   # 改为 objective_value
            'path_length': path.length,
        })
    return evaluated_paths


def summarize_run(config: MCTSConfig,
                  tree: SynthesisTree,
                  evaluator: PathEvaluator,
                  elapsed_time: float,
                  sample_id: Optional[str] = None,
                  target_rank: Optional[int] = None) -> Dict[str, Any]:
    evaluated_paths = evaluate_complete_paths(tree, evaluator)
    solved = len(evaluated_paths) > 0
    best_path = None
    if evaluated_paths:
        # 单目标：直接比较 objective_value
        best_path = max(
            evaluated_paths,
            key=lambda item: item['objective_value']
        )

    summary = {
        'sample_id': sample_id or sanitize_filename(config.project_name),
        'target_rank': target_rank,
        'project_name': config.project_name,
        'target_molecule': config.target_molecule,
        'num_commercial_molecules': len(config.commercial_molecules),
        'solved': solved,
        'num_complete_paths': len(evaluated_paths),
        'elapsed_time_sec': round(elapsed_time, 3),
        'tree_total_nodes': tree.get_statistics().total_nodes,
        'tree_molecule_nodes': tree.get_statistics().molecule_nodes,
        'tree_reaction_nodes': tree.get_statistics().reaction_nodes,
        'pareto_front_size': len(tree.complete_paths),
        'result_path': config.result_path,
        'max_iterations': config.max_iterations,
        'max_simulation_depth': config.max_simulation_depth,
        'single_fn_type': config.single_fn_type,
    }

    if best_path:
        summary.update({
            'best_path_length': best_path['path_length'],
            'best_overall_retro_score': best_path['metrics']['overall_retro_score'],
            'best_overall_feasibility': best_path['metrics']['overall_feasibility'],
            'best_overall_similarity': best_path['metrics']['overall_similarity'],
            'best_overall_yield': best_path['metrics']['overall_theoretical_yield'],
            'best_objective_sum': round(best_path['objective_value'], 6),   # 标量
        })
    else:
        summary.update({
            'best_path_length': None,
            'best_overall_retro_score': None,
            'best_overall_feasibility': None,
            'best_overall_similarity': None,
            'best_overall_yield': None,
            'best_objective_sum': None,
        })

    return summary


def run_search_with_config(mcts_config: MCTSConfig,
                           enable_graphs: bool = True,
                           sample_id: Optional[str] = None,
                           target_rank: Optional[int] = None) -> Dict[str, Any]:
    from utils.single_retro_fn import single_fn
    ensure_dir(mcts_config.result_path)
    logger = setup_logger(mcts_config.result_path)
    target_molecule = mcts_config.target_molecule
    commercial_molecules = mcts_config.commercial_molecules

    tree = SynthesisTree(target_molecule, commercial_molecules, logger=logger)
    evaluator = PathEvaluator(commercial_molecules)
    # 关键：启用单目标模式
    selector = Selector(commercial_molecules, evaluator, SelectionConfig(), single_objective=True)
    analyzer = Analyzer(mcts_config.result_path) if enable_graphs else None

    searcher = MCTSSearcher(
        tree=tree,
        single_fn_function=single_fn,
        evaluator=evaluator,
        selector=selector,
        mcts_config=mcts_config,
        analyzer=analyzer,
        logger=logger
    )

    start_time = time.time()
    searcher.search()
    elapsed_time = time.time() - start_time

    summary = summarize_run(
        config=mcts_config,
        tree=tree,
        evaluator=evaluator,
        elapsed_time=elapsed_time,
        sample_id=sample_id,
        target_rank=target_rank,
    )

    summary_path = os.path.join(mcts_config.result_path, 'run_summary.json')
    dict2json(summary, summary_path)

    paroutes_output_path = os.path.join(mcts_config.result_path, 'paroutes_routes.json')
    paroutes_routes = export_paroutes_routes(tree.complete_paths, paroutes_output_path)
    summary['paroutes_routes_file'] = paroutes_output_path
    summary['num_exported_route_groups'] = len(paroutes_routes)
    dict2json(summary, summary_path)

    return summary


def create_dataset_config(base_config: MCTSConfig,
                          dataset_name: str,
                          target_smiles: str,
                          commercial_molecules: List[str],
                          output_root: str,
                          sample_id: str) -> MCTSConfig:
    result_path = os.path.join(output_root, dataset_name, sample_id)
    return MCTSConfig(
        project_name=f'{dataset_name}_{sample_id}',
        target_molecule=target_smiles,
        commercial_molecules=commercial_molecules,
        random_seed=base_config.random_seed,
        c_param=base_config.c_param,
        max_iterations=base_config.max_iterations,
        max_simulation_depth=base_config.max_simulation_depth,
        early_stopping_threshold=base_config.early_stopping_threshold,
        real_time_feedback_interval=base_config.real_time_feedback_interval,
        enable_progressive_widening=base_config.enable_progressive_widening,
        progressive_widening_alpha=base_config.progressive_widening_alpha,
        epsilon_greedy=base_config.epsilon_greedy,
        heuristic_similarity_param=base_config.heuristic_similarity_param,
        single_fn_type=base_config.single_fn_type,
        hypervolume_improvement_threshold=base_config.hypervolume_improvement_threshold,
        pareto_front_update_frequency=base_config.pareto_front_update_frequency,
        min_iterations_before_early_stop=base_config.min_iterations_before_early_stop,
        max_iterations_without_improvement=base_config.max_iterations_without_improvement,
        result_path=result_path,
    )


def write_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(output_path, 'w', encoding='utf-8', newline='') as fileobj:
        writer = csv.DictWriter(fileobj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_table1_like_summary(dataset_name: str,
                              run_rows: List[Dict[str, Any]],
                              stock_size: int,
                              references_file: Optional[str]) -> Dict[str, Any]:
    total_targets = len(run_rows)
    solved_targets = sum(1 for row in run_rows if row.get('solved'))
    resolution_rate = solved_targets / total_targets if total_targets else 0.0
    solved_rows = [row for row in run_rows if row.get('solved')]
    elapsed_values = [row['elapsed_time_sec'] for row in run_rows if row.get('elapsed_time_sec') is not None]
    complete_path_counts = [row.get('num_complete_paths', 0) or 0 for row in run_rows]
    solved_path_lengths = [row['best_path_length'] for row in solved_rows if row.get('best_path_length') is not None]

    return {
        'dataset_name': dataset_name,
        'search_frame': 'MooRetro-ParetoMCTS',   # 名字可保留但实际已是单目标
        'configuration': {
            'single_step_policy': run_rows[0]['single_fn_type'] if run_rows else None,
            'guided_function': 'Single-objective weighted sum',
            'search_iteration': run_rows[0]['max_iterations'] if run_rows else None,
            'buyable_building_block_count': stock_size,
        },
        'total_targets': total_targets,
        'solved_targets': solved_targets,
        'resolution_rate': round(resolution_rate * 100, 2),
        'mean_elapsed_time_sec': round(
            sum(elapsed_values) / len(elapsed_values), 3
        ) if elapsed_values else None,
        'median_complete_paths': sorted(complete_path_counts)[total_targets // 2] if total_targets else 0,
        'mean_best_path_length': round(
            sum(solved_path_lengths) / len(solved_path_lengths), 3
        ) if solved_path_lengths else None,
        'references_file': references_file,
    }


def run_dataset_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if not os.path.exists(args.targets_file):
        raise FileNotFoundError(f'targets file not found: {args.targets_file}')
    if not os.path.exists(args.stock_file):
        raise FileNotFoundError(f'stock file not found: {args.stock_file}')

    targets = read_txt(args.targets_file)
    stock = read_txt(args.stock_file)

    if args.offset:
        targets = targets[args.offset:]
    if args.limit is not None:
        targets = targets[:args.limit]

    output_root = ensure_dir(args.output_root)
    dataset_root = ensure_dir(os.path.join(output_root, args.dataset_name))

    base_config = MCTSConfig(
        project_name=args.project_name or args.dataset_name,
        max_iterations=args.max_iterations,
        max_simulation_depth=args.max_simulation_depth,
        single_fn_type=args.single_fn_type,
        result_path=os.path.join(dataset_root, 'base_config_placeholder')
    )

    run_rows: List[Dict[str, Any]] = []
    all_route_groups: List[List[Dict[str, Any]]] = []

    for idx, target_smiles in enumerate(targets, start=1):
        sample_id = f'{idx:05d}_{sanitize_filename(target_smiles)}'
        config = create_dataset_config(
            base_config=base_config,
            dataset_name=args.dataset_name,
            target_smiles=target_smiles,
            commercial_molecules=stock,
            output_root=output_root,
            sample_id=sample_id,
        )
        try:
            summary = run_search_with_config(
                config,
                enable_graphs=not args.disable_graphs,
                sample_id=sample_id,
                target_rank=idx,
            )
            paroutes_routes = json2data(summary['paroutes_routes_file']) if os.path.exists(summary['paroutes_routes_file']) else []
        except Exception as exc:
            summary = {
                'sample_id': sample_id,
                'target_rank': idx,
                'project_name': config.project_name,
                'target_molecule': target_smiles,
                'num_commercial_molecules': len(stock),
                'solved': False,
                'num_complete_paths': 0,
                'elapsed_time_sec': None,
                'tree_total_nodes': None,
                'tree_molecule_nodes': None,
                'tree_reaction_nodes': None,
                'pareto_front_size': None,
                'result_path': config.result_path,
                'max_iterations': config.max_iterations,
                'max_simulation_depth': config.max_simulation_depth,
                'single_fn_type': config.single_fn_type,
                'best_path_length': None,
                'best_overall_retro_score': None,
                'best_overall_feasibility': None,
                'best_overall_similarity': None,
                'best_overall_yield': None,
                'best_objective_sum': None,
                'error': str(exc),
            }
            ensure_dir(config.result_path)
            dict2json(summary, os.path.join(config.result_path, 'run_summary.json'))
            paroutes_routes = []
        run_rows.append(summary)
        all_route_groups.append(paroutes_routes)

    summary_csv = os.path.join(dataset_root, 'dataset_summary.csv')
    write_csv(run_rows, summary_csv)

    summary_json = os.path.join(dataset_root, 'dataset_summary.json')
    dict2json(run_rows, summary_json)

    paroutes_json = os.path.join(dataset_root, 'paroutes_routes.json.gz')
    with gzip.open(paroutes_json, 'wt', encoding='utf-8') as fileobj:
        json.dump(all_route_groups, fileobj, indent=2)

    table1_like = build_table1_like_summary(
        dataset_name=args.dataset_name,
        run_rows=run_rows,
        stock_size=len(stock),
        references_file=args.references_file,
    )
    table1_like['paroutes_routes_file'] = paroutes_json
    table1_like['dataset_summary_csv'] = summary_csv
    table1_like['dataset_summary_json'] = summary_json
    table1_path = os.path.join(dataset_root, 'table1_like_summary.json')
    dict2json(table1_like, table1_path)

    run_metadata = {
        'dataset_name': args.dataset_name,
        'targets_file': args.targets_file,
        'stock_file': args.stock_file,
        'references_file': args.references_file,
        'output_root': output_root,
        'limit': args.limit,
        'offset': args.offset,
        'max_iterations': args.max_iterations,
        'max_simulation_depth': args.max_simulation_depth,
        'single_fn_type': args.single_fn_type,
        'disable_graphs': args.disable_graphs,
        'notes': [
            'solved 定义为至少找到一条完整闭合路径',
            'PaRoutes 兼容导出基于当前 MooRetro 线性路径生成递归 route tree',
            '当参考 routes 文件存在时，可继续使用 PaRoutes analysis/route_quality.py 做进一步分析',
        ],
    }
    dict2json(run_metadata, os.path.join(dataset_root, 'run_metadata.json'))

    return table1_like


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run MooRetro single-target or PaRoutes-style dataset experiments.')
    parser.add_argument('--mode', choices=['single', 'dataset'], default='single')
    parser.add_argument('--project-name', default=None)
    parser.add_argument('--target', default=None)
    parser.add_argument('--targets-file', default='/root/PaRoutes/data/n1-targets.txt')
    parser.add_argument('--stock-file', default='/root/PaRoutes/data/n1-stock.txt')
    parser.add_argument('--references-file', default=None)
    parser.add_argument('--dataset-name', default='n1')
    parser.add_argument('--output-root', default='results/paroutes_runs')
    parser.add_argument('--single-fn-type', default='biocyc')
    parser.add_argument('--max-iterations', type=int, default=10)
    parser.add_argument('--max-simulation-depth', type=int, default=10)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--disable-graphs', action='store_true')
    return parser


def test_for_single_fn():
    from utils.single_retro_fn import single_fn

    target_molecule: str = '[NH3+]CCCC[NH3+]'
    commercial_molecules = [
        'NC(=[NH2+])NCCC[C@H]([NH3+])C(=O)[O-]',
    ]
    res = single_fn(target_smile=target_molecule, building_blocks=commercial_molecules)
    print(res)


def test_for_mcts(config: Optional[MCTSConfig] = None, enable_graphs: bool = True):
    mcts_config = config or MCTSConfig()
    return run_search_with_config(mcts_config, enable_graphs=enable_graphs)


if __name__ == '__main__':
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == 'dataset':
        result = run_dataset_experiment(args)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        config = MCTSConfig(
            project_name=args.project_name or 'PWY-43',
            target_molecule=args.target or '[NH3+]CCCC[NH3+]',
            commercial_molecules=MCTSConfig().commercial_molecules,
            max_iterations=args.max_iterations,
            max_simulation_depth=args.max_simulation_depth,
            single_fn_type=args.single_fn_type,
        )
        result = test_for_mcts(config=config, enable_graphs=not args.disable_graphs)
        print(json.dumps(result, indent=2, ensure_ascii=False))