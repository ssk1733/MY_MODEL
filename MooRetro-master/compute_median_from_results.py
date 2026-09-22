#!/usr/bin/env python3
"""
独立脚本：从已有的数据集结果中计算所有 solved 目标的中位指标（不包含 Yield）。
用法：
    python compute_median_no_yield.py --summary_file /path/to/dataset_summary.json
或：
    python compute_median_no_yield.py --results_root /path/to/results/paroutes_runs/n1_top100-deep_noP
"""

import argparse
import json
import os
import statistics
from typing import List, Dict, Any


def load_from_dataset_summary(summary_file: str) -> List[Dict[str, Any]]:
    with open(summary_file, 'r') as f:
        rows = json.load(f)
    return rows


def load_from_results_root(results_root: str) -> List[Dict[str, Any]]:
    rows = []
    for subdir in os.listdir(results_root):
        subdir_path = os.path.join(results_root, subdir)
        if not os.path.isdir(subdir_path):
            continue
        summary_path = os.path.join(subdir_path, 'run_summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                rows.append(summary)
    return rows


def compute_medians_no_yield(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    solved_rows = [r for r in rows if r.get('solved', False)]
    if not solved_rows:
        print("没有找到 solved 的目标")
        return {}

    # 收集各个指标的列表
    path_lengths = []
    retro_scores = []
    feasibilities = []
    similarities = []
    complete_paths_counts = []
    recomputed_objective_sums = []   # 新综合分数 = retro + feas + sim

    for r in solved_rows:
        # 路径长度
        pl = r.get('avg_path_length')
        if pl is None:
            pl = r.get('best_path_length')
        if pl is not None:
            path_lengths.append(pl)

        # Retro Score
        rs = r.get('avg_overall_retro_score')
        if rs is None:
            rs = r.get('best_overall_retro_score')
        if rs is not None:
            retro_scores.append(rs)

        # Feasibility
        fs = r.get('avg_overall_feasibility')
        if fs is None:
            fs = r.get('best_overall_feasibility')
        if fs is not None:
            feasibilities.append(fs)

        # Similarity
        sim = r.get('avg_overall_similarity')
        if sim is None:
            sim = r.get('best_overall_similarity')
        if sim is not None:
            similarities.append(sim)

        # 完整路径个数
        cp = r.get('num_complete_paths', 0)
        if cp is not None:
            complete_paths_counts.append(cp)

        # 重新计算综合分数（不含 yield）
        if rs is not None and fs is not None and sim is not None:
            new_obj_sum = rs + fs + sim
            recomputed_objective_sums.append(new_obj_sum)

    median = {
        'num_solved': len(solved_rows),
        'median_path_length': round(statistics.median(path_lengths), 4) if path_lengths else 0,
        'median_retro_score': round(statistics.median(retro_scores), 4) if retro_scores else 0,
        'median_feasibility': round(statistics.median(feasibilities), 4) if feasibilities else 0,
        'median_similarity': round(statistics.median(similarities), 4) if similarities else 0,
        'median_objective_sum_no_yield': round(statistics.median(recomputed_objective_sums), 6) if recomputed_objective_sums else 0,
        'median_num_complete_paths': round(statistics.median(complete_paths_counts), 2) if complete_paths_counts else 0,
    }
    return median


def main():
    parser = argparse.ArgumentParser(description='计算已有结果的中位指标（不含 Yield）')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--summary_file', help='dataset_summary.json 文件路径')
    group.add_argument('--results_root', help='包含多个目标子目录的结果根目录')
    parser.add_argument('--output_json', help='可选：将中位指标保存为 JSON 文件')
    args = parser.parse_args()

    if args.summary_file:
        rows = load_from_dataset_summary(args.summary_file)
    else:
        rows = load_from_results_root(args.results_root)

    if not rows:
        print("未找到任何结果数据")
        return

    median = compute_medians_no_yield(rows)
    if not median:
        return

    print("\n========== Solved 目标中位指标（不含 Yield）==========")
    print(f"Solved 目标数: {median['num_solved']}")
    print(f"中位路径长度: {median['median_path_length']}")
    print(f"中位 Retro Score: {median['median_retro_score']}")
    print(f"中位 Feasibility: {median['median_feasibility']}")
    print(f"中位 Similarity: {median['median_similarity']}")
    print(f"中位综合分数 (Retro+Feas+Sim): {median['median_objective_sum_no_yield']}")
    print(f"中位完整路径数: {median['median_num_complete_paths']}")
    print("====================================================\n")

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(median, f, indent=2)
        print(f"中位指标已保存至: {args.output_json}")


if __name__ == '__main__':
    main()