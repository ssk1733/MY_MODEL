#!/usr/bin/env python3
"""
独立脚本：从已有的数据集结果中计算所有 solved 目标的平均指标。
用法：
    python compute_avg_from_results.py --summary_file /path/to/dataset_summary.json
或指定结果根目录自动查找所有 run_summary.json：
    python compute_avg_from_results.py --results_root /path/to/results/paroutes_runs/n5_smallscale
"""

import argparse
import json
import os
from typing import List, Dict, Any


def load_from_dataset_summary(summary_file: str) -> List[Dict[str, Any]]:
    """从 dataset_summary.json 加载所有目标的汇总数据"""
    with open(summary_file, 'r') as f:
        rows = json.load(f)
    return rows


def load_from_results_root(results_root: str) -> List[Dict[str, Any]]:
    """遍历结果根目录，收集所有 run_summary.json 文件"""
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


def compute_averages(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """计算所有 solved 目标的平均指标"""
    solved_rows = [r for r in rows if r.get('solved', False)]
    if not solved_rows:
        print("没有找到 solved 的目标")
        return {}

    n = len(solved_rows)
    avg = {
        'num_solved': n,
        'avg_path_length': 0.0,
        'avg_retro_score': 0.0,
        'avg_feasibility': 0.0,
        'avg_similarity': 0.0,
        'avg_yield': 0.0,
        'avg_objective_sum': 0.0,
    }

    for r in solved_rows:
        # 优先使用平均指标（如果存在），否则回退到最佳指标（兼容旧格式）
        path_len = r.get('avg_path_length')
        if path_len is None:
            path_len = r.get('best_path_length')
        avg['avg_path_length'] += path_len if path_len else 0

        retro = r.get('avg_overall_retro_score')
        if retro is None:
            retro = r.get('best_overall_retro_score')
        avg['avg_retro_score'] += retro if retro else 0

        feas = r.get('avg_overall_feasibility')
        if feas is None:
            feas = r.get('best_overall_feasibility')
        avg['avg_feasibility'] += feas if feas else 0

        sim = r.get('avg_overall_similarity')
        if sim is None:
            sim = r.get('best_overall_similarity')
        avg['avg_similarity'] += sim if sim else 0

        yld = r.get('avg_overall_yield')
        if yld is None:
            yld = r.get('best_overall_yield')
        avg['avg_yield'] += yld if yld else 0

        obj = r.get('avg_objective_sum')
        if obj is None:
            obj = r.get('best_objective_sum')
        avg['avg_objective_sum'] += obj if obj else 0

    for key in avg:
        if key != 'num_solved':
            avg[key] = round(avg[key] / n, 4)

    return avg


def main():
    parser = argparse.ArgumentParser(description='计算已有结果的平均指标')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--summary_file', help='dataset_summary.json 文件路径')
    group.add_argument('--results_root', help='包含多个目标子目录的结果根目录')
    parser.add_argument('--output_json', help='可选：将平均值保存为 JSON 文件')
    args = parser.parse_args()

    if args.summary_file:
        rows = load_from_dataset_summary(args.summary_file)
    else:
        rows = load_from_results_root(args.results_root)

    if not rows:
        print("未找到任何结果数据")
        return

    avg = compute_averages(rows)
    if not avg:
        return

    print("\n========== Solved 目标平均指标 ==========")
    print(f"Solved 目标数: {avg['num_solved']}")
    print(f"平均路径长度: {avg['avg_path_length']}")
    print(f"平均 Retro Score: {avg['avg_retro_score']}")
    print(f"平均 Feasibility: {avg['avg_feasibility']}")
    print(f"平均 Similarity: {avg['avg_similarity']}")
    print(f"平均 Yield: {avg['avg_yield']}")
    print(f"平均 Objective Sum: {avg['avg_objective_sum']}")
    print("========================================\n")

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(avg, f, indent=2)
        print(f"平均值已保存至: {args.output_json}")


if __name__ == '__main__':
    main()
