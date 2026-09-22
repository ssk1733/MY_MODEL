#!/usr/bin/env python3
"""
自动增加迭代次数和深度，直到找到匹配参考路线的路径。
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List, Optional, Set

# 导入 MooRetro 的搜索功能
# 注意：需要确保 test.py 在同一个目录，并且已经包含了 run_search_with_config 的定义
from test import run_search_with_config
from config.config import MCTSConfig

# RDKit 相关
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# ---------- 辅助函数（复制自之前的 test.py，确保存在）----------
_fp_cache = {}

def get_morgan_fingerprint(smiles: str, radius: int = 2, nbits: int = 2048):
    if smiles in _fp_cache:
        return _fp_cache[smiles]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nbits)
    _fp_cache[smiles] = fp
    return fp

def tanimoto_similarity(fp1, fp2):
    if fp1 is None or fp2 is None:
        return 0.0
    intersection = fp1 & fp2
    union = fp1 | fp2
    return intersection.Count() / union.Count() if union.Count() != 0 else 0.0

def reactants_set_similarity(reactants1: Set[str], reactants2: Set[str], threshold: float = 0.99) -> bool:
    if len(reactants1) != len(reactants2):
        return False
    list1 = list(reactants1)
    list2 = list(reactants2)
    n = len(list1)
    sim_matrix = []
    for smi1 in list1:
        fp1 = get_morgan_fingerprint(smi1)
        row = []
        for smi2 in list2:
            fp2 = get_morgan_fingerprint(smi2)
            sim = tanimoto_similarity(fp1, fp2)
            row.append(sim)
        sim_matrix.append(row)
    used = [False] * n
    for i in range(n):
        best_j = -1
        best_sim = -1.0
        for j in range(n):
            if not used[j] and sim_matrix[i][j] > best_sim:
                best_sim = sim_matrix[i][j]
                best_j = j
        if best_j == -1 or best_sim < threshold:
            return False
        used[best_j] = True
    return True

def extract_product_and_reactants_from_route_tree(route_tree: Dict[str, Any]):
    product = route_tree.get("smiles", "")
    if not product:
        return None, set()
    leaf_smiles = set()
    def traverse(node):
        if not node.get("children"):
            smi = node.get("smiles")
            if smi:
                leaf_smiles.add(smi)
        else:
            for child in node["children"]:
                if child.get("type") == "reaction":
                    for grandchild in child.get("children", []):
                        traverse(grandchild)
                else:
                    traverse(child)
    traverse(route_tree)
    return product, leaf_smiles

def load_reference_routes(references_file: str) -> Dict[str, List[Dict[str, Any]]]:
    if not references_file or not os.path.exists(references_file):
        return {}
    with open(references_file, 'r') as f:
        data = json.load(f)
    result = {}
    if isinstance(data, list):
        for route_tree in data:
            product = route_tree.get("smiles")
            if product:
                result.setdefault(product, []).append(route_tree)
    elif isinstance(data, dict):
        for product, routes_data in data.items():
            if isinstance(routes_data, list):
                result[product] = routes_data
            else:
                result[product] = [routes_data]
    return result

def path_matches_reference_from_tree(candidate_tree: Dict[str, Any],
                                     reference_trees: List[Dict[str, Any]],
                                     threshold: float = 0.99) -> bool:
    cand_prod, cand_react = extract_product_and_reactants_from_route_tree(candidate_tree)
    if not cand_prod:
        return False
    for ref_tree in reference_trees:
        ref_prod, ref_react = extract_product_and_reactants_from_route_tree(ref_tree)
        if len(cand_react) != len(ref_react):
            continue
        # 产物相似度
        prod_sim = tanimoto_similarity(get_morgan_fingerprint(cand_prod), get_morgan_fingerprint(ref_prod))
        if prod_sim < threshold:
            continue
        if reactants_set_similarity(cand_react, ref_react, threshold):
            return True
    return False
# ---------- 辅助函数结束 ----------

def find_matching_route(target_smiles: str,
                        stock_file: str,
                        references_file: str,
                        start_iter: int = 10,
                        step_iter: int = 20,
                        max_iter: int = 250,
                        start_depth: int = 5,
                        step_depth: int = 2,
                        max_depth: int = 12,
                        similarity_threshold: float = 0.99,
                        output_root: str = "results/auto_search"):
    # 加载参考路线
    ref_map = load_reference_routes(references_file)
    if target_smiles not in ref_map:
        print(f"错误：参考文件中未找到目标分子 {target_smiles}")
        return None
    ref_routes = ref_map[target_smiles]
    print(f"目标分子: {target_smiles}")
    print(f"参考路线数量: {len(ref_routes)}")

    # 读取商业分子列表
    with open(stock_file, 'r') as f:
        commercial_molecules = [line.strip() for line in f if line.strip()]

    # 尝试参数组合
    iter_vals = list(range(start_iter, max_iter + 1, step_iter))
    depth_vals = list(range(start_depth, max_depth + 1, step_depth))
    if iter_vals[-1] != max_iter:
        iter_vals.append(max_iter)
    if depth_vals[-1] != max_depth:
        depth_vals.append(max_depth)

    for depth in depth_vals:
        for iters in iter_vals:
            print(f"\n尝试: max_iterations={iters}, max_simulation_depth={depth}")
            result_path = os.path.join(output_root, f"iter{iters}_depth{depth}")
            config = MCTSConfig(
                project_name=f"auto_{target_smiles[:20]}",
                target_molecule=target_smiles,
                commercial_molecules=commercial_molecules,
                max_iterations=iters,
                max_simulation_depth=depth,
                single_fn_type="biocyc",
                result_path=result_path
            )
            try:
                start_time = time.time()
                summary = run_search_with_config(config, enable_graphs=False)
                elapsed = time.time() - start_time
                print(f"  搜索完成，耗时 {elapsed:.1f} 秒")
                solved = summary.get('solved', False)
                print(f"  solved: {solved}")
                if not solved:
                    print("  未找到完整路径，继续尝试")
                    continue

                # 获取候选路径树
                paroutes_file = summary.get('paroutes_routes_file')
                if not paroutes_file or not os.path.exists(paroutes_file):
                    print("  未找到 paroutes_routes.json，无法评估匹配")
                    continue
                with open(paroutes_file, 'r') as f:
                    candidate_trees = json.load(f)
                if not candidate_trees:
                    print("  候选路径为空")
                    continue

                matched = False
                for cand_tree in candidate_trees:
                    if path_matches_reference_from_tree(cand_tree, ref_routes, similarity_threshold):
                        matched = True
                        # 输出匹配的原料集合
                        _, reactants = extract_product_and_reactants_from_route_tree(cand_tree)
                        print(f"  ✓ 找到匹配的路径！原料集合: {reactants}")
                        break
                if matched:
                    print(f"\n成功！在 max_iterations={iters}, max_simulation_depth={depth} 下找到匹配路径。")
                    print(f"结果保存在: {result_path}")
                    return summary
                else:
                    print("  未找到匹配的路径，继续尝试")
            except Exception as e:
                print(f"  运行出错: {e}")
                continue
    print("\n所有尝试均未找到匹配路径。")
    return None

def main():
    parser = argparse.ArgumentParser(description='自动增加迭代次数和深度，直到找到匹配参考路线的路径。')
    parser.add_argument('--target', required=True, help='目标分子 SMILES')
    parser.add_argument('--stock_file', required=True, help='商业分子列表文件')
    parser.add_argument('--references_file', required=True, help='参考路线文件（n1-routes.json 或 n5-routes.json）')
    parser.add_argument('--start_iter', type=int, default=10, help='起始迭代次数')
    parser.add_argument('--step_iter', type=int, default=20, help='每次增加的迭代次数')
    parser.add_argument('--max_iter', type=int, default=250, help='最大迭代次数')
    parser.add_argument('--start_depth', type=int, default=5, help='起始搜索深度')
    parser.add_argument('--step_depth', type=int, default=2, help='每次增加的深度')
    parser.add_argument('--max_depth', type=int, default=12, help='最大搜索深度')
    parser.add_argument('--threshold', type=float, default=0.99, help='指纹相似度阈值')
    parser.add_argument('--output_root', default='results/auto_search', help='结果根目录')
    args = parser.parse_args()

    find_matching_route(
        target_smiles=args.target,
        stock_file=args.stock_file,
        references_file=args.references_file,
        start_iter=args.start_iter,
        step_iter=args.step_iter,
        max_iter=args.max_iter,
        start_depth=args.start_depth,
        step_depth=args.step_depth,
        max_depth=args.max_depth,
        similarity_threshold=args.threshold,
        output_root=args.output_root
    )

if __name__ == '__main__':
    main()