#!/usr/bin/env python3
"""
调试脚本：打印数据集中第一个目标的候选原料集合和参考原料集合
用法: python debug_first_target.py --dataset_summary /path/to/dataset_summary.json --references_file /path/to/n1-routes.json
"""

import argparse
import json
import os
import sys
from typing import Set

# 导入已有的辅助函数（需要确保 test.py 在同一目录或可导入）
# 如果没有，我们在这里重新定义必要的函数（简化版）
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

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

def extract_reactants_from_path(path_molecules):
    """从 SynthesisPath.molecules 提取原料 SMILES 集合"""
    last = path_molecules[-1]
    if isinstance(last, list):
        return set(last)
    else:
        return {last}

def extract_product_and_reactants_from_route_tree(route_tree):
    """从 PaRoutes 树中提取产物和原料集合"""
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

def load_reference_routes(references_file):
    """加载参考文件，返回 {target_smiles: [route_trees]}"""
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_summary', required=True, help='dataset_summary.json 路径')
    parser.add_argument('--references_file', required=True, help='参考路线文件路径')
    parser.add_argument('--target_index', type=int, default=0, help='目标索引（从0开始）')
    args = parser.parse_args()

    # 读取数据集汇总
    with open(args.dataset_summary, 'r') as f:
        rows = json.load(f)
    if not rows:
        print("数据集为空")
        return
    if args.target_index >= len(rows):
        print(f"索引超出范围，共 {len(rows)} 个目标")
        return
    row = rows[args.target_index]
    target_smiles = row['target_molecule']
    result_path = row['result_path']
    print(f"目标分子: {target_smiles}")
    print(f"结果目录: {result_path}")
    print(f"solved: {row.get('solved')}")
    if not row.get('solved'):
        print("该目标未 solved，无法提取候选路径")
        return

    # 读取候选路径的 paroutes_routes.json
    paroutes_file = os.path.join(result_path, 'paroutes_routes.json')
    if not os.path.exists(paroutes_file):
        print(f"未找到 {paroutes_file}")
        return
    with open(paroutes_file, 'r') as f:
        route_trees = json.load(f)  # 列表，每个元素是一棵树
    print(f"\n候选路径数量: {len(route_trees)}")
    for i, tree in enumerate(route_trees):
        prod, reactants = extract_product_and_reactants_from_route_tree(tree)
        print(f"  候选路径 {i+1}: 产物 = {prod}")
        print(f"              原料 = {reactants}")

    # 加载参考路线
    ref_map = load_reference_routes(args.references_file)
    if target_smiles not in ref_map:
        print(f"\n参考文件中未找到目标分子: {target_smiles}")
        return
    ref_routes = ref_map[target_smiles]
    print(f"\n参考路线数量: {len(ref_routes)}")
    for j, tree in enumerate(ref_routes):
        prod, reactants = extract_product_and_reactants_from_route_tree(tree)
        print(f"  参考路线 {j+1}: 产物 = {prod}")
        print(f"              原料 = {reactants}")

    # 可选：计算候选与参考之间的最大相似度
    print("\n=== 相似度评估（原料集合大小必须相等才可能匹配） ===")
    for i, cand_tree in enumerate(route_trees):
        cand_prod, cand_react = extract_product_and_reactants_from_route_tree(cand_tree)
        for j, ref_tree in enumerate(ref_routes):
            ref_prod, ref_react = extract_product_and_reactants_from_route_tree(ref_tree)
            if len(cand_react) != len(ref_react):
                print(f"候选{i+1} vs 参考{j+1}: 原料数量不同 ({len(cand_react)} vs {len(ref_react)})，跳过")
                continue
            # 产物相似度
            fp_cand = get_morgan_fingerprint(cand_prod)
            fp_ref = get_morgan_fingerprint(ref_prod)
            prod_sim = tanimoto_similarity(fp_cand, fp_ref)
            print(f"候选{i+1} vs 参考{j+1}: 产物相似度 = {prod_sim:.4f}")
            # 原料集合匹配度（简单比较字符串）
            if cand_react == ref_react:
                print("  原料集合完全相等（字符串）")
            else:
                print(f"  原料集合不同: cand={cand_react}, ref={ref_react}")

if __name__ == '__main__':
    main()
