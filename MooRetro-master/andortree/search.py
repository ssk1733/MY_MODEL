"""
 1. 初始化：创建根分子节点（目标分子），加入优先队列。
 2. 循环直到根节点解决或队列为空：
      a. 从优先队列中取出代价最小的节点。
      b. 如果该节点是分子节点且未扩展，则扩展它（生成反应节点和子分子节点）。
      c. 如果该节点是反应节点，则检查它是否解决（即所有子分子节点是否解决）。如果解决，则更新其父分子节点。
      d. 扩展分子节点后，将其子反应节点加入队列，同时将子反应节点的子分子节点加入队列（如果未解决）。
      e. 更新节点代价和解决状态，并传播到祖先节点。
 3. 输出解决路径。
"""

from collections import deque
import heapq
import math
import torch
import random
import logging
import os
import time
import copy
from datetime import datetime
from typing import List, Dict, Set, Tuple, Optional, Callable
import pdb 
import networkx as nx
import matplotlib.pyplot as plt
from graphviz import Digraph

import sys
sys.setrecursionlimit(5000)  # Increase limit to 5000

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  

from utils.chem import similarity_calculator
from utils.single_retro_fn import single_fn_enzretro
from utils.logger import setup_logger
from andortree import ANDORTreeBuilder

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = '0' if torch.cuda.is_available() else 'cpu'


if __name__=='__main__':
    output_dir = 'output'

    # 全局日志对象
    logger = setup_logger(log_dir=os.path.join(output_dir, 'logs'))
    # 配置参数
    # vanillin 
    target = 'COC1=C(C=CC(=C1)C=O)O'
    buyable_mols = ['COc1cc(C(=O)O)ccc1O', # vanillate
                    'C1=CC(=CC=C1/C=C/C(=O)O)[O-]', #4-Coumarate
                    'C1=CC(=C(C=C1C(=O)O)O)[O-]', # 3,4-Dihydroxybenzoate
                    ]
    # target = 'O[C@@H]1C=CC(=C[C@H]1OC(=C)C([O-])=O)C([O-])=O'
    # buyable_mols = ["C=C(OC1C=C(C(=O)OOP(=O)(O)OP(=O)(O)O)C=CC1O)C(=O)O", 
    #                 "O=P(O)(O)OP(=O)(O)OP(=O)(O)OO", "O=C(OO)C1=CCCC=C1"]
    
    # 构建AND-OR树
    builder = ANDORTreeBuilder(
        target_smiles=target,
        buyable_molecules=buyable_mols,
        retro_fn=single_fn_enzretro,
        max_depth=8,
        max_branches=5,
        max_nodes=100,
        strategy="hybrid"
    )

    # 构建树
    root = builder.build_tree()

    # try:
    #     # 可视化整个树
    #     builder.visualize_tree(format='dot')
    # except:
    #     pass

    # 生成完整报告
    # builder.generate_report()

    # 获取解决方案
    # solutions = builder.get_solution_paths()

    
    # 1. 剪枝并可视化（仅保留解决路径）
    builder.logger.info("="*60)
    builder.logger.info("开始剪枝操作...")
    
    if builder.root.solved:
        # 如果有完整解决方案，剪枝保留解决路径
        pruned_path, pruned_count = builder.prune_and_visualize("solution_paths", "svg")
        builder.logger.info(f"完整路径剪枝完成，剪除 {pruned_count} 个节点")
    else:
        # 如果没有完整方案，剪枝保留最佳部分路径
        pruned_path, pruned_count = builder.prune_and_visualize("best_partial", "svg")
        builder.logger.info(f"部分路径剪枝完成，剪除 {pruned_count} 个节点")
    
    # 2. 生成剪枝前后对比
    builder.logger.info("生成剪枝前后对比...")
    original_path, pruned_path, comparison_report = builder.compare_before_after_pruning("solution_paths")
    
    # 3. 可选：尝试不同的剪枝策略
    builder.logger.info("尝试所有剪枝策略...")
    strategies = ["solution_paths", "best_partial", "all_useful"]
    for strategy in strategies:
        try:
            strategy_path, strategy_count = builder.prune_and_visualize(strategy, "svg")
            builder.logger.info(f"策略 {strategy}: 剪除 {strategy_count} 个节点, 输出: {strategy_path}")
        except Exception as e:
            builder.logger.error(f"策略 {strategy} 执行失败: {e}")
    
    builder.logger.info("="*60)
    builder.logger.info("剪枝操作完成！")
    builder.logger.info(f"对比报告: {comparison_report}")