# dfs_searcher.py - 基于深度优先搜索的合成路径搜索器
import numpy as np
import time
import os
from typing import List, Optional
from collections import deque

from src.synthesis_tree import SynthesisTree, SynthesisPath
from src.evaluator import PathEvaluator
from config.config import MCTSConfig, SingleStepResult
from src.analyzer import Analyzer
from src.nodes import Node, MoleculeNode, ReactionNode
from utils.logger import setup_logger

class DFSSearcher:
    """基于深度优先搜索的合成路径搜索器"""
    
    def __init__(self, 
                 tree: SynthesisTree,
                 single_fn_function,
                 evaluator: PathEvaluator,
                 analyzer: Analyzer,
                 config: Optional[MCTSConfig] = None,
                 logger=None):
        
        self.tree = tree
        self.single_fn = single_fn_function
        self.evaluator = evaluator
        self.config = config or MCTSConfig()
        self.analyzer = analyzer
        self.logger = logger or setup_logger(
            os.path.join(self.config.result_path, f'dfs_search_logs_{time.time()}'))
        
        # 搜索状态
        self.best_paths = []
        self.global_pareto_front = []
        self.start_time = None
        
    def search(self, max_depth: Optional[int] = None) -> SynthesisTree:
        """执行DFS搜索"""
        max_depth = max_depth or self.config.max_simulation_depth
        self.start_time = time.time()
        self.logger.info(f"开始DFS搜索: 目标分子 {self.tree.target}")
        self.logger.info(f"最大深度: {max_depth}")
        
        # 初始化搜索队列 (分子, 当前路径)
        queue = deque([(self.tree.root, SynthesisPath([self.tree.root.molecule], []))])
        
        while queue:
            current_node, current_path = queue.popleft()
            
            # 检查是否达到最大深度
            if current_node.depth >= max_depth:
                continue
                
            # 如果当前节点是商业分子，记录完整路径
            if current_node.is_commercial:
                self._record_complete_path(current_path, current_node)
                continue
            
            # 获取可能的反应
            reactions = self._get_reactions(current_node)
            
            for reaction in reactions:
                # 创建新路径
                new_path = self._create_new_path(current_path, reaction)
                
                # 添加反应物到队列
                for reactant in reaction.children:
                    queue.append((reactant, new_path))
                    
            # 定期反馈进度
            if len(self.best_paths) % self.config.real_time_feedback_interval == 0:
                self._provide_real_time_feedback()
        
        # 完成搜索
        self._finalize_search_results()
        return self.tree
    
    def _get_reactions(self, mol_node: MoleculeNode) -> List[ReactionNode]:
        """获取可能的反应（扩展节点如果必要）"""
        # 如果节点未扩展，尝试扩展
        if mol_node.is_leaf() and not mol_node.is_commercial:
            reactions = self.single_fn(mol_node.molecule, list(self.tree.commercial_molecules))
            if reactions:
                for reaction in reactions:
                    self.tree.add_reaction(mol_node.molecule, reaction)
        
        # 返回所有可用反应
        return [child for child in mol_node.children if isinstance(child, ReactionNode)]
    
    def _create_new_path(self, current_path: SynthesisPath, reaction: ReactionNode) -> SynthesisPath:
        """创建新路径"""
        new_molecules = current_path.molecules.copy()
        new_reactions = current_path.reactions.copy()
        
        # 添加反应
        new_reactions.append(reaction.reaction_info)
        
        # 添加反应物（对于单产物反应）
        for reactant in reaction.children:
            new_molecules.append(reactant.molecule)
        
        return SynthesisPath(new_molecules, new_reactions)
    
    def _record_complete_path(self, path: SynthesisPath, node: MoleculeNode):
        """记录完整路径并更新Pareto前沿"""
        # 评估路径
        objective_vector = self.evaluator.evaluate_path(
            path.molecules, path.reactions, is_complete=True
        )
        path.objective_values = objective_vector
        path.is_complete = True
        
        # 记录到树中
        self.tree.record_path(path, is_complete=True)
        
        # 添加到最佳路径列表
        self.best_paths.append(path)
        
        # 更新全局Pareto前沿
        self._update_global_pareto_front(objective_vector)
        
        self.logger.info(f"找到完整路径: {path.molecules}")
        self.logger.info(f"目标值: {objective_vector}")
    
    def _update_global_pareto_front(self, objective_vector: np.ndarray):
        """更新全局Pareto前沿"""
        self.global_pareto_front.append(objective_vector.tolist())
        
        # 限制前沿大小
        if len(self.global_pareto_front) > self.evaluator.config.max_pareto_front_size:
            self.global_pareto_front = self.global_pareto_front[-self.evaluator.config.max_pareto_front_size:]
    
    def _provide_real_time_feedback(self):
        """提供实时反馈"""
        elapsed = time.time() - self.start_time
        self.logger.info(f"=== 搜索进度 ===")
        self.logger.info(f"已找到路径: {len(self.best_paths)}")
        self.logger.info(f"运行时间: {elapsed:.2f}秒")
        self.logger.info(f"树大小: {self.tree.get_statistics().total_nodes}")
    
    def _finalize_search_results(self):
        """完成搜索结果设置"""
        elapsed = time.time() - self.start_time
        tree_stats = self.tree.get_statistics()
        
        self.logger.info(f"========== 搜索完成 ==========")
        self.logger.info(f"总运行时间: {elapsed:.2f}秒")
        self.logger.info(f"找到完整路径: {len(self.best_paths)}")
        self.logger.info(f"树规模 - 分子节点: {tree_stats.molecule_nodes}, 反应节点: {tree_stats.reaction_nodes}")
        
        if self.analyzer:
            self.analyzer.quick_analysis(self.tree.target, self.tree)