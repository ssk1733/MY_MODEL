import copy
from typing import List, Dict, Set, Tuple, Optional, Callable

from .ANDORTreeBuilder import ANDORTreeBuilder
from .nodes import MoleculeNode, ReactionNode


class ANDORTreePruner:
    """AND-OR树剪枝器"""
    
    def __init__(self, builder: 'ANDORTreeBuilder'):
        self.builder = builder
        self.logger = builder.logger
        
    def prune_tree(self, strategy: str = "solution_paths") -> Tuple[Dict, Dict, int]:
        """
        剪枝AND-OR树，移除无用分支
        
        参数:
            strategy: 剪枝策略
                - "solution_paths": 仅保留完整解决路径
                - "best_partial": 如果无完整路径，保留最佳部分路径
                - "all_useful": 保留所有有用节点（解决的或最高相似度的）
        
        返回:
            (保留的分子节点字典, 保留的反应节点字典, 剪除的节点数)
        """
        self.logger.info(f"开始剪枝，策略: {strategy}")
        
        original_mol_count = len(self.builder.mol_nodes)
        original_rxn_count = len(self.builder.rxn_nodes)
        
        if strategy == "solution_paths":
            kept_mol_nodes, kept_rxn_nodes = self._prune_solution_paths()
        elif strategy == "best_partial":
            kept_mol_nodes, kept_rxn_nodes = self._prune_best_partial()
        elif strategy == "all_useful":
            kept_mol_nodes, kept_rxn_nodes = self._prune_all_useful()
        else:
            raise ValueError(f"未知剪枝策略: {strategy}")
        
        pruned_count = (original_mol_count - len(kept_mol_nodes)) + (original_rxn_count - len(kept_rxn_nodes))
        
        self.logger.info(f"剪枝完成:")
        self.logger.info(f"  原分子节点数: {original_mol_count} -> 保留: {len(kept_mol_nodes)}")
        self.logger.info(f"  原反应节点数: {original_rxn_count} -> 保留: {len(kept_rxn_nodes)}")
        self.logger.info(f"  总剪除节点数: {pruned_count}")
        
        return kept_mol_nodes, kept_rxn_nodes, pruned_count
    
    def _prune_solution_paths(self) -> Tuple[Dict, Dict]:
        """剪枝策略1: 仅保留完整解决路径上的节点"""
        kept_mol_nodes = {}
        kept_rxn_nodes = {}
        
        if not self.builder.root.solved:
            self.logger.warning("根节点未解决，无完整路径可剪枝")
            return {}, {}
        
        # 从根节点开始，递归收集解决路径上的所有节点
        visited = set()
        
        def collect_solution_nodes(mol_node: 'MoleculeNode'):
            if id(mol_node) in visited:
                return
            visited.add(id(mol_node))
            
            # 添加分子节点
            kept_mol_nodes[mol_node.smiles] = mol_node
            
            # 如果是可购买分子，停止递归
            if mol_node.buyable:
                return
            
            # 遍历所有已解决的反应
            for rxn in mol_node.reactions:
                if rxn.solved:
                    # 添加反应节点
                    kept_rxn_nodes[rxn.id] = rxn
                    
                    # 递归处理底物（单个）
                    if rxn.substrate:
                        collect_solution_nodes(rxn.substrate)
        
        collect_solution_nodes(self.builder.root)
        return kept_mol_nodes, kept_rxn_nodes
    
    def _prune_best_partial(self) -> Tuple[Dict, Dict]:
        """剪枝策略2: 保留最佳部分路径"""
        kept_mol_nodes = {}
        kept_rxn_nodes = {}
        
        # 如果有完整解决方案，使用solution_paths策略
        if self.builder.root.solved:
            return self._prune_solution_paths()
        
        # 找到最佳叶节点（最高相似度的未解决节点）
        best_leaves = []
        max_similarity = 0.0
        
        for mol_node in self.builder.mol_nodes.values():
            if not mol_node.solved and not mol_node.expanded:
                if mol_node.max_similarity > max_similarity:
                    max_similarity = mol_node.max_similarity
                    best_leaves = [mol_node]
                elif mol_node.max_similarity == max_similarity:
                    best_leaves.append(mol_node)
        
        self.logger.info(f"找到 {len(best_leaves)} 个最佳叶节点，最高相似度: {max_similarity:.4f}")
        
        # 从每个最佳叶节点回溯到根节点
        visited = set()
        
        def collect_path_to_root(mol_node: 'MoleculeNode'):
            if id(mol_node) in visited:
                return
            visited.add(id(mol_node))
            
            # 添加分子节点
            kept_mol_nodes[mol_node.smiles] = mol_node
            
            # 回溯到父节点
            for parent_rxn in mol_node.parent_reactions:
                if parent_rxn.id not in kept_rxn_nodes:
                    kept_rxn_nodes[parent_rxn.id] = parent_rxn
                
                if parent_rxn.parent and id(parent_rxn.parent) not in visited:
                    collect_path_to_root(parent_rxn.parent)
        
        # 从所有最佳叶节点开始回溯
        for leaf in best_leaves:
            collect_path_to_root(leaf)
        
        return kept_mol_nodes, kept_rxn_nodes
    
    def _prune_all_useful(self) -> Tuple[Dict, Dict]:
        """剪枝策略3: 保留所有有用节点"""
        kept_mol_nodes = {}
        kept_rxn_nodes = {}
        
        # 计算相似度阈值（例如保留前80%的节点）
        similarities = [node.max_similarity for node in self.builder.mol_nodes.values()]
        similarities.sort(reverse=True)
        threshold_index = min(int(len(similarities) * 0.8), len(similarities) - 1)
        similarity_threshold = similarities[threshold_index] if similarities else 0.0
        
        self.logger.info(f"相似度阈值: {similarity_threshold:.4f}")
        
        # 标记有用的分子节点
        useful_mol_nodes = set()
        for mol_node in self.builder.mol_nodes.values():
            if (mol_node.solved or 
                mol_node.buyable or 
                mol_node.max_similarity >= similarity_threshold or
                mol_node == self.builder.root):
                useful_mol_nodes.add(mol_node)
        
        # 从有用节点开始，收集连通的子图
        visited = set()
        
        def collect_connected_nodes(mol_node: 'MoleculeNode'):
            if id(mol_node) in visited:
                return
            visited.add(id(mol_node))
            
            kept_mol_nodes[mol_node.smiles] = mol_node
            
            # 收集相关反应节点
            for rxn in mol_node.reactions:
                if rxn.id not in kept_rxn_nodes:
                    kept_rxn_nodes[rxn.id] = rxn
                
                # 收集底物节点（单个，如果有用）
                if rxn.substrate and rxn.substrate in useful_mol_nodes:
                    collect_connected_nodes(rxn.substrate)
        
        # 从所有有用节点开始收集
        for mol_node in useful_mol_nodes:
            collect_connected_nodes(mol_node)
        
        return kept_mol_nodes, kept_rxn_nodes

    def create_pruned_builder(self, kept_mol_nodes: Dict, kept_rxn_nodes: Dict) -> 'ANDORTreeBuilder':
        """创建剪枝后的构建器实例"""
        # 创建新的构建器实例
        pruned_builder = copy.copy(self.builder)
        
        # 更新节点字典
        pruned_builder.mol_nodes = kept_mol_nodes
        pruned_builder.rxn_nodes = kept_rxn_nodes
        
        # 清理节点间的引用关系，确保一致性
        self._clean_node_references(pruned_builder)
        
        return pruned_builder
    
    def _clean_node_references(self, builder: 'ANDORTreeBuilder'):
        """清理剪枝后节点间的引用关系"""
        kept_mol_smiles = set(builder.mol_nodes.keys())
        kept_rxn_ids = set(builder.rxn_nodes.keys())
        
        # 清理分子节点的反应引用
        for mol_node in builder.mol_nodes.values():
            # 过滤反应列表
            mol_node.reactions = [rxn for rxn in mol_node.reactions if rxn.id in kept_rxn_ids]
            mol_node.parent_reactions = [rxn for rxn in mol_node.parent_reactions if rxn.id in kept_rxn_ids]
        
        # 清理反应节点的分子引用
        for rxn_node in builder.rxn_nodes.values():
            # 检查底物引用（单个）
            if rxn_node.substrate and rxn_node.substrate.smiles not in kept_mol_smiles:
                rxn_node.substrate = None
            
            # 检查父节点引用
            if rxn_node.parent and rxn_node.parent.smiles not in kept_mol_smiles:
                rxn_node.parent = None

