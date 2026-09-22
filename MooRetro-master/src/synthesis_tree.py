# synthesis_tree.py - 重构的合成树管理器，专注于树级别的操作和管理
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Iterator, Callable
from dataclasses import dataclass
from collections import defaultdict, deque
import time
import logging
import os
import pickle
import json
from dataclasses import dataclass, asdict, is_dataclass

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV

from config.config import SingleStepResult, EvaluationConfig
from src.nodes import Node, MoleculeNode, ReactionNode
from src.evaluator import PathMetrics
from utils.chem import similarity_calculator, canonicalize_mol_smi
from utils.utils import dict2json, data2pkl


@dataclass
class SynthesisPath:
    """合成路径表示 - 增强版本"""
    molecule_nodes: List[MoleculeNode]
    reaction_nodes: List[ReactionNode]
    objective_values: Optional[List] = None
    path_id: Optional[str] = None
    is_complete: bool = False
    _connections: Optional[List]= None
    _metrics: Optional[PathMetrics] = None

    @property
    def length(self) -> int:
        return len(self.reaction_nodes)
    
    @property
    def molecules(self) -> List[str]:
        return [mn.molecule for mn in self.molecule_nodes]
    
    @property
    def reactions(self) -> List[SingleStepResult]:
        return [rn.reaction_info for rn in self.reaction_nodes]
    
    @property
    def connections(self):
        if self._connections:
            return self._connections
        assert len(self.molecule_nodes)-len(self.reaction_nodes)==1
        connection = []
        for i, rxn in enumerate(self.reaction_nodes):
            connection.append({
                'id': f"{self.molecule_nodes[i].id}_{rxn.id}",
                'source': self.molecule_nodes[i].id,
                'target': rxn.id,
                'label': f"Score:{rxn.reaction_info.scores}\n{rxn.reaction_info.top_k}"
            })
            connection.append({
                'id': f"{rxn.id}_{self.molecule_nodes[i+1].id}",
                'source': rxn.id,
                'target': self.molecule_nodes[i+1].id,
                'label': ''
            })
        return connection
    
    @property
    def metrics(self) -> Dict:
        if self._metrics:
            return self._metrics.to_dict
        return {}

    @property
    def path_to_string(self):
        res = [f'{m.molecule}>{r.reaction_info.templates}' for m, r in zip(self.molecule_nodes[:-1], self.reaction_nodes)]
        return ">".join(res) + f'>{self.molecule_nodes[-1].molecule}'

    @property
    def to_dict(self):
        return {
            # 'is_complete': self.is_complete,
            # 'objective_values': self.objective_values,
            # 'path_string': self.path_to_string,
            'molecules': [m.to_dict for m in self.molecule_nodes],
            'reactions': [r.to_dict for r in self.reaction_nodes],
            'connection': self.connections,
            'metrics': self.metrics
        }
    


@dataclass
class TreeStatistics:
    """树统计信息"""
    total_nodes: int = 0
    molecule_nodes: int = 0
    reaction_nodes: int = 0
    connection_edges: int = 0
    max_depth: int = 0
    total_visits: int = 0
    num_complete_paths : int = 0


class SynthesisTree:
    """AND-OR合成树管理器 - 专注于树级别的操作和优化"""
    
    def __init__(self, target_molecule: str, commercial_molecules: List[str], 
                 config: Optional[EvaluationConfig] = None, logger=None):
        self.target = target_molecule
        self.commercial_molecules = list(set(commercial_molecules))
        self.config = config or EvaluationConfig()

        # 缓存和性能优化
        self.similarity_cache: Dict = {}
        # 节点索引 - 用于快速查找
        self.molecule_nodes: Dict[str, MoleculeNode] = {}
        self.reaction_nodes: Dict[str, ReactionNode] = {}
        self.connection_set = set()
        self.node_counter = 0
        
        # 树结构
        self.root = self._create_molecule_node(
            target_molecule,
            # is_commercial=self._is_commercial(target_molecule),
            depth=0,
            similarity=self._calculate_similarity(target_molecule)
        )

        # # 路径管理
        self.all_paths: List[SynthesisPath] = []
        self.complete_paths: List[SynthesisPath] = []
        self.complete_path_counter = 0
        self.path_counter = 0
        
        # 日志
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

    # ========================================================================
    # 树构建和修改方法
    # ========================================================================
    def _create_molecule_node(self, molecule: str, 
                            #   is_commercial: bool, 
                              depth: int, similarity: float=0.0) -> MoleculeNode:
        """创建分子节点并添加到索引"""
        node_id = f"mol_{self.node_counter}"
        self.node_counter += 1
        # reaction_info.similarity describes the reaction's primary reactant.
        # A reaction can contain multiple reactants, so that value must not be
        # reused to decide whether every sibling reactant is in stock.
        actual_similarity = self._calculate_similarity(molecule)
        is_commercial = actual_similarity > 0.99
        node = MoleculeNode(
            molecule=molecule,
            is_commercial=is_commercial,
            similarity_to_commercial=actual_similarity,
            depth=depth,
            node_id=node_id
        )
        
        self.molecule_nodes[node_id] = node
        return node

    
    def add_reaction(self, parent_node: MoleculeNode,
                    #  parent_molecule: str, 
                     reaction_info: SingleStepResult) -> ReactionNode:
        """添加反应到指定分子节点"""
        
        # 创建反应节点
        reaction_id = f"rxn_{self.node_counter}"
        self.node_counter += 1
        
        reaction_node = ReactionNode(
            reaction_info=reaction_info,
            parent_molecule=parent_node.molecule,
            depth=parent_node.depth + 1,
            node_id=reaction_id
        )
        
        # 链接到父节点
        parent_node.add_child(reaction_node)
        self.reaction_nodes[reaction_id] = reaction_node
        # mol -> rxn (single_score)
        self.connection_set.add((
            parent_node.id, reaction_node.id, reaction_info.scores
        ))
        
        # 添加反应物节点
        for reactant in reaction_info.reactants:
            # 创建新的分子节点（即使分子相同也创建新节点）
            reactant_node = self._create_molecule_node(
                reactant,
                # is_commercial=self._is_commercial(reactant),
                depth=reaction_node.depth + 1,
                similarity=reaction_info.similarity,
            )
            reaction_node.add_child(reactant_node)
            # rxn -> mol (similarity)
            self.connection_set.add((
                reaction_node.id, reactant_node.id, reaction_info.similarity
            ))
        
        return reaction_node
    
    def get_molecule_node_id(self, molecule: str) -> Optional[str]:
        """查找特定深度下的分子节点"""
        nodes = []
        for node in self.molecule_nodes.values():
            if node.molecule == molecule:
                nodes.append(node)
        if nodes:
            nodes.sort(key=lambda x: x.depth)
            return nodes[0].id
        return None
    

    # ========================================================================
    # 路径提取和管理
    # ========================================================================
    
    def get_path_to_node(self, node: Node) -> SynthesisPath:
        """获取从根到指定节点的路径"""
        molecules = []
        reactions = []
        connection_set = set()
        
        # 回溯路径
        path_nodes = []
        current = node
        while current:
            path_nodes.append(current)
            if current.parent:
                connection_set.add((
                    current.parent.id, current.id
                ))
            current = current.parent
        path_nodes.reverse()
        
        # 提取分子和反应
        for _, path_node in enumerate(path_nodes):
            if isinstance(path_node, MoleculeNode):
                molecules.append(path_node)
            elif isinstance(path_node, ReactionNode):
                reactions.append(path_node)
        
        # 检查路径完整性
        is_complete = self._is_path_complete(node)
        
        return SynthesisPath(
            molecule_nodes=molecules,
            reaction_nodes=reactions,
            is_complete=is_complete,
        )
    
    def record_path(self, path: SynthesisPath, is_complete: bool = False):
        if not path:
            return
        
        self.path_counter += 1 
        path.path_id = f'path_{self.path_counter}'
        self.all_paths.append(path)

        if is_complete:
            self.complete_path_counter += 1
            self.complete_paths.append(path)

    
    # ========================================================================
    # 树遍历和查询方法
    # ========================================================================
    
    def collect_tree_info(self) -> Tuple[Optional[List[Dict]], Optional[List[Dict]], Optional[List[Dict]]]:
        """获取所有节点"""
        molecule_info = []
        reaction_info = []
        connection_set = set()

        # 跟踪已访问过的MCTS树节点对象ID，避免重复处理和压栈，这是关键的性能优化。
        visited_nodes = set()

        # 栈遍历收集数据
        stack = [(self.root, None, 0)]

        while stack:
            node, parent_id, depth = stack.pop()

            # if depth > max_depth:
            #     continue

            node_id = f"{f'{node.id}_{node.visits}' if isinstance(node, MoleculeNode) else f'rxn_{id(node)}'}"
            # node_id = node.id
            
            if node_id not in visited_nodes:
                visited_nodes.add(node_id)
            
                if isinstance(node, MoleculeNode):
                    node_info = node.to_dict
                    node_info.update({'id': node_id})
                    molecule_info.append(node_info)

                else: # ReactionNode
                    node_info = node.to_dict
                    node_info.update({'id': node_id})
                    reaction_info.append(node_info)

            if parent_id:
                label = ""
                if isinstance(node, ReactionNode):
                    rxn_info = getattr(node, 'reaction_info', None)
                    if rxn_info:
                        label = f"Score: {rxn_info.scores:.3f}\nTok_k: {rxn_info.top_k}"

                connection_set.add((
                    parent_id,
                    node_id,
                    label
                )) 

            # --- 将子节点【逆序】压入栈中 ---
            # 逆序压栈可以确保当它们被弹出时，处理顺序与原始的子节点顺序一致（从左到右）。
            # 这是DFS的经典技巧。
            for child in reversed(node.children):
                if id(child) not in visited_nodes:
                    stack.append((child, node_id, depth+1))

        molecule_info.sort(key=lambda x: x['depth'])
        reaction_info.sort(key=lambda x: x['depth'])

        connection_info = [
            {
                'id': f"{parent_id}_{child_id}",
                'source': parent_id,
                'target': child_id,
                'label': label
            }
            for parent_id, child_id, label in connection_set
        ]
        return molecule_info, reaction_info, connection_info
    
    # ========================================================================
    # 统计和分析方法
    # ========================================================================
    
    def get_statistics(self, force_update: bool = False) -> TreeStatistics:        
        """获取树统计信息"""
        mol_nodes = list(self.molecule_nodes.values())
        rxn_nodes = list(self.reaction_nodes.values())
        all_nodes = mol_nodes + rxn_nodes
        connection = list(self.connection_set)
        return TreeStatistics(
            total_nodes=len(all_nodes),
            molecule_nodes=len(mol_nodes),
            reaction_nodes=len(rxn_nodes),
            connection_edges=len(connection),
            max_depth=max([n.depth for n in all_nodes], default=0),
            total_visits=sum(n.visits for n in all_nodes),
            num_complete_paths=self.complete_path_counter,
        )
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def _is_commercial(self, molecule: str) -> bool:
        """检查是否为商业分子"""
        similarity = self._calculate_similarity(molecule)
        is_commercial = similarity > 0.99
        # print(f"检查是否为商业分子: 分子 {molecule} 与商业分子最高相似度: {similarity}")
        return is_commercial
    
    def _calculate_similarity(self, molecule: str) -> float:

        if molecule in self.similarity_cache:
            max_similarity, commercial_index = self.similarity_cache[molecule]
        else:
            similarities = similarity_calculator(molecule, self.commercial_molecules)
            max_similarity = max(similarities, default=0.0)
            commercial_index = similarities.index(max_similarity) if max_similarity > 0.99 else None
            self.similarity_cache[molecule] = (max_similarity, commercial_index)
        
        return max_similarity

    def _is_path_complete(self, end_node: Node) -> bool:
        """检查到某个节点的路径是否完整"""
        if isinstance(end_node, MoleculeNode):
            return end_node.is_commercial
        elif isinstance(end_node, ReactionNode):
            return end_node.is_terminal()
        return False
        
    # ========================================================================
    # 导入导出和持久化
    # ========================================================================
    
    def _save_search_tree(self, save_path: str, filename: str = 'synthesis_tree.pkl'):
        # 定义要保存的文件名
        save_file = os.path.join(save_path, filename)
        data2pkl(self, save_file)
        return save_file
    
    def export_tree_data(self, result_path: Optional[str]=None, prune: bool = False) -> Dict:
        """导出树数据（用于保存或分析）"""
        if prune:
            # molecule_info, reaction_info, connection_info = self.reset_and_collect_tree_info()
            molecule_info, reaction_info, connection_info = self.prune_and_collect_tree_data()
        else:
            molecule_info, reaction_info, connection_info = self.collect_tree_info()
        
        all_data = {
            'target_molecule': self.target,
            'commercial_molecules': list(self.commercial_molecules),
            # 'complete_path': [cp.to_dict for cp in self.complete_paths],
            'molecule_nodes': molecule_info,
            'reaction_nodes': reaction_info,
            'connection': connection_info,
        }
        if result_path:
            filename = os.path.join(result_path, 'search_tree_all_data.json') if not prune else \
                        os.path.join(result_path, 'search_tree_pruned_data.json')
            dict2json(all_data, filename)

        return all_data
      
    def prune_and_collect_tree_data(self):
        """只保留以商业分子结束的完整路径，删除所有其他分支"""
        # 步骤1: 收集所有商业分子节点
        commercial_nodes = [node for node in self.molecule_nodes.values() if node.is_commercial]
        
        molecule_info = []
        reaction_info = []
        connection_set = set()
        complete_paths = []

        # 步骤2: 标记所有需要保留的节点（从商业节点回溯到根节点）
        # nodes_to_keep = set()
        for current in commercial_nodes:
            cp_mols, cp_rxns, cp_connections = [], [], []
            while current:
                if current.is_commercial: 
                    if current.molecule not in self.similarity_cache:
                        self._calculate_similarity(current.molecule)
                    _, commercial_index = self.similarity_cache[current.molecule]
                    if commercial_index is None:
                        # Be robust to trees saved before the per-reactant
                        # commercial-status fix. Such a node is not backed by
                        # an actual stock molecule and therefore cannot close
                        # a valid synthesis path.
                        current.is_commercial = False
                        cp_mols = []
                        cp_rxns = []
                        cp_connections = []
                        break
                    current.molecule = self.commercial_molecules[commercial_index]
                    current.id = f"molecule_{current.molecule}"
                    current.molecule = current.molecule
                molecule_info.append(current.to_dict)
                cp_mols.append(current)

                # 如果当前节点是分子节点且有父节点（反应节点）
                if isinstance(current, MoleculeNode) and current.parent:
                    # 添加父节点（反应节点）
                    reaction_info.append(current.parent.to_dict)
                    cp_rxns.append(current.parent)

                    # 1. rxn -> mol
                    connection_set.add((
                        current.parent.id,
                        current.id,
                        "" #f"Score:{current.parent.scores:.3f}\n{current.parent.top_k}"
                    ))
                    cp_connections.append({
                        'id': f"{current.parent.id}_{current.id}",
                        'source': current.parent.id,
                        'target': current.id,
                        'label': ""
                    })

                    # 移动到反应节点的父节点（上一个分子节点） mol -> rxn 
                    connection_set.add((
                        current.parent.parent.id,
                        current.parent.id,
                        f"Score:{current.parent.reaction_info.scores:.3f}\n{current.parent.reaction_info.top_k}"
                    ))

                    cp_connections.append({
                        "id": f"{current.parent.parent.id}_{current.parent.id}",
                        "source": current.parent.parent.id,
                        "target": current.parent.id,
                        "label": f"Score:{current.parent.reaction_info.scores:.3f}\n{current.parent.reaction_info.top_k}",
                    })

                    current = current.parent.parent
                else:
                    current = None
            
            if cp_mols:
                complete_paths.append(SynthesisPath(
                    molecule_nodes=cp_mols,
                    reaction_nodes=cp_rxns,
                    _connections=cp_connections,
                    is_complete=True,
                    # objective_values=[]
                ))

        self.complete_paths = complete_paths
        self.complete_path_counter = len(complete_paths)

        molecule_info.sort(key=lambda x: x['depth'])
        reaction_info.sort(key=lambda x: x['depth'])

        connection_info = [
            {
                'id': f"{parent_id}_{child_id}",
                'source': parent_id,
                'target': child_id,
                'label': label
            }
            for parent_id, child_id, label in connection_set
        ]

        return molecule_info, reaction_info, connection_info
    



