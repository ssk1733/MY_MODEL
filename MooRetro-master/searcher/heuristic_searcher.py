# simple_searcher.py - A simple, direct heuristic-driven searcher.
import numpy as np
import time
import os
import heapq  # Priority Queue
from typing import Optional

from src.synthesis_tree import SynthesisTree
from src.evaluator import PathEvaluator # Not used in search, but needed for init
from src.analyzer import Analyzer
from config.config import MCTSConfig
from src.nodes import MoleculeNode
from utils.logger import setup_logger

class HeuristicSearcher:
    """
    A simple, greedy best-first searcher driven by a heuristic score.

    This searcher replaces the complexity of MCTS with a straightforward strategy:
    1. Maintain a priority queue (frontier) of expandable molecules.
    2. The priority is based on the heuristic score of the reaction that produced the molecule.
    3. Always expand the molecule with the highest priority.
    """
    
    def __init__(self, 
                 tree: SynthesisTree,
                 single_fn_function,
                 evaluator: PathEvaluator,
                 analyzer: Analyzer,
                 mcts_config: Optional[MCTSConfig] = None,
                 logger=None,
                 ):
        
        self.tree = tree
        self.single_fn = single_fn_function
        self.analyzer = analyzer
        self.config = mcts_config or MCTSConfig()
        
        # Search state
        self.root = tree.root
        self.start_time = None
        self.expansion_count = 0
        
        # Frontier: A priority queue storing tuples of (-priority, molecule_smiles)
        # We use a negative priority because heapq is a min-heap.
        self.frontier = []
        
        # Keep track of molecules we have already expanded to avoid redundant work
        self.expanded_molecules = set()
        
        # Heuristic weights (can be tuned in MCTSConfig)
        self.w_sim = getattr(self.config, 'heuristic_w_sim', 0.8)
        self.w_score = getattr(self.config, 'heuristic_w_score', 0.2)
        
        # Logger
        self.logger = logger or setup_logger(
            os.path.join(self.config.result_path, f'simple_search_logs_{time.time()}'))
        
    def search(self, max_expansions: Optional[int] = None) -> SynthesisTree:
        """Executes the greedy best-first search."""
        max_expansions = max_expansions or self.config.max_iterations # Reuse config parameter
        self.start_time = time.time()
        
        self.logger.info(f"开始启发式搜索: 目标分子 {self.root.molecule}")
        self.logger.info(f"最大扩展次数: {max_expansions}")
        
        # Initialize the frontier with the root node
        # The root has no parent reaction, so we give it a neutral priority of 0.
        heapq.heappush(self.frontier, (0, self.root.molecule))

        while self.frontier and self.expansion_count < max_expansions:
            # 1. Select the most promising molecule from the frontier
            _, current_mol_smiles = heapq.heappop(self.frontier)
            
            if current_mol_smiles in self.expanded_molecules:
                continue

            current_node = self.tree.molecule_nodes.get(current_mol_smiles)
            if not current_node:
                self.logger.warning(f"无法在树中找到分子: {current_mol_smiles}")
                continue
                
            # Mark as expanded
            self.expanded_molecules.add(current_mol_smiles)
            self.expansion_count += 1
            
            # If it's a commercial molecule, it's a terminal node. A path has been found.
            if current_node.is_commercial:
                self.logger.info(f"✓ 找到一条完整路径，终止于商业分子: {current_mol_smiles}")
                # Record the path as complete
                path = self.tree.get_path_to_node(current_node)
                self.tree.record_path(path, is_complete=True)
                continue # Continue search for other potential paths

            self.logger.info(f"[{self.expansion_count}/{max_expansions}] Expanding: {current_mol_smiles}")

            # 2. Expand the molecule by finding possible reactions
            reactions = self.single_fn(current_mol_smiles, list(self.tree.commercial_molecules))
            if not reactions:
                self.logger.info(f"-> {current_mol_smiles} 是一个死胡同 (无可用反应).")
                continue

            # 3. Add new reactions and their reactants to the tree and frontier
            for reaction_info in reactions:
                # Add the reaction to the tree. This creates the reaction and reactant nodes.
                reaction_node = self.tree.add_reaction(current_mol_smiles, reaction_info)
                
                # Calculate the heuristic score for this new reaction
                heuristic_score = (self.w_sim * reaction_info.similarity + 
                                   self.w_score * reaction_info.scores)
                
                # Add the resulting reactant molecules to the frontier
                for reactant_node in reaction_node.children:
                    if isinstance(reactant_node, MoleculeNode):
                        if reactant_node.molecule not in self.expanded_molecules:
                            self.logger.info(f"  + 添加到前沿: {reactant_node.molecule} (score: {heuristic_score:.3f})")
                            # Use negative score for max-heap behavior
                            heapq.heappush(self.frontier, (-heuristic_score, reactant_node.molecule))
            
            if self.expansion_count % 10 == 0:
                self._provide_feedback()

        self._finalize_search_results()
        return self.tree

    def _provide_feedback(self):
        """Provides real-time feedback during the search."""
        tree_stats = self.tree.get_statistics()
        self.logger.info(f"--- 扩展 {self.expansion_count} 次后反馈 ---")
        self.logger.info(f"  - 前沿大小: {len(self.frontier)}")
        self.logger.info(f"  - 已找到完整路径: {tree_stats.num_complete_paths}")
        self.logger.info(f"  - 树大小: {tree_stats.total_nodes} 个节点")
        self.logger.info(f"  - 运行时间: {time.time() - self.start_time:.2f} 秒")

    def _finalize_search_results(self):
        """Finalizes the search and triggers analysis."""
        tree_stats = self.tree.get_statistics()
        self.logger.info("========== 启发式搜索完成 ==========")
        self.logger.info(f"总扩展次数: {self.expansion_count}")
        self.logger.info(f"总运行时间: {time.time() - self.start_time:.2f} 秒")
        self.logger.info(f"最终树规模: {tree_stats.total_nodes} 个节点")
        self.logger.info(f"找到的完整路径总数: {tree_stats.num_complete_paths}")

        if self.analyzer:
            self.logger.info("生成分析报告...")
            self.analyzer.quick_analysis(self.tree.target, self.tree)






# class SynthesisSearcher:
#     """基于启发式搜索的生物合成路径搜索器"""
    
#     def __init__(self, 
#                  tree: SynthesisTree,
#                  reaction_generator: Callable[[str, List[str]], List[Dict]],
#                  evaluator: Callable[[SynthesisPath], float],
#                  config: Dict):
#         """
#         参数:
#         - tree: 合成树实例
#         - reaction_generator: 反应生成函数 (分子, 商业分子列表) -> 反应列表
#         - evaluator: 路径评估函数 (路径) -> 分数
#         - config: 配置字典
#         """
#         self.tree = tree
#         self.reaction_generator = reaction_generator
#         self.evaluator = evaluator
#         self.config = config
#         self.logger = logging.getLogger(__name__)
        
#         # 搜索状态
#         self.open_paths = []  # 优先队列 (分数, 路径ID, 路径)
#         self.closed_nodes = set()  # 已完全扩展的节点
#         self.best_paths = []  # 最佳路径缓存
    
#     def search(self) -> List[SynthesisPath]:
#         """执行搜索"""
#         # 初始化搜索
#         self._initialize_search()
        
#         # 主搜索循环
#         while self.open_paths and len(self.best_paths) < self.config.get('max_paths', 10):
#             # 获取最有希望的路径
#             score, _, path = heapq.heappop(self.open_paths)
            
#             # 扩展当前路径
#             self._expand_path(path)
            
#             # 更新最佳路径
#             if path.is_complete:
#                 self.best_paths.append(path)
#                 self.logger.info(f"找到完整路径: 分数={-score:.3f}, 长度={path.length}")
        
#         return self.best_paths
    
#     def _initialize_search(self):
#         """初始化搜索状态"""
#         # 从目标分子开始
#         initial_path = SynthesisPath(
#             molecules=[self.tree.target],
#             reactions=[],
#             is_complete=self.tree._is_commercial(self.tree.target)
#         )
        
#         # 评估初始路径
#         score = self.evaluator(initial_path)
        
#         # 添加到优先队列 (使用负数实现最大堆)
#         heapq.heappush(self.open_paths, (-score, id(initial_path), initial_path))
        
#         self.logger.info(f"搜索初始化: 目标分子={self.tree.target}")
    
#     def _expand_path(self, path: SynthesisPath):
#         """扩展当前路径"""
#         # 获取当前分子（路径末端）
#         current_mol = path.molecules[-1]
        
#         # 如果已经是商业分子，无需扩展
#         if self.tree._is_commercial(current_mol):
#             return
        
#         # 生成可能的反应
#         reactions = self.reaction_generator(
#             current_mol, 
#             self.tree.commercial_molecules
#         )
        
#         if not reactions:
#             self.logger.debug(f"无可用反应: {current_mol}")
#             return
        
#         # 处理每个反应
#         for reaction in reactions:
#             # 创建新路径
#             new_path = self._create_new_path(path, reaction)
            
#             # 评估新路径
#             score = self.evaluator(new_path)
            
#             # 添加到优先队列
#             heapq.heappush(self.open_paths, (-score, id(new_path), new_path))
            
#             # 更新合成树
#             self._update_tree(new_path, reaction)
    
#     def _create_new_path(self, base_path: SynthesisPath, reaction: Dict) -> SynthesisPath:
#         """基于反应创建新路径"""
#         new_molecules = base_path.molecules.copy()
#         new_reactions = base_path.reactions.copy()
        
#         # 添加新反应
#         new_reactions.append(reaction)
        
#         # 添加反应物（只取第一个反应物，简化模型）
#         reactant = reaction['reactants'][0]
#         new_molecules.append(reactant)
        
#         # 检查是否完整
#         is_complete = self.tree._is_commercial(reactant)
        
#         return SynthesisPath(
#             molecules=new_molecules,
#             reactions=new_reactions,
#             is_complete=is_complete
#         )
    
#     def _update_tree(self, path: SynthesisPath, reaction: Dict):
#         """用新路径更新合成树"""
#         # 获取当前分子（路径倒数第二个）
#         parent_mol = path.molecules[-2]
        
#         # 添加反应到树
#         self.tree.add_reaction(parent_mol, reaction)
        
#         # 添加新分子到树
#         new_mol = path.molecules[-1]
#         if new_mol not in self.tree.molecule_nodes:
#             is_commercial = self.tree._is_commercial(new_mol)
#             mol_node = self.tree.molecule_nodes[new_mol] = MoleculeNode(
#                 molecule=new_mol,
#                 is_commercial=is_commercial
#             )



