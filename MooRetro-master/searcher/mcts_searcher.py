'''
# mcts_searcher.py - 重构的MCTS搜索器，严格分离四步骤，一旦找到立刻停止
import numpy as np
import random
import math
import time
import os
from typing import Optional, List
import psutil
from dataclasses import dataclass, asdict, is_dataclass
import json 

# pymoo相关导入
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV

from src.synthesis_tree import SynthesisTree, SynthesisPath
from src.evaluator import PathEvaluator
from src.analyzer import Analyzer
from src.selector import Selector
from config.config import MCTSConfig
from src.nodes import Node, MoleculeNode, ReactionNode
from utils.logger import setup_logger
from utils.utils import data2pkl, dict2json


# ============================================================================
# 1. 数据记录模块 (Data Recording)
# ============================================================================

@dataclass
class IterationRecord:
    """记录单次迭代的关键性能指标"""
    iteration: int
    elapsed_time: float
    tree_size: int
    complete_paths: int
    pareto_front_size: int
    hypervolume: float


class MCTSSearcher:
    """
    重构的MCTS搜索器，严格遵循MCTS四步骤。
    核心改进：将模拟(Simulation)与扩展(Expansion)彻底分离。
    - Expansion: 仅在主搜索树中扩展一个新节点。
    - Simulation: 从一个节点开始进行“虚拟”的快速推演（不修改主树），以获得评估分数。
    """
    
    def __init__(self, 
                 tree: SynthesisTree,
                 single_fn_function,
                 evaluator: PathEvaluator,
                 selector: Selector,
                 analyzer: Analyzer,
                 mcts_config: Optional[MCTSConfig] = None,
                 logger=None,
                 ):
        
        self.tree = tree
        self.single_fn = single_fn_function
        self.evaluator = evaluator
        self.config = mcts_config or MCTSConfig()
        self.analyzer = analyzer
        
        # MCTS状态
        self.start_time = time.time()
        self.iteration_count = 0
        self.records: List[IterationRecord] = []

        # Moo solutions
        self.global_solutions = []
        
        # 日志 
        self.logger = logger or setup_logger(
            os.path.join(self.config.result_path, f'mcts_search_logs_{time.time()}'))
        
    def search(self, max_iterations: Optional[int] = None) -> SynthesisTree:
        """执行MCTS搜索，一旦找到完整路径即提前终止（若配置要求）"""
        max_iterations = max_iterations or self.config.max_iterations
        
        self.logger.info(f"开始MCTS搜索: 目标分子 {self.tree.root.molecule}")
        self.logger.info(f"最大迭代次数: {max_iterations}")
        
        for iteration in range(max_iterations):
            self.iteration_count = iteration + 1
            
            try:
                # 1. Selection: 从根节点开始，根据UCB策略选择一个最有潜力的叶子节点。
                selected_leaf_node = self._selection()
                self.logger.info(f"[{self.iteration_count}] Selection finished. Node: {str(selected_leaf_node)}")

                # 2. Expansion: 如果这个叶子节点可以扩展，就在树上创建一个新的子节点。
                expanded_node = self._expansion(selected_leaf_node)
                self.logger.info(f"[{self.iteration_count}] Expansion finished. Expanded node: {str(expanded_node)}")

                # 模拟的起点是新扩展的节点（如果成功扩展），否则就是选择的叶子节点
                simulation_start_node = expanded_node or selected_leaf_node
                
                # 3. Simulation (Rollout): 从新节点开始，进行一次快速、随机的模拟，直到终点。
                #    此步骤不修改原始搜索树。
                objective_vector, is_complete = self._simulation(simulation_start_node)
                self.logger.info(f"[{self.iteration_count}] Simulation finished. Score: {objective_vector}, Complete: {is_complete}")
                
                # 4. Backpropagation: 将模拟结果从叶子节点反向传播至根节点，更新路径上所有节点的统计数据。
                self._backpropagation(simulation_start_node, objective_vector)
                self.logger.info(f"[{self.iteration_count}] Backpropagation finished.")
                
                # 定期提供反馈
                if self.iteration_count % self.config.real_time_feedback_interval == 0:
                    self._provide_real_time_feedback(self.iteration_count)
                
                # ========== 新增：如果已经找到完整路径，提前终止 ==========
                if self.tree.complete_paths:
                    self.logger.info(f"找到完整路径，提前终止于第 {self.iteration_count} 次迭代")
                    break
                # =======================================================
                    
            except Exception as e:
                self.logger.error(f"第 {self.iteration_count} 次迭代出错: {e}", exc_info=True)
                continue
        
        self._finalize_search_results()
        return self.tree
    
    # ========================================================================
    # MCTS四个核心步骤
    # ========================================================================
    
    def _selection(self) -> Node:
        current = self.tree.root

        while not current.is_terminal():
            if current.is_leaf():
                # 非终端叶子节点，继续返回以扩展
                return current

            if isinstance(current, MoleculeNode):
                current = self._select_best_child_ucb(current)
            elif isinstance(current, ReactionNode):
                # 反应节点只有一个产物分子，直接选择
                current = current.children[0]

        return current
    
    def _expansion(self, node: Node) -> Optional[Node]:
        """Step 2: Expansion - 如果节点不是终端节点，则扩展一个新的子节点。"""
        if node.is_terminal():
            # 终端节点（如商业分子）无法扩展
            return None
        
        if isinstance(node, MoleculeNode):
            if not node.is_expandable(): 
                return None
            
            # 为该分子查找所有可能的反应
            reactions = self.single_fn(node.molecule, 
                                       list(self.tree.commercial_molecules), 
                                       single_fn_type=self.config.single_fn_type)
            if not reactions:
                return None # 没有可行的反应
            
            chosen_reaction = self._add_reactions_and_choose_one(node, reactions)
            return chosen_reaction
        
        return None # 反应节点通常在创建时就已连接分子，无需单独扩展
    
    def _add_reactions_and_choose_one(self, parent_node: MoleculeNode, reactions: List) -> ReactionNode:
        # 将所有新反应作为子节点添加到树中
            new_reaction_nodes = []
            for reaction_info in reactions:
                # add_reaction会处理重复性检查
                # reaction_node = self.tree.add_reaction(node.molecule, reaction_info)
                reaction_node = self.tree.add_reaction(parent_node, reaction_info)
                # 只处理新创建的节点（visits为0）
                if reaction_node.visits == 0:
                    new_reaction_nodes.append(reaction_node)
                
                # 在此过程中，添加了所有的分子，但是只选择了一个，所以可能导致 没有检测到问题
                for child in reaction_node.children:
                    if child.is_terminal():
                        current_path = self.tree.get_path_to_node(child)
                        objective_vector = self.evaluator.evaluate_path(current_path.molecules, current_path.reactions, is_complete=True)
                        current_path.objective_values = objective_vector
                        self.tree.record_path(current_path, is_complete=True)
                        self.global_solutions.append(objective_vector)

            # 从新扩展的反应中，选择一个进行后续的模拟
            # 策略：优先选择与商业分子相似度高且模型分数高的反应
            if random.random() < self.config.epsilon_greedy:
                chosen_reaction = random.choice(new_reaction_nodes)
            else:
                chosen_reaction = max(new_reaction_nodes, key=lambda r: self.config.heuristic_similarity_param * r.reaction_info.similarity + \
                                      (1 - self.config.heuristic_similarity_param) * r.reaction_info.scores)
            return chosen_reaction


    def _simulation(self, start_node: Node) -> tuple[List, bool]:
        """Step 3: Simulation - 从给定节点开始进行快速的随机推演，不修改主树。"""
        
        # 1. 构建模拟的起始路径
        if isinstance(start_node, ReactionNode):
            start_node = start_node.children[0]
        current_path = self.tree.get_path_to_node(start_node)            

        # 2. 如果起始节点已经是终端，直接评估并返回
        if start_node.is_terminal():
            objective_vector = self.evaluator.evaluate_path(current_path.molecules, current_path.reactions, is_complete=True)
            current_path.objective_values = objective_vector
            self.tree.record_path(current_path, is_complete=True)
            self.global_solutions.append(objective_vector)
            return objective_vector, True

        # 3. 执行随机推演 (Rollout)
        sim_molecules = current_path.molecule_nodes.copy()
        sim_reactions = current_path.reaction_nodes.copy()
        
        # current_mol_name = sim_molecules[-1]
        if isinstance(start_node, MoleculeNode):
            current_node = start_node
        else:# ReactionNode
            current_node = start_node.children[0]
        is_complete = False
        
        # for _ in range(self.config.max_simulation_depth):
        while current_node.depth < self.config.max_simulation_depth:
            # 检查当前分子是否为商业分子
            if current_node.is_terminal():
                # 如果当前分子所commercial node，则将其 smiles 替换为 commercial——molecules中对应的smiles
                is_complete = True
                break
                
            # 获取下一步反应（不添加到主树）
            reactions = self.single_fn(current_node.molecule, 
                                       list(self.tree.commercial_molecules),
                                       single_fn_type=self.config.single_fn_type)
            if not reactions:
                break # 无法继续模拟

            chosen_reaction = self._add_reactions_and_choose_one(current_node, reactions)
            sim_reactions.append(chosen_reaction)

            # 选择下一个分子进行模拟
            current_node = chosen_reaction.children[0]
            sim_molecules.append(current_node)

        # 4. 评估整个模拟路径
        simulated_path = SynthesisPath(sim_molecules, sim_reactions, is_complete=is_complete)
        objective_vector = self.evaluator.evaluate_path(simulated_path.molecules, simulated_path.reactions, is_complete)
        simulated_path.objective_values = objective_vector
        
        # 只有完整路径才记录到树中
        if is_complete:
            self.tree.record_path(simulated_path, is_complete=True)
            self.global_solutions.append(objective_vector)
            self.logger.info(f"Simulation found a complete path of length {len(sim_reactions)}.")

        return objective_vector, is_complete
    
    def _backpropagation(self, start_node: Node, objective_vector: np.ndarray):
        """Step 4: Backpropagation - 将结果从起始节点传播回根节点。"""
        current = start_node
        while current:
            current.visits += 1
            current.solutions.append(objective_vector)
            current = current.parent
    
    # ========================================================================
    # 多目标优化相关方法
    # ========================================================================
    
    def get_pareto_front(self) -> List[np.ndarray]:
        """获取Pareto前沿"""
        if not self.global_solutions:
            return []
        
        solutions = np.array(self.global_solutions)
        nds = NonDominatedSorting()
        front_indices = nds.do(solutions, only_non_dominated_front=True)
        return solutions[front_indices].tolist()
    
    def calculate_hypervolume(self, solutions: List[np.ndarray]) -> float:
        """计算超体积"""
        if not solutions:
            return 0.0
        
        try:
            solutions_array = np.array(solutions)
            if len(solutions_array.shape) == 1:
                solutions_array = solutions_array.reshape(1, -1)
            
            hv = HV(ref_point=self.evaluator.config.get_hypervolume_reference_point())
            return float(hv(solutions_array))
        except Exception as e:
            return 0.0


    
    # ========================================================================
    # 辅助方法
    # ========================================================================

    def _select_best_child_ucb(self, parent_node: MoleculeNode) -> Node:
        """为分子节点选择最佳的反应子节点。"""
        if not parent_node.children:
            return parent_node

        # 为了平衡探索与利用，偶尔进行随机选择
        if random.random() < self.config.epsilon_greedy:
            return random.choice(parent_node.children)
        
        best_child = max(parent_node.children, key=lambda child: self._calculate_ucb_score(child, parent_node))
        return best_child
    
    def _calculate_ucb_score(self, child: Node, parent: Node) -> float:
        """计算节点的UCB值。"""
        if child.visits == 0:
            return float('inf')  # 优先选择未访问的节点
        
        # 1. Exploitation term (利用项)
        # 使用此节点下所有解决方案目标值总和的平均值
        exploitation_score = np.mean([np.sum(sol) for sol in child.solutions])
        
        # 2. Exploration term (探索项)
        exploration_score = self.config.c_param * math.sqrt(
            math.log(parent.visits) / child.visits
        )
        
        return exploitation_score + exploration_score
    
    def _provide_real_time_feedback(self, iteration: int):
        """提供实时反馈。"""
        tree_stats = self.tree.get_statistics()
        process = psutil.Process()
        mem_info = process.memory_info()
        elapsed_time = time.time() - self.start_time
        pareto_front = self.get_pareto_front()

        self.records.append(IterationRecord(
            iteration=iteration,
            elapsed_time=elapsed_time,
            tree_size=tree_stats.total_nodes,
            complete_paths=tree_stats.num_complete_paths,
            pareto_front_size=len(pareto_front), #pareto_size,
            hypervolume=self.calculate_hypervolume(pareto_front) #hypervolume
        ))

        self.logger.info(f"--- Iteration {iteration} Feedback ---")
        self.logger.info(f"  - Tree Stats: Total Nodes={tree_stats.total_nodes}, Max Depth={tree_stats.max_depth}")
        self.logger.info(f"  - Paths: {tree_stats.num_complete_paths} complete paths found.")
        self.logger.info(f"  - Performance: {elapsed_time:.2f}s elapsed, Memory={mem_info.rss / (1024**2):.2f} MB")
        
    def _finalize_search_results(self):
        """完成搜索并记录最终结果。"""
        tree_stats = self.tree.get_statistics()
        self.logger.info("========== MCTS Search Finished ==========")
        self.logger.info(f"Total Iterations: {self.iteration_count}")
        self.logger.info(f"Total Run Time: {time.time() - self.start_time:.2f} seconds")
        self.logger.info(f"Final Tree Size: {tree_stats.total_nodes} nodes (Molecules: {tree_stats.molecule_nodes}, Reactions: {tree_stats.reaction_nodes})")
        self.logger.info(f"Saving SynthesisTree to {self.tree._save_search_tree(self.config.result_path)}")
        
        # ========================================================================
        # 1. search tree all data
        # ========================================================================
        self.logger.info(f"Saving all data ...")
        tree_all_data = {
            'project': self.config.project_name,
            'target_molecule': self.tree.target,
            'commercial_molecules': list(self.tree.commercial_molecules),
            'global_solutions': self.global_solutions,
            'pareto_fronts': self.get_pareto_front(),
            }
        tree_all_data.update(asdict(self.config))
        tree_all_data.update(asdict(self.evaluator.config))
        tree_all_data.update(self.tree.export_tree_data())
        tree_all_data.update({'records': [asdict(r) for r in self.records]})
        
        dict2json(tree_all_data, os.path.join(self.config.result_path, 'synthesis_tree_all_data.json'))

        # 保存搜索树等分析结果
        if self.analyzer:
            self.logger.info("Generating analysis report for all search tree...")
            self.analyzer.plot_search_tree_graphviz(tree_all_data['molecule_nodes'], 
                                                    tree_all_data['reaction_nodes'], 
                                                    tree_all_data['connection'],
                                                    save_path=self.config.result_path,
                                                    include_node_png=False,
                                                    format='svg',
                                                    graph_name='synthesis_tree_all')

        # ========================================================================
        # 2. pruned search tree data
        # ========================================================================
        self.logger.info(f"Saving pruned data ...")
        pruned_save_path = os.path.join(self.config.result_path,'pruned')
        if not os.path.exists(pruned_save_path): os.mkdir(pruned_save_path)
        
        prune_tree_data = self.tree.export_tree_data(prune=True)
        prune_tree_data.update(asdict(self.config)) 
        dict2json(prune_tree_data, os.path.join(pruned_save_path, 'synthesis_tree_pruned_data.json'))
        

        # 保存搜索树等分析结果
        if self.analyzer:
            self.logger.info("Generating analysis report for pruned tree...")
            self.analyzer.plot_search_tree_graphviz(prune_tree_data['molecule_nodes'], 
                                                    prune_tree_data['reaction_nodes'], 
                                                    prune_tree_data['connection'],
                                                    save_path=pruned_save_path,
                                                    include_node_png=True,
                                                    format='pdf',
                                                    graph_name='synthesis_tree_pruned')
            
        
        # ========================================================================
        # 3. complete path
        # ========================================================================
        self.logger.info(f"Saving complete path ...")
        self.logger.info("ReEvaluating complete paths ...")
        # 已经得到了 cps 可以 重新评估，并计算 pareto front

        if len(self.tree.complete_paths) > 0: 
            all_cps = []
            all_cps_solutions = []
            for i, complete_path in enumerate(self.tree.complete_paths): 
                path_metrics = self.evaluator._evaluate_complete_path(complete_path.molecules,
                                                                      complete_path.reactions)
                objective_vector = self.evaluator._generate_objective_vector(path_metrics).tolist()
                complete_path._metrics = path_metrics
                complete_path.objective_values = objective_vector
                all_cps.append(complete_path)
                all_cps_solutions.append(objective_vector)

            nds = NonDominatedSorting()
            objectives = np.array(all_cps_solutions)
            fronts = nds.do(objectives, return_rank=False)
            
            all_complete_paths = []
            for front_idx, path_indices in enumerate(fronts):
                for path_idx in path_indices:
                    complete_path = all_cps[path_idx]
                    cp_data = complete_path.to_dict
                    cp_data['metrics'].update({'pareto_front': front_idx})
                    all_complete_paths.append(cp_data)
                    dict2json(cp_data, os.path.join(pruned_save_path, f'pathwayEvaluation_front{front_idx}_{path_idx}.json'))
                    if self.analyzer:
                        self.analyzer.plot_path_graphviz(complete_path, 
                                                        save_path=pruned_save_path, 
                                                        include_node_png=True, 
                                                        format='pdf',
                                                        graph_name=f'front{front_idx}_complete_paths_{path_idx}')

            cytoscape_data = {
                "search_tree": {
                    'molecules': prune_tree_data['molecule_nodes'], 
                    'reactions': prune_tree_data['reaction_nodes'], 
                    'connection': prune_tree_data['connection']
                },
                "complete_paths": all_complete_paths[:10] # only save first 10 paths
            }
            dict2json(cytoscape_data, os.path.join(pruned_save_path, 'pathwaySearch.json'))
'''
#去掉帕累托
# mcts_searcher.py - 重构的MCTS搜索器，严格分离四步骤（单目标版本）
import numpy as np
import random
import math
import time
import os
from typing import Optional, List
import psutil
from dataclasses import dataclass, asdict

# 移除 pymoo 导入
# from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
# from pymoo.indicators.hv import HV

from src.synthesis_tree import SynthesisTree, SynthesisPath
from src.evaluator import PathEvaluator
from src.analyzer import Analyzer
from src.selector import Selector
from config.config import MCTSConfig
from src.nodes import Node, MoleculeNode, ReactionNode
from utils.logger import setup_logger
from utils.utils import data2pkl, dict2json


# ============================================================================
# 1. 数据记录模块 (Data Recording)
# ============================================================================

@dataclass
class IterationRecord:
    """记录单次迭代的关键性能指标"""
    iteration: int
    elapsed_time: float
    tree_size: int
    complete_paths: int
    # 移除 pareto_front_size 和 hypervolume
    # 可保留占位或删除
    best_score: float = 0.0   # 可选：当前最佳得分


class MCTSSearcher:
    """
    单目标MCTS搜索器，严格遵循MCTS四步骤。
    """
    
    def __init__(self, 
                 tree: SynthesisTree,
                 single_fn_function,
                 evaluator: PathEvaluator,
                 selector: Selector,
                 analyzer: Analyzer,
                 mcts_config: Optional[MCTSConfig] = None,
                 logger=None,
                 ):
        
        self.tree = tree
        self.single_fn = single_fn_function
        self.evaluator = evaluator
        self.config = mcts_config or MCTSConfig()
        self.analyzer = analyzer
        
        # MCTS状态
        self.start_time = time.time()
        self.iteration_count = 0
        self.records: List[IterationRecord] = []

        # 存储所有路径的单目标得分（标量）
        self.global_solutions = []   # List[float]
        
        # 日志 
        self.logger = logger or setup_logger(
            os.path.join(self.config.result_path, f'mcts_search_logs_{time.time()}'))
        
    def search(self, max_iterations: Optional[int] = None) -> SynthesisTree:
        """执行MCTS搜索"""
        max_iterations = max_iterations or self.config.max_iterations
        
        self.logger.info(f"开始MCTS搜索: 目标分子 {self.tree.root.molecule}")
        self.logger.info(f"最大迭代次数: {max_iterations}")
        
        for iteration in range(max_iterations):
            self.iteration_count = iteration + 1
            
            try:
                # 1. Selection
                selected_leaf_node = self._selection()
                self.logger.info(f"[{self.iteration_count}] Selection finished. Node: {str(selected_leaf_node)}")

                # 2. Expansion
                expanded_node = self._expansion(selected_leaf_node)
                self.logger.info(f"[{self.iteration_count}] Expansion finished. Expanded node: {str(expanded_node)}")

                simulation_start_node = expanded_node or selected_leaf_node
                
                # 3. Simulation (返回标量得分)
                objective_value, is_complete = self._simulation(simulation_start_node)
                self.logger.info(f"[{self.iteration_count}] Simulation finished. Score: {objective_value}, Complete: {is_complete}")
                
                # 4. Backpropagation (标量)
                self._backpropagation(simulation_start_node, objective_value)
                self.logger.info(f"[{self.iteration_count}] Backpropagation finished.")
                
                # 定期提供反馈
                if self.iteration_count % self.config.real_time_feedback_interval == 0:
                    self._provide_real_time_feedback(self.iteration_count)
                    
            except Exception as e:
                self.logger.error(f"第 {self.iteration_count} 次迭代出错: {e}", exc_info=True)
                continue
        
        self._finalize_search_results()
        return self.tree
    
    # ========================================================================
    # MCTS四个核心步骤
    # ========================================================================
    
    def _selection(self) -> Node:
        current = self.tree.root
        while not current.is_terminal():
            if current.is_leaf():
                return current
            if isinstance(current, MoleculeNode):
                current = self._select_best_child_ucb(current)
            elif isinstance(current, ReactionNode):
                current = current.children[0]
        return current
    
    def _expansion(self, node: Node) -> Optional[Node]:
        if node.is_terminal():
            return None
        if isinstance(node, MoleculeNode):
            if not node.is_expandable(): 
                return None
            reactions = self.single_fn(node.molecule, 
                                       list(self.tree.commercial_molecules), 
                                       single_fn_type=self.config.single_fn_type)
            if not reactions:
                return None
            chosen_reaction = self._add_reactions_and_choose_one(node, reactions)
            return chosen_reaction
        return None
    
    def _add_reactions_and_choose_one(self, parent_node: MoleculeNode, reactions: List) -> ReactionNode:
        new_reaction_nodes = []
        for reaction_info in reactions:
            reaction_node = self.tree.add_reaction(parent_node, reaction_info)
            if reaction_node.visits == 0:
                new_reaction_nodes.append(reaction_node)
            for child in reaction_node.children:
                if child.is_terminal():
                    current_path = self.tree.get_path_to_node(child)
                    # evaluate_path 返回标量
                    objective_value = self.evaluator.evaluate_path(current_path.molecules, current_path.reactions, is_complete=True)
                    current_path.objective_values = objective_value  # 存标量
                    self.tree.record_path(current_path, is_complete=True)
                    self.global_solutions.append(objective_value)

        if random.random() < self.config.epsilon_greedy:
            chosen_reaction = random.choice(new_reaction_nodes)
        else:
            chosen_reaction = max(new_reaction_nodes, key=lambda r: self.config.heuristic_similarity_param * r.reaction_info.similarity + \
                                  (1 - self.config.heuristic_similarity_param) * r.reaction_info.scores)
        return chosen_reaction

    def _simulation(self, start_node: Node) -> tuple[float, bool]:
        """返回 (objective_value, is_complete)  objective_value 为标量"""
        if isinstance(start_node, ReactionNode):
            start_node = start_node.children[0]
        current_path = self.tree.get_path_to_node(start_node)            

        if start_node.is_terminal():
            objective_value = self.evaluator.evaluate_path(current_path.molecules, current_path.reactions, is_complete=True)
            current_path.objective_values = objective_value
            self.tree.record_path(current_path, is_complete=True)
            self.global_solutions.append(objective_value)
            return objective_value, True

        sim_molecules = current_path.molecule_nodes.copy()
        sim_reactions = current_path.reaction_nodes.copy()
        
        if isinstance(start_node, MoleculeNode):
            current_node = start_node
        else:
            current_node = start_node.children[0]
        is_complete = False
        
        while current_node.depth < self.config.max_simulation_depth:
            if current_node.is_terminal():
                is_complete = True
                break
            reactions = self.single_fn(current_node.molecule, 
                                       list(self.tree.commercial_molecules),
                                       single_fn_type=self.config.single_fn_type)
            if not reactions:
                break

            chosen_reaction = self._add_reactions_and_choose_one(current_node, reactions)
            sim_reactions.append(chosen_reaction)
            current_node = chosen_reaction.children[0]
            sim_molecules.append(current_node)

        simulated_path = SynthesisPath(sim_molecules, sim_reactions, is_complete=is_complete)
        objective_value = self.evaluator.evaluate_path(simulated_path.molecules, simulated_path.reactions, is_complete)
        simulated_path.objective_values = objective_value
        
        if is_complete:
            self.tree.record_path(simulated_path, is_complete=True)
            self.global_solutions.append(objective_value)
            self.logger.info(f"Simulation found a complete path of length {len(sim_reactions)}.")

        return objective_value, is_complete
    
    def _backpropagation(self, start_node: Node, objective_value: float):
        """反向传播标量得分"""
        current = start_node
        while current:
            current.visits += 1
            current.solutions.append(objective_value)
            current = current.parent
    
    # ========================================================================
    # 辅助方法（单目标UCB）
    # ========================================================================

    def _select_best_child_ucb(self, parent_node: MoleculeNode) -> Node:
        if not parent_node.children:
            return parent_node
        if random.random() < self.config.epsilon_greedy:
            return random.choice(parent_node.children)
        best_child = max(parent_node.children, key=lambda child: self._calculate_ucb_score(child, parent_node))
        return best_child
    
    def _calculate_ucb_score(self, child: Node, parent: Node) -> float:
        """计算单目标UCB值（child.solutions 中存储标量）"""
        if child.visits == 0:
            return float('inf')
        # 利用项：平均得分（标量）
        exploitation_score = np.mean(child.solutions) if child.solutions else 0.0
        exploration_score = self.config.c_param * math.sqrt(math.log(parent.visits) / child.visits)
        return exploitation_score + exploration_score
    
    def _provide_real_time_feedback(self, iteration: int):
        tree_stats = self.tree.get_statistics()
        process = psutil.Process()
        mem_info = process.memory_info()
        elapsed_time = time.time() - self.start_time
        # 计算当前最佳得分（如果有路径）
        best_score = max(self.global_solutions) if self.global_solutions else 0.0

        self.records.append(IterationRecord(
            iteration=iteration,
            elapsed_time=elapsed_time,
            tree_size=tree_stats.total_nodes,
            complete_paths=tree_stats.num_complete_paths,
            best_score=best_score,
        ))

        self.logger.info(f"--- Iteration {iteration} Feedback ---")
        self.logger.info(f"  - Tree Stats: Total Nodes={tree_stats.total_nodes}, Max Depth={tree_stats.max_depth}")
        self.logger.info(f"  - Paths: {tree_stats.num_complete_paths} complete paths found.")
        self.logger.info(f"  - Best Score: {best_score:.4f}")
        self.logger.info(f"  - Performance: {elapsed_time:.2f}s elapsed, Memory={mem_info.rss / (1024**2):.2f} MB")
        
    def _finalize_search_results(self):
        tree_stats = self.tree.get_statistics()
        self.logger.info("========== MCTS Search Finished ==========")
        self.logger.info(f"Total Iterations: {self.iteration_count}")
        self.logger.info(f"Total Run Time: {time.time() - self.start_time:.2f} seconds")
        self.logger.info(f"Final Tree Size: {tree_stats.total_nodes} nodes (Molecules: {tree_stats.molecule_nodes}, Reactions: {tree_stats.reaction_nodes})")
        self.logger.info(f"Saving SynthesisTree to {self.tree._save_search_tree(self.config.result_path)}")
        
        # 保存全部数据（无帕累托）
        self.logger.info("Saving all data ...")
        tree_all_data = {
            'project': self.config.project_name,
            'target_molecule': self.tree.target,
            'commercial_molecules': list(self.tree.commercial_molecules),
            'global_solutions': self.global_solutions,   # 标量列表
        }
        tree_all_data.update(asdict(self.config))
        tree_all_data.update(asdict(self.evaluator.config))
        tree_all_data.update(self.tree.export_tree_data())
        tree_all_data.update({'records': [asdict(r) for r in self.records]})
        
        dict2json(tree_all_data, os.path.join(self.config.result_path, 'synthesis_tree_all_data.json'))

        if self.analyzer:
            self.logger.info("Generating analysis report for all search tree...")
            self.analyzer.plot_search_tree_graphviz(tree_all_data['molecule_nodes'], 
                                                    tree_all_data['reaction_nodes'], 
                                                    tree_all_data['connection'],
                                                    save_path=self.config.result_path,
                                                    include_node_png=False,
                                                    format='svg',
                                                    graph_name='synthesis_tree_all')

        # 保存剪枝数据
        self.logger.info("Saving pruned data ...")
        pruned_save_path = os.path.join(self.config.result_path, 'pruned')
        if not os.path.exists(pruned_save_path): 
            os.mkdir(pruned_save_path)
        
        prune_tree_data = self.tree.export_tree_data(prune=True)
        prune_tree_data.update(asdict(self.config)) 
        dict2json(prune_tree_data, os.path.join(pruned_save_path, 'synthesis_tree_pruned_data.json'))

        if self.analyzer:
            self.logger.info("Generating analysis report for pruned tree...")
            self.analyzer.plot_search_tree_graphviz(prune_tree_data['molecule_nodes'], 
                                                    prune_tree_data['reaction_nodes'], 
                                                    prune_tree_data['connection'],
                                                    save_path=pruned_save_path,
                                                    include_node_png=True,
                                                    format='pdf',
                                                    graph_name='synthesis_tree_pruned')
        
        # 保存完整路径（无需帕累托排序，直接全部保存）
        self.logger.info("Saving complete paths ...")
        if len(self.tree.complete_paths) > 0:
            all_cps = []
            for i, complete_path in enumerate(self.tree.complete_paths):
                # 重新评估以确保指标完整
                path_metrics = self.evaluator._evaluate_complete_path(complete_path.molecules,
                                                                      complete_path.reactions)
                objective_value = self.evaluator._generate_objective_value(path_metrics)  # 标量
                complete_path._metrics = path_metrics
                complete_path.objective_values = objective_value
                cp_data = complete_path.to_dict
                cp_data['metrics']['objective_score'] = objective_value
                all_cps.append(cp_data)
                # 保存每个路径的单独文件
                dict2json(cp_data, os.path.join(pruned_save_path, f'complete_path_{i}.json'))
                if self.analyzer:
                    self.analyzer.plot_path_graphviz(complete_path, 
                                                    save_path=pruned_save_path, 
                                                    include_node_png=True, 
                                                    format='pdf',
                                                    graph_name=f'complete_path_{i}')
            # 按得分排序后保存前10条到 cytoscape 数据
            all_cps.sort(key=lambda x: x['metrics']['objective_score'], reverse=True)
            cytoscape_data = {
                "search_tree": {
                    'molecules': prune_tree_data['molecule_nodes'], 
                    'reactions': prune_tree_data['reaction_nodes'], 
                    'connection': prune_tree_data['connection']
                },
                # "complete_paths": all_cps[:10]



                "complete_paths": all_cps[:100]




            }
            dict2json(cytoscape_data, os.path.join(pruned_save_path, 'pathwaySearch.json'))


'''
# mcts_searcher.py - 重构的MCTS搜索器，严格分离四步骤
import numpy as np
import random
import math
import time
import os
from typing import Optional, List
import psutil
from dataclasses import dataclass, asdict, is_dataclass
import json 

# pymoo相关导入
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV

from src.synthesis_tree import SynthesisTree, SynthesisPath
from src.evaluator import PathEvaluator
from src.analyzer import Analyzer
from src.selector import Selector
from config.config import MCTSConfig
from src.nodes import Node, MoleculeNode, ReactionNode
from utils.logger import setup_logger
from utils.utils import data2pkl, dict2json


# ============================================================================
# 1. 数据记录模块 (Data Recording)
# ============================================================================

@dataclass
class IterationRecord:
    """记录单次迭代的关键性能指标"""
    iteration: int
    elapsed_time: float
    tree_size: int
    complete_paths: int
    pareto_front_size: int
    hypervolume: float


class MCTSSearcher:
    """
    重构的MCTS搜索器，严格遵循MCTS四步骤。
    核心改进：将模拟(Simulation)与扩展(Expansion)彻底分离。
    - Expansion: 仅在主搜索树中扩展一个新节点。
    - Simulation: 从一个节点开始进行“虚拟”的快速推演（不修改主树），以获得评估分数。
    """
    
    def __init__(self, 
                 tree: SynthesisTree,
                 single_fn_function,
                 evaluator: PathEvaluator,
                 selector: Selector,
                 analyzer: Analyzer,
                 mcts_config: Optional[MCTSConfig] = None,
                 logger=None,
                 ):
        
        self.tree = tree
        self.single_fn = single_fn_function
        self.evaluator = evaluator
        self.config = mcts_config or MCTSConfig()
        self.analyzer = analyzer
        
        # MCTS状态
        self.start_time = time.time()
        self.iteration_count = 0
        self.records: List[IterationRecord] = []

        # Moo solutions
        self.global_solutions = []
        
        # 日志 
        self.logger = logger or setup_logger(
            os.path.join(self.config.result_path, f'mcts_search_logs_{time.time()}'))
        
    def search(self, max_iterations: Optional[int] = None) -> SynthesisTree:
        """执行MCTS搜索"""
        max_iterations = max_iterations or self.config.max_iterations
        
        self.logger.info(f"开始MCTS搜索: 目标分子 {self.tree.root.molecule}")
        self.logger.info(f"最大迭代次数: {max_iterations}")
        
        for iteration in range(max_iterations):
            self.iteration_count = iteration + 1
            
            try:
                # 1. Selection: 从根节点开始，根据UCB策略选择一个最有潜力的叶子节点。
                selected_leaf_node = self._selection()
                self.logger.info(f"[{self.iteration_count}] Selection finished. Node: {str(selected_leaf_node)}")

                # 2. Expansion: 如果这个叶子节点可以扩展，就在树上创建一个新的子节点。
                expanded_node = self._expansion(selected_leaf_node)
                self.logger.info(f"[{self.iteration_count}] Expansion finished. Expanded node: {str(expanded_node)}")

                # 模拟的起点是新扩展的节点（如果成功扩展），否则就是选择的叶子节点
                simulation_start_node = expanded_node or selected_leaf_node
                
                # 3. Simulation (Rollout): 从新节点开始，进行一次快速、随机的模拟，直到终点。
                #    此步骤不修改原始搜索树。
                objective_vector, is_complete = self._simulation(simulation_start_node)
                self.logger.info(f"[{self.iteration_count}] Simulation finished. Score: {objective_vector}, Complete: {is_complete}")
                
                # 4. Backpropagation: 将模拟结果从叶子节点反向传播至根节点，更新路径上所有节点的统计数据。
                self._backpropagation(simulation_start_node, objective_vector)
                self.logger.info(f"[{self.iteration_count}] Backpropagation finished.")
                
                # 定期提供反馈
                if self.iteration_count % self.config.real_time_feedback_interval == 0:
                    self._provide_real_time_feedback(self.iteration_count)
                    
            except Exception as e:
                self.logger.error(f"第 {self.iteration_count} 次迭代出错: {e}", exc_info=True)
                continue
        
        self._finalize_search_results()
        return self.tree
    
    # ========================================================================
    # MCTS四个核心步骤
    # ========================================================================
    
    def _selection(self) -> Node:
        current = self.tree.root

        while not current.is_terminal():
            if current.is_leaf():
                # 非终端叶子节点，继续返回以扩展
                return current

            if isinstance(current, MoleculeNode):
                current = self._select_best_child_ucb(current)
            elif isinstance(current, ReactionNode):
                # 反应节点只有一个产物分子，直接选择
                current = current.children[0]

        return current
    
    def _expansion(self, node: Node) -> Optional[Node]:
        """Step 2: Expansion - 如果节点不是终端节点，则扩展一个新的子节点。"""
        if node.is_terminal():
            # 终端节点（如商业分子）无法扩展
            return None
        
        if isinstance(node, MoleculeNode):
            if not node.is_expandable(): 
                return None
            
            # 为该分子查找所有可能的反应
            reactions = self.single_fn(node.molecule, 
                                       list(self.tree.commercial_molecules), 
                                       single_fn_type=self.config.single_fn_type)
            if not reactions:
                return None # 没有可行的反应
            
            chosen_reaction = self._add_reactions_and_choose_one(node, reactions)
            return chosen_reaction
        
        return None # 反应节点通常在创建时就已连接分子，无需单独扩展
    
    def _add_reactions_and_choose_one(self, parent_node: MoleculeNode, reactions: List) -> ReactionNode:
        # 将所有新反应作为子节点添加到树中
            new_reaction_nodes = []
            for reaction_info in reactions:
                # add_reaction会处理重复性检查
                # reaction_node = self.tree.add_reaction(node.molecule, reaction_info)
                reaction_node = self.tree.add_reaction(parent_node, reaction_info)
                # 只处理新创建的节点（visits为0）
                if reaction_node.visits == 0:
                    new_reaction_nodes.append(reaction_node)
                
                # 在此过程中，添加了所有的分子，但是只选择了一个，所以可能导致 没有检测到问题
                for child in reaction_node.children:
                    if child.is_terminal():
                        current_path = self.tree.get_path_to_node(child)
                        objective_vector = self.evaluator.evaluate_path(current_path.molecules, current_path.reactions, is_complete=True)
                        current_path.objective_values = objective_vector
                        self.tree.record_path(current_path, is_complete=True)
                        self.global_solutions.append(objective_vector)

            # 从新扩展的反应中，选择一个进行后续的模拟
            # 策略：优先选择与商业分子相似度高且模型分数高的反应
            if random.random() < self.config.epsilon_greedy:
                chosen_reaction = random.choice(new_reaction_nodes)
            else:
                chosen_reaction = max(new_reaction_nodes, key=lambda r: self.config.heuristic_similarity_param * r.reaction_info.similarity + \
                                      (1 - self.config.heuristic_similarity_param) * r.reaction_info.scores)
            return chosen_reaction


    def _simulation(self, start_node: Node) -> tuple[List, bool]:
        """Step 3: Simulation - 从给定节点开始进行快速的随机推演，不修改主树。"""
        
        # 1. 构建模拟的起始路径
        if isinstance(start_node, ReactionNode):
            start_node = start_node.children[0]
        current_path = self.tree.get_path_to_node(start_node)            

        # 2. 如果起始节点已经是终端，直接评估并返回
        if start_node.is_terminal():
            objective_vector = self.evaluator.evaluate_path(current_path.molecules, current_path.reactions, is_complete=True)
            current_path.objective_values = objective_vector
            self.tree.record_path(current_path, is_complete=True)
            self.global_solutions.append(objective_vector)
            return objective_vector, True

        # 3. 执行随机推演 (Rollout)
        sim_molecules = current_path.molecule_nodes.copy()
        sim_reactions = current_path.reaction_nodes.copy()
        
        # current_mol_name = sim_molecules[-1]
        if isinstance(start_node, MoleculeNode):
            current_node = start_node
        else:# ReactionNode
            current_node = start_node.children[0]
        is_complete = False
        
        # for _ in range(self.config.max_simulation_depth):
        while current_node.depth < self.config.max_simulation_depth:
            # 检查当前分子是否为商业分子
            if current_node.is_terminal():
                # 如果当前分子所commercial node，则将其 smiles 替换为 commercial——molecules中对应的smiles
                is_complete = True
                break
                
            # 获取下一步反应（不添加到主树）
            reactions = self.single_fn(current_node.molecule, 
                                       list(self.tree.commercial_molecules),
                                       single_fn_type=self.config.single_fn_type)
            if not reactions:
                break # 无法继续模拟

            chosen_reaction = self._add_reactions_and_choose_one(current_node, reactions)
            sim_reactions.append(chosen_reaction)

            # 选择下一个分子进行模拟
            current_node = chosen_reaction.children[0]
            sim_molecules.append(current_node)

        # 4. 评估整个模拟路径
        simulated_path = SynthesisPath(sim_molecules, sim_reactions, is_complete=is_complete)
        objective_vector = self.evaluator.evaluate_path(simulated_path.molecules, simulated_path.reactions, is_complete)
        simulated_path.objective_values = objective_vector
        
        # 只有完整路径才记录到树中
        if is_complete:
            self.tree.record_path(simulated_path, is_complete=True)
            self.global_solutions.append(objective_vector)
            self.logger.info(f"Simulation found a complete path of length {len(sim_reactions)}.")

        return objective_vector, is_complete
    
    def _backpropagation(self, start_node: Node, objective_vector: np.ndarray):
        """Step 4: Backpropagation - 将结果从起始节点传播回根节点。"""
        current = start_node
        while current:
            current.visits += 1
            current.solutions.append(objective_vector)
            current = current.parent
    
    # ========================================================================
    # 多目标优化相关方法
    # ========================================================================
    
    def get_pareto_front(self) -> List[np.ndarray]:
        """获取Pareto前沿"""
        if not self.global_solutions:
            return []
        
        solutions = np.array(self.global_solutions)
        nds = NonDominatedSorting()
        front_indices = nds.do(solutions, only_non_dominated_front=True)
        return solutions[front_indices].tolist()
    
    def calculate_hypervolume(self, solutions: List[np.ndarray]) -> float:
        """计算超体积"""
        if not solutions:
            return 0.0
        
        try:
            solutions_array = np.array(solutions)
            if len(solutions_array.shape) == 1:
                solutions_array = solutions_array.reshape(1, -1)
            
            hv = HV(ref_point=self.evaluator.config.get_hypervolume_reference_point())
            return float(hv(solutions_array))
        except Exception as e:
            return 0.0


    
    # ========================================================================
    # 辅助方法
    # ========================================================================

    def _select_best_child_ucb(self, parent_node: MoleculeNode) -> Node:
        """为分子节点选择最佳的反应子节点。"""
        if not parent_node.children:
            return parent_node

        # 为了平衡探索与利用，偶尔进行随机选择
        if random.random() < self.config.epsilon_greedy:
            return random.choice(parent_node.children)
        
        best_child = max(parent_node.children, key=lambda child: self._calculate_ucb_score(child, parent_node))
        return best_child
    
    def _calculate_ucb_score(self, child: Node, parent: Node) -> float:
        """计算节点的UCB值。"""
        if child.visits == 0:
            return float('inf')  # 优先选择未访问的节点
        
        # 1. Exploitation term (利用项)
        # 使用此节点下所有解决方案目标值总和的平均值
        exploitation_score = np.mean([np.sum(sol) for sol in child.solutions])
        
        # 2. Exploration term (探索项)
        exploration_score = self.config.c_param * math.sqrt(
            math.log(parent.visits) / child.visits
        )
        
        return exploitation_score + exploration_score
    
    def _provide_real_time_feedback(self, iteration: int):
        """提供实时反馈。"""
        tree_stats = self.tree.get_statistics()
        process = psutil.Process()
        mem_info = process.memory_info()
        elapsed_time = time.time() - self.start_time
        pareto_front = self.get_pareto_front()

        self.records.append(IterationRecord(
            iteration=iteration,
            elapsed_time=elapsed_time,
            tree_size=tree_stats.total_nodes,
            complete_paths=tree_stats.num_complete_paths,
            pareto_front_size=len(pareto_front), #pareto_size,
            hypervolume=self.calculate_hypervolume(pareto_front) #hypervolume
        ))

        self.logger.info(f"--- Iteration {iteration} Feedback ---")
        self.logger.info(f"  - Tree Stats: Total Nodes={tree_stats.total_nodes}, Max Depth={tree_stats.max_depth}")
        self.logger.info(f"  - Paths: {tree_stats.num_complete_paths} complete paths found.")
        self.logger.info(f"  - Performance: {elapsed_time:.2f}s elapsed, Memory={mem_info.rss / (1024**2):.2f} MB")
        
    def _finalize_search_results(self):
        """完成搜索并记录最终结果。"""
        tree_stats = self.tree.get_statistics()
        self.logger.info("========== MCTS Search Finished ==========")
        self.logger.info(f"Total Iterations: {self.iteration_count}")
        self.logger.info(f"Total Run Time: {time.time() - self.start_time:.2f} seconds")
        self.logger.info(f"Final Tree Size: {tree_stats.total_nodes} nodes (Molecules: {tree_stats.molecule_nodes}, Reactions: {tree_stats.reaction_nodes})")
        self.logger.info(f"Saving SynthesisTree to {self.tree._save_search_tree(self.config.result_path)}")
        
        # ========================================================================
        # 1. search tree all data
        # ========================================================================
        self.logger.info(f"Saving all data ...")
        tree_all_data = {
            'project': self.config.project_name,
            'target_molecule': self.tree.target,
            'commercial_molecules': list(self.tree.commercial_molecules),
            'global_solutions': self.global_solutions,
            'pareto_fronts': self.get_pareto_front(),
            }
        tree_all_data.update(asdict(self.config))
        tree_all_data.update(asdict(self.evaluator.config))
        tree_all_data.update(self.tree.export_tree_data())
        tree_all_data.update({'records': [asdict(r) for r in self.records]})
        
        dict2json(tree_all_data, os.path.join(self.config.result_path, 'synthesis_tree_all_data.json'))

        # 保存搜索树等分析结果
        if self.analyzer:
            self.logger.info("Generating analysis report for all search tree...")
            self.analyzer.plot_search_tree_graphviz(tree_all_data['molecule_nodes'], 
                                                    tree_all_data['reaction_nodes'], 
                                                    tree_all_data['connection'],
                                                    save_path=self.config.result_path,
                                                    include_node_png=False,
                                                    format='svg',
                                                    graph_name='synthesis_tree_all')

        # ========================================================================
        # 2. pruned search tree data
        # ========================================================================
        self.logger.info(f"Saving pruned data ...")
        pruned_save_path = os.path.join(self.config.result_path,'pruned')
        if not os.path.exists(pruned_save_path): os.mkdir(pruned_save_path)
        
        prune_tree_data = self.tree.export_tree_data(prune=True)
        prune_tree_data.update(asdict(self.config)) 
        dict2json(prune_tree_data, os.path.join(pruned_save_path, 'synthesis_tree_pruned_data.json'))
        

        # 保存搜索树等分析结果
        if self.analyzer:
            self.logger.info("Generating analysis report for pruned tree...")
            self.analyzer.plot_search_tree_graphviz(prune_tree_data['molecule_nodes'], 
                                                    prune_tree_data['reaction_nodes'], 
                                                    prune_tree_data['connection'],
                                                    save_path=pruned_save_path,
                                                    include_node_png=True,
                                                    format='pdf',
                                                    graph_name='synthesis_tree_pruned')
            
        
        # ========================================================================
        # 3. complete path
        # ========================================================================
        self.logger.info(f"Saving complete path ...")
        self.logger.info("ReEvaluating complete paths ...")
        # 已经得到了 cps 可以 重新评估，并计算 pareto front

        if len(self.tree.complete_paths) > 0: 
            all_cps = []
            all_cps_solutions = []
            for i, complete_path in enumerate(self.tree.complete_paths): 
                path_metrics = self.evaluator._evaluate_complete_path(complete_path.molecules,
                                                                      complete_path.reactions)
                objective_vector = self.evaluator._generate_objective_vector(path_metrics).tolist()
                complete_path._metrics = path_metrics
                complete_path.objective_values = objective_vector
                all_cps.append(complete_path)
                all_cps_solutions.append(objective_vector)

            nds = NonDominatedSorting()
            objectives = np.array(all_cps_solutions)
            fronts = nds.do(objectives, return_rank=False)
            
            all_complete_paths = []
            for front_idx, path_indices in enumerate(fronts):
                for path_idx in path_indices:
                    complete_path = all_cps[path_idx]
                    cp_data = complete_path.to_dict
                    cp_data['metrics'].update({'pareto_front': front_idx})
                    all_complete_paths.append(cp_data)
                    dict2json(cp_data, os.path.join(pruned_save_path, f'pathwayEvaluation_front{front_idx}_{path_idx}.json'))
                    if self.analyzer:
                        self.analyzer.plot_path_graphviz(complete_path, 
                                                        save_path=pruned_save_path, 
                                                        include_node_png=True, 
                                                        format='pdf',
                                                        graph_name=f'front{front_idx}_complete_paths_{path_idx}')

            cytoscape_data = {
                "search_tree": {
                    'molecules': prune_tree_data['molecule_nodes'], 
                    'reactions': prune_tree_data['reaction_nodes'], 
                    'connection': prune_tree_data['connection']
                },
                "complete_paths": all_complete_paths[:10] # only save first 10 paths
            }
            dict2json(cytoscape_data, os.path.join(pruned_save_path, 'pathwaySearch.json'))
'''