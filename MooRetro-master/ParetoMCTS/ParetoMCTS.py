import numpy as np
import random
from typing import List, Dict, Tuple, Optional
import time
import os
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  

# pymoo相关导入
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV

from ParetoMCTS.config import SingleStepResult, EvaluationConfig, MCTSConfig, SelectionConfig
from ParetoMCTS.Evaluator import PathEvaluator
from ParetoMCTS.Selector import EnhancedSelector
from ParetoMCTS.Analyzer import Analyzer
from ParetoMCTS.nodes import Node, MoleculeNode, ReactionNode

from utils.logger import setup_logger
from utils.chem import similarity_calculator


class ParetoMCTS:
    """集成多目标PathEvaluator的Pareto MCTS算法"""
    
    def __init__(self, 
                 single_fn_function,
                 commercial_molecules: List[str],
                 mcts_config: Optional[MCTSConfig],
                 eval_config: Optional[EvaluationConfig],
                 sel_config: Optional[SelectionConfig]):
        
        self.single_fn = single_fn_function
        self.commercial_molecules = commercial_molecules
        self.mcts_config = mcts_config
        self.eval_config = eval_config
        self.sel_config = sel_config
        
        # 初始化多目标评估器
        self.evaluator = PathEvaluator(commercial_molecules, eval_config)
        self.selector = EnhancedSelector(self.commercial_molecules,
                                         self.evaluator, 
                                         self.sel_config)
        self.analyzer = Analyzer(result_path=os.path.join(self.mcts_config.result_path, 'analysis'))
        
        # 搜索统计
        self.iteration_count = 0
        self.best_hypervolume = 0.0
        self.start_time = None
        self.root = None
        self.similarity_cache = {}
        
        self.logger = setup_logger(os.path.join(self.mcts_config.result_path, 'logs'))
        self.logger.info(f"Pareto MCTS初始化完成")
        self.logger.info(f"MCTS配置: 最大迭代{self.mcts_config.max_iterations}, UCB参数{self.mcts_config.c_param}")
        self.logger.info(f"商业分子库大小: {len(commercial_molecules)}")

    def search(self, target_molecule: str) -> MoleculeNode:
        """主搜索函数"""
        self.start_time = time.time()
        
        root = MoleculeNode(
            target_molecule, 
            is_commercial=self._calculate_commercial_availability(target_molecule),
            enabled_objectives=self.eval_config.enabled_objectives)
        
        self.root = root
        self.logger.info(f"开始Pareto MCTS搜索: {target_molecule}")
        self.logger.info(f"目标分子是否为商业分子: {root.is_commercial}")
          
        for iteration in range(self.mcts_config.max_iterations):
            self.iteration_count = iteration + 1
            
            # 1. Selection
            selected_node = self.selection(root)
            self.logger.info(f"第{iteration+1}轮 - Selection: {selected_node.molecule} (深度{selected_node.depth})")
            
            # 2. Expansion
            new_node = self.expansion(selected_node)
            
            # 3. Rollout
            if new_node:
                self.logger.info(f"第{iteration+1}轮 - 扩展成功，新节点: {new_node.molecule}")
                self.logger.info(f"第{iteration+1}轮 - Rollout：{new_node.molecule} (深度{new_node.depth})")
                objective_vector = self.rollout(new_node)
                expanded_molecule = new_node.molecule
                
                # 记录路径信息
                self._record_path_if_valid(new_node, objective_vector, iteration + 1)
                
            else:
                self.logger.info(f"第{iteration+1}轮 - 未扩展新节点")
                self.logger.info(f"第{iteration+1}轮 - Rollout: {selected_node.molecule} (深度{selected_node.depth})")
                objective_vector = self.rollout(selected_node)
                expanded_molecule = None
                
                # 记录路径信息
                self._record_path_if_valid(selected_node, objective_vector, iteration + 1)
                
            # 4. Backpropagation
            self.backpropagation(new_node if new_node else selected_node, objective_vector)
            
            # 记录迭代信息用于分析
            if self.analyzer:
                current_solutions = self.get_best_solutions(root)
                hypervolume = 0.0
                if current_solutions:
                    hv = HV(ref_point=self.eval_config.get_hypervolume_reference_point())
                    hypervolume = hv(np.array(current_solutions))
                
                self.analyzer.record_iteration(
                    iteration=iteration + 1,
                    selected_node=selected_node.molecule,
                    expanded_node=expanded_molecule,
                    objective_vector=objective_vector,
                    hypervolume=hypervolume,
                    pareto_front_size=len(current_solutions),
                    elapsed_time=time.time() - self.start_time,
                    tree_size=self._count_tree_nodes(root)
                )
            
            # 实时智能引导 (保持原有代码)
            if (iteration + 1) % self.mcts_config.real_time_feedback_interval == 0:
                self._real_time_feedback(root, iteration + 1)
            
            # 早停检查 (保持原有代码)
            if self._should_early_stop():
                self.logger.info(f"早停于第{iteration + 1}次迭代")
                break

        self.logger.info(f"搜索完成，共{self.iteration_count}次迭代")

        # 新增：设置分析器的最终结果
        if self.analyzer:
            final_solutions = self.get_best_solutions(root)
            self.analyzer.set_final_results(
                search_tree=root,
                final_solutions=final_solutions,
                config=self.eval_config,
                target_molecule=target_molecule
            )
        return root
    
    def _record_path_if_valid(self, node: Node, objective_vector: np.ndarray, iteration: int):
        """记录有效路径 - 简化版本"""
        if not self.analyzer:
            return
            
        # 构建路径
        path_nodes, path_reactions = self._build_path_to_node(node)
        
        # 只记录有反应的路径
        if len(path_reactions) == 0:
            return
        
        # 准备反应信息
        reaction_info = []
        for reaction in path_reactions:
            if hasattr(reaction, 'reactants'):  # SingleStepResult对象
                reaction_dict = {
                    'reactants': reaction.reactants,
                    'scores': float(reaction.scores),
                    'templates': reaction.templates,
                    'ec': reaction.ec,
                    'confidence': float(getattr(reaction, 'confidence', 0.0))
                }
            else:  # 已经是字典
                reaction_dict = reaction
            reaction_info.append(reaction_dict)
        
        # 判断路径完整性
        is_complete = self._is_complete_path_to_node(node)
        
        # 记录路径
        self.analyzer.record_path(
            iteration=iteration,
            molecules=path_nodes,
            reactions=reaction_info,
            objective_vector=objective_vector,
            elapsed_time=time.time() - self.start_time,
            node_depth=node.depth,
            is_complete=is_complete
        )
    
    def selection(self, node: Node) -> Node:
        """选择阶段：多目标UCB策略"""
        current = node
        selection_path = []
        visited_molecules = self._get_visited_molecules()

        while current.is_expanded and current.children:
            selection_path.append(current.molecule)

            if isinstance(current, MoleculeNode):
                # OR节点: 选择下一个
                selected_child = self.selector.select_child(current, visited_molecules, self.iteration_count)
                current = selected_child if selected_child else current.children[0]

            else:  # ReactionNode 经过简化，reactionnode 的 child 只有一个
                current = current.children[0]
                
            
            if current is None or current.is_terminal():
                break
                
        self.logger.debug(f"选择路径: {' -> '.join(selection_path)} -> {current.molecule}")
        return current
    
    def expansion(self, node: Node) -> Optional[Node]:
        """扩展阶段"""
        if node.is_terminal():
            return None
        
        # 如果已扩展，用selector选择子节点
        if node.is_expanded:
            if node.children:
                visited_molecules = self._get_visited_molecules()
                return self.selector.select_child(node, visited_molecules, self.iteration_count)
            return None
        
        if isinstance(node, MoleculeNode):
            # 扩展分子节点
            reactions = self.single_fn(node.molecule, self.commercial_molecules)
            if not reactions:
                node.is_expanded = True
                return None
            
            # 创建所有反应子节点
            for reaction in reactions:
                reaction_node = ReactionNode(
                    molecule=node.molecule,
                    parent=node,
                    depth=node.depth + 1,
                    reaction_info=reaction,
                    enabled_objectives=self.eval_config.enabled_objectives
                )
                node.add_child(reaction_node)
                
                # 添加反应物子节点
                for reactant in reaction.reactants:
                    is_commercial = self._calculate_commercial_availability(reactant)
                    reactant_node = MoleculeNode(
                        molecule=reactant,
                        parent=reaction_node,
                        depth=reaction_node.depth + 1,
                        is_commercial=is_commercial,
                        enabled_objectives=self.eval_config.enabled_objectives
                    )
                    reaction_node.add_child(reactant_node)
            
            node.is_expanded = True
            
            # 用selector选择最佳子节点
            visited_molecules = self._get_visited_molecules()
            return self.selector.select_child(node, visited_molecules, self.iteration_count)
                
        else:  # ReactionNode
            node.is_expanded = True
            return node.children[0] if node.children else None

    def _get_visited_molecules(self) -> set:
        """获取已访问分子集合"""
        visited = set()
        
        def collect(node):
            visited.add(node.molecule)
            for child in node.children:
                collect(child)
        
        if hasattr(self, 'root') and self.root:
            collect(self.root)
        return visited   
    
    def _is_complete_path_to_node(self, node: Node) -> bool:
        """
        判断从根到当前节点的路径是否完整
        完整路径定义：当前节点是商业分子，或者当前节点的所有子树都能到达商业分子
        """
        # 如果当前节点是商业分子，路径完整
        if self._calculate_commercial_availability(node.molecule):
            return True
        
        # 如果是叶子节点但不是商业分子，路径不完整
        if not node.children:
            return False
        
        # 如果是ReactionNode，检查所有反应物是否都能到达商业分子
        if hasattr(node, 'reaction_info'):  # ReactionNode
            return all(self._is_complete_path_to_node(child) for child in node.children)
        
        # 如果是MoleculeNode，只要有一个子节点能到达商业分子即可
        return any(self._is_complete_path_to_node(child) for child in node.children)

    
    def rollout(self, node: Node) -> np.ndarray:
        """模拟阶段：多目标路径评估 - 返回目标向量"""
        # 构建从根到当前节点的路径
        path_nodes, path_reactions = self._build_path_to_node(node)
        self.logger.debug(f"开始模拟: 当前路径长度={len(path_nodes)}, 反应数={len(path_reactions)}")
        
        # 如果节点已经是终端，直接评估
        if node.is_terminal():
            self.logger.debug(f"终端节点评估完成")
            return self.evaluator.evaluate_path(path_nodes, path_reactions)
        
        # 否则进行随机模拟扩展
        current = node
        simulation_depth = 0
        temp_path_nodes = path_nodes.copy()
        temp_path_reactions = path_reactions.copy()

        self.logger.debug(f"开始随机模拟，最大深度={self.mcts_config.max_simulation_depth}")
        
        while (not current.is_terminal() and 
               simulation_depth < self.mcts_config.max_simulation_depth):
            
            try:
                if isinstance(current, MoleculeNode):
                    if self._calculate_commercial_availability(current.molecule):
                        self.logger.debug(f"模拟中到达商业分子: {current.molecule}")
                        break
                    
                    # 快速模拟：随机选择反应
                    reactions = self.single_fn(current.molecule, self.commercial_molecules)
                    if not reactions:
                        self.logger.debug(f"模拟中无反应可用")
                        break
                    
                    reaction = random.choice(reactions)
                    temp_path_reactions.append(reaction)
                    
                    # 选择一个反应物继续
                    if reaction.reactants:
                        next_molecule = random.choice(reaction.reactants)
                        temp_path_nodes.append(next_molecule)
                        
                        if self._calculate_commercial_availability(next_molecule):
                            self.logger.debug(f"模拟中到达商业反应物: {next_molecule}")
                            break
                        
                        # 创建临时节点继续模拟
                        current = MoleculeNode(
                            next_molecule,
                            is_commercial=self._calculate_commercial_availability(next_molecule),
                        )
                        simulation_depth += 1
                    else:
                        break
                else:
                    break
                    
            except Exception as e:
                self.logger.error(f"模拟过程中出错: {e}")
                break

        self.logger.debug(f"模拟完成: 深度={simulation_depth}, 最终路径长度={len(temp_path_nodes)}")

        return self.evaluator.evaluate_path(temp_path_nodes, temp_path_reactions)
    
    def backpropagation(self, node: Node, objective_vector: np.ndarray):
        """反向传播阶段：更新Pareto前沿"""
        current = node
        
        while current is not None:
            current.visits += 1
            current.add_solution(objective_vector)
            current = current.parent
   
    def _calculate_commercial_availability(self, query_molecule: str):
        if query_molecule in self.similarity_cache:
            similarity = self.similarity_cache[query_molecule]
        else:
            similarity = max(similarity_calculator(query_molecule, self.commercial_molecules), default=0)
            self.similarity_cache[query_molecule] = similarity

        return similarity > 0.99
    
        # 新增方法：计算搜索树节点数
    def _count_tree_nodes(self, node: Node) -> int:
        """计算搜索树节点总数"""
        count = 1
        for child in node.children:
            count += self._count_tree_nodes(child)
        return count
    
    def get_best_solutions(self, root: Node) -> List[np.ndarray]:
        """获取最佳解集（根节点的Pareto前沿）"""
        solutions = root.get_pareto_front()
        self.logger.debug(f"获取Pareto前沿解: {len(solutions)}个")
        return solutions
    
    def get_best_path_by_hypervolume(self, root: Node) -> Tuple[List[str], np.ndarray]:
        best_solutions = self.get_best_solutions(root)
        
        if not best_solutions:
            self.logger.warning("未找到Pareto最优解，返回默认路径")
            return [root.molecule], np.zeros(len(self.eval_config.enabled_objectives))
        
        # 选择最佳解：目标值总和最大
        best_solution = max(best_solutions, key=lambda sol: np.sum(sol))
        
        # 构建路径：跟随访问次数最多的路径
        path = []
        current = root
        
        while current:
            path.append(current.molecule)
            if current.children:
                current = max(current.children, key=lambda x: x.visits)
            else:
                break
        
        return path, best_solution
    
    def _build_path_to_node(self, node: Node) -> Tuple[List[str], List[SingleStepResult]]:
        """构建从根到节点的路径"""
        path = []
        reactions = []
        current = node
        
        # 向上遍历到根节点
        while current is not None:
            path.append(current.molecule)
            if isinstance(current, ReactionNode) and current.reaction_info:
                reactions.append(current.reaction_info)
            current = current.parent
        
        # 反转以获得从根到节点的顺序
        path.reverse()
        reactions.reverse()
        
        return path, reactions
    
    def _real_time_feedback(self, root: Node, iteration: int):
        """实时智能引导反馈"""
        current_solutions = self.get_best_solutions(root)
        ref_point = self.eval_config.get_hypervolume_reference_point()
        current_hypervolume = 0.0
        if current_solutions:
            hv = HV(ref_point=ref_point)
            current_hypervolume = hv(np.array(current_solutions))
        

        elapsed_time = time.time() - self.start_time
        
        self.logger.info(f"=== 实时反馈 (第{iteration}次迭代) ===")
        self.logger.info(f"执行时间: {elapsed_time:.2f}秒")
        self.logger.info(f"当前Pareto前沿大小: {len(current_solutions)}")
        self.logger.info(f"当前超体积: {current_hypervolume:.6f}")
        self.logger.info(f"超体积改进: {current_hypervolume - self.best_hypervolume:.6f}")
        
        # 添加路径发现统计
        if self.analyzer and self.analyzer.path_records:
            path_stats = self.analyzer.get_path_statistics()
            self.logger.info(f"已发现路径数: {path_stats['total_paths']}")
            self.logger.info(f"完整路径数: {path_stats['complete_paths']}")
            self.logger.info(f"路径完成率: {path_stats['completion_rate']:.2%}")
        
        if current_solutions:
            best_path, best_solution = self.get_best_path_by_hypervolume(root)
            self.logger.info(f"当前最佳路径长度: {len(best_path)}")
            self.logger.info(f"最佳解目标值: {best_solution}")
            
            # 输出Pareto前沿的多样性信息
            objectives_matrix = np.array(current_solutions)
            diversity = np.std(objectives_matrix, axis=0)
            self.logger.info(f"目标多样性: {diversity}")
        
        self.best_hypervolume = max(self.best_hypervolume, current_hypervolume)
        self.logger.info("=" * 50)
    
    def _should_early_stop(self) -> bool:
        """早停判断"""
        if self.iteration_count < 200:
            return False
        
        # 使用全局Pareto前沿进行早停判断
        global_front = self.evaluator.get_pareto_front()
        ref_point = self.eval_config.get_hypervolume_reference_point()
        current_hypervolume = 0.0
        if global_front:
            hv = HV(ref_point=ref_point)
            current_hypervolume = hv(np.array(global_front))
        
        improvement = current_hypervolume - self.best_hypervolume
        threshold = self.mcts_config.early_stopping_threshold * self.best_hypervolume * 0.01
        
        should_stop = improvement < threshold
        if should_stop:
            self.logger.info(f"早停条件满足: 改进{improvement:.6f} < 阈值{threshold:.6f}")
        
        return should_stop
    
    def generate_analysis_report(self) -> Optional[Dict]:
        """生成完整的分析报告和可视化"""
        if not self.analyzer:
            self.logger.warning("分析器未启用，无法生成报告")
            return None
        
        try:
            self.logger.info("开始生成分析报告...")
            
            # 1. 路径发现可视化
            self.logger.info("生成路径发现分析图...")
            self.analyzer.plot_path_discovery()
            
            # 2. 导出路径到CSV
            self.logger.info("导出路径信息到CSV...")
            self.analyzer.export_paths_to_csv()
            
            # 3. Pareto前沿可视化
            self.logger.info("生成Pareto前沿图...")
            self.analyzer.plot_pareto_front()
            
            # 4. 收敛性分析
            self.logger.info("生成收敛性分析图...")
            self.analyzer.plot_convergence()
            
            # 5. 支配关系分析
            self.logger.info("分析支配关系...")
            domination_analysis = self.analyzer.analyze_domination_relationships()
            
            # 6. 生成综合报告
            self.logger.info("生成综合报告...")
            comprehensive_report = self.analyzer.generate_comprehensive_report()
            
            # 7. 打印最佳路径
            self.logger.info("分析最佳路径...")
            best_paths = self.analyzer.get_best_paths(top_k=5)
            if best_paths:
                self.logger.info("=== Top 5 Best Paths ===")
                for i, path in enumerate(best_paths, 1):
                    self.logger.info(f"Path {i}: Length={path.path_length}, "
                                   f"Complete={path.is_complete}, "
                                   f"Objectives={path.objective_vector}")
            
            self.logger.info("✅ 分析报告生成完成！")
            self.logger.info(f"报告保存在: {self.analyzer.result_path}")

            # 8. 搜索树可视化
            self.logger.info("生成搜索树可视化...")
            self.analyzer.visualize_search_tree(max_depth=20)
            self.analyzer.visualize_complete_paths_tree()

            # 9. 自动收集并可视化
            # self.analyzer.visualize_search_tree(max_depth=20, save_data_json=True)

            # # 10. 手动收集数据
            # molecule_info, reaction_info, connection_info = self.analyzer.collect_tree_info()
            # self.analyzer.save_tree_data_to_json(molecule_info, reaction_info, connection_info)
            
            return comprehensive_report
            
        except Exception as e:
            self.logger.error(f"❌ 生成分析报告时出错: {e}")
            import traceback
            traceback.print_exc()
            return None