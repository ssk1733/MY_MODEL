from enum import Enum
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from ParetoMCTS.Evaluator import PathEvaluator
from ParetoMCTS.config import SelectionConfig, SelectionStrategy
from ParetoMCTS.nodes import Node
from utils.chem import similarity_calculator


class EnhancedSelector:
    """增强选择器"""
    
    def __init__(self, commercial_molecules: List[str], 
                 evaluator: 'PathEvaluator',
                 config: SelectionConfig):
        self.commercial_molecules = commercial_molecules
        self.evaluator = evaluator
        self.config = config
        
        # 缓存相似度计算结果
        self.similarity_cache = {}
        
    def calculate_commercial_similarity(self, molecule: str) -> float:
        """计算与商业分子的最大相似度"""
        if molecule in self.similarity_cache:
            return self.similarity_cache[molecule]
            
        if not self.commercial_molecules:
            return 0.0
            
        # 使用已有的相似度计算函数
        max_similarity = max(similarity_calculator(molecule, self.commercial_molecules))
        
        self.similarity_cache[molecule] = max_similarity
        return max_similarity
    
    def calculate_diversity_score(self, molecule: str, visited_molecules: set) -> float:
        """计算分子多样性得分（鼓励探索新的化学空间）"""
        if not visited_molecules:
            return 1.0
            
        # 计算与已访问分子的平均相似度，相似度越低多样性越高
        similarities = similarity_calculator(molecule, list(visited_molecules))

        avg_similarity = np.mean(similarities) if similarities else 0.0
        return 1.0 - avg_similarity  # 转换为多样性得分
    
    def similarity_guided_ucb(self, child: 'Node', parent_visits: int, 
                            visited_molecules: set, c: float = 1.4) -> float:
        """相似度引导的UCB选择"""
        if child.visits == 0:
            # 未访问节点：结合相似度和多样性
            similarity_score = self.calculate_commercial_similarity(child.molecule)
            diversity_score = self.calculate_diversity_score(child.molecule, visited_molecules)
            
            # 平衡相似度和多样性
            heuristic_score = (
                similarity_score * (1 - self.config.diversity_bonus) + 
                diversity_score * self.config.diversity_bonus
            )
            
            return float('inf') * (0.5 + heuristic_score)  # 保持高优先级但加入启发
        
        # 标准UCB计算
        standard_ucb = self.evaluator.multi_objective_ucb(
            child.solutions, child.visits, parent_visits, c
        )
        
        # 相似度引导项
        similarity_bonus = self.calculate_commercial_similarity(child.molecule)
        
        # 组合UCB和相似度
        enhanced_ucb = (
            standard_ucb * self.config.ucb_weight + 
            similarity_bonus * self.config.similarity_weight
        )
        
        return enhanced_ucb
    
    def multi_criteria_selection(self, children: List['Node'], parent_visits: int,
                                visited_molecules: set) -> 'Node':
        """多准则选择：同时考虑UCB、相似度、深度等因素"""
        if not children:
            return None
            
        best_child = None
        best_score = float('-inf')
        
        for child in children:
            # 1. UCB分量
            if child.visits == 0:
                ucb_score = 10.0  # 高优先级给未访问节点
            else:
                ucb_score = self.evaluator.multi_objective_ucb(
                    child.solutions, child.visits, parent_visits, 1.4
                )
            
            # 2. 相似度分量
            similarity_score = self.calculate_commercial_similarity(child.molecule)
            
            # 3. 深度奖励（更深的节点可能更接近商业分子）
            depth_penalty = 0.95 ** child.depth  # 深度惩罚因子
            
            # 4. 多样性奖励
            diversity_score = self.calculate_diversity_score(child.molecule, visited_molecules)
            
            # 5. 是否为终端节点的奖励
            terminal_bonus = 2.0 if child.is_terminal() else 0.0
            
            # 组合得分
            combined_score = (
                ucb_score * 0.4 +
                similarity_score * 0.3 +
                depth_penalty * 0.1 +
                diversity_score * 0.1 +
                terminal_bonus * 0.1
            )
            
            if combined_score > best_score:
                best_score = combined_score
                best_child = child
                
        return best_child
    
    def adaptive_selection(self, children: List['Node'], parent_visits: int,
                          visited_molecules: set, total_iterations: int) -> 'Node':
        """自适应选择：根据搜索进度调整策略"""
        
        # 早期：更多探索，使用相似度引导
        if total_iterations < self.config.adaptive_threshold:
            return self._select_by_similarity_guided_ucb(children, parent_visits, visited_molecules)
        
        # 后期：更多利用，使用多准则选择
        else:
            return self.multi_criteria_selection(children, parent_visits, visited_molecules)
    
    def _select_by_similarity_guided_ucb(self, children: List['Node'], 
                                       parent_visits: int, visited_molecules: set) -> 'Node':
        """基于相似度引导UCB的选择"""
        best_child = None
        best_ucb = float('-inf')
        
        for child in children:
            ucb_value = self.similarity_guided_ucb(child, parent_visits, visited_molecules)
            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_child = child
                
        return best_child
    
    def select_child(self, node: 'Node', 
                     visited_molecules: set, 
                    total_iterations: int = 0) -> 'Node':
        """主选择方法"""
        if not node.children:
            return None
            
        # 根据配置的策略选择
        if self.config.strategy == SelectionStrategy.STANDARD_UCB:
            # 标准UCB选择
            best_child = None
            best_ucb = float('-inf')
            for child in node.children:
                if child.visits == 0:
                    return child
                ucb_value = self.evaluator.multi_objective_ucb(
                    child.solutions, child.visits, node.visits, 1.4
                )
                if ucb_value > best_ucb:
                    best_ucb = ucb_value
                    best_child = child
            return best_child
            
        elif self.config.strategy == SelectionStrategy.SIMILARITY_GUIDED:
            return self._select_by_similarity_guided_ucb(node.children, node.visits, visited_molecules)
            
        elif self.config.strategy == SelectionStrategy.MULTI_CRITERIA:
            return self.multi_criteria_selection(node.children, node.visits, visited_molecules)
            
        elif self.config.strategy == SelectionStrategy.ADAPTIVE:
            return self.adaptive_selection(node.children, node.visits, visited_molecules, total_iterations)
            
        else:
            return self._select_by_similarity_guided_ucb(node.children, node.visits, visited_molecules)
