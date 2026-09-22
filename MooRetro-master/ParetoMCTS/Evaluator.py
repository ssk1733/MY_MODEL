import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Set, Union, Callable
import logging
from enum import Enum
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  

# pymoo相关导入
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV

from ParetoMCTS.config import SingleStepResult, EvaluationConfig
from ParetoMCTS.nodes import Node

class ParetoComparison(Enum):
    DOMINATES = 1
    DOMINATED = -1
    NON_DOMINATED = 0

class MolecularDescriptor:
    """分子描述符计算"""
    
    @staticmethod
    def calculate_molecular_weight(smiles: str) -> float:
        return len(smiles) * 12.0
    
    @staticmethod
    def count_rings(smiles: str) -> int:
        return smiles.count('c') + smiles.count('C')
    
    @staticmethod
    def calculate_complexity(smiles: str) -> float:
        mw = MolecularDescriptor.calculate_molecular_weight(smiles)
        rings = MolecularDescriptor.count_rings(smiles)
        heteroatoms = sum(1 for c in smiles if c in 'NOSPF')
        
        mw_score = min(1.0, mw / 500.0)
        ring_score = min(1.0, rings / 3.0)
        hetero_score = min(1.0, heteroatoms / 5.0)
        
        return (mw_score + ring_score + hetero_score) / 3.0

class ThermodynamicsCalculator:
    """热力学计算器"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.reaction_energies = {
            '1.': -50.0, '2.': -30.0, '3.': -20.0,
            '4.': -40.0, '5.': -25.0, '6.': -35.0,
        }
    
    def calculate_feasibility(self, reaction: SingleStepResult) -> float:
        ec_class = reaction.ec.split('.')[0] if '.' in reaction.ec else '1'
        base_energy = self.reaction_energies.get(ec_class, -30.0)
        
        score_factor = (reaction.scores - 0.5) * 20.0
        delta_g = base_energy + score_factor
        
        R = 8.314
        T = self.config.temperature
        
        if delta_g < 0:
            feasibility = 1.0 / (1.0 + math.exp(abs(delta_g) * 1000 / (R * T)))
            return min(1.0, 0.5 + feasibility * 0.5)
        else:
            feasibility = 1.0 / (1.0 + math.exp(delta_g * 1000 / (R * T)))
            return max(0.01, feasibility)

class PathEvaluator:
    """简化的多目标路径评估器 - 直接返回numpy数组"""
    
    def __init__(self, commercial_molecules: List[str], config: Optional[EvaluationConfig]):
        self.commercial_molecules = commercial_molecules
        self.config = config or EvaluationConfig()
        self.thermodynamics_calc = ThermodynamicsCalculator(self.config)
        self.molecular_descriptor = MolecularDescriptor()
        
        # 简化：只保留核心数据
        self.global_solutions: List[np.ndarray] = []
        self._evaluation_cache: Dict[str, np.ndarray] = {}
        self.evaluation_count = 0
        self.cache_hits = 0
        self.logger = logging.getLogger(__name__)
    
    def evaluate_path(self, path_nodes: List[Union[str, 'Node']], 
                      path_reactions: List[SingleStepResult]) -> np.ndarray:
        """路径评估 - 直接返回目标向量"""
        self.evaluation_count += 1
        
        # 简化缓存键生成
        cache_key = f"{len(path_nodes)}_{len(path_reactions)}_{hash(tuple(str(n) for n in path_nodes))}"
        if self.config.enable_cache and cache_key in self._evaluation_cache:
            self.cache_hits += 1
            return self._evaluation_cache[cache_key]
        
        # 计算各个指标
        path_length = len(path_reactions)
        yield_score = self._calculate_yield_score(path_reactions)
        thermodynamic_feasibility = self._calculate_thermodynamic_feasibility(path_reactions)
        average_step_score = self._calculate_average_step_score(path_reactions)
        complexity_score = self._calculate_complexity_score(path_nodes)
        commercial_availability_score = self._calculate_commercial_availability(path_nodes)
        synthetic_accessibility_score = self._calculate_synthetic_accessibility(path_nodes, path_reactions)
        
        # 目标映射字典
        objective_map = {
            "path_efficiency": 1.0 / max(1, path_length),
            "yield_score": yield_score,
            "thermodynamic_feasibility": thermodynamic_feasibility,
            "average_step_score": average_step_score,
            "simplicity_score": 1.0 - complexity_score,
            "commercial_availability": commercial_availability_score,
            "synthetic_accessibility": synthetic_accessibility_score
        }
        
        # 直接生成目标向量 
        obj_vector = np.array([objective_map[obj] for obj in self.config.enabled_objectives 
                              if obj in objective_map])
        
        if self.config.enable_cache:
            self._evaluation_cache[cache_key] = obj_vector
        
        # 更新全局解集
        self.global_solutions.append(obj_vector)
        return obj_vector
    
    def multi_objective_ucb(self, node_solutions: List[np.ndarray], 
                           visits: int, parent_visits: int, c: float = 1.4) -> float:
        """简化的UCB计算"""
        if visits == 0:
            return float('inf')
        
        if not node_solutions:
            return math.sqrt(math.log(parent_visits) / visits) * c
        
        # 简化：使用平均值代替超体积
        avg_performance = np.mean(node_solutions, axis=0).sum()
        exploitation = avg_performance / len(node_solutions)
        exploration = math.sqrt(math.log(parent_visits) / visits) * c
        return exploitation + exploration
    
    def calculate_hypervolume(self, solutions: List[np.ndarray]) -> float:
        """使用pymoo计算超体积"""
        if not solutions:
            return 0.0
        ref_point = self.config.get_hypervolume_reference_point()
        hv = HV(ref_point=ref_point)
        return hv(np.array(solutions))
    
    def get_pareto_front(self) -> List[np.ndarray]:
        """获取全局Pareto前沿"""
        if not self.global_solutions:
            return []
        F = np.array(self.global_solutions)
        nds = NonDominatedSorting()
        front_indices = nds.do(F, only_non_dominated_front=True)
        return F[front_indices].tolist()
    
    # ===== 私有计算方法 =====
    def _calculate_yield_score(self, path_reactions: List[SingleStepResult]) -> float:
        if not path_reactions:
            return 1.0
        return np.prod([max(0.1, r.scores) for r in path_reactions])
    
    def _calculate_thermodynamic_feasibility(self, path_reactions: List[SingleStepResult]) -> float:
        if not path_reactions:
            return 1.0
        feasibilities = [self.thermodynamics_calc.calculate_feasibility(r) for r in path_reactions]
        return np.mean(feasibilities)
    
    def _calculate_average_step_score(self, path_reactions: List[SingleStepResult]) -> float:
        if not path_reactions:
            return 1.0
        return np.mean([r.scores for r in path_reactions])
    
    def _calculate_complexity_score(self, path_nodes: List) -> float:
        if not path_nodes:
            return 0.0
        molecule_nodes = [node for node in path_nodes if hasattr(node, 'molecule')]
        if not molecule_nodes:
            return 0.0
        complexities = [self.molecular_descriptor.calculate_complexity(node.molecule) 
                       for node in molecule_nodes]
        return np.mean(complexities)
    
    def _calculate_commercial_availability(self, path_nodes: List) -> float:
        if not path_nodes:
            return 0.0
        molecule_nodes = [node for node in path_nodes if hasattr(node, 'molecule')]
        if not molecule_nodes:
            return 0.0
        commercial_count = sum(1 for node in molecule_nodes if getattr(node, 'is_commercial', False))
        return commercial_count / len(molecule_nodes)
    
    def _calculate_synthetic_accessibility(self, path_nodes: List, 
                                         path_reactions: List[SingleStepResult]) -> float:
        complexity_factor = 1.0 - self._calculate_complexity_score(path_nodes)
        if not path_reactions:
            return complexity_factor
        reaction_ease = np.mean([r.scores for r in path_reactions])
        return (complexity_factor + reaction_ease) / 2.0
    
    def get_evaluation_statistics(self) -> Dict:
        """获取评估统计信息"""
        cache_hit_rate = self.cache_hits / max(1, self.evaluation_count)
        pareto_front = self.get_pareto_front()
        
        return {
            'total_evaluations': self.evaluation_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self._evaluation_cache),
            'pareto_front_size': len(pareto_front),
            'hypervolume': self.calculate_hypervolume(pareto_front) if pareto_front else 0.0
        }