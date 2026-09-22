'''
# evaluator.py - 重新设计的多目标路径评估器
import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Callable
import logging
from enum import Enum
import random 
from dataclasses import dataclass

# pymoo相关导入
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV

from config.config import SingleStepResult, EvaluationConfig
from utils.chem import similarity_calculator, get_delta_sascore_for_feasibility
# from mooseeker.fit_funcs.get_gibbs import get_gibbs


class AggregationStrategy(Enum):
    """聚合策略枚举"""
    MEAN = "mean"              # 平均值
    MIN = "min"                # 最小值
    MAX = "max"                # 最大值
    GEOMETRIC_MEAN = "geom"    # 几何平均
    HARMONIC_MEAN = "harm"     # 调和平均
    WEIGHTED_MEAN = "weighted" # 加权平均


@dataclass
class SingleStepMetrics:
    """单步反应指标"""
    similarity_score: float      # 与商业分子的相似性
    feasibility_score: float     # 反应可行性 : delta SAscore
    retro_score: float          # 逆合成分数
    theoretical_yield: float     # 理论产量
    complexity_penalty: float   # 复杂度惩罚
    thermodynamic_feasibility: float = 0.0 # 热力学可行性
    toxicity: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            'similarity': round(self.similarity_score, 3),
            'feasibility': round(self.feasibility_score, 3),
            'retro_score': round(self.retro_score, 3),
            'yield': round(self.theoretical_yield, 3),
            'complexity': round(self.complexity_penalty, 3),
            'thermodynamic': round(self.thermodynamic_feasibility, 3),
            'toxicity': round(self.toxicity, 3),
        }


@dataclass
class PathMetrics:
    """多步路径指标"""
    step_metrics: List[SingleStepMetrics]  # 每步详细指标
    
    path_length: int                   # 路径长度
    overall_retro_score: float         # 整体逆合成分数 avg(retro_scores)
    overall_feasibility: float         # 整体可行性    min(feasibilities) 强调最难的步骤（瓶颈）
    
    overall_similarity: float          # 整体相似性 max(similarity for stepi in steps) 如果是complete path，那肯定是 1.0 
    overall_yield: float               # 整体理论产量
    overall_thermodynamics: float      # 整体热力学可行性 sum(delta Gi) 越小越表示可以自发进行 
    efficiency_score: float            # 效率分数

    overall_toxicity: float = 0.0       #化合物毒性 max ()

    @property
    def to_list(self) -> List:
        return [self.path_length, self.overall_retro_score, self.overall_feasibility]
    
    @property
    def to_dict(self) -> Dict:
        return {
            'path_length': self.path_length,
            'overall_retro_score': round(self.overall_retro_score,3),
            'overall_feasibility': round(self.overall_feasibility,3),
            'overall_similarity': round(self.overall_similarity, 3),
            'overall_theoretical_yield': round(self.overall_yield, 3),
            'overall_thermodynamics': round(self.overall_thermodynamics, 3),
            'overall_toxicity': round(self.overall_toxicity, 3),
            # 'efficiency_score': round(self.efficiency_score,3),
            # 'step_metrics': [step.to_dict() for step in self.step_metrics]
        }


class SingleStepEvaluator:

    """单步反应评估器"""
    
    def __init__(self, commercial_molecules: List[str], config: EvaluationConfig):
        self.commercial_molecules = set(commercial_molecules)
        self.config = config
        self.similarity_cache = {}
    
    def evaluate_step(self, 
                     reactant_molecule: str,
                     reaction: SingleStepResult,
                     target_molecule: str) -> SingleStepMetrics:
        """评估单步反应"""
        
        # 1. 分子相似性分数
        similarity_score = reaction.similarity if reaction.similarity else self._calculate_similarity_score(reactant_molecule)
        
        # 2. 反应可行性分数
        feasibility_score = reaction.feasibility
        # self._calculate_step_feasibility_score(target_molecule, reactant_molecule)
        
        # 3. 逆合成分数（直接使用reaction.scores）
        retro_score = float(reaction.scores)
        
        # 4. 理论产量
        theoretical_yield =reaction.yield_estimate if reaction.yield_estimate else self._calculate_theoretical_yield(reaction)
        
        # 5. 复杂度惩罚
        complexity_penalty = self._calculate_complexity_penalty(reactant_molecule, target_molecule)
        
        # 6. Toxicity
        toxicity = reaction.toxicity if reaction.toxicity else self._calculate_toxicity(reactant_molecule)

        return SingleStepMetrics(
            similarity_score=similarity_score,
            feasibility_score=feasibility_score,
            retro_score=retro_score,
            theoretical_yield=theoretical_yield,
            complexity_penalty=complexity_penalty,
            toxicity=toxicity,
        )
    
    def _calculate_similarity_score(self, molecule: str) -> float:
        """计算分子与商业分子的最高相似性"""
        if molecule in self.similarity_cache:
            return self.similarity_cache[molecule]
        
        if molecule in self.commercial_molecules:
            similarity = 1.0
        else:
            similarities = similarity_calculator(molecule, list(self.commercial_molecules))
            similarity = max(similarities) if similarities else 0.0
        
        self.similarity_cache[molecule] = similarity
        return round(similarity, 3)
    
    def _calculate_toxicity(self, reactant):
        """计算当前反应中底物分子的毒性"""
        return random.random() # [0,1]
    # def _calculate_step_feasibility_score(self, product_smiles: str, reactant_smiles: str) -> float:

    #     return get_delta_sascore_for_feasibility(product_smiles,
    #                                              reactant_smiles)
    
    def _calculate_theoretical_yield(self, reaction: SingleStepResult) -> float:
        """计算理论产量"""
        # 如果reaction有yield字段，直接使用
        # if hasattr(reaction, 'yield') and reaction.yield is not None:
        #     return float(reaction.yield)
        
        # 否则基于分数和置信度估算
        base_yield = 0.6 + 0.4 * reaction.scores  # 基础产量60-100%
        confidence_factor = getattr(reaction, 'confidence', 0.8)
        
        # 考虑置信度的影响
        theoretical_yield = base_yield * (0.8 + 0.2 * confidence_factor)
        return min(1.0, max(0.1, theoretical_yield))
    
    def _calculate_complexity_penalty(self, reactant: str, target: str) -> float:
        """计算复杂度变化惩罚"""
        # 简单的复杂度估算（可以用更复杂的描述符）
        reactant_complexity = self._estimate_complexity(reactant)
        target_complexity = self._estimate_complexity(target)
        
        # 如果反应物比目标分子复杂很多，给予惩罚
        complexity_increase = max(0, reactant_complexity - target_complexity)
        penalty = 1.0 / (1.0 + complexity_increase)  # 越复杂惩罚越重
        return penalty
    
    def _estimate_complexity(self, molecule: str) -> float:
        """估算分子复杂度"""
        # 简单的复杂度估算
        length_factor = len(molecule) / 50.0
        ring_factor = molecule.count('c') + molecule.count('C')
        hetero_factor = sum(1 for c in molecule if c in 'NOSPF')
        
        complexity = length_factor + ring_factor * 0.2 + hetero_factor * 0.1
        return complexity


class PathAggregator:
    """路径指标聚合器"""
    
    @staticmethod
    def aggregate_metrics(step_metrics: List[SingleStepMetrics], 
                          strategies: Dict[str, AggregationStrategy]) -> Dict[str, float]:
        """聚合多步指标"""
        if not step_metrics:
            return {}
        
        # 提取各指标的值
        metrics_dict = {
            'similarity': [m.similarity_score for m in step_metrics],
            'feasibility': [m.feasibility_score for m in step_metrics],
            'retro_score': [m.retro_score for m in step_metrics],
            'yield': [m.theoretical_yield for m in step_metrics],
            'complexity': [m.complexity_penalty for m in step_metrics],
            'toxicity': [m.toxicity for m in step_metrics]
        }
        
        # 应用聚合策略
        aggregated = {}
        for metric_name, values in metrics_dict.items():
            strategy = strategies.get(metric_name, AggregationStrategy.MEAN)
            aggregated[metric_name] = PathAggregator._apply_strategy(values, strategy)
        
        return aggregated
    
    @staticmethod
    def _apply_strategy(values: List[float], strategy: AggregationStrategy) -> float:
        """应用聚合策略"""
        if not values:
            return 0.0
        
        values = np.array(values)
        
        if strategy == AggregationStrategy.MEAN:
            return float(np.mean(values))
        elif strategy == AggregationStrategy.MIN:
            return float(np.min(values))
        elif strategy == AggregationStrategy.MAX:
            return float(np.max(values))
        elif strategy == AggregationStrategy.GEOMETRIC_MEAN:
            # 几何平均（适用于产量计算）
            return float(np.prod(values) ** (1.0 / len(values)))
        elif strategy == AggregationStrategy.HARMONIC_MEAN:
            # 调和平均（适用于瓶颈步骤）
            return float(len(values) / np.sum(1.0 / (values + 1e-10)))
        else:
            return float(np.mean(values))


class PathEvaluator:
    """多目标路径评估器"""
    
    def __init__(self, commercial_molecules: List[str], 
                 config: Optional[EvaluationConfig] = None,
                 ):
        
        self.config = config or EvaluationConfig()
        self.single_step_evaluator = SingleStepEvaluator(commercial_molecules, self.config)
        self.aggregator = PathAggregator()
        
        # 聚合策略配置
        self.aggregation_strategies = {
            'similarity': AggregationStrategy.MAX,         # 最高相似性
            'feasibility': AggregationStrategy.MIN,        # 最低可行性（瓶颈）
            'retro_score': AggregationStrategy.MEAN,       # 平均逆合成分数
            'yield': AggregationStrategy.GEOMETRIC_MEAN,   # 几何平均产量
            'complexity': AggregationStrategy.MEAN,        # 平均复杂度
            'toxicity': AggregationStrategy.MAX            # 最大毒性 
        }
        
        # 缓存和统计
        self._evaluation_cache = {}
        self.evaluation_count = 0
        self.cache_hits = 0
        self.global_solutions = []

    def evaluate_path(self, 
                    #   path: SynthesisPath,
                     path_molecules: List[str], 
                     path_reactions: List[SingleStepResult],
                     is_complete: bool = False
                     ) -> List:#np.ndarray:
        """评估完整路径"""
        
        # 处理不完整路径
        if not is_complete:
            objective_vector = self._evaluate_incomplete_path(path_molecules)
            
        
        else:
            path_metrics = self._evaluate_complete_path(path_molecules, path_reactions)
            # 生成目标向量
            objective_vector = self._generate_objective_vector(path_metrics)
        
        return objective_vector.tolist()
    
    def _evaluate_complete_path(self, 
                                path_molecules: List[str], 
                                path_reactions: List[SingleStepResult]) -> PathMetrics:
        self.evaluation_count += 1
        
        # 计算每步指标
        step_metrics = []
        for i, reaction in enumerate(path_reactions):
            if i < len(path_molecules) - 1:
                reactant = path_molecules[i + 1]  # 反应物
                target = path_molecules[i]        # 目标分子
                step_metric = self.single_step_evaluator.evaluate_step(reactant, reaction, target)
                step_metrics.append(step_metric)
        
        # 聚合路径指标
        aggregated_metrics = self.aggregator.aggregate_metrics(step_metrics, self.aggregation_strategies)
        
        # 计算路径级别的综合指标
        path_metrics = self._calculate_path_metrics(path_molecules, path_reactions, 
                                                   step_metrics, aggregated_metrics)
        
        return path_metrics
    
    def _evaluate_incomplete_path(self, path_molecules: List[str]) -> np.ndarray:
        """评估不完整路径（启发式惩罚）"""
        if not path_molecules:
            return np.full(len(self.config.enabled_objectives), self.config.base_penalty_value)
        
        # 计算到商业分子的最高相似性
        last_molecule = path_molecules[-1]
        similarity = self.single_step_evaluator._calculate_similarity_score(last_molecule)
        
        # 路径长度因子
        path_length_factor = np.exp(-self.config.path_length_penalty_factor * len(path_molecules))
        
        # 组合奖励因子
        reward_factor = (self.config.similarity_weight * similarity + 
                        (1 - self.config.similarity_weight) * path_length_factor)
        
        # 启发式惩罚
        heuristic_penalty = self.config.base_penalty_value * (1 - reward_factor)
        
        return np.full(len(self.config.enabled_objectives), heuristic_penalty)
    
    def _calculate_path_metrics(self, 
                               path_molecules: List[str],
                               path_reactions: List[SingleStepResult],
                               step_metrics: List[SingleStepMetrics],
                               aggregated_metrics: Dict[str, float]) -> PathMetrics:
        """计算路径综合指标"""
        
        path_length = len(path_reactions)
        
        # 从聚合指标中提取
        overall_similarity = aggregated_metrics.get('similarity', 0.0)
        overall_feasibility = aggregated_metrics.get('feasibility', 0.0)
        overall_retro_score = aggregated_metrics.get('retro_score', 0.0)
        overall_yield = aggregated_metrics.get('yield', 0.0)
        overall_toxicity = aggregated_metrics.get('toxicity', 0.0)
        
        # 计算效率分数（综合考虑路径长度和质量）
        quality_score = (overall_similarity + overall_feasibility + overall_retro_score) / 3.0
        efficiency_score = quality_score / max(1, path_length)  # 质量除以长度
        overall_themodynamics = self._calculate_thermodynamics(path_reactions)

        return PathMetrics(
            path_length=path_length,
            overall_similarity=overall_similarity,
            overall_feasibility=overall_feasibility,
            overall_retro_score=overall_retro_score,
            overall_thermodynamics=overall_themodynamics,
            overall_yield=overall_yield,
            efficiency_score=efficiency_score,
            overall_toxicity=overall_toxicity,
            step_metrics=step_metrics
        )
    
    def _calculate_thermodynamics(self, path_reactions):
        try:
            # gibbs = get_gibbs(path_reactions)
            gibbs = - random.random()
            return gibbs
        except:
            return 0.0
    
    def _generate_objective_vector(self, path_metrics: PathMetrics) -> np.ndarray:
        """生成目标向量"""
        # 目标映射字典（所有目标都是最大化）
        objective_map = {
            "path_efficiency": path_metrics.efficiency_score,
            "overall_similarity": path_metrics.overall_similarity,
            "overall_feasibility": path_metrics.overall_feasibility,
            "overall_yield": path_metrics.overall_yield,
            "overall_retro_score": path_metrics.overall_retro_score,
            "overall_toxicity": path_metrics.overall_toxicity,
            "overall_thermodynamic": path_metrics.overall_thermodynamics,
            "path_length_inv": 1.0 / max(1, path_metrics.path_length),  # 路径长度倒数
        }
        
        # 根据配置选择启用的目标
        objective_values = []
        for obj_name in self.config.enabled_objectives:
            if obj_name in objective_map:
                objective_values.append(objective_map[obj_name])
            else:
                objective_values.append(0.0)
        
        return np.array(objective_values)
    
    def _generate_cache_key(self, molecules: List[str], reactions: List[SingleStepResult]) -> str:
        """生成缓存键"""
        mol_hash = hash(tuple(molecules))
        rxn_hash = hash(tuple(r.templates for r in reactions))
        return f"{len(molecules)}_{len(reactions)}_{mol_hash}_{rxn_hash}"
    
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
            
            hv = HV(ref_point=self.config.get_hypervolume_reference_point())
            return float(hv(solutions_array))
        except Exception as e:
            return 0.0
    
    def get_evaluation_statistics(self) -> Dict:
        """获取评估统计信息"""
        pareto_front = self.get_pareto_front()
        cache_hit_rate = self.cache_hits / max(1, self.evaluation_count)
        
        return {
            'total_evaluations': self.evaluation_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self._evaluation_cache),
            'pareto_front_size': len(pareto_front),
            'hypervolume': self.calculate_hypervolume(pareto_front),
            'aggregation_strategies': {k: v.value for k, v in self.aggregation_strategies.items()}
        }
    
    def set_aggregation_strategy(self, metric: str, strategy: AggregationStrategy):
        """设置特定指标的聚合策略"""
        self.aggregation_strategies[metric] = strategy


# ========================================================================
# 使用示例和工厂函数
# ========================================================================

def create_standard_evaluator(commercial_molecules: List[str], 
                             config: Optional[EvaluationConfig] = None) -> PathEvaluator:
    """创建标准配置的评估器"""
    if config is None:
        config = EvaluationConfig()
        config.enabled_objectives = [
            "path_efficiency",
            "overall_similarity", 
            "overall_feasibility",
            "overall_yield"
        ]
    
    evaluator = PathEvaluator(commercial_molecules, config)
    
    # 设置推荐的聚合策略
    evaluator.set_aggregation_strategy('similarity', AggregationStrategy.MAX)
    evaluator.set_aggregation_strategy('feasibility', AggregationStrategy.MIN)
    evaluator.set_aggregation_strategy('yield', AggregationStrategy.GEOMETRIC_MEAN)
    evaluator.set_aggregation_strategy('retro_score', AggregationStrategy.MEAN)
    
    return evaluator


# ========================================================================
# 测试和验证函数
# ========================================================================

def test_evaluator():
    """测试评估器功能"""
    from ..config.config import SingleStepResult, EvaluationConfig
    
    # 创建测试数据
    commercial_mols = ["CCO", "CC", "CO"]  # 简单商业分子
    config = EvaluationConfig()
    config.enabled_objectives = ["path_efficiency", "overall_retro_score", "overall_feasibility"]
    
    evaluator = create_standard_evaluator(commercial_mols, config)
    
    # 创建测试路径
    test_molecules = ["CCCCCO", "CCCO", "CCO"]  # 从复杂到简单
    test_reactions = [
        SingleStepResult(
            reactants=["CCCO"], 
            scores=0.8, 
            templates="template1", 
            ec="1.1.1.1"
        ),
        SingleStepResult(
            reactants=["CCO"], 
            scores=0.9, 
            templates="template2", 
            ec="2.1.1.1"
        )
    ]
    
    # 评估路径
    result = evaluator.evaluate_path(test_molecules, test_reactions, is_complete=True)
    
    print("测试结果:")
    print(f"目标向量: {result}")
    print(f"评估统计: {evaluator.get_evaluation_statistics()}")
    
    return evaluator


if __name__ == "__main__":
    test_evaluator()
'''

# evaluator.py - 重新设计的单目标路径评估器（去掉帕累托）
import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Callable
import logging
from enum import Enum
import random 
from dataclasses import dataclass

# 不再需要 pymoo 的多目标相关导入
# from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
# from pymoo.indicators.hv import HV

from config.config import SingleStepResult, EvaluationConfig
from utils.chem import similarity_calculator, get_delta_sascore_for_feasibility


class AggregationStrategy(Enum):
    """聚合策略枚举"""
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    GEOMETRIC_MEAN = "geom"
    HARMONIC_MEAN = "harm"
    WEIGHTED_MEAN = "weighted"


@dataclass
class SingleStepMetrics:
    """单步反应指标"""
    similarity_score: float
    feasibility_score: float
    retro_score: float
    theoretical_yield: float
    complexity_penalty: float
    thermodynamic_feasibility: float = 0.0
    toxicity: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            'similarity': round(self.similarity_score, 3),
            'feasibility': round(self.feasibility_score, 3),
            'retro_score': round(self.retro_score, 3),
            'yield': round(self.theoretical_yield, 3),
            'complexity': round(self.complexity_penalty, 3),
            'thermodynamic': round(self.thermodynamic_feasibility, 3),
            'toxicity': round(self.toxicity, 3),
        }


@dataclass
class PathMetrics:
    """多步路径指标"""
    step_metrics: List[SingleStepMetrics]
    path_length: int
    overall_retro_score: float
    overall_feasibility: float
    overall_similarity: float
    overall_yield: float
    overall_thermodynamics: float
    efficiency_score: float
    overall_toxicity: float = 0.0

    @property
    def to_list(self) -> List:
        return [self.path_length, self.overall_retro_score, self.overall_feasibility]
    
    @property
    def to_dict(self) -> Dict:
        return {
            'path_length': self.path_length,
            'overall_retro_score': round(self.overall_retro_score,3),
            'overall_feasibility': round(self.overall_feasibility,3),
            'overall_similarity': round(self.overall_similarity, 3),
            'overall_theoretical_yield': round(self.overall_yield, 3),
            'overall_thermodynamics': round(self.overall_thermodynamics, 3),
            'overall_toxicity': round(self.overall_toxicity, 3),
        }


class SingleStepEvaluator:
    """单步反应评估器"""
    
    def __init__(self, commercial_molecules: List[str], config: EvaluationConfig):
        self.commercial_molecules = set(commercial_molecules)
        self.config = config
        self.similarity_cache = {}
    
    def evaluate_step(self, 
                     reactant_molecule: str,
                     reaction: SingleStepResult,
                     target_molecule: str) -> SingleStepMetrics:
        """评估单步反应"""
        similarity_score = reaction.similarity if reaction.similarity else self._calculate_similarity_score(reactant_molecule)
        feasibility_score = reaction.feasibility
        retro_score = float(reaction.scores)
        theoretical_yield = reaction.yield_estimate if reaction.yield_estimate else self._calculate_theoretical_yield(reaction)
        complexity_penalty = self._calculate_complexity_penalty(reactant_molecule, target_molecule)
        toxicity = reaction.toxicity if reaction.toxicity else self._calculate_toxicity(reactant_molecule)

        return SingleStepMetrics(
            similarity_score=similarity_score,
            feasibility_score=feasibility_score,
            retro_score=retro_score,
            theoretical_yield=theoretical_yield,
            complexity_penalty=complexity_penalty,
            toxicity=toxicity,
        )
    
    def _calculate_similarity_score(self, molecule: str) -> float:
        if molecule in self.similarity_cache:
            return self.similarity_cache[molecule]
        if molecule in self.commercial_molecules:
            similarity = 1.0
        else:
            similarities = similarity_calculator(molecule, list(self.commercial_molecules))
            similarity = max(similarities) if similarities else 0.0
        self.similarity_cache[molecule] = similarity
        return round(similarity, 3)
    
    def _calculate_toxicity(self, reactant):
        return random.random()
    
    def _calculate_theoretical_yield(self, reaction: SingleStepResult) -> float:
        base_yield = 0.6 + 0.4 * reaction.scores
        confidence_factor = getattr(reaction, 'confidence', 0.8)
        theoretical_yield = base_yield * (0.8 + 0.2 * confidence_factor)
        return min(1.0, max(0.1, theoretical_yield))
    
    def _calculate_complexity_penalty(self, reactant: str, target: str) -> float:
        reactant_complexity = self._estimate_complexity(reactant)
        target_complexity = self._estimate_complexity(target)
        complexity_increase = max(0, reactant_complexity - target_complexity)
        penalty = 1.0 / (1.0 + complexity_increase)
        return penalty
    
    def _estimate_complexity(self, molecule: str) -> float:
        length_factor = len(molecule) / 50.0
        ring_factor = molecule.count('c') + molecule.count('C')
        hetero_factor = sum(1 for c in molecule if c in 'NOSPF')
        complexity = length_factor + ring_factor * 0.2 + hetero_factor * 0.1
        return complexity


class PathAggregator:
    """路径指标聚合器"""
    
    @staticmethod
    def aggregate_metrics(step_metrics: List[SingleStepMetrics], 
                          strategies: Dict[str, AggregationStrategy]) -> Dict[str, float]:
        if not step_metrics:
            return {}
        metrics_dict = {
            'similarity': [m.similarity_score for m in step_metrics],
            'feasibility': [m.feasibility_score for m in step_metrics],
            'retro_score': [m.retro_score for m in step_metrics],
            'yield': [m.theoretical_yield for m in step_metrics],
            'complexity': [m.complexity_penalty for m in step_metrics],
            'toxicity': [m.toxicity for m in step_metrics]
        }
        aggregated = {}
        for metric_name, values in metrics_dict.items():
            strategy = strategies.get(metric_name, AggregationStrategy.MEAN)
            aggregated[metric_name] = PathAggregator._apply_strategy(values, strategy)
        return aggregated
    
    @staticmethod
    def _apply_strategy(values: List[float], strategy: AggregationStrategy) -> float:
        if not values:
            return 0.0
        values = np.array(values)
        if strategy == AggregationStrategy.MEAN:
            return float(np.mean(values))
        elif strategy == AggregationStrategy.MIN:
            return float(np.min(values))
        elif strategy == AggregationStrategy.MAX:
            return float(np.max(values))
        elif strategy == AggregationStrategy.GEOMETRIC_MEAN:
            return float(np.prod(values) ** (1.0 / len(values)))
        elif strategy == AggregationStrategy.HARMONIC_MEAN:
            return float(len(values) / np.sum(1.0 / (values + 1e-10)))
        else:
            return float(np.mean(values))


class PathEvaluator:
    """单目标路径评估器（使用加权和）"""
    
    def __init__(self, commercial_molecules: List[str], 
                 config: Optional[EvaluationConfig] = None):
        
        self.config = config or EvaluationConfig()
        self.single_step_evaluator = SingleStepEvaluator(commercial_molecules, self.config)
        self.aggregator = PathAggregator()
        
        self.aggregation_strategies = {
            'similarity': AggregationStrategy.MAX,
            'feasibility': AggregationStrategy.MIN,
            'retro_score': AggregationStrategy.MEAN,
            'yield': AggregationStrategy.GEOMETRIC_MEAN,
            'complexity': AggregationStrategy.MEAN,
            'toxicity': AggregationStrategy.MAX
        }
        
        self._evaluation_cache = {}
        self.evaluation_count = 0
        self.cache_hits = 0
        # 存储所有路径的单目标得分（不再存储向量）
        self.global_solutions = []   # 现在存储 float 标量

    def evaluate_path(self, 
                     path_molecules: List[str], 
                     path_reactions: List[SingleStepResult],
                     is_complete: bool = False
                     ) -> float:
        """评估完整路径，返回单目标得分（标量）"""
        if not is_complete:
            objective_value = self._evaluate_incomplete_path(path_molecules)
        else:
            path_metrics = self._evaluate_complete_path(path_molecules, path_reactions)
            objective_value = self._generate_objective_value(path_metrics)   # 改为标量
        return objective_value   # 直接返回 float
    
    def _evaluate_complete_path(self, 
                                path_molecules: List[str], 
                                path_reactions: List[SingleStepResult]) -> PathMetrics:
        self.evaluation_count += 1
        step_metrics = []
        for i, reaction in enumerate(path_reactions):
            if i < len(path_molecules) - 1:
                reactant = path_molecules[i + 1]
                target = path_molecules[i]
                step_metric = self.single_step_evaluator.evaluate_step(reactant, reaction, target)
                step_metrics.append(step_metric)
        
        aggregated_metrics = self.aggregator.aggregate_metrics(step_metrics, self.aggregation_strategies)
        path_metrics = self._calculate_path_metrics(path_molecules, path_reactions, 
                                                   step_metrics, aggregated_metrics)
        return path_metrics
    
    def _evaluate_incomplete_path(self, path_molecules: List[str]) -> float:
        """评估不完整路径，返回惩罚值（标量）"""
        if not path_molecules:
            return self.config.base_penalty_value
        
        last_molecule = path_molecules[-1]
        similarity = self.single_step_evaluator._calculate_similarity_score(last_molecule)
        path_length_factor = np.exp(-self.config.path_length_penalty_factor * len(path_molecules))
        reward_factor = (self.config.similarity_weight * similarity + 
                        (1 - self.config.similarity_weight) * path_length_factor)
        heuristic_penalty = self.config.base_penalty_value * (1 - reward_factor)
        return heuristic_penalty   # 标量
    
    def _calculate_path_metrics(self, 
                               path_molecules: List[str],
                               path_reactions: List[SingleStepResult],
                               step_metrics: List[SingleStepMetrics],
                               aggregated_metrics: Dict[str, float]) -> PathMetrics:
        path_length = len(path_reactions)
        overall_similarity = aggregated_metrics.get('similarity', 0.0)
        overall_feasibility = aggregated_metrics.get('feasibility', 0.0)
        overall_retro_score = aggregated_metrics.get('retro_score', 0.0)
        overall_yield = aggregated_metrics.get('yield', 0.0)
        overall_toxicity = aggregated_metrics.get('toxicity', 0.0)
        
        quality_score = (overall_similarity + overall_feasibility + overall_retro_score) / 3.0
        efficiency_score = quality_score / max(1, path_length)
        overall_thermodynamics = self._calculate_thermodynamics(path_reactions)

        return PathMetrics(
            path_length=path_length,
            overall_similarity=overall_similarity,
            overall_feasibility=overall_feasibility,
            overall_retro_score=overall_retro_score,
            overall_thermodynamics=overall_thermodynamics,
            overall_yield=overall_yield,
            efficiency_score=efficiency_score,
            overall_toxicity=overall_toxicity,
            step_metrics=step_metrics
        )
    
    def _calculate_thermodynamics(self, path_reactions):
        try:
            gibbs = - random.random()
            return gibbs
        except:
            return 0.0
    
    def _generate_objective_value(self, path_metrics: PathMetrics) -> float:
        """
        生成单目标得分（加权和）
        可根据需要调整各指标的权重
        """
        # 定义各目标权重（可根据实验调整）
        weights = {
            "overall_retro_score": 0.3,
            "overall_feasibility": 0.3,
            "overall_similarity": 0.2,
            "overall_yield": 0.2,
        }
        score = (
            weights["overall_retro_score"] * path_metrics.overall_retro_score +
            weights["overall_feasibility"] * path_metrics.overall_feasibility +
            weights["overall_similarity"] * path_metrics.overall_similarity +
            weights["overall_yield"] * path_metrics.overall_yield
        )
        return score   # 标量
    
    # ========================================================================
    # 以下多目标相关方法已废弃，但为兼容性保留空实现或移除
    # ========================================================================
    
    def get_pareto_front(self) -> List[float]:
        """已废弃：单目标模式下无帕累托前沿"""
        return []
    
    def calculate_hypervolume(self, solutions: List[float]) -> float:
        """已废弃：返回0.0"""
        return 0.0
    
    def get_evaluation_statistics(self) -> Dict:
        """获取评估统计信息（移除多目标统计）"""
        cache_hit_rate = self.cache_hits / max(1, self.evaluation_count)
        return {
            'total_evaluations': self.evaluation_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self._evaluation_cache),
            # 不再有 pareto_front_size 和 hypervolume
        }
    
    def set_aggregation_strategy(self, metric: str, strategy: AggregationStrategy):
        self.aggregation_strategies[metric] = strategy


# ========================================================================
# 工厂函数（适配单目标）
# ========================================================================

def create_standard_evaluator(commercial_molecules: List[str], 
                             config: Optional[EvaluationConfig] = None) -> PathEvaluator:
    if config is None:
        config = EvaluationConfig()
        # 单目标模式下 enabled_objectives 不再需要，但保留配置以防其他地方使用
        config.enabled_objectives = []   # 清空
    evaluator = PathEvaluator(commercial_molecules, config)
    evaluator.set_aggregation_strategy('similarity', AggregationStrategy.MAX)
    evaluator.set_aggregation_strategy('feasibility', AggregationStrategy.MIN)
    evaluator.set_aggregation_strategy('yield', AggregationStrategy.GEOMETRIC_MEAN)
    evaluator.set_aggregation_strategy('retro_score', AggregationStrategy.MEAN)
    return evaluator


def test_evaluator():
    """测试单目标评估器"""
    from config.config import SingleStepResult, EvaluationConfig
    
    commercial_mols = ["CCO", "CC", "CO"]
    config = EvaluationConfig()
    evaluator = create_standard_evaluator(commercial_mols, config)
    
    test_molecules = ["CCCCCO", "CCCO", "CCO"]
    test_reactions = [
        SingleStepResult(reactants=["CCCO"], scores=0.8, templates="template1", ec="1.1.1.1"),
        SingleStepResult(reactants=["CCO"], scores=0.9, templates="template2", ec="2.1.1.1")
    ]
    
    result = evaluator.evaluate_path(test_molecules, test_reactions, is_complete=True)
    print("测试结果（单目标得分）:", result)
    print("评估统计:", evaluator.get_evaluation_statistics())
    return evaluator


if __name__ == "__main__":
    test_evaluator()