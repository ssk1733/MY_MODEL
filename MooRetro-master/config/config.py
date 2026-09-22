import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum 
from datetime import datetime
from utils.chem import canonicalize_mol_smi

# @dataclass
# class Example:
#     """D-xylose degradation to ethylene glycol (engineered) - PWY-7178"""
#     project_name: str = f'Biocyc_PWY_7178_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
#     target_molecule: str = 'O=C[C@H](O)[C@@H](O)[C@H](O)CO'
#     commercial_molecules: List[str] = ['OCCO', 
#                             'O=CCO', 
#                             # 'O=C([O-])C(=O)C[C@H](O)CO', 
#                             # 'O=C([O-])[C@H](O)[C@@H](O)[C@H](O)CO'
#                             ]
#     max_depth: int = 6
#     single_fn_type: str = 'biocyc'

@dataclass 
class MCTSConfig:
    """MCTS配置参数"""

    project_name: str = 'PWY-43'
    target_molecule: str = '[NH3+]CCCC[NH3+]'
    commercial_molecules: List[str] = field(default_factory=lambda: [
        'NC(=[NH2+])NCCC[C@H]([NH3+])C(=O)[O-]'
    ])

    random_seed: int = 100
    c_param: float = 1.4
    max_iterations: int = 10
    max_simulation_depth: int = 10
    early_stopping_threshold: float = 0.01
    real_time_feedback_interval: int = 1
    enable_progressive_widening: bool = True
    progressive_widening_alpha: float = 0.5
    epsilon_greedy: float = 0.2
    heuristic_similarity_param: float = 0.6
    single_fn_type: str = 'biocyc'
    result_path: Optional[str] = None

    hypervolume_improvement_threshold: float = 0.001
    pareto_front_update_frequency: int = 10
    min_iterations_before_early_stop: int = 100
    max_iterations_without_improvement: int = 200

    def __post_init__(self):
        self.commercial_molecules = [smi for smi in self.commercial_molecules if smi]
        if self.result_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.result_path = f'results/{self.project_name}_{self.single_fn_type}_{timestamp}'

        # target_molecule = canonicalize_mol_smi(target_molecule)
        # commercial_molecules = [canonicalize_mol_smi(smi) for smi in commercial_molecules]



@dataclass
class SingleStepResult:
    """单步逆合成结果"""
    reactants: List[str]
    ec: str
    templates: str
    scores: float
    top_k: str = ""

    # reaction_type: str = ""

    similarity: Optional[float] = None
    feasibility: Optional[float] = None # delta_sacore_for_feasibility
    yield_estimate: Optional[float] = None  # 新增：理论产量估计
    toxicity: Optional[float] = None
    # Preserve the full template-derived precursor set for external AND-OR
    # planners while keeping ``reactants`` as MooRetro's legacy main branch.
    all_reactants: Optional[List[str]] = None
    cofactors: Optional[List[str]] = None

    # confidence: float = 0.0

    @property
    def to_dict(self):
        return {
            'ec': self.ec,
            'reactants': self.reactants[0],
            'scores': self.scores,
            'templates': self.templates,
            'topk': self.top_k,
            # 'reaction_type': self.reaction_type,
            'similarity': self.similarity,
            'feasibility': self.feasibility,
            'yield': self.yield_estimate,
            'toxicity': self.toxicity,
            'all_reactants': self.all_reactants or list(self.reactants),
            'cofactors': self.cofactors or [],
        }


@dataclass
class EvaluationConfig:
    """评估配置参数"""
    temperature: float = 298.15
    pressure: float = 1.0
    max_molecular_weight: float = 1000.0
    max_ring_count: int = 5
    max_path_length: int = 10
    max_pareto_front_size: int = 50
    enable_cache: bool = True
    cache_size: int = 10000

    # 不完整路径的启发式评估参数
    base_penalty_value: float = 10.0
    path_length_penalty_factor: float = 0.1
    similarity_weight: float = 0.8

    # 多目标优化配置
    enabled_objectives: List[str] = field(default_factory=lambda: [
        # "path_efficiency",        # 路径效率（质量/长度）
        "path_length_inv",
        "overall_retro_score",
        # "overall_similarity",     # 整体相似性
        "overall_feasibility",    # 整体可行性
        # "overall_yield"          # 整体产量
    ])
    
    # 超体积参考点
    # hypervolume_reference_point: np.ndarray = field(default_factory=lambda: np.array([0.0] * 4))
    hypervolume_reference_point: List = field(default_factory=lambda: [0.0] * 3)
    # 新增：聚合策略配置
    similarity_aggregation: str = "max"      # 相似性聚合策略：max, mean, min
    feasibility_aggregation: str = "min"     # 可行性聚合策略：min, mean, harmonic
    yield_aggregation: str = "geometric"     # 产量聚合策略：geometric, mean, min
    score_aggregation: str = "mean"          # 分数聚合策略：mean, min, weighted
    
    # 新增：指标权重配置
    similarity_weight_in_efficiency: float = 0.3
    feasibility_weight_in_efficiency: float = 0.3
    yield_weight_in_efficiency: float = 0.2
    score_weight_in_efficiency: float = 0.2
    
    # 新增：产量计算配置
    default_yield_estimate: float = 0.75    # 默认产量估计
    yield_confidence_factor: float = 0.9    # 产量置信度因子
    min_step_yield: float = 0.1             # 最小单步产量
    max_step_yield: float = 1.0             # 最大单步产量
    
    def get_hypervolume_reference_point(self) -> np.ndarray:
        """获取超体积参考点 - 根据启用的指标数量调整"""
        n_objectives = len(self.enabled_objectives)
        if len(self.hypervolume_reference_point) != n_objectives:
            # 自动调整参考点维度
            return np.array([0.0] * n_objectives)
        return self.hypervolume_reference_point
    
    def get_objective_weights(self) -> Dict[str, float]:
        """获取目标权重配置"""
        return {
            'similarity': self.similarity_weight_in_efficiency,
            'feasibility': self.feasibility_weight_in_efficiency,
            'yield': self.yield_weight_in_efficiency,
            'score': self.score_weight_in_efficiency
        }
    
    def validate_configuration(self) -> bool:
        """验证配置参数的有效性"""
        # 检查权重是否归一化
        weights = list(self.get_objective_weights().values())
        if abs(sum(weights) - 1.0) > 1e-6:
            return False
        
        # 检查产量范围
        if not (0 <= self.min_step_yield <= self.max_step_yield <= 1.0):
            return False
        
        # 检查启用的目标是否有效
        valid_objectives = {
            "path_efficiency", "overall_similarity", "overall_feasibility", 
            "overall_yield", "retro_score", "path_length_inv"
        }
        if not all(obj in valid_objectives for obj in self.enabled_objectives):
            return False
        
        return True


class SelectionStrategy(Enum):
    """选择策略枚举"""
    STANDARD_UCB = "standard_ucb"
    SIMILARITY_GUIDED = "similarity_guided" 
    MULTI_CRITERIA = "multi_criteria"
    ADAPTIVE = "adaptive"
    HYPERVOLUME_BASED = "hypervolume_based"  # 新增：基于超体积的选择


@dataclass
class SelectionConfig:
    """选择策略配置"""
    strategy: SelectionStrategy = SelectionStrategy.SIMILARITY_GUIDED
    similarity_weight: float = 0.3  # 相似度权重
    ucb_weight: float = 0.7  # UCB权重
    adaptive_threshold: int = 50  # 自适应切换阈值
    diversity_bonus: float = 0.1  # 多样性奖励
    
    # 新增：多目标选择参数
    hypervolume_weight: float = 0.4         # 超体积权重
    pareto_rank_weight: float = 0.3         # Pareto排名权重
    crowding_distance_weight: float = 0.3   # 拥挤距离权重


# 新增：评估器工厂配置
@dataclass
class EvaluatorFactoryConfig:
    """评估器工厂配置"""
    evaluator_type: str = "standard"  # standard, advanced, custom
    
    # 标准评估器配置
    use_thermodynamics: bool = True
    use_molecular_descriptors: bool = True
    use_similarity_cache: bool = True
    
    # 高级评估器配置
    enable_reaction_prediction: bool = False
    enable_synthesis_planning: bool = False
    enable_cost_estimation: bool = False
    
    # 自定义评估器配置
    custom_objective_functions: Optional[List[Callable]] = None
    custom_aggregation_functions: Optional[Dict[str, Callable]] = None


# 预定义配置模板
class ConfigTemplates:
    """预定义的配置模板"""
    
    @staticmethod
    def get_efficiency_focused_config() -> EvaluationConfig:
        """效率优先的配置"""
        config = EvaluationConfig()
        config.enabled_objectives = ["path_efficiency", "overall_yield"]
        config.similarity_weight_in_efficiency = 0.2
        config.feasibility_weight_in_efficiency = 0.2
        config.yield_weight_in_efficiency = 0.4
        config.score_weight_in_efficiency = 0.2
        config.yield_aggregation = "geometric"
        config.feasibility_aggregation = "min"
        return config
    
    @staticmethod
    def get_quality_focused_config() -> EvaluationConfig:
        """质量优先的配置"""
        config = EvaluationConfig()
        config.enabled_objectives = ["overall_similarity", "overall_feasibility", "overall_yield"]
        config.similarity_weight_in_efficiency = 0.4
        config.feasibility_weight_in_efficiency = 0.4
        config.yield_weight_in_efficiency = 0.15
        config.score_weight_in_efficiency = 0.05
        config.similarity_aggregation = "max"
        config.feasibility_aggregation = "min"
        return config
    
    @staticmethod
    def get_balanced_config() -> EvaluationConfig:
        """平衡配置"""
        config = EvaluationConfig()
        config.enabled_objectives = [
            "path_efficiency", "overall_similarity", 
            "overall_feasibility", "overall_yield"
        ]
        # 使用默认权重（已经是平衡的）
        return config
    
    @staticmethod
    def get_exploration_mcts_config() -> MCTSConfig:
        """探索优先的MCTS配置"""
        config = MCTSConfig()
        config.c_param = 2.0  # 更高的探索参数
        config.max_iterations = 1000
        config.early_stopping_threshold = 0.005
        config.min_iterations_before_early_stop = 200
        return config
    
    @staticmethod
    def get_exploitation_mcts_config() -> MCTSConfig:
        """利用优先的MCTS配置"""
        config = MCTSConfig()
        config.c_param = 0.5  # 更低的探索参数
        config.max_iterations = 500
        config.early_stopping_threshold = 0.02
        config.min_iterations_before_early_stop = 50
        return config


# 配置验证和工具函数
def validate_all_configs(eval_config: EvaluationConfig, 
                        mcts_config: MCTSConfig, 
                        selection_config: SelectionConfig) -> Tuple[bool, List[str]]:
    """验证所有配置的兼容性"""
    errors = []
    
    # 验证评估配置
    if not eval_config.validate_configuration():
        errors.append("EvaluationConfig validation failed")
    
    # 验证MCTS配置
    if mcts_config.max_iterations <= 0:
        errors.append("MCTS max_iterations must be positive")
    
    if mcts_config.c_param <= 0:
        errors.append("MCTS c_param must be positive")
    
    # 验证选择配置
    total_weight = (selection_config.hypervolume_weight + 
                   selection_config.pareto_rank_weight + 
                   selection_config.crowding_distance_weight)
    if abs(total_weight - 1.0) > 1e-6:
        errors.append("Selection weights must sum to 1.0")
    
    # 验证目标数量与参考点维度的一致性
    if len(eval_config.enabled_objectives) != len(eval_config.get_hypervolume_reference_point()):
        errors.append("Mismatch between enabled objectives and hypervolume reference point dimensions")
    
    return len(errors) == 0, errors


def create_compatible_configs(preset: str = "balanced") -> Tuple[EvaluationConfig, MCTSConfig, SelectionConfig]:
    """创建兼容的配置组合"""
    if preset == "efficiency":
        eval_config = ConfigTemplates.get_efficiency_focused_config()
        mcts_config = ConfigTemplates.get_exploitation_mcts_config()
    elif preset == "quality":
        eval_config = ConfigTemplates.get_quality_focused_config()
        mcts_config = ConfigTemplates.get_exploration_mcts_config()
    else:  # balanced
        eval_config = ConfigTemplates.get_balanced_config()
        mcts_config = MCTSConfig()
    
    selection_config = SelectionConfig()
    selection_config.strategy = SelectionStrategy.HYPERVOLUME_BASED
    
    return eval_config, mcts_config, selection_config


# 使用示例
if __name__ == "__main__":
    # 创建并验证配置
    eval_config, mcts_config, selection_config = create_compatible_configs("balanced")
    
    is_valid, errors = validate_all_configs(eval_config, mcts_config, selection_config)
    
    if is_valid:
        print("配置验证通过")
        print(f"启用的目标: {eval_config.enabled_objectives}")
        print(f"目标权重: {eval_config.get_objective_weights()}")
    else:
        print("配置验证失败:")
        for error in errors:
            print(f"  - {error}")