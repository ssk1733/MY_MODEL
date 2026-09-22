import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum 


@dataclass
class SingleStepResult:
    """单步逆合成结果"""
    reactants: List[str]
    scores: float
    templates: str
    ec: str
    confidence: float = 0.0
    reaction_type: str = ""


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
    # 添加缺失的字段
    enabled_objectives: List[str] = field(default_factory=lambda: [
        "path_efficiency", "yield_score", "average_step_score"
    ])
    hypervolume_reference_point: np.ndarray = field(default_factory=lambda: np.array([0.0] * 3))
    
    def get_hypervolume_reference_point(self) -> np.ndarray:
        """获取超体积参考点 - 根据启用的指标数量调整"""
        n_objectives = len(self.enabled_objectives)
        if len(self.hypervolume_reference_point) != n_objectives:
            # 自动调整参考点维度
            return np.array([0.0] * n_objectives)
        return self.hypervolume_reference_point


@dataclass 
class MCTSConfig:
    """MCTS配置参数"""
    c_param: float = 1.4  # UCB探索参数
    max_iterations: int = 10
    max_simulation_depth: int = 20
    early_stopping_threshold: float = 0.95  # 超体积改进阈值
    real_time_feedback_interval: int = 1  # 实时反馈间隔
    enable_progressive_widening: bool = True
    progressive_widening_alpha: float = 0.5
    result_path: str = 'results/'

class SelectionStrategy(Enum):
    """选择策略枚举"""
    STANDARD_UCB = "standard_ucb"
    SIMILARITY_GUIDED = "similarity_guided" 
    MULTI_CRITERIA = "multi_criteria"
    ADAPTIVE = "adaptive"

@dataclass
class SelectionConfig:
    """选择策略配置"""
    strategy: SelectionStrategy = SelectionStrategy.SIMILARITY_GUIDED
    similarity_weight: float = 0.3  # 相似度权重
    ucb_weight: float = 0.7  # UCB权重
    adaptive_threshold: int = 50  # 自适应切换阈值
    diversity_bonus: float = 0.1  # 多样性奖励


