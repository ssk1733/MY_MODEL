import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
import time
import graphviz
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from ParetoMCTS.config import SingleStepResult

class Node(ABC):
    """抽象节点基类"""
    def __init__(self, molecule: str, parent=None, depth: int = 0, enabled_objectives: Optional[List[str]] = None):
        self.molecule = molecule
        self.parent = parent
        self.children: List['Node'] = []
        self.visits = 0
        self.depth = depth
        # 使用配置的指标初始化节点的Pareto前沿
        self.solutions: List[np.ndarray] = []  # 存储目标向量
        # self.pareto_front = ParetoFront(enabled_objectives=enabled_objectives)
        self.is_expanded = False
        
    @abstractmethod
    def is_terminal(self) -> bool:
        pass
    
    def add_child(self, child: 'Node'):
        child.parent = self
        self.children.append(child)

    def add_solution(self, solution: np.ndarray):
        self.solutions.append(solution)
    
    def get_pareto_front(self) -> List[np.ndarray]:
        """使用pymoo获取节点的Pareto前沿"""
        if not self.solutions:
            return []
            
        F = np.array(self.solutions)
        nds = NonDominatedSorting()
        front_indices = nds.do(F, only_non_dominated_front=True)
        return F[front_indices].tolist()

class MoleculeNode(Node):
    """OR节点：表示分子"""
    def __init__(self, molecule: str, parent=None, depth: int = 0, is_commercial: bool = False, 
                 enabled_objectives: Optional[List[str]] = None):
        super().__init__(molecule, parent, depth, enabled_objectives)
        self.is_commercial = is_commercial
        
    def is_terminal(self) -> bool:
        return self.is_commercial or self.depth >= 10
    
    @property
    def info(self) -> Dict:
        return {
            'type': 'molecule',
            'smiles': self.molecule,
            'molecule': self.molecule,
            'visits': self.visits,
            'is_commercial': self.is_commercial,
            'is_terminal': self.is_terminal(),
            'solutions_count': len(self.solutions),
            'depth': self.depth,
        }

class ReactionNode(Node):
    """AND节点：表示反应"""
    def __init__(self, molecule: str, parent=None, depth: int = 0, 
                 reaction_info: Optional[SingleStepResult] = None,
                 enabled_objectives: Optional[List[str]] = None):
        super().__init__(molecule, parent, depth, enabled_objectives)
        self.reaction_info = reaction_info
        
    def is_terminal(self) -> bool:
        if not self.children:
            return False
        return all(child.is_terminal() for child in self.children)
    
    @property
    def info(self) -> Dict:
        return {
                'type': 'reaction',
                'visits': self.visits,
                'solutions_count': len(self.solutions),
                'depth': self.depth,
                'reaction_score': float(self.reaction_info.scores) if self.reaction_info else 0.0,
                'templates': self.reaction_info.templates if self.reaction_info else '',
                'ec': self.reaction_info.ec if self.reaction_info else '',
                'reactants': self.reaction_info.reactants if self.reaction_info else []
                }
