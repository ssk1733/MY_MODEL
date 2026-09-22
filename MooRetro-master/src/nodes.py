import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
import random 

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from config.config import SingleStepResult

# ============================================================================
# 节点层 (Nodes Layer) - 基础数据结构，专注于节点本身的属性和行为
# ============================================================================

class Node(ABC):
    """基础节点类 - 只包含节点自身的核心属性和方法"""
    
    def __init__(self, node_id: str, depth: int = 0):
        self.id = node_id
        self.depth = depth
        self.parent = None
        self.children = []
        
        # MCTS相关数据（节点级别）
        self.visits = 0
        self.solutions = []  # 存储通过此节点的所有目标向量
        self.complete_solutions = [] 

    
    def add_child(self, child: 'Node'):
        """添加子节点"""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    def remove_child(self, child: 'Node'):
        """移除子节点"""
        if child in self.children:
            self.children.remove(child)
            child.parent = None
    
    def is_leaf(self) -> bool:
        """检查是否为叶子节点"""
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        """检查是否为根节点"""
        return self.parent is None
    
    
    def get_ucb_value(self, parent_visits: int, c_param: float = 1.4) -> float:
        """计算UCB值（节点级别的选择依据）"""
        if self.visits == 0:
            return float('inf')  # 未访问的节点优先级最高
        
        # Exploitation: 使用最近的平均性能
        if not self.solutions:
            exploitation = 0.0
        else:
            # 使用最近10个解的平均值，避免早期差解的影响
            recent_solutions = self.solutions[-10:]
            exploitation = np.mean([np.sum(sol) for sol in recent_solutions])
        
        # Exploration: 标准UCB公式
        exploration = c_param * math.sqrt(math.log(max(parent_visits,1e-6)) / self.visits)
        
        return exploitation + exploration


    
    @abstractmethod
    def is_terminal(self) -> bool:
        """检查是否为终端节点"""
        pass
    
    def __str__(self):
        return f"{self.__class__.__name__}({self.id}, visits={self.visits}, depth={self.depth}, children={len(self.children)})"
    
    def __repr__(self):
        return self.__str__()
    
    @property
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'depth': self.depth,
            'visits': self.visits,
            'is_root': self.is_root(),
            'is_leaf': self.is_leaf(),
            'solutions_count': len(self.solutions),
        }


class MoleculeNode(Node):
    """分子节点 (OR节点) - 表示一个分子及其可能的合成路径"""
    
    def __init__(self, molecule: str, is_commercial: bool = False, 
                 similarity_to_commercial: float = 0.0, 
                 node_id = None,
                 **kwargs):
        super().__init__(node_id or f"mol_{id(self)}", **kwargs)
        self.molecule = molecule
        self.is_commercial = is_commercial
        self.similarity_to_commercial = similarity_to_commercial  # 与商业分子的最高相似性

    def is_terminal(self) -> bool:
        """检查是否为终端节点"""
        # 商业分子或者已经扩展但没有可行反应的分子
        return self.is_commercial # or (not self.is_leaf() and not self.get_available_reactions())
    
    def is_expandable(self) -> bool:
        """检查是否可以扩展"""
        return not self.is_commercial and self.is_leaf()
    
    def calculate_toxicity(self) -> float:
        toxicity = random.random()
        return round(toxicity, 3)
    
    def __str__(self):
        commercial_str = " (commercial)" if self.is_commercial else ""
        similarity_str = f" (sim={self.similarity_to_commercial:.3f})" if self.similarity_to_commercial > 0 else ""
        return f"MoleculeNode({self.molecule}{commercial_str}{similarity_str}, visits={self.visits}, depth={self.depth})"
    
    @property
    def to_dict(self) -> Dict:
        base_info = super().to_dict
        base_info.update({
            'type': 'molecule',
            'molecule': self.molecule,
            'is_commercial': self.is_commercial,
            'similarity': f'{self.similarity_to_commercial:.2f}',
            'toxicity': self.calculate_toxicity(),
            'is_terminal': self.is_terminal()
        })
        return base_info
    

class ReactionNode(Node):
    """反应节点 (AND节点) - 表示一个具体的反应及其反应物"""
    
    def __init__(self, 
                 reaction_info: SingleStepResult, 
                 parent_molecule: str, 
                 node_id = None,
                 **kwargs):
        if node_id:
            super().__init__(node_id or f"rxn_{id(self)}", **kwargs)
        else:
            super().__init__(f"rxn_{len(reaction_info.reactants)}_{reaction_info.templates}", **kwargs)
        self.reaction_info = reaction_info
        self.parent_molecule = parent_molecule
        
        # 反应特定属性
        self.feasibility_score = 0.0    # 反应可行性分数
        self.yield_estimate = 0.0       # 产量估计

    def is_terminal(self) -> bool:
        """检查反应是否完整（所有反应物都能获得）"""
        if not self.children:
            return False
        
        # AND节点：有一个子节点是商业分子即可
        return any(child.is_terminal() for child in self.children)
    
    
    def __str__(self):
        return f"ReactionNode({self.reaction_info.templates}, visits={self.visits}, depth={self.depth})"

    @property
    def to_dict(self) -> Dict:
        base_info = super().to_dict
        base_info.update(self.reaction_info.to_dict)
        base_info.update({
            'is_terminal': self.is_terminal()
        })
        return base_info

# ============================================================================
# 节点工具函数 - 简单的节点级别操作
# ============================================================================

def get_node_depth(node: Node) -> int:
    """获取节点深度"""
    return node.depth

def get_path_to_root(node: Node) -> List[Node]:
    """获取从节点到根的路径"""
    path = []
    current = node
    while current:
        path.append(current)
        current = current.parent
    return list(reversed(path))

def find_common_ancestor(node1: Node, node2: Node) -> Optional[Node]:
    """找到两个节点的共同祖先"""
    path1 = set(get_path_to_root(node1))
    path2 = get_path_to_root(node2)
    
    for node in path2:
        if node in path1:
            return node
    return None

