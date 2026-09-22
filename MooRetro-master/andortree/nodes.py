from typing import List, Dict, Set, Tuple, Optional, Callable

class MoleculeNode:
    """OR节点：表示分子"""
    __slots__ = ('smiles', 'depth', 'cost', 'expanded', 
                 'max_similarity', 'reactions', 'parent_reactions',
                 'buyable', 'solved') 
    
    def __init__(self, smiles: str):
        self.smiles = smiles          # 分子SMILES
        self.depth = 0                # 在树中的深度
        self.cost = float('inf')      # 当前最小代价估计
        self.max_similarity = 0.0     # 与buyable分子的最大相似度

        self.reactions: List['ReactionNode'] = []  # 可能的分解反应
        self.parent_reactions: List['ReactionNode'] = []  # 父反应节点
        
        self.solved = False           # 是否已解决
        self.buyable = False          # 是否可购买
        self.expanded = False         # 是否已扩展
        

class ReactionNode:
    """AND节点：表示化学反应"""
    __slots__ = ('id', 'score', 'edits', 'ec', 'parent', 'substrate', 'solved')
    
    def __init__(self, rxn_id: str, score: float, edits = None, ec = None):
        self.id = rxn_id              # 反应ID
        self.score = score            # 反应分数
        self.edits = edits
        self.ec = ec

        self.parent: Optional[MoleculeNode] = None  # 父分子节点
        self.substrate: Optional[MoleculeNode] = None  # 底物分子节点（只有一个）
        
        self.solved = False           # 是否已解决
        
