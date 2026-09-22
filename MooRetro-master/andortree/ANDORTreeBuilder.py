from collections import deque
import heapq
import math
import torch
import random
import logging
import os
import time
import copy
from datetime import datetime
from typing import List, Dict, Set, Tuple, Optional, Callable
import pdb 
import networkx as nx
import matplotlib.pyplot as plt
from graphviz import Digraph
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  

from .nodes import MoleculeNode, ReactionNode
from .ANDORTreePruner import ANDORTreePruner
from utils.chem import similarity_calculator

class ANDORTreeBuilder:
    def __init__(self, target_smiles: str, 
                 buyable_molecules: List[str],
                 retro_fn,
                 max_depth: int = 6,
                 max_branches: int = 5,
                 max_nodes: int = 1000,
                 strategy: str = "best_first",
                 log_progress: bool = True):
        """
        参数:
            target_smiles: 目标分子SMILES
            buyable_molecules: 可购买分子集合
            retro_fn: 单步逆合成函数
            max_depth: 最大搜索深度
            max_branches: 每层最大分支数
            max_nodes: 最大扩展节点数
            strategy: 节点选择策略 (best_first, depth_first, breadth_first, random, hybrid)
            log_progress: 是否记录进度
        """
        self.target = target_smiles
        self.buyable = buyable_molecules
        self.retro_fn = retro_fn
        self.max_depth = max_depth
        self.max_branches = max_branches
        self.max_nodes = max_nodes
        self.strategy = strategy
        self.log_progress = log_progress
        
        # 节点存储
        self.mol_nodes: Dict[str, MoleculeNode] = {}
        self.rxn_nodes: Dict[str, ReactionNode] = {}
        self.root: Optional[MoleculeNode] = None
        
        # 优先队列 (priority, depth, smiles, node)
        self.queue = []
        
        # 统计
        self.nodes_expanded = 0
        self.start_time = time.time()
        
        # 日志记录
        self.logger = logger
        self.logger.info(f"初始化AND-OR树构建器: 目标分子={target_smiles}")
        self.logger.info(f"配置: 最大深度={max_depth}, 最大分支={max_branches}, 最大节点={max_nodes}, 策略={strategy}")
        self.logger.info(f"可购买分子数: {len(buyable_molecules)}")

    def is_buyable(self, smiles) -> bool:
        if smiles in self.buyable:
            return True
        
        similarity = max(similarity_calculator(smiles, self.buyable), default=0)
        return similarity >= 0.99
    

    # def get_or_create_molecule_node(self, smiles: str) -> MoleculeNode:
    #     """获取或创建分子节点"""
    #     if smiles not in self.mol_nodes:
    #         node = MoleculeNode(smiles)
    #         self.mol_nodes[smiles] = node
            
    #         # 计算相似度
    #         node.max_similarity = max(similarity_calculator(smiles, self.buyable), default=0)
            
    #         # 如果是可购买分子，设置代价为0
    #         if self.is_buyable(smiles):
    #             node.cost = 0
    #     return self.mol_nodes[smiles]


    def get_or_create_molecule_node(self, smiles: str) -> MoleculeNode:
        """获取或创建分子节点"""
        if smiles not in self.mol_nodes:
            node = MoleculeNode(smiles)
            self.mol_nodes[smiles] = node
            
            # 计算并存储相似度
            node.max_similarity = max(similarity_calculator(smiles, self.buyable), default=0)
            
            # 判断是否可购买
            node.buyable = self.is_buyable(smiles)
            
            # 如果是可购买分子，设置代价为0并标记为已解决
            if node.buyable:
                node.cost = 0
                node.solved = True
                # pdb.set_trace()
                self.propagate_solution_status(node)

        return self.mol_nodes[smiles]
    
    def get_or_create_reaction_node(self, rxn_id:str, score:float, edits:str, ec:str) -> ReactionNode:
        if rxn_id not in self.rxn_nodes:
            rxn_node = ReactionNode(rxn_id, score, edits, ec)
            self.rxn_nodes[rxn_id] = rxn_node

            # 处理底物后立即检查反应节点状态 #TODO: 
            if rxn_node.substrate and rxn_node.substrate.solved:
                rxn_node.solved = True
                # 立即传播状态变化
                self.propagate_solution_status(rxn_node)

        return self.rxn_nodes[rxn_id]
    
    def calculate_heuristic_cost(self, mol_node: MoleculeNode) -> float:
        """计算分子节点的启发式代价"""
        # 如果是可购买分子，代价为0
        if self.is_buyable(mol_node.smiles):
            return 0.0
        
        # 否则使用 (1 - 最大相似度) 作为代价
        return 1.0 - mol_node.max_similarity
    
    def calculate_node_priority(self, node: MoleculeNode) -> float:
        """根据策略计算节点优先级"""
        if self.strategy == "best_first":
            # 最佳优先: 代价最小优先
            return node.cost
            
        elif self.strategy == "depth_first":
            # 深度优先: 深度最大优先 (使用负号转换为最小堆)
            return -node.depth
            
        elif self.strategy == "breadth_first":
            # 广度优先: 深度最小优先
            return node.depth
            
        elif self.strategy == "random":
            # 随机策略
            return random.random()
            
        elif self.strategy == "hybrid":
            # 混合策略: 90%最佳优先 + 10%随机探索
            return node.cost if random.random() < 0.8 else random.random()
            
        else:
            # 默认使用最佳优先
            return node.cost
    
    def push_node(self, node: MoleculeNode):
        """将节点加入优先队列，以cost为准进行排序，越小的优先级越高"""
        priority = self.calculate_node_priority(node)
        
        # 添加SMILES作为可比较字段，避免节点比较错误
        heapq.heappush(self.queue, (priority, node.depth, node.smiles, node))

    def is_node_solved(self, node: MoleculeNode) -> bool:
        """检查分子节点是否已解决"""
        # 如果是可购买分子，直接解决
        if self.is_buyable(node.smiles):
            return True
        
        # 如果有任何子反应已解决，则节点解决
        return any(rxn.solved for rxn in node.reactions)
    
    def expand_molecule_node(self, mol_node: MoleculeNode):
        """扩展分子节点（包含反应节点创建）"""
        # 1. 检查是否已经扩展，已经扩展则 返回 build tree 继续 heapq.heappop 即可
        if mol_node.expanded or mol_node.depth >= self.max_depth:
            self.logger.debug(f"跳过扩展: {mol_node.smiles} (已扩展或达到最大深度)")
            self.propagate_solution_status(mol_node)
            return
        
        # 2. 如果是可购买分子，不需要扩展
        if mol_node.buyable:
            self.logger.debug(f"可购买分子: {mol_node.smiles}")
            mol_node.solved = True
            self.propagate_solution_status(mol_node)
            return
        
        # 3. 获取逆合成反应
        self.logger.info(f"为分子节点扩展: {mol_node.smiles} (深度={mol_node.depth})")
        reactions = self.retro_fn(target_smile=mol_node.smiles, 
                                  building_blocks=self.buyable, device=device)
        # reactions = [('C[S](CCC(N)C(=O)O)CC1OC(n2cnc3c(N)ncnc32)C(O)C1O', 0.39057794213294983, '[Attaching][Group]*C:14[EC]2.1.1', '2.1.1'), ('Nc1ncnc2c1ncn2C1OC(CO)C(O)C1O', 0.17866885662078857, '[Delete][Bond]CS:13.14[Leaving][Group]*SCCC(N)C(=O)O:14[Attaching][Group]*O:13[EC]3.3.1', '3.3.1'), ('N=C(O)C(N)CCSCC1OC(n2cnc3c(N)ncnc32)C(O)C1O', 0.08408460766077042, '[Change][Bond]CN:3.2[Change][Bond]C=N:2.1[Delete][Bond]C=O:19.20[Leaving][Group]*=O:20[Attaching][Group]*=N:19[EC]3.5.4', '3.5.4'), ('Nc1ncnc2c1ncn2C1OC(CO)C(O)C1O', 0.07839066535234451, '[Delete][Bond]CS:13.14[Leaving][Group]*SCC=C(N)C(=O)O:14[Attaching][Group]*O:13[EC]3.3.1', '3.3.1'), ('CNC(CCSCC1OC(n2cnc3c(N)ncnc32)C(O)C1O)C(=O)O', 0.06567978113889694, '[Change][Bond]CO:19.20[Change][Bond]C=O:19.21[Attaching][Group]*C:18[EC]2.1.1', '2.1.1'), ('C[S](CCC(N)C(=O)O)CC1OC(n2cnc3c(N)ncnc32)C(O)C1O', 0.0605451762676239, '[Attaching][Group]*C:14[EC]2.-.-', '2.-.-'), ('Nc1ncnc2c1ncn2C1OC(CF)C(O)C1O', 0.03992713987827301, '[Delete][Bond]CS:13.14[Leaving][Group]*SCCC(N)C(=O)O:14[Attaching][Group]*F:13[EC]3.2.1', '3.2.1'), ('Nc1ncnc2c1ncn2C1OC(CF)C(O)C1O', 0.036012087017297745, '[Delete][Bond]CS:13.14[Leaving][Group]*SCC=C(N)C(=O)O:14[Attaching][Group]*F:13[EC]3.3.2', '3.3.2'), ('Nc1ncnc2c1ncn2C1OC(CSCCC(N)CO)C(O)C1O', 0.02687237784266472, '[Change][Bond]CN:3.2[Change][Bond]C=N:2.1[Delete][Bond]C=O:19.20[Leaving][Group]*=O:20[EC]1.14.14', '1.14.14')]
        
        if not reactions:
            self.logger.warning(f"未找到逆合成反应: {mol_node.smiles}")
            self.propagate_solution_status(mol_node)  # 添加状态传播
            return
        
        # 4. 标记为已扩展
        mol_node.expanded = True
        self.nodes_expanded += 1

        # 5. 限制分支数并按分数排序
        reactions = sorted(reactions, key=lambda x: x[1], reverse=True)[:self.max_branches]
        self.logger.info(f"找到 {len(reactions)} 个候选反应")
        
        for substrates, score, edits, ec in reactions:
            # 6. 创建/获取反应节点
            rxn_id = f"{mol_node.smiles}_{substrates}_{ec}" 
            self.logger.debug(f"处理反应节点: {rxn_id} (分数={score:.4f})")
            rxn_node = self.get_or_create_reaction_node(rxn_id, score, edits, ec)
            
            # 7. 链接反应节点和分子节点
            rxn_node.parent = mol_node
            mol_node.reactions.append(rxn_node)
            
            # 8. 处理底物分子 （只有一个）
            sub_node = self.get_or_create_molecule_node(substrates)
            sub_node.parent_reactions.append(rxn_node)
            rxn_node.substrate = sub_node  # 直接关联单个底物
            
            # 设置底物深度
            sub_node.depth = mol_node.depth + 1
            
            # 计算底物启发式代价
            if not sub_node.expanded and not sub_node.buyable:
                sub_node.cost = self.calculate_heuristic_cost(sub_node)
                self.logger.debug(f"底物节点启发式代价: {substrates} = {sub_node.cost:.4f} (相似度={sub_node.max_similarity:.4f})")
            
            # 9. 计算反应节点代价（反应代价 + 底物代价和）
            rxn_cost = -math.log(score + 1e-10) + sub_node.cost
            self.logger.debug(f"反应节点总代价: {rxn_id} = {rxn_cost:.4f}")

            # 10. 更新分子节点代价（取最小反应代价）
            if rxn_cost < mol_node.cost:
                mol_node.cost = rxn_cost
                self.logger.debug(f"更新分子节点最小代价: {mol_node.smiles} = {mol_node.cost:.4f}")
            
            # 11. 将未解决的底物加入队列
            if not sub_node.solved and not sub_node.expanded and sub_node.depth < self.max_depth:
                self.push_node(sub_node)

            # 12. 在反应节点构建完成后，尝试从该反应节点传播状态
            self.propagate_solution_status(rxn_node)   # 新增

        # 12. 传播解决状态到祖先
        self.propagate_solution_status(mol_node)

    def propagate_solution_status(self, start_node: MoleculeNode | ReactionNode):
        """自底向上传播解决状态"""
        # 需要更新的节点队列（可以是分子节点或反应节点）
        update_queue = deque([start_node])
        
        while update_queue:
            node = update_queue.popleft()
            changed = False  # 状态是否改变
            
            # 处理分子节点
            if isinstance(node, MoleculeNode):
                # 记录之前的状态
                old_solved = node.solved
                
                # 检查是否解决：可购买或有一个已解决的子反应
                new_solved = node.buyable or any(
                    rxn.solved for rxn in node.reactions
                )
                
                # 检测状态变化
                if new_solved != old_solved:
                    node.solved = new_solved
                    changed = True
                    self.logger.debug(f"分子节点状态改变: {node.smiles} -> {'解决' if new_solved else '未解决'}")
                    
                    # 将父反应节点加入队列
                    for parent_rxn in node.parent_reactions:
                        update_queue.append(parent_rxn)
            
            # 处理反应节点
            elif isinstance(node, ReactionNode):
                # 记录之前的状态
                old_solved = node.solved
                
                # 检查是否解决：所有底物分子都解决
                new_solved = node.substrate.solved if node.substrate else False
                
                # 检测状态变化
                if new_solved != old_solved:
                    node.solved = new_solved  # 更新节点状态
                    changed = True
                    # 将父分子节点加入队列
                    if node.parent:
                        update_queue.append(node.parent)
            
            # 如果状态改变且节点是根节点，记录信息
            if changed and node == self.root:
                self.logger.info(f"根节点解决状态更新: {'已解决' if new_solved else '未解决'}")

    def build_tree(self) -> MoleculeNode:
        """构建AND-OR搜索树"""
        self.logger.info("开始构建AND-OR树...")
        
        # 1. 初始化根节点
        self.root = self.get_or_create_molecule_node(self.target)
        self.root.cost = self.calculate_heuristic_cost(self.root)
        self.push_node(self.root)
        self.logger.info(f"根节点初始化: {self.target} (代价={self.root.cost:.4f}, 相似度={self.root.max_similarity:.4f})")
        
        # 2. 主循环
        while self.queue and self.nodes_expanded < self.max_nodes:
            # 获取优先级最高的节点
            _, _, _, current_node = heapq.heappop(self.queue)
            
            # 跳过已解决或已扩展的节点
            if current_node.solved or current_node.expanded:
                continue
                
            # 记录进度
            elapsed = time.time() - self.start_time
            self.logger.info(f"扩展节点 {self.nodes_expanded+1}/{self.max_nodes}: {current_node.smiles} "
                            f"(深度={current_node.depth}, 代价={current_node.cost:.4f}, 队列大小={len(self.queue)}, 耗时={elapsed:.2f}s)")
            
            # 扩展节点
            self.expand_molecule_node(current_node)
            
            # 如果根节点解决则提前终止
            # if self.is_node_solved(self.root):  # 现在可以正确检测到
            #     self.logger.info("根节点已解决，提前终止搜索")
            #     break
            # if self.root.solved:
            #     self.logger.info("根节点已解决，提前终止搜索")
            #     break
        
        # 记录最终状态
        elapsed = time.time() - self.start_time
        self.logger.info(f"树构建完成。扩展节点数: {self.nodes_expanded}, 总耗时: {elapsed:.2f}秒")
        self.logger.info(f"根节点解决状态: {'已解决' if self.root.solved else '未解决'}")
        self.logger.info(f"根节点最小代价: {self.root.cost:.4f}")
        
        return self.root
    
    def get_solution_paths(self):
        """获取所有解决路径（迭代实现，避免递归深度限制）"""
        if not self.root.solved:
            self.logger.warning("未找到完整解决路径")
            return []
        
        solutions = []
        #  栈元素：(当前节点, 当前路径)
        stack = [(self.root, [])]

        while stack:
            current_node, current_path = stack.pop()
            
            # 添加分子节点到路径
            new_path = current_path + [(current_node.smiles, "Molecule")]
            
            # 如果是可购买分子，记录完整路径
            if current_node.buyable:
                solutions.append(new_path)
                continue
            
            # 遍历已解决的反应
            for rxn in current_node.reactions:
                if rxn.solved and rxn.substrate:
                    # 添加反应节点到路径
                    rxn_path = new_path + [(rxn.id, f"Reaction: {rxn.id}")]
                    # 处理底物
                    stack.append((rxn.substrate, rxn_path))
        
        return solutions

    def get_best_partial_path(self) -> List[Tuple[str, str]]:
        """获取最佳部分路径（相似度最高的节点路径）"""
        # 找到相似度最高的未解决叶节点
        best_node = None
        best_similarity = 0.0
        
        for node in self.mol_nodes.values():
            # 只考虑未解决且未扩展的叶节点
            if not node.solved and not node.expanded and node.depth < self.max_depth:
                if node.max_similarity > best_similarity:
                    best_node = node
                    best_similarity = node.max_similarity
        
        if not best_node:
            self.logger.warning("未找到合适的部分路径节点")
            return []
        
        self.logger.info(f"选择部分路径节点: {best_node.smiles} (相似度={best_similarity:.4f})")
        
        # 回溯路径
        path = []
        current_node = best_node
        
        while current_node:
            # 添加分子节点
            path.insert(0, (current_node.smiles, "Molecule"))
            
            # 如果有父反应，添加反应节点
            if current_node.parent_reactions:
                parent_rxn = current_node.parent_reactions[0]
                path.insert(0, (parent_rxn.parent.smiles, f"Reaction: {parent_rxn.id}"))
                current_node = parent_rxn.parent
            else:
                break
        
        return path
      
    def visualize_path(self, path: List[Tuple[str, str]], output_dir="output"):
        """可视化解决方案路径"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"retro_path_{timestamp}.png")
        
        # 创建有向图
        G = nx.DiGraph()
        pos = {}
        
        # 添加节点
        for i, (smiles, desc) in enumerate(path):
            node_id = f"{i}_{smiles[:10]}"
            if "Reaction" in desc:
                G.add_node(node_id, label=desc, type="reaction")
                pos[node_id] = (i, 0)
            else:
                G.add_node(node_id, label=f"Molecule: {smiles[:10]}", type="molecule")
                pos[node_id] = (i, 1 if i % 2 == 0 else -1)
        
        # 添加边
        for i in range(1, len(path)):
            G.add_edge(f"{i-1}_{path[i-1][0][:10]}", f"{i}_{path[i][0][:10]}")
        
        # 绘制图形
        plt.figure(figsize=(15, 8))
        
        # 节点颜色和形状
        node_colors = []
        for node in G.nodes():
            if "reaction" in G.nodes[node].get('type', ''):
                node_colors.append('lightblue')
            else:
                node_colors.append('lightgreen' if path[int(node.split('_')[0])][0] in self.buyable else 'salmon')
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000)
        nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20)
        
        # 添加标签
        labels = {node: G.nodes[node]['label'] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        
        # 添加标题
        title = "完整逆合成路径" if self.root.solved else "部分逆合成路径 (相似度最高)"
        plt.title(title, fontsize=16)
        
        # 保存和显示
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        self.logger.info(f"路径可视化已保存至: {output_file}")
        return output_file
    
    def generate_report(self, output_dir="output"):
        """生成完整报告"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"report_{timestamp}.txt")
        
        with open(report_file, "w") as f:
            f.write("="*80 + "\n")
            f.write("AND-OR树逆合成路径搜索报告\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"目标分子: {self.target}\n")
            f.write(f"可购买分子数: {len(self.buyable)}\n")
            f.write(f"搜索策略: {self.strategy}\n")
            f.write(f"最大深度: {self.max_depth}\n")
            f.write(f"最大分支: {self.max_branches}\n")
            f.write(f"最大节点数: {self.max_nodes}\n")
            f.write(f"实际扩展节点数: {self.nodes_expanded}\n")
            f.write(f"总耗时: {time.time() - self.start_time:.2f}秒\n")
            f.write(f"根节点解决状态: {'已解决' if self.root.solved else '未解决'}\n")
            f.write(f"根节点最小代价: {self.root.cost:.4f}\n")
            f.write(f"根节点最大相似度: {self.root.max_similarity:.4f}\n\n")
            
            # 统计信息
            f.write("节点统计:\n")
            f.write("-"*80 + "\n")
            f.write(f"分子节点总数: {len(self.mol_nodes)}\n")
            f.write(f"反应节点总数: {len(self.rxn_nodes)}\n")
            
            solved_mol = sum(1 for node in self.mol_nodes.values() if node.solved)
            buyable_mol = sum(1 for node in self.mol_nodes.values() if node.buyable)
            f.write(f"已解决分子节点: {solved_mol} (其中可购买: {buyable_mol})\n")
            f.write(f"未解决分子节点: {len(self.mol_nodes) - solved_mol}\n\n")
            f.flush()
            
            # # 最佳部分路径
            # if not self.root.solved:
            #     f.write("最佳部分路径:\n")
            #     f.write("-"*80 + "\n")
            #     path = self.get_best_partial_path()
            #     for i, (smiles, desc) in enumerate(path):
            #         if "Reaction" in desc:
            #             f.write(f"步骤 {i//2+1}: {desc}\n")
            #         else:
            #             f.write(f"分子: {smiles}\n")
            #     f.write("\n")
            
            # 完整解决方案
            solutions = self.get_solution_paths()
            f.write(f"找到完整路径数: {len(solutions)}\n")
            
            if solutions:
                for i, path in enumerate(solutions, 1):
                    f.write(f"\n解决方案 #{i}:\n")
                    f.write("-"*60 + "\n")
                    for j, (smiles, desc) in enumerate(path):
                        if "Reaction" in desc:
                            f.write(f"步骤 {j//2+1}: {desc}\n")
                        else:
                            f.write(f"分子: {smiles}\n")
        
        self.logger.info(f"报告已保存至: {report_file}")
        return report_file
    
    def visualize_tree(self, output_dir="output", format="svg", 
                        max_depth=None, max_nodes=5000):
        """
        安全可视化AND-OR树（避免死循环）
        
        参数:
            output_dir: 输出目录
            format: 图像格式 (svg, png, pdf等)
            max_depth: 最大可视化深度 (None表示无限制)
            max_nodes: 最大可视化节点数 (防止过大树)
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"andor_tree_{timestamp}.{format}")
        self.logger.info(f"Visualizing ANDOR Tree to {output_file}")
        
        # 创建graphviz图
        dot = Digraph(comment="AND-OR Synthesis Tree")
        dot.attr(rankdir="TB")  # 从上到下布局
        dot.attr("node", fontname="Arial", fontsize="10")
        
        # 使用两个集合跟踪状态
        visited_nodes = set()  # 已添加到图的节点
        processed_nodes = set()  # 已处理子节点的节点
        
        # 使用栈进行迭代遍历 (节点, 父节点ID, 当前深度)
        stack = [(self.root, None, 0)]
        
        node_count = 0
        
        while stack and node_count < max_nodes:
            self.logger.info(f"There are {len(stack)} nodes in the stack")
            current_node, parent_id, depth = stack.pop()
            node_id = f"{'mol' if isinstance(current_node, MoleculeNode) else 'rxn'}_{id(current_node)}"
            
            # 检查深度限制
            if max_depth is not None and depth > max_depth:
                continue
                
            # 如果节点未添加到图中，添加它
            if node_id not in visited_nodes:
                visited_nodes.add(node_id)
                node_count += 1
                
                # 处理分子节点
                if isinstance(current_node, MoleculeNode):
                    # 创建节点标签
                    label = f"{current_node.smiles[:15]}{'...' if len(current_node.smiles) > 15 else ''}\n"
                    label += f"D: {depth} | C: {current_node.cost:.2f}\n"
                    label += f"S: {'✓' if current_node.solved else '✗'}\n"
                    label += f"B: {'✓' if current_node.buyable else '✗'}"
                    
                    # 设置节点属性
                    node_attrs = {
                        "shape": "ellipse",
                        "style": "filled",
                        "fillcolor": "#f5e722" if current_node.buyable else (
                            "#90ee90" if current_node.solved else "#ffb6c1"
                        ),
                        "fontname": "Arial",
                        "fontsize": "9"
                    }
                    
                    # 添加节点
                    dot.node(node_id, label, **node_attrs)
                    self.logger.info(f"Add MoleculeNode with {label}") 
                
                # 处理反应节点
                elif isinstance(current_node, ReactionNode):
                    # 创建节点标签
                    label = f"Rxn: {current_node.id[:15]}{'...' if len(current_node.id) > 15 else ''}\n"
                    label += f"S: {current_node.score:.4f}\n"
                    label += f"Solved: {'✓' if current_node.solved else '✗'}"
                    
                    # 设置节点属性
                    node_attrs = {
                        "shape": "box",
                        "style": "filled",
                        "fillcolor": "#add8e6" if current_node.solved else "#d3d3d3",
                        "fontname": "Arial",
                        "fontsize": "9"
                    }
                    
                    # 添加节点
                    dot.node(node_id, label, **node_attrs)
                    self.logger.info(f'Add ReactionNode with {label}')  
            
            # 添加与父节点的边
            if parent_id:
                dot.edge(parent_id, node_id)
            
            # 如果节点尚未处理其子节点
            if node_id not in processed_nodes:
                processed_nodes.add(node_id)
                
                # 处理分子节点的子反应
                if isinstance(current_node, MoleculeNode):
                    # 将子反应加入栈（逆序加入，保证顺序正确）
                    for rxn in reversed(current_node.reactions):
                        stack.append((rxn, node_id, depth))
                
                # 处理反应节点的底物分子
                elif isinstance(current_node, ReactionNode) and current_node.substrate:
                    stack.append((current_node.substrate, node_id, depth + 1))
                # elif isinstance(current_node, ReactionNode):
                #     # 将底物分子加入栈（逆序加入，保证顺序正确）
                #     for sub in reversed(current_node.substrates):
                #         stack.append((sub, node_id, depth + 1))
        
        # 渲染并保存图像
        output_path = dot.render(output_file, format=format, cleanup=True)
        self.logger.info(f"树可视化已保存至: {output_path}")
        self.logger.info(f"可视化节点数: {node_count}")
        
        if node_count >= max_nodes:
            self.logger.warning(f"达到最大节点限制 {max_nodes}，可能未完整可视化整个树")
        
        return output_path

     
    def prune_and_visualize(self, strategy: str = "solution_paths", format: str = "svg", 
                          output_dir: str = "output") -> Tuple[str, int]:
        """
        剪枝树并可视化剪枝后的结果
        
        参数:
            strategy: 剪枝策略
            format: 输出格式
            output_dir: 输出目录
            
        返回:
            (输出文件路径, 剪除节点数)
        """
        # 创建剪枝器
        pruner = ANDORTreePruner(self)
        
        # 执行剪枝
        kept_mol_nodes, kept_rxn_nodes, pruned_count = pruner.prune_tree(strategy)
        
        if not kept_mol_nodes:
            self.logger.warning("剪枝后无保留节点，跳过可视化")
            return "", pruned_count
        
        # 创建剪枝后的构建器
        pruned_builder = pruner.create_pruned_builder(kept_mol_nodes, kept_rxn_nodes)
        
        # 可视化剪枝后的树
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"pruned_tree_{strategy}_{timestamp}"
        
        # 临时替换原始构建器的可视化方法中的节点数据
        original_mol_nodes = self.mol_nodes
        original_rxn_nodes = self.rxn_nodes
        
        try:
            self.mol_nodes = kept_mol_nodes
            self.rxn_nodes = kept_rxn_nodes
            output_path = self.visualize_tree(output_dir=output_dir, format=format, 
                                            max_nodes=len(kept_mol_nodes) + len(kept_rxn_nodes))
        finally:
            # 恢复原始数据
            self.mol_nodes = original_mol_nodes  
            self.rxn_nodes = original_rxn_nodes
        
        self.logger.info(f"剪枝后树可视化完成: {output_path}")
        return output_path, pruned_count
    
    def compare_before_after_pruning(self, strategy: str = "solution_paths", 
                                   format: str = "svg", output_dir: str = "output"):
        """比较剪枝前后的树结构"""
        import os
        
        self.logger.info("生成剪枝前后对比...")
        
        # 可视化原始树
        self.logger.info("可视化原始树...")
        original_path = self.visualize_tree(output_dir=output_dir, format=format)
        
        # 剪枝并可视化
        self.logger.info("剪枝并可视化...")
        pruned_path, pruned_count = self.prune_and_visualize(strategy, format, output_dir)
        
        # 生成比较报告
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"pruning_comparison_{timestamp}.txt")
        
        with open(report_file, "w") as f:
            f.write("="*80 + "\n")
            f.write("AND-OR树剪枝对比报告\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"目标分子: {self.target}\n")
            f.write(f"剪枝策略: {strategy}\n")
            f.write(f"时间戳: {timestamp}\n\n")
            
            f.write("剪枝前统计:\n")
            f.write("-"*40 + "\n")
            f.write(f"分子节点数: {len(self.mol_nodes)}\n")
            f.write(f"反应节点数: {len(self.rxn_nodes)}\n")
            f.write(f"总节点数: {len(self.mol_nodes) + len(self.rxn_nodes)}\n")
            f.write(f"可视化文件: {original_path}\n\n")
            
            if pruned_path:
                # 重新计算剪枝后统计
                pruner = ANDORTreePruner(self)
                kept_mol_nodes, kept_rxn_nodes, _ = pruner.prune_tree(strategy)
                
                f.write("剪枝后统计:\n")
                f.write("-"*40 + "\n")
                f.write(f"分子节点数: {len(kept_mol_nodes)}\n")
                f.write(f"反应节点数: {len(kept_rxn_nodes)}\n")
                f.write(f"总节点数: {len(kept_mol_nodes) + len(kept_rxn_nodes)}\n")
                f.write(f"剪除节点数: {pruned_count}\n")
                f.write(f"保留率: {((len(kept_mol_nodes) + len(kept_rxn_nodes)) / (len(self.mol_nodes) + len(self.rxn_nodes)) * 100):.2f}%\n")
                f.write(f"可视化文件: {pruned_path}\n")
            else:
                f.write("剪枝后无保留节点\n")
        
        self.logger.info(f"对比报告已保存: {report_file}")
        return original_path, pruned_path, report_file   

