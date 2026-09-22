# analyzer.py - 简化版MCTS搜索过程分析器
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
try:
    import graphviz
except ImportError:
    graphviz = None

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

# 建议安装 pymoo 用于专业的非支配排序
try:
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False

from src.nodes import MoleculeNode, ReactionNode, Node
from src.synthesis_tree import SynthesisTree, SynthesisPath
from utils.chem import smiles_to_img

# ============================================================================
# 1. 数据记录模块 (Data Recording)
# ============================================================================

@dataclass
class IterationRecord:
    """记录单次迭代的关键性能指标"""
    iteration: int
    elapsed_time: float
    tree_size: int
    complete_paths: int
    pareto_front_size: int
    hypervolume: float


# ============================================================================
# 2. 主分析器类 (Analyzer Facade)
# ============================================================================

class Analyzer:
    """统一的数据记录和分析中心"""
    
    def __init__(self, result_path: str = 'results'):
        self.result_path = result_path
        os.makedirs(result_path, exist_ok=True)

        # 搜索状态
        # self.target_molecule = ""
        # self.search_tree = None

        self.iteration_records: List[IterationRecord] = []


    def set_data_source(self, search_tree, target_molecule: str):
        """设置数据源"""
        self.search_tree = search_tree
        self.target_molecule = target_molecule


    # ========================================================================
    # 事件回调接口 - 供MCTS使用
    # ========================================================================
    
    # def one_iteration_end(self, iteration: int, elapsed_time: float, tree: SynthesisTree):
    #     """在每次MCTS迭代结束时被调用，用于记录状态"""
    #     stats = tree.get_statistics()
        
    #     # 计算多目标相关指标
    #     complete_paths = tree.complete_paths
    #     if len(complete_paths) > 0:
    #         objectives = np.array([p.objective_values for p in complete_paths])
            
    #         # 计算Pareto前沿大小
    #         if PYMOO_AVAILABLE:
    #             # 使用pymoo进行非支配排序
    #             nds = NonDominatedSorting()
    #             fronts = nds.do(objectives, only_non_dominated_front=True)
    #             pareto_size = len(fronts)
    #         else:
    #             # 简化版：假设所有完整路径都在前沿上（不精确）
    #             pareto_size = len(complete_paths)

    #         # 超体积计算需要一个参考点 (Reference Point)
    #         # 假设目标是最大化的，参考点应该是目标空间中最差的点（或略差于）
    #         # 这里我们简单设置为所有目标维度的最小值
    #         ref_point = np.min(objectives, axis=0) - 0.1 
    #         # hypervolume = calculate_hypervolume(objectives[fronts], ref_point) # 实际计算较为复杂
    #         hypervolume = 0.0 # 暂时跳过复杂计算，可后续添加
    #     else:
    #         pareto_size = 0
    #         hypervolume = 0.0

    #     record = IterationRecord(
    #         iteration=iteration,
    #         elapsed_time=elapsed_time,
    #         tree_size=stats.total_nodes,
    #         complete_paths=len(complete_paths),
    #         pareto_front_size=pareto_size,
    #         hypervolume=hypervolume
    #     )
    #     self.iteration_records.append(record)

    # # ========================================================================
    # # 简化的便捷方法
    # # ========================================================================
    
    # # def quick_analysis(self, target_molecule: str, search_tree: SynthesisTree, prune:bool=True) -> Dict[str, str]:
    #     """一键分析"""

    #     # plots = {}
    #     # plots['convergence'] = self.generate_convergence_plot()
    #     # plots['paths'] = self.generate_path_analysis()
    #     # if molecule_info and reaction_info and connection_info:
    #     #     plots['search_tree'] = self.plot_search_tree_graphviz(molecule_info, reaction_info, connection_info)
        
    #     # report_file = self.generate_report(target_molecule)
    #     # self.print_summary(target_molecule)
        
    #     # print(f"\n✅ 分析完成!")
    #     # print(f"📊 图表: {list(plots.values())}")
    #     # print(f"📋 报告: {report_file}")
        
    #     # return {"plots": plots, "report": report_file}
    
    # def generate_report(self, target_molecule: str) -> str:
    #     """生成JSON报告"""
    #     summary = self.get_summary(target_molecule)
        
    #     # 获取最佳路径
    #     best_paths = sorted(
    #         [p for p in self.search_tree.all_paths if p.is_complete],
    #         key=lambda x: sum(x.objective_values),
    #         reverse=True
    #     )[:5]
        
    #     report = {
    #         "summary": asdict(summary),
    #         "best_paths": [asdict(p) for p in best_paths],
    #         "statistics": {
    #             # "total_iterations": len(self.iterations),
    #             "total_paths": len(self.search_tree.complete_paths),
    #             "complete_paths": len([p for p in self.paths if p.is_complete]),
    #             "avg_path_length": np.mean([len(p.molecules) for p in self.search_tree.complete_paths]) if self.search_tree.complete_paths else 0
    #         },
    #         "generation_time": time.strftime('%Y-%m-%d %H:%M:%S')
    #     }
        
    #     report_path = f"{self.result_path}/report_{int(time.time())}.json"
    #     with open(report_path, 'w', encoding='utf-8') as f:
    #         json.dump(report, f, indent=2, ensure_ascii=False)
        
    #     return report_path
    


    # def get_summary(self, target_molecule: str) -> SearchSummary:
    #     """获取搜索总结"""
    #     complete_paths = len([p for p in self.paths if p.is_complete])
    #     total_time = self.iterations[-1].elapsed_time if self.iterations else 0
        
    #     return SearchSummary(
    #         target_molecule=target_molecule,
    #         total_iterations=len(self.iterations),
    #         total_time=total_time,
    #         total_paths=len(self.paths),
    #         complete_paths=complete_paths,
    #         tree_size=self.iterations[-1].tree_size if self.iterations else 0
    #     )

    # def print_summary(self, target_molecule: str):
    #     """打印搜索总结"""
    #     summary = self.get_summary(target_molecule)
        
    #     print("=" * 50)
    #     print(f"MCTS搜索分析报告 - {target_molecule}")
    #     print("=" * 50)
    #     print(f"总迭代次数: {summary.total_iterations}")
    #     print(f"总运行时间: {summary.total_time:.2f} 秒")
    #     print(f"发现路径数: {summary.total_paths}")
    #     print(f"完整路径数: {summary.complete_paths}")
    #     print(f"成功率: {summary.complete_paths/max(1,summary.total_paths)*100:.1f}%")
        
    
    # ========================================================================
    # 分析和可视化方法
    # ========================================================================
    
    # def generate_convergence_plot(self, save_path: Optional[str] = None) -> str:
    #     """生成收敛图"""
    #     if not self.iteration_records:
    #         return ""
        
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
    #     iterations = [r.iteration for r in self.iteration_records]
    #     times = [r.elapsed_time for r in self.iteration_records]
    #     obj_sums = [sum(r.objective_values) for r in self.iteration_records]
        
    #     # 目标值趋势
    #     ax1.plot(iterations, obj_sums, 'b-', linewidth=2)
    #     ax1.set_xlabel('Iteration')
    #     ax1.set_ylabel('Objective Sum')
    #     ax1.set_title('Objective Evolution')
    #     ax1.grid(True, alpha=0.3)
        
    #     # 时间消耗
    #     ax2.plot(iterations, times, 'r-', linewidth=2)
    #     ax2.set_xlabel('Iteration')
    #     ax2.set_ylabel('Time (seconds)')
    #     ax2.set_title('Time Progress')
    #     ax2.grid(True, alpha=0.3)
        
    #     plt.tight_layout()
        
    #     if save_path is None:
    #         save_path = f"{self.result_path}/convergence_{int(time.time())}.png"
        
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     plt.close()
        
    #     return save_path
    
    # def generate_path_analysis(self, save_path: Optional[str] = None) -> str:
    #     """生成路径分析图"""
    #     if not self.paths:
    #         return ""
        
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
    #     # 路径发现时间线
    #     complete_paths = [p for p in self.paths if p.is_complete]
    #     if complete_paths:
    #         iterations = [p.iteration_found for p in complete_paths]
    #         lengths = [len(p.molecules) for p in complete_paths]
            
    #         ax1.scatter(iterations, lengths, alpha=0.7, s=50)
    #         ax1.set_xlabel('Discovery Iteration')
    #         ax1.set_ylabel('Path Length')
    #         ax1.set_title('Path Discovery Timeline')
    #         ax1.grid(True, alpha=0.3)
        
    #     # 路径长度分布
    #     all_lengths = [len(p.molecules) for p in self.paths]
    #     if all_lengths:
    #         ax2.hist(all_lengths, bins=max(5, len(set(all_lengths))), alpha=0.7)
    #         ax2.set_xlabel('Path Length')
    #         ax2.set_ylabel('Count')
    #         ax2.set_title('Path Length Distribution')
    #         ax2.grid(True, alpha=0.3)
        
    #     plt.tight_layout()
        
    #     if save_path is None:
    #         save_path = f"{self.result_path}/paths_{int(time.time())}.png"
        
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     plt.close()
        
    #     return save_path
    
    def plot_pareto_front(self, pareto_front,  save_path: Optional[str] = None) -> str:
        """可视化Pareto前沿 (假设2-3目标)"""
        # pareto_front = self.compute_pareto_front()
        if not pareto_front:
            return ""
        
        # objectives = np.array([path.objective_values for path in pareto_front])
        objectives = np.array(pareto_front)
        num_objs = objectives.shape[1]
        
        if num_objs == 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(objectives[:, 0], objectives[:, 1], c='blue', s=50)
            ax.set_xlabel('Objective 1')
            ax.set_ylabel('Objective 2')
            ax.set_title('Pareto Front')
            ax.grid(True, alpha=0.3)
        elif num_objs == 3:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2], c='blue', s=50)
            ax.set_xlabel('Objective 1')
            ax.set_ylabel('Objective 2')
            ax.set_zlabel('Objective 3')
            ax.set_title('Pareto Front')
        else:
            return ""  # 不支持>3维
        
        if save_path is None:
            save_path = os.path.join(self.result_path, "pareto_front.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        return save_path
    
    def plot_search_tree_graphviz(self,molecule_info: List[Dict], 
                                    reaction_info: List[Dict], 
                                    connection_info: List[Dict],
                                    save_path : Optional[str] = None, 
                                    format: str = 'pdf',
                                    include_node_png: bool = False,
                                    graph_name: str = 'Search_Tree'
                                    ) -> str:
        save_path = save_path or self.result_path
        if graphviz is None:
            return ""
        # 创建可视化图
        dot = graphviz.Digraph(comment=graph_name)
        dot.attr(rankdir='LR')
        # dot.attr('node', shape='box', style='rounded,filled')
        dot.attr('edge', arrowsize='1.0')
        
        #  创建分子节点（直接使用 id ）
        for info in molecule_info:
            
            shape = 'ellipse'#'none'

            if include_node_png:
                mol_imgs_path = os.path.join(save_path, 'imgs')
                if not os.path.exists(mol_imgs_path):
                    os.mkdir(mol_imgs_path)
                img_abspath = smiles_to_img(info['molecule'], name=info['id'], out_dir=mol_imgs_path)
                label = f'''<
                    <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                    <TR>
                        <TD><IMG SRC="{img_abspath}"/></TD>
                    </TR>
                    <TR>
                        <TD ALIGN="CENTER"><FONT POINT-SIZE="12">{info['molecule']}</FONT></TD>
                    </TR>
                    </TABLE>
                    >'''
                dot.node(info['id'], label=label, margin='0',shape=shape)
            else:
                label_parts = [
                    f"{info['molecule']}",
                    # f"Visits: {info['visits']}",
                    # f"Depth: {info['depth']}",
                    f"Simi: {info['similarity']}"
                    ]
                label = "\\n".join(label_parts)
                color = 'gold' if info['is_commercial'] else 'lightyellow'
                dot.node(info['id'], label=label, fillcolor=color, shape=shape)

        for info in reaction_info:
            label_parts = [
                f"EC: {info['ec']}",
                # f"Templates: {info['templates'][:10]}",
                f"Score: {info['scores']}",
                info['topk'],
            ]
            label = "\\n".join(label_parts)
            color = 'lightcoral'
            shape = 'box'

            dot.node(info['id'], label, fillcolor=color, shape=shape)
        
        for edge in connection_info:
            dot.edge(edge['source'], edge['target'], 
                     label="" #edge["label"]
                     )
        
        save_file = os.path.join(save_path, graph_name)
        # 输出多种格式，便于排查
        dot.save(save_file + ".dot")
        dot.render(save_file, format=format, cleanup=True)
        # dot.render(save_file, format="svg", cleanup=True)
        dot.render(save_file, format="pdf", cleanup=True)
        
        return save_file

    def plot_path_graphviz(self, 
                           path: SynthesisPath,
                           save_path : Optional[str] = None, 
                           format: str = 'png',
                           include_node_png: bool = False,
                           graph_name="Synthesis_path"
                           ) -> str:

        return self.plot_search_tree_graphviz(molecule_info=[m.to_dict for m in path.molecule_nodes],
                                              reaction_info=[r.to_dict for r in path.reaction_nodes],
                                              connection_info=path.connections,
                                              save_path=save_path,
                                              format=format,
                                              include_node_png=include_node_png,
                                              graph_name=graph_name,
                                              )
