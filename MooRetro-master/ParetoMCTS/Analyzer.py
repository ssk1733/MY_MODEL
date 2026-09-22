import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
import os
import json
import time
from dataclasses import dataclass
import graphviz
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV

from ParetoMCTS.nodes import Node, MoleculeNode, ReactionNode
from ParetoMCTS.config import EvaluationConfig


@dataclass
class IterationRecord:
    """单次迭代记录"""
    iteration: int
    selected_node: str
    expanded_node: Optional[str]
    objective_vector: np.ndarray
    hypervolume: float
    pareto_front_size: int
    elapsed_time: float
    tree_size: int


@dataclass
class PathRecord:
    """路径记录"""
    path_id: int
    iteration_found: int
    molecules: List[str]  # 分子序列 (从目标分子到起始材料)
    reactions: List[Dict]  # 反应序列信息
    objective_vector: np.ndarray  # 目标函数值
    path_length: int  # 路径长度
    is_complete: bool  # 是否为完整路径(到达商业分子)
    discovery_time: float  # 发现时间
    node_depth: int  # 节点深度


class Analyzer:
    """MCTS搜索过程分析器"""
    
    def __init__(self, result_path: str = 'results/analysis'):
        """
        初始化分析器
        
        Args:
            result_path: 结果保存路径
        """
        self.result_path = result_path
        os.makedirs(result_path, exist_ok=True)
        
        # 分析数据存储
        self.iteration_records: List[IterationRecord] = []
        self.path_records: List[PathRecord] = []  # 新增：路径记录
        self.path_counter = 0  # 路径计数器
        self.final_solutions: List[np.ndarray] = []
        self.search_tree: Optional[Node] = None
        self.config: Optional[EvaluationConfig] = None
        self.target_molecule: str = ""
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def record_iteration(self, iteration: int, selected_node: str, 
                        expanded_node: Optional[str], objective_vector: np.ndarray,
                        hypervolume: float, pareto_front_size: int, 
                        elapsed_time: float, tree_size: int):
        """记录单次迭代信息"""
        record = IterationRecord(
            iteration=iteration,
            selected_node=selected_node,
            expanded_node=expanded_node,
            objective_vector=objective_vector.copy(),
            hypervolume=hypervolume,
            pareto_front_size=pareto_front_size,
            elapsed_time=elapsed_time,
            tree_size=tree_size
        )
        self.iteration_records.append(record)
    
    def record_path(self, iteration: int, molecules: List[str], 
                   reactions: List[Dict], objective_vector: np.ndarray,
                   elapsed_time: float, node_depth: int, is_complete: bool = False):
        """
        记录发现的路径
        
        Args:
            iteration: 发现路径的迭代次数
            molecules: 分子序列
            reactions: 反应序列
            objective_vector: 目标函数值
            elapsed_time: 发现时间
            node_depth: 节点深度
            is_complete: 是否为完整路径
        """
        if is_complete:
            self.path_counter += 1
            
            path_record = PathRecord(
                path_id=self.path_counter,
                iteration_found=iteration,
                molecules=molecules.copy(),
                reactions=reactions.copy(),
                objective_vector=objective_vector.copy(),
                path_length=len(reactions),
                is_complete=is_complete,
                discovery_time=elapsed_time,
                node_depth=node_depth
            )
            
            self.path_records.append(path_record)
            
            print(f"Path {self.path_counter} recorded: {len(molecules)} molecules, "
                f"{len(reactions)} reactions, complete: {is_complete}")
    
    def get_path_statistics(self) -> Dict:
        """获取路径统计信息"""
        if not self.path_records:
            return {
                'total_paths': 0,
                'complete_paths': 0,
                'completion_rate': 0.0,
                'average_path_length': 0.0,
                'average_complete_path_length': 0.0
            }
        
        complete_paths = [p for p in self.path_records if p.is_complete]
        path_lengths = [p.path_length for p in self.path_records]
        complete_path_lengths = [p.path_length for p in complete_paths] if complete_paths else [0]
        
        return {
            'total_paths': len(self.path_records),
            'complete_paths': len(complete_paths),
            'completion_rate': len(complete_paths) / len(self.path_records),
            'average_path_length': np.mean(path_lengths),
            'min_path_length': np.min(path_lengths),
            'max_path_length': np.max(path_lengths),
            'average_complete_path_length': np.mean(complete_path_lengths) if complete_paths else 0,
        }
    
    def plot_path_discovery(self, save_path: Optional[str] = None):
        """绘制路径发现过程"""
        if not self.path_records:
            print("No paths recorded for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 路径发现时间线
        iterations = [p.iteration_found for p in self.path_records]
        path_lengths = [p.path_length for p in self.path_records]
        complete_mask = [p.is_complete for p in self.path_records]
        
        complete_iterations = [iterations[i] for i, complete in enumerate(complete_mask) if complete]
        incomplete_iterations = [iterations[i] for i, complete in enumerate(complete_mask) if not complete]
        complete_lengths = [path_lengths[i] for i, complete in enumerate(complete_mask) if complete]
        incomplete_lengths = [path_lengths[i] for i, complete in enumerate(complete_mask) if not complete]
        
        axes[0,0].scatter(incomplete_iterations, incomplete_lengths, 
                         c='red', alpha=0.6, label='Incomplete Paths', s=30)
        axes[0,0].scatter(complete_iterations, complete_lengths, 
                         c='green', alpha=0.8, label='Complete Paths', s=50)
        axes[0,0].set_xlabel('Iteration')
        axes[0,0].set_ylabel('Path Length')
        axes[0,0].set_title('Path Discovery Timeline')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. 路径长度分布
        axes[0,1].hist(path_lengths, bins=max(10, len(set(path_lengths))), 
                      alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,1].set_xlabel('Path Length')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Path Length Distribution')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. 完整路径vs不完整路径
        complete_count = sum(complete_mask)
        incomplete_count = len(complete_mask) - complete_count
        
        labels = ['Complete', 'Incomplete']
        sizes = [complete_count, incomplete_count]
        colors = ['lightgreen', 'lightcoral']
        
        axes[1,0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1,0].set_title('Path Completion Rate')
        
        # 4. 目标函数值演化
        if self.path_records and len(self.path_records[0].objective_vector) > 0:
            objective_values = np.array([p.objective_vector for p in self.path_records])
            
            for i in range(min(3, objective_values.shape[1])):  # 最多显示3个目标
                obj_label = self.config.enabled_objectives[i] if self.config and i < len(self.config.enabled_objectives) else f'Objective {i+1}'
                axes[1,1].scatter(iterations, objective_values[:, i], 
                                label=obj_label, alpha=0.7, s=30)
            
            axes[1,1].set_xlabel('Iteration')
            axes[1,1].set_ylabel('Objective Values')
            axes[1,1].set_title('Objective Values Evolution')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.result_path, "path_discovery_analysis.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Path discovery analysis plot saved to: {save_path}")
    
    def export_paths_to_csv(self, save_path: Optional[str] = None):
        """导出路径信息到CSV文件"""
        if not self.path_records:
            print("No paths to export")
            return
        
        if save_path is None:
            save_path = os.path.join(self.result_path, "completed_paths.csv")
        
        # 准备数据
        data = []
        for path in self.path_records:
            row = {
                'path_id': path.path_id,
                'iteration_found': path.iteration_found,
                'path_length': path.path_length,
                'is_complete': path.is_complete,
                'discovery_time': path.discovery_time,
                'node_depth': path.node_depth,
                'molecules': ' -> '.join(path.molecules),
                'num_reactions': len(path.reactions)
            }
            
            # 添加目标函数值
            if self.config:
                for i, obj_name in enumerate(self.config.enabled_objectives):
                    if i < len(path.objective_vector):
                        row[f'objective_{obj_name}'] = path.objective_vector[i]
            
            data.append(row)
        
        # 保存到CSV
        import pandas as pd
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        print(f"Paths exported to CSV: {save_path}")
    
    def get_best_paths(self, top_k: int = 10, criterion: str = 'hypervolume') -> List[PathRecord]:
        """
        获取最佳路径
        
        Args:
            top_k: 返回前k个最佳路径
            criterion: 评判标准 ('hypervolume', 'length', 'objectives_sum')
        """
        if not self.path_records:
            return []
        
        if criterion == 'length':
            # 按路径长度排序（越短越好）
            sorted_paths = sorted(self.path_records, key=lambda p: p.path_length)
        elif criterion == 'objectives_sum':
            # 按目标函数值总和排序（越大越好）
            sorted_paths = sorted(self.path_records, 
                                key=lambda p: np.sum(p.objective_vector), reverse=True)
        else:  # hypervolume
            # 使用完整路径优先，然后按目标函数值排序
            complete_paths = [p for p in self.path_records if p.is_complete]
            if complete_paths:
                sorted_paths = sorted(complete_paths, 
                                    key=lambda p: np.sum(p.objective_vector), reverse=True)
            else:
                sorted_paths = sorted(self.path_records, 
                                    key=lambda p: np.sum(p.objective_vector), reverse=True)
        
        return sorted_paths[:top_k]
    
    def set_final_results(self, search_tree: Node, final_solutions: List[np.ndarray],
                         config: EvaluationConfig, target_molecule: str):
        """设置最终结果"""
        self.search_tree = search_tree
        self.final_solutions = final_solutions
        self.config = config
        self.target_molecule = target_molecule
    
    def plot_pareto_front(self, save_path: Optional[str] = None):
        """可视化Pareto前沿"""
        if not self.final_solutions or not self.config:
            raise ValueError("最终解集或配置未设置")
            
        n_objectives = len(self.config.enabled_objectives)
        
        if n_objectives == 2:
            self._plot_2d_pareto_front(save_path)
        elif n_objectives == 3:
            self._plot_3d_pareto_front(save_path)
        else:
            self._plot_parallel_coordinates(save_path)
    
    def _plot_2d_pareto_front(self, save_path: Optional[str] = None):
        """2D Pareto front visualization"""
        solutions = np.array(self.final_solutions)
        
        # Get Pareto fronts
        nds = NonDominatedSorting()
        fronts = nds.do(solutions, return_rank=True, only_non_dominated_front=False)
        
        plt.figure(figsize=(10, 8))
        
        # Plot different front levels
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for rank in range(min(5, len(np.unique(fronts[1])))):
            mask = fronts[1] == rank
            rank_solutions = solutions[mask]
            label = f"Front {rank+1}" if rank == 0 else f"Front {rank+1}"
            plt.scatter(rank_solutions[:, 0], rank_solutions[:, 1], 
                       c=colors[rank % len(colors)], label=label, alpha=0.7, s=60)
        
        # Connect Pareto front points
        pareto_solutions = solutions[fronts[1] == 0]
        sorted_indices = np.argsort(pareto_solutions[:, 0])
        pareto_sorted = pareto_solutions[sorted_indices]
        plt.plot(pareto_sorted[:, 0], pareto_sorted[:, 1], 'r--', alpha=0.5)
        
        plt.xlabel(self.config.enabled_objectives[0])
        plt.ylabel(self.config.enabled_objectives[1])
        plt.title(f'{self.target_molecule} - 2D Pareto Front')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = os.path.join(self.result_path, "pareto_front_2d.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"2D Pareto front plot saved to: {save_path}")
    
    def _plot_3d_pareto_front(self, save_path: Optional[str] = None):
        """3D Pareto front visualization"""
        solutions = np.array(self.final_solutions)
        
        nds = NonDominatedSorting()
        fronts = nds.do(solutions, return_rank=True)
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot different front levels
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for rank in range(min(3, len(np.unique(fronts[1])))):
            mask = fronts[1] == rank
            rank_solutions = solutions[mask]
            label = f"Front {rank+1}" if rank == 0 else f"Front {rank+1}"
            ax.scatter(rank_solutions[:, 0], rank_solutions[:, 1], rank_solutions[:, 2],
                      c=colors[rank % len(colors)], label=label, alpha=0.7, s=60)
        
        ax.set_xlabel(self.config.enabled_objectives[0])
        ax.set_ylabel(self.config.enabled_objectives[1])
        ax.set_zlabel(self.config.enabled_objectives[2])
        ax.set_title(f'{self.target_molecule} - 3D Pareto Front')
        ax.legend()
        
        if save_path is None:
            save_path = os.path.join(self.result_path, "pareto_front_3d.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"3D Pareto front plot saved to: {save_path}")
    
    def _plot_parallel_coordinates(self, save_path: Optional[str] = None):
        """Parallel coordinates plot for high-dimensional Pareto front"""
        solutions = np.array(self.final_solutions)
        
        nds = NonDominatedSorting()
        fronts = nds.do(solutions, return_rank=True)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Normalize data
        normalized_solutions = (solutions - solutions.min(axis=0)) / (solutions.max(axis=0) - solutions.min(axis=0))
        
        # Plot different front levels
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for rank in range(min(3, len(np.unique(fronts[1])))):
            mask = fronts[1] == rank
            rank_solutions = normalized_solutions[mask]
            
            for solution in rank_solutions:
                ax.plot(range(len(self.config.enabled_objectives)), solution, 
                       color=colors[rank % len(colors)], alpha=0.6, linewidth=1.5)
        
        ax.set_xticks(range(len(self.config.enabled_objectives)))
        ax.set_xticklabels(self.config.enabled_objectives, rotation=45)
        ax.set_ylabel('Normalized Objective Values')
        ax.set_title(f'{self.target_molecule} - High-Dimensional Pareto Front (Parallel Coordinates)')
        ax.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = os.path.join(self.result_path, "pareto_front_parallel.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Parallel coordinates plot saved to: {save_path}")
    
    def plot_convergence(self, save_path: Optional[str] = None):
        """Plot convergence analysis"""
        if not self.iteration_records:
            raise ValueError("Iteration records are empty, please record search process first")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        iterations = [r.iteration for r in self.iteration_records]
        hypervolumes = [r.hypervolume for r in self.iteration_records]
        pareto_sizes = [r.pareto_front_size for r in self.iteration_records]
        tree_sizes = [r.tree_size for r in self.iteration_records]
        elapsed_times = [r.elapsed_time for r in self.iteration_records]
        
        # 1. Hypervolume convergence
        axes[0,0].plot(iterations, hypervolumes, 'b-', linewidth=2, label='Hypervolume')
        axes[0,0].set_xlabel('Iteration')
        axes[0,0].set_ylabel('Hypervolume')
        axes[0,0].set_title('Hypervolume Convergence')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        
        # 2. Pareto front size changes
        axes[0,1].plot(iterations, pareto_sizes, 'r-', linewidth=2, label='Front Size')
        axes[0,1].set_xlabel('Iteration')
        axes[0,1].set_ylabel('Pareto Front Size')
        axes[0,1].set_title('Pareto Front Size Evolution')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend()
        
        # 3. Search tree growth
        axes[1,0].plot(iterations, tree_sizes, 'g-', linewidth=2, label='Tree Size')
        axes[1,0].set_xlabel('Iteration')
        axes[1,0].set_ylabel('Search Tree Nodes')
        axes[1,0].set_title('Search Tree Growth')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].legend()
        
        # 4. Execution time
        axes[1,1].plot(iterations, elapsed_times, 'purple', linewidth=2, label='Cumulative Time')
        axes[1,1].set_xlabel('Iteration')
        axes[1,1].set_ylabel('Cumulative Time (seconds)')
        axes[1,1].set_title('Execution Time Evolution')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].legend()
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.result_path, "convergence_analysis.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Convergence analysis plot saved to: {save_path}")
    
    def analyze_domination_relationships(self):
        """Analyze domination relationships"""
        if not self.final_solutions:
            raise ValueError("Final solution set not set")
            
        solutions = np.array(self.final_solutions)
        n_solutions = len(solutions)
        
        # Calculate domination matrix
        domination_matrix = np.zeros((n_solutions, n_solutions))
        
        for i in range(n_solutions):
            for j in range(n_solutions):
                if i != j:
                    # Check if i dominates j
                    dominates = np.all(solutions[i] >= solutions[j]) and np.any(solutions[i] > solutions[j])
                    if dominates:
                        domination_matrix[i, j] = 1
                    elif np.all(solutions[j] >= solutions[i]) and np.any(solutions[j] > solutions[i]):
                        domination_matrix[i, j] = -1
        
        # Calculate front levels
        nds = NonDominatedSorting()
        ranks = nds.do(solutions, return_rank=True)[1]
        
        # Statistics
        unique_ranks = np.unique(ranks)
        front_sizes = [np.sum(ranks == rank) for rank in unique_ranks]
        
        print("=== Domination Relationship Analysis ===")
        print(f"Total solutions: {n_solutions}")
        print(f"Number of fronts: {len(unique_ranks)}")
        for i, size in enumerate(front_sizes):
            print(f"Front {i+1} size: {size}")
        
        # Domination statistics
        dominates_count = np.sum(domination_matrix > 0, axis=1)
        dominated_count = np.sum(domination_matrix < 0, axis=1)
        
        print(f"Average solutions dominated per solution: {np.mean(dominates_count):.2f}")
        print(f"Average solutions dominating per solution: {np.mean(dominated_count):.2f}")
        
        return {
            'domination_matrix': domination_matrix,
            'ranks': ranks,
            'front_sizes': front_sizes,
            'dominates_count': dominates_count,
            'dominated_count': dominated_count
        }
    
    def _convert_numpy_types(self, obj):
        """递归转换numpy类型为Python原生类型，用于JSON序列化"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        else:
            return obj
    
    def generate_comprehensive_report(self, save_path: Optional[str] = None):
        """Generate comprehensive analysis report"""
        if save_path is None:
            save_path = os.path.join(self.result_path, "comprehensive_report.json")
            
        # Collect statistical information
        domination_analysis = self.analyze_domination_relationships()
        print(domination_analysis)
        
        # Calculate hypervolume
        if self.final_solutions and self.config:
            hv = HV(ref_point=self.config.get_hypervolume_reference_point())
            final_hypervolume = hv(np.array(self.final_solutions))
        else:
            final_hypervolume = 0.0
        
        # Path analysis
        if self.search_tree:
            tree_stats = self._analyze_search_tree()
        else:
            tree_stats = {}
        
        # Path statistics
        path_stats = self.get_path_statistics()
        
        report = {
            'target_molecule': self.target_molecule,
            'search_summary': {
                'total_iterations': len(self.iteration_records),
                'final_hypervolume': final_hypervolume,
                'final_pareto_front_size': len(self.final_solutions),
                'total_search_time': self.iteration_records[-1].elapsed_time if self.iteration_records else 0,
                'final_tree_size': self.iteration_records[-1].tree_size if self.iteration_records else 0
            },
            'convergence_metrics': {
                'hypervolume_improvement': final_hypervolume - (self.iteration_records[0].hypervolume if self.iteration_records else 0),
                'convergence_rate': self._calculate_convergence_rate(),
                'search_efficiency': final_hypervolume / (self.iteration_records[-1].elapsed_time if self.iteration_records else 1)
            },
            'domination_analysis': {
                'front_count': len(domination_analysis['front_sizes']),
                'front_sizes': domination_analysis['front_sizes'],
                'average_dominates': float(np.mean(domination_analysis['dominates_count'])),
                'average_dominated': float(np.mean(domination_analysis['dominated_count']))
            },
            'path_analysis': path_stats,  # 新增：路径分析
            'tree_analysis': tree_stats,
            'configuration': {
                'enabled_objectives': self.config.enabled_objectives if self.config else [],
                'objective_count': len(self.config.enabled_objectives) if self.config else 0
            }
        }
        
        # Convert all numpy types to Python native types
        report = self._convert_numpy_types(report)
        
        # Save report
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Comprehensive analysis report saved to: {save_path}")
        
        # Print key information
        print("\n=== Comprehensive Analysis Report ===")
        print(f"Target molecule: {self.target_molecule}")
        print(f"Total iterations: {report['search_summary']['total_iterations']}")
        print(f"Final hypervolume: {report['search_summary']['final_hypervolume']:.6f}")
        print(f"Pareto front size: {report['search_summary']['final_pareto_front_size']}")
        print(f"Total search time: {report['search_summary']['total_search_time']:.2f} seconds")
        print(f"Search efficiency: {report['convergence_metrics']['search_efficiency']:.6f}")
        
        # Print path statistics
        if path_stats:
            print(f"\n=== Path Discovery Statistics ===")
            print(f"Total paths discovered: {path_stats['total_paths']}")
            print(f"Complete paths: {path_stats['complete_paths']}")
            print(f"Completion rate: {path_stats['completion_rate']:.2%}")
            print(f"Average path length: {path_stats['average_path_length']:.2f}")
            if path_stats['average_complete_path_length'] > 0:
                print(f"Average complete path length: {path_stats['average_complete_path_length']:.2f}")
        
        return report
    
    def _analyze_search_tree(self) -> Dict:
        """分析搜索树结构"""
        if not self.search_tree:
            return {}
        
        def traverse(node, depth=0):
            stats = {
                'total_nodes': 1,
                'molecule_nodes': 1 if isinstance(node, MoleculeNode) else 0,
                'reaction_nodes': 1 if isinstance(node, ReactionNode) else 0,
                'commercial_nodes': 1 if isinstance(node, MoleculeNode) and node.is_commercial else 0,
                'terminal_nodes': 1 if node.is_terminal() else 0,
                'max_depth': depth,
                'total_visits': node.visits,
                'branching_factors': [len(node.children)] if node.children else []
            }
            
            for child in node.children:
                child_stats = traverse(child, depth + 1)
                stats['total_nodes'] += child_stats['total_nodes']
                stats['molecule_nodes'] += child_stats['molecule_nodes']
                stats['reaction_nodes'] += child_stats['reaction_nodes']
                stats['commercial_nodes'] += child_stats['commercial_nodes']
                stats['terminal_nodes'] += child_stats['terminal_nodes']
                stats['max_depth'] = max(stats['max_depth'], child_stats['max_depth'])
                stats['total_visits'] += child_stats['total_visits']
                stats['branching_factors'].extend(child_stats['branching_factors'])
                
            return stats
        
        tree_stats = traverse(self.search_tree)
        tree_stats['average_branching_factor'] = np.mean(tree_stats['branching_factors']) if tree_stats['branching_factors'] else 0
        tree_stats['max_branching_factor'] = max(tree_stats['branching_factors']) if tree_stats['branching_factors'] else 0
        
        return tree_stats
    
    def _calculate_convergence_rate(self) -> float:
        """计算收敛率"""
        if len(self.iteration_records) < 10:
            return 0.0
            
        # 使用最后10次迭代的超体积改进来估算收敛率
        recent_hvs = [r.hypervolume for r in self.iteration_records[-10:]]
        if len(set(recent_hvs)) == 1:  # 没有改进
            return 0.0
        
        # 计算改进趋势
        improvements = np.diff(recent_hvs)
        avg_improvement = np.mean(improvements[improvements > 0])
        return avg_improvement if not np.isnan(avg_improvement) else 0.0


    def collect_tree_info(self, max_depth: int = 20) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        一次遍历收集分子、反应和连接信息，包含去重和ID安全化处理
        
        Args:
            max_depth: 最大收集深度
            
        Returns:
            Tuple: (molecule_info, reaction_info, unique_connections)
        """
        molecule_info = []
        reaction_info = []
        # connection_info = []
        connection_set = set()

        # 跟踪已访问过的MCTS树节点对象ID，避免重复处理和压栈，这是关键的性能优化。
        visited_nodes = set()

        # 栈遍历收集数据
        stack = [(self.search_tree, None, 0)]

        while stack:
            node, parent_id, depth = stack.pop()

            if depth > max_depth:
                continue

            node_id = f"{'mol' if isinstance(node, MoleculeNode) else 'rxn'}_{id(node)}"
            
            if node_id not in visited_nodes:
                visited_nodes.add(node_id)
            
                if isinstance(node, MoleculeNode):
                    node_info = {'id': node_id}
                    node_info.update(node.info)
                    molecule_info.append(node_info)

                else: # ReactionNode
                    node_info = {'id': node_id}
                    node_info.update(node.info)
                    reaction_info.append(node_info)

            if parent_id:
                label = ""
                if isinstance(node, ReactionNode):
                    rxn_info = getattr(node, 'reaction_info', None)
                    if rxn_info:
                        label = f"Score: {rxn_info.scores:.3f}"

                connection_set.add((
                    parent_id,
                    node_id,
                    label
                )) 

            # --- 将子节点【逆序】压入栈中 ---
            # 逆序压栈可以确保当它们被弹出时，处理顺序与原始的子节点顺序一致（从左到右）。
            # 这是DFS的经典技巧。
            for child in reversed(node.children):
                if id(child) not in visited_nodes:
                    stack.append((child, node_id, depth+1))

        molecule_info.sort(key=lambda x: x['depth'])
        reaction_info.sort(key=lambda x: x['depth'])

        connection_info = [
            {
                'id': f"{parent_id}_{child_id}",
                'source': parent_id,
                'target': child_id,
                'label': label
            }
            for parent_id, child_id, label in connection_set
        ]

        return molecule_info, reaction_info, connection_info

    def save_tree_data_to_json(self, molecule_info: List[Dict], reaction_info: List[Dict], 
                            unique_connections: List[Dict], save_path: Optional[str] = None) -> str:
        """
        将收集的树数据保存为JSON文件
        
        Args:
            molecule_info: 分子信息字典
            reaction_info: 反应信息列表
            unique_connections: 去重后的连接信息列表
            save_path: 保存路径
            
        Returns:
            str: 实际保存的文件路径
        """
        if save_path is None:
            save_path = os.path.join(self.result_path, "search_tree_data.json")
        
        # 准备统计信息
        stats = {
            'total_unique_molecules': len(molecule_info),
            'total_reactions': len(reaction_info),
            'total_connections': len(unique_connections),
            'generation_time': time.time()
        }
        
        # 组织完整数据
        tree_data = {
            'metadata': {
                'target_molecule': self.target_molecule if hasattr(self, 'target_molecule') else "Unknown",
                'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'statistics': stats
            },
            'molecules': molecule_info,
            'reactions': reaction_info,
            'connections': unique_connections
        }
        
        # 保存JSON文件
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(tree_data, f, indent=2, ensure_ascii=False)
        
        print(f"搜索树数据已保存到: {save_path}")
        print(f"统计信息:")
        for key, value in stats.items():
            if key != 'generation_time':
                print(f"  - {key}: {value}")
        
        return save_path

    def visualize_search_tree(self, max_depth: int = 20, 
                              save_path: Optional[str] = None,
                              save_data_json: bool = True) -> graphviz.Digraph:
        """
        可视化合并相同分子节点的搜索树
        
        Args:
            max_depth: 最大显示深度
            save_path: 保存路径
            save_data_json: 是否保存数据为JSON
            
        Returns:
            graphviz图对象
        """
        if not self.search_tree:
            raise ValueError("搜索树未设置，请先调用set_final_results")
        
        print("正在收集搜索树信息...")
        
        # 1. 一次遍历收集所有信息（已包含去重和safe_id处理）
        molecule_info, reaction_info, connection_info = self.collect_tree_info(max_depth)
        
        # 2. 保存数据为JSON（如果需要）
        if save_data_json:
            json_path = save_path + "_data.json" if save_path else None
            self.save_tree_data_to_json(molecule_info, reaction_info, connection_info, json_path)
        
        print("正在生成可视化图...")
        
        # 3. 创建可视化图
        dot = graphviz.Digraph(comment='ParetoMCTS Search Tree')
        dot.attr(rankdir='TB', size='20,16')
        dot.attr('node', shape='box', style='rounded,filled')
        dot.attr('edge', arrowsize='0.5')
        
        # 4. 创建分子节点（直接使用 id ）
        for info in molecule_info:
            if info['type'] == 'molecule':
                label_parts = [
                    f"{info['molecule']}",
                    f"Visits: {info['visits']}",
                    f"Depth: {info['depth']}"
                ]
                label = "\\n".join(label_parts)

                color = 'gold' if info['is_commercial'] else 'lightyellow'
                shape = 'ellipse'

            elif info['type'] == 'reaction':
                label_parts = [
                    f"EC: {info['ec']}",
                    f"Templates: {info['templates'][:10]}"
                    f"Score: {info['reaction_score']:.3f}",
                    f"Depth: {info['depth']}"
                ]
                label = "\\n".join(label_parts)
                color = 'lightcoral'
                shape = 'box'

            dot.node(info['id'], label, fillcolor=color, shape=shape)
        
        for edge in connection_info:
            dot.edge(edge['source'], edge['target'], label=edge['label'])

        # 保存
        if save_path is None:
            save_path = os.path.join(self.result_path, f"search_tree.dot")
        
        dot.save(save_path)
        print(f"✅ Graphviz文件已保存: {save_path}")
        
        return dot

    def visualize_complete_paths_tree(self, max_depth: int = 20, 
                                    show_visits: bool = True,
                                    show_solutions: bool = True,
                                    save_path: Optional[str] = None) -> Optional[graphviz.Digraph]:
        """
        可视化仅包含完整路径的剪枝搜索树，并合并相同的分子节点。
        """
        if not self.search_tree:
            raise ValueError("搜索树未设置，请先调用 set_final_results。")
        
        # 步骤 1: 找到所有属于完整路径的节点
        complete_path_nodes = set()
        def find_complete_paths(node: Node, current_path: List[Node]):
            current_path.append(node)
            if isinstance(node, MoleculeNode) and node.is_commercial:
                for path_node in current_path:
                    complete_path_nodes.add(id(path_node))
            for child in node.children:
                if child.depth <= max_depth:
                    find_complete_paths(child, current_path.copy())
        
        find_complete_paths(self.search_tree, [])
        
        if not complete_path_nodes:
            print("在搜索树中未找到完整路径。")
            return None
        
        print(f"在完整路径中找到 {len(complete_path_nodes)} 个节点。")
        
        # 步骤 2: 创建剪枝后的可视化
        dot = graphviz.Digraph(comment='MCTS Complete Paths Tree')
        dot.attr(rankdir='TB', size='20,16')
        dot.attr('node', shape='box', style='rounded,filled')
        dot.attr('edge', arrowsize='0.5')

        # 同样使用字典来跟踪已添加的分子节点
        existing_nodes = {}

        def add_node_to_graph(node: Node, parent_node_id: Optional[str] = None):
            """递归添加属于完整路径的节点，并合并分子节点。"""
            if node.depth > max_depth or id(node) not in complete_path_nodes:
                return

            node_id = f"node_{id(node)}"

            if isinstance(node, MoleculeNode):
                if node.molecule in existing_nodes:
                    if parent_node_id:
                        dot.edge(parent_node_id, existing_nodes[node.molecule])
                    return
                existing_nodes[node.molecule] = node_id
            
            color, shape = 'lightgrey', 'box'
            if isinstance(node, MoleculeNode):
                shape = 'ellipse'
                color = 'gold' if node.is_commercial else 'lightyellow'
                
                label_parts = [f"{node.molecule}"]
                if show_visits: label_parts.append(f"Visits: {node.visits}")
                if show_solutions: label_parts.append(f"Solutions: {len(node.solutions)}")
                label_parts.append(f"Depth: {node.depth}")
                label = "\\n".join(label_parts)

            elif isinstance(node, ReactionNode):
                color = 'lightcoral'
                shape = 'box'
                label_parts = [node.reaction_info.ec,
                               f'{node.reaction_info.templates[:10]}']
                label = "\\n".join(label_parts)

            
            dot.node(node_id, label, fillcolor=color, shape=shape)

            if parent_node_id:
                edge_label = ""
                if isinstance(node, ReactionNode) and node.reaction_info:
                    edge_label = f"Score: {node.reaction_info.scores:.3f}"
                dot.edge(parent_node_id, node_id, label=edge_label)

            for child in node.children:
                add_node_to_graph(child, node_id)
        
        add_node_to_graph(self.search_tree)
        
        if save_path is None:
            save_path = os.path.join(self.result_path, "complete_paths_tree_merged")
        dot.save(save_path)
            
        return dot


