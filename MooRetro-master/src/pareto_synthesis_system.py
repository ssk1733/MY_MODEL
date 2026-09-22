# pareto_synthesis_system.py - 极简主系统
import os
from typing import List, Optional, Dict, Tuple

from .synthesis_tree import SynthesisTree, SynthesisPath
from .mcts_searcher import MCTSSearcher
from ..config.config import EvaluationConfig, MCTSConfig, SelectionConfig
from .evaluator import PathEvaluator
from .selector import EnhancedSelector
from .analyzer import Analyzer


class ParetoSynthesisSystem:
    """Pareto多目标合成路径设计系统 - 极简版"""
    
    def __init__(self, 
                 single_fn_function,
                 commercial_molecules: List[str],
                 eval_config: Optional[EvaluationConfig] = None,
                 mcts_config: Optional[MCTSConfig] = None,
                 sel_config: Optional[SelectionConfig] = None,
                 enable_analysis: bool = True):
        
        # 配置
        self.eval_config = eval_config or EvaluationConfig()
        self.mcts_config = mcts_config or MCTSConfig()
        self.sel_config = sel_config or SelectionConfig()
        
        # 核心组件
        self.single_fn = single_fn_function
        self.commercial_molecules = commercial_molecules
        
        # 初始化组件
        self.evaluator = PathEvaluator(commercial_molecules, self.eval_config)
        self.selector = EnhancedSelector(commercial_molecules, self.evaluator, self.sel_config)
        
        # 可选分析器
        self.analyzer = None
        if enable_analysis:
            analysis_path = os.path.join(self.mcts_config.result_path, 'analysis')
            self.analyzer = Analyzer(result_path=analysis_path)
        
        # 运行时状态
        self.tree = None
        self.searcher = None
    
    def search_synthesis_paths(self, target_molecule: str, 
                              max_iterations: Optional[int] = None) -> SynthesisTree:
        """搜索合成路径"""
        # 创建树和搜索器
        self.tree = SynthesisTree(target_molecule, self.commercial_molecules)
        self.searcher = MCTSSearcher(
            tree=self.tree,
            single_fn_function=self.single_fn,
            evaluator=self.evaluator,
            selector=self.selector,
            mcts_config=self.mcts_config,
            analyzer=self.analyzer
        )
        
        # 执行搜索
        return self.searcher.search(max_iterations)
    
    def get_results(self) -> Dict:
        """获取搜索结果"""
        if not self.tree:
            return {}
        
        return {
            'best_paths': self.tree.get_best_paths(top_k=10),
            'pareto_solutions': self.tree.root.get_pareto_front(),
            'tree_stats': self.tree.get_statistics(),
            'search_stats': {
                'iterations': self.searcher.iteration_count if self.searcher else 0,
                'best_hypervolume': self.searcher.best_hypervolume if self.searcher else 0,
            },
            'evaluation_stats': self.evaluator.get_evaluation_statistics()
        }
    
    def export_visualization(self, complete_paths_only: bool = False) -> Tuple[str, str]:
        """导出可视化"""
        if not self.analyzer:
            return "", ""
        
        if complete_paths_only:
            return self.analyzer.export_complete_paths()
        else:
            return self.analyzer.export_full_tree()
    
    def generate_report(self) -> Optional[Dict]:
        """生成分析报告"""
        if not self.analyzer:
            return None
        
        try:
            self.analyzer.plot_path_discovery()
            self.analyzer.export_paths_to_csv()
            self.analyzer.plot_pareto_front()
            self.analyzer.plot_convergence()
            self.analyzer.visualize_search_tree(max_depth=20)
            return self.analyzer.generate_comprehensive_report()
        except Exception as e:
            print(f"生成报告出错: {e}")
            return None


# 便捷创建函数
def create_system(single_fn, commercial_molecules: List[str], **kwargs) -> ParetoSynthesisSystem:
    """便捷创建函数"""
    return ParetoSynthesisSystem(
        single_fn_function=single_fn,
        commercial_molecules=commercial_molecules,
        eval_config=EvaluationConfig(**kwargs.get('eval_config', {})),
        mcts_config=MCTSConfig(**kwargs.get('mcts_config', {})),
        sel_config=SelectionConfig(**kwargs.get('sel_config', {})),
        enable_analysis=kwargs.get('enable_analysis', True)
    )


# 使用示例
def example():
    # 创建系统
    system = create_system(
        single_fn=your_single_step_function,
        commercial_molecules=["mol1", "mol2"],
        mcts_config={'max_iterations': 500}
    )
    
    # 搜索
    tree = system.search_synthesis_paths("target_molecule")
    
    # 获取结果
    results = system.get_results()
    print(f"找到 {len(results['best_paths'])} 条路径")
    
    # 可视化
    system.export_visualization()
    system.generate_report()
    
    return system