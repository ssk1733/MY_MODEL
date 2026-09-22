#!/usr/bin/env python3
"""
独立脚本：从已有的 MooRetro 结果文件中生成合成树图像。
用法：
    python generate_images_from_results.py --result_dir /path/to/result_dir
示例：
    python generate_images_from_results.py --result_dir results/paroutes_runs/smoke/00001_xxx
"""

import argparse
import json
import os
import sys

# 尝试导入 graphviz
try:
    from graphviz import Digraph
except ImportError:
    print("错误：未安装 graphviz Python 包。请运行: pip install graphviz")
    sys.exit(1)

# 尝试设置 graphviz 可执行文件路径（如果系统安装但找不到）
def setup_graphviz_path():
    import subprocess
    import shutil
    # 尝试找到 dot 命令
    dot_path = shutil.which('dot')
    if dot_path:
        # 设置环境变量
        os.environ["PATH"] += os.pathsep + os.path.dirname(dot_path)
        try:
            from graphviz import backend
            backend.set_executable(executable=dot_path, format='pdf')
        except:
            pass
        return True
    else:
        # 常见安装路径
        common_paths = [
            '/usr/bin/dot',
            '/usr/local/bin/dot',
            '/opt/homebrew/bin/dot',
            '/home/conda/bin/dot',
        ]
        for path in common_paths:
            if os.path.exists(path):
                os.environ["PATH"] += os.pathsep + os.path.dirname(path)
                try:
                    from graphviz import backend
                    backend.set_executable(executable=path, format='pdf')
                except:
                    pass
                return True
        return False

def load_paroutes_tree(paroutes_file):
    """加载 paroutes_routes.json 文件（单个目标）"""
    with open(paroutes_file, 'r') as f:
        data = json.load(f)
    # data 可能是一个列表（多条路线），取第一条
    if isinstance(data, list) and len(data) > 0:
        return data[0]
    return data

def build_graph_from_tree(tree_node, graph=None, parent=None, edge_label=""):
    """递归构建 graphviz 图"""
    if graph is None:
        graph = Digraph(comment='Synthesis Tree', format='pdf')
        graph.attr(rankdir='TB')  # 从上到下
    
    node_id = str(id(tree_node))
    node_label = tree_node.get('smiles', '')
    # 截断过长的 SMILES
    if len(node_label) > 50:
        node_label = node_label[:47] + '...'
    node_attrs = {'shape': 'box', 'style': 'filled', 'fillcolor': 'lightblue'}
    if tree_node.get('in_stock'):
        node_attrs['fillcolor'] = 'lightgreen'
    graph.node(node_id, label=node_label, **node_attrs)
    
    if parent:
        graph.edge(parent, node_id, label=edge_label)
    
    children = tree_node.get('children', [])
    for child in children:
        if child.get('type') == 'reaction':
            # 反应节点，作为中间节点
            react_id = str(id(child))
            react_label = child.get('smiles', '')
            if len(react_label) > 40:
                react_label = react_label[:37] + '...'
            graph.node(react_id, label=react_label, shape='ellipse', style='filled', fillcolor='lightyellow')
            graph.edge(node_id, react_id)
            # 反应节点的 children 是分子节点
            for grandchild in child.get('children', []):
                build_graph_from_tree(grandchild, graph, react_id, "")
        else:
            # 分子节点
            build_graph_from_tree(child, graph, node_id, "")
    return graph

def main():
    parser = argparse.ArgumentParser(description='从结果文件生成合成树图像')
    parser.add_argument('--result_dir', required=True, help='结果目录（包含 paroutes_routes.json 或 pruned/synthesis_tree_pruned_data.json）')
    parser.add_argument('--output', help='输出文件路径（不含扩展名），默认保存在 result_dir 下')
    parser.add_argument('--format', default='pdf', choices=['pdf', 'png', 'svg'], help='输出格式')
    args = parser.parse_args()
    
    # 设置 graphviz 路径
    if not setup_graphviz_path():
        print("警告：未找到 graphviz 的 dot 命令。请确保已安装 graphviz 软件。")
        print("Ubuntu/Debian: sudo apt-get install graphviz")
        print("CentOS/RHEL: sudo yum install graphviz")
        print("MacOS: brew install graphviz")
        # 仍然尝试继续，可能失败
    
    # 查找 paroutes_routes.json
    paroutes_file = os.path.join(args.result_dir, 'paroutes_routes.json')
    if not os.path.exists(paroutes_file):
        # 尝试 pruned 目录下的 synthesis_tree_pruned_data.json
        paroutes_file = os.path.join(args.result_dir, 'pruned', 'synthesis_tree_pruned_data.json')
    if not os.path.exists(paroutes_file):
        print(f"错误：在 {args.result_dir} 中找不到 paroutes_routes.json 或 pruned/synthesis_tree_pruned_data.json")
        sys.exit(1)
    
    print(f"加载结果文件: {paroutes_file}")
    tree_data = load_paroutes_tree(paroutes_file)
    if not tree_data:
        print("错误：无法解析路线数据")
        sys.exit(1)
    
    print("构建图像...")
    graph = build_graph_from_tree(tree_data)
    
    # 确定输出路径
    if args.output:
        output_base = args.output
    else:
        output_base = os.path.join(args.result_dir, f'synthesis_tree')
    
    output_path = graph.render(filename=output_base, format=args.format, cleanup=True)
    print(f"图像已保存至: {output_path}")

if __name__ == '__main__':
    main()